"""Stage 3: raw sitting text -> individual speech records.

Two-tier approach:
  1. Regex heuristics find "SPEAKER NAME:" boundaries (title-prefixed or
     ALL-CAPS runs followed by a colon) — cheap, and correct for the large
     majority of well-formatted text-layer sittings.
  2. Chunks the heuristics can't confidently split (too long, no plausible
     speaker label, suspiciously short) are merged and handed to the local
     Ollama instruct model, which returns structured {speaker, text} JSON.

This intentionally does NOT special-case per-Lok-Sabha formatting quirks
the way the old repo's notebook did (a near-duplicate regex block per Lok
Sabha) — the LLM fallback exists precisely to absorb that variance instead
of hand-tuning a new regex block per session.

Idempotent: only processes sittings where segmented_at IS NULL and
extracted_at IS NOT NULL.

Usage:
    python -m pipeline.segment [--lok-sabha 16] [--limit 50]
"""
import argparse
import logging
import re

from pipeline.db import execute, fetch_all, get_conn
from pipeline.llm import generate_json
from pipeline.speakers import canonicalize_speaker

log = logging.getLogger("pipeline.segment")

START_MARKERS = [
    "in the Chair]",
    "The Lok Sabha met at",
    "OBITUARY REFERENCE",
    "LOK SABHA DEBATES",
]
END_MARKERS = [
    "The Lok Sabha then adjourned",
    "(The National Song was played",
    "NATIONAL SONG",
]

NOISE_PATTERNS = [
    re.compile(r"\(Interruptions\)", re.IGNORECASE),
    re.compile(r"\(व्यवधान\)"),
    re.compile(r"\[?\(?Placed in Library,?\s*See No\.\s*LT\s*\d*\d\d\d\d/\d\d/\d\d\)?\]?"),
    re.compile(r"\d{1,2}\.\d{1,2}\.20\d\d"),
    re.compile(r"\d{1,2}[.:]\d{2}\s*hrs\.?", re.IGNORECASE),
    re.compile(r"_+"),
    re.compile(r"…"),
]

SPEAKER_TITLES = (
    r"SHRI|SHRIMATI|SMT\.?|DR\.?|KUMARI|PROF\.?|MR\.?|MRS\.?|MS\.?|"
    r"THE\s+MINISTER\s+OF[^:\n]{0,60}|THE\s+DEPUTY\s+SPEAKER|THE\s+SPEAKER|"
    r"MR\.?\s+SPEAKER|MADAM\s+SPEAKER|MR\.?\s+CHAIRMAN|"
    # "HON. SPEAKER:"/"SECRETARY-GENERAL:" are distinct printed labels from
    # "MR. SPEAKER"/"THE SPEAKER" already above — found missing by the
    # segmentation accuracy audit (confirmed real, frequent: e.g. 24
    # occurrences of "HON. SPEAKER:" in one otherwise-clean LS16 sitting,
    # each one previously silently bleeding into whichever speech matched
    # immediately before it instead of being its own turn).
    r"HON\.?\s+SPEAKER|HON\.?\s+DEPUTY\s+SPEAKER|HON\.?\s+CHAIRMAN|HON\.?\s+CHAIRPERSON|"
    r"SECRETARY[\s-]?GENERAL|"
    # Hindi-script honorifics — speaker turns attributed in Devanagari (own
    # name written in Hindi rather than transliterated) were previously
    # invisible to this pattern and got silently absorbed into whichever
    # preceding English-labeled speech was still "open".
    # माननीय सभापति / सभापति महोदय (Chairperson, used when someone other than
    # the Speaker is presiding) is the same class of gap as अध्यक्ष below —
    # confirmed 93 occurrences in one LS17 sitting alone, previously all
    # silently bled into the preceding matched speaker.
    r"श्रीमती|श्री|डॉ[०.]?|कुमारी|प्रो[०.]?|"
    r"माननीय\s+अध्यक्ष|अध्यक्ष\s+महोदय|माननीय\s+सभापति|सभापति\s+महोदय"
)
SPEAKER_PATTERN = re.compile(
    rf"(?P<speaker>(?:{SPEAKER_TITLES})[A-Za-zऀ-ॿ.,'()\-\d ]{{0,80}}?)\s*:\s*",
)

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Real Devanagari letters/vowel-signs only, excluding danda punctuation
# (। ॥, U+0964-0965), digits (०-९, U+0966-096F), and the abbreviation
# sign (U+0970) — used only by detect_language(), not the broader
# DEVANAGARI pattern above (which stays intentionally inclusive for
# extract.py's OCR-trigger decision, a different and legitimately
# conservative use case). Confirmed by hand: every speech in the corpus
# whose only "Devanagari" content was digits/danda (a Devanagari-numeral
# footnote marker, "।, ॥" used like Roman numerals "I, II") had zero real
# Hindi content — these were misclassified as language='mixed' and sent
# through the Hindi->English translation model, which can catastrophically
# corrupt long English/tabular content it was never meant to touch
# (confirmed: a large government-answer data table lost an entire state's
# row and degenerated into repeated-token loops after being needlessly
# translated over a single stray digit).
DEVANAGARI_LETTER = re.compile(r"[ऀ-ॣॱ-ॿ]")

MIN_SEGMENT_CHARS = 8
MAX_SPEAKER_CHARS = 90
# A single regex-matched speaker turn this long almost certainly swallowed
# multiple real turns without noticing — e.g. a continuous President's-
# Address-style monologue where the *first* speaker label matched but no
# further one did for tens of thousands of characters, producing one
# enormous "speech" instead of a segmentation failure being flagged.
#
# This used to be 8000, on the reasoning that a real 71K-char match had
# stalled downstream translation. That translation-stalling problem is now
# fixed properly at the translation stage instead (pipeline/translate.py
# splits into sentences before calling the model, rather than silently
# truncating a giant input) — so this threshold no longer needs to protect
# translation, only genuine mis-segmentation. Measured on real LS18 sittings:
# at 8000, a single sitting can have 40+ "oversized" matches, the vast
# majority in the 8K-30K range and reading as ordinary (if long) single
# speeches, not merges — each one escalated to the LLM fallback anyway,
# at ~140s/call (confirmed by hand: a 6000-char chunk took 138s for
# llama3.1:8b-instruct to re-segment, JSON-grammar-constrained decoding of a
# ~2000-token response). That's 1-2+ hours of avoidable LLM time per sitting
# for turns that almost certainly didn't need re-segmenting at all. Raised
# to comfortably exceed ordinary long speeches while still catching the
# genuinely pathological merges (confirmed real examples at 70-71K chars).
MAX_SEGMENT_CHARS = 40000


def trim_to_debate_body(text: str) -> str:
    start = 0
    for marker in START_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            start = idx
            break
    end = len(text)
    for marker in END_MARKERS:
        # rfind, not find: "The Lok Sabha then adjourned" (etc.) also marks
        # every intra-day recess (lunch break, etc.), not just the actual
        # end of the sitting — the real end-of-day marker is the LAST match.
        idx = text.rfind(marker)
        if idx != -1 and idx >= start:
            end = idx
            break
    return text[start:end]


def clean_speech_text(text: str) -> str:
    for pattern in NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> str:
    devanagari_chars = len(DEVANAGARI_LETTER.findall(text))
    if devanagari_chars == 0:
        return "en"
    if devanagari_chars / max(len(text), 1) > 0.15:
        return "hi"
    return "mixed"


def _balance_trailing_paren(speaker: str) -> str:
    """Drop a trailing ")" that has no matching "(" in the same speaker string
    — observed in practice (e.g. "SHRI BABUL SUPRIYO)"): the non-greedy
    SPEAKER_PATTERN match overshoots onto a stray close-paren from a nearby
    bracketed aside instead of stopping at the name's actual end."""
    if speaker.endswith(")") and speaker.count("(") < speaker.count(")"):
        return speaker[:-1].rstrip()
    return speaker


def regex_segment(body: str):
    """Yield (speaker_raw, text) tuples using the SPEAKER_PATTERN boundaries."""
    matches = list(SPEAKER_PATTERN.finditer(body))
    segments = []
    for i, m in enumerate(matches):
        speaker = re.sub(r"\s+", " ", m.group("speaker")).strip()
        speaker = _balance_trailing_paren(speaker)
        text_start = m.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        raw_text = body[text_start:text_end]
        segments.append((speaker, raw_text))
    return segments


def is_noise(text: str) -> bool:
    """True for a rough segment that's just procedural noise (a bare
    "(Interruptions)", a timestamp, a "Placed in Library" note between two
    real speeches) once NOISE_PATTERNS are stripped — not a parsing failure,
    just nothing worth keeping or re-segmenting."""
    return len(clean_speech_text(text)) < MIN_SEGMENT_CHARS


LLM_SEGMENT_PROMPT = """You are splitting a noisy OCR/PDF transcript excerpt from an Indian \
Lok Sabha debate into individual speeches. Identify each speaker's name (as printed, do not \
invent one) and the text of what they said. Merge stray fragments into the nearest real \
speech rather than inventing empty entries. If a speech's speaker cannot be confidently \
identified from the text, set "speaker" to null — never invent a placeholder value (not "...", \
not a bare honorific like "Shri" or "Dr." with no name attached, not "Unknown").

Return strict JSON, e.g.: {{"speeches": [{{"speaker": "Shri Ram Manohar Naidu", "text": "..the \
speech text.."}}, {{"speaker": null, "text": "..a speech whose speaker is unclear.."}}]}}

Transcript excerpt:
---
{chunk}
---
"""


LLM_CHUNK_CHARS = 6000

# Honorifics on their own, with nothing else, aren't a name — the LLM fallback
# has been observed returning these (and bare "..."/"null" placeholders, the
# latter presumably the model outputting the *string* "null" instead of a
# real JSON null) despite the prompt saying not to invent one. Treated the
# same as a null speaker.
_BARE_HONORIFICS = {
    "shri", "shrimati", "smt", "dr", "kumari", "prof", "mr", "mrs", "ms",
    "श्री", "श्रीमती", "डॉ", "कुमारी", "प्रो",
    "null", "none", "unknown", "n/a", "na",
}


def _is_placeholder_speaker(speaker: str) -> bool:
    stripped = speaker.strip().rstrip(".:")
    if not stripped:
        return True
    if not re.search(r"[A-Za-zऀ-ॿ]", stripped):
        return True  # punctuation/symbols only, e.g. "..."
    return stripped.lower() in _BARE_HONORIFICS


def llm_segment_chunk(chunk: str):
    result = generate_json(LLM_SEGMENT_PROMPT.format(chunk=chunk[:LLM_CHUNK_CHARS]))
    if not result or not isinstance(result.get("speeches"), list):
        return []
    out = []
    for item in result["speeches"]:
        # The model doesn't always honor the requested shape (has been
        # observed returning plain strings instead of {speaker, text}
        # objects) — skip anything that isn't a dict rather than crashing
        # the whole sitting.
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker") or "").strip()
        text = str(item.get("text", "")).strip()
        if speaker and _is_placeholder_speaker(speaker):
            speaker = ""
        if text:
            out.append((speaker or None, text))
    return out


def segment_sitting_text(raw_text: str):
    body = trim_to_debate_body(raw_text)
    rough_segments = regex_segment(body)

    final_segments = []
    ambiguous_run = []

    if not rough_segments and len(body.strip()) >= MIN_SEGMENT_CHARS:
        # No speaker-colon boundaries found at all (e.g. a continuous
        # President's-address-style sitting with no back-and-forth) — hand
        # the whole body to the LLM fallback instead of silently yielding
        # zero speeches.
        ambiguous_run.append((None, body))

    def flush_ambiguous():
        if not ambiguous_run:
            return
        merged = "\n".join(f"{sp or ''}: {tx}" for sp, tx in ambiguous_run)
        # llm_segment_chunk only looks at the first LLM_CHUNK_CHARS of what
        # it's given — split ourselves so a long run (e.g. a whole
        # speaker-marker-free sitting) doesn't silently lose everything past
        # the first chunk.
        for i in range(0, len(merged), LLM_CHUNK_CHARS):
            chunk = merged[i : i + LLM_CHUNK_CHARS]
            if SPEAKER_PATTERN.search(chunk) is None:
                # No recognizable speaker-label boundary anywhere in this
                # window — genuinely continuous unlabeled prose (a President's
                # Address paragraph windowed arbitrarily at LLM_CHUNK_CHARS,
                # confirmed by the segmentation audit: entire sittings with
                # zero SPEAKER_PATTERN matches had EVERY window like this).
                # Asking the LLM to "identify each speaker" here has been
                # observed, on real sittings, to hallucinate plausible-looking
                # but fabricated speaker names (a single dictionary word, a
                # historical figure being quoted within the speech, a vocative
                # phrase like "Honorable Members") instead of reporting null
                # as instructed — so skip the LLM call and keep this as one
                # unattributed segment rather than trusting invented splits.
                final_segments.append((None, chunk))
            else:
                final_segments.extend(llm_segment_chunk(chunk))
        ambiguous_run.clear()

    for speaker, text in rough_segments:
        if is_noise(text):
            continue
        if _is_placeholder_speaker(speaker):
            # A regex match can land on a bare honorific too (e.g. "MS" with
            # no name attached, confirmed in real LS18 output) — same
            # placeholder problem as the LLM fallback, not a mis-segmented
            # boundary, so just drop the fake attribution rather than
            # escalating text that's probably fine.
            speaker = None
        if len(speaker or "") > MAX_SPEAKER_CHARS:
            # The match itself looks wrong (garbled/oversized speaker) — the
            # speaker attribution can't be trusted, so hand the whole thing
            # to the LLM fallback rather than just the excess length.
            ambiguous_run.append((speaker, text))
            continue
        if len(text) > MAX_SEGMENT_CHARS:
            # The speaker match is a normal one here — trust it rather than
            # paying for a full LLM re-segmentation of a speech whose author
            # is already known. Keep speaker + the first MAX_SEGMENT_CHARS
            # as a real segment as-is; only the remainder (which might hide
            # an actual missed speaker change further in) goes to the LLM.
            flush_ambiguous()
            final_segments.append((speaker, text[:MAX_SEGMENT_CHARS]))
            ambiguous_run.append((None, text[MAX_SEGMENT_CHARS:]))
            continue
        flush_ambiguous()
        final_segments.append((speaker, text))
    flush_ambiguous()

    cleaned = []
    for speaker, text in final_segments:
        text = clean_speech_text(text)
        if len(text) < MIN_SEGMENT_CHARS:
            continue
        cleaned.append((speaker, text, detect_language(text)))
    return cleaned


def run(lok_sabha_number: int | None, limit: int | None):
    with get_conn() as conn:
        query = (
            "select id, raw_text, sitting_date from sittings "
            "where extracted_at is not null and segmented_at is null and raw_text is not null"
        )
        params = []
        if lok_sabha_number is not None:
            query += " and lok_sabha_number = %s"
            params.append(lok_sabha_number)
        query += " order by id"
        if limit is not None:
            query += " limit %s"
            params.append(limit)

        rows = fetch_all(conn, query, params)
        log.info("segmenting %d sittings", len(rows))

        for row in rows:
            try:
                speeches = segment_sitting_text(row["raw_text"])
            except Exception:
                log.exception("failed to segment sitting %s", row["id"])
                continue

            for idx, (speaker, text, language) in enumerate(speeches):
                execute(
                    conn,
                    """
                    insert into speeches
                        (sitting_id, sequence_index, speaker_raw, speaker_canonical, language, text_original)
                    values (%s, %s, %s, %s, %s, %s)
                    on conflict (sitting_id, sequence_index) do nothing
                    """,
                    (
                        row["id"], idx, speaker,
                        canonicalize_speaker(speaker, row["sitting_date"]),
                        language, text,
                    ),
                )
            execute(conn, "update sittings set segmented_at = now() where id = %s", (row["id"],))
            conn.commit()
            log.info("sitting %s: %d speeches", row["id"], len(speeches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lok-sabha", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run(args.lok_sabha, args.limit)


if __name__ == "__main__":
    main()
