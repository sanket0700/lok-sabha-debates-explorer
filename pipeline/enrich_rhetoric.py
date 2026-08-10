"""Stage 5e: rhetorical-mode classification — what a speaker is *doing*,
not how the text sounds.

Generic positive/negative sentiment is a poor fit for parliamentary debate: a
substantive, well-argued critique of a policy reads as "negative" even though
it's exactly what opposition scrutiny is supposed to look like. This stage
classifies each speech into one of config.RHETORIC_TAXONOMY's labels
(supportive / critical / constructive / confrontational / procedural)
instead — single-label, since testing multi-label zero-shot on this taxonomy
produced near-meaningless output (almost every label scoring >0.7 on almost
every speech; see PROGRESS.md).

Two refinements on top of the plain classifier, both added after real
prototyping found the plain zero-shot pass alone wasn't reliable enough:

1. Structural short-circuits (_STRUCTURAL_PROCEDURAL_PATTERNS): a large
   fraction of this corpus is recognizable procedural boilerplate (Written
   Question openers, "I beg to lay/move" motions, structured "(a)...(b)..."
   written answers, the presiding officer managing order) that a generic
   NLI classifier doesn't reliably recognize as procedural — confirmed by
   hand that it inconsistently mislabeled the *same* boilerplate structure
   as "critical" depending on unrelated wording differences. These patterns
   resolve ~56% of the corpus without a model call at all: cheaper, and
   more reliable than asking the model to guess on something a regex
   already answers confidently.

2. Minister role-context (pipeline.speakers.resolve_minister_role): for the
   remainder, knowing *who* is speaking is real signal a text-only
   classifier can't recover — a minister explaining their own policy reads
   differently than an unattributed floor speech. Confirmed by hand this
   must run strictly *after* the structural check, not before: tested
   against written-answer content the structural patterns hadn't yet
   caught, the role-injected classifier systematically over-biased plain
   factual answers toward "supportive" just because a minister said them.

Speeches under RHETORIC_MIN_CHARS get a 'too_short' sentinel label instead
of a real classification — confirmed by hand that the classifier still
returns a confident-looking score on bare interjections ("Why is that
relevant?") with no real rhetorical content to judge. aggregate.py and the
app exclude this label from trends/display.

Idempotent: only processes speeches with no row yet in `rhetoric`.

Usage:
    python -m pipeline.enrich_rhetoric [--lok-sabha 16] [--limit 500]
"""
import argparse
import logging
import re

from pipeline.config import RHETORIC_MIN_CHARS, RHETORIC_TAXONOMY, TOPIC_MODEL, get_device
from pipeline.db import execute, fetch_all, get_conn
from pipeline.speakers import resolve_minister_role

log = logging.getLogger("pipeline.enrich_rhetoric")

# Presiding officer / procedural-secretary roles — these speeches are almost
# always the Chair managing order or reading out routine business, not
# taking a rhetorical stance. Matches speaker_canonical, not text content.
_PROCEDURAL_SPEAKER_PATTERN = re.compile(
    r"speaker|chairperson|chairman|deputy|अध्यक्ष|सभापति|secretary.general|महासचिव", re.IGNORECASE
)

# Written-Question opener ("Will the Minister ... be pleased to state") —
# confirmed 17.5% of the corpus.
_WRITTEN_QUESTION_PATTERN = re.compile(r"^\s*(will|would) the (hon'?ble )?minister", re.IGNORECASE)

# Structured written-answer body ("(a): ..." / "(a) and (b): ..."), including
# the confirmed "(a)" -> "(8)" OCR digit confusion — same class of noise
# already documented for segmentation (e.g. "SHRI 8. MAHTAB"). ~9.6% of the
# corpus, before the (8) variant; that variant alone is a further 0.3%.
_WRITTEN_ANSWER_PATTERN = re.compile(r"^\s*\((a|8)\)", re.IGNORECASE)

# Formal motion/procedural-tabling openers: "I beg to lay...", "I beg to
# move...", "I rise to move...", "I move: That...", "I table...". Includes
# "|" as an alternative to "I" — a confirmed common OCR/extraction
# misreading of the capital letter (same failure mode found in the
# translate-quality review this project already fixed via detect_language()
# — see PROGRESS.md). Consolidated into one pattern rather than chasing
# every possible phrasing indefinitely; residual variants not covered here
# are a known, accepted gap, not something to keep expanding forever.
_MOTION_OPENER_PATTERN = re.compile(
    r"^\s*(madam|sir|mr\.?\s*speaker|hon\.?\s*(chairperson|speaker)\s*sir)?,?\s*"
    r"[i|]\s*(beg to lay|beg to move|rise to move|move\s*:|table the)",
    re.IGNORECASE,
)


def _structural_bypass(speaker_canonical: str | None, text: str) -> str | None:
    if speaker_canonical and _PROCEDURAL_SPEAKER_PATTERN.search(speaker_canonical):
        return "speaker-role"
    head = text[:120]
    if _WRITTEN_QUESTION_PATTERN.search(head):
        return "written-question"
    if _MOTION_OPENER_PATTERN.search(text[:60]):
        return "motion-opener"
    if _WRITTEN_ANSWER_PATTERN.search(text[:15]):
        return "written-answer"
    return None


_classifier = None


def _load():
    global _classifier
    if _classifier is not None:
        return _classifier
    from transformers import pipeline

    device = get_device()
    log.info("loading rhetoric model %s on device=%s (first call only)...", TOPIC_MODEL, device)
    _classifier = pipeline("zero-shot-classification", model=TOPIC_MODEL, device=device)
    return _classifier


_LABEL_BY_HYPOTHESIS = {v: k for k, v in RHETORIC_TAXONOMY.items()}
_HYPOTHESES = list(RHETORIC_TAXONOMY.values())


def classify_one(classifier, speaker_canonical, lok_sabha_number, sitting_date, text):
    """Returns (label, score, method). score is None for structural bypasses."""
    if len(text) < RHETORIC_MIN_CHARS:
        return "too_short", None, "skipped:too_short"

    bypass = _structural_bypass(speaker_canonical, text)
    if bypass:
        return "procedural", None, f"bypass:{bypass}"

    role = resolve_minister_role(speaker_canonical, lok_sabha_number, sitting_date)
    premise = text if not role else f"[Spoken by the {role}, responding as a government minister.] " + text
    out = classifier([premise], candidate_labels=_HYPOTHESES, multi_label=False, truncation=True)[0]
    label = _LABEL_BY_HYPOTHESIS[out["labels"][0]]
    method = f"minister:{role}" if role else "plain"
    return label, float(out["scores"][0]), method


def run(
    lok_sabha_number: int | None,
    limit: int | None,
    batch_size: int = 16,
    sitting_ids: list[int] | None = None,
):
    classifier = _load()
    with get_conn() as conn:
        query = """
            select sp.id, sp.speaker_canonical, sp.text_english,
                   si.lok_sabha_number, si.sitting_date
            from speeches sp
            join sittings si on si.id = sp.sitting_id
            left join rhetoric r on r.speech_id = sp.id
            where sp.text_english is not null and r.speech_id is null
        """
        params = []
        if lok_sabha_number is not None:
            query += " and si.lok_sabha_number = %s"
            params.append(lok_sabha_number)
        if sitting_ids:
            query += " and sp.sitting_id = any(%s)"
            params.append(sitting_ids)
        query += " order by sp.id"
        if limit is not None:
            query += " limit %s"
            params.append(limit)

        rows = fetch_all(conn, query, params)
        log.info("classifying rhetoric for %d speeches", len(rows))

        for i, row in enumerate(rows):
            label, score, method = classify_one(
                classifier,
                row["speaker_canonical"],
                row["lok_sabha_number"],
                row["sitting_date"],
                row["text_english"],
            )
            execute(
                conn,
                """
                insert into rhetoric (speech_id, label, score, method)
                values (%s, %s, %s, %s)
                on conflict (speech_id) do update set label = excluded.label, score = excluded.score, method = excluded.method
                """,
                (row["id"], label, score, method),
            )
            if i % batch_size == 0:
                conn.commit()
                log.info("rhetoric: %d/%d", i, len(rows))
        conn.commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lok-sabha", type=int, default=None)
    parser.add_argument("--sitting-id", type=int, action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run(args.lok_sabha, args.limit, args.batch_size, args.sitting_id)


if __name__ == "__main__":
    main()
