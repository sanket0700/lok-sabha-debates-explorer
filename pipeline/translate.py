"""Stage 4: translate Hindi (and mixed-language) speeches to English.

English-only speeches are simply copied through — no model call. Everything
downstream (NER, topic tagging, sentiment, embeddings) runs on text_english
only, which is the key simplification from the rebuild plan: one English
model stack instead of separate Hindi/English models per task.

Uses AI4Bharat's IndicTrans2 (local HF model), the current best open-source
Indic MT system, via the companion IndicTransToolkit for pre/post-processing.

Idempotent: only processes speeches where text_english IS NULL.

Usage:
    python -m pipeline.translate [--lok-sabha 16] [--limit 500]
"""
import argparse
import logging
import re

from pipeline.config import TRANSLATE_MODEL, get_device
from pipeline.db import execute, fetch_all, get_conn

log = logging.getLogger("pipeline.translate")

# translate_batch tokenizes with max_length=256 and truncation=True — IndicTrans2
# is meant to be used one sentence at a time (that's what IndicProcessor's
# pre/post-processing is designed around), not fed a whole multi-paragraph
# speech as a single unit. Splitting into sentences here, rather than raising
# max_length, is the correct fix: it's also what keeps translation quality
# high (seq2seq MT degrades on very long single inputs) — confirmed by hand
# that feeding a whole speech silently truncated everything past ~150-200
# words, which would have affected most medium-or-longer Hindi speeches, not
# just outliers.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[।.!?])\s+")
# This machine has 16GB unified memory total. num_beams=5 with a 32-sentence
# batch on MPS reported an 18.13GB "max allowed" ceiling — already larger
# than physical RAM — and ran for ~12 hours real time before being killed:
# repeated "MPS backend out of memory" errors (each one wasting the time to
# fail, not just failing fast) plus catastrophic swap thrashing (13GB/14GB
# swap in use, 650M+ swap I/O events) once accumulated allocations crossed
# what the machine actually has. Beam search multiplies KV-cache memory by
# num_beams; that's the dominant lever, not batch size. Cut both.
SENTENCE_BATCH_SIZE = 8
NUM_BEAMS = 1  # greedy — trades some translation quality for staying inside 16GB


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
    return sentences or [text.strip()]

_model = None
_tokenizer = None
_processor = None


def _load():
    """Lazy-load the (large) translation model on first use only."""
    global _model, _tokenizer, _processor
    if _model is not None:
        return
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor

    log.info("loading translation model %s (first call only)...", TRANSLATE_MODEL)
    _tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_MODEL, trust_remote_code=True)
    _model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL, trust_remote_code=True)
    device = get_device()
    log.info("moving translation model to device=%s", device)
    _model = _model.to(device)
    _model.eval()
    _processor = IndicProcessor(inference=True)


def translate_batch(texts: list[str], src_lang: str = "hin_Deva", tgt_lang: str = "eng_Latn") -> list[str]:
    import torch

    _load()
    device = get_device()
    batch = _processor.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = _tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = _model.generate(
            **inputs, use_cache=True, min_length=0, max_length=256, num_beams=NUM_BEAMS
        )
    decoded = _tokenizer.batch_decode(generated, skip_special_tokens=True)
    result = _processor.postprocess_batch(decoded, lang=tgt_lang)

    # Without this, MPS allocations accumulate across calls (fragmentation,
    # not a real leak) until the process falls off the memory cliff — this is
    # what turned a should-be-fast task into a ~12-hour run that never
    # recovered. Cheap per-call, and this machine has no memory to spare.
    if device == "mps":
        torch.mps.empty_cache()

    return result


def run(lok_sabha_number: int | None, limit: int | None, batch_size: int = 16):
    with get_conn() as conn:
        # English-only speeches: pass through, no model call needed.
        execute(
            conn,
            """
            update speeches set text_english = text_original
            where text_english is null and language = 'en'
              and (%s is null or sitting_id in (select id from sittings where lok_sabha_number = %s))
            """,
            (lok_sabha_number, lok_sabha_number),
        )
        conn.commit()

        query = (
            "select id, text_original from speeches "
            "where text_english is null and language in ('hi', 'mixed')"
        )
        params = []
        if lok_sabha_number is not None:
            query += " and sitting_id in (select id from sittings where lok_sabha_number = %s)"
            params.append(lok_sabha_number)
        query += " order by id"
        if limit is not None:
            query += " limit %s"
            params.append(limit)

        rows = fetch_all(conn, query, params)
        log.info("translating %d speeches", len(rows))

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]

            per_speech_sentences = [split_sentences(r["text_original"]) for r in batch]
            all_sentences = [s for sentences in per_speech_sentences for s in sentences]

            try:
                translated_sentences = []
                for j in range(0, len(all_sentences), SENTENCE_BATCH_SIZE):
                    translated_sentences.extend(
                        translate_batch(all_sentences[j : j + SENTENCE_BATCH_SIZE])
                    )
            except Exception:
                log.exception("batch translation failed for speech ids %s", [r["id"] for r in batch])
                continue

            idx = 0
            for row, sentences in zip(batch, per_speech_sentences):
                translation = " ".join(translated_sentences[idx : idx + len(sentences)])
                idx += len(sentences)
                execute(
                    conn,
                    "update speeches set text_english = %s where id = %s",
                    (translation, row["id"]),
                )
            conn.commit()
            log.info("translated %d/%d", min(i + batch_size, len(rows)), len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lok-sabha", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run(args.lok_sabha, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
