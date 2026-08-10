"""Stage 5c: topic tagging against the fixed issue taxonomy (config.TOPIC_TAXONOMY).

Zero-shot multi-label classification (a speech can touch more than one
topic) via a local NLI model — no training data needed, and the taxonomy is
small/hand-picked so the Insights dashboard's topic trends stay legible
(see config.py for the rationale).

Idempotent: only processes speeches where topics_processed_at IS NULL.
Only scores above TOPIC_SCORE_THRESHOLD are stored.

Usage:
    python -m pipeline.enrich_topic [--lok-sabha 16] [--limit 500]
"""
import argparse
import logging

from pipeline.config import TOPIC_MODEL, TOPIC_TAXONOMY, get_device
from pipeline.db import execute, fetch_all, get_conn

log = logging.getLogger("pipeline.enrich_topic")

TOPIC_SCORE_THRESHOLD = 0.5

_classifier = None


def _load():
    global _classifier
    if _classifier is not None:
        return _classifier
    from transformers import pipeline

    device = get_device()
    log.info("loading topic model %s on device=%s (first call only)...", TOPIC_MODEL, device)
    _classifier = pipeline("zero-shot-classification", model=TOPIC_MODEL, device=device)
    return _classifier


def ensure_topics(conn) -> dict:
    for name in TOPIC_TAXONOMY:
        execute(conn, "insert into topics (name) values (%s) on conflict (name) do nothing", (name,))
    conn.commit()
    rows = fetch_all(conn, "select id, name from topics")
    return {r["name"]: r["id"] for r in rows}


def run(
    lok_sabha_number: int | None,
    limit: int | None,
    batch_size: int = 8,
    sitting_ids: list[int] | None = None,
):
    classifier = _load()
    with get_conn() as conn:
        topic_ids = ensure_topics(conn)

        query = (
            "select id, text_english from speeches "
            "where text_english is not null and topics_processed_at is null"
        )
        params = []
        if lok_sabha_number is not None:
            query += " and sitting_id in (select id from sittings where lok_sabha_number = %s)"
            params.append(lok_sabha_number)
        # See pipeline/translate.py's --sitting-id for why: lets multiple
        # concurrent workers sharing one DB split the backlog by sitting
        # without racing on the same rows.
        if sitting_ids:
            query += " and sitting_id = any(%s)"
            params.append(sitting_ids)
        query += " order by id"
        if limit is not None:
            query += " limit %s"
            params.append(limit)

        rows = fetch_all(conn, query, params)
        log.info("tagging topics for %d speeches", len(rows))

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [r["text_english"] for r in batch]
            # Token-based truncation (model_max_length=1024 for
            # facebook/bart-large-mnli), not the old 1500-char slice —
            # confirmed by hand that a ~375-token budget (roughly what
            # 1500 chars gives in English) left well under half the
            # model's real 1024-token context unused for longer speeches.
            results = classifier(texts, candidate_labels=TOPIC_TAXONOMY, multi_label=True, truncation=True)
            if isinstance(results, dict):
                results = [results]

            for row, result in zip(batch, results):
                for label, score in zip(result["labels"], result["scores"]):
                    if score >= TOPIC_SCORE_THRESHOLD:
                        execute(
                            conn,
                            """
                            insert into speech_topics (speech_id, topic_id, score)
                            values (%s, %s, %s)
                            on conflict (speech_id, topic_id) do update set score = excluded.score
                            """,
                            (row["id"], topic_ids[label], score),
                        )
                execute(
                    conn,
                    "update speeches set topics_processed_at = now() where id = %s",
                    (row["id"],),
                )
            conn.commit()
            log.info("topics: %d/%d", min(i + batch_size, len(rows)), len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lok-sabha", type=int, default=None)
    parser.add_argument("--sitting-id", type=int, action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run(args.lok_sabha, args.limit, args.batch_size, args.sitting_id)


if __name__ == "__main__":
    main()
