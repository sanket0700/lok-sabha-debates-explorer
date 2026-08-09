"""Stage 5d: sentiment/tone scoring over text_english.

Score is a signed continuous value in [-1, 1] (P(positive) - P(negative)
from the full 3-class softmax, not just the top label) so the Insights
"sentiment/polarization over time" trend has more resolution than a bare
pos/neg/neutral label would give.

Idempotent: only processes speeches with no row yet in `sentiment`.

Usage:
    python -m pipeline.enrich_sentiment [--lok-sabha 16] [--limit 500]
"""
import argparse
import logging

from pipeline.config import SENTIMENT_MODEL, get_device
from pipeline.db import execute, fetch_all, get_conn

log = logging.getLogger("pipeline.enrich_sentiment")

_classifier = None


def _load():
    global _classifier
    if _classifier is not None:
        return _classifier
    from transformers import pipeline

    device = get_device()
    log.info("loading sentiment model %s on device=%s (first call only)...", SENTIMENT_MODEL, device)
    _classifier = pipeline(
        "sentiment-analysis", model=SENTIMENT_MODEL, tokenizer=SENTIMENT_MODEL, top_k=None, device=device
    )
    return _classifier


def _score_and_label(class_scores: list[dict]) -> tuple[float, str]:
    by_label = {c["label"].lower(): c["score"] for c in class_scores}
    positive = by_label.get("positive", 0.0)
    negative = by_label.get("negative", 0.0)
    score = positive - negative
    label = max(class_scores, key=lambda c: c["score"])["label"].lower()
    return score, label


def run(lok_sabha_number: int | None, limit: int | None, batch_size: int = 16):
    classifier = _load()
    with get_conn() as conn:
        query = """
            select s.id, s.text_english from speeches s
            left join sentiment sen on sen.speech_id = s.id
            where s.text_english is not null and sen.speech_id is null
        """
        params = []
        if lok_sabha_number is not None:
            query += " and s.sitting_id in (select id from sittings where lok_sabha_number = %s)"
            params.append(lok_sabha_number)
        query += " order by s.id"
        if limit is not None:
            query += " limit %s"
            params.append(limit)

        rows = fetch_all(conn, query, params)
        log.info("scoring sentiment for %d speeches", len(rows))

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [r["text_english"][:512] for r in batch]
            results = classifier(texts)

            for row, class_scores in zip(batch, results):
                score, label = _score_and_label(class_scores)
                execute(
                    conn,
                    """
                    insert into sentiment (speech_id, score, label) values (%s, %s, %s)
                    on conflict (speech_id) do update set score = excluded.score, label = excluded.label
                    """,
                    (row["id"], score, label),
                )
            conn.commit()
            log.info("sentiment: %d/%d", min(i + batch_size, len(rows)), len(rows))


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
