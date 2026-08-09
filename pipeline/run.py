"""Single entrypoint to run pipeline stages, individually or in order.

Every stage is idempotent (see each module's docstring for exactly what
"already done" means for that stage), so re-running `all` after a partial
or failed run just picks up where it left off.

Usage:
    python -m pipeline.run status
    python -m pipeline.run scrape --lok-sabha 16 --limit 20 --dry-run
    python -m pipeline.run all --lok-sabha 16 --limit 50
    python -m pipeline.run extract segment translate --lok-sabha 17
"""
import argparse
import logging

from pipeline import (
    aggregate,
    embed,
    enrich_geocode,
    enrich_ner,
    enrich_sentiment,
    enrich_topic,
    extract,
    scrape,
    segment,
    translate,
)
from pipeline.db import fetch_all, get_conn

log = logging.getLogger("pipeline.run")

# Order matters: each stage depends on the previous one's output.
STAGES = ["scrape", "extract", "segment", "translate", "ner", "geocode", "topic", "sentiment", "embed", "aggregate"]


def run_stage(stage: str, lok_sabha: int | None, limit: int | None, dry_run: bool):
    log.info("=== stage: %s ===", stage)
    if stage == "scrape":
        lok_sabhas = [lok_sabha] if lok_sabha else scrape.LOK_SABHA_NUMBERS
        for ls in lok_sabhas:
            scrape.scrape_lok_sabha(ls, limit, dry_run)
    elif stage == "extract":
        extract.run(lok_sabha, limit)
    elif stage == "segment":
        segment.run(lok_sabha, limit)
    elif stage == "translate":
        translate.run(lok_sabha, limit)
    elif stage == "ner":
        enrich_ner.run(lok_sabha, limit)
    elif stage == "geocode":
        enrich_geocode.run(limit)
    elif stage == "topic":
        enrich_topic.run(lok_sabha, limit)
    elif stage == "sentiment":
        enrich_sentiment.run(lok_sabha, limit)
    elif stage == "embed":
        embed.run(lok_sabha, limit)
    elif stage == "aggregate":
        aggregate.run()
    else:
        raise ValueError(f"unknown stage: {stage}")


def print_status():
    with get_conn() as conn:
        rows = fetch_all(
            conn,
            """
            select
                lok_sabha_number,
                count(*) as sittings,
                count(scraped_at) as scraped,
                count(extracted_at) as extracted,
                count(segmented_at) as segmented
            from sittings
            group by lok_sabha_number
            order by lok_sabha_number
            """,
        )
        print(f"{'LS':<4}{'sittings':<10}{'scraped':<10}{'extracted':<11}{'segmented':<10}")
        for r in rows:
            print(
                f"{r['lok_sabha_number']:<4}{r['sittings']:<10}{r['scraped']:<10}"
                f"{r['extracted']:<11}{r['segmented']:<10}"
            )

        speech_stats = fetch_all(
            conn,
            """
            select
                count(*) as total,
                count(text_english) as translated,
                count(ner_processed_at) as ner_done,
                count(topics_processed_at) as topics_done,
                (select count(*) from sentiment) as sentiment_done,
                (select count(*) from speech_embeddings) as embeddings_done
            from speeches
            """,
        )[0]
        print()
        for key, value in speech_stats.items():
            print(f"{key:<16}{value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", help=f"one or more of: status, all, {', '.join(STAGES)}")
    parser.add_argument("--lok-sabha", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="scrape stage only: discover, don't download")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if "status" in args.stages:
        print_status()
        return

    stages = STAGES if "all" in args.stages else args.stages
    unknown = set(stages) - set(STAGES)
    if unknown:
        parser.error(f"unknown stage(s): {unknown}. valid stages: {STAGES}")

    for stage in stages:
        run_stage(stage, args.lok_sabha, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
