"""One-off: reapply enrich_ner.canonicalize() to PERSON entities extracted
before honorific-stripping existed, without re-running NER.

Cheap (pure string transform on already-extracted entity text), so a direct
Python backfill instead of a SQL regex — keeps the normalization logic in
exactly one place (enrich_ner.canonicalize) rather than duplicating it in SQL.

Usage:
    python -m pipeline.backfill_entity_canonicalization
"""
import argparse
import logging

from pipeline.db import execute, fetch_all, get_conn
from pipeline.enrich_ner import canonicalize

log = logging.getLogger("pipeline.backfill_entity_canonicalization")


def run():
    with get_conn() as conn:
        rows = fetch_all(conn, "select id, text, canonical_text from entities where type = 'PERSON'")
        log.info("checking %d PERSON entities", len(rows))

        changed = 0
        for i, row in enumerate(rows):
            new_canonical = canonicalize(row["text"], "PERSON")
            if new_canonical != row["canonical_text"]:
                execute(
                    conn, "update entities set canonical_text = %s where id = %s",
                    (new_canonical, row["id"]),
                )
                changed += 1
            if i % 1000 == 0:
                conn.commit()
        conn.commit()
        log.info("updated %d/%d entities", changed, len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run()


if __name__ == "__main__":
    main()
