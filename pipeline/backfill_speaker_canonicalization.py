"""One-off: populate speaker_canonical for speeches segmented before that
column/pipeline/speakers.py existed, without re-running segmentation.

Usage:
    python -m pipeline.backfill_speaker_canonicalization
"""
import argparse
import logging

from pipeline.db import execute, fetch_all, get_conn
from pipeline.speakers import canonicalize_speaker

log = logging.getLogger("pipeline.backfill_speaker_canonicalization")


def run():
    with get_conn() as conn:
        rows = fetch_all(
            conn,
            """
            select sp.id, sp.speaker_raw, si.sitting_date
            from speeches sp join sittings si on si.id = sp.sitting_id
            where sp.speaker_raw is not null
            """,
        )
        log.info("backfilling speaker_canonical for %d speeches", len(rows))

        for i, row in enumerate(rows):
            canonical = canonicalize_speaker(row["speaker_raw"], row["sitting_date"])
            execute(
                conn, "update speeches set speaker_canonical = %s where id = %s",
                (canonical, row["id"]),
            )
            if i % 500 == 0:
                conn.commit()
        conn.commit()
        log.info("done")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run()


if __name__ == "__main__":
    main()
