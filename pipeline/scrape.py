"""Stage 1: scrape Lok Sabha debate PDFs + sitting metadata from elibrary.sansad.in.

Scoped to the pilot Lok Sabhas only (config.LOK_SABHA_NUMBERS = [16, 17]).
Resumable: a sitting is skipped if its source_pdf_url already has a row in
`sittings`. Rate-limited via config.SCRAPE_REQUEST_DELAY_SECONDS.

NOTE: the original site this scraper targeted, eparlib.nic.in, was retired
and its content migrated to elibrary.sansad.in — a DSpace 7 app whose
collection browse pages are client-rendered Angular, not static HTML. This
hits the DSpace REST API (config.ELIBRARY_API_URL) directly instead:
discover/search/objects, filtered by the "loksabhanumber" facet and scoped
to the "Lok Sabha Debates (Text)" collection (config.ELIBRARY_DEBATES_SCOPE),
then one items/{id}?embed=bundles/bitstreams fetch per hit for the PDF link.
Both the scope UUID and facet name were confirmed by hand against the live
API — they (or the API shape) could still drift if the site is
restructured, so run with --limit 5 --dry-run first and confirm rows look
sane before a real run.

Usage:
    python -m pipeline.scrape --lok-sabha 16 --limit 20 --dry-run
    python -m pipeline.scrape --lok-sabha 16
    python -m pipeline.scrape --lok-sabha 17
"""
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import requests

from pipeline.config import (
    ELIBRARY_API_URL,
    ELIBRARY_DEBATES_SCOPE,
    LOK_SABHA_NUMBERS,
    PDF_DIR,
    SCRAPE_REQUEST_DELAY_SECONDS,
    SCRAPE_USER_AGENT,
)
from pipeline.db import execute, fetch_one, get_conn

log = logging.getLogger("pipeline.scrape")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": SCRAPE_USER_AGENT, "Accept": "application/json"})

PAGE_SIZE = 20


def _get_json(url: str, **params) -> dict:
    resp = SESSION.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(SCRAPE_REQUEST_DELAY_SECONDS)
    return resp.json()


def _first_metadata_value(metadata: dict, field: str):
    values = metadata.get(field)
    return values[0]["value"] if values else None


def discover_items(lok_sabha_number: int):
    """Page through the elibrary discover-search API, yielding item dicts."""
    page = 0
    while True:
        data = _get_json(
            f"{ELIBRARY_API_URL}/discover/search/objects",
            scope=ELIBRARY_DEBATES_SCOPE,
            size=PAGE_SIZE,
            page=page,
            **{"f.loksabhanumber": f"{lok_sabha_number},equals"},
        )
        search_result = data["_embedded"]["searchResult"]
        objects = search_result.get("_embedded", {}).get("objects", [])
        if not objects:
            return
        for obj in objects:
            yield obj["_embedded"]["indexableObject"]
        page += 1
        if page >= search_result["page"]["totalPages"]:
            return


def find_pdf_bitstream(item_id: str):
    """Return (pdf_url, filename) for an item's ORIGINAL-bundle PDF, or (None, None)."""
    data = _get_json(f"{ELIBRARY_API_URL}/core/items/{item_id}", embed="bundles/bitstreams")
    bundles = data.get("_embedded", {}).get("bundles", {}).get("_embedded", {}).get("bundles", [])
    for bundle in bundles:
        if bundle.get("name") != "ORIGINAL":
            continue
        bitstreams = (
            bundle.get("_embedded", {}).get("bitstreams", {}).get("_embedded", {}).get("bitstreams", [])
        )
        for bitstream in bitstreams:
            name = bitstream.get("name", "")
            if name.lower().endswith(".pdf"):
                return bitstream["_links"]["content"]["href"], name
    return None, None


def already_scraped(conn, pdf_url: str) -> bool:
    return fetch_one(conn, "select 1 from sittings where source_pdf_url = %s", (pdf_url,)) is not None


def download_pdf(pdf_url: str, filename: str, lok_sabha_number: int) -> Path:
    ls_dir = PDF_DIR / f"{lok_sabha_number:02d}"
    ls_dir.mkdir(parents=True, exist_ok=True)
    dest = ls_dir / filename
    if not dest.exists():
        resp = SESSION.get(pdf_url, timeout=60)
        resp.raise_for_status()
        time.sleep(SCRAPE_REQUEST_DELAY_SECONDS)
        dest.write_bytes(resp.content)
    return dest


def scrape_lok_sabha(lok_sabha_number: int, limit: int | None, dry_run: bool):
    count = 0
    with get_conn() as conn:
        for item in discover_items(lok_sabha_number):
            if limit is not None and count >= limit:
                break

            metadata = item.get("metadata", {})
            sitting_date_str = _first_metadata_value(metadata, "dc.date.issued")
            sitting_date = (
                datetime.strptime(sitting_date_str, "%Y-%m-%d").date() if sitting_date_str else None
            )

            try:
                pdf_url, filename = find_pdf_bitstream(item["id"])
            except requests.RequestException as exc:
                log.warning("failed to fetch item %s: %s", item["id"], exc)
                continue
            if pdf_url is None:
                log.warning("no pdf bitstream found for item %s (handle=%s)", item["id"], item.get("handle"))
                continue
            if already_scraped(conn, pdf_url):
                log.debug("already scraped, skipping: %s", pdf_url)
                continue

            log.info("[LS%s] %s (date=%s)", lok_sabha_number, pdf_url, sitting_date)
            if dry_run:
                count += 1
                continue

            try:
                pdf_path = download_pdf(pdf_url, filename, lok_sabha_number)
            except requests.RequestException as exc:
                log.warning("failed to download %s: %s", pdf_url, exc)
                continue

            execute(
                conn,
                """
                insert into sittings
                    (lok_sabha_number, sitting_date, source_pdf_url, source_pdf_path, scraped_at)
                values (%s, %s, %s, %s, now())
                on conflict (source_pdf_url) do nothing
                """,
                (lok_sabha_number, sitting_date, pdf_url, str(pdf_path)),
            )
            conn.commit()
            count += 1
    log.info("[LS%s] done: %d sittings processed", lok_sabha_number, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lok-sabha", type=int, choices=LOK_SABHA_NUMBERS, action="append")
    parser.add_argument("--limit", type=int, default=None, help="cap sittings per Lok Sabha (pilot-safe)")
    parser.add_argument("--dry-run", action="store_true", help="discover + log only, no download/insert")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    lok_sabhas = args.lok_sabha or LOK_SABHA_NUMBERS
    for ls in lok_sabhas:
        scrape_lok_sabha(ls, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
