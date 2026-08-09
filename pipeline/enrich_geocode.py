"""Stage 5b: geocode place-like entities (GPE/LOC/FAC) via Nominatim.

Same free, no-API-key geocoder the old repo used (it also tried the paid
Google Maps API — dropped here per the "no paid API" decision). An
in-process cache avoids re-geocoding the same place name twice within a
run, since many entities share a canonical_text (e.g. "India" mentioned in
hundreds of speeches) — mirrors the old repo's dedup-by-address approach,
just applied to outgoing requests instead of only to the output CSV.

Idempotent: only processes entities of a geocodable type with no row yet in
`locations`.

Usage:
    python -m pipeline.enrich_geocode [--limit 2000]
"""
import argparse
import logging
import time

import requests

from pipeline.config import (
    ENTITY_TYPES_TO_GEOCODE,
    NOMINATIM_REQUEST_DELAY_SECONDS,
    NOMINATIM_USER_AGENT,
)
from pipeline.db import execute, fetch_all, get_conn

log = logging.getLogger("pipeline.enrich_geocode")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Nominatim's own "importance" score (roughly 0-1), applied to results from
# BOTH passes. Turns out generic/junk NER extractions ("The", "North East",
# scheme names misparsed as places) don't just cause problems on the
# worldwide fallback — India alone has enough OSM data density that a
# free-text search for almost any word matches *something* (a shop, a
# historic-site marker, a random building), so the India-restricted pass
# needs the same filter. Measured by hand: genuine places (Delhi 0.70,
# Mumbai 0.72, Mahabubabad 0.42, the Ganga river 0.66, Nagar Haveli 0.25)
# score at least ~0.08; true junk matches ("the"→a railway junction,
# "north east"→a shop, "greenfield highway"→a random building) score
# 0.00004-0.00008, six orders of magnitude lower. 0.01 sits cleanly between
# the two clusters observed on both passes.
MIN_IMPORTANCE = 0.01

# The worldwide (non-India) pass needs a *much* stricter floor than the India
# pass. A full re-run with MIN_IMPORTANCE=0.01 on both passes showed real
# countries resolving correctly (Nepal, Sri Lanka, UAE, Russia...) but also
# generic Indian-context terms matching random low-relevance places abroad at
# moderate importance: "National Highways"→a road in England (0.08), "Kharif"
# (an Indian crop season)→a district in Yemen (0.38), "State"→a building in
# Boston (0.37), "Covid"→a place literally named that in New Zealand (0.04),
# "Latin America"→a garden in California (0.07). Every confirmed-genuine
# country/major city measured, by contrast, scored 0.76-0.90 (China, Pakistan,
# Bangladesh, Beijing, Washington, Nepal). 0.5 sits cleanly between those two
# clusters — comfortably rejects the false matches above without excluding any
# real country/major-city result seen so far. Deliberately not applied to the
# India-restricted pass, which needs to stay at the lower MIN_IMPORTANCE floor
# to keep matching legitimate small Indian places (districts, rivers, dams).
WORLDWIDE_MIN_IMPORTANCE = 0.5

# A handful of common English words spaCy's GPE/LOC/FAC tagger occasionally
# misfires on, that MIN_IMPORTANCE alone doesn't reliably catch: they can
# coincidentally match a *real* (if obscure/irrelevant) place somewhere in
# the world at moderate importance — confirmed "the" matches a real Brazilian
# airport at 0.42, "north east" a real New York town at 0.44, both well
# above the junk/legitimate importance split observed elsewhere. This is a
# deliberately small, conservative list of clear non-place words, not an
# attempt at general disambiguation — genuinely ambiguous cases (e.g. "China"
# the country vs. a same-named Indian district; "Kashmir" the region vs. a
# same-named Rajasthan village) remain an open problem beyond this fix's
# scope, noted in PROGRESS.md rather than solved here.
NON_PLACE_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "here", "there",
    "north", "south", "east", "west",
    "north east", "north west", "south east", "south west",
    "northeast", "northwest", "southeast", "southwest",
}

_cache: dict[str, tuple | None] = {}


def _search(place: str, countrycodes: str | None):
    params = {"q": place, "format": "json", "limit": 1, "accept-language": "en"}
    if countrycodes:
        params["countrycodes"] = countrycodes
    resp = requests.get(
        NOMINATIM_URL,
        params=params,
        headers={"User-Agent": NOMINATIM_USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    time.sleep(NOMINATIM_REQUEST_DELAY_SECONDS)
    return resp.json()


def geocode(place: str):
    """Two-pass: India-restricted first, worldwide fallback second.

    Every entity here comes from an Indian Lok Sabha debate transcript, so most
    ambiguous names (e.g. "Central Hall") should resolve to their Indian match
    rather than whatever ranks highest globally — that's what the India-only pass
    is for. But debates also genuinely discuss foreign places (China, Pakistan,
    Washington, the UN...), and a hard `countrycodes=in` restriction with no
    fallback was forcing *every* geocoded result into India, silently dropping or
    mis-locating anything actually foreign (confirmed: 0 of 2226 distinct resolved
    addresses were outside India). The worldwide pass only runs when the
    India-restricted search finds nothing, so it doesn't reintroduce the original
    ambiguity problem for names that do have a real Indian match.
    """
    if place in _cache:
        return _cache[place]
    if place in NON_PLACE_STOPWORDS:
        _cache[place] = None
        return None

    def _good(matches, threshold):
        return [r for r in matches if float(r.get("importance", 0)) >= threshold]

    results = []
    try:
        results = _good(_search(place, countrycodes="in"), MIN_IMPORTANCE)
        if not results:
            results = _good(_search(place, countrycodes=None), WORLDWIDE_MIN_IMPORTANCE)
    except requests.RequestException as exc:
        log.warning("geocode failed for %r: %s", place, exc)
        results = []

    result = None
    if results:
        top = results[0]
        result = (top["display_name"], float(top["lat"]), float(top["lon"]))
    _cache[place] = result
    return result


def run(limit: int | None):
    placeholders = ", ".join(["%s"] * len(ENTITY_TYPES_TO_GEOCODE))
    with get_conn() as conn:
        query = f"""
            select e.id, e.canonical_text from entities e
            left join locations l on l.entity_id = e.id
            where e.type in ({placeholders}) and l.entity_id is null
            order by e.id
        """
        params = list(ENTITY_TYPES_TO_GEOCODE)
        if limit is not None:
            query += " limit %s"
            params.append(limit)

        rows = fetch_all(conn, query, params)
        log.info("geocoding %d entity mentions", len(rows))

        for i, row in enumerate(rows):
            result = geocode(row["canonical_text"])
            if result is not None:
                formatted_address, lat, lon = result
                execute(
                    conn,
                    """
                    insert into locations (entity_id, formatted_address, lat, lon)
                    values (%s, %s, %s, %s)
                    on conflict (entity_id) do nothing
                    """,
                    (row["id"], formatted_address, lat, lon),
                )
            if i % 100 == 0:
                conn.commit()
                log.info("geocoded %d/%d", i, len(rows))
        conn.commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run(args.limit)


if __name__ == "__main__":
    main()
