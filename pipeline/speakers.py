"""Speaker canonicalization: resolve `speeches.speaker_raw` (as printed —
"Shri Narendra Modi", "The Prime Minister", "Modi ji"...) to one consistent
identity per real person, so speaker_stats shows a cumulative record per
person instead of one row per alias.

Two layers, in order:
  1. Honorific/possessive stripping (pipeline/text_normalize.py) — handles
     salutation variants of an already-named speaker ("Shri Narendra Modi"
     vs "Narendra Modi" vs "Modi ji").
  2. A small, explicit, date-scoped role→person table — handles a printed
     transcript attribution that's a *role*, not a name at all ("The Prime
     Minister", "Honourable Prime Minister"). This can't be inferred from
     the text alone (confirmed: unlike every other minister role in this
     corpus, which is always printed with the name attached in parens, e.g.
     "THE MINISTER OF RAILWAYS (SHRI SURESH PRABHU)", the PM role appears
     completely bare everywhere it occurs) — it's real-world domain
     knowledge (who held the role, and when), so it's a manually maintained
     table, not something auto-derived. Deliberately small: only add an
     entry once a bare-role duplicate is actually confirmed in the data
     (checked by hand: Speaker/Deputy Speaker/Leader of Opposition don't
     have this problem in the pilot corpus — the Chair is consistently
     referred to by title only, never conflated with a named mention, so
     there's nothing to consolidate there).
"""
from datetime import date

from pipeline.text_normalize import strip_person_honorifics

# (role text after strip_person_honorifics, canonical person, valid_from, valid_to)
# valid_to=None means "still holds as of this pipeline's pilot scope".
SPEAKER_ROLE_ALIASES: list[tuple[str, str, date, date | None]] = [
    ("prime minister", "narendra modi", date(2014, 5, 26), None),
]


def canonicalize_speaker(speaker_raw: str | None, sitting_date: date | None) -> str | None:
    if not speaker_raw:
        return None
    normalized = strip_person_honorifics(speaker_raw)
    for role, canonical, valid_from, valid_to in SPEAKER_ROLE_ALIASES:
        if normalized != role:
            continue
        if sitting_date is None:
            continue  # can't confirm the date is in range — leave as the role, not a guess
        if valid_from <= sitting_date and (valid_to is None or sitting_date <= valid_to):
            return canonical
    return normalized


# Used by pipeline/enrich_rhetoric.py to bias rhetorical-mode classification:
# a minister explaining/defending their own portfolio reads differently than
# an unattributed floor speech, and that's real signal a text-only classifier
# can't recover on its own.
#
# (speaker_canonical, portfolio text, lok_sabha_number, valid_from, valid_to)
# — deliberately keyed per Lok Sabha term, not just per person: the same
# person can hold different portfolios across terms (confirmed in this
# corpus: Nirmala Sitharaman appears in LS16/17/18 and was NOT Finance
# Minister the whole time; Pralhad Joshi's LS16 floor speeches predate him
# holding any ministerial role at all) or be reshuffled mid-term (a real,
# separate case this same per-term/per-portfolio-entry structure supports —
# just add another entry for the same person+Lok Sabha with a narrower
# valid_from/valid_to). Only entries verified against a real source
# (pmindia.gov.in / Cabinet Secretariat portfolio listings, checked by hand)
# are included — a person reappearing in an earlier, unlisted term is a
# deliberate gap, not an oversight; see PROGRESS.md for what's verified vs.
# what would need further research to extend.
MINISTER_ROLE_HISTORY: list[tuple[str, str, int, date, date | None]] = [
    # Modi 3.0 council of ministers, sworn in 2024-06-09 — verified against
    # the official PM India / Cabinet Secretariat portfolio listing.
    ("amit shah", "Minister of Home Affairs and Cooperation", 18, date(2024, 6, 9), None),
    ("nirmala sitharaman", "Minister of Finance and Corporate Affairs", 18, date(2024, 6, 9), None),
    ("nitin jairam gadkari", "Minister of Road Transport and Highways", 18, date(2024, 6, 9), None),
    ("hardeep singh puri", "Minister of Petroleum and Natural Gas", 18, date(2024, 6, 9), None),
    ("jagat prakash nadda", "Minister of Health and Family Welfare", 18, date(2024, 6, 9), None),
    ("pralhad joshi", "Minister of Consumer Affairs, Food and Public Distribution", 18, date(2024, 6, 9), None),
    ("tokhan sahu", "Minister of State for Housing and Urban Affairs", 18, date(2024, 6, 9), None),
    ("shobha karandlaje", "Minister of State for MSME and Labour", 18, date(2024, 6, 9), None),
    ("kamlesh paswan", "Minister of State for Rural Development", 18, date(2024, 6, 9), None),
    ("dra sekhar pemmasani", "Minister of State for Rural Development and Communications", 18, date(2024, 6, 9), None),
    ("ram nath thakur", "Minister of State for Agriculture and Farmers Welfare", 18, date(2024, 6, 9), None),
    ("nityanand rai", "Minister of State for Home Affairs", 18, date(2024, 6, 9), None),
]


def resolve_minister_role(
    speaker_canonical: str | None, lok_sabha_number: int | None, sitting_date: date | None
) -> str | None:
    if not speaker_canonical or not lok_sabha_number or not sitting_date:
        return None
    for person, role, ls, valid_from, valid_to in MINISTER_ROLE_HISTORY:
        if person != speaker_canonical or ls != lok_sabha_number:
            continue
        if valid_from <= sitting_date and (valid_to is None or sitting_date <= valid_to):
            return role
    return None
