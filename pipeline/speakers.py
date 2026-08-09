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
