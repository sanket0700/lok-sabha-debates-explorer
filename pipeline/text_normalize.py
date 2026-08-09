"""Shared person-name normalization, used by both entity canonicalization
(pipeline/enrich_ner.py) and speaker canonicalization (pipeline/speakers.py).

Kept in one place so the two don't quietly drift apart — a name should
normalize the same way whether it came from spaCy NER or from a
segmentation speaker label.
"""
import re

_HONORIFIC_PREFIX = re.compile(
    r"^(the|his\s+excellency|her\s+excellency|shri|shrimati|smt\.?|dr\.?|kumari|prof\.?|"
    r"mr\.?|mrs\.?|ms\.?|honourable|hon\.?ble|hon\.?)\s+"
)
# Space-separated suffixes only — fused forms like "Modiji" are left alone:
# some real surnames legitimately end in "-ji" (e.g. "Balaji"), and blind
# suffix removal without a word boundary risks corrupting them.
_HONORIFIC_SUFFIX = re.compile(r"\s+(ji|saab|sahib|saheb)$")
_POSSESSIVE_SUFFIX = re.compile(r"['’]s$")


def strip_person_honorifics(text: str) -> str:
    """Lowercase/trim a person name and strip honorific prefixes/suffixes and
    a trailing possessive. Does NOT attempt to merge a bare surname/short name
    into a full name (e.g. "Modi" vs "Narendra Modi") — several surnames in
    this corpus belong to more than one real, distinct person, so that merge
    isn't safe by string matching alone.
    """
    normalized = text.strip().lower()
    prev = None
    while prev != normalized:
        prev = normalized
        normalized = _POSSESSIVE_SUFFIX.sub("", normalized)
        normalized = _HONORIFIC_SUFFIX.sub("", normalized)
        normalized = _HONORIFIC_PREFIX.sub("", normalized)
    normalized = normalized.strip()
    return normalized or text.strip().lower()
