"""Regression tests for pipeline/speakers.py — pure logic, no model or DB
needed.
"""
from datetime import date

from pipeline.speakers import canonicalize_speaker, resolve_minister_role


class TestCanonicalizeSpeaker:
    def test_pm_role_resolves_to_person_within_term(self):
        assert canonicalize_speaker("The Prime Minister", date(2020, 1, 1)) == "narendra modi"

    def test_pm_role_unresolved_without_date(self):
        # Can't confirm the date is in range — leave as the (honorific-
        # stripped) role, not a guess at the person.
        assert canonicalize_speaker("The Prime Minister", None) == "prime minister"

    def test_honorific_stripped(self):
        assert canonicalize_speaker("Shri Narendra Modi", date(2020, 1, 1)) == "narendra modi"


class TestResolveMinisterRole:
    """Bug: a flat {name: role} mapping breaks the moment a minister is
    reshuffled mid-term, or the same person appears in an earlier Lok
    Sabha term under a different (or no) portfolio — confirmed real in this
    corpus (Pralhad Joshi, Nirmala Sitharaman, Hardeep Singh Puri all
    appear across multiple terms). Role history must be scoped per
    (person, Lok Sabha term, date range), not just per person.
    """

    def test_current_term_minister_resolves(self):
        role = resolve_minister_role("amit shah", 18, date(2025, 1, 1))
        assert role == "Minister of Home Affairs and Cooperation"

    def test_same_person_earlier_term_not_resolved(self):
        # Real case: Nirmala Sitharaman appears in LS16 (2017) and LS17,
        # before/outside her verified LS18 Finance Minister tenure — no
        # entry exists for those terms, so no role should be guessed.
        role = resolve_minister_role("nirmala sitharaman", 16, date(2017, 8, 8))
        assert role is None

    def test_before_verified_tenure_start_not_resolved(self):
        role = resolve_minister_role("amit shah", 18, date(2024, 1, 1))
        assert role is None

    def test_unknown_person_not_resolved(self):
        assert resolve_minister_role("some random backbench mp", 18, date(2025, 1, 1)) is None

    def test_missing_inputs_not_resolved(self):
        assert resolve_minister_role(None, 18, date(2025, 1, 1)) is None
        assert resolve_minister_role("amit shah", None, date(2025, 1, 1)) is None
        assert resolve_minister_role("amit shah", 18, None) is None
