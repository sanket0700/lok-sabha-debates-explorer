"""Regression tests for pipeline/segment.py — pure regex/logic, no model or
DB needed, so these run fast in CI. Each test is anchored to a real bug
found and fixed during development (see PROGRESS.md for the full story);
the point is to catch a *regression*, not to re-derive the finding.
"""
from unittest.mock import patch

from pipeline.segment import (
    SPEAKER_PATTERN,
    detect_language,
    segment_sitting_text,
)


class TestDetectLanguage:
    """Bug: Devanagari digits (०-९) and danda punctuation (। ॥) were being
    counted as "Hindi content", sending near-pure-English speeches through
    the Hindi->English translation model needlessly — confirmed to
    catastrophically corrupt a real government written-answer table.
    """

    def test_pure_english_with_stray_danda_stays_english(self):
        # Real case: OCR misread the pronoun "I" as the Devanagari danda "।"
        # a few times in an otherwise fully English floor speech.
        text = (
            "Now, the Home Minister in introducing this Bill attempted to invoke "
            "Nehru Ji in his support and । think that the expression that my friend "
            "was quoting in Tamil was that the devil can quote scriptures."
        )
        assert detect_language(text) == "en"

    def test_devanagari_numeral_footnote_marker_stays_english(self):
        # Real case: a Devanagari numeral used as a footnote marker in an
        # otherwise-English written answer, not real Hindi content.
        text = "The government has approved the scheme.१ Details are in the annexure."
        assert detect_language(text) == "en"

    def test_genuine_hindi_still_detected(self):
        text = "माननीय अध्यक्ष महोदय, मैं इस विषय पर सदन का ध्यान आकर्षित करना चाहता हूँ।"
        assert detect_language(text) == "hi"

    def test_genuine_code_switched_content_still_mixed(self):
        # A real handful of Hindi words embedded in an English speech is
        # legitimate code-switching and should still trigger translation.
        text = (
            "Sir, I want to raise a point. जय हिन्द, जय भारत। "
            "This scheme has helped many farmers across the country improve their income."
        )
        assert detect_language(text) == "mixed"


class TestSpeakerTitles:
    """Bug: SPEAKER_TITLES was missing several real, recurring printed
    labels, causing those turns to silently bleed into whichever speech
    matched immediately before them instead of starting their own turn.
    """

    def test_hon_speaker_matches(self):
        assert SPEAKER_PATTERN.search("HON. SPEAKER: Please continue.")

    def test_secretary_general_matches(self):
        assert SPEAKER_PATTERN.search("SECRETARY-GENERAL: Sir, I beg to lay on the Table...")

    def test_hindi_chairperson_matches(self):
        assert SPEAKER_PATTERN.search("माननीय सभापति : माननीय सदस्यगण, मुझे सूचित करना है")


class TestLLMFallbackHallucination:
    """Bug: the LLM segmentation fallback fabricated plausible-looking but
    fake speaker names when handed a chunk with no recognizable speaker
    boundary at all (e.g. continuous President's-Address-style prose),
    despite the prompt explicitly saying not to invent one.
    """

    def test_chunk_with_no_speaker_pattern_skips_llm_entirely(self):
        # No SPEAKER_PATTERN match anywhere in this text — should never
        # reach the LLM at all, regardless of what it might return.
        body = (
            "In the Chair] The Lok Sabha met at eleven of the clock. "
            "This House resolves that the President's address be discussed "
            "at length over the coming sessions with due consideration."
        )
        with patch("pipeline.segment.generate_json") as mock_llm:
            segment_sitting_text(body)
            mock_llm.assert_not_called()
