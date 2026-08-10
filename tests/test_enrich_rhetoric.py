"""Regression tests for pipeline/enrich_rhetoric.py's structural bypass
patterns — pure regex, no model or DB needed. Each pattern here was added
because a generic zero-shot classifier demonstrably could not tell this
boilerplate apart from genuine rhetorical content (see PROGRESS.md for the
prototyping history — two iterations found broken before this shipped).
"""
from pipeline.enrich_rhetoric import RHETORIC_MIN_CHARS, _structural_bypass, classify_one


class TestStructuralBypass:
    def test_speaker_role_bypass(self):
        assert _structural_bypass("माननीय अध्यक्ष", "I have given the ruling. You sit down.") == "speaker-role"

    def test_english_speaker_role_bypass(self):
        assert _structural_bypass("secretary-general", "Sir, I beg to lay on the table a copy.") == "speaker-role"

    def test_written_question_bypass(self):
        text = "Will the Minister of Railways be pleased to state: (a) the status of the project"
        assert _structural_bypass(None, text) == "written-question"

    def test_written_answer_bypass(self):
        text = "(a) and (b): The scheme is being implemented across the country."
        assert _structural_bypass(None, text) == "written-answer"

    def test_written_answer_ocr_digit_confusion_bypass(self):
        # Real case: "(a)" OCR-misread as "(8)" — same class of noise as
        # "SHRI 8. MAHTAB" already documented for segmentation.
        text = "(8) and (b): As per the National Cooperative Database (NCD)..."
        assert _structural_bypass(None, text) == "written-answer"

    def test_motion_opener_bypass(self):
        assert _structural_bypass(None, "Sir, I beg to move the following:- That in pursuance of...") == "motion-opener"

    def test_motion_opener_pipe_for_i_ocr_confusion_bypass(self):
        # Real case: extracted "I" became a pipe character "|" — same
        # failure mode already fixed once for detect_language().
        assert _structural_bypass(None, "Sir, | beg to lay on the Table a copy of the report.") == "motion-opener"

    def test_genuine_floor_speech_not_bypassed(self):
        text = "The whole of India is burning because of your policies and it is time you listened."
        assert _structural_bypass(None, text) is None


class TestClassifyOne:
    def test_too_short_gets_sentinel_label_no_classifier_call(self):
        text = "Why is that relevant?"
        assert len(text) < RHETORIC_MIN_CHARS
        label, score, method = classify_one(None, None, 18, None, text)
        assert label == "too_short"
        assert score is None
        assert method == "skipped:too_short"

    def test_structural_bypass_short_circuits_before_classifier_call(self):
        # classifier is None here deliberately — if this reached the
        # classifier it would crash, so a clean return proves the bypass
        # fired instead of falling through.
        label, score, method = classify_one(
            None, "माननीय अध्यक्ष", 18, None, "I have given the ruling. Please sit down."
        )
        assert label == "procedural"
        assert score is None
        assert method == "bypass:speaker-role"
