"""
brain — Recall Scorer Unit Tests

Tests for the pure computation module servers/recall_scorer.py which
implements the layered evaluation engine (regex, BART, embeddings, scorer).

These tests verify the COMPUTATION, not the database plumbing:
  - Regex patterns fire on expected inputs
  - Score accumulation produces correct signs
  - Evidence-to-precision mapping works at boundaries
  - Signal classification covers all branches
  - Empty/degraded inputs don't crash

== TEST INTEGRITY RULE ==

If any test fails: STOP. Do not change the test OR the code. Report to Tom
what the test expected vs what the code returned. Ask whether the test
expectation is wrong or the code has a bug. Wait for answer before proceeding.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.recall_scorer import (
    compute_regex_signals,
    score_recall,
    evidence_to_precision,
    classify_followup_signal,
    determine_match_method,
    _empty_bart,
    _empty_emb,
)


class TestRegexSignals(unittest.TestCase):
    """Tests for L0 regex pattern detection."""

    def test_affirm_strong_perfect(self):
        """Strong affirmation: 'Perfect, and how...'

        VERIFIES: AFFIRM_STRONG fires on direct agreement word + continuation.
        SIGNALS: If this misses, the most common agreement form is undetected.
        FUTURE: Add more continuation words if real conversations show misses.
        """
        s = compute_regex_signals("Perfect, and how do we handle shared state?")
        self.assertGreater(s["affirm_strong"], 0)

    def test_affirm_strong_maps_to(self):
        """'Maps to' pattern — the paraphrase agreement that was missing in v2.

        VERIFIES: AFFIRM_STRONG catches 'maps to what I was thinking'.
        SIGNALS: If this misses, paraphrase agreement (P3 scenario) regresses to undetected.
        FUTURE: No changes needed — this was the key fix from v2 to v3.
        """
        s = compute_regex_signals("That maps to what I was thinking.")
        self.assertGreater(s["affirm_strong"], 0)

    def test_affirm_strong_good_framing(self):
        """'Good framing' pattern — the subtle agreement that was missing in v2.

        VERIFIES: AFFIRM_STRONG catches 'That's a good framing'.
        SIGNALS: If this misses, P6 (subtle recognition) regresses.
        FUTURE: No changes needed.
        """
        s = compute_regex_signals("That's a good framing. So the module becomes a recognizer?")
        self.assertGreater(s["affirm_strong"], 0)

    def test_affirm_strong_makes_sense(self):
        """'Makes sense' is a strong affirmation.

        VERIFIES: Common agreement phrase detected.
        SIGNALS: Basic regex coverage gap if this fails.
        FUTURE: No changes needed.
        """
        s = compute_regex_signals("That makes sense. How many lines is brain.py now?")
        self.assertGreater(s["affirm_strong"], 0)

    def test_redirect_strong(self):
        """Strong redirect: 'Not what I asked about'.

        VERIFIES: REDIRECT_STRONG fires on explicit topic rejection.
        SIGNALS: If this misses, clear negative cases will score neutral.
        FUTURE: No changes needed.
        """
        s = compute_regex_signals("No, that's not what I asked about. I need help with Docker.")
        self.assertGreater(s["redirect_strong"], 0)

    def test_complaint_pattern(self):
        """Complaint: 'Why did you bring that up?'

        VERIFIES: COMPLAINT fires when user objects to recall itself.
        SIGNALS: Missed complaints mean reinforcement bias goes undetected.
        FUTURE: No changes needed.
        """
        s = compute_regex_signals("Why did you bring up the brain philosophy?")
        self.assertGreater(s["complaint"], 0)

    def test_extension_pattern(self):
        """Extension: 'Should we also apply this pattern?'

        VERIFIES: EXTENSION fires when user extends recalled content.
        SIGNALS: Extensions are strong positive signals — missing them hurts recall.
        FUTURE: No changes needed.
        """
        s = compute_regex_signals("Should we also apply this pattern to the recall module?")
        self.assertGreater(s["extension"], 0)

    def test_opening_word_positive(self):
        """Positive opening word classification.

        VERIFIES: fup_opens_positive correctly classifies agreement openers.
        SIGNALS: Opening word is a fast short-circuit signal.
        FUTURE: No changes needed.
        """
        s = compute_regex_signals("Yes, exactly — check the brain first.")
        self.assertTrue(s["fup_opens_positive"])
        self.assertFalse(s["fup_opens_negative"])

    def test_opening_word_negative(self):
        """Negative opening word classification.

        VERIFIES: fup_opens_negative correctly classifies rejection openers.
        SIGNALS: Opening word is a fast short-circuit signal.
        FUTURE: No changes needed.
        """
        s = compute_regex_signals("No, that's not related to my question.")
        self.assertTrue(s["fup_opens_negative"])
        self.assertFalse(s["fup_opens_positive"])

    def test_no_signals_on_neutral(self):
        """Neutral message produces no strong signals.

        VERIFIES: Neutral messages don't false-positive on intent patterns.
        SIGNALS: If affirm or redirect fire on neutral text, scoring is noisy.
        FUTURE: No changes needed.
        """
        s = compute_regex_signals("Can you actually help me with the code?")
        # "actually" may fire mild redirect, but strong should be zero
        self.assertEqual(s["affirm_strong"], 0)
        self.assertEqual(s["redirect_strong"], 0)


class TestScorer(unittest.TestCase):
    """Tests for the combined score_recall function."""

    def test_score_positive_scenario(self):
        """Clear positive signals produce positive evidence.

        VERIFIES: Strong affirm + extension + positive opener → evidence > 0.
        SIGNALS: If evidence is negative here, the weight calibration is wrong.
        FUTURE: No changes needed.
        """
        regex = compute_regex_signals("Perfect, and should we also apply this to the recall module?")
        bart = _empty_bart()  # No BART — regex-only
        emb = _empty_emb()
        emb["resp_words"] = 50
        emb["depth_ratio"] = 0.4

        evidence, confidence, reasons, n = score_recall(regex, bart, emb)
        self.assertGreater(evidence, 0, "Positive scenario should have positive evidence")
        self.assertGreater(confidence, 0, "Should have some confidence")

    def test_score_negative_scenario(self):
        """Clear negative signals produce negative evidence.

        VERIFIES: Strong redirect + complaint + negative opener → evidence < 0.
        SIGNALS: If evidence is positive here, negative weighting is broken.
        FUTURE: No changes needed.
        """
        regex = compute_regex_signals("No, that's not what I asked about. Why did you bring that up?")
        bart = _empty_bart()
        emb = _empty_emb()
        emb["resp_words"] = 50
        emb["depth_ratio"] = 0.4

        evidence, confidence, reasons, n = score_recall(regex, bart, emb)
        self.assertLess(evidence, 0, "Negative scenario should have negative evidence")

    def test_score_ultra_short_response(self):
        """Ultra-short response gets penalized.

        VERIFIES: resp_words < 5 adds a strong negative signal.
        SIGNALS: If this doesn't produce negative evidence, thin responses go undetected.
        FUTURE: No changes needed.
        """
        regex = compute_regex_signals("Can you help?")
        bart = _empty_bart()
        emb = _empty_emb()
        emb["resp_words"] = 2  # "OK."
        emb["depth_ratio"] = 0.01

        evidence, confidence, reasons, n = score_recall(regex, bart, emb)
        self.assertLess(evidence, 0, "Ultra-short response should be negative")

    def test_score_returns_reasons(self):
        """Score function returns structured reasons.

        VERIFIES: Reasons list is non-empty and correctly structured.
        SIGNALS: If reasons are missing, evaluation_metadata will be empty.
        FUTURE: No changes needed.
        """
        regex = compute_regex_signals("That makes sense.")
        evidence, confidence, reasons, n = score_recall(regex, _empty_bart(), _empty_emb())
        self.assertGreater(len(reasons), 0)
        # Each reason is (weight, text, direction)
        for w, r, d in reasons:
            self.assertIsInstance(w, float)
            self.assertIsInstance(r, str)
            self.assertIn(d, ("+", "-", "?"))


class TestEvidenceMapping(unittest.TestCase):
    """Tests for evidence_to_precision and classify_followup_signal."""

    def test_evidence_to_precision_positive(self):
        """Positive evidence maps to precision > 0.5.

        VERIFIES: evidence +0.5 with high confidence → precision around 0.75.
        SIGNALS: If mapping is wrong, recall weights will be inverted.
        FUTURE: No changes needed.
        """
        p = evidence_to_precision(0.5, 0.8)
        self.assertIsNotNone(p)
        self.assertGreater(p, 0.5)

    def test_evidence_to_precision_negative(self):
        """Negative evidence maps to precision < 0.5.

        VERIFIES: evidence -0.5 with high confidence → precision around 0.25.
        SIGNALS: If mapping is wrong, bad recalls will be reinforced.
        FUTURE: No changes needed.
        """
        p = evidence_to_precision(-0.5, 0.8)
        self.assertIsNotNone(p)
        self.assertLess(p, 0.5)

    def test_evidence_to_precision_low_confidence(self):
        """Low confidence returns None — can't make a determination.

        VERIFIES: confidence < 0.15 → None (no score stored).
        SIGNALS: If this returns a score, uncertain evaluations pollute precision data.
        FUTURE: No changes needed.
        """
        p = evidence_to_precision(0.5, 0.1)
        self.assertIsNone(p)

    def test_classify_positive(self):
        """Positive evidence + sufficient confidence → 'positive' signal.

        VERIFIES: classify_followup_signal returns 'positive' for evidence > 0.1.
        SIGNALS: If wrong, followup_signal column has incorrect values.
        FUTURE: No changes needed.
        """
        self.assertEqual(classify_followup_signal(0.5, 0.5), "positive")

    def test_classify_negative(self):
        """Negative evidence → 'negative' signal.

        VERIFIES: classify_followup_signal returns 'negative' for evidence < -0.1.
        SIGNALS: If wrong, negative signals not counted in summary.
        FUTURE: No changes needed.
        """
        self.assertEqual(classify_followup_signal(-0.5, 0.5), "negative")

    def test_classify_neutral(self):
        """Near-zero evidence → 'neutral' signal.

        VERIFIES: Evidence between -0.1 and 0.1 with confidence → neutral.
        SIGNALS: Ambiguous cases correctly classified as neutral.
        FUTURE: No changes needed.
        """
        self.assertEqual(classify_followup_signal(0.0, 0.5), "neutral")

    def test_classify_uncertain(self):
        """Low confidence → 'uncertain' signal regardless of evidence.

        VERIFIES: confidence < 0.15 always returns 'ask_operator'.
        SIGNALS: If uncertain cases get classified as positive/negative, data quality drops.
        ADJUSTED: 'uncertain' renamed to 'ask_operator' (2026-03-22) — genuinely uncertain
        cases should ask the operator for explicit feedback instead of guessing.
        """
        self.assertEqual(classify_followup_signal(0.9, 0.1), "ask_operator")
        self.assertEqual(classify_followup_signal(-0.9, 0.1), "ask_operator")

    def test_determine_match_method(self):
        """Match method string reflects active layers.

        VERIFIES: All combinations produce correct method strings.
        SIGNALS: If wrong, we can't tell which layers were active for a given evaluation.
        FUTURE: No changes needed.
        """
        self.assertEqual(determine_match_method(True, True), "regex+emb+bart")
        self.assertEqual(determine_match_method(False, True), "regex+emb")
        self.assertEqual(determine_match_method(True, False), "regex+bart")
        self.assertEqual(determine_match_method(False, False), "regex")


if __name__ == "__main__":
    unittest.main()
