"""Tests for the unified recall scoring module.

Verifies the research-grounded scoring formula:
  final = semantic_base * (1.0 + recency + emotion + frequency_penalty + confidence)

Key properties tested:
  - Semantic is gatekeeper: zero base = zero final
  - Modulators are bounded: ~0.81x to ~1.55x
  - Frequency is a penalty, not a boost
  - Created_at drives recency, not last_accessed
  - None/missing values handled gracefully
"""

import unittest
import sys
from datetime import datetime, timedelta

sys.path.insert(0, '.')

from servers.recall_scoring import (
    unified_score,
    freshness_from_created,
    frequency_penalty,
    confidence_boost,
)


class TestUnifiedScore(unittest.TestCase):
    """Core scoring formula tests."""

    def test_zero_semantic_returns_zero(self):
        """Zero relevance = zero final, regardless of other signals."""
        self.assertEqual(unified_score(0.0, emotion=1.0, access_count=0, confidence=1.0), 0.0)

    def test_negative_semantic_returns_zero(self):
        self.assertEqual(unified_score(-0.5), 0.0)

    def test_pure_semantic_passes_through(self):
        """With no modulators, score ≈ base (modulator = 1.0)."""
        score = unified_score(0.7)
        self.assertAlmostEqual(score, 0.7, places=2)

    def test_emotion_cannot_boost_zero(self):
        """GANE model: emotion amplifies priority, can't create it."""
        self.assertEqual(unified_score(0.0, emotion=1.0), 0.0)

    def test_emotion_amplifies_relevance(self):
        """High emotion on relevant node > same node without emotion."""
        base = unified_score(0.7, emotion=0.0)
        with_emotion = unified_score(0.7, emotion=0.9)
        self.assertGreater(with_emotion, base)

    def test_modulator_upper_bound(self):
        """Max modulator should not exceed ~1.55x."""
        # Best case: just created, max emotion, low access, high confidence
        now = datetime.utcnow().isoformat() + 'Z'
        score = unified_score(1.0, created_at=now, emotion=1.0,
                              access_count=0, confidence=1.0)
        self.assertLess(score, 1.6)
        self.assertGreater(score, 1.4)

    def test_modulator_lower_bound(self):
        """Min modulator should not go below ~0.81x."""
        old = (datetime.utcnow() - timedelta(days=365)).isoformat() + 'Z'
        score = unified_score(1.0, created_at=old, emotion=0.0,
                              access_count=1000, confidence=0.1)
        self.assertGreater(score, 0.75)
        self.assertLess(score, 1.0)

    def test_high_frequency_penalizes(self):
        """Hub nodes (high access) should score lower than low-access."""
        low_access = unified_score(0.7, access_count=5)
        high_access = unified_score(0.7, access_count=500)
        self.assertGreater(low_access, high_access)

    def test_recent_beats_old(self):
        """Recently created node beats old node, same semantic score."""
        now = datetime.utcnow().isoformat() + 'Z'
        old = (datetime.utcnow() - timedelta(days=60)).isoformat() + 'Z'
        recent = unified_score(0.7, created_at=now)
        ancient = unified_score(0.7, created_at=old)
        self.assertGreater(recent, ancient)

    def test_all_none_defaults(self):
        """All optional params as None should not crash."""
        score = unified_score(0.5, created_at=None, emotion=0,
                              access_count=0, confidence=None)
        self.assertAlmostEqual(score, 0.5, places=2)


class TestFreshnessFromCreated(unittest.TestCase):
    """Recency from creation time, not access time."""

    def test_just_created(self):
        now = datetime.utcnow().isoformat() + 'Z'
        self.assertAlmostEqual(freshness_from_created(now), 0.30, places=2)

    def test_one_day_old(self):
        yesterday = (datetime.utcnow() - timedelta(hours=20)).isoformat() + 'Z'
        boost = freshness_from_created(yesterday)
        self.assertGreater(boost, 0.10)
        self.assertLess(boost, 0.25)

    def test_one_week_old(self):
        week_ago = (datetime.utcnow() - timedelta(days=5)).isoformat() + 'Z'
        boost = freshness_from_created(week_ago)
        self.assertGreater(boost, 0.0)
        self.assertLess(boost, 0.15)

    def test_ancient(self):
        old = (datetime.utcnow() - timedelta(days=365)).isoformat() + 'Z'
        self.assertEqual(freshness_from_created(old), 0.0)

    def test_none_returns_zero(self):
        self.assertEqual(freshness_from_created(None), 0.0)

    def test_invalid_string_returns_zero(self):
        self.assertEqual(freshness_from_created("not a date"), 0.0)

    def test_empty_string_returns_zero(self):
        self.assertEqual(freshness_from_created(""), 0.0)


class TestFrequencyPenalty(unittest.TestCase):
    """Hub penalty — high access = low diagnostic value."""

    def test_below_threshold_no_penalty(self):
        self.assertEqual(frequency_penalty(5), 0.0)
        self.assertEqual(frequency_penalty(20), 0.0)

    def test_zero_access_no_penalty(self):
        self.assertEqual(frequency_penalty(0), 0.0)

    def test_above_threshold_negative(self):
        pen = frequency_penalty(100)
        self.assertLess(pen, 0)

    def test_extreme_access_capped(self):
        pen = frequency_penalty(10000)
        self.assertGreaterEqual(pen, -0.10)

    def test_moderate_access_small_penalty(self):
        pen = frequency_penalty(50)
        self.assertLess(pen, 0)
        self.assertGreater(pen, -0.05)


class TestConfidenceBoost(unittest.TestCase):
    """Consolidation strength modifier."""

    def test_neutral_returns_zero(self):
        self.assertAlmostEqual(confidence_boost(0.7), 0.0, places=3)

    def test_high_confidence_positive(self):
        self.assertGreater(confidence_boost(1.0), 0)

    def test_low_confidence_negative(self):
        self.assertLess(confidence_boost(0.1), 0)

    def test_none_returns_zero(self):
        self.assertEqual(confidence_boost(None), 0.0)

    def test_clamped_high(self):
        self.assertLessEqual(confidence_boost(1.0), 0.045)

    def test_clamped_low(self):
        self.assertGreaterEqual(confidence_boost(0.0), -0.09)


if __name__ == '__main__':
    unittest.main()
