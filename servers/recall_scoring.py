"""Unified recall scoring — one formula for all retrieval paths.

This module computes the final relevance score for a recalled node.
It's a PURE module: no database access, no Brain dependency.
All constants come from brain_constants.py.

Architecture:
  semantic_score (from z-weighted embedding groups) is the BASE.
  Other signals MODULATE within bounded ranges.
  Zero semantic = zero final. No signal can override irrelevance.

Biology mapping:
  semantic_score → pattern completion (hippocampal CA3)
  freshness      → hippocampal fresh trace (created_at, NOT last_accessed)
  emotion        → amygdala modulation (GANE model: winner-take-more)
  frequency      → cue overload penalty (high-access = low diagnostic value)
  confidence     → consolidation strength (validated = stronger trace)

Formula:
  final = base * (1.0 + recency_boost + emotion_boost + frequency_penalty + confidence_boost)
  Max modulator: ~1.55x. Min modulator: ~0.81x. Bounded.

Called from:
  brain_recall.py recall() STEP 6 — embedding path
  brain_recall.py _keyword_recall() — keyword fallback path
  Both paths use the SAME formula. No more two-path scoring divergence.

Tested: 2026-04-02 via eval/brain_eval.py and eval/decode_funnel.py.
"""

import math
from datetime import datetime
from typing import Optional

from .brain_constants import (
    FRESHNESS_BANDS,
    EMOTION_AMPLIFICATION,
    FREQUENCY_PENALTY_THRESHOLD,
    FREQUENCY_PENALTY_SCALE,
    FREQUENCY_PENALTY_MAX,
    CONFIDENCE_NEUTRAL,
    CONFIDENCE_SCALE,
)


def unified_score(
    semantic_score: float,
    created_at: Optional[str] = None,
    emotion: float = 0,
    access_count: int = 0,
    confidence: Optional[float] = None,
) -> float:
    """Compute final recall score for a node.

    Args:
        semantic_score: Base relevance from embeddings (z-weighted top2-avg)
                        + any pre-scoring adjustments (title match, critical, etc.)
                        This is the gatekeeper — zero base = zero final.
        created_at:     ISO timestamp of node creation. Drives freshness boost.
                        Uses creation time, NOT last_accessed (avoids self-fulfilling loop).
        emotion:        Node emotion value (-1 to +1). Abs value amplifies relevance.
        access_count:   How many times this node has been recalled. High = hub penalty.
        confidence:     Node confidence (0-1). Higher than neutral = mild boost.

    Returns:
        Final score. Always >= 0. Zero if semantic_score <= 0.
    """
    if semantic_score <= 0:
        return 0.0

    recency = freshness_from_created(created_at)
    emotion_boost = abs(emotion or 0) * EMOTION_AMPLIFICATION
    freq_pen = frequency_penalty(access_count)
    conf_boost = confidence_boost(confidence)

    modulator = 1.0 + recency + emotion_boost + freq_pen + conf_boost
    return semantic_score * modulator


def freshness_from_created(created_at: Optional[str]) -> float:
    """Compute recency boost from node creation time.

    Uses FRESHNESS_BANDS from constants. Recent knowledge gets a boost.
    Old knowledge gets no penalty (boost = 0), just no advantage.

    Returns 0.0 on None or parse failure (safe default — no boost, no penalty).
    """
    if not created_at:
        return 0.0

    try:
        # Parse ISO timestamp — handle both 'Z' suffix and '+00:00' offset
        ts = created_at.replace('Z', '+00:00')
        created_dt = datetime.fromisoformat(ts)
        now = datetime.utcnow()
        hours_ago = (now - created_dt.replace(tzinfo=None)).total_seconds() / 3600

        for band in FRESHNESS_BANDS:
            if hours_ago <= band['max_hours']:
                return band['boost']

        return 0.0
    except Exception:
        return 0.0


def frequency_penalty(access_count: int) -> float:
    """Compute hub penalty from access count.

    High-access nodes are less diagnostic (cue overload principle).
    Below threshold: no penalty. Above: log-scaled, capped.

    Returns: 0 to -FREQUENCY_PENALTY_MAX (always negative or zero).
    """
    if not access_count or access_count <= FREQUENCY_PENALTY_THRESHOLD:
        return 0.0

    penalty = math.log(access_count / FREQUENCY_PENALTY_THRESHOLD) * FREQUENCY_PENALTY_SCALE
    return -min(penalty, FREQUENCY_PENALTY_MAX)


def confidence_boost(confidence: Optional[float]) -> float:
    """Compute consolidation strength modifier from confidence.

    Maps confidence to a small boost/penalty around CONFIDENCE_NEUTRAL.
    High confidence (validated) gets mild boost. Low confidence gets mild penalty.

    Returns: approximately -0.09 to +0.045. 0.0 on None.
    """
    if confidence is None:
        return 0.0

    raw = (confidence - CONFIDENCE_NEUTRAL) * CONFIDENCE_SCALE
    return max(-0.09, min(0.045, raw))
