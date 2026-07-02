#!/usr/bin/env python3
"""THE shared LAF eval metrics — one definition, every probe imports from here.

The code review (2026-07-01) found need@k computed THREE different ways across the
probe suite (first-match need reconstruction / empty-needs-counted-as-0 / empty-needs-
excluded), making cross-probe tables non-commensurable: gold24_diagnostic read ~4%
lower than the beam probes for identical rankings purely from denominator choice.
This module is the fix — the canonical definitions, with the semantics settled:

  • need-collapse: iterate cue['needs'] DIRECTLY (a need is met if ANY of its gold
    nodes ranks ≤ k; a node listed under two needs counts for both — no first-match
    reconstruction).
  • empty-needs cues are UNMEASURABLE, not zero: they are excluded from the need@k
    denominator (a cue with no gold can't be hit; counting it as 0 punishes rankers
    for corpus composition). Callers report n alongside.
"""
import numpy as np


def zscore(x, elig, n):
    """Standardize x over the eligible finite entries → unit variance; 0 elsewhere.
    The LAF fusion normalizer: gains stay pure influence dials only if every
    operator field passes through this exact form."""
    m = elig & np.isfinite(x)
    o = np.zeros(n)
    if int(m.sum()) > 2 and np.std(x[m]) > 1e-9:
        o[m] = (x[m] - x[m].mean()) / x[m].std()
    return o


def ranks(scores, elig, master):
    """{node_id: 1-based rank} over eligible finite-scored nodes, score desc."""
    s = np.where(elig & np.isfinite(scores), scores, -np.inf)
    return {master[i]: r + 1 for r, i in enumerate(np.argsort(-s))}


def best_ranks(rank_map, needs):
    """{need: best (min) rank over its gold nodes}; None if none ranked."""
    out = {}
    for need, nids in needs.items():
        rs = [rank_map.get(nm) for nm in nids if rank_map.get(nm) is not None]
        out[need] = min(rs) if rs else None
    return out


def need_hit_at(rank_map, needs, k):
    """Need-collapsed hit@k for one cue: fraction of needs with ANY gold node ≤ k.
    Returns None for an empty-needs cue (unmeasurable — exclude from averages)."""
    if not needs:
        return None
    br = best_ranks(rank_map, needs)
    return sum(1 for r in br.values() if r is not None and r <= k) / len(br)


def brought_lost(rank_map, needs, ref_best_ranks, k=25):
    """(brought, lost) @k vs a reference config's best_ranks for the same cue.
    brought = needs this config reaches@k that the reference missed; lost = the
    reverse. The decomposition that tells reach from reshuffling."""
    br = best_ranks(rank_map, needs)
    brought = lost = 0
    for need, r in br.items():
        r0 = ref_best_ranks.get(need)
        now = r is not None and r <= k
        was = r0 is not None and r0 <= k
        if now and not was:
            brought += 1
        elif was and not now:
            lost += 1
    return brought, lost


# ── score-vector convenience forms (what the layer probes call in their config loops) ──

def need_hit(scores, elig, master, needs, k):
    """need_hit_at from a raw score vector (0.0 for an empty-needs cue — probes
    exclude those at precompute, so None never reaches an average here)."""
    return need_hit_at(ranks(scores, elig, master), needs, k) or 0.0


def need_bl(scores, elig, master, needs, ref_best_ranks, k=25):
    """brought_lost from a raw score vector."""
    return brought_lost(ranks(scores, elig, master), needs, ref_best_ranks, k)
