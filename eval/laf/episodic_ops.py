#!/usr/bin/env python3
"""Episodic activation operators — the proper THREE-WAY version (encoded / picked / dropped).

The LAF episodic layer seeds from SIMILAR PAST SURFACE-MOMENTS, not from the cue↔node
cosine the other operators use. `recall_episodes(query=cue, older_than=cutoff, scale='s0')`
returns past conversation traces ranked by cosine `_score` = how similar that past moment is
to the current cue. Around each similar past moment the brain's nodes played one of three
roles (joined structurally by stop via `trace_links.episode_node_roles`), each weighted by
that moment's `_score`:

  • ENCODED — nodes CREATED/REVISED in/after the moment  → +activation (what the brain learned)
  • PICKED  — nodes recall SURFACED and Haiku SELECTED    → +activation (worth surfacing here)
  • DROPPED — nodes recall surfaced as candidates but
              Haiku did NOT pick                          → −inhibition (NOT worth surfacing)

This file supersedes `gold24_matrix.py:episodic_field` (which MERGES surfaced+encoded and
ignores dropped). Each role is its own operator returning a per-node vector over a master
index, so LAF can z-score + gain-weight them independently (the picked/encoded gains add,
the dropped gain subtracts).

──────────────────────────────────────────────────────────────────────────────────────────
THE `dropped` SHARPENING — ÷prevalence, NOT raw count  (the load-bearing design choice)
──────────────────────────────────────────────────────────────────────────────────────────
Haiku selects only ~3–5 of ~25 candidates, so a node being "dropped" once is almost always
just the top-5 cap, NOT a judgment. The real inhibition signal is the DROP-RATE weighted by
context-similarity: a node that is repeatedly a candidate across SIMILAR moments and
consistently NOT picked.

    inhibition(node) = Σ_m  score(m)·1{node dropped in m}
                       ─────────────────────────────────────
                       Σ_m  score(m)·1{node a candidate in m}

The denominator (÷prevalence) is what turns "dropped once because of the cap" (rate ≈ 1 but
prevalence ≈ 1 — a single weak moment) into a calibrated rate: a node dropped in 8/8 similar
moments where it was offered earns full inhibition; a node dropped 1/1 earns the same RATE but
the consumer can additionally gate on prevalence (Σ candidate weight) to ignore thin evidence.
We return BOTH the rate (the vector) and expose prevalence via `episodic_dropped_detail` so
the caller can apply a min-prevalence floor without recomputing.

──────────────────────────────────────────────────────────────────────────────────────────
THE OPEN DESIGN QUESTION — "what is a similar MOMENT?"  (pluggable seam, Tom to define)
──────────────────────────────────────────────────────────────────────────────────────────
Two decisions are deliberately NOT hardcoded — they are parameters with a clear seam:

  1. MOMENT WINDOW (`window`) — what counts as ONE moment around a matched s0 trace.
       • 'turn'  (DEFAULT) — one s0 trace = one moment. The roles are read at exactly that
         stop. Simplest, what the current `episodic_field` does implicitly.
       • ('window', N) — a ±N-turn window around the matched stop is ONE moment; roles are
         unioned across stops [stop−N, stop+N] and carry the matched trace's score. This is
         the seam for "a moment is a small conversational neighbourhood, not a single turn."
       The window is applied in `_episode_targets` by expanding each matched trace into the
       target stops fed to `episode_node_roles`. Tom will define the real algorithm; this
       file makes both swappable and reports the 'turn' default.

  2. SIMILARITY METRIC (`score_fn`) — how a moment's weight is computed.
       • DEFAULT — `recall_episodes`'s own cosine `_score` (cue ↔ s0-trace-embedding).
       • SWAPPABLE — pass `score_fn(episode, cue, brain) -> float` to override (e.g.
         max-sim over the moment's window, a learned transition weight, prev-anchor cosine).
       The seam is the `score_fn` parameter; default reads `episode['_score']`.

Both decisions also bear on "what's INCLUDED in a moment" — with `window='turn'` only the
matched turn's roles count; with a ±N window the surrounding turns' picks/drops/encodes are
folded in. That is exactly the knob Tom is thinking about.

Run via the audit harness: ./dev python3 eval/laf/gold24_episodic_audit.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from servers.scales.s1.trace_links import (  # noqa: E402
    gather_roles, episode_node_roles,
)

# Default count of similar past moments to seed from (top-K by similarity).
DEFAULT_TOP_MOMENTS = 15


# ─────────────────────────── the moment-definition seam ───────────────────────────
def default_score(episode, cue=None, brain=None):
    """Default moment weight = recall_episodes' own cosine `_score` (cue↔s0 trace).

    The SIMILARITY-METRIC seam: pass a different `score_fn(episode, cue, brain)` to
    `episodic_roles` to swap in max-sim / learned-transition / prev-anchor weighting.
    Falls back to 0.0 for an unscored episode (e.g. a time-ranked, query-less pull).
    """
    return float(episode.get("_score") or 0.0)


def _expand_window(stop, window):
    """The MOMENT-WINDOW seam: the set of stops that make up ONE moment.

    window='turn'        → {stop}                       (one trace = one moment)
    window=('window', N) → {stop-N, ..., stop, ..., stop+N}   (a ±N-turn neighbourhood)
    """
    if window == "turn" or window is None:
        return {stop}
    if isinstance(window, (tuple, list)) and len(window) == 2 and window[0] == "window":
        n = int(window[1])
        return set(range(stop - n, stop + n + 1))
    raise ValueError("unknown moment window spec: %r (use 'turn' or ('window', N))" % (window,))


def _stop_of_chain(chain_id):
    """Local stop parser mirroring trace_links._stop_of (chain tail int, else None)."""
    if not chain_id:
        return None
    tail = str(chain_id).rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _episode_targets(episodes, window):
    """Expand matched s0 episodes into the target 'turn' records episode_node_roles joins.

    With window='turn' each episode is its own target (1:1). With a ±N window each episode
    fans out into 2N+1 synthetic targets sharing the episode's score — so the roles at the
    surrounding stops are folded into that ONE moment. Returns:
        (targets, score_of_target_id)  where targets are {id, chain_id} records and
        score_of_target_id maps each synthetic target id → its parent episode score.
    The chain_id we synthesize reuses the episode's session_short so the stop join is exact.
    """
    targets = []
    score_of = {}
    for e in episodes:
        chain = e.get("chain_id") or ""
        stop = _stop_of_chain(chain)
        if stop is None:
            continue
        # session_short is the middle segment of an s0 chain: s0-{short}-{stop}
        parts = chain.split("-")
        short = parts[1] if len(parts) >= 3 else ""
        score = float(e.get("_score") or 0.0)
        for s in _expand_window(stop, window):
            tid = "%s@%d" % (e.get("id") or chain, s)
            targets.append({"id": tid, "chain_id": "s0-%s-%d" % (short, s)})
            # a stop reachable from multiple episodes' windows keeps the MAX score
            score_of[tid] = max(score_of.get(tid, 0.0), score)
    return targets, score_of


# ─────────────────────────── the shared seed → roles pull ───────────────────────────
def episodic_roles(brain, cue, cutoff, *, top=DEFAULT_TOP_MOMENTS,
                   window="turn", score_fn=default_score):
    """Seed from similar past moments and return their per-target three-way roles.

    The shared substrate the three operators sit on:
      1. recall_episodes(query=cue, older_than=cutoff, scale='s0', limit=top) — the
         similar PAST surface-moments, each carrying a similarity `_score`.
      2. group episodes by session, pull each session's (recall/surface/encode) trace
         streams via gather_roles, and join by stop via episode_node_roles.
      3. apply the moment-window expansion (the seam) so a target = a moment.

    Returns a list of per-moment role records:
        [{'score': float, 'encoded': [full_ids], 'picked': [short_ids],
          'dropped': [short_ids]}, ...]
    Empty list when recall_episodes returns nothing (caller should retry/reword,
    NOT treat empty as truth — see the audit harness).
    """
    ep = brain.recall_episodes(query=cue, older_than=cutoff, scale="s0", limit=top)
    episodes = ep.get("episodes", []) if isinstance(ep, dict) else []
    if not episodes:
        return []

    # window-expanded targets + their parent-moment scores
    targets, score_of = _episode_targets(episodes, window)
    # the score_fn seam: recompute each EPISODE's weight, then propagate to its targets
    # (default_score just reads _score, so the default path is unchanged).
    if score_fn is not default_score:
        ep_score = {(e.get("id") or e.get("chain_id")): float(score_fn(e, cue, brain) or 0.0)
                    for e in episodes}
        # re-derive target scores under the custom metric (max over contributing episodes)
        score_of = {}
        for e in episodes:
            chain = e.get("chain_id") or ""
            stop = _stop_of_chain(chain)
            if stop is None:
                continue
            parts = chain.split("-")
            short = parts[1] if len(parts) >= 3 else ""
            s = ep_score.get(e.get("id") or chain, 0.0)
            for st in _expand_window(stop, window):
                tid = "%s@%d" % (e.get("id") or chain, st)
                score_of[tid] = max(score_of.get(tid, 0.0), s)

    # group targets by session so we pull each session's trace streams once
    by_sess = defaultdict(list)
    sess_of_episode = {}
    for e in episodes:
        sess_of_episode[e.get("id") or e.get("chain_id")] = e.get("session_id")
    # map each target back to its episode's session via the id prefix
    targets_by_sess = defaultdict(list)
    for t in targets:
        ep_id = t["id"].rsplit("@", 1)[0]
        sess = sess_of_episode.get(ep_id)
        if sess:
            targets_by_sess[sess].append(t)

    records = []
    for sess, sess_targets in targets_by_sess.items():
        recall_tr, surf_tr, enc_tr = gather_roles(brain, sess)
        roles = episode_node_roles(recall_tr, surf_tr, enc_tr, sess_targets)
        for tid, role in roles.items():
            records.append({
                "score": score_of.get(tid, 0.0),
                "encoded": role.get("encoded", []),
                "picked": role.get("picked", []),
                "dropped": role.get("dropped", []),
            })
    return records


# ─────────────────────────── id-width resolution ───────────────────────────
def _short_to_full(idx):
    """Map 8-char short id → full id, from the master index keys.

    picked/dropped roles are recorded as 8-char short ids (what surface traces store);
    encoded roles are full ids. The master `idx` is full-id keyed, so short ids must be
    resolved. A short id that collides (two full ids share a prefix) is dropped from the
    map (ambiguous → skip, never mis-attribute) and logged via the returned collision count.
    """
    by_short = defaultdict(list)
    for full in idx:
        by_short[full[:8]].append(full)
    out = {}
    collisions = 0
    for short, fulls in by_short.items():
        if len(fulls) == 1:
            out[short] = fulls[0]
        else:
            collisions += 1
    return out, collisions


def _resolve(node_id, idx, s2f):
    """Resolve a role node id (short OR full) to a master row index, or None."""
    if node_id in idx:                     # already a full id in the master
        return idx[node_id]
    full = s2f.get(node_id)                # short → full
    return idx.get(full) if full else None


# ─────────────────────────── the three operators ───────────────────────────
def episodic_encoded(brain, cue, cutoff, idx, n, *,
                     top=DEFAULT_TOP_MOMENTS, window="turn", score_fn=default_score,
                     _records=None):
    """+activation: per node, the best similar-moment score where it was ENCODED.

    A node created/revised in/after a moment similar to the cue is what the brain LEARNED
    in a situation like this → activate it. Per-node value = max over moments of that
    moment's similarity score. Returns an [n] non-negative vector over the master index.
    `_records` lets the audit reuse one episodic_roles pull across all three operators.
    """
    vec = np.zeros(n, dtype=np.float64)
    records = _records if _records is not None else episodic_roles(
        brain, cue, cutoff, top=top, window=window, score_fn=score_fn)
    s2f, _ = _short_to_full(idx)
    for r in records:
        s = r["score"]
        for node in set(r["encoded"]):
            i = _resolve(node, idx, s2f)
            if i is not None and s > vec[i]:
                vec[i] = s
    return vec


def episodic_picked(brain, cue, cutoff, idx, n, *,
                    top=DEFAULT_TOP_MOMENTS, window="turn", score_fn=default_score,
                    _records=None):
    """+activation: per node, the best similar-moment score where Haiku PICKED it.

    A node recall surfaced AND Haiku selected in a similar moment was JUDGED worth
    surfacing in a situation like this → activate it. Per-node value = max over moments
    of similarity score. Returns an [n] non-negative vector over the master index.
    """
    vec = np.zeros(n, dtype=np.float64)
    records = _records if _records is not None else episodic_roles(
        brain, cue, cutoff, top=top, window=window, score_fn=score_fn)
    s2f, _ = _short_to_full(idx)
    for r in records:
        s = r["score"]
        for node in set(r["picked"]):
            i = _resolve(node, idx, s2f)
            if i is not None and s > vec[i]:
                vec[i] = s
    return vec


def episodic_dropped_detail(brain, cue, cutoff, idx, n, *,
                            top=DEFAULT_TOP_MOMENTS, window="turn",
                            score_fn=default_score, _records=None):
    """The ÷prevalence inhibition, with prevalence exposed. Returns (rate, prevalence).

    rate[i]       = Σ_m score(m)·1{i dropped in m} / Σ_m score(m)·1{i candidate in m}
    prevalence[i] = Σ_m score(m)·1{i candidate in m}    (the denominator — evidence mass)

    rate is the context-weighted drop-RATE: a node consistently NOT picked across SIMILAR
    moments where it WAS a candidate. prevalence lets the caller floor on thin evidence
    (a 1/1 drop has rate=1 but tiny prevalence). Both are [n] non-negative vectors; rate is
    the magnitude of inhibition (the LAF consumer SUBTRACTS gain·zscore(rate)).
    """
    drop_w = np.zeros(n, dtype=np.float64)   # Σ score over moments node was DROPPED
    cand_w = np.zeros(n, dtype=np.float64)   # Σ score over moments node was a CANDIDATE
    records = _records if _records is not None else episodic_roles(
        brain, cue, cutoff, top=top, window=window, score_fn=score_fn)
    s2f, _ = _short_to_full(idx)
    for r in records:
        s = r["score"]
        dropped = set(r["dropped"])
        picked = set(r["picked"])
        # candidate pool at this moment = picked ∪ dropped (everything offered to Haiku)
        for node in dropped | picked:
            i = _resolve(node, idx, s2f)
            if i is None:
                continue
            cand_w[i] += s
            if node in dropped:
                drop_w[i] += s
    rate = np.zeros(n, dtype=np.float64)
    nz = cand_w > 1e-12
    rate[nz] = drop_w[nz] / cand_w[nz]
    return rate, cand_w


def episodic_dropped(brain, cue, cutoff, idx, n, *,
                     top=DEFAULT_TOP_MOMENTS, window="turn", score_fn=default_score,
                     min_prevalence=0.0, _records=None):
    """−inhibition: the ÷prevalence context-weighted drop-RATE (see episodic_dropped_detail).

    Per node, Σ score·1{dropped} / Σ score·1{candidate} over similar moments — high when a
    node is repeatedly offered yet consistently NOT picked. Returned as a NON-NEGATIVE
    magnitude; the LAF layer applies it with a NEGATIVE gain (subtracts it). `min_prevalence`
    zeros out nodes whose candidate-evidence mass is below the floor (ignore thin 1/1 drops).
    Returns an [n] vector over the master index.
    """
    rate, prevalence = episodic_dropped_detail(
        brain, cue, cutoff, idx, n, top=top, window=window,
        score_fn=score_fn, _records=_records)
    if min_prevalence > 0.0:
        rate = np.where(prevalence >= min_prevalence, rate, 0.0)
    return rate
