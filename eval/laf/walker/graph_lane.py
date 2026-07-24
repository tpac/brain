"""Graph lane (design B, spec 8a93d799) — the primitive + its GATE.

A STANDALONE LAF lane that lights up 1-hop neighbors of the maxsim-top5
seeds, filtered to the base union (co_accessed ∪ semantic-desc≥80) and
SCORED continuously (seed_z × why_cos × convergence × type_prior). Replaces
the inert graph OPERATOR (54777ca7) — a single static z-lane in the
product-of-experts sum, NOT an iterative spread through the settling engine.

Two substrates, one scorer:
  reach  — activation over ALL master nodes (rank gold among 7684 → reach@5)
  pool   — the same activation restricted to a turn's candidate rows

THE SCORER is modular (SCORE_SPEC): every factor can be ablated to a
constant so the audit can measure which factor earns its place. Absolute
scale is irrelevant — the lane is support-z normalized downstream; only the
ordering among the ~16 filtered neighbors matters.

SELF-CHECK (the graph analog of GATE 0): reproduce the committed
corpus_v2_hop_refine anatomy — 345 clean-valid misses, 71 rescue rows,
base-union 52/71 rescues kept at 16.1 noise/turn. If the seed→neighbor→
filter doesn't reproduce those, NOTHING downstream is trustworthy.

Read-only. Run:  ./dev python3 eval/laf/walker/graph_lane.py
"""
import json
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

from walker_db import OUT_DIR, WALKER_DB, open_ro, open_brain_ro

sys.path.insert(0, str(OUT_DIR))
from lambda_probe import zn                                           # noqa: E402
from layer_readout_probe import lane_z                              # noqa: E402
from servers.recall_laf import zscore_variant                       # noqa: E402

CUTOFF = '2026-05-11'
K_SEEDS = 5
DESC_MIN = 80                       # base-union semantic-edge desc threshold
GENERIC = ('related_to', 'related', 'community_member')

# ── the modular scorer (spec 8a93d799 §SCORE) ───────────────────────────
# Each factor maps to a multiplier; ablate by setting the toggle False (the
# factor becomes a constant 1.0). Measured priors from corpus_v2_hop_refine:
#   type_prior  lesson 21% of rescues vs 6% noise → 3.5×; architecture 2×;
#               decision/community anti (noise-heavy) → 0.5/0.7×
#   convergence ≥2 seeds triples rescue rate → 2×; ≥3 → 3×
#   why_cos     rescue median 0.661 vs noise 0.605 (a ranker, not a knife)
TYPE_PRIOR = {'lesson': 3.5, 'architecture': 2.0, 'decision': 0.5,
              'community': 0.7}
CONV_MULT = {1: 1.0, 2: 2.0}        # ≥3 clamps to the last entry (3.0 below)
CONV_MULT_3PLUS = 3.0

DEFAULT_SCORE_SPEC = {
    'use_seed_z': True,             # × maxsim z of the activating seed(s)
    'seed_agg': 'max',              # 'max' | 'sum' over reaching seeds
    'use_why_cos': True,            # × best edge-why cosine (sem channel)
    'co_acc_cos': 0.60,             # neutral why for co_acc-only (no edge text)
    'use_convergence': True,        # × CONV_MULT[n_seeds]
    'use_type_prior': True,         # × TYPE_PRIOR[tgt_type]
}


def iso(ts):
    try:
        return datetime.fromisoformat((ts or '').replace('Z', '+00:00'))
    except Exception:
        return None


def build_adjacency(b, m2i):
    """{row: [(nbr_row, rel, desc_len, edge_created_at, why_emb, coacc_n,
    last_strengthened)]} — bidirectional, active edges only. row-space is the
    field-cache master index (m2i: node_id → row)."""
    adj = defaultdict(list)
    for src, tgt, rel, dlen, ecr, emb, cac, lstr in b.execute(
            "SELECT e.source_id, e.target_id, r.relation, "
            "LENGTH(COALESCE(r.description,'')), e.created_at, r.embedding, "
            "e.co_access_count, e.last_strengthened "
            "FROM edges e JOIN edge_relations r ON r.edge_id=e.edge_id "
            "WHERE (r.archived IS NULL OR r.archived=0)"):
        si, ti = m2i.get(src), m2i.get(tgt)
        if si is None or ti is None:
            continue
        ev = np.frombuffer(emb, dtype=np.float32) if emb is not None else None
        rec = (rel, dlen or 0, ecr, ev, cac or 0, lstr)
        adj[si].append((ti,) + rec)
        adj[ti].append((si,) + rec)
    return adj


def build_node_meta(b, m2i):
    """{row: (type, content_len)}."""
    meta = {}
    for nid, typ, clen in b.execute(
            'SELECT id, type, LENGTH(content) FROM nodes'):
        if nid in m2i:
            meta[m2i[nid]] = (typ, clen or 0)
    return meta


def build_qvecs(w):
    """{(sess, epoch, seq): q_vec} for labeled turns."""
    qv = {}
    for sess, epoch, seq, blob in w.execute(
            'SELECT session_id, epoch, seq, q_vec FROM turns WHERE labeled=1'):
        if blob is not None:
            qv[(sess, epoch, seq)] = np.frombuffer(blob, dtype=np.float32)
    return qv


def seed_rows(lanes_mm, row, S, n):
    """maxsim-top5 seed rows + a {row: maxsim_z} lookup. Seeds are a pure
    function of (cue, graph) — the drift-proof choice (spec §SEEDS)."""
    raw_mx = lanes_mm[row].astype(np.float32)[S['op0'], 0]       # maxsim=lane0
    mxz = lane_z(raw_mx, 'maxsim', np.isfinite(raw_mx), n)
    fin = np.where(np.isfinite(mxz), mxz, -np.inf)
    seeds = [int(x) for x in np.argsort(-fin)[:K_SEEDS]]
    return seeds, {si: float(mxz[si]) for si in seeds}


def aggregate_neighbors(seeds, seed_z, adj, qv, turn_dt):
    """Per unique 1-hop neighbor, aggregate over the edges that reached it
    from the seeds. Time-honest: edges created after the turn are invisible.
    Returns {nbr_row: agg}."""
    neigh = {}
    for si in seeds:
        for (oi, rel, dlen, ecr, ev, cac, lstr) in adj.get(si, ()):
            edt = iso(ecr)
            if turn_dt and edt and edt > turn_dt:
                continue
            cos = float(qv @ ev) if (qv is not None and ev is not None) else None
            d = neigh.setdefault(oi, {
                'seeds': set(), 'seed_z': [], 'best_cos': None,
                'best_dlen': 0, 'has_coacc': False, 'has_sem': False})
            d['seeds'].add(si)
            d['seed_z'].append(seed_z[si])
            if rel == 'co_accessed':
                d['has_coacc'] = True
            elif rel not in GENERIC:
                d['has_sem'] = True
                if dlen > d['best_dlen']:
                    d['best_dlen'] = dlen
                if cos is not None and (d['best_cos'] is None
                                        or cos > d['best_cos']):
                    d['best_cos'] = cos
    return neigh


def passes_filter(d):
    """Base union (spec §FILTER, binary — stop here): co_accessed ∪
    semantic-edge-with-desc≥80. Every further hard cut measured to lose
    rescues faster than noise — do NOT add gates."""
    return d['has_coacc'] or (d['has_sem'] and d['best_dlen'] >= DESC_MIN)


def score_neighbor(d, tgt_type, spec):
    """Continuous activation = seed_z × why_cos × convergence × type_prior,
    each factor gated by the spec (ablation → constant 1.0)."""
    act = 1.0
    if spec['use_seed_z']:
        sz = d['seed_z']
        act *= (max(sz) if spec['seed_agg'] == 'max' else sum(sz))
    if spec['use_why_cos']:
        act *= d['best_cos'] if d['best_cos'] is not None else spec['co_acc_cos']
    if spec['use_convergence']:
        ns = len(d['seeds'])
        act *= CONV_MULT_3PLUS if ns >= 3 else CONV_MULT.get(ns, 1.0)
    if spec['use_type_prior']:
        act *= TYPE_PRIOR.get(tgt_type, 1.0)
    return act


def graph_activation(seeds, seed_z, adj, qv, turn_dt, node_meta, n,
                     spec=DEFAULT_SCORE_SPEC):
    """Raw graph activation over ALL n master rows (0 = not a kept neighbor).
    Also returns the kept-neighbor rows for support bookkeeping."""
    neigh = aggregate_neighbors(seeds, seed_z, adj, qv, turn_dt)
    act = np.zeros(n)
    kept = []
    for oi, d in neigh.items():
        if not passes_filter(d):
            continue
        tgt_type = node_meta.get(oi, (None, 0))[0]
        act[oi] = score_neighbor(d, tgt_type, spec)
        kept.append(oi)
    return act, kept


# ── SELF-CHECK: reproduce the committed hop_refine anatomy ───────────────
def _self_check():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    fields = np.load(OUT_DIR / 'field_cache.npy', mmap_mode='r')
    lanes_mm = np.load(OUT_DIR / 'lane_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    master = idx['master']
    m2i = {sid: i for i, sid in enumerate(master)}
    n = idx['n_nodes']
    verds = {json.loads(x)['key']: json.loads(x)
             for x in (OUT_DIR / 'corpus_v2_verdicts.jsonl').open()}
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}

    w = open_ro(WALKER_DB)
    qvecs = build_qvecs(w)
    w.close()
    b = open_brain_ro()
    node_meta = build_node_meta(b, m2i)
    adj = build_adjacency(b, m2i)
    b.close()

    n_miss = 0
    rescue_rows = 0
    base_rescue_kept = 0
    base_noise_kept = 0
    for t in idx['turns']:
        key = '%s/%d/%d' % tuple(t['key'])
        v = verds.get(key)
        bd = bundles.get(key)
        if not v or v['verdict'] != 'valid' or not bd or (bd['ts'] or '') < CUTOFF:
            continue
        # gold row + mix rank (0.65 f0 + 0.35 mh), tie-fair — the miss gate
        gi = t.get('gold_i')
        if gi is None:
            continue
        gr = t['cand_rows'][gi]
        if gr < 0:
            continue
        F = fields[t['row']].astype(np.float32)
        f0 = F[S['op0']]
        a1, f1 = F[S['anchor1']], F[S['op1']]
        f2 = F[S['op2']]
        a2 = F[S['anchor2']] if 'anchor2' in S else None
        parts = [(0.5, a1), (0.25, f1), (0.0625, f2)]
        if a2 is not None:
            parts.insert(2, (0.125, a2))
        mh = np.zeros(n)
        pres = np.zeros(n, dtype=bool)
        for wt, fld in parts:
            if fld is None or np.isnan(fld).all():
                continue
            fin = np.isfinite(fld)
            mh += wt * np.where(fin, fld, 0.0)
            pres |= fin
        mh[~pres] = np.nan
        if not np.isfinite(f0[gr]):
            continue
        mix = 0.65 * zn(f0) + 0.35 * zn(mh)
        fin = np.where(np.isfinite(mix), mix, -np.inf)
        if int((fin > mix[gr]).sum()) + 1 <= 5:
            continue                                    # a hit — not a miss
        n_miss += 1
        turn_dt = iso(bd['ts'])
        qv = qvecs.get(tuple(t['key']))
        seeds, sz = seed_rows(lanes_mm, t['row'], S, n)
        neigh = aggregate_neighbors(seeds, sz, adj, qv, turn_dt)
        for oi, d in neigh.items():
            is_rescue = (oi == gr)
            if is_rescue:
                rescue_rows += 1
            if passes_filter(d):
                if is_rescue:
                    base_rescue_kept += 1
                else:
                    base_noise_kept += 1

    print('# graph_lane self-check vs committed hop_refine anatomy\n')
    exp = {'n_miss': 345, 'rescue_rows': 71, 'base_rescue_kept': 52,
           'noise_per_turn': 16.1}
    got = {'n_miss': n_miss, 'rescue_rows': rescue_rows,
           'base_rescue_kept': base_rescue_kept,
           'noise_per_turn': round(base_noise_kept / max(1, n_miss), 1)}
    ok = True
    for k in exp:
        match = abs(got[k] - exp[k]) <= (0.2 if k == 'noise_per_turn' else 0)
        ok = ok and match
        print('  %-18s expected %-7s got %-7s %s'
              % (k, exp[k], got[k], 'OK' if match else 'MISMATCH'))
    print('\n%s' % ('SELF-CHECK PASS — primitive reproduces the anatomy'
                    if ok else 'SELF-CHECK FAIL — do NOT trust downstream'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(_self_check())
