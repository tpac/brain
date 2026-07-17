"""Settling probes — renorm confound control, graph-spread settling,
MMR lateral inhibition at delivery (Tom go, 2026-07-17).

A. RENORM CONFOUND CONTROL (pool, val turns, K8-γ0.7 op-cue):
   renorm's soft_r win could be implicit re-weighting (each z dilutes
   what's already accumulated) rather than real nonlinearity. Controls:
     - fitted-linear: per-message weights FITTED on train soft-pairs
       (the best any linear mesh can do) — renorm must beat it;
     - order-shuffle: renorm with per-turn random add order — if the
       gain survives shuffling, it's rescaling, not dynamics;
     - coverage: single-message turns must rank identically to linear.

B. GRAPH-SPREAD SETTLING (full-field, sampled val turns, K2-γ0.7):
   final = z(A) + β·z(spread(relu(z(A)))), spread = degree-normalized
   undirected flow over noise-excluded edges (fan damping built in),
   edges cutoff-masked at the turn ts. hops ∈ {1,2}, β ∈ {0.3,0.5}.
   The only settling variant that can ADD reach — judged on pool-entry
   (@25) and soft_r, not Haiku top-1 agreement.

C. MMR LATERAL INHIBITION AT DELIVERY (same full-field rig):
   deliver 5 from the top-50 greedily, score' = z(s) − λ·max_cos(chosen)
   over _primary vectors, λ ∈ {0.3,0.7}. Directional probe: redundancy
   of the delivered set vs coverage; the full test needs need-clusters.

Run:  ./dev python3 eval/laf/walker/settling_probes.py
Out:  settling_probes.md, settling_probes.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import WALKER_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from q1_sweep import GAINS, load, gate_provenance, weights          # noqa: E402
from moment_grids import cfg_for, message_fields, mesh, eval_arm    # noqa: E402
from moment_influence import sample_turns, turn_fields              # noqa: E402
from p3_fit import fit_logistic                                     # noqa: E402
from reach_leg import rank_rows                                     # noqa: E402
from soft_usage import auc                                          # noqa: E402
from servers.recall_laf import _zscore                              # noqa: E402

sys.path.insert(0, str(REPO / 'eval' / 'laf'))
from operators import graph_spread                                  # noqa: E402

CFG_A = cfg_for(8, 0.7)          # probe-A base: the settling result's config
CFG_B = cfg_for(2, 0.7)          # probe-B base: fits the j<=2 field rig
N_SAMPLE = 120
SEED = 20260717
SOFT_MARGIN = 0.10
REPORT = WALKER_DIR / 'settling_probes.md'
OUT = WALKER_DIR / 'settling_probes.json'


# ------------------------------------------------------------------ A

def renorm_mesh(F, ww, order):
    nc = F.shape[0]
    A = np.zeros(nc)
    touched = np.zeros(nc, dtype=bool)
    for col in order:
        f = F[:, col]
        fin = np.isfinite(f)
        if not fin.any():
            continue
        A = A + np.where(fin, ww[col] * f, 0.0)
        A = _zscore(A, nc)
        touched |= fin
    return np.where(touched, A, np.nan)


def probe_a(turns):
    w = weights(CFG_A)
    cache = {}
    for td in turns:
        cache[td.key] = message_fields(td, CFG_A, w)

    # fitted-linear control: per-message weights on train soft-pairs
    rows = []
    n_msg = next(iter(cache.values()))[0].shape[1]
    for td in turns:
        if td.val:
            continue
        F, _ = cache[td.key]
        X = np.where(np.isfinite(F), F, 0.0)
        fin = np.flatnonzero(np.isfinite(td.soft))
        if len(fin) < 2:
            continue
        s = td.soft[fin]
        d = s[:, None] - s[None, :]
        wi, li = np.nonzero(d >= SOFT_MARGIN)
        if len(wi):
            rows.append(X[fin[wi]] - X[fin[li]])
    D = np.concatenate(rows)
    w_fit = fit_logistic(D)

    rng = np.random.default_rng(SEED)
    orders = {td.key: rng.permutation(n_msg) for td in turns}

    def run(score_of):
        return eval_arm(turns, CFG_A, w, score_fn=score_of)

    arms = {
        'linear': run(lambda td: mesh(*cache[td.key], 'linear')),
        'renorm': run(lambda td: renorm_mesh(*cache[td.key],
                                             range(n_msg))),
        'renorm_shuffled': run(lambda td: renorm_mesh(
            *cache[td.key], orders[td.key])),
        'fitted_linear': run(lambda td: np.where(
            np.any(np.isfinite(cache[td.key][0]), axis=1),
            np.where(np.isfinite(cache[td.key][0]),
                     cache[td.key][0], 0.0) @ w_fit, np.nan)),
    }

    # coverage: single-message turns must rank identically linear/renorm
    bad = 0
    for td in turns:
        F, ww = cache[td.key]
        live = [c for c in range(n_msg)
                if np.isfinite(F[:, c]).any()]
        if len(live) != 1:
            continue
        a = mesh(F, ww, 'linear')
        b = renorm_mesh(F, ww, range(n_msg))
        fa = np.isfinite(a)
        if not np.array_equal(np.argsort(-a[fa]), np.argsort(-b[fa])):
            bad += 1
    arms['coverage_single_msg_order_mismatches'] = bad
    arms['fitted_weights'] = [round(float(x), 4) for x in w_fit]
    return arms


# ------------------------------------------------------------------ B

def build_adjacency_ts(brain, idx):
    """operators.build_adjacency + per-edge created_at for cutoff masks."""
    na = brain.aspects.by_name('noise')
    noise = list(na.edge_relations) if na else []
    ph = ','.join('?' * len(noise)) if noise else "''"
    rows = brain.conn.execute(
        "SELECT e.source_id, e.target_id, SUM(COALESCE(er.weight,1.0)), "
        "MIN(COALESCE(e.created_at,'')) FROM edge_relations er "
        "JOIN edges e ON e.edge_id = er.edge_id "
        "WHERE (er.archived IS NULL OR er.archived=0) "
        "AND er.relation NOT IN (%s) "
        "GROUP BY e.source_id, e.target_id" % ph, noise).fetchall()
    src, dst, w, ets = [], [], [], []
    for s, t, wt, ts in rows:
        if s in idx and t in idx and s != t:
            src.append(idx[s])
            dst.append(idx[t])
            w.append(float(wt or 1.0))
            ets.append(ts or '')
    return (np.asarray(src), np.asarray(dst),
            np.asarray(w, dtype=np.float32), np.asarray(ets, dtype='<U40'))


def adj_asof(adj_full, ts, n):
    src, dst, w, ets = adj_full
    m = ets < ts                      # '' (unknown) sorts before any ISO → kept
    s, d, ww = src[m], dst[m], w[m]
    degree = np.zeros(n, dtype=np.float32)
    if s.size:
        np.add.at(degree, s, ww)
        np.add.at(degree, d, ww)
    return s, d, ww, degree


def probe_bc(walker, pool_of):
    from q1_sweep import compose, stack_messages
    from servers.recall_laf import LafV1Engine, _unit
    from tests.isolated_brain import IsolatedBrain
    rng = np.random.default_rng(SEED)
    sample = sample_turns(walker, rng)
    # soft label = soft_usage.soft_max — the live quality label. The first
    # run joined candidates.used_next_* (7 positives in 93k rows, the label
    # soft_usage.py itself declares dead) — 2026-07-17 audit finding 3.
    soft_of = {}
    for sess, epoch, seq, nid, sm in walker.execute(
            "SELECT session_id, epoch, seq, node_id, soft_max "
            "FROM soft_usage WHERE soft_max IS NOT NULL"):
        soft_of[(sess, epoch, seq, nid)] = float(sm)

    B_ARMS = ['base', 'spread h1 b0.3', 'spread h1 b0.5', 'spread h2 b0.3']
    ranks = {a: [] for a in B_ARMS}
    soft_xy = {a: ([], []) for a in B_ARMS}
    brought = lost = 0
    mmr = {lam: {'sel_in_5': 0, 'n_turn': 0, 'redundancy': [],
                 'base_redundancy': []} for lam in (0.3, 0.7)}
    w_b = weights(CFG_B)
    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_titles(env.brain)
            eng._refresh_traces(env.brain)
        eng._brain_ref = env.brain
        n = eng._n
        idx = {nid: i for i, nid in enumerate(eng._master[:n])}
        adj_full = build_adjacency_ts(env.brain, idx)
        prim = eng._mats['_primary'][:n]
        for i, ((sess, epoch, seq, stop, ts, op_text, qb), sel_ids) \
                in enumerate(sample):
            q0 = _unit(qb)
            if q0 is None:
                continue
            node_mask, trace_mask = eng._asof_masks(ts, n)
            op, an = turn_fields(eng, trace_mask, sess, stop, ts,
                                 op_text, q0)
            sel_rows = [r for r in (eng._resolve(nid) for nid in sel_ids)
                        if r is not None]
            if not sel_rows:
                continue
            mats, ww = {}, None
            for ln in GAINS:
                mats[ln], ww = stack_messages(op[ln], an[ln], w_b, CFG_B)
            base = compose(mats, ww, CFG_B, n, mask=node_mask)
            zb = _zscore(np.where(node_mask, base, np.nan), n)
            zb0 = np.where(np.isfinite(zb), zb, 0.0)
            adj = adj_asof(adj_full, ts, n)
            fields = {'base': zb}
            for name, hops, beta in (('spread h1 b0.3', 1, 0.3),
                                     ('spread h1 b0.5', 1, 0.5),
                                     ('spread h2 b0.3', 2, 0.3)):
                g = graph_spread(np.maximum(zb0, 0.0), adj, hops=hops)
                fields[name] = zb0 + beta * _zscore(
                    np.where(node_mask, g, np.nan), n)
            orders = {}
            for name, s in fields.items():
                order = rank_rows(s, node_mask)
                orders[name] = order
                pos = np.empty(n, dtype=int)
                pos[order] = np.arange(1, n + 1)
                ranks[name].extend(int(pos[r]) for r in sel_rows)
            pool = [(nid, soft_of.get((sess, epoch, seq, nid)))
                    for nid in pool_of.get((sess, epoch, seq), [])]
            for name, s in fields.items():
                xs, ys = soft_xy[name]
                for nid, sv in pool:
                    r = eng._resolve(nid)
                    if r is not None and sv is not None \
                            and np.isfinite(s[r]):
                        xs.append(float(s[r]))
                        ys.append(sv)
            top_b = set(orders['base'][:25].tolist())
            top_s = set(orders['spread h1 b0.3'][:25].tolist())
            brought += len([r for r in sel_rows
                            if r in top_s and r not in top_b])
            lost += len([r for r in sel_rows
                         if r in top_b and r not in top_s])
            # ---- C: MMR delivery from base top-50
            top50 = orders['base'][:50]
            v50 = prim[top50]
            sim = v50 @ v50.T
            zs50 = zb[top50]
            base5 = list(range(5))
            for lam in (0.3, 0.7):
                chosen = []
                cand = list(range(50))
                while len(chosen) < 5 and cand:
                    if chosen:
                        pen = sim[np.ix_(cand, chosen)].max(axis=1)
                    else:
                        pen = np.zeros(len(cand))
                    j = int(np.argmax(zs50[cand] - lam * pen))
                    chosen.append(cand.pop(j))
                dv = mmr[lam]
                dv['n_turn'] += 1
                dv['sel_in_5'] += len(
                    set(top50[chosen].tolist())
                    & set(sel_rows)) / max(len(sel_rows), 1)
                dv['redundancy'].append(float(
                    sim[np.ix_(chosen, chosen)][np.triu_indices(5, 1)]
                    .mean()))
                dv['base_redundancy'].append(float(
                    sim[np.ix_(base5, base5)][np.triu_indices(5, 1)]
                    .mean()))
            if (i + 1) % 20 == 0:
                print('  field %d/%d' % (i + 1, len(sample)))
    out_b = {}
    for a in B_ARMS:
        r = np.array(ranks[a])
        xs, ys = soft_xy[a]
        out_b[a] = {'sel_at_1': float((r == 1).mean()),
                    'sel_in_5': float((r <= 5).mean()),
                    'sel_in_25': float((r <= 25).mean()),
                    'median_rank': float(np.median(r)),
                    'soft_r_pool': (float(np.corrcoef(xs, ys)[0, 1])
                                    if len(xs) > 9 else None)}
    out_b['brought_vs_lost_h1b03_at25'] = [brought, lost]
    out_c = {str(lam): {
        'sel_in_5': d['sel_in_5'] / d['n_turn'],
        'redundancy': float(np.mean(d['redundancy'])),
        'base_redundancy': float(np.mean(d['base_redundancy']))}
        for lam, d in mmr.items()}
    return out_b, out_c


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    pool_of = defaultdict(list)
    for sess, epoch, seq, nid in walker.execute(
            "SELECT session_id, epoch, seq, node_id FROM candidates "
            "WHERE node_id IS NOT NULL"):
        pool_of[(sess, epoch, seq)].append(nid)

    lines = ['# settling_probes — renorm control, graph spread, MMR', '']

    a = probe_a(turns)
    lines.append('## A — renorm confound control (%s, op-cue, val)'
                 % CFG_A['name'])
    lines.append('| arm | sel@1 | sel-in-5 | AUC | soft_r |')
    lines.append('|---|---|---|---|---|')
    for k in ('linear', 'renorm', 'renorm_shuffled', 'fitted_linear'):
        m = a[k]
        lines.append('| %s | %.3f | %.3f | %.4f | %.3f |'
                     % (k, m['sel_at_1'], m['sel_in_5'], m['auc'],
                        m['soft_r']))
    lines.append('')
    lines.append('- coverage (single-message turns, linear vs renorm '
                 'order mismatches): %d'
                 % a['coverage_single_msg_order_mismatches'])
    lines.append('- fitted per-message weights: %s' % a['fitted_weights'])
    lines.append('')
    print('probe A done')

    b, c = probe_bc(walker, pool_of)
    walker.close()
    lines.append('## B — graph-spread settling (full field, %s, %d turns)'
                 % (CFG_B['name'], N_SAMPLE))
    lines.append('| arm | sel@1 | sel@5 | sel@25 | median | soft_r(pool) |')
    lines.append('|---|---|---|---|---|---|')
    for k in ('base', 'spread h1 b0.3', 'spread h1 b0.5', 'spread h2 b0.3'):
        m = b[k]
        lines.append('| %s | %.3f | %.3f | %.3f | %.0f | %s |'
                     % (k, m['sel_at_1'], m['sel_in_5'], m['sel_in_25'],
                        m['median_rank'],
                        '%.3f' % m['soft_r_pool'] if m['soft_r_pool']
                        is not None else '—'))
    lines.append('- spread h1 b0.3 @25: brought %d / lost %d selected '
                 'nodes vs base' % tuple(b['brought_vs_lost_h1b03_at25']))
    lines.append('')
    lines.append('## C — MMR delivery (top-5 from base top-50)')
    lines.append('| λ | sel-in-5 (share) | delivered redundancy | '
                 'base redundancy |')
    lines.append('|---|---|---|---|')
    for lam, m in c.items():
        lines.append('| %s | %.3f | %.3f | %.3f |'
                     % (lam, m['sel_in_5'], m['redundancy'],
                        m['base_redundancy']))
    lines.append('')

    OUT.write_text(json.dumps({'A': {k: v for k, v in a.items()},
                               'B': b, 'C': c}, indent=1, default=str))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
