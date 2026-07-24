"""LAF composition audit — the enrichment lane as a first-class 6th lane.

Answers Tom's three questions on the REACH substrate (rank gold among all
7684 nodes → reach@5, where the enrichment lane's rescue value actually lives):

  RECORD   per-(turn × lane) atomic dump (support, gold rank-in-lane tie-fair,
           sole-reacher, peak) for all 6 lanes incl enrichment → laf_atoms.jsonl
  GAINS?   add-one / leave-one-out marginal reach + JOINT coordinate-ascent
           refit of the 6-gain vector — does adding enrichment shift the others?
           (LAF composition is non-additive: cd74b974)
  SETTLE   per-lane displacement — mean gold-rank MOVEMENT each lane causes
           (b6a4dc6b: a field is measured by how it moves anchors)

Composition (production shape): f0 = Σ_lane gain·zscore(lane_op0); the graph
lane joins the op0 sum (it seeds from the current message). Slot arbitration
stays fixed: mix = 0.65·zn(f0) + 0.35·zn(mh). Each lane is z-scored to unit
variance first, so a gain is a pure influence dial.

GATE 0 (parity) runs first: recompose f0 from lanes at current gains, assert
== field_cache. Baseline reach must reproduce the committed 51%. Precompute
z-lanes + enrichment_z + mh ONCE per turn → gain vectors re-score in milliseconds.

Read-only. Run:  ./dev python3 eval/laf/walker/laf_lane_audit.py
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from walker_db import OUT_DIR, WALKER_DB, open_ro, open_brain_ro

sys.path.append(str(OUT_DIR))   # DATA dir may be another tree: append so
                                 # THIS tree's code wins, while main-tree-only
                                 # helpers (lambda_probe, miss_anatomy) resolve
from lambda_probe import zn                                          # noqa: E402
from layer_readout_probe import lane_z                             # noqa: E402
from miss_anatomy import rank_in                                    # noqa: E402
from servers.recall_laf import zscore_variant                      # noqa: E402
import enrichment_lane as GL                                            # noqa: E402

CUTOFF = '2026-05-11'
LANES = ('maxsim', 'sit', 'idf', 'pick', 'enc')       # cache lane order
SPARSE = ('pick', 'enc', 'idf')                        # support-z lanes
GAINS = {'maxsim': 1.0, 'sit': 0.5, 'idf': 0.5, 'pick': 0.5, 'enc': 0.3}
LAM = 0.65                                             # op0-vs-history mix
PARITY_TOL = 5e-3
GAMMA = 0.5
REPORT = OUT_DIR / 'laf_lane_audit.md'
ATOMS = OUT_DIR / 'laf_atoms.jsonl'


def znorm(raw, lane, alive, n):
    kind = 'support' if lane in SPARSE else 'current'
    src = np.where(np.isfinite(raw), raw, 0.0) if lane in SPARSE else raw
    return zscore_variant(src, n, mask=alive, kind=kind)


def moment_history(F, S, n):
    a1, f1, f2 = F[S['anchor1']], F[S['op1']], F[S['op2']]
    a2 = F[S['anchor2']] if 'anchor2' in S else None
    parts = [(GAMMA, a1), (GAMMA ** 2, f1), (GAMMA ** 4, f2)]
    if a2 is not None:
        parts.insert(2, (GAMMA ** 3, a2))
    mh = np.zeros(n)
    pres = np.zeros(n, dtype=bool)
    for wt, fld in parts:
        if fld is None or np.isnan(fld).all():
            continue
        fin = np.isfinite(fld)
        mh += wt * np.where(fin, fld, 0.0)
        pres |= fin
    mh[~pres] = np.nan
    return mh


def load_turns():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    fields = np.load(OUT_DIR / 'field_cache.npy', mmap_mode='r')
    lanes_mm = np.load(OUT_DIR / 'lane_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    m2i = {sid: i for i, sid in enumerate(idx['master'])}
    n = idx['n_nodes']
    verds = {json.loads(x)['key']: json.loads(x)
             for x in (OUT_DIR / 'corpus_v2_verdicts.jsonl').open()}
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}
    w = open_ro(WALKER_DB)
    qvecs = GL.build_qvecs(w)
    w.close()
    b = open_brain_ro()
    node_meta = GL.build_node_meta(b, m2i)
    adj = GL.build_adjacency(b, m2i)
    b.close()
    return (idx, fields, lanes_mm, S, m2i, n, verds, bundles, qvecs,
            node_meta, adj)


def gate0(idx, fields, lanes_mm, S, n):
    worst, checked = 0.0, 0
    for t in idx['turns'][:60]:
        stored = fields[t['row'], S['op0']].astype(np.float64)
        if np.isnan(stored).all():
            continue
        Lr = lanes_mm[t['row']].astype(np.float64)
        alive = np.isfinite(Lr[S['op0'], 0])
        rec = np.zeros(n)
        for li, ln in enumerate(LANES):
            rec += GAINS[ln] * znorm(Lr[S['op0'], li], ln, alive, n)
        rec[~alive] = np.nan
        both = np.isfinite(rec) & np.isfinite(stored)
        if both.any():
            worst = max(worst, float(np.abs(rec[both] - stored[both]).max()))
            checked += 1
    if worst >= PARITY_TOL:
        raise SystemExit('GATE 0 FAIL |Δ| %.3g over %d turns' % (worst, checked))
    return worst, checked


def build(spec=GL.DEFAULT_SCORE_SPEC):
    """Precompute per valid turn: z-lanes (5), enrichment_z, mh, gold row, meta.
    Everything a gain vector needs to re-score in one dot product."""
    (idx, fields, lanes_mm, S, m2i, n, verds, bundles, qvecs,
     node_meta, adj) = load_turns()
    worst, checked = gate0(idx, fields, lanes_mm, S, n)
    print('GATE 0 parity OK |Δ|max %.2e over %d turns\n' % (worst, checked))

    turns = []
    for t in idx['turns']:
        key = '%s/%d/%d' % tuple(t['key'])
        v = verds.get(key)
        bd = bundles.get(key)
        if not v or v['verdict'] != 'valid' or not bd or (bd['ts'] or '') < CUTOFF:
            continue
        gi = t.get('gold_i')
        if gi is None:
            continue
        gr = t['cand_rows'][gi]
        if gr < 0:
            continue
        F = fields[t['row']].astype(np.float32)
        if np.isnan(F[S['op0']]).all() or not np.isfinite(F[S['op0']][gr]):
            continue
        Lr = lanes_mm[t['row']].astype(np.float32)
        alive = np.isfinite(Lr[S['op0'], 0])
        zl = {ln: znorm(Lr[S['op0'], li], ln, alive, n)
              for li, ln in enumerate(LANES)}
        # RAW (pre-normalization) episodic activations — a merged epi lane has
        # to be fused BEFORE support-z, otherwise "merging" is just the
        # additive sum of two z-lanes that separate gains already express.
        raw_epi = {ln: np.where(np.isfinite(Lr[S['op0'], LANES.index(ln)]),
                                Lr[S['op0'], LANES.index(ln)],
                                0.0).astype(np.float32)
                   for ln in ('pick', 'enc')}
        mh = moment_history(F, S, n)
        turn_dt = GL.iso(bd['ts'])
        qv = qvecs.get(tuple(t['key']))
        seeds, sz = GL.seed_rows(lanes_mm, t['row'], S, n)
        g_raw, kept = GL.enrichment_activation(seeds, sz, adj, qv, turn_dt,
                                          node_meta, n, spec)
        enrichment_z = zscore_variant(g_raw, n, mask=alive, kind='support')
        # gold's convergence (n seeds reaching it) — a per-message feature
        neigh = GL.aggregate_neighbors(seeds, sz, adj, qv, turn_dt)
        gd = neigh.get(gr)
        # per-candidate labels (within-pool substrate — dynamic-gain fit)
        cand_rows = np.array(t['cand_rows'])
        soft = np.array([np.nan if x is None else x for x in t['soft']]) \
            if t.get('soft') else np.full(len(cand_rows), np.nan)
        sel = np.array(t['sel'], dtype=bool) if t.get('sel') \
            else np.zeros(len(cand_rows), dtype=bool)
        # graph-shape features (observable at inference — no gold peeking)
        n_conv2 = sum(1 for d in neigh.values() if len(d['seeds']) >= 2)
        max_conv = max((len(d['seeds']) for d in neigh.values()), default=0)
        turns.append({
            'key': key, 'sess': t['key'][0],
            'stratum': v['stratum'],
            'door': 'door-1' if v['stratum'] == 'cue' else 'door-2',
            'gr': gr, 'alive': alive,
            'zl': zl, 'raw_epi': raw_epi,
            'enrichment_z': enrichment_z, 'enrichment_raw': g_raw, 'mh': mh,
            'enrichment_support': len(kept), 'enrichment_n_conv2': n_conv2,
            'enrichment_max_conv': max_conv,
            'cand_rows': cand_rows, 'soft': soft, 'sel': sel,
            'gold_type': node_meta.get(gr, (None, 0))[0],
            'gold_in_enrichment': gr in kept,
            'gold_seeds': len(gd['seeds']) if gd else 0,
            'cur_maxz': float(np.nanmax(zl['maxsim'])),
        })
    built = len(turns)
    # ONE universe: turns the baseline mix can rank (gold finite in f0 AND
    # mh). Golds NaN in mh are unrankable by the current mix — a separate
    # question (absence from moment history), excluded here so reach, sweep,
    # LOO, and ceiling all share the SAME denominator. Rankability is
    # gain-invariant (gains scale finite lanes; mh is fixed).
    turns = [t for t in turns if mix_rank(t, GAINS, 0.0) is not None]
    print('valid turns: %d built · %d rankable-in-mix (cue %d / window %d / '
          'session %d)\n'
          % (built, len(turns),
             sum(t['stratum'] == 'cue' for t in turns),
             sum(t['stratum'] == 'window' for t in turns),
             sum(t['stratum'] == 'session' for t in turns)))
    return turns, n


def f0_of(t, gains, g_enrichment):
    f = np.zeros_like(t['enrichment_z'])
    for ln in LANES:
        f = f + gains[ln] * t['zl'][ln]
    if g_enrichment:
        f = f + g_enrichment * t['enrichment_z']
    f[~t['alive']] = np.nan
    return f


def mix_rank(t, gains, g_enrichment):
    f0 = f0_of(t, gains, g_enrichment)
    mix = LAM * zn(f0) + (1 - LAM) * zn(t['mh'])
    return rank_in(mix, t['gr'])


def reach(turns, gains, g_enrichment, stratum=None, at=5):
    sub = [t for t in turns if stratum is None or t['stratum'] == stratum]
    h = n = 0
    for t in sub:
        r = mix_rank(t, gains, g_enrichment)
        if r is None:
            continue
        n += 1
        h += int(r <= at)
    return 100.0 * h / n if n else 0.0, n


def strata_row(turns, gains, g_enrichment):
    return {s: reach(turns, gains, g_enrichment, s)[0]
            for s in (None, 'cue', 'window', 'session')}


def main():
    turns, n = build()

    # baseline parity (must ~= committed 51%)
    base_all, N = reach(turns, GAINS, 0.0)
    L = ['# LAF composition audit — enrichment as a 6th lane (reach substrate)', '',
         'n=%d clean valids ≥%s · composition f0=Σgain·z(lane), '
         'mix=%.2f·zn(f0)+%.2f·zn(mh) · tie-fair ranks' % (N, CUTOFF, LAM, 1 - LAM),
         '', 'CROSS-CHECK baseline reach@5 = %.0f%% (committed 51%%) — %s'
         % (base_all, 'MATCH' if abs(base_all - 51) <= 2 else 'DRIFT!'), '']

    # ── RECORD: atomic per-(turn × lane) dump + descriptive ──────────────
    ALL6 = LANES + ('enrichment',)
    with ATOMS.open('w') as fh:
        for t in turns:
            rec = {'key': t['key'], 'stratum': t['stratum'],
                   'gold_type': t['gold_type'], 'cur_maxz': t['cur_maxz'],
                   'mix_rank': mix_rank(t, GAINS, 0.0),
                   'enrichment_support': t['enrichment_support'],
                   'gold_in_enrichment': t['gold_in_enrichment'],
                   'lanes': {}}
            zg = {**t['zl'], 'enrichment': t['enrichment_z']}
            reach_ranks = {}
            for ln in ALL6:
                z = zg[ln]
                gz = z[t['gr']]
                support = int(np.sum(np.abs(z) > 1e-9))
                gr_rank = rank_in(z, t['gr'])
                reach_ranks[ln] = gr_rank
                rec['lanes'][ln] = {
                    'support': support,
                    'gold_z': float(gz) if np.isfinite(gz) else None,
                    'gold_rank': gr_rank,
                    'peak': float(np.nanmax(z)) if np.isfinite(z).any() else None,
                }
            reachers = [ln for ln, r in reach_ranks.items()
                        if r is not None and r <= 5]
            rec['sole_reacher'] = reachers[0] if len(reachers) == 1 else None
            fh.write(json.dumps(rec) + '\n')

    # descriptive: per lane, hits vs misses (mix), gold-rank≤5 rate + sole
    hits = [t for t in turns if (mix_rank(t, GAINS, 0.0) or 99) <= 5]
    miss = [t for t in turns if (mix_rank(t, GAINS, 0.0) or 99) > 5]
    L += ['## RECORD — per-lane descriptive (hits vs misses), 6 lanes',
          '', 'support = #nonzero z · gold≤5 = lane ALONE ranks gold ≤5 '
          '(tie-fair) · sole = lane is the only ≤5 reacher',
          '', '| lane | grp | support | gold z (mean) | gold≤5 |',
          '|---|---|---|---|---|']
    for ln in ALL6:
        for lbl, grp in (('hit', hits), ('miss', miss)):
            zs = [({**t['zl'], 'enrichment': t['enrichment_z']})[ln] for t in grp]
            sup = np.mean([np.sum(np.abs(z) > 1e-9) for z in zs])
            gzs = [z[t['gr']] for t, z in zip(grp, zs)
                   if np.isfinite(z[t['gr']])]
            g5 = 100 * np.mean([1 if (rank_in(z, t['gr']) or 99) <= 5 else 0
                                for t, z in zip(grp, zs)])
            L.append('| %s | %s | %.0f | %s | %.0f%% |'
                     % (ln, lbl, sup,
                        '%.2f' % np.mean(gzs) if gzs else '—', g5))
    # sole-reacher census incl enrichment
    sole = Counter()
    for t in turns:
        zg = {**t['zl'], 'enrichment': t['enrichment_z']}
        rr = [ln for ln in ALL6 if (rank_in(zg[ln], t['gr']) or 99) <= 5]
        if len(rr) == 1:
            sole[rr[0]] += 1
    L += ['', '**Sole-reacher census** (gold reached ≤5 by exactly one lane):',
          '', '| lane | sole count | %% of valids |', '|---|---|---|']
    for ln, c in sole.most_common():
        L.append('| %s | %d | %.0f%% |' % (ln, c, 100 * c / N))
    L.append('')

    # ── GAINS? — enrichment gain sweep (per stratum) ──────────────────────────
    L += ['## GAINS? (1) enrichment-gain sweep — reach@5 per stratum', '',
          '| gain_enrichment | all | cue | window | session |', '|---|---|---|---|---|']
    GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5]
    best_g, best_r = 0.0, base_all
    for g in GRID:
        r = strata_row(turns, GAINS, g)
        if r[None] > best_r:
            best_r, best_g = r[None], g
        L.append('| %.2f | %.1f%% | %.1f%% | %.1f%% | %.1f%% |'
                 % (g, r[None], r['cue'], r['window'], r['session']))
    L += ['', '- best fixed gain_enrichment = **%.2f** → reach@5 %.1f%% '
          '(baseline %.1f%%, %+.1fpp)' % (best_g, best_r, base_all,
                                          best_r - base_all), '']

    # ceiling: oracle graph (a miss is rescued iff its gold is a base-union
    # graph neighbor). Same N as baseline. CROSS-CHECK rescuable ≈ 52 (the
    # committed hop_refine base-union count) — a self-verifying gate.
    rescuable = sum(1 for t in turns
                    if mix_rank(t, GAINS, 0.0) > 5 and t['gold_in_enrichment'])
    ceil_pct = base_all + 100.0 * rescuable / N
    xc = 'MATCH' if abs(rescuable - 52) <= 3 else 'DRIFT vs committed 52!'
    L += ['- rescuable misses (gold ∈ base-union graph neighbors): **%d** / '
          '%d misses — cross-check vs committed 52: %s' % (rescuable, N - int(round(base_all * N / 100)), xc),
          '- REACH CEILING (oracle graph, every rescuable miss converted): '
          '%.1f%% (%+.1fpp) — the prize; gain/scoring decides real conversion'
          % (ceil_pct, ceil_pct - base_all), '']

    # ── GAINS? — marginal (LOO) at current gains + best graph ────────────
    L += ['## GAINS? (2) leave-one-out — does each lane earn its place?', '',
          'reach@5 with one lane zeroed (enrichment at best fixed gain=%.2f). '
          'Big drop = load-bearing; ~0 = dead weight; rise = harmful.'
          % best_g, '', '| lane zeroed | reach@5 | Δ vs full |', '|---|---|---|']
    full6 = dict(GAINS)
    full_r, _ = reach(turns, full6, best_g)
    L.append('| (none — full+enrichment) | %.1f%% | — |' % full_r)
    for ln in LANES:
        g = {k: (0.0 if k == ln else v) for k, v in full6.items()}
        r, _ = reach(turns, g, best_g)
        L.append('| %s | %.1f%% | %+.1fpp |' % (ln, r, r - full_r))
    r_nog, _ = reach(turns, full6, 0.0)
    L.append('| enrichment | %.1f%% | %+.1fpp |' % (r_nog, r_nog - full_r))
    L.append('')

    # ── GAINS? — joint MULTI-START coordinate-ascent refit ───────────────
    L += ['## GAINS? (3) joint refit — does adding enrichment shift the others?',
          '', 'MULTI-START coordinate ascent (keep best over diverse inits, '
          '4 passes each) on reach@5, grid {0,.25,.5,.75,1,1.25,1.5}. Greedy '
          'ascent from one start is unreliable (coupled gains → local optima); '
          'multi-start + a clean "graph on the tuned base" row guard it. '
          'Non-additive: cd74b974.', '']
    CA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    INITS = [GAINS,
             {ln: 0.5 for ln in LANES},
             {ln: (1.0 if ln == 'maxsim' else 0.0) for ln in LANES},
             {ln: (1.5 if ln == 'maxsim' else 0.25) for ln in LANES}]

    def coord_ascent(init_g, init_gg, include_enrichment, passes=4):
        g, gg = dict(init_g), init_gg
        keys = list(LANES) + (['enrichment'] if include_enrichment else [])
        for _ in range(passes):
            for k in keys:
                best_val, best_score = None, -1
                for cand in CA_GRID:
                    if k == 'enrichment':
                        s, _ = reach(turns, g, cand)
                    else:
                        s, _ = reach(turns, {**g, k: cand}, gg)
                    if s > best_score:
                        best_score, best_val = s, cand
                if k == 'enrichment':
                    gg = best_val
                else:
                    g[k] = best_val
        r, _ = reach(turns, g, gg)
        return g, gg, r

    def multistart(include_enrichment, extra_inits=()):
        best = None
        for ig in list(INITS) + list(extra_inits):
            g, gg, r = coord_ascent(ig, best_g if include_enrichment else 0.0,
                                    include_enrichment)
            if best is None or r > best[2]:
                best = (g, gg, r)
        return best

    g_nog, _, r_refit_nog = multistart(False)
    # +enrichment search also SEEDED from the no-enrichment optimum (gain_enrichment=0
    # reachable → +enrichment ≥ no-enrichment by construction; any shortfall is ascent
    # noise, not a real "graph hurts").
    g_wg, gg_wg, r_refit_wg = multistart(True, extra_inits=[g_nog])
    # clean marginal: graph swept ON TOP of the fixed no-enrichment optimum
    gtop, rtop = 0.0, r_refit_nog
    for g in CA_GRID:
        r, _ = reach(turns, g_nog, g)
        if r > rtop:
            rtop, gtop = r, g
    L += ['| arm | maxsim | sit | idf | pick | enc | enrichment | reach@5 |',
          '|---|---|---|---|---|---|---|---|',
          '| current (shipped) | 1.00 | 0.50 | 0.50 | 0.50 | 0.30 | 0.00 | %.1f%% |'
          % base_all,
          '| refit, no enrichment | %.2f | %.2f | %.2f | %.2f | %.2f | 0.00 | %.1f%% |'
          % (g_nog['maxsim'], g_nog['sit'], g_nog['idf'], g_nog['pick'],
             g_nog['enc'], r_refit_nog),
          '| no-enrichment optimum + graph on top | %.2f | %.2f | %.2f | %.2f | %.2f'
          ' | %.2f | %.1f%% |'
          % (g_nog['maxsim'], g_nog['sit'], g_nog['idf'], g_nog['pick'],
             g_nog['enc'], gtop, rtop),
          '| refit + graph (joint) | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.1f%% |'
          % (g_wg['maxsim'], g_wg['sit'], g_wg['idf'], g_wg['pick'],
             g_wg['enc'], gg_wg, r_refit_wg), '']

    # ── SETTLE — per-lane displacement (gold-rank movement) ──────────────
    L += ['## SETTLE — per-lane displacement (mean gold-rank pull)', '',
          'Δrank = rank(without lane) − rank(full+enrichment), over turns where '
          'gold rankable. Positive = the lane pulls the gold UP the field '
          '(b6a4dc6b: a field is measured by how it moves anchors).',
          '', '| lane | median Δrank | mean Δrank | % turns helped |',
          '|---|---|---|---|']
    base_ranks = {t['key']: mix_rank(t, full6, best_g) for t in turns}
    for ln in ALL6:
        if ln == 'enrichment':
            gains_wo, gg_wo = full6, 0.0
        else:
            gains_wo, gg_wo = {**full6, ln: 0.0}, best_g
        deltas = []
        for t in turns:
            rb = base_ranks[t['key']]
            rw = mix_rank(t, gains_wo, gg_wo)
            if rb is not None and rw is not None:
                deltas.append(rw - rb)
        deltas = np.array(deltas, dtype=float)
        L.append('| %s | %+.1f | %+.1f | %.0f%% |'
                 % (ln, np.median(deltas), np.mean(deltas),
                    100 * np.mean(deltas > 0)))
    L.append('')

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\natomic record → %s' % ATOMS)
    return 0


if __name__ == '__main__':
    sys.exit(main())
