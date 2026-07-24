"""Router investigation — bug-gated, richly decomposed, held-out fit.

Tom: 'double-check yourself (bugs → wrong results); add more breakdowns —
lane relationships, size of nodes per lane, avg weight/variance — the more
signal the better we deduce a pattern.'

Structure:
  GATE 0  parity self-check — recompose F0 from op0 lanes at GAINS, assert
          == field_cache. If the lane math is wrong, HARD FAIL here (no
          silently-wrong downstream). Also cross-checks reach@5 vs the
          committed corpus_v2_eval number.
  PART A  per-lane descriptive (op0): support (#active nodes), peak z, mean,
          std, top-2 gap, gold rank, sole-reacher rate — HITS vs MISSES.
  PART B  lane RELATIONSHIPS: pairwise top-25 Jaccard (do lanes agree?),
          and gold-carrier co-occurrence.
  PART C  held-out per-message router fit with 4 anti-hallucination guards:
          (1) gold-blind features, (2) session-grouped 5-fold CV, (3) beat
          BEST-FIXED not current, (4) shuffle control + door↔door transfer.

Machinery imported (Turn, lane_z, rank_in, zn, lambda_star, GAINS,
zscore_variant). Cache-only. Run: ./dev python3 eval/laf/walker/corpus_v2_router.py
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from walker_db import OUT_DIR

sys.path.insert(0, str(OUT_DIR))
from q1_sweep import GAINS                                          # noqa: E402
from mesh_fit_probe import Turn                                     # noqa: E402
from lambda_probe import zn, lambda_star                           # noqa: E402
from layer_readout_probe import lane_z                             # noqa: E402
from miss_anatomy import rank_in                                   # noqa: E402
from servers.recall_laf import zscore_variant                     # noqa: E402

CUTOFF = '2026-05-11'
LANES = ('maxsim', 'sit', 'idf', 'pick', 'enc')
SPARSE = ('pick', 'enc', 'idf')
PARITY_TOL = 5e-3


def top2gap(z):
    fin = z[np.isfinite(z)]
    if fin.size < 2:
        return 0.0
    s = np.sort(fin)[::-1]
    return float(s[0] - s[1])


def topset(z, k=25):
    fin = np.where(np.isfinite(z), z, -np.inf)
    return set(np.argsort(-fin)[:k].tolist())


def main():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    fields = np.load(OUT_DIR / 'field_cache.npy', mmap_mode='r')
    lanes_mm = np.load(OUT_DIR / 'lane_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    n = idx['n_nodes']
    verds = {json.loads(x)['key']: json.loads(x)
             for x in (OUT_DIR / 'corpus_v2_verdicts.jsonl').open()}
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}

    # ══ GATE 0: parity self-check ══
    worst = 0.0
    checked = 0
    for t in idx['turns'][:60]:
        row = t['row']
        stored = fields[row, S['op0']].astype(np.float64)
        if np.isnan(stored).all():
            continue
        L = lanes_mm[row].astype(np.float64)
        mx = L[S['op0'], LANES.index('maxsim')]
        alive = np.isfinite(mx)
        rec = np.zeros(n)
        for li, ln in enumerate(LANES):
            col = L[S['op0'], li]
            kind = 'support' if ln in SPARSE else 'current'
            src = np.where(np.isfinite(col), col, 0.0) if ln in SPARSE else col
            rec += GAINS[ln] * zscore_variant(src, n, mask=alive, kind=kind)
        rec[~alive] = np.nan
        both = np.isfinite(rec) & np.isfinite(stored)
        if both.any():
            worst = max(worst, float(np.abs(rec[both] - stored[both]).max()))
            checked += 1
    if worst >= PARITY_TOL:
        raise SystemExit('PARITY FAIL: |Δ| %.4g over %d turns — lane math '
                         'does NOT reproduce field_cache; downstream invalid.'
                         % (worst, checked))
    print('GATE 0 parity OK: |Δ|max %.2e over %d turns (lane recompose == cache)\n'
          % (worst, checked))

    # collect per-turn per-lane stats over clean valids
    recs = []
    for t in idx['turns']:
        key = '%s/%d/%d' % tuple(t['key'])
        v = verds.get(key)
        b = bundles.get(key)
        if not v or v['verdict'] != 'valid' or not b or (b['ts'] or '') < CUTOFF:
            continue
        tt = Turn(t, fields, S)
        if tt.gr < 0 or tt.mh is None or np.isnan(tt.fields[0]).all():
            continue
        gr = tt.gr
        L = lanes_mm[t['row']].astype(np.float32)
        mx = L[S['op0'], LANES.index('maxsim')]
        alive = np.isfinite(mx)
        lz, lstat = {}, {}
        for li, ln in enumerate(LANES):
            raw = L[S['op0'], li]
            z = lane_z(raw, ln, alive, n)
            lz[ln] = z
            # support = # nodes with nonzero RAW activation (what "came back")
            support = int(np.sum(np.isfinite(raw) & (np.abs(raw) > 1e-9)))
            fin = z[np.isfinite(z)]
            lstat[ln] = {
                'support': support,
                'peak': float(np.nanmax(z)) if fin.size else np.nan,
                'mean': float(np.nanmean(z)) if fin.size else np.nan,
                'std': float(np.nanstd(z)) if fin.size else np.nan,
                'gap': top2gap(z),
                'grank': rank_in(z, gr),
            }
        mhr = rank_in(tt.mh, gr)
        rk = lambda_star(zn(tt.fields[0]), zn(tt.mh), gr, grid=np.array([0.65]))
        mix = min(rk.values()) if rk else None
        f0r = rank_in(tt.fields[0], gr)
        # sole-reacher: lanes that ALONE rank gold ≤5
        reachers = [ln for ln in LANES
                    if lstat[ln]['grank'] is not None and lstat[ln]['grank'] <= 5]
        recs.append({
            'key': key, 'sess': t['key'][0],
            'door': 'door-1' if v['stratum'] == 'cue' else 'door-2',
            'lz': {ln: topset(lz[ln]) for ln in LANES}, 'lstat': lstat,
            'mhr': mhr, 'f0': f0r, 'mix': mix,
            'hit': mix is not None and mix <= 5, 'reachers': reachers,
            'cur_maxz': lstat['maxsim']['peak'],
        })
    N = len(recs)
    hits = [r for r in recs if r['hit']]
    miss = [r for r in recs if not r['hit']]

    # cross-check reach vs committed number
    reach5 = 100.0 * len(hits) / N
    print('CROSS-CHECK reach@5 = %.0f%% (corpus_v2_eval.md reported 51%%) — '
          '%s\n' % (reach5, 'MATCH' if abs(reach5 - 51) <= 2 else 'DRIFT!'))

    L = ['# Router investigation — clean valids ≥%s, n=%d' % (CUTOFF, N), '']

    # ══ PART A: per-lane descriptive, hits vs misses ══
    L += ['## A. Per-lane descriptive (op0) — HITS vs MISSES', '',
          'support = # nodes with nonzero raw activation ("size of what came '
          'back"). peak/mean/std/gap = z-field stats. grank = gold rank in '
          'that lane alone. sole% = lane ALONE reaches gold ≤5.', '',
          '| lane | grp | support | peak z | mean | std | top2-gap | gold rank≤5 |',
          '|---|---|---|---|---|---|---|---|']
    def lane_block(ln):
        for lbl, grp in (('hit', hits), ('miss', miss)):
            s = [r['lstat'][ln] for r in grp]
            sup = np.mean([x['support'] for x in s])
            pk = np.mean([x['peak'] for x in s if np.isfinite(x['peak'])])
            mn = np.mean([x['mean'] for x in s if np.isfinite(x['mean'])])
            sd = np.mean([x['std'] for x in s if np.isfinite(x['std'])])
            gp = np.mean([x['gap'] for x in s])
            g5 = 100*np.mean([1 if (x['grank'] and x['grank'] <= 5) else 0
                              for x in s])
            L.append('| %s | %s | %.0f | %.2f | %.3f | %.2f | %.2f | %.0f%% |'
                     % (ln, lbl, sup, pk, mn, sd, gp, g5))
    for ln in LANES:
        lane_block(ln)
    L.append('')

    # ══ PART B: lane relationships ══
    L += ['## B. Lane relationships — pairwise top-25 Jaccard (agreement)', '',
          'How much do lanes agree on their top-25 nodes? Low Jaccard = '
          'independent lanes (composition adds reach); high = redundant.', '',
          '| pair | mean Jaccard |', '|---|---|']
    jacc = defaultdict(list)
    for r in recs:
        for i, a in enumerate(LANES):
            for bl in LANES[i+1:]:
                sa, sb = r['lz'][a], r['lz'][bl]
                u = len(sa | sb)
                jacc[(a, bl)].append(len(sa & sb)/u if u else 0.0)
    for pair, vals in sorted(jacc.items(), key=lambda x: -np.mean(x[1])):
        L.append('| %s∩%s | %.2f |' % (pair[0], pair[1], np.mean(vals)))
    L.append('')

    # sole-reacher census — the "one lane great" test done rigorously
    L += ['## B2. Sole-reacher census — which lane ALONE reaches golds no '
          'other does', '',
          'Of golds reached (≤5) by exactly ONE lane, which lane? This is the '
          '"one lane great where others fail" pattern, measured.', '',
          '| lane | sole-reacher count | as %% of all golds |', '|---|---|---|']
    sole = Counter()
    n_sole = 0
    for r in recs:
        if len(r['reachers']) == 1:
            sole[r['reachers'][0]] += 1
            n_sole += 1
    for ln, c in sole.most_common():
        L.append('| %s | %d | %.0f%% |' % (ln, c, 100*c/N))
    L += ['', '- %d/%d golds (%.0f%%) are reached by exactly one lane — the '
          'conditional-lane population. %d reached by ≥2 lanes (redundant), '
          '%d by none.'
          % (n_sole, N, 100*n_sole/N,
             sum(1 for r in recs if len(r['reachers']) >= 2),
             sum(1 for r in recs if len(r['reachers']) == 0)), '']

    # ══ PART C: held-out router fit ══
    L += ['## C. Held-out per-message router fit (4 guards)', '']
    # gold-blind feature matrix: lane peaks, gaps, supports, M_h-vs-maxsim conf
    def feats(r):
        s = r['lstat']
        return np.array([
            s['maxsim']['peak'], s['maxsim']['gap'], s['maxsim']['std'],
            s['sit']['peak'], s['idf']['peak'],
            np.log1p(s['pick']['support']), np.log1p(s['enc']['support']),
            np.log1p(s['idf']['support']),
        ])
    X = np.array([feats(r) for r in recs])
    # target: per-turn best λ over grid (oracle) — the thing a router predicts
    GRID = np.round(np.arange(0.0, 1.0001, 0.1), 1)
    best_lam = []
    for t in idx['turns']:
        pass
    # recompute per-rec oracle λ and per-λ rank (need f0z/mhz)
    lam_rank = {}  # key -> {lam: rank}
    for t in idx['turns']:
        key = '%s/%d/%d' % tuple(t['key'])
        if key not in {r['key'] for r in recs}:
            continue
        tt = Turn(t, fields, S)
        rk = lambda_star(zn(tt.fields[0]), zn(tt.mh), tt.gr, grid=GRID)
        lam_rank[key] = rk
    sess = np.array([r['sess'] for r in recs])
    usess = list(dict.fromkeys(sess.tolist()))
    # 5 session-grouped folds
    folds = {s: i % 5 for i, s in enumerate(usess)}
    fold = np.array([folds[s] for s in sess])

    def reach_at(keys, lam_map):
        h = 0
        for k in keys:
            rk = lam_rank.get(k)
            if not rk:
                continue
            lam = lam_map(k)
            r = rk.get(lam, min(rk.values()))
            if r is not None and r <= 5:
                h += 1
        return 100.0*h/len(keys) if keys else 0.0

    # best FIXED lam on full set (the honest baseline to beat)
    keys_all = [r['key'] for r in recs]
    fixed_scores = {lam: reach_at(keys_all, lambda k, lm=lam: lm) for lam in GRID}
    best_fixed_lam = max(fixed_scores, key=fixed_scores.get)
    best_fixed = fixed_scores[best_fixed_lam]
    oracle = 100.0*np.mean([1 if (lam_rank[k] and min(lam_rank[k].values()) <= 5)
                            else 0 for k in keys_all])

    # held-out router: fit λ = f(features) via ridge on train oracle-λ, apply on test
    from numpy.linalg import lstsq
    key2i = {r['key']: i for i, r in enumerate(recs)}
    y_orac = np.array([min(lam_rank[r['key']], key=lam_rank[r['key']].get)
                       if lam_rank[r['key']] else best_fixed_lam for r in recs])
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xs = np.hstack([Xs, np.ones((N, 1))])

    def cv_reach(Xmat):
        hit = 0
        for fi in range(5):
            tr = fold != fi
            te = fold == fi
            beta, *_ = lstsq(Xmat[tr], y_orac[tr], rcond=None)
            pred = np.clip(Xmat[te] @ beta, 0, 1)
            pred = np.round(pred, 1)
            for j, r in enumerate([recs[i] for i in np.where(te)[0]]):
                rk = lam_rank[r['key']]
                lam = float(pred[j])
                lam = min(GRID, key=lambda g: abs(g-lam))
                rr = rk.get(lam, min(rk.values())) if rk else None
                if rr is not None and rr <= 5:
                    hit += 1
        return 100.0*hit/N

    router_reach = cv_reach(Xs)
    # shuffle control: permute features across turns, refit
    rngseed = np.arange(N)[::-1]           # deterministic permutation (no RNG)
    Xshuf = np.hstack([Xs[rngseed, :-1], np.ones((N, 1))])
    shuf_reach = cv_reach(Xshuf)

    L += ['| config | reach@5 | vs best-fixed |', '|---|---|---|',
          '| best FIXED λ=%.1f | %.1f%% | — |' % (best_fixed_lam, best_fixed),
          '| held-out router (CV) | %.1f%% | %+.1fpp |' % (router_reach, router_reach - best_fixed),
          '| SHUFFLE control | %.1f%% | %+.1fpp |' % (shuf_reach, shuf_reach - best_fixed),
          '| oracle-λ (ceiling) | %.1f%% | %+.1fpp |' % (oracle, oracle - best_fixed),
          '']
    # door↔door transfer
    d1 = np.array([r['door'] == 'door-1' for r in recs])
    def transfer(train_mask, test_mask):
        beta, *_ = lstsq(Xs[train_mask], y_orac[train_mask], rcond=None)
        pred = np.clip(Xs[test_mask] @ beta, 0, 1)
        hit = 0
        tekeys = [recs[i] for i in np.where(test_mask)[0]]
        for j, r in enumerate(tekeys):
            rk = lam_rank[r['key']]
            lam = min(GRID, key=lambda g: abs(g-float(pred[j])))
            rr = rk.get(lam, min(rk.values())) if rk else None
            if rr is not None and rr <= 5:
                hit += 1
        return beta, 100.0*hit/len(tekeys)
    b_d1, r_d1ond2 = transfer(d1, ~d1)
    b_d2, r_d2ond1 = transfer(~d1, d1)
    signflip = int(np.sum(np.sign(b_d1[:-1]) != np.sign(b_d2[:-1])))
    L += ['**Cross-population transfer (the hallucination killer):**',
          '- fit door-1 → apply door-2: %.1f%% (door-2 best-fixed %.1f%%)'
          % (r_d1ond2, reach_at([r['key'] for r in recs if r['door'] == 'door-2'],
                                lambda k: best_fixed_lam)),
          '- fit door-2 → apply door-1: %.1f%%' % r_d2ond1,
          '- learned-weight SIGN FLIPS across doors: %d / %d features %s'
          % (signflip, len(b_d1)-1,
             '(β unstable → fitting population, not signal)' if signflip >= 3
             else '(β stable → real signal)'), '',
          '## Verdict',
          '- router beats best-fixed by %+.1fpp on held-out; shuffle by %+.1fpp. '
          '%s' % (router_reach - best_fixed, shuf_reach - best_fixed,
                  'Router > shuffle AND > 0 → real (proceed).'
                  if router_reach - best_fixed > 1.5 and router_reach > shuf_reach + 1
                  else 'Router ≈ shuffle or ≤ best-fixed → NO cheap router; '
                       'lever is a new reach signal + the confidence gate.')]

    (OUT_DIR / 'corpus_v2_router.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
