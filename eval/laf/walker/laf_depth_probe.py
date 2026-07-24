"""Depth (@5/@10/@25) + the enc⊕pick merge test.

THREE QUESTIONS (Tom, 2026-07-24):
  1. @5 is one harsh threshold — expand every verdict to @10 and @25. This
     matters concretely: the enrichment lane is HEAVY-TAILED (laf_gate_audit T12 —
     rank 419→168, 80→23 rescues swamped by 1-rank taxes), and a rescue that
     lands at rank 23 scores 0 at @5 but converts at @25. A lane can be
     negative at @5 and positive at @25; only measuring both distinguishes
     "worthless" from "wrong threshold".
  2. Does `enc` hurt because it is TOO SPARSE (support ~59 of 7684)? If so,
     fusing it into `pick` BEFORE normalization should help: the union has
     larger support, so support-z is stabler and the lane stops manufacturing
     variance from a handful of nonzeros (the z-inflation pathology 0ccc5481).
     Merging after z-scoring would be meaningless — that is exactly what two
     separate gains already express. So the fuse is on RAW activations.
  3. Per-lane solo reach by cue sharpness (folded in from the episodic
     question) — is there ANY lane to lean on when the cosine field is flat?

Reuses the parity-checked fast scorer from laf_gate_audit. Read-only.
Run:  ./dev python3 eval/laf/walker/laf_depth_probe.py
"""
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))                 # `servers` — before any use
sys.path.insert(0, str(OUT_DIR))
from servers.recall_laf import zscore_variant                       # noqa: E402
from lambda_probe import zn                                         # noqa: E402
import laf_lane_audit as A                                         # noqa: E402
import laf_gate_audit as G                                         # noqa: E402

LANES5 = A.LANES
DEPTHS = (5, 10, 25)
GRID = G.GRID
GGRID = G.GGRID
REPORT = OUT_DIR / 'laf_depth_probe.md'


def merged_epi(t, mode):
    """Fuse pick+enc on RAW activations, then support-z once."""
    p, e = t['raw_epi']['pick'], t['raw_epi']['enc']
    if mode == 'max':
        raw = np.maximum(p, e)
    elif mode == 'sum':
        raw = p + e
    else:
        raise ValueError(mode)
    n = len(raw)
    return zscore_variant(raw.astype(np.float64), n, mask=t['alive'],
                          kind='support')


def prep_merged(turns, mode):
    """prep() but with pick/enc replaced by ONE fused epi lane.
    Lane order: maxsim, sit, idf, epi, graph  (5 columns)."""
    out = []
    for t in turns:
        U = np.flatnonzero(t['alive'])
        epi = merged_epi(t, mode)
        cols = [t['zl']['maxsim'][U], t['zl']['sit'][U], t['zl']['idf'][U],
                epi[U], t['enrichment_z'][U]]
        ZU = np.column_stack(cols).astype(np.float64)
        zmh = zn(t['mh'])[U]
        gpos = int(np.searchsorted(U, t['gr']))
        if gpos >= len(U) or U[gpos] != t['gr']:
            continue
        out.append({'Z': ZU, 'zmh': zmh, 'g': gpos, 't': t})
    return out


def reach_vec(prepped, gvec, at):
    """reach@at for an explicit gain vector over prepped's column order."""
    h, tot = 0, 0
    for p in prepped:
        r = G.rank_fast(p, np.asarray(gvec, dtype=float))
        if r is None:
            continue
        tot += 1
        h += int(r <= at)
    return 100.0 * h / tot if tot else 0.0


def main():
    turns, n = A.build()
    P = G.prep(turns)
    tt = [p['t'] for p in P]
    N = len(P)
    fold = G.folds_of(tt)
    ship = dict(A.GAINS)
    L = ['# Depth probe (@5/@10/@25) + the enc⊕pick merge', '',
         'n=%d clean valids ≥%s · fast scorer (parity-checked in '
         'laf_gate_audit) · tie-fair ranks' % (N, A.CUTOFF), '']

    # ══ D1. every Door-1/Door-2 verdict at three depths ══
    L += ['## D1. Depth expansion — the same arms at @5 / @10 / @25', '',
          'Held-out arms use the per-fold gains fit on TRAIN only (same CV as '
          'laf_gate_audit T1). Graph arms use the tuned base.', '',
          '| arm | @5 | @10 | @25 |', '|---|---|---|---|']
    # shipped
    row = []
    for at in DEPTHS:
        r, _ = G.reach_fast(P, ship, 0.0, at=at)
        row.append(r)
    L.append('| shipped gains | %.1f%% | %.1f%% | %.1f%% |' % tuple(row))
    ship_row = row

    # held-out refit per depth (gains fit maximizing TRAIN reach@that depth)
    heldout = {}
    for at in DEPTHS:
        h = np.full(N, np.nan)
        for f in range(5):
            tr = [P[i] for i in range(N) if fold[i] != f]
            te_i = [i for i in range(N) if fold[i] == f]
            best, bs = None, -1.0
            for init in (A.GAINS, {ln: 0.5 for ln in LANES5},
                         {ln: (1.5 if ln == 'maxsim' else 0.25)
                          for ln in LANES5}):
                g5 = dict(init)
                for _ in range(3):
                    for ln in LANES5:
                        bv, bsc = g5[ln], -1.0
                        for c in GRID:
                            s, _ = G.reach_fast(tr, {**g5, ln: c}, 0.0, at=at)
                            if s > bsc:
                                bsc, bv = s, c
                        g5[ln] = bv
                s, _ = G.reach_fast(tr, g5, 0.0, at=at)
                if s > bs:
                    bs, best = s, dict(g5)
            hh = G.hits_fast([P[i] for i in te_i], best, 0.0, at=at)
            for k, i in enumerate(te_i):
                h[i] = hh[k]
        heldout[at] = h
        print('held-out refit @%d done' % at)
    L.append('| refit (held-out) | ' + ' | '.join(
        '%.1f%%' % (100 * np.nanmean(heldout[at])) for at in DEPTHS) + ' |')
    L.append('| Δ vs shipped | ' + ' | '.join(
        '%+.1fpp' % (100 * np.nanmean(heldout[at]) - ship_row[i])
        for i, at in enumerate(DEPTHS)) + ' |')
    for at in DEPTHS:
        md, sd, lo, hi = G.boot_delta(heldout[at],
                                      G.hits_fast(P, ship, 0.0, at=at))
        L.append('| ⤷ bootstrap 95%% CI @%d | %s |'
                 % (at, '%+.2fpp [%+.2f, %+.2f] → %s'
                    % (md, lo, hi, 'REAL' if lo > 0 else
                       ('HARMFUL' if hi < 0 else 'NOISE'))))
    L.append('')

    # enrichment at three depths, tuned base, gain swept
    g_in, _, _ = G.multistart(P)
    L += ['### D1b. Enrichment lane at three depths (tuned base, gain swept)', '',
          'The heavy-tail test: a rescue landing at rank 23 is invisible at @5 '
          'and counts at @25.', '',
          '| gain_enrichment | @5 | @10 | @25 |', '|---|---|---|---|']
    for g in GGRID:
        row = []
        for at in DEPTHS:
            r, _ = G.reach_fast(P, g_in, g, at=at)
            row.append(r)
        L.append('| %.2f | %.1f%% | %.1f%% | %.1f%% |' % (g, *row))
    base = {at: G.reach_fast(P, g_in, 0.0, at=at)[0] for at in DEPTHS}
    bestg = {}
    for at in DEPTHS:
        cand = [(G.reach_fast(P, g_in, g, at=at)[0], g) for g in GGRID]
        bestg[at] = max(cand)[1]
    L += ['', '- best enrichment gain per depth: ' + ' · '.join(
        '@%d → %.2f (%+.1fpp)'
        % (at, bestg[at], G.reach_fast(P, g_in, bestg[at], at=at)[0] - base[at])
        for at in DEPTHS), '']

    # ══ D2. per-lane SOLO reach at three depths ══
    L += ['## D2. Per-lane solo reach (lane alone ranks gold ≤ k)', '',
          '| lane | support (med) | @5 | @10 | @25 |', '|---|---|---|---|---|']
    ALL6 = tuple(LANES5) + ('enrichment',)
    for li, ln in enumerate(ALL6):
        sup, rows = [], {at: 0 for at in DEPTHS}
        tot = 0
        for p in P:
            z = p['Z'][:, li]
            sup.append(int(np.sum(np.abs(z) > 1e-9)))
            gv = z[p['g']]
            fin = np.where(np.isfinite(z), z, -np.inf)
            rk = int((fin > gv).sum()) + (int((fin == gv).sum()) - 1) / 2.0 + 1
            tot += 1
            for at in DEPTHS:
                rows[at] += int(rk <= at)
        L.append('| %s | %.0f | %.0f%% | %.0f%% | %.0f%% |'
                 % (ln, np.median(sup),
                    *[100 * rows[at] / tot for at in DEPTHS]))
    L.append('')

    # ══ D3. lane solo reach by cue sharpness (the episodic question) ══
    cmz = np.array([t['cur_maxz'] for t in tt])
    qs = np.percentile(cmz, [25, 50, 75])
    quart = np.digitize(cmz, qs)
    L += ['## D3. Is there a lane to lean on when the cosine field is flat?',
          '', 'Per-lane solo reach@10 by cur_maxz quartile. If every lane '
          'degrades together, no reweighting can rescue a vague cue — they all '
          'read the same flat geometry.', '',
          '| lane | Q1 (flattest) | Q2 | Q3 | Q4 (sharpest) | Q1−Q4 |',
          '|---|---|---|---|---|---|']
    for li, ln in enumerate(ALL6):
        vals = []
        for q in range(4):
            idx = [i for i in range(N) if quart[i] == q]
            hit = 0
            for i in idx:
                p = P[i]
                z = p['Z'][:, li]
                gv = z[p['g']]
                fin = np.where(np.isfinite(z), z, -np.inf)
                rk = int((fin > gv).sum()) + (int((fin == gv).sum()) - 1) / 2.0 + 1
                hit += int(rk <= 10)
            vals.append(100.0 * hit / max(1, len(idx)))
        L.append('| %s | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %+.0fpp |'
                 % (ln, *vals, vals[0] - vals[3]))
    L.append('')

    # ══ D4. enc ⊕ pick merge ══
    L += ['## D4. Does `enc` hurt because it is too sparse? (the merge test)',
          '', 'Fuse pick+enc on RAW activations then support-z ONCE, so the '
          'union carries larger support. Merging after z would just be the '
          'additive sum two separate gains already express.', '']
    # mechanism diagnostics first
    L += ['### D4a. Mechanism — does fusing actually fix the sparsity?', '',
          '| lane | median support | median peak z | mean gold z |',
          '|---|---|---|---|']
    diag = {}
    for name in ('pick', 'enc', 'epi_max', 'epi_sum'):
        sups, peaks, gz = [], [], []
        for p in P:
            t = p['t']
            if name in ('pick', 'enc'):
                z = t['zl'][name]
            else:
                z = merged_epi(t, name.split('_')[1])
            sups.append(int(np.sum(np.abs(z) > 1e-9)))
            fin = z[np.isfinite(z)]
            if fin.size:
                peaks.append(float(np.nanmax(z)))
            if np.isfinite(z[t['gr']]):
                gz.append(float(z[t['gr']]))
        diag[name] = (np.median(sups), np.median(peaks), np.mean(gz))
        L.append('| %s | %.0f | %.2f | %+.2f |' % (name, *diag[name]))
    L.append('')

    # arms at three depths
    L += ['### D4b. Arms — separate vs merged (best gain per arm, in-sample)',
          '', '| arm | best gains | @5 | @10 | @25 |', '|---|---|---|---|---|']

    def sweep_sep(at):
        """best (pick,enc) pair at shipped-others."""
        best, bs = None, -1.0
        for gp in GRID:
            for ge in GRID:
                r, _ = G.reach_fast(P, {**ship, 'pick': gp, 'enc': ge},
                                    0.0, at=at)
                if r > bs:
                    bs, best = r, (gp, ge)
        return best, bs

    def sweep_merged(mode, at, pm):
        best, bs = None, -1.0
        for ge in GRID:
            gv = [ship['maxsim'], ship['sit'], ship['idf'], ge, 0.0]
            r = reach_vec(pm, gv, at)
            if r > bs:
                bs, best = r, ge
        return best, bs

    PM = {m: prep_merged(turns, m) for m in ('max', 'sum')}
    rows = {}
    for at in DEPTHS:
        (gp, ge), r_sep = sweep_sep(at)
        rows.setdefault('separate (pick,enc)', {})[at] = (r_sep, '%.2f/%.2f' % (gp, ge))
        for m in ('max', 'sum'):
            gm, rm = sweep_merged(m, at, PM[m])
            rows.setdefault('merged epi (%s)' % m, {})[at] = (rm, '%.2f' % gm)
        r_po, _ = G.reach_fast(P, {**ship, 'enc': 0.0}, 0.0, at=at)
        rows.setdefault('pick only (enc=0)', {})[at] = (r_po, '%.2f/0' % ship['pick'])
        r_sh, _ = G.reach_fast(P, ship, 0.0, at=at)
        rows.setdefault('shipped (0.5/0.3)', {})[at] = (r_sh, '0.50/0.30')
    for name in ('shipped (0.5/0.3)', 'pick only (enc=0)',
                 'separate (pick,enc)', 'merged epi (max)',
                 'merged epi (sum)'):
        L.append('| %s | %s | %.1f%% | %.1f%% | %.1f%% |'
                 % (name, rows[name][5][1],
                    *[rows[name][at][0] for at in DEPTHS]))
    L.append('')
    L += ['- NOTE: these are in-sample best-gain arms (a fair arm-vs-arm '
          'comparison, but each has the same in-sample optimism ~+0.8pp as '
          'Door 1). Only a difference LARGER than that is interesting.', '']

    # held-out merged vs separate at @10 (the one CV that matters here)
    L += ['### D4c. Held-out: merged vs separate (fit gains on TRAIN, @10)',
          '', '| arm | held-out reach@10 |', '|---|---|']
    for label, pset, lanes_n in (('separate (5 lanes)', P, 5),
                                 ('merged epi max (4 lanes)', PM['max'], 4)):
        h = np.full(len(pset), np.nan)
        for f in range(5):
            fo = G.folds_of([p['t'] for p in pset])
            tr = [pset[i] for i in range(len(pset)) if fo[i] != f]
            te_i = [i for i in range(len(pset)) if fo[i] == f]
            if lanes_n == 5:
                g5 = dict(ship)
                for _ in range(2):
                    for ln in LANES5:
                        bv, bsc = g5[ln], -1.0
                        for c in GRID:
                            s, _ = G.reach_fast(tr, {**g5, ln: c}, 0.0, at=10)
                            if s > bsc:
                                bsc, bv = s, c
                        g5[ln] = bv
                hh = G.hits_fast([pset[i] for i in te_i], g5, 0.0, at=10)
            else:
                gv = [ship['maxsim'], ship['sit'], ship['idf'], 0.5, 0.0]
                for _ in range(2):
                    for j in range(4):
                        bv, bsc = gv[j], -1.0
                        for c in GRID:
                            cand = list(gv)
                            cand[j] = c
                            s = reach_vec(tr, cand, 10)
                            if s > bsc:
                                bsc, bv = s, c
                        gv[j] = bv
                hh = []
                for i in te_i:
                    r = G.rank_fast(pset[i], np.array(gv))
                    hh.append(np.nan if r is None else float(r <= 10))
                hh = np.array(hh)
            for k, i in enumerate(te_i):
                h[i] = hh[k]
        L.append('| %s | %.1f%% |' % (label, 100 * np.nanmean(h)))
        print('held-out merge arm %s done' % label)
    L.append('')

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
