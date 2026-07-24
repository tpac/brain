"""Door 1 (held-out gain retune) + Door 2 (enrichment as a GATED ACTION).

WHY THIS EXISTS. laf_lane_audit answered "are the gains mistuned?" and "does
graph add reach?" IN-SAMPLE, and read reach@5 only. Two consequences, both
corrections Tom pushed:
  * an in-sample-tuned baseline is an INFLATED FLOOR — measuring a new lane's
    marginal against it understates the lane. Door 1 makes the floor honest
    (session-grouped CV) so Door 2 is a fair comparison.
  * lanes serve SORTING too, and reach@5 is a threshold metric blind to
    rank movement that doesn't cross 5. Door 2 measures the within-pool
    ordering substrate (picked/soft AUC) and the rank-movement distribution.

DOOR 2's FORM (the reframe): graph is not an always-on lane but a GATED
ACTION fired when the moment is reach-starved. Grounds: 8b3ef4f4 (cur_maxz
predicts difficulty +19pp Q1→Q4 but indicates NO better lane → use it as an
action trigger); 768e827a + 5381db10 (focus/fatigue wash out as GLOBAL
additive lanes, value lives in a conditional regime); biology 4e82d914
(phase GATING switches mode multiplicatively; it does not re-weight a sum).
So: global-vs-conditional is the measurement, not an aside.

FAST SCORER: the CV needs ~10^3 re-scorings. Per turn we precompute the
alive-restricted lane matrix ZU [n_alive × 6] and zn(mh); a gain vector is
then one matvec. PARITY-CHECKED against laf_lane_audit.reach (the audited
slow path) on shipped + random gain vectors before any result is trusted.

Read-only. Run:  ./dev python3 eval/laf/walker/laf_gate_audit.py
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from walker_db import OUT_DIR, open_brain_ro

sys.path.insert(0, str(OUT_DIR))
from lambda_probe import zn                                          # noqa: E402
import laf_lane_audit as A                                          # noqa: E402
from laf_dynamic_gains import cv_auc, LANES6                        # noqa: E402

LANES5 = A.LANES
FOLDS = 5
LAM = A.LAM                          # 0.65 op0-vs-history mix
GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
GGRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
N_BOOT = 2000
SEED = 20260724
REPORT = OUT_DIR / 'laf_gate_audit.md'


# ── fast scorer ──────────────────────────────────────────────────────────
def prep(turns):
    """Per turn: alive-restricted lane matrix + zn(mh) + gold position."""
    out = []
    for t in turns:
        U = np.flatnonzero(t['alive'])
        cols = [t['zl'][ln][U] for ln in LANES5] + [t['enrichment_z'][U]]
        ZU = np.column_stack(cols).astype(np.float64)
        zmh = zn(t['mh'])[U]
        gpos = int(np.searchsorted(U, t['gr']))
        if gpos >= len(U) or U[gpos] != t['gr']:
            continue                              # gold not alive — excluded
        out.append({'Z': ZU, 'zmh': zmh, 'g': gpos, 't': t})
    return out


def rank_fast(p, g6):
    f0 = p['Z'] @ g6
    if f0.size <= 2 or f0.std() <= 1e-9:
        return None
    zf0 = (f0 - f0.mean()) / f0.std()
    mix = LAM * zf0 + (1 - LAM) * p['zmh']
    gv = mix[p['g']]
    if not np.isfinite(gv):
        return None
    fin = np.where(np.isfinite(mix), mix, -np.inf)
    greater = int((fin > gv).sum())
    ties = int((fin == gv).sum())
    return greater + (ties - 1) / 2.0 + 1


def hits_fast(prepped, g5, enrichment_gain=0.0, at=5):
    """Per-turn hit indicators. enrichment_gain: scalar or per-turn array."""
    gg = (np.full(len(prepped), float(enrichment_gain))
          if np.isscalar(enrichment_gain) else np.asarray(enrichment_gain, dtype=float))
    hv = []
    for i, p in enumerate(prepped):
        g6 = np.array([g5[ln] for ln in LANES5] + [gg[i]])
        r = rank_fast(p, g6)
        hv.append(np.nan if r is None else float(r <= at))
    return np.array(hv)


def reach_fast(prepped, g5, enrichment_gain=0.0, at=5):
    h = hits_fast(prepped, g5, enrichment_gain, at)
    m = np.isfinite(h)
    return (100.0 * h[m].mean() if m.any() else 0.0), int(m.sum())


def ranks_fast(prepped, g5, enrichment_gain=0.0):
    gg = (np.full(len(prepped), float(enrichment_gain))
          if np.isscalar(enrichment_gain) else np.asarray(enrichment_gain, dtype=float))
    out = []
    for i, p in enumerate(prepped):
        g6 = np.array([g5[ln] for ln in LANES5] + [gg[i]])
        out.append(rank_fast(p, g6))
    return out


def parity_check(turns, prepped):
    """Fast scorer must reproduce the audited slow path (laf_lane_audit)."""
    rng = np.random.default_rng(1)
    cases = [dict(A.GAINS), {ln: 0.5 for ln in LANES5},
             {ln: float(rng.uniform(0, 1.5)) for ln in LANES5}]
    worst = 0.0
    for g5 in cases:
        for gg in (0.0, 0.5):
            slow, _ = A.reach(turns, g5, gg)
            fast, _ = reach_fast(prepped, g5, gg)
            worst = max(worst, abs(slow - fast))
    if worst > 0.15:
        raise SystemExit('FAST-SCORER PARITY FAIL: |Δreach| %.3fpp' % worst)
    return worst


# ── gain fitting ─────────────────────────────────────────────────────────
def coord_ascent(prepped, init5, init_g, fit_enrichment, passes=3):
    g5, gg = dict(init5), float(init_g)
    for _ in range(passes):
        for ln in LANES5:
            best, bs = g5[ln], -1.0
            for c in GRID:
                s, _ = reach_fast(prepped, {**g5, ln: c}, gg)
                if s > bs:
                    bs, best = s, c
            g5[ln] = best
        if fit_enrichment:
            best, bs = gg, -1.0
            for c in GGRID:
                s, _ = reach_fast(prepped, g5, c)
                if s > bs:
                    bs, best = s, c
            gg = best
    return g5, gg


def multistart(prepped, fit_enrichment=False, extra=()):
    inits = [A.GAINS, {ln: 0.5 for ln in LANES5},
             {ln: (1.5 if ln == 'maxsim' else 0.25) for ln in LANES5}]
    best = None
    for i5 in list(inits) + list(extra):
        g5, gg = coord_ascent(prepped, i5, 0.5 if fit_enrichment else 0.0,
                              fit_enrichment)
        r, _ = reach_fast(prepped, g5, gg)
        if best is None or r > best[2]:
            best = (g5, gg, r)
    return best


def folds_of(turns):
    sess = sorted({t['sess'] for t in turns})
    fo = {s: i % FOLDS for i, s in enumerate(sess)}
    return np.array([fo[t['sess']] for t in turns])


def boot_delta(hA, hB, n_boot=N_BOOT, seed=SEED):
    """Paired turn-level bootstrap of (reachA − reachB) in pp."""
    m = np.isfinite(hA) & np.isfinite(hB)
    a, b = hA[m], hB[m]
    rng = np.random.default_rng(seed)
    n = len(a)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(100.0 * (a[i].mean() - b[i].mean()))
    d = np.array(d)
    return float(d.mean()), float(d.std()), float(np.percentile(d, 2.5)), \
        float(np.percentile(d, 97.5))


def main():
    turns, n = A.build()
    prepped = prep(turns)
    worst = parity_check(turns, prepped)
    print('fast-scorer parity OK (|Δreach| %.3fpp vs audited slow path)\n'
          % worst)
    fold = folds_of([p['t'] for p in prepped])
    P = prepped
    tt = [p['t'] for p in P]
    N = len(P)

    L = ['# Door 1 + Door 2 — held-out gains, then enrichment as a gated action',
         '', 'n=%d clean valids ≥%s · %d-fold session-grouped CV · tie-fair '
         'ranks · fast scorer parity vs audited path |Δ| %.3fpp'
         % (N, A.CUTOFF, FOLDS, worst), '']

    ship = dict(A.GAINS)
    h_ship = hits_fast(P, ship, 0.0)
    r_ship = 100 * np.nanmean(h_ship)
    L += ['Shipped-gain baseline reach@5 = **%.1f%%** (cross-check vs '
          'committed 51%%: %s)' % (r_ship,
                                   'MATCH' if abs(r_ship - 51) <= 2 else 'DRIFT'),
          '']

    # ══════════════ DOOR 1 ══════════════
    L += ['## DOOR 1 — is the fixed-gain retune real out-of-sample?', '']

    # T1 fold-by-fold
    L += ['### T1. Fold-by-fold held-out reach@5 (gains fit on TRAIN only)',
          '', '| fold | train n | test n | shipped | refit (held-out) | Δ |',
          '|---|---|---|---|---|---|']
    fold_gains, h_refit = {}, np.full(N, np.nan)
    for f in range(FOLDS):
        tr = [P[i] for i in range(N) if fold[i] != f]
        te_i = [i for i in range(N) if fold[i] == f]
        te = [P[i] for i in te_i]
        g5, _, _ = multistart(tr)
        fold_gains[f] = g5
        hs = hits_fast(te, ship, 0.0)
        hr = hits_fast(te, g5, 0.0)
        for k, i in enumerate(te_i):
            h_refit[i] = hr[k]
        L.append('| %d | %d | %d | %.1f%% | %.1f%% | %+.1fpp |'
                 % (f, len(tr), len(te), 100 * np.nanmean(hs),
                    100 * np.nanmean(hr),
                    100 * (np.nanmean(hr) - np.nanmean(hs))))
        print('fold %d done' % f)
    r_cv = 100 * np.nanmean(h_refit)
    md, sd, lo, hi = boot_delta(h_refit, h_ship)
    L += ['| **pooled** | — | %d | **%.1f%%** | **%.1f%%** | **%+.1fpp** |'
          % (N, r_ship, r_cv, r_cv - r_ship), '',
          '- paired turn bootstrap (×%d) of the held-out Δ: **%+.2fpp** '
          '(sd %.2f, 95%% CI [%+.2f, %+.2f]) → %s'
          % (N_BOOT, md, sd, lo, hi,
             'REAL (CI excludes 0)' if lo > 0 else
             ('HARMFUL (CI below 0)' if hi < 0 else
              'INSIDE NOISE (CI spans 0)')), '']
    # in-sample reference for the inflation estimate
    g_in, _, r_in = multistart(P)
    L += ['- in-sample refit (the earlier number): %.1f%% (%+.1fpp). '
          'Optimism = in-sample − held-out = **%+.1fpp** — the inflation '
          'that made Door 2 look worthless.'
          % (r_in, r_in - r_ship, r_in - r_cv), '']

    # T2 gain stability across folds
    L += ['### T2. Per-fold learned gains — stability is the generalization tell',
          '', '| fold | ' + ' | '.join(LANES5) + ' |',
          '|' + '---|' * (len(LANES5) + 1)]
    for f in range(FOLDS):
        L.append('| %d | ' % f + ' | '.join('%.2f' % fold_gains[f][ln]
                                            for ln in LANES5) + ' |')
    L.append('| **shipped** | ' + ' | '.join('%.2f' % ship[ln]
                                             for ln in LANES5) + ' |')
    L.append('| **in-sample** | ' + ' | '.join('%.2f' % g_in[ln]
                                               for ln in LANES5) + ' |')
    L += ['', '| lane | fold mean | fold sd | shipped | verdict |',
          '|---|---|---|---|---|']
    for ln in LANES5:
        vals = np.array([fold_gains[f][ln] for f in range(FOLDS)])
        agree = 'STABLE' if vals.std() <= 0.2 else 'UNSTABLE'
        L.append('| %s | %.2f | %.2f | %.2f | %s |'
                 % (ln, vals.mean(), vals.std(), ship[ln], agree))
    L.append('')

    # T3 held-out by stratum
    L += ['### T3. Held-out reach@5 by stratum', '',
          '| stratum | n | shipped | refit (held-out) | Δ |',
          '|---|---|---|---|---|']
    for s in ('cue', 'window', 'session'):
        idx = [i for i in range(N) if tt[i]['stratum'] == s]
        a, b = h_refit[idx], h_ship[idx]
        L.append('| %s | %d | %.1f%% | %.1f%% | %+.1fpp |'
                 % (s, len(idx), 100 * np.nanmean(b), 100 * np.nanmean(a),
                    100 * (np.nanmean(a) - np.nanmean(b))))
    L.append('')

    # T4 held-out LOO — does enc stay dead out of sample?
    L += ['### T4. Held-out leave-one-out (fit without the lane, test)', '',
          'Per fold: refit the remaining gains on TRAIN with the lane forced '
          'to 0, evaluate on TEST. Honest "does this lane earn its place".',
          '', '| lane zeroed | held-out reach@5 | Δ vs full refit |',
          '|---|---|---|']
    L.append('| (none) | %.1f%% | — |' % r_cv)
    loo = {}
    for ln in LANES5:
        h = np.full(N, np.nan)
        for f in range(FOLDS):
            tr = [P[i] for i in range(N) if fold[i] != f]
            te_i = [i for i in range(N) if fold[i] == f]
            g5, _, _ = multistart(tr)
            g5 = {**g5, ln: 0.0}
            hr = hits_fast([P[i] for i in te_i], g5, 0.0)
            for k, i in enumerate(te_i):
                h[i] = hr[k]
        loo[ln] = h
        L.append('| %s | %.1f%% | %+.1fpp |'
                 % (ln, 100 * np.nanmean(h), 100 * np.nanmean(h) - r_cv))
        print('LOO %s done' % ln)
    L.append('')

    # T5 1-D sensitivity curves (shape, not just argmax)
    L += ['### T5. Gain sensitivity curves (others held at shipped)', '',
          'reach@5 as ONE gain sweeps. Flat = the dial barely matters; '
          'peaked = it does. Shows WHY an argmax can be noise.', '',
          '| lane | ' + ' | '.join('%.2f' % g for g in GRID) + ' |',
          '|' + '---|' * (len(GRID) + 1)]
    for ln in LANES5:
        row = []
        for g in GRID:
            r, _ = reach_fast(P, {**ship, ln: g}, 0.0)
            row.append('%.1f' % r)
        L.append('| %s | ' % ln + ' | '.join(row) + ' |')
    row = []
    for g in GRID:
        r, _ = reach_fast(P, ship, g)
        row.append('%.1f' % r)
    L.append('| enrichment | ' + ' | '.join(row) + ' |')
    L.append('')

    # ══════════════ DOOR 2 ══════════════
    L += ['## DOOR 2 — enrichment as a gated action (vs always-on lane)', '']

    cmz = np.array([t['cur_maxz'] for t in tt])
    qs = np.percentile(cmz, [25, 50, 75])
    quart = np.digitize(cmz, qs)          # 0..3

    # T7 gate calibration
    L += ['### T7. Gate calibration — cur_maxz quartiles (is the moment '
          'reach-starved?)', '',
          '| quartile | cur_maxz range | n | shipped reach@5 | enrichment-rescuable '
          '| graph support (median) | gold∈enrichment |', '|---|---|---|---|---|---|---|']
    for q in range(4):
        idx = [i for i in range(N) if quart[i] == q]
        rng_lo = cmz[idx].min()
        rng_hi = cmz[idx].max()
        resc = sum(1 for i in idx
                   if not (h_ship[i] == 1) and tt[i]['gold_in_enrichment'])
        sup = np.median([tt[i]['enrichment_support'] for i in idx])
        ging = 100 * np.mean([tt[i]['gold_in_enrichment'] for i in idx])
        L.append('| Q%d | %.2f–%.2f | %d | %.1f%% | %d (%.0f%% of its misses) '
                 '| %.0f | %.0f%% |'
                 % (q + 1, rng_lo, rng_hi, len(idx),
                    100 * np.nanmean(h_ship[idx]), resc,
                    100 * resc / max(1, sum(1 for i in idx if h_ship[i] != 1)),
                    sup, ging))
    L += ['', '- Q1→Q4 shipped spread: %+.1fpp (8b3ef4f4 measured +19pp — '
          'replication check)'
          % (100 * np.nanmean(h_ship[quart == 3])
             - 100 * np.nanmean(h_ship[quart == 0])), '']

    # T8 gated arms, held-out (threshold + gain fit on train)
    L += ['### T8. Gated vs always-on graph — held-out', '',
          'ALWAYS-ON: one gain for every turn. GATED: fire enrichment only when '
          'cur_maxz ≤ threshold (fit threshold+gain on TRAIN, apply on TEST). '
          'Base gains = the held-out refit per fold.', '',
          '| arm | held-out reach@5 | Δ vs refit base | fires on |',
          '|---|---|---|---|']
    THRESH = [None] + list(np.percentile(cmz, [25, 50, 75, 100]))

    def gated_vec(idxs, thr, g):
        return np.array([g if (thr is None or cmz[i] <= thr) else 0.0
                         for i in idxs])

    h_always = np.full(N, np.nan)
    h_gated = np.full(N, np.nan)
    fired = np.zeros(N)
    chosen = []
    for f in range(FOLDS):
        tr_i = [i for i in range(N) if fold[i] != f]
        te_i = [i for i in range(N) if fold[i] == f]
        tr, te = [P[i] for i in tr_i], [P[i] for i in te_i]
        g5 = fold_gains[f]
        # always-on: best single gain on train
        bg, bs = 0.0, -1.0
        for g in GGRID:
            s, _ = reach_fast(tr, g5, g)
            if s > bs:
                bs, bg = s, g
        ha = hits_fast(te, g5, bg)
        # gated: best (threshold, gain) on train
        bt, bgg, bs2 = None, 0.0, -1.0
        for thr in THRESH:
            for g in GGRID:
                s, _ = reach_fast(tr, g5, gated_vec(tr_i, thr, g))
                if s > bs2:
                    bs2, bt, bgg = s, thr, g
        hg = hits_fast(te, g5, gated_vec(te_i, bt, bgg))
        for k, i in enumerate(te_i):
            h_always[i] = ha[k]
            h_gated[i] = hg[k]
            fired[i] = 1.0 if (bt is None or cmz[i] <= bt) else 0.0
        chosen.append((bt, bgg, bg))
        print('gate fold %d: thr=%s g=%.2f (always %.2f)'
              % (f, 'none' if bt is None else '%.2f' % bt, bgg, bg))
    L += ['| refit base (no enrichment) | %.1f%% | — | — |' % r_cv,
          '| + graph ALWAYS-ON | %.1f%% | %+.1fpp | 100%% |'
          % (100 * np.nanmean(h_always), 100 * np.nanmean(h_always) - r_cv),
          '| + graph GATED | %.1f%% | %+.1fpp | %.0f%% |'
          % (100 * np.nanmean(h_gated), 100 * np.nanmean(h_gated) - r_cv,
             100 * fired.mean()), '']
    ma, sa, loa, hia = boot_delta(h_always, h_refit)
    mg, sg, log_, hig = boot_delta(h_gated, h_refit)
    mgv, sgv, logv, higv = boot_delta(h_gated, h_always)
    L += ['- always-on vs base: %+.2fpp (95%% CI [%+.2f, %+.2f])'
          % (ma, loa, hia),
          '- gated vs base: %+.2fpp (95%% CI [%+.2f, %+.2f])'
          % (mg, log_, hig),
          '- gated vs always-on: %+.2fpp (95%% CI [%+.2f, %+.2f])'
          % (mgv, logv, higv),
          '- per-fold chosen (threshold, gated gain, always gain): %s'
          % '; '.join('(%s, %.2f, %.2f)'
                      % ('none' if a is None else '%.2f' % a, b, c)
                      for a, b, c in chosen), '']

    # T9 conditional lift — the washout test
    bestg, bs = 0.0, -1.0
    for g in GGRID:
        s, _ = reach_fast(P, g_in, g)
        if s > bs:
            bs, bestg = s, g
    # CHARACTERIZATION gain: what enrichment DOES when it fires, independent of
    # whether the optimizer wants it. Without this, a best-gain of 0.00
    # makes T9/T12/T13 compare gain-0 against gain-0 (vacuous tables).
    charg = bestg if bestg > 0 else 0.5
    L += ['### T9. Global vs conditional lift — the washout test', '',
          'Graph reach@5 delta measured WITHIN each cur_maxz quartile. If '
          'value concentrated in LOW quartiles while the global average is '
          '~0, the lane washed out (768e827a pattern) and gating would '
          'recover it.', '',
          '- optimizer\'s best fixed enrichment gain on the tuned base: **%.2f**'
          % bestg,
          '- characterization gain used below: **%.2f**%s' % (charg,
            ' (forced nonzero — the optimizer wanted 0, so a 0-vs-0 table '
            'would say nothing about the lane\'s behaviour)'
            if bestg == 0 else ''), '',
          '| quartile | n | base | +enrichment | Δ |', '|---|---|---|---|---|']
    h_b = hits_fast(P, g_in, 0.0)
    h_g = hits_fast(P, g_in, charg)
    for q in range(4):
        idx = [i for i in range(N) if quart[i] == q]
        L.append('| Q%d | %d | %.1f%% | %.1f%% | %+.1fpp |'
                 % (q + 1, len(idx), 100 * np.nanmean(h_b[idx]),
                    100 * np.nanmean(h_g[idx]),
                    100 * (np.nanmean(h_g[idx]) - np.nanmean(h_b[idx]))))
    L += ['| **all** | %d | %.1f%% | %.1f%% | %+.1fpp |'
          % (N, 100 * np.nanmean(h_b), 100 * np.nanmean(h_g),
             100 * (np.nanmean(h_g) - np.nanmean(h_b))), '']

    # T9b forced-gate arms — characterize gating instead of letting the
    # optimizer collapse to gain 0 (where every threshold ties and the
    # gated-vs-always comparison becomes vacuous).
    L += ['### T9b. Forced gate arms (gain %.2f) — held-out base, gate not fit'
          % charg, '',
          'Fire enrichment only below a cur_maxz threshold. Gate NOT fitted here — '
          'each threshold reported directly, so a tie at gain 0 cannot hide '
          'the comparison.', '',
          '| gate | fires on | reach@5 | Δ vs no-enrichment |', '|---|---|---|---|']
    base_r, _ = reach_fast(P, g_in, 0.0)
    for lbl, thr in ([('always-on', None)]
                     + [('cur_maxz ≤ Q%d' % (k + 1), qs[k])
                        for k in range(3)]):
        vec = np.array([charg if (thr is None or cmz[i] <= thr) else 0.0
                        for i in range(N)])
        r, _ = reach_fast(P, g_in, vec)
        L.append('| %s | %.0f%% | %.1f%% | %+.1fpp |'
                 % (lbl, 100 * np.mean(vec > 0), r, r - base_r))
    L.append('')

    # T12 rank-movement distribution + churn (reach@5 is threshold-blind)
    L += ['### T12. Rank movement — what reach@5 cannot see', '',
          'Gold rank with enrichment off vs on (tuned base, characterization gain '
          '%.2f). reach@5 only counts crossings of 5; this shows the whole '
          'distribution — a lane can move golds a lot and score 0.0pp.'
          % charg, '',
          '| metric | value |', '|---|---|']
    r_off = ranks_fast(P, g_in, 0.0)
    r_on = ranks_fast(P, g_in, charg)
    d = np.array([(b - a) for a, b in zip(r_on, r_off)
                  if a is not None and b is not None], dtype=float)
    gained = sum(1 for a, b in zip(r_on, r_off)
                 if a is not None and b is not None and a <= 5 < b)
    lost = sum(1 for a, b in zip(r_on, r_off)
               if a is not None and b is not None and b <= 5 < a)
    L += ['| turns where enrichment moved the gold at all | %d (%.0f%%) |'
          % (int((d != 0).sum()), 100 * (d != 0).mean()),
          '| median Δrank (improvement, moved turns) | %+.1f |'
          % (np.median(d[d != 0]) if (d != 0).any() else 0.0),
          '| mean Δrank (all turns) | %+.2f |' % d.mean(),
          '| p90 improvement | %+.0f |' % np.percentile(d, 90),
          '| p10 (worst regression) | %+.0f |' % np.percentile(d, 10),
          '| golds GAINED into @5 | %d |' % gained,
          '| golds LOST from @5 | %d |' % lost,
          '| net @5 | %+d |' % (gained - lost), '']

    # T14 rescue anatomy — where enrichment's prize lives
    L += ['### T14. Rescue anatomy — which golds are graph-reachable', '',
          '| slice | rescuable golds | share of slice misses |',
          '|---|---|---|']
    miss_i = [i for i in range(N) if h_ship[i] != 1]
    for key, label in (('stratum', 'stratum'), ('gold_type', 'gold type')):
        vals = Counter(tt[i][key] for i in miss_i if tt[i]['gold_in_enrichment'])
        tot = Counter(tt[i][key] for i in miss_i)
        for v, c in vals.most_common(8):
            L.append('| %s=%s | %d | %.0f%% of %d |'
                     % (label, v, c, 100 * c / tot[v], tot[v]))
    L.append('')
    conv = Counter(tt[i]['gold_seeds'] for i in miss_i if tt[i]['gold_in_enrichment'])
    L += ['- gold convergence among rescuable (how many seeds reached it): %s'
          % ', '.join('%d seeds: %d' % (k, v) for k, v in sorted(conv.items())),
          '']

    # T10/T11 sorting substrate — the measurement laf_lane_audit skipped
    L += ['### T10. SORTING substrate — within-pool ordering (held-out CV)',
          '', 'Does enrichment improve the order Haiku consumes? Pairwise logistic '
          'on candidate lane-z, session-CV. picked = echo-prone; soft-usage = '
          'answer-need (the honest target).', '',
          '| lanes | picked-AUC | soft-AUC |', '|---|---|---|']
    for label, lanes in (('5 lanes (no enrichment)', LANES5),
                         ('6 lanes (+enrichment)', LANES6)):
        pa = cv_auc(turns, lanes, False, 'picked')
        sa = cv_auc(turns, lanes, False, 'soft')
        L.append('| %s | %.4f | %.4f |' % (label, pa, sa))
        print('sorting CV %s done' % label)
    L.append('')

    lowq = [t for t, q in zip(turns, np.digitize(
        np.array([x['cur_maxz'] for x in turns]), qs)) if q == 0]
    L += ['### T11. SORTING inside the fire regime (Q1 cur_maxz only, n=%d)'
          % len(lowq), '',
          'The gated hypothesis: enrichment earns its place where the moment is '
          'reach-starved. Same CV, restricted to Q1.', '',
          '| lanes | picked-AUC | soft-AUC |', '|---|---|---|']
    for label, lanes in (('5 lanes (no enrichment)', LANES5),
                         ('6 lanes (+enrichment)', LANES6)):
        try:
            pa = cv_auc(lowq, lanes, False, 'picked')
            sa = cv_auc(lowq, lanes, False, 'soft')
            L.append('| %s | %.4f | %.4f |' % (label, pa, sa))
        except Exception as e:
            L.append('| %s | n/a (%s) | n/a |' % (label, type(e).__name__))
        print('Q1 sorting CV %s done' % label)
    L.append('')

    # T13 eyeball — the cases, with titles
    idxm = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    master = idxm['master']
    b = open_brain_ro()
    titles = dict(b.execute('SELECT id, title FROM nodes').fetchall())
    b.close()
    L += ['### T13. Eyeball — turns where enrichment moved the gold most', '',
          '| Δrank | rank off→on | stratum | cur_maxz | gold type | seeds | gold title |',
          '|---|---|---|---|---|---|---|']
    mv = sorted(((r_off[i] - r_on[i], i) for i in range(N)
                 if r_off[i] is not None and r_on[i] is not None
                 and r_off[i] != r_on[i]), reverse=True)
    for dlt, i in mv[:8] + mv[-4:]:
        nid = master[tt[i]['gr']]
        L.append('| %+.0f | %.0f→%.0f | %s | %.2f | %s | %d | %s |'
                 % (dlt, r_off[i], r_on[i], tt[i]['stratum'],
                    tt[i]['cur_maxz'], tt[i]['gold_type'] or '?',
                    tt[i]['gold_seeds'],
                    (titles.get(nid) or nid)[:60]))
    L.append('')

    # verdict
    door1 = ('REAL' if lo > 0 else ('NOISE' if hi > 0 else 'HARMFUL'))
    gate_beats = higv > 0 and mgv > 0
    L += ['## Verdict', '',
          '- **Door 1** (fixed-gain retune, held-out): %+.2fpp, 95%% CI '
          '[%+.2f, %+.2f] → **%s**. In-sample optimism %+.1fpp.'
          % (md, lo, hi, door1, r_in - r_cv),
          '- **Door 2 always-on graph**: %+.2fpp (CI [%+.2f, %+.2f])'
          % (ma, loa, hia),
          '- **Door 2 gated graph**: %+.2fpp (CI [%+.2f, %+.2f]); gated vs '
          'always-on %+.2fpp (CI [%+.2f, %+.2f]) → gating %s'
          % (mg, log_, hig, mgv, logv, higv,
             'HELPS' if gate_beats else 'not separable from always-on'), '']

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
