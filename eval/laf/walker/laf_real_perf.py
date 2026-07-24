"""Real performance of every candidate at @5 — the endo target, done properly.

WHY THIS EXISTS. Earlier probes measured candidates against a baseline that
had already absorbed their headroom (non-additivity, cd74b974), in-sample
(+0.8pp optimism), on a single binary threshold. Each of those can make a
real idea look dead. Tom: "before we kill ideas, run them in a way we know
their real performance."

THE POWER PROBLEM (the reason "no effect" was the wrong words). At n=707 the
paired bootstrap sd on a reach delta is ~0.9pp, so the MINIMUM DETECTABLE
EFFECT at 95% is ~1.8pp. Every candidate dismissed so far had a delta BELOW
that. This harness therefore reports MDE, and adds rank-based metrics that
use more information per turn than a binary hit.

TARGET = @5, because endo-recall is algorithmic and takes the top 5 — there
is no Haiku selector to filter noise, so @5 is both the decision metric and
the whole story. @10/@25 gains are irrelevant to it.

DESIGN (each choice fixes one way an idea could be under-measured):
  1. JOINT REFIT PER ARM   -- every arm gets its own gains, fit on TRAIN, so
                              no arm is scored under a rival's tuning
  2. HELD-OUT 5-fold       -- session-grouped; no in-sample optimism
  3. reach@5 (decision) + MRR and mean-delta-rank (sensitivity) -- catches
                              movement that does not cross the threshold
  4. PAIRED BOOTSTRAP CI   -- and MDE, so "no difference" comes with "than what"

Read-only. Run:  ./dev python3 eval/laf/walker/laf_real_perf.py
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from walker_db import OUT_DIR, open_brain_ro

REPO = __file__.rsplit('/eval/', 1)[0]
sys.path.insert(0, REPO)
sys.path.append(str(OUT_DIR))
from servers.recall_laf import zscore_variant                       # noqa: E402
from lambda_probe import zn                                         # noqa: E402
from miss_anatomy import rank_in                                    # noqa: E402
import enrichment_lane as EL                                       # noqa: E402
import laf_lane_audit as A                                         # noqa: E402
import laf_gate_audit as G                                         # noqa: E402
from enrichment_widen import load_communities, community_expand    # noqa: E402

FOLDS = 5
GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
N_BOOT = 4000
SEED = 20260724
AT = 5
REPORT = OUT_DIR / 'laf_real_perf.md'


def epi_fused(t, mode):
    p, e = t['raw_epi']['pick'], t['raw_epi']['enc']
    raw = np.maximum(p, e) if mode == 'max' else (p + e)
    return zscore_variant(raw.astype(np.float64), len(raw),
                          mask=t['alive'], kind='support')


def enrichment_variant(t, adj, node_meta, lanes_mm, S, n, k, of_node=None,
                       members=None, corridor=None, cohesion=None):
    """Enrichment z at seed-count k, optionally + community/corridor members
    (BOOST only, never a filter -- 48fcb7c3/73a98824: 31% of nodes have no
    membership).

    REVIEW FIX: a community-admitted node has no edge to score on (why_cos is
    undefined), so it is scored from real structure — the activating seed's z
    times that community's internal_fraction, "how much of this community's
    story is internal" (77b2617c). Corridors are low-cohesion by construction
    (dbf9146e), so loose-bundle admissions arrive weak, which is the correct
    semantics. Replaces a hand-picked 0.5 constant that left this arm's
    scoring partly fabricated."""
    seeds, sz = EL.seed_rows(lanes_mm, t['row_idx'], S, n, k=k)
    raw, kept = EL.enrichment_activation(seeds, sz, adj, t['qv'],
                                         t['turn_dt'], node_meta, n)
    if of_node is not None:
        touched = {c for s in seeds for c in of_node.get(s, ())
                   if corridor is None or c in corridor}
        for c in touched:
            coh = (cohesion or {}).get(c)
            if coh is None:
                continue                     # no cohesion data → make no claim
            zc = max((sz[s] for s in seeds if c in of_node.get(s, ())),
                     default=None)
            if zc is None:
                continue
            for oi in members[c]:
                if raw[oi] == 0.0:
                    raw[oi] = zc * coh
    return zscore_variant(raw, n, mask=t['alive'], kind='support')


def build_arms(turns, lanes_mm, S, n, adj, node_meta, of_node, members,
               corridor, cohesion):
    """Per-arm per-turn lane matrix (alive-restricted) + zmh + gold pos."""
    print('building arm matrices...')
    variants = {}
    variants['enr_k5'] = [enrichment_variant(t, adj, node_meta, lanes_mm, S, n, 5)
                          for t in turns]
    print('  enr_k5 done')
    variants['enr_k20'] = [enrichment_variant(t, adj, node_meta, lanes_mm, S, n, 20)
                           for t in turns]
    print('  enr_k20 done')
    variants['enr_k20_corr'] = [
        enrichment_variant(t, adj, node_meta, lanes_mm, S, n, 20,
                           of_node, members, corridor, cohesion)
        for t in turns]
    print('  enr_k20_corr done')
    variants['epi_max'] = [epi_fused(t, 'max') for t in turns]
    variants['epi_sum'] = [epi_fused(t, 'sum') for t in turns]
    print('  epi fuses done')

    # ('FIXED', lanes) = score at shipped gains, no refit. A bare list = refit.
    # (Was `(lanes, None)` for BOTH baseline and the refit arm, with
    # fixed=isinstance(spec, tuple) — so the refit arm silently did not refit,
    # returned the baseline exactly, and zeroed the MDE derived from it.)
    ARMS = {
        'baseline (shipped gains, no refit)': ('FIXED', list(A.LANES)),
        'refit 5 lanes': list(A.LANES),
        'drop enc': ['maxsim', 'sit', 'idf', 'pick'],
        'drop sit+enc': ['maxsim', 'idf', 'pick'],
        'epi_max fuse': ['maxsim', 'sit', 'idf', 'epi_max'],
        'epi_sum fuse': ['maxsim', 'sit', 'idf', 'epi_sum'],
        'enrichment K=5': list(A.LANES) + ['enr_k5'],
        'enrichment K=20': list(A.LANES) + ['enr_k20'],
        'enrichment K=20 + corridors': list(A.LANES) + ['enr_k20_corr'],
        'epi_max + enrichment K=20+corr': ['maxsim', 'sit', 'idf', 'epi_max',
                                           'enr_k20_corr'],
    }
    out = {}
    for name, spec in ARMS.items():
        is_fixed = isinstance(spec, tuple) and spec[0] == 'FIXED'
        lanes = spec[1] if is_fixed else spec
        prepped = []
        for i, t in enumerate(turns):
            U = np.flatnonzero(t['alive'])
            cols = []
            for ln in lanes:
                z = (variants[ln][i] if ln in variants else t['zl'][ln])
                cols.append(z[U])
            prepped.append({'Z': np.column_stack(cols).astype(np.float64),
                            'zmh': zn(t['mh'])[U],
                            'g': int(np.searchsorted(U, t['gr'])),
                            'sess': t['sess'], 'ok': U[int(np.searchsorted(
                                U, t['gr']))] == t['gr'] if len(U) else False})
        out[name] = {'lanes': lanes, 'prepped': [p for p in prepped if p['ok']],
                     'fixed': is_fixed}
    # The paired bootstrap indexes arms POSITIONALLY, so every arm must carry
    # the identical turn set in the identical order. True by construction
    # today (the `ok` filter depends only on `alive`/`gr`, both arm-invariant)
    # but asserted because a silent divergence would compare different turns
    # and still print a plausible CI.
    sig = {nm: [p['sess'] for p in a['prepped']] for nm, a in out.items()}
    ref_nm, ref_sig = next(iter(sig.items()))
    for nm, s in sig.items():
        if s != ref_sig:
            raise SystemExit('PAIRING BROKEN: arm %r has a different turn set '
                             'than %r (%d vs %d turns) — the paired bootstrap '
                             'would compare different turns'
                             % (nm, ref_nm, len(s), len(ref_sig)))
    return out


def rank_of(p, gvec):
    f0 = p['Z'] @ gvec
    if f0.size <= 2 or f0.std() <= 1e-9:
        return None
    zf0 = (f0 - f0.mean()) / f0.std()
    mix = A.LAM * zf0 + (1 - A.LAM) * p['zmh']
    gv = mix[p['g']]
    if not np.isfinite(gv):
        return None
    fin = np.where(np.isfinite(mix), mix, -np.inf)
    return int((fin > gv).sum()) + (int((fin == gv).sum()) - 1) / 2.0 + 1


def score_set(prepped, gvec, at=AT):
    """(hits, recip_ranks, ranks) per turn — NaN where unrankable."""
    h, rr, rk = [], [], []
    for p in prepped:
        r = rank_of(p, gvec)
        if r is None:
            h.append(np.nan); rr.append(np.nan); rk.append(np.nan)
        else:
            h.append(float(r <= at)); rr.append(1.0 / r); rk.append(r)
    return np.array(h), np.array(rr), np.array(rk)


def refit(prepped, lanes, passes=3):
    """Joint coordinate ascent on reach@5, multi-start."""
    inits = [np.array([A.GAINS.get(ln, 0.5) for ln in lanes], dtype=float),
             np.full(len(lanes), 0.5),
             np.array([1.5 if ln == 'maxsim' else 0.25 for ln in lanes])]
    best, bs = None, -1.0
    for g0 in inits:
        g = g0.copy()
        for _ in range(passes):
            for j in range(len(lanes)):
                bv, bsc = g[j], -1.0
                for c in GRID:
                    cand = g.copy(); cand[j] = c
                    hh, _, _ = score_set(prepped, cand)
                    s = np.nanmean(hh)
                    if s > bsc:
                        bsc, bv = s, c
                g[j] = bv
        hh, _, _ = score_set(prepped, g)
        s = np.nanmean(hh)
        if s > bs:
            bs, best = s, g.copy()
    return best


def boot(a, b, n_boot=N_BOOT, seed=SEED, scale=100.0):
    """Paired turn-level bootstrap of (a − b). scale=100 for rate metrics
    reported in percentage points; scale=1 for metrics already in their own
    units (MRR lives in [0,1] — scaling it by 100 inflated ΔMRR 100×)."""
    m = np.isfinite(a) & np.isfinite(b)
    x, y = a[m], b[m]
    rng = np.random.default_rng(seed)
    d = np.array([scale * (x[i].mean() - y[i].mean())
                  for i in (rng.integers(0, len(x), len(x))
                            for _ in range(n_boot))])
    return float(d.mean()), float(d.std()), float(np.percentile(d, 2.5)), \
        float(np.percentile(d, 97.5))


def main():
    turns, n = A.build()
    # re-open the graph + community structure (A.build does not return them)
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    lanes_mm = np.load(OUT_DIR / 'lane_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    m2i = {sid: i for i, sid in enumerate(idx['master'])}
    row_of = {tuple(t['key']): t['row'] for t in idx['turns']}
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}
    from walker_db import WALKER_DB, open_ro
    w = open_ro(WALKER_DB)
    qvecs = EL.build_qvecs(w)
    w.close()
    b = open_brain_ro()
    node_meta = EL.build_node_meta(b, m2i)
    adj = EL.build_adjacency(b, m2i)
    of_node, members, corridor, cohesion = load_communities(b, m2i)
    b.close()
    for t in turns:
        # t['key'] is the 'sess/epoch/seq' string laf_lane_audit builds; the
        # row/qvec maps are keyed by the (sess, int, int) tuple.
        parts = t['key'].split('/') if isinstance(t['key'], str) else None
        kk = (parts[0], int(parts[1]), int(parts[2])) if parts else t['key']
        t['row_idx'] = row_of[kk]
        t['qv'] = qvecs.get(kk)
        bd = bundles.get(t['key'])
        t['turn_dt'] = EL.iso(bd['ts']) if bd else None

    arms = build_arms(turns, lanes_mm, S, n, adj, node_meta, of_node,
                      members, corridor, cohesion)

    # session-grouped folds (shared across arms — same turns, same split)
    ref = arms['baseline (shipped gains, no refit)']['prepped']
    sess = sorted({p['sess'] for p in ref})
    fold_of = {s: i % FOLDS for i, s in enumerate(sess)}

    results = {}
    for name, arm in arms.items():
        pr, lanes = arm['prepped'], arm['lanes']
        fold = np.array([fold_of[p['sess']] for p in pr])
        H = np.full(len(pr), np.nan); R = np.full(len(pr), np.nan)
        K = np.full(len(pr), np.nan)
        if arm['fixed']:
            g = np.array([A.GAINS[ln] for ln in lanes], dtype=float)
            H, R, K = score_set(pr, g)
            gains_used = [g]
        else:
            gains_used = []
            for f in range(FOLDS):
                tr = [pr[i] for i in range(len(pr)) if fold[i] != f]
                te_i = [i for i in range(len(pr)) if fold[i] == f]
                g = refit(tr, lanes)
                gains_used.append(g)
                hh, rr, kk2 = score_set([pr[i] for i in te_i], g)
                for j, i in enumerate(te_i):
                    H[i], R[i], K[i] = hh[j], rr[j], kk2[j]
        results[name] = {'H': H, 'R': R, 'K': K, 'lanes': lanes,
                         'gains': gains_used}
        print('%-34s reach@5 %.1f%%  MRR %.4f'
              % (name, 100 * np.nanmean(H), np.nanmean(R)))

    base = results['baseline (shipped gains, no refit)']
    # MDE from the MEDIAN paired-delta sd across all real arms — not from one
    # arm (a degenerate arm would zero it, which is exactly what happened when
    # the refit arm silently returned the baseline).
    sds = [boot(r['H'], base['H'])[1] for nm, r in results.items()
           if not nm.startswith('baseline')]
    ref_sd = float(np.median(sds))
    mde = 1.96 * ref_sd

    L = ['# Real performance of every candidate at @5 (endo target)', '',
         'n=%d clean valids ≥%s · held-out session-grouped %d-fold · EVERY arm '
         'jointly refit on TRAIN (so no arm is scored under a rival\'s tuning) '
         '· paired bootstrap ×%d'
         % (len(base['H']), A.CUTOFF, FOLDS, N_BOOT), '',
         '**Power.** paired-delta sd ≈ %.2fpp → **MDE(95%%) ≈ %.1fpp**. A null '
         'here means "no effect ≥ %.1fpp", NOT "no effect". Every candidate '
         'dismissed earlier today had a delta below this bar.'
         % (ref_sd, mde, mde), '',
         '**Reading it.** reach@5 is the DECISION metric (endo takes top-5). '
         'MRR and mean Δrank are SENSITIVITY metrics — they use the whole rank, '
         'so they see movement that never crosses 5. An arm that moves MRR but '
         'not reach@5 is a real component, just not shippable alone.', '',
         '**Two baselines, because one column cannot separate the two effects.** '
         '"Δ vs shipped" mixes *this lane set is better* with *refitting at all '
         'helps*. "Δ vs refit-5" holds refitting constant and isolates the '
         'lane-set change — that is the comparison that says whether a lane '
         'earns its place.', '',
         '**Multiplicity.** %d arms are compared here, each with a ~±%.1fpp CI. '
         'Picking the largest and quoting its own CI overstates it (winner\'s '
         'curse); a best-of-%d needs roughly a %.1fpp effect to mean what a '
         'single %.1fpp CI would.'
         % (len(results) - 1, mde, len(results) - 1, mde * 1.4, mde), '',
         '| arm | reach@5 | Δ vs shipped | 95% CI | Δ vs refit-5 | MRR | Δ MRR (95% CI) | verdict |',
         '|---|---|---|---|---|---|---|---|']
    ref5 = results.get('refit 5 lanes')
    for name, r in results.items():
        if name.startswith('baseline'):
            L.append('| %s | %.1f%% | — | — | — | %.4f | — | reference |'
                     % (name, 100 * np.nanmean(r['H']), np.nanmean(r['R'])))
            continue
        md, sd, lo, hi = boot(r['H'], base['H'])
        rmd, rsd, rlo, rhi = boot(r['R'], base['R'], scale=1.0)
        if ref5 is not None and name != 'refit 5 lanes':
            m5, _, l5, h5 = boot(r['H'], ref5['H'])
            vs5 = '%+.2f [%+.2f, %+.2f]' % (m5, l5, h5)
        else:
            vs5 = '—'
        if lo > 0:
            v = '**REAL @5**'
        elif hi < 0:
            v = 'HARMFUL @5'
        elif rlo > 0:
            v = 'sub-threshold signal'
        elif rhi < 0:
            v = 'rank-harmful'
        else:
            v = 'not measurable (<MDE)'
        L.append('| %s | %.1f%% | %+.2fpp | [%+.2f, %+.2f] | %s | %.4f | %+.4f '
                 '[%+.4f, %+.4f] | %s |'
                 % (name, 100 * np.nanmean(r['H']), md, lo, hi, vs5,
                    np.nanmean(r['R']), rmd, rlo, rhi, v))
    L.append('')

    # mean Δrank (most sensitive paired statistic)
    L += ['## Mean Δrank vs baseline (negative = gold ranked better)', '',
          '| arm | mean Δrank | median Δrank | turns improved |',
          '|---|---|---|---|']
    for name, r in results.items():
        if name.startswith('baseline'):
            continue
        m = np.isfinite(r['K']) & np.isfinite(base['K'])
        d = r['K'][m] - base['K'][m]
        L.append('| %s | %+.2f | %+.1f | %.0f%% |'
                 % (name, d.mean(), np.median(d), 100 * np.mean(d < 0)))
    L.append('')

    # the gains each arm actually learned (stability across folds)
    L += ['## Learned gains per arm (fold mean ± sd)', '',
          'Fold sd is the generalization tell: a gain that swings across folds '
          'is fitting the fold, not the signal.', '',
          '| arm | learned gains |', '|---|---|']
    for name, r in results.items():
        Gs = np.array(r['gains'])
        if Gs.ndim != 2:
            continue
        txt = ' · '.join('%s %.2f±%.2f' % (ln, Gs[:, j].mean(), Gs[:, j].std())
                         for j, ln in enumerate(r['lanes']))
        L.append('| %s | %s |' % (name, txt))
    L.append('')

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
