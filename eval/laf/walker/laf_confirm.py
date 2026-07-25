"""CONFIRMATORY test of the one arm that cleared @5 — and its attribution.

The 10-arm run (laf_real_perf, e34feda) was EXPLORATORY: it found
`enrichment K=20 + corridors` at +2.55pp [+0.71, +4.38], but a best-of-9
carries a winner's curse that inflates the bar to ~2.6pp — right where the
result sits. The legitimate escape is not a bigger CI; it is a FOCUSED
CONFIRMATORY run, pre-registered, with few comparisons.

PRE-REGISTERED BEFORE RUNNING:
  ARMS (3 only, so the multiplicity bar stays ~1.9pp, not ~2.6pp):
    A  shipped gains (reference)
    B  refit 5 lanes                      -- retuning alone
    C  refit 5 lanes + enrichment K=20+corridors  -- the candidate
  PASS for the candidate: C vs A CI excludes 0 in BOTH corpora AND holds
    across every fold seed. A single-seed win is a fold-assignment artifact.
  ATTRIBUTION (the question that decides production cost): C vs B. If that
    CI excludes 0 the LANE earns its keep (new code). If not, retuning gets
    most of it and the ship is a K-store value change with no new lane.
  ROBUSTNESS: 3 session->fold permutations x 2 corpora.
    quality corpus  turn-date >= 2026-05-11 (n~707)   -- PRIMARY
    wide corpus     all valid golds (n~1000)          -- SENSITIVITY ONLY,
      never pooled with the primary: pre-cutoff golds are a different era
      (strong-tier composition varies by month), so it tests robustness of
      direction, not a bigger sample for the headline.
  PER STRATUM: reported for every arm. 449fb9a7 is explicit that a blended
    walker reach number hides the populations; cue / window / session must be
    visible, especially because a CONCENTRATED effect can clear a 1.9pp floor
    that a diffuse one never will.

Read-only. Run:  ./dev python3 eval/laf/walker/laf_confirm.py
"""
import sys

import numpy as np

from walker_db import OUT_DIR, open_brain_ro

REPO = __file__.rsplit('/eval/', 1)[0]
sys.path.insert(0, REPO)
sys.path.append(str(OUT_DIR))
from lambda_probe import zn                                         # noqa: E402
import enrichment_lane as EL                                       # noqa: E402
import laf_lane_audit as A                                         # noqa: E402
import laf_real_perf as RP                                         # noqa: E402
from enrichment_widen import load_communities                      # noqa: E402

FOLD_SEEDS = (0, 1, 2)
FOLDS = 5
PASSES = 2
REPORT = OUT_DIR / 'laf_confirm.md'


def refit_light(prepped, lanes, passes=PASSES):
    """Coordinate ascent, 2 inits x 2 passes (the confirmatory run trades a
    little fit quality for running 3 arms x 3 seeds x 2 corpora)."""
    inits = [np.array([A.GAINS.get(ln, 0.5) for ln in lanes], dtype=float),
             np.array([1.25 if ln == 'maxsim' else 0.4 for ln in lanes])]
    best, bs = None, -1.0
    for g0 in inits:
        g = g0.copy()
        for _ in range(passes):
            for j in range(len(lanes)):
                bv, bsc = g[j], -1.0
                for c in RP.GRID:
                    cand = g.copy(); cand[j] = c
                    s = np.nanmean(RP.score_set(prepped, cand)[0])
                    if s > bsc:
                        bsc, bv = s, c
                g[j] = bv
        s = np.nanmean(RP.score_set(prepped, g)[0])
        if s > bs:
            bs, best = s, g.copy()
    return best


def run_corpus(label, cutoff):
    """Build one corpus and evaluate the 3 arms over all fold seeds."""
    old = A.CUTOFF
    A.CUTOFF = cutoff                     # build() reads the module global
    try:
        turns, n = A.build()
    finally:
        A.CUTOFF = old
    import json
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
        parts = t['key'].split('/')
        kk = (parts[0], int(parts[1]), int(parts[2]))
        t['row_idx'] = row_of[kk]
        t['qv'] = qvecs.get(kk)
        bd = bundles.get(t['key'])
        t['turn_dt'] = EL.iso(bd['ts']) if bd else None

    enr = [RP.enrichment_variant(t, adj, node_meta, lanes_mm, S, n, 20,
                                 of_node, members, corridor, cohesion)
           for t in turns]
    print('  %s: enrichment variant built' % label)

    ARMS = {'A shipped': (list(A.LANES), True),
            'B refit 5': (list(A.LANES), False),
            'C refit 5 + enrichment K=20+corr': (list(A.LANES) + ['ENR'], False)}
    prep = {}
    for name, (lanes, fixed) in ARMS.items():
        rows = []
        for i, t in enumerate(turns):
            U = np.flatnonzero(t['alive'])
            cols = [(enr[i] if ln == 'ENR' else t['zl'][ln])[U] for ln in lanes]
            gpos = int(np.searchsorted(U, t['gr']))
            if gpos >= len(U) or U[gpos] != t['gr']:
                continue
            rows.append({'Z': np.column_stack(cols).astype(np.float64),
                         'zmh': zn(t['mh'])[U], 'g': gpos,
                         'sess': t['sess'], 'stratum': t['stratum']})
        prep[name] = (rows, lanes, fixed)
    lens = {k: len(v[0]) for k, v in prep.items()}
    if len(set(lens.values())) != 1:
        raise SystemExit('PAIRING BROKEN across arms: %s' % lens)

    out = {}
    for seed in FOLD_SEEDS:
        rng = np.random.default_rng(seed)
        sess = sorted({p['sess'] for p in prep['A shipped'][0]})
        perm = rng.permutation(len(sess))
        fold_of = {sess[perm[i]]: i % FOLDS for i in range(len(sess))}
        for name, (rows, lanes, fixed) in prep.items():
            fold = np.array([fold_of[p['sess']] for p in rows])
            H = np.full(len(rows), np.nan)
            if fixed:
                H = RP.score_set(rows, np.array([A.GAINS[ln] for ln in lanes]))[0]
            else:
                for f in range(FOLDS):
                    tr = [rows[i] for i in range(len(rows)) if fold[i] != f]
                    te_i = [i for i in range(len(rows)) if fold[i] == f]
                    g = refit_light(tr, lanes)
                    hh = RP.score_set([rows[i] for i in te_i], g)[0]
                    for j, i in enumerate(te_i):
                        H[i] = hh[j]
            out[(seed, name)] = H
            print('    seed %d %-34s reach@5 %.1f%%'
                  % (seed, name, 100 * np.nanmean(H)))
    strata = [p['stratum'] for p in prep['A shipped'][0]]
    return out, strata, lens['A shipped']


def main():
    L = ['# Confirmatory test — does the winning arm replicate?', '',
         'The 10-arm run was EXPLORATORY (best-of-9 → ~2.6pp bar). This is a '
         'PRE-REGISTERED CONFIRMATORY run: 3 arms only, so the bar returns to '
         '~1.9pp; 3 session→fold permutations; 2 corpora; per stratum.', '',
         '**Pass criterion (pre-declared):** C vs A must exclude 0 in BOTH '
         'corpora AND across every fold seed. **Attribution:** C vs B decides '
         'whether the LANE earns new production code or whether retuning '
         'alone (a K-store value change) captures it.', '']

    for label, cutoff in (('quality (≥2026-05-11) — PRIMARY', '2026-05-11'),
                          ('wide (all valid golds) — SENSITIVITY', '0000')):
        print('=== %s ===' % label)
        out, strata, n = run_corpus(label, cutoff)
        L += ['## %s · n=%d' % (label, n), '',
              '| fold seed | A shipped | B refit 5 | C +enrichment | C−A (95% CI) | C−B (95% CI) |',
              '|---|---|---|---|---|---|']
        ca_ok, cb_ok = [], []
        for seed in FOLD_SEEDS:
            HA = out[(seed, 'A shipped')]
            HB = out[(seed, 'B refit 5')]
            HC = out[(seed, 'C refit 5 + enrichment K=20+corr')]
            ma, _, la, ha = RP.boot(HC, HA)
            mb, _, lb, hb = RP.boot(HC, HB)
            ca_ok.append(la > 0)
            cb_ok.append(lb > 0)
            L.append('| %d | %.1f%% | %.1f%% | %.1f%% | %+.2f [%+.2f, %+.2f] | '
                     '%+.2f [%+.2f, %+.2f] |'
                     % (seed, 100 * np.nanmean(HA), 100 * np.nanmean(HB),
                        100 * np.nanmean(HC), ma, la, ha, mb, lb, hb))
        L += ['', '- **C vs A excludes 0 in %d/%d fold seeds** · C vs B in %d/%d'
              % (sum(ca_ok), len(ca_ok), sum(cb_ok), len(cb_ok)), '']

        # per stratum, pooled over fold seeds (449fb9a7: never quote the blend)
        L += ['### Per stratum (fold seeds pooled)', '',
              '| stratum | n | A shipped | C +enrichment | Δ |',
              '|---|---|---|---|---|']
        st = np.array(strata)
        for s in ('cue', 'window', 'session'):
            m = st == s
            if not m.any():
                continue
            a = np.nanmean([np.nanmean(out[(sd, 'A shipped')][m])
                            for sd in FOLD_SEEDS])
            c = np.nanmean([np.nanmean(
                out[(sd, 'C refit 5 + enrichment K=20+corr')][m])
                for sd in FOLD_SEEDS])
            L.append('| %s | %d | %.1f%% | %.1f%% | %+.1fpp |'
                     % (s, int(m.sum()), 100 * a, 100 * c, 100 * (c - a)))
        L.append('')

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
