"""Role-expansion arms — does completing the enc role set revive the lane?

The enc lane carries created∪revised only and carries 0 door-1 hits (see
corpus_v2_eval.md). The role-expansion hypothesis (Tom, 2026-07-29): the lane
is dead BECAUSE it holds the wrong half of the encode delta — nodes CREATED
at a past moment didn't exist before it, while the nodes the encoder
CONNECTED TO (and the ones Anchor hand-wrote) are the pre-existing resonant
population door-1 golds come from. Substrate: roles_lane_cache.npy
(conn/auth, validated backfill — role_backfill_audit.py) + lane_cache.npy.

ARMS (all refit arms get gains AND λ fit per door, laf_doors protocol):
  A shipped     — production lanes, shipped gains, λ=0.65 (reproduction gate)
  B refit-5     — production lanes, refit (THE CONTROL: any win must beat
                  this, not A — refitting alone was in-sample +2pp once)
  C enc∪conn    — enc fused with conn (raw max, THEN support-z — fusing after
                  z would just average two z-inflated lanes)
  D epi_max     — pick∪enc∪conn as ONE episodic lane (the composition that
                  survived held-out before: 48b69e1c)
  E epi_max+auth — D plus auth as its own support-z lane (auth is too thin
                  for its own arm: 265 trace-era ids)

Pass rule (pre-registered): vs B, CI excludes 0 in every seed, on door-1.
Door-2 recorded, not blended. 5 session→fold permutations · 2 corpora ·
paired bootstrap. Echo note: conn/auth are NOT derived from Haiku picks
(edges + anchor nodes), but the collinearity ledger (36.5% same-window
picked) travels with any win.

Run:  ./dev python3 eval/laf/walker/role_arms.py        (cache-only)
"""
import json
import sys

import numpy as np

from walker_db import OUT_DIR

sys.path.append(str(OUT_DIR))
import laf_doors as D                                               # noqa: E402
import laf_lane_audit as A                                          # noqa: E402
import laf_real_perf as RP                                          # noqa: E402

REPO = __file__.rsplit('/eval/', 1)[0]
sys.path.insert(0, REPO)
from servers.recall_laf import zscore_variant                       # noqa: E402

REPORT = OUT_DIR / 'role_arms.md'
KS = (5, 10, 25)

ARMS = {
    'A shipped': (list(A.LANES), True),
    'B refit-5': (list(A.LANES), False),
    'C enc+conn': (['maxsim', 'sit', 'idf', 'pick', 'enc_conn'], False),
    'D epi_max': (['maxsim', 'sit', 'idf', 'epi'], False),
    'E epi+auth': (['maxsim', 'sit', 'idf', 'epi', 'auth'], False),
}
PAIRS = (('B refit-5', 'A shipped'), ('C enc+conn', 'B refit-5'),
         ('D epi_max', 'B refit-5'), ('E epi+auth', 'B refit-5'),
         ('E epi+auth', 'D epi_max'))


def inject_role_lanes(turns):
    """Add enc_conn / epi / auth / conn z-lanes onto each turn's zl dict.

    Raw fusion BEFORE support-z (laf_lane_audit's epi rule); NaN → 0 in the
    raw episodic lanes (0 = no activation, the lanes' native zero-sea)."""
    ridx = json.loads((OUT_DIR / 'roles_lane_index.json').read_text())
    fidx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    if ridx['master_hash'] != fidx['master_hash']:
        raise SystemExit('roles_lane_cache master_hash != field_cache — '
                         'rebuild roles_lane_cache.py')
    roles = np.load(OUT_DIR / 'roles_lane_cache.npy', mmap_mode='r')
    lanes_mm = np.load(OUT_DIR / 'lane_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(fidx['slots'])}
    n = fidx['n_nodes']
    i_pick, i_enc = A.LANES.index('pick'), A.LANES.index('enc')
    for t in turns:
        row = t['row_idx']
        Lr = lanes_mm[row]
        raw = {}
        for nm, li in (('pick', i_pick), ('enc', i_enc)):
            x = Lr[S['op0'], li].astype(np.float64)
            raw[nm] = np.where(np.isfinite(x), x, 0.0)
        conn = roles[row, 0].astype(np.float64)
        auth = roles[row, 1].astype(np.float64)
        alive = t['alive']
        z = lambda x: zscore_variant(x, n, mask=alive, kind='support')
        t['zl']['conn'] = z(conn)
        t['zl']['auth'] = z(auth)
        t['zl']['enc_conn'] = z(np.maximum(raw['enc'], conn))
        t['zl']['epi'] = z(np.maximum(np.maximum(raw['pick'], raw['enc']),
                                      conn))
    return turns


def eval_door(turns, enr, strata):
    """laf_doors.eval_door with this module's ARMS (theirs is module-global)."""
    prep = {nm: (D.prep_rows(turns, enr, strata, lanes), lanes, fixed)
            for nm, (lanes, fixed) in ARMS.items()}
    lens = {k: len(v[0]) for k, v in prep.items()}
    if len(set(lens.values())) != 1:
        raise SystemExit('PAIRING BROKEN: %s' % lens)
    out, lams = {}, {}
    ref_key = next(iter(prep))
    for seed in D.FOLD_SEEDS:
        rng = np.random.default_rng(seed)
        sess = sorted({p['sess'] for p in prep[ref_key][0]})
        perm = rng.permutation(len(sess))
        fold_of = {sess[perm[i]]: i % D.FOLDS for i in range(len(sess))}
        for nm, (rows, lanes, fixed) in prep.items():
            fold = np.array([fold_of[p['sess']] for p in rows])
            H = {k: np.full(len(rows), np.nan) for k in KS}
            if fixed:
                gv = np.array([A.GAINS[l] for l in lanes])
                for k in KS:
                    H[k] = D.hits_lam(rows, gv, 0.65, at=k)
            else:
                for f in range(D.FOLDS):
                    tr = [rows[i] for i in range(len(rows)) if fold[i] != f]
                    te = [i for i in range(len(rows)) if fold[i] == f]
                    g, lam = D.refit_gains_lam(tr, lanes)   # fit at @5
                    lams.setdefault(nm, []).append(lam)
                    for k in KS:
                        hh = D.hits_lam([rows[i] for i in te], g, lam, at=k)
                        for j, i in enumerate(te):
                            H[k][i] = hh[j]
            for k in KS:
                out[(seed, nm, k)] = H[k]
    return out, lens[ref_key], lams


def main():
    L = ['# Role-expansion arms — conn/auth into the episodic lane', '',
         'Control is **B refit-5** (any win must beat refitting alone). '
         'Door-1 is the exam; door-2 recorded. 5 fold-permutations, '
         'paired bootstrap, pass = CI>0 across all seeds.', '']
    for clabel, cutoff in (('quality (≥2026-05-11)', '2026-05-11'),
                           ('wide (all valid golds)', '0000')):
        print('=== corpus %s ===' % clabel)
        turns, enr, n = D.build_corpus(cutoff)
        inject_role_lanes(turns)
        for dlabel, strata in D.DOORS:
            out, dn, lams = eval_door(turns, enr, strata)
            L += ['## %s · %s · n=%d' % (clabel, dlabel, dn), '']
            for k in KS:
                L += ['### reach@%d%s' % (k, '' if k == 5 else
                                          ' (gains fit at @5)'), '',
                      '| seed | ' + ' | '.join(ARMS) + ' | ' +
                      ' | '.join('%s−%s' % (a.split()[0], b.split()[0])
                                 for a, b in PAIRS) + ' |',
                      '|' + '---|' * (1 + len(ARMS) + len(PAIRS))]
                wins = {p: [] for p in PAIRS}
                for seed in D.FOLD_SEEDS:
                    cells = ['%.1f%%' % (100 * np.nanmean(out[(seed, nm, k)]))
                             for nm in ARMS]
                    diffs = []
                    for a, b in PAIRS:
                        m, _, lo, hi = RP.boot(out[(seed, a, k)],
                                               out[(seed, b, k)])
                        wins[(a, b)].append(lo > 0)
                        diffs.append('%+.2f [%+.2f, %+.2f]' % (m, lo, hi))
                    L.append('| %d | %s | %s |' % (seed, ' | '.join(cells),
                                                   ' | '.join(diffs)))
                L += ['', '- CI>0 seeds: ' +
                      ' · '.join('%s−%s **%d/%d**'
                                 % (a.split()[0], b.split()[0], sum(w),
                                    len(w))
                                 for (a, b), w in wins.items()), '']
                print('  %s n=%d @%d: %s' % (
                    dlabel.split(' — ')[0], dn, k,
                    ', '.join('%s-%s %d/5' % (a.split()[0], b.split()[0],
                                              sum(w))
                              for (a, b), w in wins.items())))
            L += ['- fitted λ: ' + ' · '.join(
                      '%s %s' % (nm.split()[0],
                                 '/'.join('%.2f' % x for x in sorted(set(v))))
                      for nm, v in lams.items()), '']
    REPORT.write_text('\n'.join(L) + '\n')
    print('wrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
