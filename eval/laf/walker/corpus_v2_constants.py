"""Is it the CONSTANTS? — config comparison + composition help/hurt ledger.

Tom's question: could the recall shortfall be the fixed factors we use
(the per-lane GAINS, the F0/M_h blend λ=0.65)? Test empirically on the clean
corpus (valid golds, turn-date ≥ CUTOFF):

  A. reach@5 under single components vs the composite — if the composed
     field does NOT beat its own maxsim lane, the extra lanes/gains are
     net-neutral-or-harmful (a constants problem).
  B. composition help/hurt ledger — turns where maxsim-alone ranks the gold
     ≤5 but the λ-mix pushes it out (HURT) vs the reverse (RESCUED). Net
     sign says whether the fixed composite earns its place.
  C. oracle-λ uplift — best per-turn F0/M_h blend vs the fixed 0.65. The gap
     is the ceiling of 'just fix the blend constant' — and if it's large,
     the lesson is NO single constant works (→ per-message gain, not a retune).

Cache-only. Machinery imported (Turn, lane_z, rank_in, zn, lambda_star).
Run: ./dev python3 eval/laf/walker/corpus_v2_constants.py [cutoff]
"""
import json
import sys

import numpy as np

from walker_db import OUT_DIR

sys.path.insert(0, str(OUT_DIR))
from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn, lambda_star                            # noqa: E402
from layer_readout_probe import lane_z                              # noqa: E402
from miss_anatomy import rank_in                                    # noqa: E402

CUTOFF = sys.argv[1] if len(sys.argv) > 1 else '2026-05-11'
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
LANES = ('maxsim', 'sit', 'idf', 'pick', 'enc')


def main():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    fields = np.load(OUT_DIR / 'field_cache.npy', mmap_mode='r')
    lanes_mm = np.load(OUT_DIR / 'lane_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    n_nodes = idx['n_nodes']
    verds = {json.loads(x)['key']: json.loads(x)
             for x in (OUT_DIR / 'corpus_v2_verdicts.jsonl').open()}
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}

    rows = []
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
        mxz = lane_z(L[S['op0'], LANES.index('maxsim')], 'maxsim', alive, n_nodes)
        r_mxonly = rank_in(mxz, gr)
        r_f0 = rank_in(tt.fields[0], gr)
        r_mh = rank_in(tt.mh, gr)
        rk_fixed = lambda_star(zn(tt.fields[0]), zn(tt.mh), gr,
                               grid=np.array([0.65]))
        r_mix = min(rk_fixed.values()) if rk_fixed else None
        rk_all = lambda_star(zn(tt.fields[0]), zn(tt.mh), gr, grid=GRID)
        r_orac = min(rk_all.values()) if rk_all else None
        rows.append({'door': 'door-1' if v['stratum'] == 'cue' else 'door-2',
                     'mxonly': r_mxonly, 'f0': r_f0, 'mh': r_mh,
                     'mix': r_mix, 'orac': r_orac})

    def reach(rs, fld, k=5):
        vals = [r[fld] for r in rs if r[fld] is not None]
        return 100.0 * sum(x <= k for x in vals) / len(vals) if vals else 0.0

    print('# Is it the constants? — clean corpus, valid golds ≥%s, n=%d\n'
          % (CUTOFF, len(rows)))
    print('## A. reach@5 by component (single ranking vs composite)\n')
    print('| slice | n | maxsim-only | F0 (composed) | M_h | λ-mix (0.65) | oracle-λ |')
    print('|---|---|---|---|---|---|---|')
    for lbl, sub in (('ALL', rows),
                     ('door-1', [r for r in rows if r['door'] == 'door-1']),
                     ('door-2', [r for r in rows if r['door'] == 'door-2'])):
        print('| %s | %d | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% |'
              % (lbl, len(sub), reach(sub, 'mxonly'), reach(sub, 'f0'),
                 reach(sub, 'mh'), reach(sub, 'mix'), reach(sub, 'orac')))

    print('\n## B. composition help/hurt ledger (maxsim-only vs λ-mix @5)\n')
    print('| slice | HURT (mx≤5, mix>5) | RESCUED (mx>5, mix≤5) | net |')
    print('|---|---|---|---|')
    for lbl, sub in (('ALL', rows),
                     ('door-1', [r for r in rows if r['door'] == 'door-1']),
                     ('door-2', [r for r in rows if r['door'] == 'door-2'])):
        hurt = sum(1 for r in sub if r['mxonly'] and r['mix']
                   and r['mxonly'] <= 5 < r['mix'])
        resc = sum(1 for r in sub if r['mxonly'] and r['mix']
                   and r['mix'] <= 5 < r['mxonly'])
        print('| %s | %d | %d | %+d |' % (lbl, hurt, resc, resc - hurt))

    print('\n## C. oracle-λ uplift over fixed 0.65 (ceiling of just retuning the blend)\n')
    for lbl, sub in (('ALL', rows),
                     ('door-1', [r for r in rows if r['door'] == 'door-1']),
                     ('door-2', [r for r in rows if r['door'] == 'door-2'])):
        print('- %s: fixed %.0f%% → oracle %.0f%%  (+%.0fpp per-turn headroom)'
              % (lbl, reach(sub, 'mix'), reach(sub, 'orac'),
                 reach(sub, 'orac') - reach(sub, 'mix')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
