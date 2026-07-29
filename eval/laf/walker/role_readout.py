"""Role readout — did the new roles TOUCH the golds at all?

The role-arms null has two possible anatomies, and this splits them:
  (a) DATA GAP    — golds never receive conn/auth activation (the encoder's
                    connect-targets at similar past moments just aren't the
                    golds) → no lane shape could have worked;
  (b) BURIED      — golds DO receive activation but the composed mix doesn't
                    lift them into the top-K (the pipeline-burial pattern,
                    c02f37bb) → composition/mechanism problem, data is live.

Per door-1/door-2 turn, under the SHIPPED mix (production gains, λ=0.65):
  - shipped tie-fair rank of the gold → hit@5 / @10 / @25 / beyond buckets
  - gold's raw conn / auth / pick / enc activation (>0 = the role reached it)
  - for conn>0 golds: within-lane rank (would conn ALONE put it top-5? the
    lane-oracle read) and the lane's support (crowding)

Cache-only, no fitting. Run:  ./dev python3 eval/laf/walker/role_readout.py
"""
import json
import sys

import numpy as np

from walker_db import OUT_DIR

sys.path.append(str(OUT_DIR))
import laf_doors as D                                               # noqa: E402
import laf_lane_audit as A                                          # noqa: E402
from lambda_probe import zn                                         # noqa: E402
from role_arms import inject_role_lanes                             # noqa: E402

REPORT = OUT_DIR / 'role_readout.md'


def main():
    fidx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    S = {s: i for i, s in enumerate(fidx['slots'])}
    roles = np.load(OUT_DIR / 'roles_lane_cache.npy', mmap_mode='r')
    lanes_mm = np.load(OUT_DIR / 'lane_cache.npy', mmap_mode='r')
    i_pick, i_enc = A.LANES.index('pick'), A.LANES.index('enc')
    gains = np.array([A.GAINS[l] for l in A.LANES])

    L = ['# Role readout — do conn/auth reach the golds?', '']
    for clabel, cutoff in (('quality (≥2026-05-11)', '2026-05-11'),
                           ('wide (all valid golds)', '0000')):
        turns, enr, n = D.build_corpus(cutoff)
        inject_role_lanes(turns)
        for dlabel, strata in D.DOORS:
            rows = []
            for t in turns:
                if t['stratum'] not in strata:
                    continue
                U = np.flatnonzero(t['alive'])
                gpos = int(np.searchsorted(U, t['gr']))
                if gpos >= len(U) or U[gpos] != t['gr']:
                    continue
                p = {'Z': np.column_stack(
                        [t['zl'][ln][U] for ln in A.LANES]).astype(np.float64),
                     'zmh': zn(t['mh'])[U], 'g': gpos}
                r = D.rank_lam(p, gains, 0.65)
                if r is None:
                    continue
                row = t['row_idx']
                gr = t['gr']
                conn_v = float(roles[row, 0, gr])
                auth_v = float(roles[row, 1, gr])
                Lr = lanes_mm[row]
                pk = Lr[S['op0'], i_pick, gr]
                en = Lr[S['op0'], i_enc, gr]
                conn_lane = np.asarray(roles[row, 0], dtype=np.float64)[U]
                support = int((conn_lane > 0).sum())
                lane_rank = (int((conn_lane > conn_v).sum()) + 1
                             if conn_v > 0 else None)
                rows.append({'rank': r, 'conn': conn_v, 'auth': auth_v,
                             'pick': float(pk) if np.isfinite(pk) else 0.0,
                             'enc': float(en) if np.isfinite(en) else 0.0,
                             'support': support, 'lane_rank': lane_rank})
            n_t = len(rows)

            def bucket(lo, hi):
                return [x for x in rows if lo < x['rank'] <= hi]

            buckets = (('hit@5', bucket(0, 5)), ('6–10', bucket(5, 10)),
                       ('11–25', bucket(10, 25)),
                       ('beyond-25', bucket(25, 10**9)))
            L += ['## %s · %s · n=%d' % (clabel, dlabel, n_t), '',
                  '| shipped-rank bucket | n | gold conn>0 | gold auth>0 | '
                  'gold pick>0 | gold enc>0 | conn lane-rank≤5 | '
                  'median lane support |',
                  '|---|---|---|---|---|---|---|---|']
            for bname, B in buckets:
                if not B:
                    L.append('| %s | 0 | | | | | | |' % bname)
                    continue
                nb = len(B)
                c = sum(1 for x in B if x['conn'] > 0)
                a = sum(1 for x in B if x['auth'] > 0)
                pk = sum(1 for x in B if x['pick'] > 0)
                en = sum(1 for x in B if x['enc'] > 0)
                lo5 = sum(1 for x in B if x['lane_rank'] and
                          x['lane_rank'] <= 5)
                sup = sorted(x['support'] for x in B)[nb // 2]
                L.append('| %s | %d | %d (%.0f%%) | %d (%.0f%%) | %d (%.0f%%)'
                         ' | %d (%.0f%%) | %d (%.0f%%) | %d |'
                         % (bname, nb, c, 100.0 * c / nb, a, 100.0 * a / nb,
                            pk, 100.0 * pk / nb, en, 100.0 * en / nb,
                            lo5, 100.0 * lo5 / nb, sup))
            misses = bucket(5, 10**9)
            m_conn = [x for x in misses if x['conn'] > 0]
            L += ['', '- misses (rank>5): %d; gold conn-touched: %d (%.0f%%);'
                  ' of those, conn-lane-rank≤5: %d — the lane-oracle rescue'
                  ' ceiling' % (len(misses), len(m_conn),
                                100.0 * len(m_conn) / max(len(misses), 1),
                                sum(1 for x in m_conn if x['lane_rank'] and
                                    x['lane_rank'] <= 5)), '']
            print('%s %s: n=%d, miss conn-touch %d/%d' % (
                clabel, dlabel.split(' — ')[0], n_t, len(m_conn),
                len(misses)))
    REPORT.write_text('\n'.join(L) + '\n')
    print('wrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
