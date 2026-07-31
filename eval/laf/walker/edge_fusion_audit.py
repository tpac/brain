"""Self-audit of the cross-lane census — is the divergence a BASE-RATE artifact?

The census reported RISK ratios (gold presence % / non-gold presence %). That
measure is not base-rate invariant: the `current` lane group lights most of
the pool (presence near ceiling → ratios compress toward 1) while `episodic`
lights few nodes (presence low → ratios have room to move). A pure degree
effect (golds have more edges, so more chances of ANY lit neighbour) would
therefore show up as a LARGER apparent lift in the sparse group — exactly the
pattern claimed as evidence for lane provenance.

Three checks, band 6–25 (the claimed effect's home):
  S1 SUPPORT — how many nodes each lane group actually lights per turn. If
     episodic lights ~5 and current ~2000, the cells were never comparable.
  S2 ODDS RATIO — base-rate-invariant re-statement of the same cells.
  S3 DEGREE-MATCHED — per turn, compare the gold against same-band non-golds
     of SIMILAR DEGREE (0.7–1.4× the gold's distinct-partner count). Kills the
     "golds simply have more edges" explanation.

KILL CRITERION (pre-stated): if the episodic-vs-current divergence collapses
below ~1.2× on the odds-ratio AND degree-matched views, the cross-lane claim
is a base-rate artifact and the thesis loses its positive evidence.

Read-only. Run: VECLIB_MAXIMUM_THREADS=3 nice -n 19 \
                ./dev python3 eval/laf/walker/edge_fusion_audit.py
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_brain_ro

sys.path.append(str(OUT_DIR))
import laf_doors as D                                               # noqa: E402
import laf_lane_audit as A                                          # noqa: E402
from lambda_probe import zn                                         # noqa: E402
from edge_fusion_census import (LAM, TOP_N, LIT_Z, LANE_GROUPS,     # noqa: E402
                                verb_class, iso)

REPORT = OUT_DIR / 'edge_fusion_audit.md'
BAND = (6, 25)
CELLS = ('complementary', 'hebbian', 'similarity', 'corrective_strict')
DEG_LO, DEG_HI = 0.7, 1.4


def main():
    aspects = json.loads(
        (Path(__file__).resolve().parents[3] /
         'servers/scales/s2/aspects_v1.json').read_text())
    corrective_all = set(
        (aspects.get('correction_improvement') or {}).get('edge_relations') or [])

    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    m2i = {sid: i for i, sid in enumerate(idx['master'])}
    n_nodes = idx['n_nodes']

    b = open_brain_ro()
    adj = defaultdict(list)
    partners = defaultdict(set)
    for src, tgt, rel, created in b.execute(
            "SELECT e.source_id, e.target_id, r.relation, e.created_at "
            "FROM edges e JOIN edge_relations r ON r.edge_id=e.edge_id "
            "WHERE (r.archived IS NULL OR r.archived=0)"):
        si, ti = m2i.get(src), m2i.get(tgt)
        if si is None or ti is None:
            continue
        vc = verb_class(rel, corrective_all)
        adj[si].append((ti, vc, created))
        adj[ti].append((si, vc, created))
        partners[si].add(ti)
        partners[ti].add(si)
    b.close()
    deg = {k: len(v) for k, v in partners.items()}

    groups = list(LANE_GROUPS) + ['history']
    support = defaultdict(list)                       # group -> [count/turn]
    # raw 2x2 per (vclass, group): gold_with, gold_tot, non_with, non_tot
    tab = defaultdict(lambda: [0, 0, 0, 0])
    # degree-matched: per turn gold presence vs matched non-gold presence rate
    matched = defaultdict(lambda: [[], []])           # -> [gold[], nonmean[]]
    gold_deg, nong_deg = [], []

    turns, _enr, n = D.build_corpus('2026-05-11')
    for t in turns:
        U = np.flatnonzero(t['alive'])
        if U.size < 20:
            continue
        Z = np.column_stack([t['zl'][ln][U] for ln in A.LANES])
        f0 = Z @ np.array([A.GAINS[ln] for ln in A.LANES])
        if not np.isfinite(f0).any() or f0.std() <= 1e-9:
            continue
        zf0 = (f0 - f0.mean()) / f0.std()
        zmh = zn(t['mh'])[U]
        mix = LAM * zf0 + (1.0 - LAM) * zmh
        fin = np.where(np.isfinite(mix), mix, -np.inf)
        order_local = np.argsort(-fin)
        top_nodes = U[order_local[:TOP_N]]
        gpos = np.flatnonzero(top_nodes == t['gr'])

        lit = {}
        for gname, lanes in LANE_GROUPS.items():
            arr = np.full(n_nodes, -np.inf)
            arr[U] = np.nanmax(np.vstack([t['zl'][ln][U] for ln in lanes]),
                               axis=0)
            lit[gname] = arr
        harr = np.full(n_nodes, -np.inf)
        harr[U] = zmh
        lit['history'] = harr
        for gname in groups:
            support[gname].append(int((lit[gname][U] >= LIT_Z).sum()))

        turn_dt = t.get('turn_dt')

        def feats(ni):
            out = set()
            for (oi, vc, created) in adj.get(int(ni), ()):
                edt = iso(created)
                if turn_dt and edt and edt > turn_dt:
                    continue
                for gname in groups:
                    if lit[gname][oi] >= LIT_Z:
                        out.add((vc, gname))
            return out

        band_nodes = [int(x) for i, x in enumerate(top_nodes, 1)
                      if BAND[0] <= i <= BAND[1]]
        gold_in_band = gpos.size and BAND[0] <= int(gpos[0]) + 1 <= BAND[1]
        band_feats = {ni: feats(ni) for ni in band_nodes}
        for ni, fs in band_feats.items():
            is_gold = (ni == t['gr'])
            for vc in CELLS:
                for gname in groups:
                    cell = tab[(vc, gname)]
                    if is_gold:
                        cell[1] += 1
                        cell[0] += 1 if (vc, gname) in fs else 0
                    else:
                        cell[3] += 1
                        cell[2] += 1 if (vc, gname) in fs else 0
        if gold_in_band:
            gd = deg.get(int(t['gr']), 0)
            gold_deg.append(gd)
            pool = [ni for ni in band_nodes if ni != t['gr']
                    and DEG_LO * gd <= deg.get(ni, 0) <= DEG_HI * gd]
            nong_deg.extend(deg.get(ni, 0) for ni in band_nodes
                            if ni != t['gr'])
            if len(pool) >= 3:
                for vc in CELLS:
                    for gname in groups:
                        g = 1.0 if (vc, gname) in band_feats[t['gr']] else 0.0
                        m = np.mean([1.0 if (vc, gname) in band_feats[ni]
                                     else 0.0 for ni in pool])
                        matched[(vc, gname)][0].append(g)
                        matched[(vc, gname)][1].append(m)

    def orat(gw, gt, nw, nt):
        a, b_, c, d = gw, gt - gw, nw, nt - nw
        if min(a, b_, c, d) == 0:      # Haldane correction
            a, b_, c, d = a + .5, b_ + .5, c + .5, d + .5
        return (a / b_) / (c / d)

    L = ['# Cross-lane census — self-audit (base-rate / degree confounds)', '',
         '## S1. Lane-group SUPPORT (nodes lit at z≥%.1f, per turn)' % LIT_Z, '',
         '| lane group | median | p25 | p75 | mean |', '|---|---|---|---|---|']
    for gname in groups:
        v = sorted(support[gname])
        if not v:
            continue
        L.append('| %s | %d | %d | %d | %.0f |'
                 % (gname, v[len(v) // 2], v[len(v) // 4], v[3 * len(v) // 4],
                    float(np.mean(v))))
    L += ['', 'Degree: gold median %d vs non-gold median %d (band %d–%d)'
          % (int(np.median(gold_deg or [0])), int(np.median(nong_deg or [0])),
             *BAND), '',
          '## S2+S3. Band %d–%d: risk ratio vs ODDS ratio vs DEGREE-MATCHED'
          % BAND, '',
          '| verb class | lane | gold% | non% | risk ratio | ODDS ratio | '
          'matched gold% | matched non% | matched diff |',
          '|---|---|---|---|---|---|---|---|---|']
    for vc in CELLS:
        for gname in groups:
            gw, gt, nw, nt = tab[(vc, gname)]
            if not gt or not nt:
                continue
            gr_, nr_ = gw / gt, nw / nt
            rr = (gr_ / nr_) if nr_ > 0 else float('nan')
            orr = orat(gw, gt, nw, nt)
            mg, mn = matched[(vc, gname)]
            mgv = float(np.mean(mg)) if mg else float('nan')
            mnv = float(np.mean(mn)) if mn else float('nan')
            L.append('| %s | %s | %.1f%% | %.1f%% | %.2f× | **%.2f×** | %.1f%% '
                     '| %.1f%% | %+.1fpp |'
                     % (vc, gname, 100 * gr_, 100 * nr_, rr, orr,
                        100 * mgv, 100 * mnv, 100 * (mgv - mnv)))
    L += ['', '### Divergence check (episodic vs current, same verb class)', '',
          '| verb class | risk-ratio divergence | ODDS divergence | '
          'matched-diff episodic | matched-diff current |', '|---|---|---|---|---|']
    for vc in CELLS:
        ge, gc = tab[(vc, 'episodic')], tab[(vc, 'current')]
        if not (ge[1] and gc[1] and ge[3] and gc[3]):
            continue
        rr_e = (ge[0] / ge[1]) / max(ge[2] / ge[3], 1e-9)
        rr_c = (gc[0] / gc[1]) / max(gc[2] / gc[3], 1e-9)
        or_e = orat(*ge)
        or_c = orat(*gc)
        me = matched[(vc, 'episodic')]
        mc = matched[(vc, 'current')]
        d_e = (np.mean(me[0]) - np.mean(me[1])) if me[0] else float('nan')
        d_c = (np.mean(mc[0]) - np.mean(mc[1])) if mc[0] else float('nan')
        L.append('| %s | %.2f× | **%.2f×** | %+.1fpp | %+.1fpp |'
                 % (vc, rr_e / max(rr_c, 1e-9), or_e / max(or_c, 1e-9),
                    100 * d_e, 100 * d_c))
    L += ['', 'matched pairs: %d turns' % len(matched[(CELLS[0], 'episodic')][0])]
    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
