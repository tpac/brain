"""Reach attribution on the current-miss golds — of the golds the composed
field buried (in top-25, ranked >5), which single LANE or CUE-SLOT would have
ranked them into top-5? The realizable-selector question (8bcc8c96): is the
signal already in a lane we compute (→ reweight/select recovers it) or stuck
(→ needs new signal)?

SCOPE: pool-resident re-ranking, NOT out-of-pool reach — the walker only holds
the ~25 candidates recall already pulled, so a gold not in the pool is invisible
here (that's the endo_reverse_regress full-node analysis, separate & heavier).

Lanes tested per current-miss gold, each z-scored over the turn's pool, gold
ranked; "reaches" = rank < 5:
  current message (j0): the 6 maxsim views individually, their nanmax, sit, idf
  episodic:            pick, enc
  history cue:         best maxsim over prior slots op1..8 / anchor1..8

Run: ./dev python3 eval/laf/walker/reach_probe.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import (_zscore, _zscore_support,           # noqa: E402
                                MAXSIM_VIEWS, SUPPORT_ZERO_SEA_LANES)
from q1_sweep import load, gate_provenance, V_OP                    # noqa: E402
from p3_fit import fit_logistic                                     # noqa: E402
from definitive_fit import turn_features, pairs_soft, FEATURES      # noqa: E402
from episodic_roles import K_MAX                                    # noqa: E402

VIEWS = [v.strip('_') for v in MAXSIM_VIEWS]        # title primary high_meta ...


def current_miss(feats):
    """Reuse the mesh decomposition: gold = argmax soft >= 90th pctile,
    field-rank >= 5, current-alone-rank >= 5. Returns [(td, gold_idx)]."""
    content = [i for i, f in enumerate(FEATURES)
               if not f.startswith(('pick·', 'enc·'))]
    w = fit_logistic(pairs_soft(feats)[:, content])
    wmap = dict(zip(content, w))
    cur_cols, car_cols, me_col = [], [], None
    for c in content:
        f = FEATURES[c]
        if f == 'M_e_f':
            me_col = c
        elif int(f.split('·')[1].lstrip('opanchor')) == 0:
            cur_cols.append(c)
        else:
            car_cols.append(c)
    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td, _ in feats if np.isfinite(td.soft).any()])
    hi = np.percentile(allsoft, 90)
    out = []
    for td, X in feats:
        if not td.val or not np.isfinite(td.soft).any():
            continue
        cur = X[:, cur_cols] @ np.array([wmap[c] for c in cur_cols])
        car = X[:, car_cols] @ np.array([wmap[c] for c in car_cols])
        score = cur + car + X[:, me_col] * wmap[me_col]
        g = int(np.nanargmax(td.soft))
        if td.soft[g] < hi:
            continue
        if int((score > score[g]).sum()) < 5:            # field got it top-5
            continue
        if int((cur > cur[g]).sum()) < 5:                # carried-flood, not miss
            continue
        out.append((td, g))
    return out, hi


def per_view_rows(walker, keys):
    """{key: {node_id: {view/sit/idf: value}}} for j=0 (current message)."""
    d = defaultdict(dict)
    nv = len(V_OP)
    cols = ', '.join(V_OP)
    for row in walker.execute(
            "SELECT session_id, epoch, seq, node_id, %s, sit_op, idf_op "
            "FROM cand_turn_scores WHERE j=0" % cols):
        key = row[:3]
        if key not in keys:
            continue
        vals = row[4:4 + nv]
        rec = {VIEWS[i]: vals[i] for i in range(nv)}
        rec['sit'], rec['idf'] = row[4 + nv], row[5 + nv]
        d[key][row[3]] = rec
    epi = defaultdict(dict)
    for row in walker.execute(
            "SELECT session_id, epoch, seq, node_id, pick_op, enc_op "
            "FROM cand_turn_episodic WHERE j=0"):
        key = row[:3]
        if key not in keys:
            continue
        epi[key][row[3]] = {'pick': row[4], 'enc': row[5]}
    return d, epi


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    feats = [(td, turn_features(td)) for td in turns]
    miss, hi = current_miss(feats)
    keys = {td.key for td, _ in miss}
    rowd, epi = per_view_rows(walker, keys)
    walker.close()

    lanes = VIEWS + ['maxsim', 'sit', 'idf', 'pick', 'enc', 'hist_maxsim']
    reach = {ln: 0 for ln in lanes}
    support = {ln: 0 for ln in lanes}
    union = dense_union = stuck = 0
    N = len(miss)
    for td, g in miss:
        nc = len(td.cands)
        rec = rowd.get(td.key, {})
        er = epi.get(td.key, {})
        # build per-lane raw vectors over the pool
        lv = {}
        for v in VIEWS:
            lv[v] = np.array([rec.get(n, {}).get(v, np.nan) for n in td.cands],
                             float)
        lv['maxsim'] = np.nanmax(np.stack([lv[v] for v in VIEWS]), axis=0)
        lv['sit'] = np.array([rec.get(n, {}).get('sit', np.nan)
                              for n in td.cands], float)
        # support-zero-sea lanes: absent == 0 (no activation), not NaN
        lv['idf'] = np.array([rec.get(n, {}).get('idf', 0.0) or 0.0
                              for n in td.cands], float)
        lv['pick'] = np.array([er.get(n, {}).get('pick', 0.0) or 0.0
                               for n in td.cands], float)
        lv['enc'] = np.array([er.get(n, {}).get('enc', 0.0) or 0.0
                              for n in td.cands], float)
        hist = np.concatenate([td.op['maxsim'][:, 1:K_MAX + 1],
                               td.anchor['maxsim'][:, 1:K_MAX + 1]], axis=1)
        with np.errstate(all='ignore'):
            lv['hist_maxsim'] = np.where(np.all(np.isnan(hist), axis=1), np.nan,
                                         np.nanmax(hist, axis=1))
        got = dense_got = False
        for ln in lanes:
            base = ln.split('_')[0] if ln == 'hist_maxsim' else ln
            if base in SUPPORT_ZERO_SEA_LANES:
                z = _zscore_support(lv[ln], nc)
                support[ln] += int((lv[ln] != 0).sum())
            else:
                z = _zscore(lv[ln], nc)
                support[ln] += int(np.isfinite(lv[ln]).sum())
            if int((z > z[g]).sum()) < 5:                # gold in top-5 on lane
                reach[ln] += 1
                got = True
                if ln not in ('pick', 'enc'):            # dense lanes only
                    dense_got = True
        union += int(got)
        dense_union += int(dense_got)
        stuck += int(not got)

    print('current-miss golds (pool-resident, field-rank>5): %d  '
          '(soft hi=%.2f)' % (N, hi))
    print('\nlane                reaches top-5   %     avg support/pool')
    for ln in sorted(lanes, key=lambda k: -reach[k]):
        print('  %-16s  %5d        %4.0f%%      %.1f'
              % (ln, reach[ln], 100 * reach[ln] / N, support[ln] / N))
    print('\n  UNION (>=1 lane)      %5d     %4.0f%%' % (union, 100 * union / N))
    print('  UNION dense (no pick/enc) %5d %4.0f%%' % (dense_union, 100 * dense_union / N))
    print('  STUCK (no lane)    %5d        %4.0f%%' % (stuck, 100 * stuck / N))
    return 0


if __name__ == '__main__':
    sys.exit(main())
