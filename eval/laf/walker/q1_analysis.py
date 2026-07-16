"""Q1 post-verdict analysis — contributors, not new claims (Tom 2026-07-15).

Three decompositions of the RANK result, all on the same walker v6 table and
the same June+ holdout metric the verdict used:
  1. axis marginals — every grid axis's effect distribution (readable
     comparison, replaces eyeballing 673 rows)
  2. lane contributions — winner shape with each lane dropped (marginal
     value) and each lane solo (standalone power)
  3. message contributions — the winner's turn anatomy: j=0 prompt,
     j=1 previous-op, j=1 previous-anchor, ablated explicitly

No new configs are scored as candidates — this decomposes the registered
winner. Output: q1_analysis.json (the artifact's data feed) + stdout.

Run:  ./dev python3 eval/laf/walker/q1_analysis.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import open_walker, WALKER_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import q1_sweep                                                   # noqa: E402
from q1_sweep import GAINS, compose, weights, configs, auc        # noqa: E402

WINNER = 'K1-exp0.5-turnsum-zsum-opanchor-me0'
OUT = WALKER_DIR / 'q1_analysis.json'


def val_auc(turns, score_fn):
    sel, drp = [], []
    for td in turns:
        if not td.val:
            continue
        s = score_fn(td)
        sel.append(s[td.sel])
        drp.append(s[~td.sel])
    return auc(np.concatenate(sel), np.concatenate(drp))


def axis_marginals(results):
    axes = {
        'K': lambda n: re.match(r'K(\d+)', n).group(1),
        'decay': lambda n: (re.search(r'-(exp[\d.]+|pow[\d.]+|uniform)-', n)
                            or [None, '—'])[1],
        'composition': lambda n: 'turnsum' if 'turnsum' in n else 'turnmax',
        'aggregation': lambda n: 'zsum' if '-zsum-' in n else 'lane',
        'texts': lambda n: 'op+anchor' if 'opanchor' in n else 'op',
        'M_e': lambda n: 'δ' + (re.search(r'me([\d.]+)', n) or [None, '0']
                                )[1].replace('0', 'off', 1)
               if n.endswith('me0') else 'δ' + n.rsplit('me', 1)[1],
    }
    out = {}
    for ax, f in axes.items():
        groups = defaultdict(list)
        for r in results:
            if r['name'] == 'K0':
                if ax == 'K':
                    groups['0'].append(r['d_val'])
                continue
            groups[f(r['name'])].append(r['d_val'])
        out[ax] = {k: {'best': max(v), 'median': float(np.median(v)),
                       'worst': min(v), 'n': len(v)}
                   for k, v in sorted(groups.items())}
    return out


def main():
    results = json.loads((WALKER_DIR / 'q1_sweep_full.json').read_text())
    walker = open_walker()
    q1_sweep.gate_provenance(walker)
    turns = q1_sweep.load(walker)
    walker.close()
    cfg = next(c for c in configs() if c['name'] == WINNER)
    k0 = configs()[0]
    w = weights(cfg)

    # ---- lane contributions on the winner shape ----
    def with_gains(g):
        return val_auc(turns, lambda td: q1_sweep.score_turn(td, cfg, w,
                                                             gains=g))
    full = with_gains(None)
    k0_auc = val_auc(turns, lambda td: q1_sweep.score_turn(td, k0,
                                                           weights(k0)))
    lanes = {}
    for ln in GAINS:
        dropped = {k: v for k, v in GAINS.items() if k != ln}
        solo = {ln: GAINS[ln]}
        lanes[ln] = {'gain': GAINS[ln],
                     'drop': with_gains(dropped) - full,   # marginal value
                     'solo': with_gains(solo)}             # standalone AUC

    # ---- message contributions (winner anatomy, explicit columns) ----
    # slots: [j0-op(q_vec), j1-op, j1-anchor], base weights [1, .5, .5]
    def msg_auc(ww_mask):
        def score(td):
            nc = len(td.cands)
            mats = {}
            for ln in GAINS:
                m = np.concatenate([td.op[ln][:, :2],
                                    td.anchor[ln][:, 1:2]], axis=1)
                mats[ln] = m
            ww = np.array([1.0, 0.5, 0.5]) * np.asarray(ww_mask, dtype=float)
            return compose(mats, ww, cfg, nc)
        return val_auc(turns, score)
    messages = {
        'full (j0 + j1-op + j1-anchor)': msg_auc([1, 1, 1]),
        'drop j1-anchor (op-only history)': msg_auc([1, 1, 0]),
        'drop j1-op (anchor-only history)': msg_auc([1, 0, 1]),
        'j0 only (≡ K0 shape)': msg_auc([1, 0, 0]),
    }

    out = {'winner': WINNER, 'k0_val_auc': k0_auc, 'winner_val_auc': full,
           'axis_marginals': axis_marginals(results),
           'lane_contributions': lanes, 'message_contributions': messages}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:3000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
