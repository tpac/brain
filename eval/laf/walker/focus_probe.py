"""Focus-operator smoke test — does the refocus contrast lane help held-out
soft_r, non-destructively, before touching the frozen definitive_fit?

Refocus (Tom, 2026-07-18, Option 1 additive): the current message plays two
roles — additive evidence AND a suppressor of carried-but-stale nodes. As an
additive fit lane it is a precomputed per-candidate column (like M_e_f):

    current = z(maxsim · op0)                 # what the current message lights
    carried = z(maxsim over j>=1, op+anchor)  # what history carried in
    focus   = max(0, carried - current)       # stale carry to suppress

High focus == a node the thread lit up but THIS message doesn't — the topic-
pivot inertia class. The fit learns rho (expected negative). We reuse the
definitive_fit machinery verbatim and only add the one column, so the base
arms reproduce definitive_fit exactly and the delta is the operator's alone.

Run: ./dev python3 eval/laf/walker/focus_probe.py
"""
import sys
from pathlib import Path

import numpy as np

from walker_db import open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import _zscore                              # noqa: E402
from q1_sweep import load, gate_provenance                          # noqa: E402
from p3_fit import fit_logistic                                     # noqa: E402
from episodic_roles import K_MAX                                    # noqa: E402
from definitive_fit import (turn_features, pairs_soft, val_metrics,  # noqa: E402
                            FEATURES)


def focus_col(td):
    """max(0, carried - current) per candidate, maxsim lane, z-space."""
    nc = len(td.cands)
    cur = _zscore(td.op['maxsim'][:, 0], nc)
    hist = np.concatenate([td.op['maxsim'][:, 1:K_MAX + 1],
                           td.anchor['maxsim'][:, 1:K_MAX + 1]], axis=1)
    with np.errstate(all='ignore'):
        carried_raw = np.where(np.all(np.isnan(hist), axis=1), np.nan,
                               np.nanmax(hist, axis=1))
    carried = _zscore(carried_raw, nc)
    return np.maximum(0.0, carried - cur)


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    walker.close()

    feats = [(td, np.column_stack([turn_features(td), focus_col(td)]))
             for td in turns]
    FEAT2 = FEATURES + ['focus']
    fi = len(FEATURES)                                   # focus column index

    content = [i for i, f in enumerate(FEAT2)
               if not f.startswith(('pick·', 'enc·')) and f != 'focus']
    arms = {
        'S_content        ': content,
        'S_content+focus  ': content + [fi],
    }

    D = pairs_soft(feats)
    print('train soft pairs: %d · %d turns · %d val'
          % (len(D), len(turns), sum(td.val for td in turns)))
    print()
    print('%-18s  val_soft_r  val_AUC  sel@1   M_e_f    focus' % 'arm')
    for name, cols in arms.items():
        w = fit_logistic(D[:, cols])
        m = val_metrics(feats, w, cols)
        wd = dict(zip(cols, w))
        me = wd.get(FEATURES.index('M_e_f'), float('nan'))
        fo = wd.get(fi, float('nan'))
        print('%s  %+.4f    %.4f   %.3f  %+.3f  %+.3f'
              % (name, m['soft_r'], m['auc'], m['sel_at_1'], me, fo))
    return 0


if __name__ == '__main__':
    sys.exit(main())
