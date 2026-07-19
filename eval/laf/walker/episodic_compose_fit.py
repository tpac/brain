"""The composition test / ship gate (Tom, 2026-07-18): does adding pick_rec +
enc_graph to the FULL fitted z-blend beat the champion on held-out soft_r AND
reach@5 — or does z-competition with the dense lanes wash it out (as routing
did, 88414714)?

Champion = S_content (content lanes j0..j8, pick/enc zeroed). Arms add the ext
lanes (from cand_turn_episodic_ext, per-turn support-z) as extra fit features.
Session-grouped 5-fold CV (works on both corpora; avoids pool60's build-time
val-split degeneracy). Held-out soft_r (Pearson vs soft_max) + reach@5 (gold =
argmax soft >= 90th pctile in top-5 by composed score).

Run: ./dev python3 eval/laf/walker/episodic_compose_fit.py  (after ext_build)
"""
import sys
from pathlib import Path

import numpy as np

from walker_db import open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import _zscore_support                      # noqa: E402
from q1_sweep import load, gate_provenance                          # noqa: E402
from p3_fit import fit_logistic                                     # noqa: E402
from definitive_fit import turn_features, FEATURES                  # noqa: E402

SOFT_MARGIN = 0.10
FOLDS = 5


def pairs(feats, keys, cols):
    rows = []
    for td, X in feats:
        if td.key not in keys or not np.isfinite(td.soft).any():
            continue
        fin = np.flatnonzero(np.isfinite(td.soft))
        if len(fin) < 2:
            continue
        s = td.soft[fin]
        wi, li = np.nonzero((s[:, None] - s[None, :]) >= SOFT_MARGIN)
        if len(wi):
            rows.append(X[fin[wi]][:, cols] - X[fin[li]][:, cols])
    return np.concatenate(rows) if rows else np.zeros((0, len(cols)))


def evaluate(feats, keys, cols, w, hi):
    sx, sy, top = [], [], []
    for td, X in feats:
        if td.key not in keys:
            continue
        s = X[:, cols] @ w
        m = np.isfinite(td.soft) & np.isfinite(s)
        if m.sum() > 2:
            sx.append(s[m]); sy.append(td.soft[m])
        if np.isfinite(td.soft).any():
            g = int(np.nanargmax(td.soft))
            if td.soft[g] >= hi:
                top.append(int((s > s[g]).sum()) < 5)
    r = float(np.corrcoef(np.concatenate(sx), np.concatenate(sy))[0, 1]) \
        if sx else float('nan')
    return r, (np.mean(top) if top else float('nan')), len(top)


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    ext = {}
    for s, e, q, nid, pr, eg in walker.execute(
            "SELECT session_id, epoch, seq, node_id, pick_rec, enc_graph "
            "FROM cand_turn_episodic_ext"):
        ext[(s, e, q, nid)] = (pr, eg)
    walker.close()

    # base content features + appended per-turn support-z ext lanes
    content = [i for i, f in enumerate(FEATURES)
               if not f.startswith(('pick·', 'enc·'))]
    feats = []
    for td in turns:
        X = turn_features(td)
        nc = len(td.cands)
        pr = np.array([ext.get((*td.key, nid), (0.0, 0.0))[0]
                       for nid in td.cands])
        eg = np.array([ext.get((*td.key, nid), (0.0, 0.0))[1]
                       for nid in td.cands])
        X = np.column_stack([X, _zscore_support(pr, nc),
                             _zscore_support(eg, nc)])
        feats.append((td, X))
    PR, EG = len(FEATURES), len(FEATURES) + 1
    arms = {
        'champion (S_content)': content,
        '+pick_rec': content + [PR],
        '+pick_rec+enc_graph': content + [PR, EG],
        '+enc_graph only': content + [EG],
    }
    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td in turns if np.isfinite(td.soft).any()])
    hi = np.percentile(allsoft, 90)

    sess = sorted({td.key[0] for td in turns})
    fold_of = {s: i % FOLDS for i, s in enumerate(sess)}
    print('turns %d · sessions %d · %d-fold session-grouped CV · soft hi=%.2f'
          % (len(turns), len(sess), FOLDS, hi))
    print('\narm                     held-out soft_r    reach@5')
    for name, cols in arms.items():
        rs, ts = [], []
        for f in range(FOLDS):
            tr = {td.key for td in turns if fold_of[td.key[0]] != f}
            te = {td.key for td in turns if fold_of[td.key[0]] == f}
            D = pairs(feats, tr, cols)
            if len(D) < 10:
                continue
            w = fit_logistic(D)
            r, t, _ = evaluate(feats, te, cols, w, hi)
            if np.isfinite(r):
                rs.append(r)
            if np.isfinite(t):
                ts.append(t)
        print('  %-22s  %+.4f           %4.1f%%'
              % (name, np.mean(rs), 100 * np.mean(ts)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
