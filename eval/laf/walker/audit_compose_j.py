"""Foundations audit (c) runner: the composition gate with episodic channels
entering at HISTORY slots (op1 / anchor1 / op2, from audit_ext_j_build) in
addition to the j0 cue — does episodic signal compose better through history
cues than it did at j0 (where it washed)?

Arms: champion · +j0 (the original wash arm) · +history slots · +all slots.
Reports held-out soft_r, reach@5, and current-miss-subset recovery/churn.

Run: ./dev python3 eval/laf/walker/audit_compose_j.py   (pool60 via env)
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import _zscore_support                      # noqa: E402
from q1_sweep import load, gate_provenance                          # noqa: E402
from p3_fit import fit_logistic                                     # noqa: E402
from definitive_fit import turn_features, FEATURES                  # noqa: E402
from episodic_compose_fit import pairs, FOLDS                       # noqa: E402

J_SLOTS = ('op1', 'anchor1', 'op2')


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    ext0 = {}
    for s, e, q, nid, pr, eg in walker.execute(
            "SELECT session_id, epoch, seq, node_id, pick_rec, enc_graph "
            "FROM cand_turn_episodic_ext"):
        ext0[(s, e, q, nid)] = (pr, eg)
    extj = defaultdict(dict)
    for s, e, q, nid, slot, pr, eg in walker.execute(
            "SELECT session_id, epoch, seq, node_id, slot, pick_rec, "
            "enc_graph FROM cand_turn_episodic_ext_j"):
        extj[(s, e, q, nid)][slot] = (pr, eg)
    walker.close()

    content = [i for i, f in enumerate(FEATURES)
               if not f.startswith(('pick·', 'enc·'))]
    feats = []
    for td in turns:
        X = turn_features(td)
        nc = len(td.cands)
        cols = []
        for src, slot in [(ext0, None)] + [(extj, s) for s in J_SLOTS]:
            for ch in (0, 1):
                if slot is None:
                    raw = np.array([src.get((*td.key, nid), (0.0, 0.0))[ch]
                                    for nid in td.cands])
                else:
                    raw = np.array([src.get((*td.key, nid), {}).get(
                        slot, (0.0, 0.0))[ch] for nid in td.cands])
                cols.append(_zscore_support(raw, nc))
        X = np.column_stack([X] + cols)
        feats.append((td, X))
    base = len(FEATURES)
    # ext feature indices: [j0_pr, j0_eg, op1_pr, op1_eg, an1_pr, an1_eg,
    #                       op2_pr, op2_eg]
    J0 = [base, base + 1]
    HIST = list(range(base + 2, base + 8))
    arms = {
        'champion (S_content)': content,
        '+j0 (orig wash arm)': content + J0,
        '+history slots only': content + HIST,
        '+all slots': content + J0 + HIST,
    }

    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td in turns if np.isfinite(td.soft).any()])
    hi = np.percentile(allsoft, 90)
    sess = sorted({td.key[0] for td in turns})
    fold_of = {s: i % FOLDS for i, s in enumerate(sess)}
    print('turns %d · sessions %d · soft hi=%.2f' % (len(turns), len(sess), hi))

    # per-fold champion scores for the miss-subset comparison
    champ_scores = {}
    arm_scores = {name: {} for name in arms}
    for f in range(FOLDS):
        tr = {td.key for td in turns if fold_of[td.key[0]] != f}
        te = {td.key for td in turns if fold_of[td.key[0]] == f}
        for name, cols in arms.items():
            w = fit_logistic(pairs(feats, tr, cols))
            for td, X in feats:
                if td.key in te:
                    arm_scores[name][td.key] = X[:, cols] @ w
        for td, X in feats:
            if td.key in te:
                champ_scores[td.key] = arm_scores['champion (S_content)'][
                    td.key]

    print('\narm                     held-out soft_r  reach@5   miss-recov'
          '   hits-lost')
    for name in arms:
        sx, sy, top = [], [], []
        miss = rec = nhit = lost = 0
        for td, X in feats:
            s = arm_scores[name].get(td.key)
            if s is None:
                continue
            m = np.isfinite(td.soft) & np.isfinite(s)
            if m.sum() > 2:
                sx.append(s[m])
                sy.append(td.soft[m])
            if not np.isfinite(td.soft).any():
                continue
            g = int(np.nanargmax(td.soft))
            if td.soft[g] < hi:
                continue
            h = int((s > s[g]).sum()) < 5
            top.append(h)
            c = champ_scores[td.key]
            ch = int((c > c[g]).sum()) < 5
            if ch:
                nhit += 1
                lost += int(not h)
            else:
                miss += 1
                rec += int(h)
        r = float(np.corrcoef(np.concatenate(sx), np.concatenate(sy))[0, 1])
        print('  %-22s  %+.4f         %5.1f%%    %d/%d (%.1f%%)   %d/%d'
              % (name, r, 100 * np.mean(top), rec, miss,
                 100 * rec / max(1, miss), lost, nhit))
    return 0


if __name__ == '__main__':
    sys.exit(main())
