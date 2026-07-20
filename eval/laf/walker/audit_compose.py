"""Foundations audit (2026-07-20, Tom's mandate): re-verify the composition
wash-out verdict on pick_rec/enc_graph before building on it.

(a) fitted-gain forensics — is w_pick ~0 because the lane is REDUNDANT with
    the dense blend, or because support-z leaves it structurally unable to
    express itself (variance artifact) in the Bradley-Terry pair space?
    Prints per-fold fitted weights, D-column stds (can the lane move pairs?),
    and the z_pick vs champion-score correlation on support rows.
(b) current-miss subset — reach@5 measured ONLY on the golds the champion
    misses (rank>=5), where the all-golds aggregate could mask a subset
    gain; plus a frozen-champion gain sweep on z_pick (could ANY gain
    recover misses without churning champion hits?).

Run: ./dev python3 eval/laf/walker/audit_compose.py   (pool60 via env vars)
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
from episodic_compose_fit import pairs, FOLDS                       # noqa: E402


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

    content = [i for i, f in enumerate(FEATURES)
               if not f.startswith(('pick·', 'enc·'))]
    feats, sup_pr, sup_eg, turn_has = [], [], [], 0
    for td in turns:
        X = turn_features(td)
        nc = len(td.cands)
        pr = np.array([ext.get((*td.key, nid), (0.0, 0.0))[0]
                       for nid in td.cands])
        eg = np.array([ext.get((*td.key, nid), (0.0, 0.0))[1]
                       for nid in td.cands])
        sup_pr.append((pr != 0).mean())
        sup_eg.append((eg != 0).mean())
        turn_has += int((pr != 0).any())
        X = np.column_stack([X, _zscore_support(pr, nc),
                             _zscore_support(eg, nc)])
        feats.append((td, X))
    PR, EG = len(FEATURES), len(FEATURES) + 1

    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td in turns if np.isfinite(td.soft).any()])
    hi = np.percentile(allsoft, 90)
    sess = sorted({td.key[0] for td in turns})
    fold_of = {s: i % FOLDS for i, s in enumerate(sess)}

    print('turns %d · pick_rec support %.1f%% of cands/turn (%.0f%% of turns '
          'have any) · enc_graph %.1f%%'
          % (len(turns), 100 * np.mean(sup_pr),
             100 * turn_has / len(turns), 100 * np.mean(sup_eg)))

    # ---- (a) fitted-gain forensics on the +both arm, per fold ----
    cols_b = content + [PR, EG]
    print('\n(a) fold   w_pick    w_enc    |w_content| mean/max'
          '    D-col std pick/enc (content median)')
    per_turn = {}
    for f in range(FOLDS):
        tr = {td.key for td in turns if fold_of[td.key[0]] != f}
        te = {td.key for td in turns if fold_of[td.key[0]] == f}
        w_c = fit_logistic(pairs(feats, tr, content))
        D_b = pairs(feats, tr, cols_b)
        w_b = fit_logistic(D_b)
        dstd = D_b.std(axis=0)
        wc_abs = np.abs(w_b[:len(content)])
        print('     %d    %+.4f  %+.4f    %.3f / %.3f'
              '            %.3f / %.3f (%.3f)'
              % (f, w_b[-2], w_b[-1], wc_abs.mean(), wc_abs.max(),
                 dstd[-2], dstd[-1], np.median(dstd[:len(content)])))
        for td, X in feats:
            if td.key in te:
                per_turn[td.key] = {'champ': X[:, content] @ w_c,
                                    'both': X[:, cols_b] @ w_b,
                                    'z_pr': X[:, PR]}

    # redundancy: does z_pick just re-rank what the champion already ranks?
    cz, cc = [], []
    for td, X in feats:
        sc = per_turn.get(td.key)
        if sc is None:
            continue
        m = sc['z_pr'] != 0
        if m.sum() > 2:
            cz.append(sc['z_pr'][m])
            cc.append(sc['champ'][m])
    r = float(np.corrcoef(np.concatenate(cz), np.concatenate(cc))[0, 1])
    print('    corr(z_pick, champion score) on support rows: %+.3f' % r)

    # ---- (b) current-miss subset ----
    n_gold = miss = rec_both = lost_both = miss_sup = hit_sup = 0
    for td, X in feats:
        sc = per_turn.get(td.key)
        if sc is None or not np.isfinite(td.soft).any():
            continue
        g = int(np.nanargmax(td.soft))
        if td.soft[g] < hi:
            continue
        n_gold += 1
        champ_hit = int((sc['champ'] > sc['champ'][g]).sum()) < 5
        both_hit = int((sc['both'] > sc['both'][g]).sum()) < 5
        if champ_hit:
            lost_both += int(not both_hit)
            hit_sup += int(sc['z_pr'][g] != 0)
        else:
            miss += 1
            rec_both += int(both_hit)
            miss_sup += int(sc['z_pr'][g] != 0)
    print('\n(b) gold turns %d · champion misses %d (%.1f%%)'
          % (n_gold, miss, 100 * miss / max(1, n_gold)))
    print('    +both recovers %d/%d misses (%.1f%%) · loses %d/%d champion '
          'hits' % (rec_both, miss, 100 * rec_both / max(1, miss),
                    lost_both, n_gold - miss))
    print('    pick_rec support ON THE GOLD: missed %d/%d (%.0f%%) · hit '
          '%d/%d (%.0f%%)'
          % (miss_sup, miss, 100 * miss_sup / max(1, miss),
             hit_sup, n_gold - miss,
             100 * hit_sup / max(1, n_gold - miss)))

    print('\n    gain sweep  s = champ + g*z_pick     overall   miss-subset'
          '   churn(lost hits)')
    for gain in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6):
        tot = hit = mrec = lost = 0
        for td, X in feats:
            sc = per_turn.get(td.key)
            if sc is None or not np.isfinite(td.soft).any():
                continue
            g = int(np.nanargmax(td.soft))
            if td.soft[g] < hi:
                continue
            s = sc['champ'] + gain * sc['z_pr']
            h = int((s > s[g]).sum()) < 5
            ch = int((sc['champ'] > sc['champ'][g]).sum()) < 5
            tot += 1
            hit += int(h)
            if not ch:
                mrec += int(h)
            elif not h:
                lost += 1
        print('      g=%.2f                              %5.1f%%    %5.1f%%'
              '        %d' % (gain, 100 * hit / max(1, tot),
                              100 * mrec / max(1, miss), lost))
    return 0


if __name__ == '__main__':
    sys.exit(main())
