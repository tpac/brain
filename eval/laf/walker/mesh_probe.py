"""Meshing mechanics — how carried (history) activation combines with the
current node across strength strata, on the FIELD score (not Haiku surface).

The fitted A1 (S_content) score is linear, so it splits exactly:
    score(i) = current(i) + carried(i) + me(i)
    current  = sum over j0   slots  of  w·z(lane)     # this message
    carried  = sum over j>=1 slots  of  w·z(lane)     # prior turns' mesh
    me       = w_M_e_f · fat                          # fatigue

Two outputs:
 1. MESHING TABLE — pool val candidates, bin by `current` tercile
    (strong/med/low current node), and within each bin ask whether `carried`
    separates soft-relevant from soft-irrelevant, and how often carried
    rescues a soft-high node into top-5 vs buries one out (rank-by-score vs
    rank-by-current-only).
 2. FAILED CASES — val turns where the most soft-relevant candidate is ranked
    OUTSIDE top-5 by the field score. Dump each candidate's decomposition +
    node_ids to JSON (titles fetched via the brain afterward), so we can ask
    algorithmically WHY the gold didn't make the cut.

Run: ./dev python3 eval/laf/walker/mesh_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from q1_sweep import load, gate_provenance                          # noqa: E402
from p3_fit import fit_logistic                                     # noqa: E402
from definitive_fit import turn_features, pairs_soft, FEATURES      # noqa: E402

SOFT_HI = 0.60                     # "clearly relevant to the next response"
OUT = Path('/private/tmp/claude-503/-Users-tpac-brain/'
           '4e1b0747-c5c6-4c35-8a8c-564bb2b8b4ca/scratchpad/mesh_cases.json')


def col_roles():
    """content column index -> ('current'|'carried'|'me')."""
    content = [i for i, f in enumerate(FEATURES)
               if not f.startswith(('pick·', 'enc·'))]
    role = {}
    for i in content:
        f = FEATURES[i]
        if f == 'M_e_f':
            role[i] = 'me'
        else:
            j = int(f.split('·')[1].lstrip('opanchor'))   # trailing slot int
            role[i] = 'current' if j == 0 else 'carried'
    return content, role


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    # cue text + node lists for the drill
    op_text = {}
    for sess, epoch, seq, txt in walker.execute(
            "SELECT session_id, epoch, seq, op_text FROM turns"):
        op_text[(sess, epoch, seq)] = txt
    walker.close()

    feats = [(td, turn_features(td)) for td in turns]
    content, role = col_roles()
    w = fit_logistic(pairs_soft(feats)[:, content])
    wmap = dict(zip(content, w))
    cur_cols = [c for c in content if role[c] == 'current']
    car_cols = [c for c in content if role[c] == 'carried']
    me_col = [c for c in content if role[c] == 'me'][0]

    def parts(X):
        cur = X[:, cur_cols] @ np.array([wmap[c] for c in cur_cols])
        car = X[:, car_cols] @ np.array([wmap[c] for c in car_cols])
        me = X[:, me_col] * wmap[me_col]
        return cur, car, me

    # ---- pooled val rows for the meshing table ----
    CUR, CAR, SOFT = [], [], []
    for td, X in feats:
        if not td.val:
            continue
        cur, car, _ = parts(X)
        CUR.append(cur); CAR.append(car); SOFT.append(td.soft)
    CUR = np.concatenate(CUR); CAR = np.concatenate(CAR)
    SOFT = np.concatenate(SOFT)
    fin = np.isfinite(SOFT)
    hi_thr = np.nanpercentile(SOFT, 90)               # real relevance tail
    q = np.quantile(CUR, [1 / 3, 2 / 3])
    strat = np.digitize(CUR, q)                       # 0 low,1 med,2 high
    names = ['low ', 'med ', 'high']
    print('val candidates: %d (soft-labeled %d) · hi=90th pctile soft=%.2f'
          % (len(CUR), int(fin.sum()), hi_thr))
    print('soft pctiles: ' + ' '.join(
        '%d%%=%.2f' % (p, np.nanpercentile(SOFT, p))
        for p in (50, 75, 90, 95)))
    print()
    print('current   n_cand   carried|soft>=90   carried|soft<90    Δ      '
          'corr(car,soft)')
    for s in (2, 1, 0):
        m = strat == s
        hi = m & fin & (SOFT >= hi_thr)
        lo = m & fin & (SOFT < hi_thr)
        chi, clo = CAR[hi].mean(), CAR[lo].mean()
        mf = m & fin
        r = (np.corrcoef(CAR[mf], SOFT[mf])[0, 1]
             if mf.sum() > 2 else float('nan'))
        print('%s      %6d      %+.3f (%5d)    %+.3f (%5d)   %+.3f    %+.3f'
              % (names[s], int(m.sum()), chi, int(hi.sum()),
                 clo, int(lo.sum()), chi - clo, r))

    # ---- carried-flood vs current-miss: of the failures, how many would
    # current-alone have surfaced (carried buried the gold = refocus-fixable)
    # vs never reached (current-miss = refocus can't help)? per-turn gold =
    # argmax soft with soft >= hi_thr. ----
    flood = miss = ok = 0
    for td, X in feats:
        if not td.val:
            continue
        cur, car, me = parts(X)
        score = cur + car + me
        soft = td.soft
        if not np.isfinite(soft).any():
            continue
        g = int(np.nanargmax(soft))
        if soft[g] < hi_thr:
            continue
        rank_full = int((score > score[g]).sum())
        rank_cur = int((cur > cur[g]).sum())
        if rank_full < 5:
            ok += 1
        elif rank_cur < 5:
            flood += 1                                 # current had it, carried buried it
        else:
            miss += 1                                  # current never reached it
    tot = ok + flood + miss
    print('\nper-turn gold (argmax soft >= 90th pctile), N=%d turns:' % tot)
    print('  in top-5 by field score:        %d (%.0f%%)' % (ok, 100 * ok / tot))
    print('  FAILED — carried-flood (refocus-fixable): %d (%.0f%%)'
          % (flood, 100 * flood / tot))
    print('  FAILED — current-miss (maxsim reach):     %d (%.0f%%)'
          % (miss, 100 * miss / tot))

    # ---- failed cases: soft-hi gold ranked outside top-5 by field ----
    cases = []
    for td, X in feats:
        if not td.val:
            continue
        cur, car, me = parts(X)
        score = cur + car + me
        soft = td.soft
        if not np.isfinite(soft).any():
            continue
        g = int(np.nanargmax(soft))
        if soft[g] < hi_thr:
            continue
        rank_g = int((score > score[g]).sum())         # 0-based rank of gold
        if rank_g < 5:
            continue
        kind = 'flood' if int((cur > cur[g]).sum()) < 5 else 'miss'
        order = (-score).argsort()
        def rec(i):
            return {'node': td.cands[i], 'cur': round(float(cur[i]), 3),
                    'car': round(float(car[i]), 3), 'me': round(float(me[i]), 3),
                    'score': round(float(score[i]), 3),
                    'soft': None if not np.isfinite(soft[i])
                    else round(float(soft[i]), 3),
                    'sel': bool(td.sel[i])}
        cases.append({
            'key': list(td.key), 'cue': (op_text.get(td.key) or '')[:240],
            'kind': kind, 'gold_rank': rank_g, 'n_cand': len(td.cands),
            'gold': rec(g), 'top5': [rec(int(i)) for i in order[:5]]})
    cases.sort(key=lambda c: -c['gold']['soft'])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, indent=1))
    print('\nfailed cases (soft-hi gold ranked >5): %d  → %s' % (len(cases), OUT))
    for c in cases[:2]:
        print('\n--- gold rank %d/%d · soft %.2f · cue: %s'
              % (c['gold_rank'], c['n_cand'], c['gold']['soft'],
                 c['cue'][:90].replace('\n', ' ')))
        g = c['gold']
        print('   GOLD %s  cur%+.2f car%+.2f me%+.2f = %+.2f'
              % (g['node'], g['cur'], g['car'], g['me'], g['score']))
        for r in c['top5']:
            print('   #    %s  cur%+.2f car%+.2f me%+.2f = %+.2f  soft%s'
                  % (r['node'], r['cur'], r['car'], r['me'], r['score'],
                     r['soft']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
