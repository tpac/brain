"""Router step-0 gate — is there a GOLD-BLIND signal that spots poor cases?

Before fitting any dynamic-gain router (which risks hallucinating on the
oracle), the honest question: do the LOW-REACH turns cluster on an observable
feature? If yes → a router has signal to exploit. If poor cases are
feature-uniform → no dynamic constant helps, stop.

Two gold-blind features per turn (op0):
  cur_maxz — the maxsim lane's top z-peak (did THIS message find a sharp
             match; the prior work's best-but-weak router signal, e0ace594)
  cur_gap  — top1 − top2 z of maxsim (peakedness / decisiveness)
plus has_question, op_len (cue-shape proxies from deaf1fb8).

Reports, on clean valid golds (≥CUTOFF):
  A. reach@5 by cur_maxz quartile — does confidence predict reach GOLD-BLIND?
  B. feature profile of HITS vs MISSES — do poor cases separate?
  C. descriptive: which lane ranks the gold best, by cur_maxz quartile — does
     the winning lane actually FLIP with confidence? (uses gold — DESCRIPTIVE
     ONLY, the 'is there signal' check, NOT the router)

Cache-only. Run: ./dev python3 eval/laf/walker/corpus_v2_router_gate.py [cutoff]
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
LANES = ('maxsim', 'sit', 'idf', 'pick', 'enc')


def top2gap(z):
    fin = z[np.isfinite(z)]
    if fin.size < 2:
        return 0.0
    s = np.sort(fin)[::-1]
    return float(s[0] - s[1])


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

    R = []
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
        zlanes = {ln: lane_z(L[S['op0'], LANES.index(ln)], ln, alive, n_nodes)
                  for ln in LANES}
        cur_maxz = float(np.nanmax(zlanes['maxsim']))
        cur_gap = top2gap(zlanes['maxsim'])
        rk = lambda_star(zn(tt.fields[0]), zn(tt.mh), gr, grid=np.array([0.65]))
        mix = min(rk.values()) if rk else None
        # descriptive best-lane (uses gold — signal check only). INCLUDE the
        # history composite M_h: the documented flip (deaf1fb8) is current-vs-
        # history, so M_h must be a candidate or the flip is invisible.
        lane_ranks = {ln: rank_in(zlanes[ln], gr) for ln in LANES}
        lane_ranks['M_h'] = rank_in(tt.mh, gr)
        lane_ranks = {k: r for k, r in lane_ranks.items() if r is not None}
        best_lane = min(lane_ranks, key=lambda k: lane_ranks[k]) if lane_ranks else None
        op = b['op_text'] or ''
        R.append({'door': 'door-1' if v['stratum'] == 'cue' else 'door-2',
                  'cur_maxz': cur_maxz, 'cur_gap': cur_gap,
                  'has_q': '?' in op, 'op_len': len(op),
                  'mix': mix, 'hit': mix is not None and mix <= 5,
                  'best_lane': best_lane})
    N = len(R)
    print('# Router step-0 gate — clean valid golds ≥%s, n=%d\n' % (CUTOFF, N))

    # A. reach by cur_maxz quartile (GOLD-BLIND predictor)
    mz = sorted(r['cur_maxz'] for r in R)
    q = [mz[int(N*f)] for f in (0.25, 0.5, 0.75)]
    def qb(x):
        return 0 if x < q[0] else 1 if x < q[1] else 2 if x < q[2] else 3
    print('## A. reach@5 by cur_maxz quartile (gold-blind — does confidence predict reach?)\n')
    print('| quartile | cur_maxz range | n | reach@5 |')
    print('|---|---|---|---|')
    edges = ['<%.2f' % q[0], '%.2f-%.2f' % (q[0], q[1]),
             '%.2f-%.2f' % (q[1], q[2]), '>%.2f' % q[2]]
    for qi in range(4):
        sub = [r for r in R if qb(r['cur_maxz']) == qi]
        print('| Q%d | %s | %d | %.0f%% |'
              % (qi+1, edges[qi], len(sub),
                 100*sum(r['hit'] for r in sub)/max(1, len(sub))))

    # B. feature profile hits vs misses
    print('\n## B. feature profile — HITS vs MISSES (gold-blind features)\n')
    hits = [r for r in R if r['hit']]
    miss = [r for r in R if not r['hit']]
    print('| feature | hits (n=%d) | misses (n=%d) | separation |' % (len(hits), len(miss)))
    print('|---|---|---|---|')
    for f, lbl in (('cur_maxz', 'cur_maxz (mean)'), ('cur_gap', 'cur_gap (mean)'),
                   ('op_len', 'op_len (median)')):
        h = np.mean([r[f] for r in hits]) if f != 'op_len' else np.median([r[f] for r in hits])
        m = np.mean([r[f] for r in miss]) if f != 'op_len' else np.median([r[f] for r in miss])
        print('| %s | %.2f | %.2f | %+.2f |' % (lbl, h, m, h - m))
    hq = 100*sum(r['has_q'] for r in hits)/len(hits)
    mq = 100*sum(r['has_q'] for r in miss)/len(miss)
    print('| has_question %% | %.0f%% | %.0f%% | %+.0fpp |' % (hq, mq, hq - mq))

    # C. best-lane by cur_maxz quartile (DESCRIPTIVE — does winner flip?)
    print('\n## C. best-ranking lane by cur_maxz quartile (descriptive — does the winner FLIP?)\n')
    print('| quartile | n | maxsim | M_h | sit | idf | pick | enc | dominant |')
    print('|---|---|---|---|---|---|---|---|---|')
    for qi in range(4):
        sub = [r for r in R if qb(r['cur_maxz']) == qi and r['best_lane']]
        from collections import Counter
        c = Counter(r['best_lane'] for r in sub)
        n = max(1, len(sub))
        dom = c.most_common(1)[0][0] if c else '-'
        print('| Q%d | %d | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %s |'
              % (qi+1, len(sub), 100*c['maxsim']/n, 100*c['M_h']/n, 100*c['sit']/n,
                 100*c['idf']/n, 100*c['pick']/n, 100*c['enc']/n, dom))

    print('\n## Read')
    lo = [r for r in R if qb(r['cur_maxz']) == 0]
    hi = [r for r in R if qb(r['cur_maxz']) == 3]
    lo_r = 100*sum(r['hit'] for r in lo)/max(1, len(lo))
    hi_r = 100*sum(r['hit'] for r in hi)/max(1, len(hi))
    print('- cur_maxz spread: Q1 reach %.0f%% vs Q4 reach %.0f%% (%+.0fpp gold-blind separation)'
          % (lo_r, hi_r, hi_r - lo_r))
    print('- If separation is large AND the best-lane flips across quartiles → router SIGNAL exists, proceed to held-out fit.')
    print('- If flat / uniform → no cheap router, the fixed blend is right; lever is a new reach signal (graph/next-move).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
