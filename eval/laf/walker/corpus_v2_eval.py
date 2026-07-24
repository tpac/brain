"""Corpus-v2 eval — honest baselines DECOMPOSED (Tom: 'analyze from different
aspects, not raw high-level KPIs — what got in, what didn't, the patterns').

On the CLEAN corpus (valid golds only, turn-date >= CUTOFF), per gold:
  - mix rank (static λ=0.65) → HIT@5 / miss
  - field ranks (F0, M_h) and op0 lane ranks (maxsim/sit/idf/pick/enc)
  - best-lane and which lane holds it → miss class (REACHABLE/ALMOST/
    BURIED/BARELY) and, for HITS, the carrier lane (attribution)

Then breaks HITS and MISSES down by door / stratum / gold-type / age, and
names the displacer lane on misses. Cache-only, zero tokens.

Machinery imported: Turn, lane_z, rank_in, classify, zn, lambda_star.
Run: ./dev python3 eval/laf/walker/corpus_v2_eval.py [cutoff_iso]
Out: OUT_DIR/corpus_v2_eval.md
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from walker_db import OUT_DIR

sys.path.insert(0, str(OUT_DIR))
from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn, lambda_star                            # noqa: E402
from layer_readout_probe import lane_z                              # noqa: E402
from miss_anatomy import rank_in, classify                          # noqa: E402

CUTOFF = sys.argv[1] if len(sys.argv) > 1 else '2026-05-11'
LAM = 0.65
LANES = ('maxsim', 'sit', 'idf', 'pick', 'enc')
REPORT = OUT_DIR / 'corpus_v2_eval.md'


def ageband(a):
    if a is None:
        return '?'
    for lbl, hi in (('≤1d', 1), ('1-7d', 7), ('7-21d', 21), ('21-45d', 45)):
        if a < hi:
            return lbl
    return '>45d'


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

    recs = []
    for t in idx['turns']:
        key = '%s/%d/%d' % tuple(t['key'])
        v = verds.get(key)
        b = bundles.get(key)
        if not v or v['verdict'] != 'valid' or not b:
            continue
        if (b['ts'] or '') < CUTOFF:
            continue
        tt = Turn(t, fields, S)
        if tt.gr < 0 or tt.mh is None or np.isnan(tt.fields[0]).all():
            continue
        gr = tt.gr
        # composed-field ranks
        f0r = rank_in(tt.fields[0], gr)
        mhr = rank_in(tt.mh, gr)
        # op0 lane ranks (the query-side lanes — door-1 substrate)
        L = lanes_mm[t['row']].astype(np.float32)      # [slots × lanes × n]
        mx = L[S['op0'], LANES.index('maxsim')]
        alive = np.isfinite(mx)
        lane_ranks = {}
        for li, ln in enumerate(LANES):
            z = lane_z(L[S['op0'], li], ln, alive, n_nodes)
            lane_ranks[ln] = rank_in(z, gr)
        # λ-mix rank
        rk = lambda_star(zn(tt.fields[0]), zn(tt.mh), gr, grid=np.array([LAM]))
        mix = min(rk.values()) if rk else None
        # best reachable across all held fields/lanes
        cand = [r for r in ([f0r, mhr] + list(lane_ranks.values()))
                if r is not None]
        best = min(cand) if cand else None
        best_lane = None
        allr = dict(lane_ranks, F0=f0r, M_h=mhr)
        if best is not None:
            best_lane = min((k for k, r in allr.items() if r == best),
                            key=lambda k: allr[k])
        recs.append({
            'key': key, 'stratum': v['stratum'], 'gtype': (b['gold'] or {}).get('type'),
            'age': (b['gold'] or {}).get('age_days'), 'strong': (b.get('telemetry') or {}).get('strong'),
            'mix': mix, 'f0': f0r, 'mh': mhr, 'lanes': lane_ranks,
            'best': best, 'best_lane': best_lane,
            'hit': mix is not None and mix <= 5,
            'missclass': classify(best) if best is not None else 'BARELY',
        })

    N = len(recs)
    door = lambda r: 'door-1' if r['stratum'] == 'cue' else 'door-2'
    L = ['# Corpus-v2 eval — decomposed honest baselines (cutoff %s)' % CUTOFF,
         '', 'Valid golds only, turn-date ≥ %s. n=%d. Static λ=%.2f mix.'
         % (CUTOFF, N, LAM), '']

    # ---- 1. reach by door × stratum ----
    L += ['## 1. Reach — by door and stratum', '',
          '| slice | n | reach@5 | reach@25 | median mix |',
          '|---|---|---|---|---|']
    def reach_row(lbl, sub):
        m = [r['mix'] for r in sub if r['mix'] is not None]
        if not m:
            return
        L.append('| %s | %d | %.0f%% | %.0f%% | %.0f |'
                 % (lbl, len(sub), 100*sum(x <= 5 for x in m)/len(m),
                    100*sum(x <= 25 for x in m)/len(m), np.median(m)))
    for st in ('cue', 'window', 'session'):
        reach_row(st, [r for r in recs if r['stratum'] == st])
    reach_row('**DOOR-1**', [r for r in recs if door(r) == 'door-1'])
    reach_row('**DOOR-2**', [r for r in recs if door(r) == 'door-2'])
    L.append('')

    # ---- 2. WHAT GOT IN — hit attribution ----
    hits = [r for r in recs if r['hit']]
    L += ['## 2. What got in — carrier lane on the %d HITS' % len(hits), '',
          'Which held field/lane ranks the gold best (what recall is leaning '
          'on when it succeeds).', '',
          '| carrier | door-1 | door-2 | total |', '|---|---|---|---|']
    car = defaultdict(lambda: [0, 0])
    for r in hits:
        car[r['best_lane'] or '?'][0 if door(r) == 'door-1' else 1] += 1
    for k in sorted(car, key=lambda k: -sum(car[k])):
        L.append('| %s | %d | %d | %d |' % (k, car[k][0], car[k][1], sum(car[k])))
    L.append('')
    # hit rate by gold type + age
    L += ['**Hit@5 by gold type (n≥15):**', '',
          '| gtype | n | hit@5 |', '|---|---|---|']
    byt = defaultdict(list)
    for r in recs:
        byt[r['gtype']].append(r)
    for t, sub in sorted(byt.items(), key=lambda x: -len(x[1])):
        if len(sub) >= 15:
            L.append('| %s | %d | %.0f%% |'
                     % (t, len(sub), 100*sum(x['hit'] for x in sub)/len(sub)))
    L += ['', '**Hit@5 by gold age:**', '', '| age | n | hit@5 |', '|---|---|---|']
    bya = defaultdict(list)
    for r in recs:
        bya[ageband(r['age'])].append(r)
    for a in ('≤1d', '1-7d', '7-21d', '21-45d', '>45d'):
        if a in bya:
            sub = bya[a]
            L.append('| %s | %d | %.0f%% |'
                     % (a, len(sub), 100*sum(x['hit'] for x in sub)/len(sub)))
    L.append('')

    # ---- 3. WHAT DIDN'T — miss anatomy on valids ----
    miss = [r for r in recs if not r['hit']]
    L += ['## 3. What didn\'t — miss anatomy on the %d MISSES' % len(miss), '',
          'Best rank the gold achieves in ANY held field/lane → is it a '
          'RANKING problem (reachable, machinery can fix) or an ENCODING '
          'problem (buried everywhere)?', '',
          '| miss class | door-1 | door-2 | total | meaning |',
          '|---|---|---|---|---|']
    meaning = {'REACHABLE': 'best≤5 — remix/compose problem',
               'ALMOST': 'best≤25 — selection problem',
               'BURIED': 'best≤100 — calibration/crowding',
               'BARELY': 'best>100 — encode-side, no signal'}
    mc = defaultdict(lambda: [0, 0])
    for r in miss:
        mc[r['missclass']][0 if door(r) == 'door-1' else 1] += 1
    for cls in ('REACHABLE', 'ALMOST', 'BURIED', 'BARELY'):
        L.append('| %s | %d | %d | %d | %s |'
                 % (cls, mc[cls][0], mc[cls][1], sum(mc[cls]), meaning[cls]))
    L.append('')
    # for REACHABLE/ALMOST misses: which lane WOULD have reached (the recoverable map)
    reach_miss = [r for r in miss if r['best'] is not None and r['best'] <= 25]
    L += ['**Recoverable misses (best≤25): which lane holds the gold '
          '(n=%d)**' % len(reach_miss), '',
          '| holding lane | count | share |', '|---|---|---|']
    hold = Counter(r['best_lane'] for r in reach_miss)
    for k, c in hold.most_common():
        L.append('| %s | %d | %.0f%% |' % (k, c, 100*c/max(1, len(reach_miss))))
    L.append('')
    # BARELY/BURIED by door — the encoding-limited tail
    L += ['**Encode-limited tail (best>100 = BARELY) by door & stratum:**', '',
          '| stratum | BARELY n | share of stratum misses |', '|---|---|---|']
    for st in ('cue', 'window', 'session'):
        sm = [r for r in miss if r['stratum'] == st]
        bar = [r for r in sm if r['missclass'] == 'BARELY']
        if sm:
            L.append('| %s | %d | %.0f%% |'
                     % (st, len(bar), 100*len(bar)/len(sm)))
    L.append('')

    # ---- 4. door-2 ceiling verdict ----
    d2 = [r for r in recs if door(r) == 'door-2']
    d2m = [r for r in d2 if not r['hit']]
    d2_reach = sum(1 for r in d2m if r['best'] is not None and r['best'] <= 25)
    d2_barely = sum(1 for r in d2m if r['missclass'] == 'BARELY')
    L += ['## 4. Door-2 ceiling verdict', '',
          '- door-2 valid golds: %d · hit@5 %.0f%% · misses %d'
          % (len(d2), 100*sum(r['hit'] for r in d2)/max(1, len(d2)), len(d2m)),
          '- of the misses: %d (%.0f%%) are REACHABLE/ALMOST (best≤25 — a '
          'ranking/composition fix recovers them), %d (%.0f%%) BARELY '
          '(encode-side, machinery can\'t help)'
          % (d2_reach, 100*d2_reach/max(1, len(d2m)),
             d2_barely, 100*d2_barely/max(1, len(d2m))),
          '- **read:** door-2 is %s'
          % ('RANKING-limited — worth building the running field'
             if d2_reach >= d2_barely else
             'ENCODE-limited — fix encoding before machinery'), '']

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
