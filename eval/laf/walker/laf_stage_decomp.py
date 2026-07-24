"""Where is the loss — field ORDERING or surface SELECTION?

Every probe in this arc measured the FIELD (gold's rank among 7684 nodes).
But production is two stages: field ranks -> ~25-candidate pool -> Haiku
picks 3-5 -> Anchor sees it. reach@5 is a proxy for stage 1 and says nothing
about stage 2, which I had never measured.

WHAT THIS CORPUS CAN AND CANNOT ANSWER (checked, not assumed):
  * The gold is in the pool BY CONSTRUCTION -- gold_i indexes cand_rows, so
    P(gold in pool) == 1 here. This corpus cannot measure pool-ENTRY failure,
    and the 78% reach@25 figure is a COUNTERFACTUAL about where a LAF field
    cut at 25 would land, not production's actual pool.
  * P(gold PICKED | gold in pool) IS measurable: `sel` holds the real Haiku
    picks for the real turn. That is the selection loss, and it is the gap
    this probe exists to size.

CAUSAL CAVEAT (stated because it bounds the conclusion): the pool and `sel`
were produced by the CHAMPION path, so Haiku saw the champion's ordering, not
the LAF ordering computed here. Relating LAF pool-rank to `sel` therefore
measures ASSOCIATION -- "is the field's ordering aligned with what Haiku
picked" -- not causation. The causal question (does reordering change picks?)
needs the frame_replay A/B on the live path; nothing here substitutes for it.

Still decisive for direction: if picks concentrate hard at the top of the
field ordering, ordering is worth improving; if picks are flat across field
rank, the field is not what drives selection and the lever is the surface.

Read-only. Run:  ./dev python3 eval/laf/walker/laf_stage_decomp.py
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from walker_db import OUT_DIR, open_brain_ro

sys.path.append(str(OUT_DIR))
from lambda_probe import zn                                          # noqa: E402
from miss_anatomy import rank_in                                     # noqa: E402
import laf_lane_audit as A                                          # noqa: E402
import laf_gate_audit as G                                          # noqa: E402

REPORT = OUT_DIR / 'laf_stage_decomp.md'


def iso(ts):
    try:
        return datetime.fromisoformat((ts or '').replace('Z', '+00:00'))
    except Exception:
        return None


def main():
    turns, n = A.build()
    P = G.prep(turns)
    tt = [p['t'] for p in P]
    N = len(P)
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    master = idx['master']
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}

    b = open_brain_ro()
    meta = {}
    for nid, typ, created in b.execute(
            'SELECT id, type, created_at FROM nodes'):
        meta[nid] = (typ, created)
    b.close()

    # per turn: field score over the pool, gold's pool-rank, was it picked
    rows = []
    for i, t in enumerate(tt):
        cr = t['cand_rows']
        gi = int(np.flatnonzero(cr == t['gr'])[0]) if (cr == t['gr']).any() else None
        if gi is None:
            continue
        # field score (shipped gains) restricted to the pool
        f0 = np.zeros_like(t['enrichment_z'])
        for ln in A.LANES:
            f0 = f0 + A.GAINS[ln] * t['zl'][ln]
        f0[~t['alive']] = np.nan
        mix = A.LAM * zn(f0) + (1 - A.LAM) * zn(t['mh'])
        ok = cr >= 0
        pool_scores = np.where(np.isfinite(mix[cr[ok]]), mix[cr[ok]], -np.inf)
        gscore = mix[t['gr']]
        if not np.isfinite(gscore):
            continue
        greater = int((pool_scores > gscore).sum())
        ties = int((pool_scores == gscore).sum())
        pool_rank = greater + (ties - 1) / 2.0 + 1        # tie-fair, in-pool
        nid = master[t['gr']]
        typ, created = meta.get(nid, (None, None))
        bd = bundles.get(t['key'])
        tdt = iso(bd['ts']) if bd else None
        cdt = iso(created)
        age = (tdt - cdt).days if (tdt and cdt and tdt >= cdt) else None
        rows.append({
            'picked': bool(t['sel'][gi]) if len(t['sel']) > gi else False,
            'pool_rank': pool_rank, 'pool_size': int(ok.sum()),
            'n_picks': int(t['sel'].sum()),
            'field_rank': rank_in(mix, t['gr']),
            'soft': float(t['soft'][gi]) if np.isfinite(t['soft'][gi]) else None,
            'stratum': t['stratum'], 'type': typ, 'age': age,
            'cur_maxz': t['cur_maxz'],
        })
    M = len(rows)
    picked = [r for r in rows if r['picked']]

    L = ['# Where is the loss — field ORDERING or surface SELECTION?', '',
         'n=%d clean valids ≥%s · pool median %d candidates · Haiku picks '
         'median %d' % (M, A.CUTOFF,
                        int(np.median([r['pool_size'] for r in rows])),
                        int(np.median([r['n_picks'] for r in rows]))),
         '',
         '**What this corpus can answer.** The gold is in the pool BY '
         'CONSTRUCTION (gold_i indexes cand_rows), so P(gold in pool)=1 and '
         'pool-ENTRY failure is NOT measurable here — the 78% reach@25 quoted '
         'earlier is a counterfactual about a LAF field cut at 25, not '
         "production's real pool. What IS measurable is P(gold PICKED | in "
         'pool), from the real `sel`.', '',
         '**Causal caveat.** The pool and `sel` came from the CHAMPION path, '
         'so Haiku saw the champion ordering, not the LAF ordering computed '
         'here. Field-rank↔pick relations below are ASSOCIATION, not '
         'causation; the causal test is the frame_replay A/B.', '']

    # ══ THE HEADLINE ══
    p_picked = 100.0 * len(picked) / M
    L += ['## The selection loss', '',
          '| quantity | value |', '|---|---|',
          '| P(gold picked \\| gold in pool) | **%.1f%%** |' % p_picked,
          '| chance rate (n_picks / pool_size) | %.1f%% |'
          % (100.0 * np.mean([r['n_picks'] / max(1, r['pool_size'])
                              for r in rows])),
          '| gold in field-top-2 of pool | %.1f%% |'
          % (100.0 * np.mean([r['pool_rank'] <= 2 for r in rows])),
          '| gold in field-top-5 of pool | %.1f%% |'
          % (100.0 * np.mean([r['pool_rank'] <= 5 for r in rows])), '',
          '- Haiku selects ~%.1f of ~%.0f candidates, so even a perfect field '
          'ordering leaves a hard selection: the gold must land in those few '
          'picks.' % (np.mean([r['n_picks'] for r in rows]),
                      np.mean([r['pool_size'] for r in rows])), '']

    # ══ THE DECISIVE TABLE: P(picked) vs field rank within pool ══
    L += ['## Does the field ordering align with what Haiku picks?', '',
          'If P(picked) rises steeply with the field\'s in-pool rank, ordering '
          'is worth improving. If it is flat, the field is not what drives '
          'selection and the lever is the surface.', '',
          '| gold field-rank in pool | turns | picked | P(picked) |',
          '|---|---|---|---|']
    bands = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 25), (26, 99)]
    for lo, hi in bands:
        sub = [r for r in rows if lo <= r['pool_rank'] <= hi]
        if not sub:
            continue
        k = sum(1 for r in sub if r['picked'])
        L.append('| %s | %d | %d | %.0f%% |'
                 % ('%d' % lo if lo == hi else '%d–%d' % (lo, hi),
                    len(sub), k, 100.0 * k / len(sub)))
    L.append('')

    # the lever estimate
    top1 = [r for r in rows if r['pool_rank'] <= 1]
    p_top1 = 100.0 * np.mean([r['picked'] for r in top1]) if top1 else 0.0
    L += ['- **P(picked | field ranks gold #1 in pool) = %.0f%%** vs overall '
          '%.0f%%. If perfect field ordering were achievable, that difference '
          '(%+.0fpp) bounds what reordering could buy — an upper bound, since '
          'Haiku saw a different ordering.'
          % (p_top1, p_picked, p_top1 - p_picked), '']

    # ══ what distinguishes picked from unpicked ══
    def split(title, keyfn, min_n=10):
        buck = defaultdict(lambda: [0, 0])
        for r in rows:
            k = keyfn(r)
            if k is None:
                continue
            buck[k][0] += int(r['picked'])
            buck[k][1] += 1
        out = ['### %s' % title, '',
               '| bucket | n | picked | P(picked) |', '|---|---|---|---|']
        for k, (a, c) in sorted(buck.items(), key=lambda x: -x[1][1]):
            if c < min_n:
                continue
            out.append('| %s | %d | %d | %.0f%% |' % (k, c, a, 100.0 * a / c))
        return out + ['']

    L += ['## What distinguishes a picked gold from an unpicked one', '']
    L += split('By stratum', lambda r: r['stratum'])
    L += split('By gold type', lambda r: r['type'])
    L += split('By age at recall', lambda r: (
        None if r['age'] is None else
        ('0–7d' if r['age'] <= 7 else '8–30d' if r['age'] <= 30
         else '31–90d' if r['age'] <= 90 else '90d+')))
    L += split('By soft-usage label', lambda r: (
        None if r['soft'] is None else
        ('soft ≥0.7' if r['soft'] >= 0.7 else 'soft 0.3–0.7'
         if r['soft'] >= 0.3 else 'soft <0.3')))
    L += split('By cue sharpness', lambda r: (
        'Q1 (flat)' if r['cur_maxz'] < np.percentile(
            [x['cur_maxz'] for x in rows], 25)
        else 'Q4 (sharp)' if r['cur_maxz'] > np.percentile(
            [x['cur_maxz'] for x in rows], 75) else 'Q2–Q3'))

    # ══ the unpicked, ranked well by the field: the surface's own misses ══
    good_rank_unpicked = [r for r in rows
                          if r['pool_rank'] <= 5 and not r['picked']]
    L += ['## The surface\'s own misses (field ranked gold ≤5 in pool, not picked)',
          '', '- **%d turns (%.0f%% of all)** — the field did its job and the '
          'gold still was not selected. This population is what surface work '
          'would target; it is invisible to any reach metric.'
          % (len(good_rank_unpicked), 100.0 * len(good_rank_unpicked) / M), '',
          '| slice | count |', '|---|---|']
    for k, v in Counter(r['type'] for r in good_rank_unpicked).most_common(8):
        L.append('| type=%s | %d |' % (k, v))
    for k, v in Counter(r['stratum'] for r in good_rank_unpicked).most_common():
        L.append('| stratum=%s | %d |' % (k, v))
    L.append('')

    # ══ verdict ══
    steep = p_top1 - p_picked
    L += ['## Verdict', '',
          '- selection loss: gold is in the pool 100%% of the time (by '
          'construction) yet picked only **%.1f%%** of the time.' % p_picked,
          '- field ordering is %s with picks: P(picked) goes from %.0f%% at '
          'in-pool rank #1 to %.0f%% at ranks 11–25.'
          % ('strongly associated' if steep > 15 else 'weakly associated',
             p_top1,
             100.0 * np.mean([r['picked'] for r in rows
                              if 11 <= r['pool_rank'] <= 25] or [0])),
          '- **direction**: %s'
          % ('field ordering still carries real signal into selection — '
             'improving in-pool rank is worth it, AND the surface has its own '
             'independent loss (see the misses table).' if steep > 15 else
             'field ordering barely moves selection — the lever is the '
             'surface (prompt / render / how many it picks), not more lanes.'),
          '']

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
