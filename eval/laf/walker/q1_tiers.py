"""Q1 tier-graded ranking quality — gold/silver top-k, K0 vs winner (Tom).

Judge-independent quality check: on the 24 gold cues (as_of engine path,
same validated cue fields), where do the tiered nodes actually LAND —
top-1 / top-5 / top-25 — under K0 vs the rank-leg winner? Unlike the rank
leg's pick-AUC (which carries pick-lane label echo), the tiers were minted
by blind outcome-anchored judges — no Haiku in the loop.

Reports per tier (gold_plus / gold / silver_plus / silver): node-level
placements aggregated over cues, plus median rank shift.

Run:  ./dev python3 eval/laf/walker/q1_tiers.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, GOLD_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import LafV1Engine, _unit                  # noqa: E402
from q1_sweep import GAINS, compose, stack_messages, weights, configs  # noqa: E402
from reach_leg import load_cues, cue_fields, rank_rows             # noqa: E402

WINNER = 'K1-exp0.5-turnsum-zsum-opanchor-me0'
TIERS = ('gold_plus', 'gold', 'silver_plus', 'silver')
OUT = OUT_DIR / 'q1_tiers.json'


def main():
    cfg = next(c for c in configs() if c['name'] == WINNER)
    k0 = configs()[0]
    cues = load_cues()
    gold = json.loads((GOLD_DIR / 'frozen_gold_24.json').read_text())

    import servers.embedder as embedder
    from tests.isolated_brain import IsolatedBrain
    placements = {t: {'k0': [], 'win': []} for t in TIERS}
    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_titles(env.brain)
            eng._refresh_traces(env.brain)
        eng._brain_ref = env.brain
        n = eng._n
        for cue in cues:
            q0 = _unit(embedder.embed_query(cue['text']))
            node_mask, trace_mask = eng._asof_masks(cue['cutoff'], n)
            op, an = cue_fields(eng, trace_mask, cue, q0)
            tiers = gold[cue['cue_id']]['tiers']
            for label, c in (('k0', k0), ('win', cfg)):
                w = weights(c)
                mats, ww = {}, None
                for ln in GAINS:
                    mats[ln], ww = stack_messages(op[ln], an[ln], w, c)
                s = compose(mats, ww, c, n, mask=node_mask)
                order = rank_rows(s, node_mask)
                rank_of = {eng._master[r][:8]: i + 1
                           for i, r in enumerate(order)}
                for t in TIERS:
                    for it in tiers.get(t, []):
                        placements[t][label].append(
                            rank_of.get(it['node_id']))   # None = not in field

    lines = ['# q1_tiers — gold/silver placement, K0 vs winner', '',
             '| tier | n | arm | top-1 | top-5 | top-25 | median rank |',
             '|---|---|---|---|---|---|---|']
    data = {}
    for t in TIERS:
        for label in ('k0', 'win'):
            rs = placements[t][label]
            known = [r for r in rs if r is not None]
            row = {'n': len(rs), 'unresolved': len(rs) - len(known),
                   'top1': sum(1 for r in known if r <= 1),
                   'top5': sum(1 for r in known if r <= 5),
                   'top25': sum(1 for r in known if r <= 25),
                   'median_rank': float(np.median(known)) if known else None}
            data.setdefault(t, {})[label] = row
            lines.append('| %s | %d | %s | %d | %d | %d | %s |'
                         % (t, row['n'], label, row['top1'], row['top5'],
                            row['top25'],
                            '%.0f' % row['median_rank']
                            if row['median_rank'] else '—'))
    OUT.write_text(json.dumps(data, indent=1))
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
