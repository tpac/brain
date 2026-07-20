"""Q1 reach leg — winner config vs K0 on gold-24 (§20.5 pre-declared
aggregate, reach half).

Runs AFTER the shuffle control holds. Composes the rank-leg winner's moment
shape over the SAME cue fields the base-parity check validated (reach_leg
cue_fields: production engine lanes, as_of=cutoff, production episodic),
through the SAME composer (q1_sweep.compose, masked z). Reports
need-reach @5/@25, winner vs K0, per cue.

Pass criterion (pre-committed): +>=2 needs @25 on gold-24 (AND LongMemEval
beyond variance — that leg trails, flagged, not silently skipped).

Run:  ./dev python3 eval/laf/walker/reach_winner.py <config-name>
"""
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import LafV1Engine, _unit                  # noqa: E402
from q1_sweep import GAINS, compose, stack_messages, weights, configs  # noqa: E402
from reach_leg import load_cues, cue_fields, rank_rows, needs_reach    # noqa: E402

REPORT = OUT_DIR / 'reach_winner.md'


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if not name:
        raise SystemExit('usage: reach_winner.py <config-name>')
    cfg = next((c for c in configs() if c['name'] == name), None)
    if cfg is None:
        raise SystemExit('unknown config %r' % name)
    if cfg['me'][0] != 'off':
        raise SystemExit('gold cues carry no session fatigue replay — run '
                         'the me0 twin of this shape')
    k0 = configs()[0]
    cues = load_cues()

    import servers.embedder as embedder
    from tests.isolated_brain import IsolatedBrain
    per_cue = []
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
            row = {}
            for label, c in (('k0', k0), ('win', cfg)):
                w = weights(c)
                mats, ww = {}, None
                for ln in GAINS:
                    mats[ln], ww = stack_messages(op[ln], an[ln], w, c)
                s = compose(mats, ww, c, n, mask=node_mask)
                order = rank_rows(s, node_mask)
                for at in (5, 25):
                    hit, tot = needs_reach(eng, order, cue['needs'], at)
                    row['%s_%d' % (label, at)] = (hit, tot)
            per_cue.append((cue['cue_id'], row))

    lines = ['# reach_winner — gold-24, %s vs K0 (§20.5 reach leg)' % name,
             '', '| cue | K0 @5 | K0 @25 | win @5 | win @25 |', '|---|---|---|---|---|']
    tot = {k: [0, 0] for k in ('k0_5', 'k0_25', 'win_5', 'win_25')}
    for cid, row in per_cue:
        for k, (h, t) in row.items():
            tot[k][0] += h
            tot[k][1] += t
        lines.append('| %s | %d/%d | %d/%d | %d/%d | %d/%d |'
                     % (cid, *row['k0_5'], *row['k0_25'],
                        *row['win_5'], *row['win_25']))
    lines.append('')
    d25 = tot['win_25'][0] - tot['k0_25'][0]
    d5 = tot['win_5'][0] - tot['k0_5'][0]
    lines.append('**Totals:** K0 %d/%d @5, %d/%d @25 → winner %d/%d @5, '
                 '%d/%d @25 · **Δ@5 %+d needs · Δ@25 %+d needs**'
                 % (*tot['k0_5'], *tot['k0_25'], *tot['win_5'],
                    *tot['win_25'], d5, d25))
    lines.append('')
    lines.append('**Pre-committed reach criterion (+≥2 needs @25): %s**'
                 % ('MET' if d25 >= 2 else 'NOT MET — no ship on reach, '
                    'regardless of the rank win'))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
