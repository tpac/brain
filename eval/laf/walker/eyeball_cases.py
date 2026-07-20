"""Eyeball cases — Tom's qualitative read (2026-07-17).

For val turns with a clear quality signal (max soft_max >= MIN_SOFT,
pool >= 8), rank the pool under K0 (prompt-only) and the winner moment
arm (K1-exp0.5 zsum op+anchor). Target node = the pool's highest-soft
candidate (what the actual response drew on most).

  SUCCESS  = moment lifts the target into the top-3 from K0-deep
  FAILURE-A = moment BURIES a target K0 had high (moment hurt)
  FAILURE-B = both arms miss it (rank > 10 in both — the residual)

Prints the conversational context (previous anchor + current op text)
with top-5 titles per arm, soft values, ✓ = Haiku-selected that turn.

Run:  ./dev python3 eval/laf/walker/eyeball_cases.py
Out:  eyeball_cases.md
"""
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_walker, open_brain_ro

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from q1_sweep import load, gate_provenance, configs, score_turn, \
    weights                                                        # noqa: E402

WINNER = 'K1-exp0.5-turnsum-zsum-opanchor-me0'
MIN_SOFT = 0.80
MIN_POOL = 8
N_PER_CLASS = 4
REPORT = OUT_DIR / 'eyeball_cases.md'


def ranks_of(s):
    order = np.argsort(-s)
    r = np.empty(len(s), dtype=int)
    r[order] = np.arange(1, len(s) + 1)
    return r, order


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    texts = {}
    for sess, epoch, seq, op_t, an_t in walker.execute(
            "SELECT session_id, epoch, seq, op_text, anchor_text "
            "FROM turns"):
        texts[(sess, epoch, seq)] = (op_t or '', an_t or '')
    walker.close()

    cfg_k0 = configs()[0]
    cfg_w = next(c for c in configs() if c['name'] == WINNER)
    w0, ww = weights(cfg_k0), weights(cfg_w)

    cases = []
    for td in turns:
        if not td.val or len(td.cands) < MIN_POOL:
            continue
        if not np.isfinite(td.soft).any():
            continue
        tgt = int(np.nanargmax(td.soft))
        if td.soft[tgt] < MIN_SOFT:
            continue
        s0 = score_turn(td, cfg_k0, w0)
        s1 = score_turn(td, cfg_w, ww)
        r0, o0 = ranks_of(s0)
        r1, o1 = ranks_of(s1)
        cases.append((td, tgt, int(r0[tgt]), int(r1[tgt]), s0, s1,
                      o0, o1))

    succ = sorted([c for c in cases if c[3] <= 3 and c[2] >= 6],
                  key=lambda c: -(c[2] - c[3]))[:N_PER_CLASS]
    hurt = sorted([c for c in cases if c[2] <= 3 and c[3] >= 6],
                  key=lambda c: -(c[3] - c[2]))[:N_PER_CLASS]
    miss = sorted([c for c in cases if c[2] > 10 and c[3] > 10],
                  key=lambda c: -(c[0].soft[c[1]]))[:N_PER_CLASS]

    ids = {nid for c in succ + hurt + miss for nid in c[0].cands}
    b = open_brain_ro()
    titles = dict(b.execute(
        'SELECT id, title FROM nodes WHERE id IN (%s)'
        % ','.join('?' * len(ids)), list(ids)))
    b.close()

    lines = ['# eyeball_cases — K0 vs moment (winner K1), val turns',
             '',
             '- target = the pool node my actual response drew on most '
             '(highest soft_max, ≥ %.2f)' % MIN_SOFT,
             '- ✓ = Haiku selected that turn · [T] = the target node', '']

    def emit(tag, c):
        td, tgt, rk0, rk1, s0, s1, o0, o1 = c
        sess, epoch, seq = td.key
        op_t, _ = texts.get(td.key, ('', ''))
        # previous turn's anchor text = the j1-anchor cue
        prev_an = ''
        for pseq in range(seq - 1, max(seq - 6, -1), -1):
            t = texts.get((sess, epoch, pseq))
            if t and t[1]:
                prev_an = t[1]
                break
        lines.append('## %s · %s/%s/%s · target K0 #%d → moment #%d'
                     % (tag, sess[:8], epoch, seq, rk0, rk1))
        lines.append('')
        lines.append('**my previous response (j1-anchor cue):** %s'
                     % (prev_an[:300].replace('\n', ' ') or '(none)'))
        lines.append('')
        lines.append('**operator message (j0 cue):** %s'
                     % op_t[:300].replace('\n', ' '))
        lines.append('')
        lines.append('**target [T]:** %s (soft %.2f)'
                     % ((titles.get(td.cands[tgt]) or td.cands[tgt])[:90],
                        td.soft[tgt]))
        lines.append('')
        lines.append('| # | K0 (prompt only) | moment (K1) |')
        lines.append('|---|---|---|')

        def fmt(i):
            t = (titles.get(td.cands[i]) or td.cands[i])[:58]
            m = ''
            if i == tgt:
                m += ' **[T]**'
            if td.sel[i]:
                m += ' ✓'
            sv = (' s=%.2f' % td.soft[i]
                  if np.isfinite(td.soft[i]) else '')
            return t + m + sv
        for r in range(5):
            lines.append('| %d | %s | %s |' % (r + 1, fmt(o0[r]),
                                               fmt(o1[r])))
        lines.append('')

    lines.append('# ── SUCCESSES — moment lifts the needed node ──')
    lines.append('')
    for c in succ:
        emit('SUCCESS', c)
    lines.append('# ── FAILURES A — moment BURIES what K0 had ──')
    lines.append('')
    for c in hurt:
        emit('MOMENT-HURT', c)
    lines.append('# ── FAILURES B — both arms miss it ──')
    lines.append('')
    for c in miss:
        emit('BOTH-MISS', c)

    print('pool: %d candidate cases · success %d · hurt %d · both-miss %d'
          % (len(cases), len(succ), len(hurt), len(miss)))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('wrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
