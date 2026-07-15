"""H1 packet — §20.6 checkpoint 1: the moment-fidelity read.

Samples ~17 reconstructed moments STRATIFIED across the classes where a wrong
reconstruction would be silent (untraced-legacy micro-turns, superseded steering
turns, epoch seams, no-recall turns) plus normal controls across months, and
renders each as the walker sees it: the turn + its preceding stack, roles,
flags, timing. Tom reads them against his memory of the conversations — the one
check no math can do.

Deterministic seed → rerunnable; emits eval/laf/walker/h1_moments.md.
Run:  ./dev python3 eval/laf/walker/h1_packet.py
"""
import random
import sys
from pathlib import Path

from walker_db import open_walker, WALKER_DIR

RNG = random.Random(11)
STACK_DEPTH = 4          # previous turns rendered per moment
OP_CAP, ANCHOR_CAP = 280, 200

STRATA = [
    ('untraced legacy micro-turn (pre-06-08 s0 loss)', 3,
     "SELECT session_id, epoch, seq FROM turns WHERE flags LIKE '%untraced_legacy%' ORDER BY ts"),
    ('superseded turn (steering/interrupt — later turn shares its stop)', 2,
     "SELECT session_id, epoch, seq FROM turns WHERE flags LIKE '%superseded%'"
     " AND flags NOT LIKE '%untraced_legacy%' ORDER BY ts"),
    ('epoch seam (first turns of epoch >= 1)', 3,
     "SELECT session_id, epoch, seq FROM turns WHERE epoch >= 1 AND seq <= 1 AND labeled=1 ORDER BY ts"),
    ('no-recall turn (register_only or hook miss)', 2,
     "SELECT session_id, epoch, seq FROM turns WHERE flags LIKE '%no_recall%' ORDER BY ts"),
    ('normal labeled, deep history (seq >= 8)', 4,
     "SELECT session_id, epoch, seq FROM turns WHERE labeled=1 AND seq >= 8 AND flags='[]' ORDER BY ts"),
    ('normal labeled, session opening (seq <= 2)', 3,
     "SELECT session_id, epoch, seq FROM turns WHERE labeled=1 AND seq <= 2 AND epoch=0 AND flags='[]' ORDER BY ts"),
]


def clip(text, n):
    text = (text or '').replace('\n', ' ⏎ ').strip()
    return text[:n] + ('…' if len(text) > n else '')


def render_moment(walker, idx, label, sess, epoch, seq):
    rows = walker.execute(
        "SELECT seq, stop, ts, op_text, anchor_text, flags, labeled FROM turns "
        "WHERE session_id=? AND epoch=? AND seq BETWEEN ? AND ? ORDER BY seq",
        (sess, epoch, max(0, seq - STACK_DEPTH), seq)).fetchall()
    date = rows[-1][2][:16].replace('T', ' ') if rows and rows[-1][2] else '?'
    out = ['### Moment %d — %s' % (idx, label),
           '`session %s · epoch %d · seq %d · %s`' % (sess[:8], epoch, seq, date), '']
    for r_seq, r_stop, r_ts, op, anchor, flags, labeled in rows:
        j = seq - r_seq
        marker = '**→ THE MOMENT**' if j == 0 else 'j=%d' % j
        flag_note = '' if flags == '[]' else '  ⚠ %s' % flags
        out.append('- %s%s' % (marker, flag_note))
        out.append('  - **Tom:** %s' % clip(op, OP_CAP))
        if anchor:
            out.append('  - *Anchor:* %s' % clip(anchor, ANCHOR_CAP))
    out.append('')
    out.append('**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________')
    out.append('')
    out.append('---')
    out.append('')
    return out


def main():
    walker = open_walker()
    lines = [
        '# H1 — moment-fidelity read (walker Stage 1)', '',
        'Each block is a moment exactly as the walker reconstructs it: the turn (→) plus',
        'the previous %d turns of its epoch, oldest first. Read against your memory:' % STACK_DEPTH,
        'are these the right turns, in the right order, from the right conversation?',
        'Untraced-legacy turns show the prompt recovered from the recall trace (no',
        'Anchor response ever existed). Superseded turns never got their own Stop —',
        'a steering message / interrupt / notification landed first; the combined',
        'response attaches to the LAST turn of the stop. Epoch seams are',
        'post-resume/compaction restarts — the stack deliberately does NOT cross them.', '',
        '---', '']
    idx = 0
    for label, n, sql in STRATA:
        pool = walker.execute(sql).fetchall()
        for sess, epoch, seq in RNG.sample(pool, min(n, len(pool))):
            idx += 1
            lines.extend(render_moment(walker, idx, label, sess, epoch, seq))
    (WALKER_DIR / 'h1_moments.md').write_text('\n'.join(lines) + '\n')
    walker.close()
    print('h1_moments.md written — %d moments' % idx)
    return 0


if __name__ == '__main__':
    sys.exit(main())
