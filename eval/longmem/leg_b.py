"""Leg B — the engine path (§20.18): turn-by-turn REAL brain.recall on a
frozen pooled corpus, as_of at each cue, arms as registered config JSONs.

Per arm: a FRESH copy of the corpus brain + a fresh Brain instance (no
cross-arm cache/fatigue state), the arm's gain table registered as the
`recall_laf` interaction (the exact switch Stage-3 would flip), then one
recall per labeled walker cue with `as_of` = the cue's O-row timestamp and
`query` = the build's own stored 500-char recall query.

Arms: A0 (production defaults, no override), A0f / A1 / A1a from
arm_tables.py (frozen definitive-fit weights — never refit here).
Controls run through the same wired path:
  C1  shuffle — A1 gains with a DONOR session id (previous session in date
      order, cyclic): the stack is another session's history, positions
      preserved. Must NOT beat A0f.
  C2  identity — on session-first cues (empty stack) A1 must equal A0f
      row-level: same ids, same scores (<1e-9). Any divergence = wiring FAIL.
Instruments:
  soft_r per arm — engine score vs walker soft_max over (cue, candidate)
      rows (G5: the composition transfer across z-universes, measured).
  latency per arm — informational, feeds §20.17 G4.

Run:  ./dev python3 eval/longmem/leg_b.py --corpus 74aea3 [--arms A0,A0f,A1,A1a,C1]
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'laf' / 'walker'))

from corpus import corpus_dir  # noqa: E402

os.environ['BRAIN_RECALL_VARIANT'] = 'laf_v1'   # before any servers import

DEFAULT_ARMS = ('A0', 'A0f', 'A1', 'A1a', 'C1')
RECALL_LIMIT = 25


def load_cues(walker_path, cue_side='op', logs_db=None):
    """Cues in corpus order: (session_id, epoch, seq, ts, query).

    cue_side='op' (default): each labeled turn's operator query at its
    O-row ts — the production recall position.

    cue_side='stop' (endo, §20.18 exploratory): recall fires at MY stop —
    query = Anchor's own response to turn t (500 cap, the production query
    cap), as_of = that assistant_message's trace ts + 1ms (turn t fully in
    the past, turn t+1 not yet begun; the stack's j1 is the just-finished
    exchange). Session-last turns have no next moment and are skipped.
    Labels for a stop cue at t = soft_usage at t+1 (usage vs the NEXT
    response — anticipatory recall's self-supervising label); the caller
    joins with seq+1."""
    conn = sqlite3.connect('file:%s?mode=ro' % walker_path, uri=True)
    soft = {}
    for row in conn.execute(
            'SELECT session_id, epoch, seq, node_id, soft_max FROM soft_usage '
            'WHERE soft_max IS NOT NULL'):
        soft[(row[0], row[1], row[2], row[3])] = row[4]
    if cue_side == 'op':
        cues = conn.execute(
            'SELECT session_id, epoch, seq, ts, query_stored FROM turns '
            'WHERE labeled=1 ORDER BY ts').fetchall()
        conn.close()
        return cues, soft
    # stop side: anchor text + the anchor trace's own ts
    rows = conn.execute(
        'SELECT session_id, epoch, seq, ts, anchor_text, anchor_trace_id '
        'FROM turns WHERE anchor_text IS NOT NULL ORDER BY ts').fetchall()
    last_seq = {}
    for sid, ep, seq, *_ in rows:
        last_seq[(sid, ep)] = max(last_seq.get((sid, ep), -1), seq)
    tconn = sqlite3.connect('file:%s?mode=ro' % logs_db, uri=True)
    anchor_ts = dict(tconn.execute(
        "SELECT id, created_at FROM trace_events "
        "WHERE ref_type='assistant_message'"))
    tconn.close()
    conn.close()
    cues = []
    for sid, ep, seq, ts, anchor, atid in rows:
        if seq >= last_seq[(sid, ep)]:
            continue                      # no next moment to predict
        ats = anchor_ts.get(atid)
        if not ats:
            continue
        # +1ms past the assistant message: strict as_of masks keep turn t
        # visible and everything of t+1 in the future
        from datetime import datetime, timedelta
        as_of = (datetime.fromisoformat(ats) +
                 timedelta(milliseconds=1)).isoformat()
        cues.append((sid, ep, seq, as_of, (anchor or '')[:500]))
    return cues, soft


def fresh_brain_copy(corpus_pooled, work_dir):
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    for f in ('brain.db', 'brain_logs.db'):
        shutil.copy(corpus_pooled / f, work_dir / f)
    from servers.brain import Brain
    return Brain(db_path=str(work_dir / 'brain.db'))


def donor_map(cues):
    """C1: cue session → previous session in first-appearance order (cyclic).
    Sessions are date-ordered in the pooled corpus, so the donor's history
    exists before the cue's as_of."""
    order = []
    for sid, *_ in cues:
        if sid not in order:
            order.append(sid)
    return {sid: order[i - 1] for i, sid in enumerate(order)}


def run_arm(name, corpus_pooled, work_root, cues, gains_json):
    from build_corpus import _apply_interaction_override
    brain = fresh_brain_copy(corpus_pooled, work_root / name)
    if gains_json is not None:
        _apply_interaction_override(brain, 'recall_laf', template='',
                                    parameters=gains_json)
    donors = donor_map(cues) if name == 'C1' else None
    rows = []
    t_arm = time.time()
    for sid, epoch, seq, ts, query in cues:
        call_sid = donors[sid] if donors else sid
        t0 = time.time()
        out = brain.recall(query or '', limit=RECALL_LIMIT,
                           session_id=call_sid, as_of=ts, source='leg_b')
        ms = int((time.time() - t0) * 1000)
        cands = [(r.get('id'), r.get('effective_activation'))
                 for r in out.get('results', [])]
        rows.append({'key': [sid, epoch, seq], 'ms': ms, 'cands': cands})
    wall = time.time() - t_arm
    try:
        brain.close()
    except Exception:
        pass
    return rows, wall


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() >= 3 else float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    p.add_argument('--arms', default=','.join(DEFAULT_ARMS))
    p.add_argument('--work', default=None,
                   help='work root (default <corpus_dir>/leg_b[/stop])')
    p.add_argument('--cue-side', dest='cue_side', default='op',
                   choices=('op', 'stop'),
                   help="'stop' = endo/anticipatory: cue is Anchor's own "
                        "response, labels are the NEXT turn's usage")
    args = p.parse_args()

    cdir = Path(corpus_dir(args.corpus))
    pooled = cdir / 'pooled'
    walker_db = cdir / 'walker' / 'walker.db'
    work_root = Path(args.work) if args.work else \
        (cdir / 'leg_b' if args.cue_side == 'op' else cdir / 'leg_b' / 'stop')
    work_root.mkdir(parents=True, exist_ok=True)

    from arm_tables import arm as arm_table   # frozen fit → gain tables
    cues, soft = load_cues(walker_db, cue_side=args.cue_side,
                           logs_db=pooled / 'brain_logs.db')
    if args.cue_side == 'stop':
        # anticipatory label: usage at the NEXT moment — shift the join
        soft = {(k[0], k[1], k[2] - 1, k[3]): v for k, v in soft.items()}
    print('[leg_b] %d cues (%s side), arms: %s'
          % (len(cues), args.cue_side, args.arms), flush=True)

    arm_gains = {
        'A0': None,
        'A0f': json.dumps(arm_table('A0f')),
        'A1': json.dumps(arm_table('A1')),
        'A1t': json.dumps(arm_table('A1t')),
        'A1a': json.dumps(arm_table('A1a')),
        'A1k3': json.dumps(arm_table('A1k3')),
        'A1stk': json.dumps(arm_table('A1stk')),  # stack-only (o0 zeroed) —
                                                  # the walker-faithful endo
                                                  # shape: the cue text rides
                                                  # only as history, no j0
        'C1': json.dumps(arm_table('A1')),   # A1 gains, donor stack
    }

    results, walls = {}, {}
    for name in [a.strip() for a in args.arms.split(',') if a.strip()]:
        print('[leg_b] arm %s ...' % name, flush=True)
        rows, wall = run_arm(name, pooled, work_root, cues, arm_gains[name])
        results[name] = rows
        walls[name] = wall
        (work_root / ('%s.jsonl' % name)).write_text(
            '\n'.join(json.dumps(r) for r in rows) + '\n')
        print('[leg_b] arm %s done in %.1fs (%.0fms/recall)'
              % (name, wall, 1000 * wall / max(len(rows), 1)), flush=True)

    # ── soft_r per arm (engine path, G5) ────────────────────────────────
    report = {'n_cues': len(cues), 'soft_r': {}, 'latency_ms': {},
              'c2': None}
    for name, rows in results.items():
        xs, ys = [], []
        for r in rows:
            key = tuple(r['key'])
            for nid, score in r['cands']:
                sm = soft.get((*key, nid))
                if sm is not None and score is not None:
                    xs.append(score)
                    ys.append(sm)
        report['soft_r'][name] = pearson(xs, ys)
        ms = [r['ms'] for r in rows]
        report['latency_ms'][name] = {
            'p50': float(np.percentile(ms, 50)),
            'p95': float(np.percentile(ms, 95))}

    # ── C2: empty-stack identity A1 ≡ A0f. Empty stack ⟺ seq==0 in the
    # epoch — the FIRST LABELED cue is not enough: a no_recall seq-0 turn is
    # unlabeled yet still enters the next cue's stack as history (it was a
    # real turn; recall just didn't fire), so seq≥1 cues legitimately
    # diverge. Caught live: dev20 session i072d57b, exactly this shape.
    # OP SIDE ONLY: at a stop cue, seq0's stack contains the just-finished
    # turn — A1 ≢ A0f there BY DESIGN, so the identity property doesn't
    # exist on the stop side. ──
    if 'A1' in results and 'A0f' in results and args.cue_side == 'op':
        firsts = {(sid, epoch, seq) for sid, epoch, seq, ts, _ in cues
                  if seq == 0}
        mismatches = []
        a0f_by_key = {tuple(r['key']): r for r in results['A0f']}
        for r in results['A1']:
            key = tuple(r['key'])
            if key not in firsts:
                continue
            other = a0f_by_key.get(key)
            ids1 = [c[0] for c in r['cands']]
            ids0 = [c[0] for c in other['cands']]
            s1 = np.array([c[1] or 0 for c in r['cands']], float)
            s0 = np.array([c[1] or 0 for c in other['cands']], float)
            if ids1 != ids0 or (len(s1) == len(s0)
                                and np.abs(s1 - s0).max() > 1e-9):
                mismatches.append(key)
        report['c2'] = {'first_cues': len(firsts),
                        'mismatches': [list(k) for k in mismatches],
                        'pass': not mismatches}

    out = work_root / 'leg_b_report.json'
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'c2'}, indent=1))
    print('C2:', report['c2'])
    print('report → %s' % out)


if __name__ == '__main__':
    main()
