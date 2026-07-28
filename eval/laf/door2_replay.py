#!/usr/bin/env python3
"""Door-2 replay: this conversation's turns through LAF with the moment stack.

For each S1R turn of the two sessions (original stream + fork), score the
node universe three ways:
  door1  — LafV1Engine.scores() with empty moment table (dormancy invariant:
           bit-identical to live production laf_v1)
  door2  — same engine, moment_K=3 + a moment_gains table mapped from the
           walker P3 fit (p3_fit.json A_full_picked coefficients), stack
           pulled from THIS conversation via get_conversation(session_id),
           as_of-cut at the turn's timestamp
  live   — the top-25 the production pipeline actually served (from the O
           trace), for reference (includes floors/supplements beyond scores)

Per turn: top-25 diff (door2 vs door1) + rank of a hindsight watchlist —
the nodes I (Anchor, having lived the turns) know the turn actually needed.
Read-only: IsolatedBrain copy, scores() is read-only by construction.
"""
import json
import sys
import numpy as np

ROOT = '/Users/tpac/brain/.claude/worktrees/ecstatic-einstein-874830'
sys.path.insert(0, ROOT)

ORIG = 'fb78aab9-3e7c-414d-bc6c-695297f151aa'
FORK = '96c2693f-b497-409a-b5bd-c44eeae2d08b'

# hindsight watchlist per turn-prefix (first 30 chars of query)
WATCH = {
    'Not sure we need to strengthen': ['1ec0497a', 'a630b29e', 'be9e7fdf'],
    '1 - co_accessed is part of noi': ['06c69912', 'c3f37710', '1b8fdb1d'],
    'The doc is your work so i dont': ['2ace76f3', 'c478373d', '48ac820b', 'b0b22f55'],
    'Ok...... So good we opened thi': ['3a507484', '05c08df1', '302c33f0', '55104178', 'b2f97fb1'],
    'Sounds good. Worth remembering': ['3a507484', '05c08df1', 'b2f97fb1'],
    'Hey. Weird experiment.\nI\'ve fo': ['500b2b23', 'a7b0c067', '29311172', '137302a6'],
    '1. This shit always happens on': ['27590e1b', 'c7f31d6e', 'a1f25f4b', '81c3982b'],
}


def moment_gains_from_p3():
    """Map p3_fit A_full_picked coefficients -> the wiring's gain-table keys.

    'maxsim·j1-op' -> 'maxsim_o1'; 'sit·j2-anchor' -> 'sit_a2';
    'idf·j0-op' -> 'idf_o0' (the o0 override). 'tail' has no slot in the
    §20.17 wiring -> skipped (reported).
    """
    coef = json.load(open(ROOT + '/eval/laf/walker/p3_fit.json'))[
        'results']['A_full_picked']['coef']
    table, skipped = {}, []
    for k, v in coef.items():
        lane, _, slot = k.partition('·')
        if lane not in ('maxsim', 'sit', 'idf'):
            skipped.append(k)
            continue
        if slot == 'tail':
            skipped.append(k)
            continue
        jpart, _, side = slot.partition('-')
        if not jpart.startswith('j') or side not in ('op', 'anchor'):
            skipped.append(k)
            continue
        s = 'o' if side == 'op' else 'a'
        table['%s_%s%s' % (lane, s, jpart[1:])] = float(v)
    return table, skipped


def top25(score_map):
    return [nid for nid, _ in sorted(score_map.items(),
                                     key=lambda kv: -kv[1])[:25]]


def main():
    from tests.isolated_brain import IsolatedBrain
    from servers.embedder import embed_query
    from servers.recall_laf import get_engine, DEFAULT_CONFIG

    table, skipped = moment_gains_from_p3()
    print('moment_gains table: %d keys (skipped %d: %s)' % (
        len(table), len(skipped), ','.join(skipped)[:120]))

    with IsolatedBrain() as env:
        brain = env.brain
        eng = get_engine(brain)
        base_cfg = dict(eng.config(brain))

        turns = []
        for sid in (ORIG, FORK):
            rows = brain.query_traces(scale='s1', ref_type='recall',
                                      session_id=sid, limit=50)
            rows = rows.get('events', rows) if isinstance(rows, dict) else rows
            for r in rows:
                md = r.get('metadata') or {}
                if isinstance(md, str):
                    try:
                        md = json.loads(md)
                    except Exception:
                        md = {}
                q = md.get('query') or ''
                if not q:
                    continue
                cands = md.get('candidates') or []
                live = [c.split('|')[0] for c in cands if isinstance(c, str)]
                ts = (r.get('created_at') or '').replace(' ', 'T')
                turns.append({'sid': sid, 'ts': ts, 'q': q, 'live': live})
        turns.sort(key=lambda t: t['ts'])
        # keep only today's window (this conversation)
        turns = [t for t in turns if t['ts'] >= '2026-07-27T19:30']
        print('turns to replay: %d' % len(turns))

        for t in turns:
            qv = embed_query(t['q'][:2000])
            qv = np.frombuffer(qv, dtype=np.float32) if isinstance(qv, (bytes, bytearray)) else qv
            label = t['q'][:60].replace('\n', ' ')
            print('\n===== [%s] %s | %s' % (t['sid'][:8], t['ts'][11:19], label))

            import re as _re

            def cues_of(q):
                parts = [c.strip() for c in _re.split(r'[\n]+|(?<=[.?!]) ', q)]
                return [c for c in parts if len(c) >= 15][:8] or [q]

            arms = {}
            for arm, cfg_over in (
                    ('door1', {'moment_K': 0, 'moment_gains': {}}),
                    ('cues', {'moment_K': 0, 'moment_gains': {}}),
                    ('door2', {'moment_K': 3, 'moment_gains': table})):
                cfg = dict(base_cfg)
                cfg.update(cfg_over)
                eng.config = (lambda c: (lambda _brain: c))(cfg)
                try:
                    if arm == 'cues':
                        sm = {}
                        cs = cues_of(t['q'])
                        for cue in cs:
                            cv = embed_query(cue[:2000])
                            cv = np.frombuffer(cv, dtype=np.float32) if isinstance(cv, (bytes, bytearray)) else cv
                            csm, _ = eng.scores(brain, cue, cv,
                                                as_of=t['ts'],
                                                session_id=t['sid'])
                            for nid, sc in csm.items():
                                if sc > sm.get(nid, 0.0):
                                    sm[nid] = sc
                        print('  cues arm: %d cues' % len(cs))
                    else:
                        sm, _tele = eng.scores(brain, t['q'], qv,
                                               as_of=t['ts'],
                                               session_id=t['sid'])
                except Exception as e:
                    print('  %s FAILED: %s' % (arm, e))
                    sm = {}
                arms[arm] = top25(sm)
                if arm == 'door2':
                    led = eng._last_moment_ledger
                    print('  door2 stack ledger: %s' % (led,))

            d1, d2 = arms['door1'], arms['door2']
            dc = arms['cues']
            entered = [n for n in d2 if n not in d1]
            left = [n for n in d1 if n not in d2]
            print('  door2 vs door1: %d entered, %d left top-25' % (
                len(entered), len(left)))
            titles = {}
            allids = sorted(set(d1) | set(d2) | set(t['live'][:25]))
            for k in range(0, len(allids), 400):
                ch = allids[k:k + 400]
                ph = ','.join('?' * len(ch))
                for row in brain.conn.execute(
                        'SELECT id, title FROM nodes WHERE id IN (%s)' % ph, ch):
                    titles[row[0]] = (row[1] or '')[:56]
            for n in entered[:8]:
                print('    + %s %s' % (n[:8], titles.get(n, '?')))
            for n in left[:8]:
                print('    - %s %s' % (n[:8], titles.get(n, '?')))

            centered = [n for n in dc if n not in d1]
            print('  cues vs door1: %d entered' % len(centered))
            wkey = next((k for k in WATCH if t['q'].startswith(k)), None)
            if wkey:
                print('  watchlist ranks (live | door1 | cues | door2):')
                for w in WATCH[wkey]:
                    full = next((x for x in allids if x.startswith(w)), None)

                    def rk(lst, node=full, pre=w):
                        for i, x in enumerate(lst):
                            if x.startswith(pre):
                                return i + 1
                        return '-'
                    print('    %s  %s | %s | %s | %s  %s' % (
                        w, rk(t['live'][:25]), rk(d1), rk(dc), rk(d2),
                        titles.get(full, '')))


if __name__ == '__main__':
    main()
