"""E1 eyeball cases — the blocking qualitative gate (§20.18), engine-path runs.

For each cue, compare the ENGINE's delivered lists (leg_b A0 vs A1) against
the soft-usage labels and render the full moment: conversation window, cue,
the response the assistant actually gave, and each arm's top-5 with titles,
soft values, and markers. Three bins:

  SUCCESS   A1 lifts the cue's highest-soft labeled node into its top-5
            while A0 had it outside (or absent) — the moment stack paid.
  HURT      A0 had a high-soft node in top-5 that A1 dropped/buried —
            the moment stack displaced something the response used.
  BOTH-MISS evidence cues (has_answer) whose item gold node EXISTED at the
            cue's as_of, yet neither arm delivered it top-25.

Markers: ★ item gold node · soft=— means the node is off the build pool
(engine reached outside the labeled set — soft labels can't see it; that's
what this eyeball is for).

Run:  ./dev python3 eval/longmem/e1_cases.py --corpus 74aea3
Out:  <corpus_dir>/leg_b/e1_cases.md
"""
import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import corpus_dir  # noqa: E402

N_PER_BIN = 5
TOP = 5
_DATE_PREFIX = re.compile(r'^\[Current date:[^\]]*\]\s*')


def strip(t):
    return _DATE_PREFIX.sub('', t or '')


def open_ro(path):
    return sqlite3.connect('file:%s?mode=ro' % path, uri=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    args = p.parse_args()

    cdir = Path(corpus_dir(args.corpus))
    manifest = json.loads((cdir / 'manifest.json').read_text())
    walker = open_ro(cdir / 'walker' / 'walker.db')
    brain = open_ro(cdir / 'pooled' / 'brain.db')

    titles = dict(brain.execute('SELECT id, title FROM nodes'))
    created = dict(brain.execute('SELECT id, created_at FROM nodes'))
    soft = {(r[0], r[1], r[2], r[3]): r[4] for r in walker.execute(
        'SELECT session_id, epoch, seq, node_id, soft_max FROM soft_usage '
        'WHERE soft_max IS NOT NULL')}
    turn_text = {(r[0], r[1], r[2]): (strip(r[3]), r[4], r[5]) for r in
                 walker.execute('SELECT session_id, epoch, seq, op_text, '
                                'anchor_text, ts FROM turns')}

    runs = {}
    for arm in ('A0', 'A1'):
        runs[arm] = {tuple(r['key']): r for r in
                     map(json.loads, open(cdir / 'leg_b' / ('%s.jsonl' % arm)))}

    gold_by_qid = {it['qid']: {m['node_id'] for m in it['gold_scan']['matches']}
                   for it in manifest['items'] if it['answerable']}

    def qid_of(sid):
        m = re.match(r'^i[0-9a-f]{7}-(.+)-s\d+$', sid)
        return m.group(1) if m else sid

    # evidence cue keys (has_answer user turns of answerable items)
    oracle = json.loads((Path(__file__).parent / 'data' /
                         manifest['config']['oracle']).read_text())
    evidence_keys = {}
    for item in oracle:
        qid = item['question_id']
        if qid not in gold_by_qid:
            continue
        for sess_idx, session in enumerate(item.get('haystack_sessions', [])):
            h = hashlib.sha1(('%s|%d' % (qid, sess_idx)).encode()).hexdigest()
            sid = 'i%s-%s-s%d' % (h[:7], qid, sess_idx)
            for t in session:
                if t.get('role') == 'user' and t.get('has_answer'):
                    for key, (op, _, _) in turn_text.items():
                        if key[0] == sid and op.startswith(t['content'][:200]):
                            evidence_keys[key] = qid

    def rank_of(cands, nid):
        for i, (cid, _) in enumerate(cands):
            if cid == nid:
                return i + 1
        return None

    successes, hurts, both_miss = [], [], []
    for key in runs['A1']:
        if key not in runs['A0']:
            continue
        a0, a1 = runs['A0'][key]['cands'], runs['A1'][key]['cands']
        labeled = [(nid, sm) for (nid, sm) in
                   ((nid, soft.get((*key, nid))) for nid, _ in (a0 + a1))
                   if sm is not None]
        if labeled:
            target, t_soft = max(labeled, key=lambda x: x[1])
            r0, r1 = rank_of(a0, target), rank_of(a1, target)
            if t_soft >= 0.70:
                if (r1 or 99) <= TOP < (r0 or 99):
                    successes.append((t_soft, key, target, r0, r1))
                elif (r0 or 99) <= TOP < (r1 or 99):
                    hurts.append((t_soft, key, target, r0, r1))
        qid = evidence_keys.get(key)
        if qid:
            alive = {g for g in gold_by_qid[qid]
                     if (created.get(g) or '9999') <= (turn_text[key][2] or '')}
            if alive and not any(rank_of(a0, g) or rank_of(a1, g)
                                 for g in alive):
                both_miss.append((key, qid, alive))

    successes.sort(reverse=True)
    hurts.sort(reverse=True)

    L = ['# E1 eyeball cases — corpus `%s` (engine path, A0 vs A1)' % args.corpus,
         '',
         'bins: %d successes / %d hurts / %d both-miss (rendering top %d each)'
         % (len(successes), len(hurts), len(both_miss), N_PER_BIN), '']

    def render_case(key, target=None, r0=None, r1=None, t_soft=None,
                    gold=frozenset()):
        sid, epoch, seq = key
        for j in (2, 1):
            prev = turn_text.get((sid, epoch, seq - j))
            if prev:
                L.append('> **T-%d op:** %s' % (j, prev[0][:220]))
                if prev[1]:
                    L.append('> **T-%d anchor:** %s' % (j, prev[1][:220]))
        op, anchor, _ = turn_text[key]
        L.append('> **CUE:** %s' % op[:300])
        L.append('> **response given:** %s' % (anchor or '')[:300])
        L.append('')
        if target:
            L.append('target (highest-soft labeled): `%s` "%s" soft=%.2f — '
                     'A0 rank %s, A1 rank %s'
                     % (target[:8], titles.get(target, '?')[:70], t_soft,
                        r0 or '>25', r1 or '>25'))
            L.append('')
        for arm in ('A0', 'A1'):
            cands = runs[arm][key]['cands'][:TOP]
            L.append('**%s top-%d:**' % (arm, TOP))
            other = {c[0] for c in runs['A1' if arm == 'A0' else 'A0'][key]['cands'][:TOP]}
            for i, (nid, _) in enumerate(cands):
                sm = soft.get((*key, nid))
                mark = '★' if nid in gold else (' ' if nid in other else '+')
                L.append('%d. %s `%s` %s — soft=%s'
                         % (i + 1, mark, (nid or '?')[:8],
                            titles.get(nid, '(?)')[:70],
                            '%.2f' % sm if sm is not None else '—'))
            L.append('')

    L.append('## SUCCESSES — A1 lifted the used node into top-%d' % TOP)
    for t_soft, key, target, r0, r1 in successes[:N_PER_BIN]:
        L.append('\n### %s / seq %d' % (key[0], key[2]))
        render_case(key, target, r0, r1, t_soft,
                    gold_by_qid.get(qid_of(key[0]), frozenset()))

    L.append('## HURTS — A1 displaced a used node from top-%d' % TOP)
    for t_soft, key, target, r0, r1 in hurts[:N_PER_BIN]:
        L.append('\n### %s / seq %d' % (key[0], key[2]))
        render_case(key, target, r0, r1, t_soft,
                    gold_by_qid.get(qid_of(key[0]), frozenset()))

    L.append('## BOTH-MISS — evidence cue, gold alive at as_of, neither arm '
             'delivered')
    for key, qid, alive in both_miss[:N_PER_BIN]:
        L.append('\n### %s / seq %d (item %s)' % (key[0], key[2], qid))
        L.append('gold alive: %s' % ', '.join(
            '`%s` "%s"' % (g[:8], titles.get(g, '?')[:60]) for g in alive))
        L.append('')
        render_case(key, gold=alive)

    out = cdir / 'leg_b' / 'e1_cases.md'
    out.write_text('\n'.join(L) + '\n')
    print('bins: %d successes / %d hurts / %d both-miss'
          % (len(successes), len(hurts), len(both_miss)))
    print('report → %s' % out)


if __name__ == '__main__':
    main()
