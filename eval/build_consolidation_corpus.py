#!/usr/bin/env python3
"""Build a ground-truth consolidation corpus.

Step 1 (--dump): pull N real candidate clusters from the live decoder against a
production snapshot, write each cluster's full member content to a JSON file so a
human (Anchor) can read them and assign the EXPECTED action (the ground truth).

Step 2 (manual): Anchor reads the dump, writes expected labels into a corpus file
(cluster_id → {action, survivor?, absorbed?, rationale}).

Step 3 (--score, later): run the candidate prompt over the same frozen clusters
and compare actual decisions to the labels.

This replaces the noisy per-run LLM grader on a one-sided cold-start set with a
fixed, human-judged ground truth covering both axes (merge AND keep).

Usage:
    ./dev python3 eval/build_consolidation_corpus.py --dump --n 20 --out eval/corpus/clusters.json
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def member_rows(brain, ids):
    if not ids:
        return []
    ph = ','.join('?' * len(ids))
    rows = {}
    for r in brain.conn.execute(
            "SELECT id, type, title, content, locked, critical "
            "FROM nodes WHERE id IN (%s)" % ph, ids):
        rows[r[0]] = {
            'id': r[0][:8], 'type': r[1], 'title': r[2] or '',
            'content': (r[3] or '')[:700], 'locked': bool(r[4]) or bool(r[5]),
        }
    return [rows.get(nid, {'id': nid[:8], 'type': '?', 'missing': True}) for nid in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', action='store_true')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--out', default='eval/corpus/clusters.json')
    args = ap.parse_args()

    from tests.isolated_brain import IsolatedBrain
    from eval.s2_consolidation_eval import run_decoder

    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        decode = run_decoder(brain)
        clusters = decode.get('clusters', [])[:args.n]
        out = []
        for i, c in enumerate(clusters):
            out.append({
                'cluster_id': i,
                'pre_class': c.get('pre_class'),
                'content_cosine': round(c.get('content_cosine_max', 0), 3),
                'title_cosine': round(c.get('title_cosine_max', 0), 3),
                'members': member_rows(brain, c.get('nodes', [])),
                'node_ids': c.get('nodes', []),   # full ids, for the frozen run later
            })
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print('Dumped %d clusters → %s' % (len(out), args.out))
        # compact human-readable index
        from collections import Counter
        print('pre_class mix:', dict(Counter(c['pre_class'] for c in out)))


if __name__ == '__main__':
    main()
