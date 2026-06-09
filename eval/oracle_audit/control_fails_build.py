#!/usr/bin/env python3
"""CONTROL-FAILS CORPUS BUILDER — the missing foundation: a corpus where CONTROL recall actually
fails, with gold defined INDEPENDENTLY of recall (not harvested from recall output = the flaw in
recall_corpus_v2 that made Control pass by construction).

Independent gold = each node's STORED `question` field (the question it was encoded to answer).
query = that question, gold = that node. Run Control recall; bucket by where the gold lands:
  HIT@5   rank 1-5   (surfaces to awareness)   <- control succeeds
  POOL    rank 6-25  (retrieved, not surfaced) <- borderline
  MISS    not in 25  (not retrieved)           <- control FAILS

CAVEAT (honest): the node's own question is FAVORABLE to recall (written to match the node), so the
fail rate here is a LOWER BOUND — real-world phrasing (oblique, different vocabulary) is harder.
If control fails meaningfully even on own-question, there IS a real retrieval problem + a corpus to
fix it on. If it ~never fails, recall is genuinely good enough and we stop.

Daemon-safe (IsolatedBrain). Dumps the POOL+MISS pairs to control_fails_corpus.json.
Usage: ./dev python3 eval/oracle_audit/control_fails_build.py
"""
import sys, json, random
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

SAMPLE = 200
random.seed(7)
TYPES = ('fact', 'decision', 'lesson', 'insight', 'mechanism', 'finding', 'concept',
         'rule', 'principle', 'correction', 'moment', 'reflection', 'context', 'method')


with IsolatedBrain() as env:
    b = env.brain
    # independent gold: (node_id, question) from stored question field, substantive types, live nodes
    rows = b.conn.execute(
        """SELECT m.node_id, m.value, n.type, n.title
           FROM node_metadata_kv m JOIN nodes n ON n.id = m.node_id
           WHERE m.key = 'question' AND COALESCE(n.archived,0) = 0
             AND n.type IN (%s) AND length(m.value) > 8""" % ','.join('?' * len(TYPES)),
        TYPES).fetchall()
    print("nodes with a stored question (substantive types):", len(rows))
    if len(rows) > SAMPLE:
        rows = random.sample(rows, SAMPLE)

    def rank_of(nid, query):
        out = b.recall(query=query, limit=25)
        res = out.get('results', []) if isinstance(out, dict) else (out or [])
        ids = [(r.get('id') or r.get('node_id'))[:8] for r in res]
        tgt = nid[:8]
        return (ids.index(tgt) + 1) if tgt in ids else None

    buckets = {'HIT@5': 0, 'POOL': 0, 'MISS': 0}
    fails = []
    for i, (nid, q, ntype, title) in enumerate(rows):
        rk = rank_of(nid, q)
        if rk and rk <= 5:
            buckets['HIT@5'] += 1
        elif rk:
            buckets['POOL'] += 1
            fails.append({'gold': nid[:8], 'type': ntype, 'rank': rk, 'bucket': 'POOL',
                          'question': q, 'title': title[:70]})
        else:
            buckets['MISS'] += 1
            fails.append({'gold': nid[:8], 'type': ntype, 'rank': None, 'bucket': 'MISS',
                          'question': q, 'title': title[:70]})
        if (i + 1) % 50 == 0:
            print("  ...%d/%d" % (i + 1, len(rows)))

    n = len(rows)
    print("\n=== CONTROL recall on each node's OWN question (n=%d) ===" % n)
    for k in ('HIT@5', 'POOL', 'MISS'):
        print("  %-6s %3d  (%.0f%%)" % (k, buckets[k], 100.0 * buckets[k] / max(n, 1)))
    print("  CONTROL FAILS (POOL+MISS, gold not in top-5): %d/%d (%.0f%%)"
          % (buckets['POOL'] + buckets['MISS'], n, 100.0 * (buckets['POOL'] + buckets['MISS']) / max(n, 1)))

    print("\n=== sample of the failures (the corpus to fix on) ===")
    for f in sorted(fails, key=lambda x: (x['rank'] is not None, x['rank'] or 999))[:12]:
        print("  [%s rank=%s] %s" % (f['bucket'], f['rank'], (f['question'] or '')[:60]))
        print("        gold %s (%s) %s" % (f['gold'], f['type'], f['title']))

    with open(f'{ROOT}/eval/oracle_audit/control_fails_corpus.json', 'w') as fp:
        json.dump({'n_sampled': n, 'buckets': buckets, 'fails': fails}, fp, indent=2)
    print("\nwrote control_fails_corpus.json (%d control-failures: gold known, recall missed)" % len(fails))
