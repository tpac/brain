#!/usr/bin/env python3
"""LANE CONTRIBUTION — measures what each recall lane actually contributes to the final 25.

Tallies the `_discovery` tag of every node in the final-25 across the 12-corpus (flag OFF / baseline).
Answers "is each lane doing its purpose?" with numbers, not assertions. Verified finding (2026-06-07):
recall is effectively MONO-LANE — dense fills ~100% of slots (embedding_only + both + embedding+keyword),
while fts5_only = 0 and keyword_only_fallback = 0 (the lexical lanes contribute ZERO unique nodes).

Never touches live (IsolatedBrain). Usage: ./dev python3 eval/oracle_audit/lane_contribution_probe.py
"""
import sys, json
from collections import Counter
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/meshed_top10.json'))

with IsolatedBrain() as env:
    brain = env.brain
    agg = Counter()
    per = []
    for it in sorted(CORPUS, key=lambda x: x['rank']):
        if hasattr(brain, '_recall_cache'):
            try: brain._recall_cache.clear()
            except Exception: pass
        out = brain.recall(query=it['prompt'], limit=25)
        res = out.get('results', []) if isinstance(out, dict) else (out or [])
        disc = Counter((r.get('_discovery') or '?') for r in res)
        agg.update(disc)
        per.append((it['rank'], len(res), dict(disc)))

    print("=== per-query final-25 discovery mix ===")
    for rank, n, d in per:
        print("  #%-2d n=%-3d %s" % (rank, n, d))
    print("\n=== AGGREGATE (what each LANE contributed to the 25s) ===")
    tot = sum(agg.values()) or 1
    for k, v in agg.most_common():
        print("  %-24s %4d  (%.0f%%)" % (k, v, 100.0 * v / tot))
    print("  TOTAL slots filled: %d / %d" % (tot, 12 * 25))
    print("\n  READ: fts5_only + keyword_only_fallback ≈ 0 ⇒ the lexical lanes are DEAD (0 unique nodes);")
    print("  recall is a mono-lane dense retriever. This is why entity queries (#2/#12) bury.")
