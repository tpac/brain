#!/usr/bin/env python3
"""Diagnostic: is the BRAIN_ENRICH_SCORE knob actually changing rankings, and is
STEP 3.5 enrichment even firing? Counts enrichment-vector coverage and diffs the
top-25 ordering avg2 vs max for every control query."""
import os, sys, json, re
# Import the repo this script physically lives in (worktree-safe — hardcoding
# /Users/tpac/brain would silently import the MAIN checkout, not the worktree).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

QS = json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
TIMED = {'episode'}


def cut(q):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', q.get('moment', ''))
    return (m.group(1) + 'T23:59:59+00:00') if (m and q['mode'] in TIMED) else None


def topids(b, q, n=25):
    c = cut(q)
    filt = {"created_at": {"lte": c}} if c else None
    lim = 200 if c else n
    try:
        out = b.recall(query=q['query'], limit=lim, filter=filt)
    except Exception:
        out = b.recall(query=q['query'], limit=lim)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    return [(r.get('id') or r.get('node_id'))[:8] for r in res][:n]


with IsolatedBrain() as env:
    b = env.brain
    # 1. Enrichment-vector coverage by type
    from collections import Counter
    by_type = Counter()
    per_node = Counter()
    for r in b._vec_dal.get_all_vectors():
        if r.get('embedding'):
            by_type[r['vector_type']] += 1
            per_node[r['node_id']] += 1
    multi = sum(1 for n, c in per_node.items() if c >= 2)
    print("=== enrichment-vector coverage ===")
    print("vectors by type:", dict(by_type))
    print("nodes with >=2 vectors: %d / %d (%.0f%%)"
          % (multi, len(per_node), 100.0 * multi / max(len(per_node), 1)))

    # 2. Ordering diff avg2 vs max across all queries
    print("\n=== ordering diff: avg2 vs max (top-25) ===")
    changed = 0
    for q in QS:
        os.environ['BRAIN_ENRICH_SCORE'] = 'avg2'
        if hasattr(b, '_recall_cache') and isinstance(b._recall_cache, dict):
            b._recall_cache.clear()
        a = topids(b, q)
        os.environ['BRAIN_ENRICH_SCORE'] = 'max'
        if hasattr(b, '_recall_cache') and isinstance(b._recall_cache, dict):
            b._recall_cache.clear()
        m = topids(b, q)
        os.environ.pop('BRAIN_ENRICH_SCORE', None)
        if a != m:
            changed += 1
            # first divergence index
            div = next((i for i in range(min(len(a), len(m))) if a[i] != m[i]), min(len(a), len(m)))
            print("  #%-4s %-8s CHANGED (first divergence @rank %d)  len %d→%d"
                  % (q['id'], q['mode'], div, len(a), len(m)))
    print("\nqueries with changed top-25 ordering: %d / %d" % (changed, len(QS)))
    if changed == 0:
        print("→ knob produces identical rankings. Either enrichment never wins over")
        print("  primary, or <2 vectors per node so the avg/max branch never runs.")
