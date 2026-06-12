#!/usr/bin/env python3
"""BOOST DECOMPOSITION for EP5 — close the arithmetic. For each node in the recall
top-15 (+ the buried gold dabb3078), compare the pipeline's own boost delta
(effective_activation − embedding_similarity) against a FAITHFUL replication of the
title-match boost: terms = set(query.lower().split()) (no stopwords, punctuation
KEPT), match = substring containment, boost = fraction × TITLE_MATCH_BOOST.
If replicated ≈ delta, the title boost is quantitatively confirmed as the burial
mechanism. Residual = situation boost / penalties. Daemon-safe (IsolatedBrain)."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402
from servers.brain_constants import TITLE_MATCH_BOOST  # noqa: E402

QUERY = "what did we do on the last session we worked on ex.co?"
TERMS = set(QUERY.lower().split())   # EXACTLY what the pipeline does (line 1337)
BURIED = 'dabb3078'


def repl_boost(title):
    t = (title or '').lower()
    if not t or not TERMS:
        return 0.0, []
    hits = [q for q in TERMS if q in t]
    return (len(hits) / len(TERMS)) * TITLE_MATCH_BOOST, hits


with IsolatedBrain() as env:
    b = env.brain
    print("query terms as pipeline sees them (%d):" % len(TERMS), sorted(TERMS))
    if hasattr(b, '_recall_cache') and isinstance(b._recall_cache, dict):
        b._recall_cache.clear()
    out = b.recall(query=QUERY, limit=100)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])

    rows = []
    for i, r in enumerate(res, 1):
        nid = (r.get('id') or r.get('node_id') or '')[:8]
        emb = r.get('embedding_similarity')
        fin = r.get('effective_activation')
        ti = r.get('title') or ''
        rows.append((i, nid, emb, fin, ti))

    print("\nrank  node      emb    final  delta  repl_title  resid  hits | title")
    targets = rows[:15] + [r for r in rows if r[1] == BURIED]
    for i, nid, emb, fin, ti in targets:
        if emb is None or fin is None:
            print("%4d  %s  (missing scores: emb=%s fin=%s) %s" % (i, nid, emb, fin, ti[:40]))
            continue
        delta = fin - emb
        rb, hits = repl_boost(ti)
        resid = delta - rb
        mark = " <-- BURIED GOLD" if nid == BURIED else ""
        print("%4d  %s  %.3f  %.3f  %+.3f  %+.3f      %+.3f  %-28s | %s%s"
              % (i, nid, emb, fin, delta, rb, resid,
                 ",".join(sorted(hits))[:28], ti[:42], mark))

    # Aggregate: how well does replicated title boost explain the deltas?
    diffs = []
    for i, nid, emb, fin, ti in rows:
        if emb is None or fin is None:
            continue
        rb, _ = repl_boost(ti)
        diffs.append(abs((fin - emb) - rb))
    if diffs:
        import statistics
        print("\n|delta − replicated_title_boost| over %d scored nodes: median=%.4f  mean=%.4f  max=%.4f"
              % (len(diffs), statistics.median(diffs), statistics.mean(diffs), max(diffs)))
        print("(≈0 median → title boost alone explains the reranking; larger residuals = situation boost / penalties)")

    # The flood census: how many of the 100 titles each query term hits
    print("\n--- term flood census (substring hits across the %d returned titles) ---" % len(rows))
    for q in sorted(TERMS):
        n = sum(1 for _, _, _, _, ti in rows if q in (ti or '').lower())
        print("  %-10s %3d/%d" % (repr(q), n, len(rows)))
