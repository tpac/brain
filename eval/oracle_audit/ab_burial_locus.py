#!/usr/bin/env python3
"""BURIAL LOCUS for EP5. Finding 2: nodes with rank-5 cosine vanish from recall.
Hypothesis: STEP 6 title-match boost lifts lower-cosine nodes (esp. 'session'/'last'
in title) above high-cosine ones. Recall with a huge limit, find the buried nodes'
true rank, and split the nodes ABOVE them into 'higher cosine' (legit) vs 'lower cosine
but boosted above' (the burial). Daemon-safe (IsolatedBrain)."""
import os, sys, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402
from servers.brain_constants import TITLE_MATCH_BOOST, NOISE_FLOOR_THRESHOLD  # noqa: E402

QUERY = "what did we do on the last session we worked on ex.co?"
BURIED = ['8359cf1d', 'dabb3078']
_STOP = set("what did we do on the last we a an the to of in is are i you it".split())
QTERMS = {t for t in re.findall(r"[a-z0-9.]+", QUERY.lower()) if t not in _STOP}


with IsolatedBrain() as env:
    b = env.brain
    print("TITLE_MATCH_BOOST=%.2f  NOISE_FLOOR=%.2f" % (TITLE_MATCH_BOOST, NOISE_FLOOR_THRESHOLD))
    print("query terms (post-stop):", sorted(QTERMS))
    if hasattr(b, '_recall_cache') and isinstance(b._recall_cache, dict):
        b._recall_cache.clear()
    out = b.recall(query=QUERY, limit=5000)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    rows = []
    for i, r in enumerate(res, 1):
        nid = (r.get('id') or r.get('node_id') or '')[:8]
        rows.append((i, nid, r.get('embedding_similarity'), (r.get('title') or '')))
    print("recall returned %d results (limit 5000)" % len(rows))

    by_id = {nid: (rk, sim, ti) for rk, nid, sim, ti in rows}
    for bnid in BURIED:
        if bnid not in by_id:
            print("\n%s: NOT in returned set at all" % bnid); continue
        rk, sim, ti = by_id[bnid]
        print("\n=== %s  rank=%d  emb_sim=%.3f  '%s' ===" % (bnid, rk, sim or 0, ti[:50]))
        above = [r for r in rows if r[0] < rk]
        # nodes above with LOWER cosine than the buried node = boosted past it
        boosted = [r for r in above if (r[2] is None) or (sim is not None and r[2] < sim)]
        legit = [r for r in above if r[2] is not None and sim is not None and r[2] >= sim]
        print("  nodes above it: %d   | higher-cosine (legit): %d   | lower-cosine but boosted above: %d"
              % (len(above), len(legit), len(boosted)))
        # of the boosted-above, how many have a query term in their title?
        tmatch = [r for r in boosted if any(t in (r[3] or '').lower() for t in QTERMS)]
        print("  of those %d boosted-above, %d have a query term in their TITLE (title-boost burial)"
              % (len(boosted), len(tmatch)))
        print("  top 12 boosted-above (rank | emb_sim | matched-title-terms | title):")
        for rk2, nid2, sim2, ti2 in sorted(boosted, key=lambda x: x[0])[:12]:
            mt = [t for t in QTERMS if t in (ti2 or '').lower()]
            print("    r%-4d cos=%-5s %-22s %s" % (
                rk2, ('%.3f' % sim2) if sim2 is not None else '—',
                ",".join(mt) or '(no title hit)', (ti2 or '')[:46]))
