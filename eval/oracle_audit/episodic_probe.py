#!/usr/bin/env python3
"""EPISODIC probe — the surface we built and never used. Isolated copy only.

#11 ("what did we do on the LAST SESSION we worked on ex.co") is an EPISODIC query.
We've been answering it semantically (node vectors). The episodic surface — S0 traces with
their own embeddings (trace_embeddings) + source_refs (node_source_refs reverse index) — has
been sitting unused. This tests whether it answers #11 NATURALLY:

  1. recall #11 over trace_embeddings (cosine) — do ex.co-session turns surface?
  2. coverage check: how many traces are embedded, how many mention ex.co, how many of THOSE are embedded
  3. source_ref hop: top traces -> node_source_refs reverse-lookup -> do ex.co nodes appear?

Never touches live. Usage: ./dev python3 eval/oracle_audit/episodic_probe.py
"""
import os, sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/meshed_top10.json'))
KNOWN_EXCO = {'e62cc595','dabb3078','af92b2cb','30d88dd0','b3bda662','5fe121db',
              '8359cf1d','5410f4be','ef2f3276','41d31ca5','671d1f22','598d78a8'}

def is_exco(s):
    s = (s or '').lower()
    return 'ex.co' in s or 'exco' in s

with IsolatedBrain() as env:
    brain = env.brain
    from servers import embedder
    lc, bc = brain.logs_conn, brain.conn

    # 1. pull all trace embeddings
    rows = lc.execute("SELECT trace_id, vector, text, created_at FROM trace_embeddings").fetchall()
    total_traces = lc.execute("SELECT COUNT(*) FROM trace_events").fetchone()[0]
    print("\n=== EPISODIC probe (isolated) ===")
    print("  trace_events=%d  trace_embeddings=%d (%.0f%% embedded)"
          % (total_traces, len(rows), 100*len(rows)/max(total_traces,1)))

    # coverage: traces mentioning ex.co among the EMBEDDED set (text is on trace_embeddings)
    exco_embedded = sum(1 for _, _, t, _ in rows if is_exco(t))
    print("  embedded traces mentioning ex.co: %d" % exco_embedded)

    def rank_traces(query, topn=15):
        qv = embedder.embed_query(query)
        scored = [(embedder.cosine_similarity(qv, vec), tid, txt, ts) for tid, vec, txt, ts in rows]
        scored.sort(key=lambda x: -x[0])
        return scored[:topn]

    def best_exco_trace_rank(query, limit=200):
        qv = embedder.embed_query(query)
        scored = sorted([(embedder.cosine_similarity(qv, vec), txt) for _, vec, txt, _ in rows], key=lambda x: -x[0])
        for i, (_, txt) in enumerate(scored[:limit], 1):
            if is_exco(txt):
                return i
        return None

    q11 = next(it['prompt'] for it in CORPUS if it['rank'] == 11)
    print("\n--- #11: %s" % q11)
    print("  best ex.co-mentioning trace rank in episodic recall: %s" % best_exco_trace_rank(q11))
    print("  top-12 traces for #11 (cos | date | ex.co? | text):")
    top = rank_traces(q11, 12)
    for cos, tid, txt, ts in top:
        flag = 'EXCO' if is_exco(txt) else '    '
        print("   %.3f %s %s %s %s" % (cos, tid, (ts or '')[:10], flag, (txt or '').replace('\n',' ')[:72]))

    # sanity: "ex.co" alone over traces
    print("\n--- sanity: 'ex.co' alone over traces, top-6 ---")
    for cos, tid, txt, ts in rank_traces("ex.co", 6):
        print("   %.3f %s %s %s" % (cos, tid, (ts or '')[:10], (txt or '').replace('\n',' ')[:64]))

    # 3. source_ref hop: top #11 traces -> nodes
    print("\n--- source_ref hop: nodes anchored to #11's top-15 traces ---")
    top15_ids = [tid for _, tid, _, _ in rank_traces(q11, 15)]
    placeholders = ','.join('?' * len(top15_ids))
    nsr = bc.execute(
        "SELECT trace_id, node_id FROM node_source_refs WHERE trace_id IN (%s)" % placeholders, top15_ids).fetchall()
    print("  node_source_refs hit for %d of top-15 traces -> %d node links" % (len({t for t, _ in nsr}), len(nsr)))
    hop_nodes = {n[:8] for _, n in nsr}
    hop_exco = hop_nodes & KNOWN_EXCO
    print("  hopped nodes: %d unique; KNOWN_EXCO among them: %s" % (len(hop_nodes), hop_exco or 'none'))
    for nid in list(hop_nodes)[:8]:
        t = bc.execute("SELECT title FROM nodes WHERE id=?", (nid,)).fetchone()
        print("     %s %s %s" % (nid, '[EXCO]' if nid in KNOWN_EXCO else '      ', (t[0] if t else '')[:48]))

    # overall coverage of source_refs (how dense is the hop in general?)
    n_nodes = bc.execute("SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
    n_with_refs = bc.execute("SELECT COUNT(DISTINCT node_id) FROM node_source_refs").fetchone()[0]
    print("\n  source_ref coverage: %d/%d nodes have source_refs (%.0f%%)"
          % (n_with_refs, n_nodes, 100*n_with_refs/max(n_nodes,1)))
