#!/usr/bin/env python3
"""GRAPH-COHERENCE re-rank probe (Tom's idea) — isolated copy.

Hypothesis: in #11's top-100 candidate pool, the buried EX.CO nodes (rank 36-59 by score)
form a COHERENT cluster (mutual edges / shared community), while top-scored brain-dev nodes
are structurally ISOLATED w.r.t. the candidate set. If so, re-ranking by graph coherence —
not by score — promotes the cluster: "filter a 31-spot node over a 4th."

Measures, for #11's top-100 candidates:
  - intra-candidate edge degree per node (how connected to OTHER candidates)
  - community membership (shared community = a coherence signal)
  - contrast: EX.CO nodes' coherence vs the top-scored nodes' coherence
  - a simulated coherence re-rank: do EX.CO nodes rise into top-30?

Never touches live. Usage: ./dev python3 eval/oracle_audit/graph_rerank_probe.py
"""
import os, sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/meshed_top10.json'))
KNOWN_EXCO = {'e62cc595','dabb3078','af92b2cb','30d88dd0','b3bda662','5fe121db',
              '8359cf1d','5410f4be','ef2f3276','41d31ca5','671d1f22','598d78a8'}

with IsolatedBrain() as env:
    brain = env.brain
    conn = brain.conn
    q11 = next(it['prompt'] for it in CORPUS if it['rank'] == 11)
    if hasattr(brain, '_recall_cache'):
        try: brain._recall_cache.clear()
        except Exception: pass
    out = brain.recall(query=q11, limit=100)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    cands = [(r.get('id') or r.get('node_id'), r.get('effective_activation')) for r in res]
    cand_ids = [c[0] for c in cands]
    cset = set(cand_ids)
    rank = {cid: i for i, cid in enumerate(cand_ids, 1)}
    score = {cid: s for cid, s in cands}
    print("\n=== GRAPH-COHERENCE re-rank probe (#11, top-%d pool) ===" % len(cand_ids))
    exco_in_pool = [c for c in cand_ids if c[:8] in KNOWN_EXCO]
    print("  EX.CO nodes in the top-100 pool: %d  (ranks: %s)"
          % (len(exco_in_pool), sorted(rank[c] for c in exco_in_pool)))

    # intra-candidate edges (structural; exclude Hebbian co_accessed/emergent_bridge)
    ph = ','.join('?' * len(cand_ids))
    rows = conn.execute(
        """SELECT e.source_id, e.target_id, er.relation
           FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id
           WHERE e.source_id IN (%s) AND e.target_id IN (%s)
             AND er.relation NOT IN ('co_accessed','emergent_bridge')""" % (ph, ph),
        cand_ids + cand_ids).fetchall()
    from collections import defaultdict
    deg = defaultdict(int)
    rels = defaultdict(int)
    for s, t, rel in rows:
        if s in cset and t in cset and s != t:
            deg[s] += 1; deg[t] += 1; rels[rel] += 1
    print("  intra-candidate edges: %d  (relations: %s)" % (len(rows), dict(rels)))

    # community membership among candidates
    crows = conn.execute(
        """SELECT e.source_id, e.target_id FROM edges e JOIN edge_relations er ON er.edge_id=e.edge_id
           WHERE er.relation='community_member' AND e.source_id IN (%s)""" % ph, cand_ids).fetchall()
    comm = {}
    for s, t in crows:
        comm[s] = t[:8]
    comm_count = defaultdict(list)
    for cid in cand_ids:
        if cid in comm:
            comm_count[comm[cid]].append(cid)
    print("  candidates with a community: %d ; top communities by member-count in pool:" % len(comm))
    for cm, members in sorted(comm_count.items(), key=lambda x: -len(x[1]))[:5]:
        ex = sum(1 for m in members if m[:8] in KNOWN_EXCO)
        print("     community %s: %d candidates (%d EX.CO)" % (cm, len(members), ex))

    def line(cid):
        t = conn.execute("SELECT title FROM nodes WHERE id=?", (cid,)).fetchone()
        return "r%-3d sc=%.3f deg=%-2d comm=%s %s %s" % (
            rank[cid], score.get(cid) or 0, deg.get(cid, 0), comm.get(cid, '--'),
            '[EXCO]' if cid[:8] in KNOWN_EXCO else '      ', (t[0] if t else '')[:40])

    print("\n  -- EX.CO nodes in pool (rank, score, intra-degree, community) --")
    for cid in sorted(exco_in_pool, key=lambda c: rank[c]):
        print("   ", line(cid))
    print("\n  -- top-8 by SCORE (the 'rank 4' isolated-high-scorers?) --")
    for cid in cand_ids[:8]:
        print("   ", line(cid))

    # contrast
    import statistics as st
    exco_deg = [deg.get(c, 0) for c in exco_in_pool]
    top10_deg = [deg.get(c, 0) for c in cand_ids[:10]]
    print("\n  CONTRAST intra-degree: EX.CO mean=%.1f median=%d  |  top-10-by-score mean=%.1f median=%d"
          % (st.mean(exco_deg) if exco_deg else 0, int(st.median(exco_deg)) if exco_deg else 0,
             st.mean(top10_deg), int(st.median(top10_deg))))

    # simulated coherence re-rank: sort by (intra-degree desc, then score) — pure structure-first
    by_deg = sorted(cand_ids, key=lambda c: (-deg.get(c, 0), -(score.get(c) or 0)))
    deg_rank = {c: i for i, c in enumerate(by_deg, 1)}
    best_exco_deg = min((deg_rank[c] for c in exco_in_pool), default=None)
    # community-coherence re-rank: boost members of the dominant EX.CO-bearing community
    print("\n  SIMULATED re-ranks (best EX.CO rank under each):")
    print("     by score (baseline):            %s" % (min((rank[c] for c in exco_in_pool), default=None)))
    print("     by intra-degree (structure):    %s" % best_exco_deg)
