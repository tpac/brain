#!/usr/bin/env python3
"""LAF field-activation probe — REACH diagnosis on 3 far-miss golds.

For each far-miss gold, compute its rank in the full eligible field under each
field-vector ALONE (`_primary` base + question/situation/title/... additional
activations), plus graph-1hop / community / episodic-availability flags.

Single metric: REACH — does a signal pull the gold from far-120+ into top-25
(contention) or top-5 (would-hit)? This DISCOVERS which additional field-activation
carries each far-gold. It is NOT a win metric — the corpus A/B (control-gated) is the judge.

Adjudicates a live tension:
  - 8bcc8c96: "multi-field matching is a dead end (primary dominates)" — measured AGGREGATE.
  - reverse-look (this session): `question` is a gated REACH lever for FAR-misses.
If question-cosine doesn't rank these golds better than _primary, the hypothesis is dead.

Run: ./dev python3 eval/laf/field_activation_probe.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain          # noqa: E402
from servers import embedder                             # noqa: E402
from endo_baseline_recall import load_corpus             # noqa: E402

CASES = {
    "operator_msg_0131": "spread vs CS/bio  (cross-domain abstraction)",
    "operator_msg_0363": "co_access as trace<->node connection",
    "operator_msg_0118": "RRF / recency / reranker",
}


def reach_label(rank):
    if rank is None:
        return "no-vector"
    if rank <= 5:
        return "TOP5"
    if rank <= 25:
        return "top25"
    if rank <= 120:
        return "pool"
    return "FAR"


def main():
    corpus = {c["id"]: c for c in load_corpus()}
    with IsolatedBrain() as env:
        if not embedder.is_ready():
            embedder.load_model()
        conn = env.brain.conn

        vtypes = [r[0] for r in conn.execute(
            "SELECT DISTINCT vector_type FROM node_enrichments").fetchall()]
        print("vector_types available:", vtypes)

        # Per vector-type: ids, created_at, matrix
        vt = {}
        for v in vtypes:
            rows = conn.execute(
                "SELECT n.id, n.created_at, e.embedding FROM node_enrichments e "
                "JOIN nodes n ON n.id=e.node_id WHERE e.vector_type=? AND n.archived=0",
                (v,)).fetchall()
            if not rows:
                continue
            ids = [r[0] for r in rows]
            created = np.array([r[1] or "" for r in rows])
            M = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
            vt[v] = (ids, created, M)

        for cue_id, note in CASES.items():
            c = corpus[cue_id]
            cv = np.frombuffer(embedder.embed_batch([c["query"]], kind="query")[0],
                               dtype=np.float32)
            cutoff, golds = c["cutoff"], c["gold_essential"]
            print("\n" + "=" * 74)
            print("%s — %s\n  gold(s): %s" % (cue_id, note, golds))

            # --- field-activation reach: gold rank under each vector type ---
            primary_order = None
            for v in vtypes:
                if v not in vt:
                    continue
                ids, created, M = vt[v]
                elig = created < cutoff
                sc = np.where(elig, M @ cv, -np.inf)
                order = np.argsort(-sc)
                rank_of = {ids[order[r]]: r + 1 for r in range(len(order))
                           if np.isfinite(sc[order[r]])}
                cos_of = {ids[j]: float(sc[j]) for j in range(len(ids))
                          if np.isfinite(sc[j])}
                if v == "_primary":
                    primary_order = [ids[i] for i in order[:25]]
                cells = []
                for g in golds:
                    rk = rank_of.get(g)
                    cells.append("%s rank=%-5s cos=%-6s [%s]"
                                 % (g, rk, round(cos_of.get(g, 0), 3) if g in cos_of else "-",
                                    reach_label(rk)))
                print("  [%-12s] %s" % (v, "   ".join(cells)))

            # --- graph-1hop: is gold a 1-hop neighbor of the _primary top-25? ---
            if primary_order:
                ph = ",".join("?" * len(primary_order))
                nbrs = set()
                try:
                    for row in conn.execute(
                            "SELECT target_id FROM edges WHERE source_id IN (%s) AND archived=0" % ph,
                            primary_order):
                        nbrs.add(row[0])
                    for row in conn.execute(
                            "SELECT source_id FROM edges WHERE target_id IN (%s) AND archived=0" % ph,
                            primary_order):
                        nbrs.add(row[0])
                except Exception as e:
                    print("  [graph-1hop  ] (skipped: %s)" % e)
                    nbrs = None
                if nbrs is not None:
                    for g in golds:
                        print("  [graph-1hop  ] %s: %s"
                              % (g, "YES — 1-hop from _primary top-25" if g in nbrs else "no"))

                # --- community: gold shares a community-node with the top-25? ---
                def comms(nid):
                    try:
                        rows = conn.execute(
                            "SELECT e.target_id FROM edges e JOIN edge_relations r ON r.edge_id=e.edge_id "
                            "JOIN nodes n ON n.id=e.target_id "
                            "WHERE e.source_id=? AND r.relation='community_member' AND n.type='community' "
                            "UNION SELECT e.source_id FROM edges e JOIN edge_relations r ON r.edge_id=e.edge_id "
                            "JOIN nodes n ON n.id=e.source_id "
                            "WHERE e.target_id=? AND r.relation='community_member' AND n.type='community'",
                            (nid, nid)).fetchall()
                        return {r[0] for r in rows}
                    except Exception:
                        return set()
                top_comms = set()
                for t in primary_order:
                    top_comms |= comms(t)
                for g in golds:
                    gc = comms(g)
                    print("  [community   ] %s: comms=%s shared_with_top25=%s"
                          % (g, list(gc)[:3], "YES" if (gc & top_comms) else "no"))

            # --- episodic availability: traces near gold's created_at ---
            for g in golds:
                try:
                    node = env.brain.get_node(g)
                    ca = node.get("created_at") if node else None
                    src_turn = node.get("source_turn_id") if node else None
                    n_tr = env.brain.logs_conn.execute(
                        "SELECT COUNT(*) FROM trace_events WHERE created_at <= ?", (ca,)).fetchone()[0]
                    print("  [episodic    ] %s: created_at=%s source_turn_id=%s  traces_before=%s"
                          % (g, ca, src_turn, n_tr))
                except Exception as e:
                    print("  [episodic    ] %s: (skipped: %s)" % (g, e))


if __name__ == "__main__":
    main()
