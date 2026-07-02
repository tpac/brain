#!/usr/bin/env python3
"""Graph-fix benchmark: is the residual REALIZABLY reachable by a typed-edge hop? (§18.21)

B (the judge lens-tags) showed judges reached 34/53 residual needs via `graph` — but by following
a SPECIFIC edge from a node they found (often via the outcome). The question that decides whether
a graph operator helps REALIZABLY: are those residual nodes 1-hop (or 2-hop) over a TYPED
(noise-excluded) edge from a node that is *cosine-reachable from the cue* (top-25 of maxsim /
primary / episodic)? If yes → a "1-hop-from-reachable" operator captures them, and we see which
edge relations do the work. If no → the graph reach is seeded from outcome-found nodes and is
gated on the predictor (same wall as cos_outcome), not a realizable graph fix.

This is benchmark-FIRST: it sizes the realizable graph ceiling on the residual + names the edge
types BEFORE we wire an operator (the matrix's blunt z-scored 1-hop spread already reaches only 3).

Run (daemon maintenance-locked): ./dev python3 eval/laf/gold24_graph_probe.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from operators import MAXSIM_GROUPS, query_vec, build_field_matrices, maxsim_field, primary_field  # noqa: E402
from gold24_diagnostic import load_cues                              # noqa: E402
from gold24_matrix import episodic_field                            # noqa: E402

MATRIX = os.path.join(os.path.dirname(__file__), "gold24_matrix.json")
SEED_TOPK = 25       # "cosine-reachable from the cue" = top-25 of a realizable signal
RESID_REACH = 25     # a need is residual if NO signal put any of its nodes ≤25 (matches the matrix)


def typed_neighbors(brain, idx):
    """node_id → [(neighbor_id, relation)] over noise-excluded, undirected edges (relation kept)."""
    na = brain.aspects.by_name("noise")
    noise = list(na.edge_relations) if na else []
    ph = ",".join("?" * len(noise)) if noise else "''"
    rows = brain.conn.execute(
        "SELECT e.source_id, e.target_id, er.relation FROM edge_relations er "
        "JOIN edges e ON e.edge_id = er.edge_id "
        "WHERE (er.archived IS NULL OR er.archived = 0) AND er.relation NOT IN (%s)" % ph,
        noise).fetchall()
    nbr = defaultdict(list)
    for s, t, rel in rows:
        if s in idx and t in idx and s != t:
            nbr[s].append((t, rel)); nbr[t].append((s, rel))
    return nbr


def topk_ids(scores, eligible, master, k):
    s = np.where(eligible & np.isfinite(scores), scores, -np.inf)
    return {master[i] for i in np.argsort(-s)[:k]}


def main():
    cues = {c["id"]: c for c in load_cues()}
    mtx = json.load(open(MATRIX))["matrix"]

    # residual needs (recomputed from the persisted matrix ranks — matches the matrix run)
    residual = {}    # cue_id → {need: [node_ids]}
    for cid, nodes in mtx.items():
        needs = defaultdict(list)
        for nid, row in nodes.items():
            if row.get("tier") in ("gold_plus", "gold") and row.get("need"):
                needs[row["need"]].append(nid)
        res = {nd: nids for nd, nids in needs.items()
               if not any(any((nodes[n]["signals"][s]["rank"] or 1e9) <= RESID_REACH
                              for s in nodes[n]["signals"]) for n in nids)}
        if res:
            residual[cid] = res
    n_resid = sum(len(v) for v in residual.values())
    print("residual needs (no signal ≤%d): %d across %d cues" % (RESID_REACH, n_resid, len(residual)))

    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        master, idx, mats = build_field_matrices(brain, model, list(MAXSIM_GROUPS))
        ca = np.array([dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall()).get(n, "")
                       for n in master])
        nbr = typed_neighbors(brain, idx)

        hop1 = hop2 = both0 = 0
        rel1 = Counter()           # relations on the 1-hop edge that connects residual→reachable seed
        seed_via = Counter()       # which signal put the connecting SEED in reach
        for cid, needs in residual.items():
            c = cues[cid]
            qv = query_vec(c["query"])
            if qv is None:
                continue
            elig = (ca != "") & (ca <= c["cutoff"])
            ms = maxsim_field(qv, mats, list(MAXSIM_GROUPS))
            pr = primary_field(qv, mats)
            epi = episodic_field(brain, c["query"], c["cutoff"], idx, len(master))
            seed_sets = {"maxsim": topk_ids(ms, elig, master, SEED_TOPK),
                         "primary": topk_ids(pr, elig, master, SEED_TOPK),
                         "episodic": topk_ids(epi, elig, master, SEED_TOPK)}
            S = set().union(*seed_sets.values())
            for need, nids in needs.items():
                got1 = got2 = False
                for nid in nids:
                    one = nbr.get(nid, [])
                    hits1 = [(nb, rel) for nb, rel in one if nb in S]
                    if hits1:
                        got1 = True
                        for nb, rel in hits1:
                            rel1[rel] += 1
                            for sname, sset in seed_sets.items():
                                if nb in sset:
                                    seed_via[sname] += 1; break
                        break
                    # 2-hop: neighbor's neighbor in S
                    for nb, _ in one:
                        if any(nb2 in S for nb2, _ in nbr.get(nb, [])):
                            got2 = True; break
                if got1:
                    hop1 += 1
                elif got2:
                    hop2 += 1
                else:
                    both0 += 1

        print("\n=== REALIZABLE GRAPH REACH on the %d residual needs ===" % n_resid)
        print("  1-hop from a cosine-reachable seed (typed edge): %d  (%.0f%%)"
              % (hop1, 100 * hop1 / (n_resid or 1)))
        print("  2-hop (not 1):                                   %d  (%.0f%%)"
              % (hop2, 100 * hop2 / (n_resid or 1)))
        print("  unreachable even at 2-hop from cosine seeds:     %d  (%.0f%%)  ← graph can't help; predictor-gated"
              % (both0, 100 * both0 / (n_resid or 1)))
        print("\n  edge RELATIONS that connect residual→reachable (the operator should weight these):")
        for rel, n in rel1.most_common(12):
            print("    %-22s %d" % (rel, n))
        print("\n  which signal supplied the connecting SEED (where to spread FROM):")
        for s, n in seed_via.most_common():
            print("    %-12s %d" % (s, n))


if __name__ == "__main__":
    main()
