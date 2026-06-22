#!/usr/bin/env python3
"""Edge-seed vs node-seed A/B on the frozen endo gold corpus.

Tests the HippoRAG query→triple-linking lever (their biggest ablation, +20.9 recall):
does seeding the candidate ranking from cosine(cue, EDGE embedding) → endpoint nodes
beat the standard cosine(cue, NODE embedding)? The edge embedding is the brain's
compose_edge_text(relation, description) — i.e. the query is matched against the
*relationship's* semantics, a non-node-cosine lane.

Both arms use the SAME nomic-Q model + per-cue cutoff + scorer, so only the seed signal
differs — a clean A/B. Read-only against a brain.db snapshot (no daemon contention).

Handoff from stream 8c28df4e; lever credited to this stream (HippoRAG mapping).
The question: additive on the CURRENT flat embedder, or gated on the embedding upgrade?

Run: ./dev python3 eval/oracle_audit/emb_bench/edge_seed_ab.py
"""
import json, os, sys, sqlite3
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench import score_one, KS, FastEmbedBackend, _normalize  # reuse identical scorer + embedder

DATA = "/tmp/emb_bench"
SNAP = os.path.join(DATA, "snapshot", "brain.db")
# noise relations production excludes from traversal (d1d1a90c) — structural/usage, not semantic
NOISE = ("co_accessed", "community_member", "related_to", "related", "co_anchored", "emergent_bridge")


def load_vecs():
    cues = json.load(open(os.path.join(DATA, "cues.json")))
    created = {n["id"]: n["created_at"] for n in json.load(open(os.path.join(DATA, "nodes.json")))}
    con = sqlite3.connect(f"file:{SNAP}?mode=ro", uri=True)

    # node _primary embeddings (the node-seed arm + the geometry edges live in)
    nids, nvecs = [], []
    for nid, blob in con.execute(
        "SELECT node_id, embedding FROM node_enrichments "
        "WHERE vector_type='_primary' AND embedding IS NOT NULL"):
        if nid in created:
            nids.append(nid); nvecs.append(np.frombuffer(blob, np.float32))
    nvecs = _normalize(np.array(nvecs))
    nidx = {nid: i for i, nid in enumerate(nids)}

    # semantic edge embeddings + endpoints + created_at
    qmarks = ",".join("?" * len(NOISE))
    evecs, eends, ecreated = [], [], []
    for blob, src, tgt, eca in con.execute(
        f"""SELECT er.embedding, e.source_id, e.target_id, e.created_at
            FROM edge_relations er JOIN edges e ON er.edge_id = e.edge_id
            WHERE er.embedding IS NOT NULL AND COALESCE(er.archived,0)=0
              AND er.relation NOT IN ({qmarks})""", NOISE):
        evecs.append(np.frombuffer(blob, np.float32)); eends.append((src, tgt)); ecreated.append(eca or "")
    evecs = _normalize(np.array(evecs))
    con.close()
    return cues, created, nids, nvecs, nidx, evecs, eends, ecreated


def main():
    cues, created, nids, nvecs, nidx, evecs, eends, ecreated = load_vecs()
    print(f"nodes(_primary): {len(nids)}  semantic-edge vecs: {len(evecs)}  cues: {len(cues)}")

    model = FastEmbedBackend("nomic-ai/nomic-embed-text-v1.5-Q")
    cue_vecs = model.embed([c["query"] for c in cues], "search_query: ")

    def node_seed(i, c):
        sims = nvecs @ cue_vecs[i]
        order = np.argsort(-sims)
        return [nids[j] for j in order if created.get(nids[j], "") <= c["cutoff"]]

    def edge_seed(i, c):
        esims = evecs @ cue_vecs[i]
        score = {}
        for k, (src, tgt) in enumerate(eends):
            if ecreated[k] and ecreated[k] > c["cutoff"]:
                continue                                   # edge must pre-exist the cue
            s = float(esims[k])
            for nid in (src, tgt):
                if created.get(nid, "") <= c["cutoff"] and nid in nidx and s > score.get(nid, -9.0):
                    score[nid] = s                          # node = best matching incident edge
        return [nid for nid, _ in sorted(score.items(), key=lambda x: -x[1])]

    rows = {"node_seed": [], "edge_seed": []}
    cover = []                                              # edge-seed gold reachability
    for i, c in enumerate(cues):
        ess, helpful = c["gold_essential"], c.get("gold_helpful", [])
        ns = node_seed(i, c); es = edge_seed(i, c)
        for arm, ranked in (("node_seed", ns), ("edge_seed", es)):
            m = score_one(ranked, ess, helpful); m["source"] = c["source"]; rows[arm].append(m)
        es_set = set(es)
        cover.append(len([g for g in ess if g in es_set]) / len(ess) if ess else 0.0)

    def agg(arm, rws):
        n = len(rws)
        mean = lambda k: sum(r[k] for r in rws if r.get(k) is not None) / max(n, 1)
        return (f"{arm:11s} n={n:3d} | hit@1 {mean('hit1_ess'):.0%}  hit@5 {mean('hit5_ess'):.0%}  "
                f"hit@25 {mean('hit25_ess'):.0%} | recall@5 {mean('recall5_ess'):.0%}  nDCG@5 {mean('ndcg5'):.2f}")

    print("\n" + "=" * 90)
    print("EDGE-SEED vs NODE-SEED — nomic-Q, 73-cue endo gold, identical scorer/cutoff")
    print("=" * 90)
    for arm in ("node_seed", "edge_seed"):
        print(agg(arm, rows[arm]))
    print(f"\nedge-seed gold reachability (essential gold that is a semantic-edge endpoint): {sum(cover)/len(cover):.0%}")
    for src in ("operator_msg", "anchor_turn"):
        print(f"  -- {src} --")
        for arm in ("node_seed", "edge_seed"):
            print("   " + agg(arm, [r for r in rows[arm] if r["source"] == src]))

    json.dump({"node_seed": rows["node_seed"], "edge_seed": rows["edge_seed"],
               "edge_reach": sum(cover) / len(cover)},
              open(os.path.join(HERE, "results", "edge_seed_ab.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
