#!/usr/bin/env python3
"""Substrate pack dump for the vector-behavior geometry bench (geometry.py).

Extends dump_corpus.py's node/cue dump with every OTHER text substrate the
LAF engine embeds, so geometry.py can compare embedders on vector behavior
per substrate — not just gold hit-rates:

  nodes.json        (via dump_corpus.main() — title/content + situation/question KV)
  cues.json         (via dump_corpus.main() — the 73-cue endo gold, bench.py compat)
  edges.json        edge-relation rows (relation, description, source, target)
                    + probe pair sets (correction pairs, community sibling pairs)
  episodic.json     rendered trace texts from trace_embeddings (what the
                    episodic pick/enc lanes actually rank)
  door1_cues.json   corpus_v2 walker bundles: per-turn query + gold id + ts
                    cutoff (the Door-1 corpus — 3.8k turns, ~700 with golds)
  pack_meta.json    counts + sha256 per file — future runs comparable by hash

Safety: same WAL-aware read-only snapshot pattern as dump_corpus.py; the
live DBs are never opened with a writer.

Run: ./dev python3 eval/oracle_audit/emb_bench/dump_pack.py
Out: /tmp/emb_bench/*.json
"""
import hashlib
import json
import os
import random
import shutil
import sqlite3
import time

import dump_corpus  # sibling module — owns nodes.json/cues.json + brain.db snapshot

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = dump_corpus.OUT                      # /tmp/emb_bench
SNAP = dump_corpus.SNAP                    # /tmp/emb_bench/snapshot
BUNDLES = os.path.join(REPO, "eval", "laf", "walker", "corpus_v2_bundles.jsonl")
ASPECTS = os.path.join(REPO, "servers", "scales", "s2", "aspects_v1.json")

DBDIR = os.environ.get("BRAIN_DB_DIR") or os.path.expanduser("~/AgentsContext/brain")
LOGS_LIVE = os.path.join(DBDIR, "brain_logs.db")

EPISODIC_SAMPLE = 6000
SEED = 20260807


def _snapshot_logs():
    """WAL-aware copy of brain_logs.db (same pattern as dump_corpus.snapshot)."""
    for suffix in ("", "-wal", "-shm"):
        src = LOGS_LIVE + suffix
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(SNAP, "brain_logs.db" + suffix))
    return os.path.join(SNAP, "brain_logs.db")


def dump_edges(con):
    """All live edge-relation rows + the two graph-derived probe pair sets."""
    rows = con.execute(
        """SELECT er.relation, er.description, e.source_id, e.target_id
           FROM edge_relations er JOIN edges e ON e.edge_id = er.edge_id
           WHERE COALESCE(er.archived, 0) = 0"""
    ).fetchall()

    aspects = json.load(open(ASPECTS))
    correction_rels = set(aspects["correction_improvement"]["edge_relations"])

    relations, correction_pairs, community_groups = [], [], {}
    for relation, description, src, tgt in rows:
        relations.append({
            "relation": relation, "description": description or "",
            "source_id": src, "target_id": tgt,
        })
        if relation in correction_rels:
            correction_pairs.append([src, tgt])
        elif relation == "community_member":
            # community node is the actor (source); member is the target
            community_groups.setdefault(src, []).append(tgt)

    # community sibling pairs: two members of the same community (same topic).
    rng = random.Random(SEED)
    sibling_pairs = []
    for members in community_groups.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        for _ in range(min(len(members), 6)):     # bounded per community
            a, b = rng.sample(members, 2)
            sibling_pairs.append([a, b])

    return {
        "relations": relations,
        "correction_pairs": correction_pairs,
        "community_sibling_pairs": sibling_pairs,
        "n_communities": len(community_groups),
    }


def dump_episodic(logs_db):
    """Rendered trace texts — exactly what the episodic lanes rank (the text
    column store_embeddings persists alongside each vector)."""
    con = sqlite3.connect(f"file:{logs_db}?mode=ro", uri=True)
    rows = con.execute(
        """SELECT trace_id, text, created_at FROM trace_embeddings
           WHERE text != '' ORDER BY trace_id"""
    ).fetchall()
    con.close()
    rng = random.Random(SEED)
    if len(rows) > EPISODIC_SAMPLE:
        rows = rng.sample(rows, EPISODIC_SAMPLE)
    return [{"trace_id": t, "text": txt, "created_at": c} for t, txt, c in rows]


def dump_door1():
    """corpus_v2 walker bundles → flat cue rows. gold_id may be None — those
    turns still serve the label-free ranking blocks; gold rows feed block E."""
    cues = []
    with open(BUNDLES) as f:
        for line in f:
            b = json.loads(line)
            gold = b.get("gold") or {}
            cues.append({
                "key": b["key"], "ts": b["ts"],
                "query": b["op_text"],
                "gold_id": gold.get("id"),
                "stratum": b.get("v0_stratum"),
            })
    return cues


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    t0 = time.time()
    dump_corpus.main()                       # nodes.json + cues.json + brain.db snapshot

    con = sqlite3.connect(
        f"file:{os.path.join(SNAP, 'brain.db')}?mode=ro", uri=True)
    edges = dump_edges(con)
    con.close()
    json.dump(edges, open(os.path.join(OUT, "edges.json"), "w"))

    episodic = dump_episodic(_snapshot_logs())
    json.dump(episodic, open(os.path.join(OUT, "episodic.json"), "w"))

    door1 = dump_door1()
    json.dump(door1, open(os.path.join(OUT, "door1_cues.json"), "w"))

    files = ["nodes.json", "cues.json", "edges.json", "episodic.json", "door1_cues.json"]
    meta = {
        "n_edge_relations": len(edges["relations"]),
        "n_correction_pairs": len(edges["correction_pairs"]),
        "n_community_sibling_pairs": len(edges["community_sibling_pairs"]),
        "n_episodic": len(episodic),
        "n_door1_cues": len(door1),
        "n_door1_with_gold": sum(1 for c in door1 if c["gold_id"]),
        "sha256_16": {f: _sha(os.path.join(OUT, f)) for f in files},
        "seed": SEED,
        "dumped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 2),
    }
    json.dump(meta, open(os.path.join(OUT, "pack_meta.json"), "w"), indent=1)
    print(f"edges: {meta['n_edge_relations']}  "
          f"correction_pairs: {meta['n_correction_pairs']}  "
          f"sibling_pairs: {meta['n_community_sibling_pairs']}")
    print(f"episodic: {meta['n_episodic']}  door1: {meta['n_door1_cues']} "
          f"({meta['n_door1_with_gold']} with gold)")
    print(f"pack complete -> {OUT} in {meta['elapsed_s']}s")


if __name__ == "__main__":
    main()
