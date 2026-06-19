#!/usr/bin/env python3
"""Dump the node universe + the 73-cue gold corpus to flat JSON for the
embedding-model head-to-head bench.

Why a dump (not live recall): the bench re-embeds the WHOLE node set with each
candidate model and ranks by pure cosine — model-isolating. That work happens in
a scratch venv (sentence-transformers/torch) that must not import the brain. So
we extract once, here, under the brain venv, and hand the candidates a plain JSON.

Safety: never opens the LIVE brain.db with a writer. Copies brain.db (+ -wal/-shm
for a consistent WAL snapshot, exactly as tests/isolated_brain.py does) to a temp
snapshot, then reads that copy read-only. The daemon is untouched.

Run: ./dev python3 eval/oracle_audit/emb_bench/dump_corpus.py
Out: /tmp/emb_bench/nodes.json, /tmp/emb_bench/cues.json, /tmp/emb_bench/meta.json
"""
import json, os, shutil, sqlite3, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "..", "endo_corpus", "endo_gold_corpus.json")
OUT = "/tmp/emb_bench"
SNAP = os.path.join(OUT, "snapshot")
os.makedirs(SNAP, exist_ok=True)

DBDIR = os.environ.get("BRAIN_DB_DIR") or os.path.expanduser("~/AgentsContext/brain")
LIVE = os.path.join(DBDIR, "brain.db")

# KV fields we keep (everything Mode A needs is title+content from the nodes
# table; the rest enable a later full-engine Mode B without a second dump).
KV_KEYS = ("situation", "question", "reasoning", "user_raw_quote",
           "anchor_raw_quote", "correction_pattern", "source_context", "keywords")


def snapshot():
    """Consistent WAL-aware copy of the live brain.db into SNAP."""
    for suffix in ("", "-wal", "-shm"):
        src = LIVE + suffix
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(SNAP, "brain.db" + suffix))
    return os.path.join(SNAP, "brain.db")


def main():
    t0 = time.time()
    db = snapshot()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        """SELECT id, type, title, content, created_at, emotion, access_count,
                  confidence
           FROM nodes
           WHERE COALESCE(archived,0)=0 AND title IS NOT NULL"""
    ).fetchall()

    nodes = {}
    for r in rows:
        nodes[r["id"]] = {
            "id": r["id"], "type": r["type"],
            "title": r["title"] or "", "content": r["content"] or "",
            "created_at": r["created_at"],
            "emotion": r["emotion"] or 0.0,
            "access_count": r["access_count"] or 0,
            "confidence": r["confidence"],
        }

    # attach kv fields in one scan
    qmarks = ",".join("?" * len(KV_KEYS))
    for nid, key, val in con.execute(
        f"SELECT node_id, key, value FROM node_metadata_kv WHERE key IN ({qmarks})",
        KV_KEYS,
    ).fetchall():
        if nid in nodes and val:
            nodes[nid][key] = val
    con.close()

    node_list = list(nodes.values())
    gold = json.load(open(GOLD))

    # sanity: how many gold ids actually exist in the (active) node set
    node_ids = set(nodes)
    missing = set()
    for c in gold:
        for g in c["gold_essential"] + c.get("gold_helpful", []):
            if g not in node_ids:
                missing.add(g)

    json.dump(node_list, open(os.path.join(OUT, "nodes.json"), "w"))
    json.dump(gold, open(os.path.join(OUT, "cues.json"), "w"))
    clens = sorted(len(n["title"]) + len(n["content"]) for n in node_list)
    meta = {
        "n_nodes": len(node_list),
        "n_cues": len(gold),
        "node_doc_charlen_min_med_max": [clens[0], clens[len(clens) // 2], clens[-1]],
        "gold_ids_missing_from_active_nodes": sorted(missing),
        "live_db": LIVE,
        "dumped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 2),
    }
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)

    print(f"nodes: {len(node_list)}  cues: {len(gold)}")
    print(f"doc charlen (title+content) min/med/max: {meta['node_doc_charlen_min_med_max']}")
    print(f"gold ids missing from active nodes: {len(missing)}"
          + (f" -> {sorted(missing)[:8]}" if missing else " (all gold present)"))
    print(f"wrote -> {OUT}/{{nodes,cues,meta}}.json in {meta['elapsed_s']}s")


if __name__ == "__main__":
    main()
