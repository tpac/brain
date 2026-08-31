"""Encoder-quality diff between two frozen corpora — the seed-pack A/B's
graph-side read.

The sweep (sweep.py) answers "did QA outcomes move"; this answers the register
question underneath it: the seed pack is the encoder's only early catalog, so
its prose register should transfer into what the encoder writes (situations,
questions, reasoning, quotes, edge whys). Walks each corpus item's frozen
brain.db read-only and aggregates per-arm stats over encoder-authored nodes,
plus verification rows (seed count + generation marker per brain) and the
gold-scan-hit-a-seed contamination check (brain id:9e3afc4d).

Second read, when sweep runs are passed: SEED CROWDING. A bigger pack is more
identity nodes competing for candidate slots in a nearly-empty graph — the
concrete way a richer pack could make recall WORSE. Joins each sweep rep's
recall candidates against the corpus graph to get the seed share of the
candidate list, per arm.

USE
    ./dev python3 eval/longmem/pack_quality.py --corpus-a <hash> --corpus-b <hash> \
        --labels oldpack,newpack [--sweep-a <run>] [--sweep-b <run>] \
        [--out eval/longmem/reports/pack_ab.json]
"""
import argparse
import json
import os
import re
import sqlite3
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.longmem.corpus import load_manifest

FIELD_KEYS = ("situation", "question", "reasoning",
              "their_raw_quote", "my_raw_quote")
# Developmental-seed register: self-revision / expiry language in encoder prose.
REVISE_MARKERS = re.compile(
    r"revis(?:e|ing) (?:this|it)|until (?:we|I|the)|expir|outgrow|scaffold|"
    r"supersede", re.I)
CORRECTION_SHAPE = re.compile(r"ASSUMED.+REALITY.+PATTERN", re.S)
WHEN_TRIGGER = re.compile(r"^\s*When\b", re.I)


def _ro(db_path):
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def item_stats(brain_dir: str, gold_match_ids=()) -> dict:
    con = _ro(os.path.join(brain_dir, "brain.db"))
    try:
        seed_count = con.execute(
            "SELECT COUNT(*) FROM nodes WHERE encoding_source='anchor:seed'"
        ).fetchone()[0]
        marker = (con.execute(
            "SELECT value FROM brain_meta WHERE key='seed_pack_generation'"
        ).fetchone() or ("",))[0]
        s2_count = con.execute(
            "SELECT COUNT(*) FROM nodes WHERE archived=0"
            " AND encoding_source LIKE 's2:%'").fetchone()[0]
        # Which seeds did the answerability gold-scan match? (contamination
        # read, brain id:9e3afc4d — the scan can satisfy gold from seed prose)
        gold_seed_hits = 0
        for mid in gold_match_ids:
            row = con.execute(
                "SELECT encoding_source FROM nodes WHERE id LIKE ? LIMIT 1",
                (str(mid) + "%",)).fetchone()
            if row and row[0] == "anchor:seed":
                gold_seed_hits += 1
        # The S1E scribe's writes land as encoding_source='anchor' in this
        # harness (brain_batch default); seeds are 'anchor:seed', S2 is 's2:%'.
        nodes = con.execute(
            "SELECT id, type, title, content, emotion, emotion_label, confidence"
            " FROM nodes WHERE archived=0 AND encoding_source = 'anchor'"
        ).fetchall()
        ids = [n[0] for n in nodes]
        kv = {}
        edges = []
        if ids:
            ph = ",".join("?" * len(ids))
            for nid, k, v in con.execute(
                    f"SELECT node_id, key, value FROM node_metadata_kv"
                    f" WHERE node_id IN ({ph})", ids):
                kv.setdefault(nid, {})[k] = v
            edges = con.execute(
                f"SELECT e.source_id, e.target_id, r.relation, r.description"
                f" FROM edges e JOIN edge_relations r ON r.edge_id = e.edge_id"
                f" WHERE r.archived=0 AND r.relation != 'community_member'"
                f" AND (e.source_id IN ({ph}) OR e.target_id IN ({ph}))",
                ids + ids).fetchall()
    finally:
        con.close()

    st = {
        "seed_count": seed_count,
        "marker": marker,
        "s2_nodes": s2_count,
        "gold_seed_hits": gold_seed_hits,
        "n_nodes": len(nodes),
        "types": {},
        "field_cov": {k: 0 for k in FIELD_KEYS},
        "when_trigger": 0,
        "content_chars": [],
        "situation_chars": [],
        "reasoning_chars": [],
        "emotion_nonneutral": 0,
        "abs_emotion": [],
        "confidences": [],
        "conf_at_1": 0,
        "corrections": 0,
        "corrections_shaped": 0,
        "revise_marker_nodes": 0,
        "n_edges": len(edges),
        "edges_with_desc": 0,
        "edge_desc_chars": [],
        "relations": {},
    }
    for nid, ntype, title, content, emotion, elabel, conf in nodes:
        st["types"][ntype] = st["types"].get(ntype, 0) + 1
        st["content_chars"].append(len(content or ""))
        meta = kv.get(nid, {})
        for k in FIELD_KEYS:
            if (meta.get(k) or "").strip():
                st["field_cov"][k] += 1
        situ = meta.get("situation") or ""
        if situ:
            st["situation_chars"].append(len(situ))
            if WHEN_TRIGGER.match(situ):
                st["when_trigger"] += 1
        if meta.get("reasoning"):
            st["reasoning_chars"].append(len(meta["reasoning"]))
        if (elabel or "neutral") != "neutral":
            st["emotion_nonneutral"] += 1
        st["abs_emotion"].append(abs(emotion or 0.0))
        if conf is not None:
            st["confidences"].append(conf)
            if conf >= 1.0:
                st["conf_at_1"] += 1
        if ntype == "correction":
            st["corrections"] += 1
            if CORRECTION_SHAPE.search(content or ""):
                st["corrections_shaped"] += 1
        if REVISE_MARKERS.search(content or ""):
            st["revise_marker_nodes"] += 1
    for _, _, rel, desc in edges:
        st["relations"][rel] = st["relations"].get(rel, 0) + 1
        if (desc or "").strip():
            st["edges_with_desc"] += 1
            st["edge_desc_chars"].append(len(desc))
    return st


def _pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def _source_map(brain_dir: str) -> dict:
    """8-char node id → encoding_source, for one frozen brain."""
    con = _ro(os.path.join(brain_dir, "brain.db"))
    try:
        return {r[0][:8]: (r[1] or "")
                for r in con.execute(
                    "SELECT id, encoding_source FROM nodes")}
    finally:
        con.close()


def _cand_id(c) -> str:
    """Candidate entries are dicts or 'id|title|score|type' strings."""
    if isinstance(c, dict):
        return (c.get("id") or c.get("node_id") or "")[:8]
    return str(c).split("|")[0].strip()[:8]


def seed_crowding(corpus_hash: str, run_name: str, reports_dir: str) -> dict:
    """Seed share of recall candidates and selections, per sweep rep.

    The bigger-pack risk: in a nearly-empty graph, identity nodes compete with
    the lived nodes an answer actually needs. If the new pack crowds candidate
    slots harder, that is a real cost even when the register improves.
    """
    manifest = load_manifest(corpus_hash)
    srcs = {}
    for it in manifest["items"]:
        srcs[it["qid"]] = _source_map(it["brain_dir"])
    items_dir = os.path.join(reports_dir, run_name, "items")
    if not os.path.isdir(items_dir):
        return {"error": f"no sweep artifacts at {items_dir}"}
    per_item, cand_tot, seed_tot, sel_tot, sel_seed = {}, 0, 0, 0, 0
    for leaf in sorted(os.listdir(items_dir)):
        blob_path = os.path.join(items_dir, leaf, "recall.json")
        if not os.path.exists(blob_path):
            continue
        with open(blob_path) as f:
            blob = json.load(f)
        qid = leaf.split("-r")[0]
        smap = srcs.get(qid, {})
        cands = [_cand_id(c) for c in blob.get("candidates", [])]
        sels = [_cand_id(s) for s in blob.get("selected", [])]
        n_seed = sum(1 for c in cands if smap.get(c) == "anchor:seed")
        n_sel_seed = sum(1 for s in sels if smap.get(s) == "anchor:seed")
        per_item[leaf] = {
            "candidates": len(cands), "seed_candidates": n_seed,
            "seed_pct": _pct(n_seed, len(cands)),
            "selected": len(sels), "seed_selected": n_sel_seed,
        }
        cand_tot += len(cands); seed_tot += n_seed
        sel_tot += len(sels); sel_seed += n_sel_seed
    return {
        "run": run_name,
        "candidates_total": cand_tot,
        "seed_candidates_total": seed_tot,
        "seed_share_of_candidates_pct": _pct(seed_tot, cand_tot),
        "selected_total": sel_tot,
        "seed_selected_total": sel_seed,
        "seed_share_of_selected_pct": _pct(sel_seed, sel_tot),
        "per_item": per_item,
    }


def arm_report(corpus_hash: str) -> dict:
    manifest = load_manifest(corpus_hash)
    if not manifest:
        raise SystemExit(f"no corpus {corpus_hash}")
    items = {}
    agg = None
    seed_gold_hits = []
    for it in manifest["items"]:
        match_ids = [m.get("node_id") for m in
                     (it.get("gold_scan", {}).get("matches") or [])
                     if isinstance(m, dict) and m.get("node_id")]
        st = item_stats(it["brain_dir"], gold_match_ids=match_ids)
        items[it["qid"]] = st
        if st["gold_seed_hits"]:
            seed_gold_hits.append(it["qid"])
        if agg is None:
            agg = {k: (dict(v) if isinstance(v, dict) else
                       list(v) if isinstance(v, list) else v)
                   for k, v in st.items() if k not in ("marker",)}
        else:
            for k, v in st.items():
                if k == "marker":
                    continue
                if isinstance(v, dict):
                    for dk, dv in v.items():
                        agg[k][dk] = agg[k].get(dk, 0) + dv
                elif isinstance(v, list):
                    agg[k].extend(v)
                else:
                    agg[k] += v
    n = agg["n_nodes"]
    def _mm(lst):
        return {"mean": round(statistics.mean(lst), 1),
                "median": statistics.median(lst)} if lst else {}
    summary = {
        "corpus_hash": corpus_hash,
        "label": manifest.get("label"),
        "seed_pack": (manifest.get("config") or {}).get("seed_pack"),
        "items": len(manifest["items"]),
        "answerable": manifest.get("answerable_count"),
        "markers": sorted({s["marker"] for s in items.values()}),
        "seed_counts": sorted({s["seed_count"] for s in items.values()}),
        "encoder_nodes_total": n,
        "s2_nodes_total": agg["s2_nodes"],
        "nodes_per_item": _mm([s["n_nodes"] for s in items.values()]),
        "types": dict(sorted(agg["types"].items(), key=lambda x: -x[1])),
        "field_cov_pct": {k: _pct(v, n) for k, v in agg["field_cov"].items()},
        "when_trigger_pct_of_situations": _pct(agg["when_trigger"],
                                               agg["field_cov"]["situation"]),
        "content_chars": _mm(agg["content_chars"]),
        "situation_chars": _mm(agg["situation_chars"]),
        "reasoning_chars": _mm(agg["reasoning_chars"]),
        "emotion_nonneutral_pct": _pct(agg["emotion_nonneutral"], n),
        "abs_emotion_mean": round(statistics.mean(agg["abs_emotion"]), 2)
                            if agg["abs_emotion"] else 0,
        "confidence_mean": round(statistics.mean(agg["confidences"]), 3)
                           if agg["confidences"] else None,
        "confidence_at_1_pct": _pct(agg["conf_at_1"], len(agg["confidences"])),
        "corrections": agg["corrections"],
        "corrections_shaped": agg["corrections_shaped"],
        "revise_marker_nodes_pct": _pct(agg["revise_marker_nodes"], n),
        "edges_touching_encoder_nodes": agg["n_edges"],
        "edges_per_node": round(agg["n_edges"] / n, 2) if n else 0,
        "edge_desc_pct": _pct(agg["edges_with_desc"], agg["n_edges"]),
        "edge_desc_chars": _mm(agg["edge_desc_chars"]),
        "relation_vocab": len(agg["relations"]),
        "relations": dict(sorted(agg["relations"].items(), key=lambda x: -x[1])),
        "gold_scan_seed_hits": sorted(set(seed_gold_hits)),
    }
    return {"summary": summary, "per_item": items}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-a", required=True)
    p.add_argument("--corpus-b", required=True)
    p.add_argument("--labels", default="A,B")
    p.add_argument("--sweep-a", default=None,
                   help="sweep run name for corpus-a → seed-crowding read")
    p.add_argument("--sweep-b", default=None,
                   help="sweep run name for corpus-b → seed-crowding read")
    p.add_argument("--out", default=None, help="write full JSON here")
    args = p.parse_args()
    la, lb = (args.labels.split(",") + ["A", "B"])[:2]

    a = arm_report(args.corpus_a)
    b = arm_report(args.corpus_b)

    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "reports")
    if args.sweep_a:
        a["seed_crowding"] = seed_crowding(args.corpus_a, args.sweep_a,
                                           reports_dir)
    if args.sweep_b:
        b["seed_crowding"] = seed_crowding(args.corpus_b, args.sweep_b,
                                           reports_dir)

    print(f"\n{'metric':<38} {la:>18} {lb:>18}")
    print("-" * 76)
    sa, sb = a["summary"], b["summary"]
    for k in ("items", "answerable", "encoder_nodes_total", "s2_nodes_total",
              "nodes_per_item",
              "field_cov_pct", "when_trigger_pct_of_situations",
              "content_chars", "situation_chars", "reasoning_chars",
              "emotion_nonneutral_pct", "abs_emotion_mean",
              "confidence_mean", "confidence_at_1_pct",
              "corrections", "corrections_shaped", "revise_marker_nodes_pct",
              "edges_touching_encoder_nodes", "edges_per_node",
              "edge_desc_pct", "edge_desc_chars", "relation_vocab",
              "seed_counts", "markers", "gold_scan_seed_hits"):
        va, vb = sa.get(k), sb.get(k)
        print(f"{k:<38} {json.dumps(va):>18} {json.dumps(vb):>18}")
    print(f"\ntypes {la}: {json.dumps(sa['types'])}")
    print(f"types {lb}: {json.dumps(sb['types'])}")
    print(f"relations {la}: {json.dumps(sa['relations'])}")
    print(f"relations {lb}: {json.dumps(sb['relations'])}")

    if a.get("seed_crowding") or b.get("seed_crowding"):
        print("\nseed crowding (share of recall candidates that are seeds)")
        for lbl, arm in ((la, a), (lb, b)):
            sc = arm.get("seed_crowding") or {}
            if sc.get("error"):
                print(f"  {lbl}: {sc['error']}")
                continue
            if not sc:
                continue
            print(f"  {lbl:<10} candidates={sc['candidates_total']:<5} "
                  f"seeds={sc['seed_candidates_total']:<4} "
                  f"({sc['seed_share_of_candidates_pct']}%)   "
                  f"selected={sc['selected_total']:<4} "
                  f"seed-selected={sc['seed_selected_total']} "
                  f"({sc['seed_share_of_selected_pct']}%)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({la: a, lb: b}, f, indent=2)
        print(f"\nfull report → {args.out}")


if __name__ == "__main__":
    main()
