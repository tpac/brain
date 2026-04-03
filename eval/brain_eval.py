#!/usr/bin/env python3
"""Brain Evaluation Framework — measures encoding, decoding, and the loop.

Usage:
    # Decode-only: test recall quality on current brain
    BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 eval/brain_eval.py --mode decode

    # Fresh brain: encode conversations then test recall
    python3 eval/brain_eval.py --mode fresh

    # Full loop: encode A, query A, encode B, query B, measure contamination
    python3 eval/brain_eval.py --mode loop

    # Compare two result files
    python3 eval/brain_eval.py --mode compare results/run_A.json results/run_B.json

    # Specific conversation only
    python3 eval/brain_eval.py --mode decode --conv conv_001

    # With a label for the result file
    python3 eval/brain_eval.py --mode decode --label "title_embedding_v1"
"""
import sys, os, json, argparse, time, shutil, sqlite3
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'servers'))

from eval.eval_kpis import (
    enc_completeness, enc_overencoding, enc_dedup, enc_metadata_richness,
    dec_recall_at_k, dec_mrr, dec_hub_concentration, dec_false_positive_rate,
    dec_cross_topic_contamination, format_kpi_summary,
)


def load_corpus(corpus_dir: str, conv_filter: str = None):
    """Load conversation corpus with annotations."""
    conversations = []
    for f in sorted(os.listdir(corpus_dir)):
        if not f.endswith('.json') or not f.startswith('conv_'):
            continue
        if conv_filter and conv_filter not in f:
            continue

        with open(os.path.join(corpus_dir, f)) as fh:
            data = json.load(fh)

        conv = {
            "id": data.get("id", f.replace(".json", "")),
            "file": f,
            "category": data.get("category", "unknown"),
            "exchanges": data.get("exchanges", []),
            "ground_truth": data.get("ground_truth", {}),
        }

        # Load extended annotations if they exist
        ann_path = os.path.join(corpus_dir, "annotations", f)
        if os.path.exists(ann_path):
            with open(ann_path) as af:
                conv["annotations"] = json.load(af)

        conversations.append(conv)

    return conversations


def run_decode_eval(brain_db_path: str, conversations: list, verbose: bool = True):
    """Run decoding evaluation against an existing brain.

    Uses the decode_funnel QUERIES (which have proper expected node IDs) plus
    any annotated queries from the corpus conversations.
    """
    from servers.brain import Brain

    db_file = os.path.join(brain_db_path, "brain.db") if os.path.isdir(brain_db_path) else brain_db_path
    brain = Brain(db_path=db_file)
    results = []

    # Primary source: decode_funnel queries (have proper expected node IDs)
    from eval.decode_funnel import QUERIES as FUNNEL_QUERIES

    for q in FUNNEL_QUERIES:
        query = q.get("query", "")
        expected = list(q.get("expected", []))
        cat = q.get("category", "unknown")

        if not query:
            continue

        recall_result = brain.recall(query=query, limit=25)
        returned = recall_result.get("results", [])

        result = {
            "query": query,
            "category": cat,
            "description": q.get("description", ""),
            "expected_ids": expected,
            "returned_ids": [r.get("id", "") for r in returned],
            "returned_scores": [r.get("effective_activation", 0) for r in returned],
            "returned_titles": [r.get("title", "")[:60] for r in returned],
        }
        results.append(result)

    brain.close()

    # Compute KPIs
    queries_with_expected = [r for r in results if r.get("expected_ids")]
    false_positive_queries = [r for r in results if r.get("category") == "false_positive"]
    all_results = results

    kpis = {
        "recall@8": dec_recall_at_k(queries_with_expected, k=8),
        "recall@25": dec_recall_at_k(queries_with_expected, k=25),
        "mrr": dec_mrr(queries_with_expected),
        "hub_concentration@8": dec_hub_concentration(all_results, k=8),
        "hub_concentration@25": dec_hub_concentration(all_results, k=25),
        "false_positive_rate": dec_false_positive_rate(false_positive_queries),
    }

    # Per-category breakdown
    categories = {}
    for r in queries_with_expected:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    category_kpis = {}
    for cat, cat_results in sorted(categories.items()):
        category_kpis[cat] = {
            "recall@8": dec_recall_at_k(cat_results, k=8),
            "recall@25": dec_recall_at_k(cat_results, k=25),
            "count": len(cat_results),
        }

    if verbose:
        print("\n" + "=" * 60)
        print("DECODE EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nQueries: {len(results)} total, {len(queries_with_expected)} with expected IDs, "
              f"{len(false_positive_queries)} false positive tests")
        print(f"\nOverall:")
        print(format_kpi_summary(kpis))

        print(f"\nPer category:")
        print(f"  {'Category':<20s} {'R@8':>6s} {'R@25':>6s} {'Count':>6s}")
        print(f"  {'-'*42}")
        for cat, ck in category_kpis.items():
            r8 = ck["recall@8"]["score"]
            r25 = ck["recall@25"]["score"]
            n = ck["count"]
            print(f"  {cat:<20s} {r8:>5.0%} {r25:>5.0%} {n:>6d}")

        # Show hub nodes
        hubs = kpis["hub_concentration@8"]
        if hubs.get("top_hubs"):
            print(f"\nTop hub nodes:")
            for h in hubs["top_hubs"][:5]:
                title = brain_db_path  # Can't lookup since brain is closed
                print(f"  {h['count']:3d}x  {h['id'][:12]}")

        # Show misses
        recall8 = kpis["recall@8"]
        missed = [q for q in recall8.get("per_query", []) if not q.get("found")]
        if missed:
            print(f"\nMissed queries ({len(missed)}):")
            for m in missed[:15]:
                print(f"  ✗ {m['query']}")

    return {"kpis": kpis, "category_kpis": category_kpis, "results": results, "mode": "decode"}


def run_fresh_eval(corpus_dir: str, conversations: list, verbose: bool = True):
    """Run full loop evaluation on a fresh brain.

    1. Create empty brain in /tmp
    2. For each conversation: encode exchanges, then run decode queries
    3. Measure encoding quality + decode quality + cross-contamination
    """
    # Create temp brain
    tmp_dir = '/tmp/brain_eval_%d' % int(time.time())
    os.makedirs(tmp_dir, exist_ok=True)

    from servers.brain import Brain
    brain = Brain(tmp_dir)

    all_kpis = {"encoding": {}, "decoding": {}, "loop": {}}
    all_results = []
    conv_node_ids = {}  # conv_id -> set of node IDs created

    for conv in conversations:
        gt = conv.get("ground_truth", {})
        exchanges = conv.get("exchanges", [])
        encode_targets = gt.get("encode_targets", [])
        decode_queries = gt.get("decode_queries", [])

        if verbose:
            print(f"\n--- {conv['id']} ({len(exchanges)} exchanges) ---")

        # Encode the conversation
        # Simulate what the encoding agent would see
        encoded_nodes = []
        for exchange in exchanges:
            role = exchange.get("role", "user")
            content = exchange.get("content", "")
            if role == "user":
                # Store in message stream for encoding agent context
                brain.store_exchange(content, "", session_id=conv["id"])

        # Run encoding agent on this conversation
        # For now, we use remember() directly with the encode targets
        # In the full implementation, we'd run the actual encoding agent
        created_ids = set()
        for target in encode_targets:
            try:
                result = brain.remember(
                    type=target.get("type", "lesson"),
                    title=target.get("topic", "Untitled"),
                    content="Encoded from eval conversation %s" % conv["id"],
                    project=conv["category"],
                )
                node_id = result.get("id", "")
                if node_id:
                    created_ids.add(node_id)
                    encoded_nodes.append({
                        "id": node_id,
                        "title": target.get("topic", ""),
                        "type": target.get("type", ""),
                    })
            except Exception as e:
                if verbose:
                    print(f"  encode error: {e}")

        conv_node_ids[conv["id"]] = created_ids

        # Encoding KPIs
        if encode_targets:
            completeness = enc_completeness(encoded_nodes, encode_targets)
            overencoding = enc_overencoding(encoded_nodes, encode_targets)
            all_kpis["encoding"][conv["id"]] = {
                "completeness": completeness,
                "overencoding": overencoding,
                "nodes_created": len(encoded_nodes),
            }
            if verbose:
                print(f"  Encoded: {len(encoded_nodes)} nodes, "
                      f"completeness: {completeness['score']:.0%}, "
                      f"overencoding: {overencoding['score']:.0%}")

        # Decode KPIs
        for q in decode_queries:
            query = q.get("query", "")
            if not query:
                continue

            recall_result = brain.recall(query=query, limit=25)
            returned = recall_result.get("results", [])

            result = {
                "query": query,
                "conv_id": conv["id"],
                "expected_ids": list(created_ids),  # All nodes from this conv
                "expected_topics": q.get("expected_topics", []),
                "returned_ids": [r.get("id", "") for r in returned],
                "returned_scores": [r.get("effective_activation", 0) for r in returned],
            }
            all_results.append(result)

    # Decode KPIs (aggregate across all conversations)
    queries_with_expected = [r for r in all_results if r.get("expected_ids")]
    all_kpis["decoding"] = {
        "recall@8": dec_recall_at_k(queries_with_expected, k=8),
        "recall@25": dec_recall_at_k(queries_with_expected, k=25),
        "mrr": dec_mrr(queries_with_expected),
        "hub_concentration@8": dec_hub_concentration(all_results, k=8),
    }

    # Loop KPI: sequential contamination
    # Check if conv A's nodes appear in conv B's queries
    from eval.eval_kpis import loop_sequential_contamination
    if len(conversations) >= 2:
        conv_a = conversations[0]
        conv_b_results = [r for r in all_results if r.get("conv_id") == conversations[1]["id"]]
        contamination = loop_sequential_contamination(
            conv_b_results, conv_node_ids.get(conv_a["id"], set()))
        all_kpis["loop"]["sequential_contamination"] = contamination

    brain.close()

    # Cleanup
    try:
        shutil.rmtree(tmp_dir)
    except:
        pass

    if verbose:
        print("\n" + "=" * 60)
        print("FRESH BRAIN EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nEncoding:")
        for conv_id, enc_kpis in all_kpis["encoding"].items():
            print(f"  {conv_id}: {enc_kpis['nodes_created']} nodes, "
                  f"completeness={enc_kpis['completeness']['score']:.0%}")
        print(f"\nDecoding:")
        print(format_kpi_summary(all_kpis["decoding"]))
        if "sequential_contamination" in all_kpis.get("loop", {}):
            sc = all_kpis["loop"]["sequential_contamination"]
            print(f"\nLoop:")
            print(f"  sequential_contamination: {sc['score']:.1%}")

    return {"kpis": all_kpis, "results": all_results, "mode": "fresh"}


def save_results(results: dict, label: str = "", results_dir: str = "eval/results"):
    """Save results with timestamp and label."""
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    label_part = f"_{label}" if label else ""
    mode = results.get("mode", "unknown")
    filename = f"{ts}_{mode}{label_part}.json"
    path = os.path.join(results_dir, filename)

    # Add metadata
    results["timestamp"] = ts
    results["label"] = label

    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")
    return path


def compare_results(path_a: str, path_b: str):
    """Compare two result files and show deltas."""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    print(f"\n{'='*60}")
    print(f"COMPARISON: {os.path.basename(path_a)} vs {os.path.basename(path_b)}")
    print(f"{'='*60}")

    kpis_a = a.get("kpis", {})
    kpis_b = b.get("kpis", {})

    # For decode mode, kpis are flat
    # For fresh mode, kpis are nested (encoding/decoding/loop)
    def flatten_kpis(kpis):
        flat = {}
        for k, v in kpis.items():
            if isinstance(v, dict) and "score" in v:
                flat[k] = v["score"]
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict) and "score" in v2:
                        flat[f"{k}.{k2}"] = v2["score"]
        return flat

    flat_a = flatten_kpis(kpis_a)
    flat_b = flatten_kpis(kpis_b)

    all_keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))

    print(f"\n{'KPI':<35s} {'A':>8s} {'B':>8s} {'Delta':>8s}")
    print("-" * 60)
    for key in all_keys:
        va = flat_a.get(key)
        vb = flat_b.get(key)
        if va is not None and vb is not None:
            delta = vb - va
            arrow = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "=")
            # For hub concentration and false positive, lower is better
            if "hub" in key or "false" in key or "contamination" in key or "overencoding" in key:
                arrow = "▲" if delta < -0.01 else ("▼" if delta > 0.01 else "=")
            print(f"  {key:<33s} {va:>7.1%} {vb:>7.1%} {delta:>+7.1%} {arrow}")
        elif va is not None:
            print(f"  {key:<33s} {va:>7.1%} {'n/a':>8s}")
        elif vb is not None:
            print(f"  {key:<33s} {'n/a':>8s} {vb:>7.1%}")


def main():
    parser = argparse.ArgumentParser(description="Brain Evaluation Framework")
    parser.add_argument("--mode", choices=["decode", "fresh", "loop", "compare"],
                        default="decode", help="Evaluation mode")
    parser.add_argument("--conv", default=None, help="Filter to specific conversation")
    parser.add_argument("--label", default="", help="Label for result file")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("files", nargs="*", help="Result files for compare mode")
    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    corpus_dir = str(Path(__file__).resolve().parent / "corpus")

    if args.mode == "compare":
        if len(args.files) != 2:
            print("Compare mode requires exactly 2 result files")
            sys.exit(1)
        compare_results(args.files[0], args.files[1])
        return

    conversations = load_corpus(corpus_dir, args.conv)
    if not conversations:
        print(f"No conversations found in {corpus_dir}")
        sys.exit(1)

    print(f"Loaded {len(conversations)} conversations")

    if args.mode == "decode":
        brain_db_dir = os.environ.get("BRAIN_DB_DIR", "")
        if not brain_db_dir:
            print("Set BRAIN_DB_DIR environment variable")
            sys.exit(1)
        results = run_decode_eval(brain_db_dir, conversations, args.verbose)

    elif args.mode == "fresh":
        results = run_fresh_eval(corpus_dir, conversations, args.verbose)

    elif args.mode == "loop":
        results = run_fresh_eval(corpus_dir, conversations, args.verbose)

    save_results(results, args.label)


if __name__ == "__main__":
    main()
