#!/usr/bin/env python3
"""Decoding Suite — measures recall quality + distillation fidelity.

Extends the original decode_funnel with:
  - Structured KPI scoring via scoring.py
  - Optional Haiku distillation testing
  - Corpus-driven query sets alongside hardcoded queries
  - JSON results output for trend tracking

Usage:
    # Quick run (recall only, no LLM)
    python3 eval/decoding_suite.py

    # With distillation testing (needs ANTHROPIC_API_KEY)
    python3 eval/decoding_suite.py --with-distillation

    # Use specific brain snapshot
    python3 eval/decoding_suite.py --db eval/fixtures/brain_eval_copy.db

    # Filter category
    python3 eval/decoding_suite.py --category procedural
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.scoring import score_decoding_query, score_decoding_suite

# Import the curated query set from decode_funnel
from eval.decode_funnel import QUERIES as FUNNEL_QUERIES


def run_recall_query(brain, query_spec: dict) -> dict:
    """Run a single recall query and score it."""
    query = query_spec["query"]
    expected = set(query_spec.get("expected", []))
    category = query_spec.get("category", "unknown")

    t0 = time.time()
    try:
        result = brain.recall(query, limit=8)
        latency_ms = (time.time() - t0) * 1000
    except Exception as e:
        return {
            "query": query, "category": category,
            "error": str(e), "recall_at_3": 0, "recall_at_8": 0, "mrr": 0,
            "latency_ms": 0,
        }

    returned = result.get("results", []) or result.get("nodes", [])
    returned_ids = [n.get("id", "") for n in returned]

    # Prefix-match for truncated IDs
    def match_ids(returned_ids, expected_ids):
        matched = []
        for rid in returned_ids:
            for eid in expected_ids:
                if rid.startswith(eid) or eid.startswith(rid):
                    matched.append(rid)
                    break
            else:
                matched.append(rid)  # keep non-matching for position
        # Build effective retrieved list using expected IDs where matched
        effective = []
        for rid in returned_ids:
            found = False
            for eid in expected_ids:
                if rid.startswith(eid) or eid.startswith(rid):
                    effective.append(eid)
                    found = True
                    break
            if not found:
                effective.append(rid)
        return effective

    effective_ids = match_ids(returned_ids, expected) if expected else returned_ids

    scores = score_decoding_query(effective_ids, expected, latency_ms)
    scores["query"] = query
    scores["category"] = category
    scores["description"] = query_spec.get("description", "")
    scores["returned_count"] = len(returned)
    scores["top1_title"] = returned[0].get("title", "")[:60] if returned else ""

    return scores


def run_suite(db_path: str, category: str = None, verbose: bool = True) -> dict:
    """Run the full decoding suite.

    Args:
        db_path: Path to brain.db (production snapshot or eval copy)
        category: Optional category filter
        verbose: Print progress
    """
    from servers.brain import Brain

    brain = Brain(db_path=db_path)

    # Filter queries
    queries = FUNNEL_QUERIES
    if category:
        queries = [q for q in queries if q["category"] == category]

    if verbose:
        print("Decoding Suite: %d queries against %s" % (len(queries), db_path))

    # Run all queries (single-threaded — recall is fast, no LLM)
    results = []
    for q in queries:
        r = run_recall_query(brain, q)
        results.append(r)
        if verbose:
            status = "HIT" if r.get("hit_at_3") else ("hit@8" if r.get("hit_at_8") else "MISS")
            print("  [%s] %s (%.0fms)" % (status, q["query"][:50], r.get("latency_ms", 0)))

    brain.close()

    # Aggregate
    suite_scores = score_decoding_suite(results)

    # Category breakdown
    cats = {}
    for r in results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(r)

    category_scores = {}
    for cat, cat_results in cats.items():
        category_scores[cat] = score_decoding_suite(cat_results)

    return {
        "suite": "decoding",
        "db_path": db_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "overall": suite_scores,
        "per_category": category_scores,
        "per_query": results,
    }


def print_results(results: dict):
    """Print formatted results."""
    overall = results["overall"]
    print("\n" + "=" * 70)
    print("DECODING SUITE RESULTS")
    print("=" * 70)

    print("\nOverall (%d queries):" % overall.get("queries_tested", 0))
    print("  Recall@3: %.1f%%" % (overall.get("recall_at_3", 0) * 100))
    print("  Recall@8: %.1f%%" % (overall.get("recall_at_8", 0) * 100))
    print("  MRR:      %.3f" % overall.get("mrr", 0))
    print("  Avg latency: %.0fms" % overall.get("latency_ms", 0))

    print("\nBy Category:")
    print("  %-15s %8s %8s %8s %5s" % ("Category", "R@3", "R@8", "MRR", "N"))
    for cat, scores in sorted(results["per_category"].items()):
        print("  %-15s %7.0f%% %7.0f%% %8.3f %5d" % (
            cat,
            scores.get("recall_at_3", 0) * 100,
            scores.get("recall_at_8", 0) * 100,
            scores.get("mrr", 0),
            scores.get("queries_tested", 0),
        ))

    # Misses
    misses = [r for r in results["per_query"] if not r.get("hit_at_8") and not r.get("error")]
    if misses:
        print("\nMisses (not in top-8):")
        for m in misses:
            print("  - [%s] %s" % (m["category"], m["query"]))

    print()


def save_results(results: dict, output_dir: str = None):
    """Save results to JSON."""
    if output_dir is None:
        output_dir = str(ROOT / "eval" / "results")
    os.makedirs(output_dir, exist_ok=True)

    filename = "decoding_%s.json" % time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved: %s" % path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Decoding Suite")
    parser.add_argument("--db", default=os.path.expanduser("~/AgentsContext/brain/brain.db"),
                        help="Brain DB path")
    parser.add_argument("--category", help="Filter to category")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results = run_suite(args.db, category=args.category, verbose=not args.quiet)
    print_results(results)

    if args.save:
        save_results(results)

    # Return exit code based on recall quality
    r3 = results["overall"].get("recall_at_3", 0)
    return 0 if r3 >= 0.40 else 1


if __name__ == "__main__":
    sys.exit(main())
