"""Harness — stratified micro-suite runner.

Orchestrates:
  1. Load LongMemEval oracle JSON
  2. Select stratified N items (default: 10 = 2 per axis × 5 axes)
  3. For each item: fresh brain (reset to seeds) → replay haystack → query → answerer → record
  4. Write hypothesis.jsonl + a run report

Usage:
  python3 eval/longmem/harness.py              # run with defaults
  python3 eval/longmem/harness.py --items 5    # smaller for debugging
  python3 eval/longmem/harness.py --seed 42    # reproducible selection
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 5 axes per LongMemEval paper; information-extraction collapses the three single-session types
AXES = {
    "info_extraction": ["single-session-user", "single-session-assistant", "single-session-preference"],
    "multi_session": ["multi-session"],
    "temporal": ["temporal-reasoning"],
    "knowledge_update": ["knowledge-update"],
    "abstention": None,  # question_id suffix "_abs" rather than question_type
}


def _item_axis(item: Dict[str, Any]) -> str:
    """Classify an item into one of 5 axes."""
    if item["question_id"].endswith("_abs"):
        return "abstention"
    qt = item["question_type"]
    for axis, types in AXES.items():
        if types and qt in types:
            return axis
    return "other"


def stratified_sample(data: List[Dict[str, Any]], per_axis: int = 2,
                      seed: int = 42) -> List[Dict[str, Any]]:
    """Pick `per_axis` items from each of the 5 axes. Returns ~10 items."""
    rng = random.Random(seed)
    by_axis: Dict[str, List[Dict[str, Any]]] = {a: [] for a in AXES}
    for item in data:
        a = _item_axis(item)
        if a in by_axis:
            by_axis[a].append(item)

    picked = []
    for axis, items in by_axis.items():
        if not items:
            print(f"[harness] WARN axis {axis} has no items, skipping", flush=True)
            continue
        # Prefer smaller items for speed (sort by total turn count)
        items_sorted = sorted(items, key=lambda i: sum(len(s) for s in i.get("haystack_sessions", [])))
        # Take from the bottom half (smaller) for efficiency
        pool = items_sorted[: max(per_axis * 4, len(items_sorted) // 2)]
        rng.shuffle(pool)
        picked.extend(pool[:per_axis])
    return picked


def run_item(brain, item: Dict[str, Any], item_idx: int, total: int) -> Dict[str, Any]:
    """Run one item end-to-end. Returns result dict."""
    from eval.longmem.replay import replay_item, query_brain
    from eval.longmem.answerer import answer_question
    from eval.longmem.fresh_brain import reset_to_seeds

    # Reset to seeds — each item is independent per LongMemEval semantics
    reset_to_seeds(brain)

    qid = item["question_id"]
    axis = _item_axis(item)
    n_turns = sum(len(s) for s in item.get("haystack_sessions", []))
    print(f"\n{'='*70}")
    print(f"[harness] item {item_idx+1}/{total} qid={qid} axis={axis} turns={n_turns}", flush=True)
    gold_str = str(item['answer']) if not isinstance(item['answer'], str) else item['answer']
    print(f"[harness] Q: {item['question'][:120]}", flush=True)
    print(f"[harness] A (gold): {gold_str[:120]}", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    ingest_session_id = f"ingest-{qid}"
    ingest_stats = replay_item(brain, ingest_session_id, item["haystack_sessions"],
                               log_prefix=f"[item {item_idx+1}]")
    ingest_ms = int((time.time() - t0) * 1000)

    q_result = query_brain(brain, item["question"], item.get("question_date"))
    a_result = answer_question(item["question"], q_result["additional_context"],
                               item.get("question_date"))

    print(f"[harness] hypothesis: {a_result['hypothesis'][:200]}", flush=True)
    print(f"[harness] abstained: {a_result['abstained']}, had context: {a_result['has_context']}", flush=True)

    return {
        "question_id": qid,
        "question_type": item["question_type"],
        "axis": axis,
        "question": item["question"],
        "answer_gold": item["answer"],
        "hypothesis": a_result["hypothesis"],
        "abstained": a_result["abstained"],
        "has_context": a_result["has_context"],
        "ingest": ingest_stats,
        "ingest_ms": ingest_ms,
        "query_s1r_ms": q_result["s1r_ms"],
        "answer_ms": a_result["elapsed_ms"],
        "answer_tokens_in": a_result.get("tokens_in", 0),
        "answer_tokens_out": a_result.get("tokens_out", 0),
        "total_item_ms": ingest_ms + q_result["s1r_ms"] + a_result["elapsed_ms"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=2, help="per-axis item count (total = items × 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracle", default="eval/longmem/data/longmemeval_oracle.json")
    parser.add_argument("--run_name", default=None, help="report file suffix (default: timestamp)")
    args = parser.parse_args()

    # Load env — override empty vars (setdefault skips empty strings, per known bug)
    from pathlib import Path
    envf = Path(".env")
    if envf.exists():
        for line in envf.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                key, val = k.strip(), v.strip().strip('"').strip("'")
                if not os.environ.get(key):  # missing OR empty
                    os.environ[key] = val

    with open(args.oracle) as f:
        data = json.load(f)
    print(f"[harness] loaded {len(data)} oracle items from {args.oracle}", flush=True)

    picked = stratified_sample(data, per_axis=args.items, seed=args.seed)
    print(f"[harness] selected {len(picked)} items:", flush=True)
    for i, item in enumerate(picked):
        axis = _item_axis(item)
        n = sum(len(s) for s in item.get("haystack_sessions", []))
        print(f"  {i+1}. {item['question_id']:<24} axis={axis:<18} turns={n}", flush=True)

    from eval.longmem.fresh_brain import create_fresh_eval_brain
    brain = create_fresh_eval_brain()

    results = []
    t_run0 = time.time()
    for i, item in enumerate(picked):
        try:
            r = run_item(brain, item, i, len(picked))
            results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[harness] item {i+1} FAILED: {e}", flush=True)
            results.append({"question_id": item["question_id"], "error": str(e)})
    total_ms = int((time.time() - t_run0) * 1000)

    # Write outputs
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    hypotheses_path = os.path.join(reports_dir, f"hypotheses_{run_name}.jsonl")
    with open(hypotheses_path, "w") as f:
        for r in results:
            if "hypothesis" in r:
                f.write(json.dumps({"question_id": r["question_id"], "hypothesis": r["hypothesis"]}) + "\n")

    report_path = os.path.join(reports_dir, f"run_{run_name}.json")
    with open(report_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "items_count": len(results),
            "total_ms": total_ms,
            "config": {"items_per_axis": args.items, "seed": args.seed},
            "results": results,
        }, f, indent=2)

    print(f"\n[harness] done in {total_ms/1000:.1f}s")
    print(f"[harness] hypotheses → {hypotheses_path}")
    print(f"[harness] report     → {report_path}")


if __name__ == "__main__":
    main()
