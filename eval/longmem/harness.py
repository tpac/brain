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


def _snapshot_error_count(brain) -> int:
    """Count brain_errors rows now — returns a monotonic counter used to detect
    errors logged during a single item's ingest/query phase."""
    try:
        return brain.logs_conn.execute("SELECT COUNT(*) FROM brain_errors").fetchone()[0]
    except Exception:
        return 0


def _new_errors_since(brain, baseline_count: int) -> List[Dict[str, Any]]:
    """Return error rows logged since baseline_count. Limited to 20 for sanity."""
    try:
        rows = brain.logs_conn.execute(
            "SELECT error_type, error_message, context FROM brain_errors "
            "ORDER BY id DESC LIMIT ?", (20,)).fetchall()
    except Exception:
        return []
    current = _snapshot_error_count(brain)
    n_new = max(0, current - baseline_count)
    if n_new == 0:
        return []
    return [{"type": r[0], "message": (r[1] or "")[:200], "context": (r[2] or "")[:100]}
            for r in rows[:n_new]]


def run_item(item: Dict[str, Any], item_idx: int, total: int,
             run_name: str = None, keep_db: bool = False) -> Dict[str, Any]:
    """Run one item end-to-end. Each item gets its OWN brain DB for isolation.

    Per-item DB (brain-eval-{run_name}/{qid}/) means:
      - No reset_to_seeds leftovers (cross-item contamination impossible)
      - Inspectable post-hoc (keep_db=True preserves the DB for debugging)
      - Prerequisite for parallel execution (each process writes to its own file)

    Returns result dict with inline judge + failure class.
    """
    from eval.longmem.replay import replay_item, query_brain
    from eval.longmem.answerer import answer_question
    from eval.longmem.fresh_brain import create_fresh_eval_brain, per_item_brain_dir
    from eval.longmem.judge import judge_one
    from eval.longmem.classifier import classify_failure

    qid = item["question_id"]
    item_db_path = per_item_brain_dir(qid, run_name=run_name)
    brain = create_fresh_eval_brain(path=item_db_path, wipe=True)
    err_baseline = _snapshot_error_count(brain)

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
                               haystack_dates=item.get("haystack_dates"),
                               log_prefix=f"[item {item_idx+1}]")
    ingest_ms = int((time.time() - t0) * 1000)

    q_result = query_brain(brain, item["question"], item.get("question_date"))
    a_result = answer_question(item["question"], q_result["additional_context"],
                               item.get("question_date"))

    print(f"[harness] hypothesis: {a_result['hypothesis'][:200]}", flush=True)
    print(f"[harness] abstained: {a_result['abstained']}, had context: {a_result['has_context']}", flush=True)

    # Inline judge — grade now so classifier knows if this item needs diagnosis
    j = judge_one(item["question"], item["answer"], a_result["hypothesis"])
    correct = j["correct"]
    print(f"[harness] judge: {'✓' if correct else '✗'} ({j['raw']})", flush=True)

    # Classify failures while brain + traces are still live (before next reset_to_seeds)
    failure_info = {}
    if not correct:
        failure_info = classify_failure(
            brain, item["question"], item["answer"], a_result["hypothesis"],
            q_result["query_session_id"], a_result["has_context"], a_result["abstained"])
        print(f"[harness] failure: {failure_info['failure_bucket']} — {failure_info['failure_reason'][:140]}",
              flush=True)

    # Surface any silent errors logged during this item (prevents "passed the test
    # but something broke mid-ingest" blind spots).
    new_errors = _new_errors_since(brain, err_baseline)
    if new_errors:
        print(f"[harness] {len(new_errors)} new brain_error rows this item", flush=True)
        for e in new_errors[:5]:
            print(f"    {e['type']}: {e['message'][:120]}", flush=True)

    result = {
        "question_id": qid,
        "question_type": item["question_type"],
        "axis": axis,
        "question": item["question"],
        "answer_gold": item["answer"],
        "hypothesis": a_result["hypothesis"],
        "abstained": a_result["abstained"],
        "has_context": a_result["has_context"],
        "correct": correct,
        "judge_raw": j["raw"],
        "brain_errors_new": new_errors,
        **failure_info,
        "ingest": ingest_stats,
        "ingest_ms": ingest_ms,
        "query_s1r_ms": q_result["s1r_ms"],
        "query_session_id": q_result["query_session_id"],
        "answer_ms": a_result["elapsed_ms"],
        "answer_tokens_in": a_result.get("tokens_in", 0),
        "answer_tokens_out": a_result.get("tokens_out", 0),
        "total_item_ms": ingest_ms + q_result["s1r_ms"] + a_result["elapsed_ms"],
        "brain_db_path": item_db_path,
    }

    # Release the per-item brain's handles before any cleanup.
    try:
        brain.close()
    except Exception:
        pass

    # Cleanup per-item DB unless --keep_dbs was passed.
    if not keep_db:
        try:
            import shutil
            if os.path.isdir(item_db_path):
                shutil.rmtree(item_db_path)
        except Exception as e:
            print(f"[harness] cleanup failed for {qid}: {e}", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=2, help="per-axis item count (total = items × 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracle", default="eval/longmem/data/longmemeval_oracle.json")
    parser.add_argument("--run_name", default=None, help="report file suffix (default: timestamp)")
    parser.add_argument("--keep_dbs", action="store_true",
                        help="keep per-item brain DBs after each item (for post-hoc inspection)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes (default 1 = serial). Each worker loads its own embedder (~1GB).")
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

    # Compute run_name up front — each item's brain lives under brain-eval-{run_name}/{qid}/
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    t_run0 = time.time()
    if args.workers <= 1:
        # Serial path (backward compatible)
        for i, item in enumerate(picked):
            try:
                r = run_item(item, i, len(picked), run_name=run_name,
                             keep_db=args.keep_dbs)
                results.append(r)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[harness] item {i+1} FAILED: {e}", flush=True)
                results.append({"question_id": item["question_id"], "error": str(e)})
    else:
        # Parallel path — ProcessPoolExecutor, each worker loads its own embedder.
        # Per-item DBs are already isolated (brain-eval-{run_name}/{qid}/), so no contention.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"[harness] running {len(picked)} items across {args.workers} workers", flush=True)
        results_by_idx: Dict[int, Dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_item, item, i, len(picked), run_name, args.keep_dbs): (i, item)
                for i, item in enumerate(picked)
            }
            done_count = 0
            for fut in as_completed(futures):
                i, item = futures[fut]
                try:
                    r = fut.result()
                    results_by_idx[i] = r
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[harness] item {i+1} ({item['question_id']}) FAILED: {e}", flush=True)
                    results_by_idx[i] = {"question_id": item["question_id"], "error": str(e)}
                done_count += 1
                print(f"[harness] progress: {done_count}/{len(picked)} done", flush=True)
        # Preserve original order for reporting
        results = [results_by_idx[i] for i in range(len(picked))]
    total_ms = int((time.time() - t_run0) * 1000)

    # Write outputs
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    hypotheses_path = os.path.join(reports_dir, f"hypotheses_{run_name}.jsonl")
    with open(hypotheses_path, "w") as f:
        for r in results:
            if "hypothesis" in r:
                f.write(json.dumps({"question_id": r["question_id"], "hypothesis": r["hypothesis"]}) + "\n")

    # Aggregate inline-graded results
    graded = [r for r in results if "correct" in r]
    correct_count = sum(1 for r in graded if r["correct"])
    overall = correct_count / len(graded) if graded else 0
    by_axis: Dict[str, List[bool]] = {}
    by_bucket: Dict[str, int] = {}
    for r in graded:
        by_axis.setdefault(r["axis"], []).append(r["correct"])
        if not r["correct"] and r.get("failure_bucket"):
            by_bucket[r["failure_bucket"]] = by_bucket.get(r["failure_bucket"], 0) + 1

    report_path = os.path.join(reports_dir, f"run_{run_name}.json")
    with open(report_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "items_count": len(results),
            "correct_count": correct_count,
            "overall_score": overall,
            "axis_scores": {a: sum(v) / len(v) for a, v in by_axis.items() if v},
            "axis_counts": {a: len(v) for a, v in by_axis.items()},
            "failure_buckets": by_bucket,
            "total_ms": total_ms,
            "config": {"items_per_axis": args.items, "seed": args.seed},
            "results": results,
        }, f, indent=2)

    # Per-item brain dirs live at ~/AgentsContext/brain-eval-{run_name}/{qid}/.
    # If --keep_dbs was passed, run_item left them in place for inspection.
    # Otherwise they're cleaned per-item; the containing dir may be empty.
    if args.keep_dbs:
        base_dir = os.path.expanduser(f"~/AgentsContext/brain-eval-{run_name}")
        if os.path.isdir(base_dir):
            print(f"[harness] per-item brains preserved → {base_dir}", flush=True)

    print(f"\n[harness] done in {total_ms/1000:.1f}s")
    print(f"[harness] overall: {overall:.1%} ({correct_count}/{len(graded)})")
    print(f"[harness] hypotheses → {hypotheses_path}")
    print(f"[harness] report     → {report_path}")

    # Render friendly markdown report
    try:
        from eval.longmem.report import render_report
        md_path = render_report(report_path)
        print(f"[harness] markdown   → {md_path}")
    except Exception as e:
        print(f"[harness] report render failed: {e}", flush=True)


if __name__ == "__main__":
    main()
