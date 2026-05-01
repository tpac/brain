"""Re-query kept brain DBs with a different recall variant.

Use after running harness.py --keep_dbs. The kept per-item DBs already
contain ingested haystack (the slow part). This script loops over them
and re-runs ONLY the query phase under a different BRAIN_RECALL_VARIANT —
so A/B/C comparisons isolate the recall change, not ingest variance.

Usage:
    BRAIN_RECALL_VARIANT=cluster python3 eval/longmem/requery.py \\
        --run_name baseline_18item --variant_label B_cluster

Reads brain DBs from brain-eval-{run_name}/ (one per question_id).
Writes hypotheses_requery_{variant_label}.jsonl alongside the original
hypotheses JSONL.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True,
                        help="Name of the kept run (matches harness --run_name)")
    parser.add_argument("--variant_label", required=True,
                        help="Tag for output file (e.g. B_cluster, C_l4)")
    parser.add_argument("--oracle", default="eval/longmem/data/longmemeval_oracle.json")
    parser.add_argument("--report_dir", default="eval/longmem/reports")
    args = parser.parse_args()

    # Verify the variant flag is set — this script is variant-agnostic;
    # the env var drives the recall path.
    variant_env = os.environ.get('BRAIN_RECALL_VARIANT', 'baseline')
    print(f"[requery] BRAIN_RECALL_VARIANT={variant_env}", flush=True)
    print(f"[requery] writing as label={args.variant_label}", flush=True)

    # Load env from .env if present
    envf = Path(".env")
    if envf.exists():
        for line in envf.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                key, val = k.strip(), v.strip().strip('"').strip("'")
                if not os.environ.get(key):
                    os.environ[key] = val

    # Load oracle for question lookup
    with open(args.oracle) as f:
        oracle = {item["question_id"]: item for item in json.load(f)}

    from eval.longmem.replay import query_brain
    from eval.longmem.answerer import answer_question
    from eval.longmem.judge import judge_one
    from eval.longmem.fresh_brain import per_item_brain_dir
    from servers.brain import Brain

    # Find kept DBs by scanning the brain-eval-{run_name}/ dir
    base_dir = Path(os.path.expanduser("~/AgentsContext")) / f"brain-eval-{args.run_name}"
    if not base_dir.exists():
        print(f"[requery] FAIL: {base_dir} does not exist — did you run with --keep_dbs?",
              file=sys.stderr)
        sys.exit(1)

    item_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    print(f"[requery] found {len(item_dirs)} per-item DBs in {base_dir}", flush=True)

    out_path = Path(args.report_dir) / f"hypotheses_requery_{args.variant_label}.jsonl"
    results = []
    t_start = time.time()

    for i, item_dir in enumerate(item_dirs):
        qid = item_dir.name
        item = oracle.get(qid)
        if not item:
            print(f"[requery] {qid}: not in oracle (skipping)", flush=True)
            continue

        print(f"\n[requery] {i+1}/{len(item_dirs)} {qid} variant={variant_env}", flush=True)
        print(f"[requery] Q: {item['question'][:120]}", flush=True)

        # Open the kept brain
        try:
            brain = Brain(db_path=str(item_dir / "brain.db"))
        except Exception as e:
            print(f"[requery] {qid}: open failed — {e}", flush=True)
            continue

        try:
            t0 = time.time()
            q_result = query_brain(brain, item["question"], item.get("question_date"))
            a_result = answer_question(item["question"], q_result["additional_context"],
                                       item.get("question_date"))
            j = judge_one(item["question"], item["answer"], a_result["hypothesis"])
            correct = j["correct"]
            elapsed = int((time.time() - t0) * 1000)
            print(f"[requery] judge: {'✓' if correct else '✗'} ({j['raw']}) — {elapsed}ms", flush=True)
            print(f"[requery] hypothesis: {a_result['hypothesis'][:160]}", flush=True)

            results.append({
                "question_id": qid,
                "question_type": item["question_type"],
                "question": item["question"],
                "answer_gold": item["answer"],
                "hypothesis": a_result["hypothesis"],
                "abstained": a_result["abstained"],
                "has_context": a_result["has_context"],
                "correct": correct,
                "judge_raw": j["raw"],
                "additional_context_chars": len(q_result["additional_context"] or ""),
                "additional_context": q_result["additional_context"],
                "query_s1r_ms": q_result["s1r_ms"],
                "answer_ms": a_result["elapsed_ms"],
                "variant_env": variant_env,
                "variant_label": args.variant_label,
            })
        except Exception as e:
            print(f"[requery] {qid}: query failed — {e}", flush=True)
            results.append({"question_id": qid, "error": str(e),
                           "variant_label": args.variant_label})
        finally:
            try:
                brain.close()
            except Exception:
                pass

    # Write JSONL
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[requery] wrote {len(results)} hypotheses → {out_path}", flush=True)

    # Summary
    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    pct = 100 * correct / total if total else 0
    print(f"[requery] {args.variant_label}: {correct}/{total} = {pct:.1f}%", flush=True)
    print(f"[requery] wall clock: {(time.time() - t_start):.1f}s", flush=True)


if __name__ == "__main__":
    main()
