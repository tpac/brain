"""Stage 2 — sweep recall over a frozen corpus (fast, zero encoding).

Loads the per-item brains a corpus build froze, then runs only the cheap
suffix — query → answer → judge → classify — under a given recall/surface
config, optionally N times for variance. No haystack replay, no S1E, no S2.
That's ~100× cheaper per item than a full run, which is what turns recall
iteration from hours into minutes.

Honest A/B: two sweeps over the SAME corpus_hash with different --surface
differ ONLY in recall. Each sweep copies the frozen brain into its own work
dir, so both arms start from byte-identical encoded graphs and the query
phase can't mutate the shared corpus.

New diagnostic — S2-reached-recall: for each query we check whether any
S2-origin node (community / consolidation / healer, `encoding_source` like
's2:%') showed up among recall candidates / selected. That's the direct
probe for the "communities-not-in-recall" bug — the fractal loop where an
S2 Δ should become S1R's next O and doesn't.

Per-item artifacts are written in the same shape harness.py produces
(reports/{run}/items/{qid}/{result,meta,recall}.json), so compare_arms.py
and cost_summary.py work unchanged (encoder cost reads $0 — correct, the
sweep does no encoding).

USE
    ./dev python3 eval/longmem/sweep.py --corpus c7f3a1 --label armA_v8
    ./dev python3 eval/longmem/sweep.py --corpus c7f3a1 --surface eval/prompts/surface_v9.txt --label armB_v9 --variance 3
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.longmem.harness import _apply_surface_override
from eval.longmem.replay import query_brain
from eval.longmem.answerer import answer_question
from eval.longmem.judge import judge_one
from eval.longmem.classifier import classify_failure, _read_s1r_trace
from eval.longmem.fresh_brain import create_fresh_eval_brain
from eval.longmem.corpus import load_manifest, corpus_item_dir


def _load_env() -> None:
    envf = Path(".env")
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            key, val = k.strip(), v.strip().strip('"').strip("'")
            if not os.environ.get(key):
                os.environ[key] = val


def _norm_id(x) -> str:
    """Normalize a candidate/selected entry to an 8-char id key."""
    if isinstance(x, dict):
        x = x.get("id") or x.get("node_id") or ""
    return (str(x) or "")[:8]


def _s2_origin_ids(brain) -> set:
    """8-char ids of live S2-authored nodes (community / consolidation / healer)."""
    try:
        rows = brain.conn.execute(
            "SELECT id FROM nodes WHERE archived = 0 AND encoding_source LIKE 's2:%'"
        ).fetchall()
    except Exception:
        return set()
    return {_norm_id(r[0]) for r in rows}


def _s2_reach(s2_ids: set, trace: dict) -> dict:
    """Did any S2-origin node surface for this query? The loop-closure check."""
    if not trace:
        return {"s2_total": len(s2_ids), "s2_in_candidates": 0, "s2_in_selected": 0}
    cand = {_norm_id(c) for c in trace.get("candidates", [])}
    sel = {_norm_id(s) for s in trace.get("selected", [])}
    return {
        "s2_total": len(s2_ids),
        "s2_in_candidates": len(s2_ids & cand),
        "s2_in_selected": len(s2_ids & sel),
    }


def _write_item_artifacts(reports_dir, run_name, artifact_qid, result, recall_blob, meta_blob):
    """Mirror harness's per-item artifact shape so compare_arms/cost_summary work."""
    item_dir = os.path.join(reports_dir, run_name, "items", artifact_qid)
    os.makedirs(item_dir, exist_ok=True)
    for fname, blob in (("result.json", result), ("recall.json", recall_blob),
                        ("meta.json", meta_blob)):
        try:
            with open(os.path.join(item_dir, fname), "w") as f:
                json.dump(blob, f, indent=2)
        except Exception as e:
            print(f"[sweep] artifact write failed {fname}: {e}", flush=True)


def sweep(corpus_hash: str, surface: str, variance: int, label: str,
          qids: str = None) -> str:
    _load_env()
    manifest = load_manifest(corpus_hash)
    if not manifest:
        print(f"[sweep] no corpus {corpus_hash} — build it first with build_corpus.py",
              file=sys.stderr)
        sys.exit(1)

    if surface != "active":
        os.environ["BRAIN_SURFACE_VARIANT"] = "v5_agentic"
        if not Path(surface).exists():
            print(f"[sweep] --surface path not found: {surface}", file=sys.stderr)
            sys.exit(2)

    # Score EVERY item. The answerability gate's gold-scan requires ALL gold
    # terms in ONE node, but composed / multi-session answers are spread across
    # MANY nodes — no single node carries all the terms — so a strict-scan
    # "unanswerable" is often a FALSE negative (verified: bc149d6b had 13 richly
    # on-topic nodes yet scanned unanswerable). The scan can't reliably exclude,
    # so don't hard-exclude on it. Score all items and let the per-item failure
    # bucket separate genuine ENCODE_MISS (coverage) from recall-pipeline
    # failures. Abstention items are scored too (the test is whether they abstain).
    items = list(manifest["items"])
    gate_unanswerable = [it["qid"] for it in items
                         if not it.get("answerable") and it.get("axis") != "abstention"]
    if qids:
        wanted = {q.strip() for q in qids.split(",") if q.strip()}
        items = [it for it in items if it["qid"] in wanted]
    print(f"[sweep] corpus {corpus_hash} loaded — scoring {len(items)} item(s); "
          f"{len(gate_unanswerable)} flagged unanswerable by the gate, scored anyway "
          f"for coverage: {gate_unanswerable}", flush=True)
    print(f"[sweep] surface={'override:'+surface if surface != 'active' else 'active'} "
          f"variance={variance}", flush=True)

    run_name = label or f"sweep_{corpus_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    work_root = os.path.expanduser(f"~/AgentsContext/sweep-work-{run_name}")
    os.makedirs(work_root, exist_ok=True)

    results = []
    t_run0 = time.time()

    for it in items:
        qid = it["qid"]
        axis = it["axis"]
        question = it["question"]
        gold = it["gold"]
        qdate = it.get("question_date")

        # Copy the frozen brain into this run's work dir → byte-identical start,
        # query phase can't mutate the shared corpus.
        src = corpus_item_dir(corpus_hash, qid)
        work_qid = os.path.join(work_root, qid)
        if os.path.isdir(work_qid):
            shutil.rmtree(work_qid)
        # Skip brain-*.json scratch dumps (encoding-prompt / surface-selected /
        # judge-result / recall-candidates): since BRAIN_TMP_DIR now routes
        # ephemeral files into the brain dir, the build phase drops them in the
        # corpus dir. The frozen work copy should hold only the DB snapshot, not
        # build-time scratch. (brain.db / brain.db-wal / brain.db-shm don't match
        # the 'brain-*.json' pattern, so the DB is copied intact.)
        shutil.copytree(src, work_qid,
                        ignore=shutil.ignore_patterns('brain-*.json'))

        brain = create_fresh_eval_brain(path=work_qid, wipe=False)
        if surface != "active":
            _apply_surface_override(brain, surface)
        s2_ids = _s2_origin_ids(brain)

        for rep in range(variance):
            artifact_qid = qid if variance == 1 else f"{qid}-r{rep}"
            t0 = time.time()
            qr = query_brain(brain, question, qdate)
            ar = answer_question(question, qr["additional_context"], qdate)
            j = judge_one(question, gold, ar["hypothesis"])
            correct = j["correct"]

            trace = _read_s1r_trace(brain, qr["query_session_id"]) or {}
            reach = _s2_reach(s2_ids, trace)

            failure_info = {}
            if not correct:
                if axis == "abstention":
                    # An abstention miss = fabricated an answer when the haystack
                    # had none. The recall-funnel classifier would mislabel it
                    # (gold legitimately absent → spurious ENCODE_MISS), so bucket
                    # it directly instead.
                    failure_info = {
                        "failure_bucket": "ABSTENTION_FAIL",
                        "failure_reason": "did not abstain — fabricated an answer when the haystack had none",
                    }
                else:
                    failure_info = classify_failure(
                        brain, question, gold, ar["hypothesis"],
                        qr["query_session_id"], ar["has_context"], ar["abstained"])

            total_ms = int((time.time() - t0) * 1000)
            mark = "✓" if correct else "✗"
            bucket = failure_info.get("failure_bucket", "")
            print(f"[sweep] {artifact_qid:<28} axis={axis:<16} {mark} "
                  f"{bucket} | s2_reach cand={reach['s2_in_candidates']}/{reach['s2_total']} "
                  f"sel={reach['s2_in_selected']}", flush=True)

            result = {
                "question_id": qid,
                "variance_idx": None if variance == 1 else rep,
                "axis": axis,
                "question": question,
                "answer_gold": gold,
                "hypothesis": ar["hypothesis"],
                "abstained": ar["abstained"],
                "has_context": ar["has_context"],
                "correct": correct,
                "judge_raw": j["raw"],
                "comparison": j.get("comparison", ""),
                "judge_reasoning": j.get("reasoning", ""),
                "s2_reach": reach,
                **failure_info,
                "ingest_ms": 0,            # sweep does no encoding
                "s1r_ms": qr["s1r_ms"],
                "answer_ms": ar["elapsed_ms"],
                "total_item_ms": total_ms,
            }
            recall_blob = {
                "query_session_id": qr["query_session_id"],
                "query": trace.get("query", question),
                "candidates": trace.get("candidates", []),
                "selected": trace.get("selected", []),
                "context": trace.get("context", qr.get("additional_context", "")),
                # Probe-fidelity + per-arm tool economics: the agentic surface
                # always writes a non-empty tool_trace (a no-fire run still has
                # its round record); v4 single-shot writes []. Without these,
                # tool_trace_summary reads nothing and an A/B can't prove it
                # exercised the agentic path.
                "tool_trace": trace.get("tool_trace", []),
                "surface_variant": trace.get("surface_variant", ""),
                "s2_reach": reach,
                "answerer_response": {
                    "hypothesis": ar["hypothesis"],
                    "abstained": ar["abstained"],
                    "has_context": ar["has_context"],
                    "tokens_in": ar.get("tokens_in", 0),
                    "tokens_out": ar.get("tokens_out", 0),
                    "elapsed_ms": ar.get("elapsed_ms", 0),
                },
            }
            meta_blob = {"axis": axis, "question": question, "gold": gold,
                         "corpus_hash": corpus_hash}
            _write_item_artifacts(reports_dir, run_name, artifact_qid,
                                  result, recall_blob, meta_blob)
            results.append(result)

        try:
            brain.close()
        except Exception:
            pass
        shutil.rmtree(work_qid, ignore_errors=True)

    shutil.rmtree(work_root, ignore_errors=True)
    total_ms = int((time.time() - t_run0) * 1000)

    _write_run_report(reports_dir, run_name, corpus_hash, surface, variance,
                      results, total_ms, gate_unanswerable)
    return run_name


def _stats(hits):
    floats = [1.0 if h else 0.0 for h in hits]
    return {"mean": sum(floats) / len(floats) if floats else 0,
            "stddev": statistics.pstdev(floats) if len(floats) > 1 else 0.0,
            "n": len(floats)}


def _write_run_report(reports_dir, run_name, corpus_hash, surface, variance,
                      results, total_ms, gate_unanswerable):
    correct_count = sum(1 for r in results if r["correct"])
    overall = correct_count / len(results) if results else 0
    by_axis, by_qid, by_bucket, by_comparison = {}, {}, {}, {}
    reach_cand_sum = reach_sel_sum = reach_total_sum = reps_with_s2 = 0
    for r in results:
        by_axis.setdefault(r["axis"], []).append(r["correct"])
        by_qid.setdefault(r["question_id"], []).append(r["correct"])
        if r.get("comparison"):
            by_comparison[r["comparison"]] = by_comparison.get(r["comparison"], 0) + 1
        if not r["correct"] and r.get("failure_bucket"):
            by_bucket[r["failure_bucket"]] = by_bucket.get(r["failure_bucket"], 0) + 1
        reach = r.get("s2_reach", {})
        reach_cand_sum += reach.get("s2_in_candidates", 0)
        reach_sel_sum += reach.get("s2_in_selected", 0)
        reach_total_sum += reach.get("s2_total", 0)
        if reach.get("s2_total", 0) > 0:
            reps_with_s2 += 1

    axis_stats = {a: _stats(v) for a, v in by_axis.items() if v}
    per_qid_stats = {q: _stats(v) for q, v in by_qid.items()} if variance > 1 else {}

    # Recall-conditional pass rate: drop genuine ENCODE_MISS items from the
    # denominator so encode-coverage gaps don't mask the recall signal. (Raw
    # `overall` keeps everything; this isolates "recall quality given the fact
    # was encoded." ABSTENTION_FAIL stays in — abstaining IS the test there.)
    encode_miss = by_bucket.get("ENCODE_MISS", 0)
    recall_denom = len(results) - encode_miss
    recall_conditional = correct_count / recall_denom if recall_denom else 0.0

    report = {
        "run_name": run_name,
        "stage": "sweep",
        "corpus_hash": corpus_hash,
        "surface": surface,
        "variance": variance,
        "items_count": len(results),
        "correct_count": correct_count,
        "overall_score": overall,
        "axis_scores": {a: s["mean"] for a, s in axis_stats.items()},
        "axis_counts": {a: s["n"] for a, s in axis_stats.items()},
        "axis_stats": axis_stats,
        "per_qid_stats": per_qid_stats,
        "comparison_counts": by_comparison,
        "failure_buckets": by_bucket,
        # S2-reached-recall funnel — the loop-closure probe.
        "s2_reach": {
            "reps_total": len(results),
            "reps_with_s2_nodes": reps_with_s2,
            "s2_in_candidates_sum": reach_cand_sum,
            "s2_in_selected_sum": reach_sel_sum,
            "s2_total_sum": reach_total_sum,
        },
        "encode_miss_count": encode_miss,
        "recall_conditional_pass": recall_conditional,
        "gate_flagged_unanswerable": gate_unanswerable,
        "total_ms": total_ms,
        "results": results,
    }
    report_path = os.path.join(reports_dir, f"run_{run_name}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[sweep] done in {total_ms/1000:.1f}s", flush=True)
    print(f"[sweep] overall (raw): {overall:.1%} ({correct_count}/{len(results)})", flush=True)
    print(f"[sweep] recall-conditional (excl. {encode_miss} ENCODE_MISS): "
          f"{recall_conditional:.1%} ({correct_count}/{recall_denom})", flush=True)
    print(f"[sweep] failure buckets: {json.dumps(by_bucket)}", flush=True)
    print(f"[sweep] S2-reached-recall: {reps_with_s2}/{len(results)} reps had S2 nodes; "
          f"{reach_cand_sum} surfaced as candidates, {reach_sel_sum} selected", flush=True)
    print(f"[sweep] report → {report_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="corpus_hash from build_corpus.py")
    p.add_argument("--surface", default="active",
                   help="'active' or a path to a surface prompt file to test at query time")
    p.add_argument("--variance", type=int, default=1, help="repeat each item N times")
    p.add_argument("--label", default=None, help="run name (default: sweep_{corpus}_{ts})")
    p.add_argument("--qids", default=None, help="comma-separated subset of corpus qids")
    args = p.parse_args()
    sweep(args.corpus, args.surface, args.variance, args.label, qids=args.qids)


if __name__ == "__main__":
    main()
