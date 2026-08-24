"""Harness — stratified micro-suite runner.

Orchestrates:
  1. Load LongMemEval oracle JSON
  2. Select stratified N items (default: 10 = 2 per axis × 5 axes)
  3. For each item: fresh brain (reset to seeds) → replay haystack → query → answerer → record
  4. Stream hypotheses + per-item results to JSONL as items complete
  5. Write aggregate report.json at the end

Usage:
  python3 eval/longmem/harness.py              # run with defaults
  python3 eval/longmem/harness.py --items 5    # smaller for debugging
  python3 eval/longmem/harness.py --seed 42    # reproducible selection
  python3 eval/longmem/harness.py --smoke-test # 1 item per axis, fail-fast preflight

Reliability guardrails (so a 30+ min run doesn't fail 20 min deep):
  - _preflight() validates env, oracle, disk, embedder importability before
    any work happens. Run automatically at main() start.
  - --smoke-test runs 1 item per axis (5 items, ~3-5 min wall) and exits
    non-zero on any pipeline failure. Do this BEFORE any long run.
  - Per-item streaming writes: hypotheses_{run}.jsonl and results_{run}.jsonl
    are appended after each item completes, so a crash at item N preserves
    items 1..N-1.
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, TextIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.interaction_override import override_interaction


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
    """Count error rows now — a monotonic counter to detect errors logged during
    a single item's ingest/query phase.

    Errors live in `debug_log` (event_type='error') via `brain._log_error`, NOT a
    `brain_errors` table. The prior query here hit a non-existent table and the
    bare `except` swallowed it to 0 — so this guardrail was a silent no-op. Fixed
    2026-05-29 (same bug class flagged in docs/WRITE-TXN-ISOLATION-ROOTFIX.md)."""
    try:
        return brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type='error'").fetchone()[0]
    except Exception as e:
        # Loud: a failed read here makes the "silent errors" guardrail (and
        # the smoke gate's brain_errors_new threshold) pass vacuously.
        print(f"[harness] WARN error-count read failed ({e}) — "
              f"brain_errors guardrail is blind this item", flush=True)
        return 0


def _new_errors_since(brain, baseline_count: int) -> List[Dict[str, Any]]:
    """Return error rows logged since baseline_count. Limited to 20 for sanity.

    Error detail is JSON in `debug_log.metadata` ({error, type, context})."""
    try:
        rows = brain.logs_conn.execute(
            "SELECT source, metadata FROM debug_log WHERE event_type='error' "
            "ORDER BY id DESC LIMIT ?", (20,)).fetchall()
    except Exception as e:
        print(f"[harness] WARN error-rows read failed ({e}) — "
              f"brain_errors guardrail is blind this item", flush=True)
        return []
    current = _snapshot_error_count(brain)
    n_new = max(0, current - baseline_count)
    if n_new == 0:
        return []
    out = []
    for source, meta_json in rows[:n_new]:
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except Exception:
            meta = {}
        out.append({
            "type": meta.get("type") or source,
            "message": (meta.get("error") or "")[:200],
            "context": (meta.get("context") or "")[:100],
        })
    return out


def _preflight(oracle_path: str, min_oracle_items: int = 100,
               min_disk_gb: float = 1.0) -> None:
    """Validate environment + dependencies before launching a long run.

    Raises RuntimeError with a clear message on any failure. Catches the
    "20 min deep then we discover X is broken" class of bug — every check
    here is for something that, if broken, fails ALL items not just one.

    Args:
        oracle_path: Path to oracle JSON to validate.
        min_oracle_items: Lower bound on oracle item count (sanity, not strict).
        min_disk_gb: Minimum free space in ~/AgentsContext/ before run.
    """
    failures = []

    # 1. ANTHROPIC_API_KEY — judge + answerer both need it. Tested AFTER .env
    #    load so the standard project setup works.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        failures.append(
            "ANTHROPIC_API_KEY not set — judge + answerer will fail every item. "
            "Either export it or put it in .env at the repo root."
        )

    # 2. Oracle JSON — exists, parses, has reasonable item count.
    if not os.path.exists(oracle_path):
        failures.append(f"Oracle JSON not found at {oracle_path}")
    else:
        try:
            with open(oracle_path) as f:
                data = json.load(f)
            if not isinstance(data, list):
                failures.append(f"Oracle JSON is not a list (got {type(data).__name__})")
            elif len(data) < min_oracle_items:
                failures.append(
                    f"Oracle has {len(data)} items, expected >= {min_oracle_items}")
        except json.JSONDecodeError as e:
            failures.append(f"Oracle JSON parse error: {e}")
        except Exception as e:
            failures.append(f"Oracle JSON read error: {e}")

    # 3. Disk space at ~/AgentsContext/ — each per-item brain takes ~5–10 MB,
    #    50 items in flight ≈ 500 MB peak before cleanup. Be conservative.
    base = os.path.expanduser("~/AgentsContext")
    try:
        os.makedirs(base, exist_ok=True)
        usage = shutil.disk_usage(base)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_disk_gb:
            failures.append(
                f"~/AgentsContext/ has {free_gb:.2f} GB free, need >= {min_disk_gb} GB. "
                f"Per-item brains accumulate even with cleanup if items run in parallel.")
    except Exception as e:
        failures.append(f"Disk-space check failed at {base}: {e}")

    # 4. Embedder importable — model load itself is slow, so we just check
    #    the import path, which catches most "module renamed / venv broken"
    #    failures without paying the 1.5s load cost.
    try:
        import importlib
        importlib.import_module("servers.embedder")
    except Exception as e:
        failures.append(f"servers.embedder import failed: {e}")

    if failures:
        msg = "\n  - ".join(["[preflight] FAILED:"] + failures)
        raise RuntimeError(msg)
    print("[preflight] OK — env + oracle + disk + embedder import all good", flush=True)


def _open_streaming_writers(reports_dir: str, run_name: str) -> Dict[str, TextIO]:
    """Open append-mode writers for hypotheses + per-item results.

    Returns dict with 'hypotheses' and 'results' keys, each a file handle
    that the loop can append-and-flush to. Caller is responsible for
    closing them.
    """
    hypotheses_path = os.path.join(reports_dir, f"hypotheses_{run_name}.jsonl")
    results_path = os.path.join(reports_dir, f"results_{run_name}.jsonl")
    return {
        'hypotheses': open(hypotheses_path, "a", buffering=1),  # line-buffered
        'results': open(results_path, "a", buffering=1),
        'hypotheses_path': hypotheses_path,
        'results_path': results_path,
    }


def _stream_write_result(writers: Dict, result: Dict[str, Any]) -> None:
    """Append one result to both jsonl streams. Flushes on each write so a
    process crash preserves what completed."""
    if "hypothesis" in result:
        writers['hypotheses'].write(json.dumps({
            "question_id": result["question_id"],
            "hypothesis": result["hypothesis"]}) + "\n")
        writers['hypotheses'].flush()
    writers['results'].write(json.dumps(result) + "\n")
    writers['results'].flush()


def _apply_s1e_override(brain, override_path: str) -> None:
    """Point this brain's s1e at a candidate prompt file (--s1e-override).

    Tests a candidate without editing the code default, which would deploy
    it to every install at the next restart. Preserves the effective config;
    only the template changes.
    """
    override_interaction(brain, 's1e', template=open(override_path).read(),
                         set_by='eval-s1e-override')


def _apply_surface_override(brain, override_path: str,
                            params_json: str = None) -> None:
    """Point this brain's surface at a candidate prompt file (--surface-override).

    Companion to the BRAIN_SURFACE_VARIANT env var (set by main() when
    --surface-override is passed) — the env var picks the tool-use loop in
    surface.py; this makes the override prompt the one Haiku actually reads.

    params_json: optional config for the override (e.g. '{"layout":
    "xml_v13"}' — the surface renderer reads `layout` from the active config,
    so template and user-content layout flip atomically). None preserves the
    effective config.
    """
    override_interaction(brain, 'surface', template=open(override_path).read(),
                         parameters=params_json,
                         set_by='eval-surface-override')


def run_item(item: Dict[str, Any], item_idx: int, total: int,
             run_name: str = None, keep_db: bool = False,
             s1e_override: Optional[str] = None,
             surface_override: Optional[str] = None,
             variance_idx: Optional[int] = None) -> Dict[str, Any]:
    """Run one item end-to-end. Each item gets its OWN brain DB for isolation.

    Per-item DB (brain-eval-{run_name}/{qid}/) means:
      - No reset_to_seeds leftovers (cross-item contamination impossible)
      - Inspectable post-hoc (keep_db=True preserves the DB for debugging)
      - Prerequisite for parallel execution (each process writes to its own file)

    Per-item ARTIFACTS (eval/longmem/reports/{run}/items/{qid}/) are dumped
    BEFORE the brain handles close — durable post-hoc analysis without
    re-running. See eval/ARTIFACTS.md.

    When variance_idx is set, the per-item DB and artifact paths get a
    suffix (-r{variance_idx}) so the same qid can be run N times in
    parallel without collision. The result dict carries variance_idx
    through so the aggregator can compute per-item mean/stddev.

    Returns result dict with inline judge + failure class.
    """
    from eval.longmem.replay import replay_item, query_brain
    from eval.longmem.answerer import answer_question
    from eval.longmem.fresh_brain import create_fresh_eval_brain, per_item_brain_dir
    from eval.longmem.judge import judge_one
    from eval.longmem.classifier import classify_failure, _read_s1r_trace
    from eval.longmem.artifacts import EvalArtifactsDumper

    qid = item["question_id"]
    # Variance runs share qid but get a -r{idx} suffix to keep per-item DBs
    # and artifact dirs distinct. variance_idx=None means single-run, no suffix.
    artifact_qid = qid if variance_idx is None else f"{qid}-r{variance_idx}"
    item_db_path = per_item_brain_dir(qid, run_name=run_name,
                                      variance_idx=variance_idx)
    brain = create_fresh_eval_brain(path=item_db_path, wipe=True)

    # Optional s1e prompt override — register a new version over the seeded
    # v1 BEFORE haystack replay so the override is what the encoder uses.
    # A failed override must ABORT the item, not degrade it: continuing
    # encodes/surfaces on the code default while the run records the result
    # as a treatment-arm measurement — a silently corrupt A/B. (For surface:
    # BRAIN_SURFACE_VARIANT stays set by main(), compounding the mismatch.)
    # The abort must not leak the per-item brain — the caller catches item
    # exceptions and continues, so an unclosed Brain per aborted item piles up.
    try:
        if s1e_override:
            _apply_s1e_override(brain, s1e_override)
            print(f"[harness] item {item_idx+1}: s1e override applied from "
                  f"{s1e_override}", flush=True)
        # Optional surface prompt override — register + activate v5 in this
        # eval brain. Companion env var BRAIN_SURFACE_VARIANT=v5_agentic is
        # set by main() when --surface-override is passed, so this brain's
        # surface call uses the agentic tool-use loop reading the v5 prompt.
        if surface_override:
            _apply_surface_override(brain, surface_override)
            print(f"[harness] item {item_idx+1}: surface override applied from "
                  f"{surface_override}", flush=True)
    except Exception:
        try:
            brain.close()
        except Exception:
            pass
        raise

    err_baseline = _snapshot_error_count(brain)

    axis = _item_axis(item)
    n_turns = sum(len(s) for s in item.get("haystack_sessions", []))
    print(f"\n{'='*70}")
    print(f"[harness] item {item_idx+1}/{total} qid={qid} axis={axis} turns={n_turns}", flush=True)
    gold_str = str(item['answer']) if not isinstance(item['answer'], str) else item['answer']
    print(f"[harness] Q: {item['question'][:120]}", flush=True)
    print(f"[harness] A (gold): {gold_str[:120]}", flush=True)
    print(f"{'='*70}", flush=True)

    # Artifacts dumper — writes durable per-item bytes to eval/longmem/reports/.
    # Calls are no-ops if the dumper itself fails; an artifact failure must
    # NEVER kill the eval. Wrap each dump call in try/except.
    dumper: Optional[EvalArtifactsDumper] = None
    try:
        dumper = EvalArtifactsDumper(run_name=run_name or 'adhoc', qid=artifact_qid)
        dumper.dump_meta(
            axis=axis,
            question=item["question"],
            gold=gold_str,
            question_date=item.get("question_date"),
            haystack_dates=item.get("haystack_dates", []),
            haystack_session_ids=item.get("haystack_session_ids", []),
            haystack_turn_count=n_turns,
        )
        # Snapshot interactions BEFORE ingest — captures the prompt versions
        # the encoder/surfacer will use during this run. (Re-dumped after
        # ingest is overkill — interactions don't mutate during eval.)
        dumper.dump_interactions(brain)
    except Exception as e:
        print(f"[harness] artifacts init failed (non-fatal): {e}", flush=True)

    t0 = time.time()
    from eval.longmem.corpus import ingest_session_id as _ingest_sid
    ingest_session_id = _ingest_sid(qid)
    ingest_stats = replay_item(brain, ingest_session_id, item["haystack_sessions"],
                               haystack_dates=item.get("haystack_dates"),
                               log_prefix=f"[item {item_idx+1}]",
                               dumper=dumper)
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
            q_result["query_session_id"], a_result["has_context"], a_result["abstained"],
            context=q_result["additional_context"])
        print(f"[harness] failure: {failure_info['failure_bucket']} — {failure_info['failure_reason'][:140]}",
              flush=True)

    # Surface any silent errors logged during this item (prevents "passed the test
    # but something broke mid-ingest" blind spots).
    new_errors = _new_errors_since(brain, err_baseline)
    if new_errors:
        print(f"[harness] {len(new_errors)} new brain_error rows this item", flush=True)
        for e in new_errors[:5]:
            print(f"    {e['type']}: {e['message'][:120]}", flush=True)

    # Artifacts dumps — happen AFTER classify (so we can include classifier
    # evidence in recall.json) but BEFORE brain.close + cleanup.
    if dumper:
        try:
            dumper.dump_traces(brain)
            dumper.dump_nodes(brain)
            dumper.dump_edges(brain)
            # Recall trace — re-read here so artifacts have the parsed shape;
            # classifier already read it once for failure_evidence, but the
            # parse is cheap and decouples this dump from classifier internals.
            s1r = _read_s1r_trace(brain, q_result["query_session_id"]) or {}
            dumper.dump_recall(
                query_session_id=q_result["query_session_id"],
                query=s1r.get("query", item["question"]),
                candidates=s1r.get("candidates", []),
                selected=s1r.get("selected", []),
                dropped=s1r.get("dropped", []),
                context=s1r.get("context", q_result.get("additional_context", "")),
                classifier_evidence=failure_info.get("failure_evidence"),
                answerer_response={
                    "hypothesis": a_result["hypothesis"],
                    "abstained": a_result["abstained"],
                    "has_context": a_result["has_context"],
                    "tokens_in": a_result.get("tokens_in", 0),
                    "tokens_out": a_result.get("tokens_out", 0),
                    "elapsed_ms": a_result.get("elapsed_ms", 0),
                },
                tool_trace=s1r.get("tool_trace", []),
                surface_variant=s1r.get("surface_variant", ""),
            )
        except Exception as e:
            print(f"[harness] artifacts dump failed (non-fatal): {e}", flush=True)

    result = {
        "question_id": qid,
        "variance_idx": variance_idx,
        "question_type": item["question_type"],
        "axis": axis,
        "question": item["question"],
        "answer_gold": item["answer"],
        "hypothesis": a_result["hypothesis"],
        "abstained": a_result["abstained"],
        "has_context": a_result["has_context"],
        "correct": correct,
        "judge_raw": j["raw"],
        "comparison": j.get("comparison", ""),
        "judge_reasoning": j.get("reasoning", ""),
        # Failure-mode markers, keys absent on clean rows: an answerer API
        # error (e.g. a 529 killing the call) or an unparseable judge output
        # would otherwise score as an indistinguishable brain miss.
        **({"answerer_error": a_result["error"]} if a_result.get("error") else {}),
        **({"judge_parse_failed": True} if j.get("judge_parse_failed") else {}),
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

    # Mirror the result dict to artifacts dir — final piece of the per-item
    # bundle for retrospective analysis (analyzer.py loads this alongside
    # traces/nodes/edges/recall).
    if dumper:
        try:
            dumper.dump_result(result)
        except Exception as e:
            print(f"[harness] artifacts result dump failed (non-fatal): {e}", flush=True)

        # Capture every encoder + surface agent call for offline replay.
        # Encoder prompts + judge results live in the item brain's
        # {db_dir}/payloads/ (payload recorder); the surfaced ids live in the
        # surface_selected traces (recorded via dump_recall). We copy the
        # payloads into the item's agent_calls/ subdir so we can replay any
        # call with a future prompt revision without re-running the eval.
        # Paired with the interactions.jsonl snapshot (system prompts) this
        # is full replay.
        try:
            stats = dumper.dump_agent_calls(session_ids=[
                ingest_session_id, q_result["query_session_id"],
            ])
            print(f"[harness] agent_calls captured: "
                  f"encoder={stats['encoder_calls']} "
                  f"surface={stats['surface_calls']}"
                  + (f" — {stats['errors']} copy error(s), see "
                     f"agent_calls/_manifest.json" if stats.get('errors') else ""),
                  flush=True)
        except Exception as e:
            print(f"[harness] agent_calls capture failed (non-fatal): {e}",
                  flush=True)

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


def _summarize_smoke(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inspect smoke-test results — return {ok, failures} for exit-code decision.

    A smoke item PASSES the pipeline if (regardless of correct/incorrect):
      - no top-level exception ('error' key absent)
      - hypothesis was produced
      - judge returned a verdict ('correct' key present)
      - brain_errors_new is small enough to suggest a working ingest

    Correctness is NOT a smoke-test criterion — we're proving the pipeline
    runs end-to-end, not that the brain is good at the task.
    """
    failures = []
    PIPELINE_ERROR_THRESHOLD = 10  # >10 brain_errors in one item == something broke
    for r in results:
        qid = r.get("question_id", "?")
        if "error" in r:
            failures.append(f"{qid}: top-level exception: {r['error'][:120]}")
            continue
        if "hypothesis" not in r:
            failures.append(f"{qid}: no hypothesis produced")
            continue
        if "correct" not in r:
            failures.append(f"{qid}: judge did not return verdict")
            continue
        n_err = len(r.get("brain_errors_new") or [])
        if n_err > PIPELINE_ERROR_THRESHOLD:
            sample = "; ".join(e["type"] for e in (r.get("brain_errors_new") or [])[:3])
            failures.append(
                f"{qid}: {n_err} brain_errors during item (>{PIPELINE_ERROR_THRESHOLD}) — {sample}")
    return {"ok": not failures, "failures": failures, "total": len(results)}


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
    parser.add_argument("--variance", type=int, default=1,
                        help="Replicate each picked item N times (default 1). "
                             "Each replicate runs in its own per-item brain "
                             "(qid-r{idx}) so they can run in parallel. The "
                             "aggregator reports per-axis mean/stddev across "
                             "replicates so you can tell signal from noise.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 1 item per axis (~3-5 min), validate the pipeline end-to-end, "
                             "exit non-zero on any pipeline failure. Run BEFORE long runs.")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip env/oracle/disk/embedder health checks (debug only).")
    parser.add_argument("--min-oracle-items", type=int, default=10,
                        help="Preflight floor on oracle item count (default 10 — small corpora "
                             "like realchat_oracle.json have 15 items; longmem oracle has 500).")
    parser.add_argument("--qids", default=None,
                        help="Comma-separated qids to run (overrides stratified sampling). "
                             "Use to re-run specific failures with rich artifacts: "
                             "--qids 58470ed2,b86304ba,852ce960. Order is preserved.")
    parser.add_argument("--s1e-override", default=None,
                        help="Path to s1e prompt file to deploy as an override in "
                             "each fresh eval brain (e.g. eval/prompts/s1e_v15_3.txt). "
                             "Used to test prompt revisions on the eval before landing "
                             "them as the code default.")
    parser.add_argument("--surface-override", default=None,
                        help="Path to surface prompt file to register AND activate in each "
                             "fresh eval brain (e.g. eval/surface_v5_prompt.txt). "
                             "Auto-sets BRAIN_SURFACE_VARIANT=v5_agentic in the harness env "
                             "so the agentic tool-use loop runs. Live brain is untouched.")
    args = parser.parse_args()

    # If --surface-override is set, opt the harness into the agentic variant.
    # Live daemon is unaffected — this only sets the env var for THIS process.
    if args.surface_override:
        os.environ['BRAIN_SURFACE_VARIANT'] = 'v5_agentic'
        print('[harness] BRAIN_SURFACE_VARIANT=v5_agentic '
              '(surface-override active: %s)' % args.surface_override, flush=True)

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

    # Preflight — fail fast BEFORE we burn time loading data, picking items,
    # or starting the loop. The whole point is to surface config bugs in the
    # first 2 seconds, not 20 minutes deep.
    if not args.skip_preflight:
        try:
            _preflight(args.oracle, min_oracle_items=args.min_oracle_items)
        except RuntimeError as e:
            print(str(e), file=sys.stderr, flush=True)
            sys.exit(2)

    # Smoke-test mode: override args for a fast, fail-fast end-to-end check.
    # 1 item per axis × 5 axes = 5 items. Serial. Keep DBs for post-mortem.
    if args.smoke_test:
        print("[harness] SMOKE TEST mode: 1 item per axis, serial, keep DBs", flush=True)
        args.items = 1
        args.workers = 1
        args.keep_dbs = True
        if not args.run_name:
            args.run_name = "smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(args.oracle) as f:
        data = json.load(f)
    print(f"[harness] loaded {len(data)} oracle items from {args.oracle}", flush=True)

    # --qids overrides stratified_sample for targeted re-runs (e.g. when
    # investigating specific failures with rich artifacts).
    if args.qids:
        wanted = [q.strip() for q in args.qids.split(",") if q.strip()]
        by_id = {item["question_id"]: item for item in data}
        picked = []
        missing = []
        for q in wanted:
            if q in by_id:
                picked.append(by_id[q])
            else:
                missing.append(q)
        if missing:
            print(f"[harness] WARN qids not in oracle: {missing}", flush=True)
        if not picked:
            print("[harness] no valid qids — exiting", file=sys.stderr)
            sys.exit(1)
        print(f"[harness] --qids picked {len(picked)} items "
              f"(skipping stratified sampling)", flush=True)
    else:
        picked = stratified_sample(data, per_axis=args.items, seed=args.seed)
    print(f"[harness] selected {len(picked)} items:", flush=True)
    for i, item in enumerate(picked):
        axis = _item_axis(item)
        n = sum(len(s) for s in item.get("haystack_sessions", []))
        print(f"  {i+1}. {item['question_id']:<24} axis={axis:<18} turns={n}", flush=True)

    # Expand picked into (item, variance_idx) tasks. variance_idx=None when
    # --variance=1 (preserves single-run qid paths). With --variance N, each
    # item is replicated N times — each replicate is its own brain DB and
    # its own artifact dir at qid-r{idx}.
    if args.variance > 1:
        tasks = [(item, vidx) for item in picked for vidx in range(args.variance)]
        print(f"[harness] variance={args.variance} → {len(tasks)} total runs "
              f"({len(picked)} items × {args.variance} replicates)", flush=True)
    else:
        tasks = [(item, None) for item in picked]

    # Compute run_name up front — each item's brain lives under brain-eval-{run_name}/{qid}/
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Reports dir + streaming writers — open BEFORE the loop so partial
    # progress is preserved if the process dies. JSONL append + flush per
    # item; aggregate report.json still gets written at the end.
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    writers = _open_streaming_writers(reports_dir, run_name)
    print(f"[harness] streaming results → {writers['results_path']}", flush=True)

    results = []
    t_run0 = time.time()
    s1e_override_path = args.s1e_override or None
    if s1e_override_path:
        # Verify file exists at startup so we fail-fast, not per-item
        from pathlib import Path
        if not Path(s1e_override_path).exists():
            print(f"[harness] --s1e-override path not found: {s1e_override_path}",
                  file=sys.stderr)
            sys.exit(2)
        print(f"[harness] s1e override active: {s1e_override_path}", flush=True)

    surface_override_path = args.surface_override or None
    if surface_override_path:
        from pathlib import Path
        if not Path(surface_override_path).exists():
            print(f"[harness] --surface-override path not found: {surface_override_path}",
                  file=sys.stderr)
            sys.exit(2)
        print(f"[harness] surface override active: {surface_override_path}", flush=True)

    try:
        if args.workers <= 1:
            # Serial path (backward compatible). When --variance=1, tasks
            # carries variance_idx=None and per-item paths stay unchanged.
            for i, (item, vidx) in enumerate(tasks):
                try:
                    r = run_item(item, i, len(tasks), run_name=run_name,
                                 keep_db=args.keep_dbs,
                                 s1e_override=s1e_override_path,
                                 surface_override=surface_override_path,
                                 variance_idx=vidx)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[harness] task {i+1} FAILED: {e}", flush=True)
                    r = {"question_id": item["question_id"],
                         "variance_idx": vidx, "error": str(e)}
                results.append(r)
                _stream_write_result(writers, r)
        else:
            # Parallel path — ProcessPoolExecutor, each worker loads its own embedder.
            # Per-item DBs are already isolated (brain-eval-{run_name}/{qid}[-r{idx}]/), so no contention.
            from concurrent.futures import ProcessPoolExecutor, as_completed
            print(f"[harness] running {len(tasks)} tasks across {args.workers} workers", flush=True)
            results_by_idx: Dict[int, Dict[str, Any]] = {}
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(run_item, item, i, len(tasks), run_name,
                                args.keep_dbs, s1e_override_path,
                                surface_override_path, vidx): (i, item, vidx)
                    for i, (item, vidx) in enumerate(tasks)
                }
                done_count = 0
                for fut in as_completed(futures):
                    i, item, vidx = futures[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"[harness] task {i+1} ({item['question_id']}"
                              f"{f'-r{vidx}' if vidx is not None else ''}) FAILED: {e}",
                              flush=True)
                        r = {"question_id": item["question_id"],
                             "variance_idx": vidx, "error": str(e)}
                    results_by_idx[i] = r
                    # Stream as completed (out of order is fine for jsonl)
                    _stream_write_result(writers, r)
                    done_count += 1
                    print(f"[harness] progress: {done_count}/{len(tasks)} done", flush=True)
            # Preserve original order for the aggregate report
            results = [results_by_idx[i] for i in range(len(tasks))]
    finally:
        # Always close streams — guarantees the partial jsonl is on disk.
        try:
            writers['hypotheses'].close()
            writers['results'].close()
        except Exception:
            pass
    total_ms = int((time.time() - t_run0) * 1000)
    hypotheses_path = writers['hypotheses_path']

    # Aggregate inline-graded results
    import statistics
    graded = [r for r in results if "correct" in r]
    correct_count = sum(1 for r in graded if r["correct"])
    overall = correct_count / len(graded) if graded else 0
    by_axis: Dict[str, List[bool]] = {}
    by_bucket: Dict[str, int] = {}
    by_comparison: Dict[str, int] = {}
    by_qid: Dict[str, List[bool]] = {}
    for r in graded:
        by_axis.setdefault(r["axis"], []).append(r["correct"])
        by_qid.setdefault(r["question_id"], []).append(r["correct"])
        if r.get("comparison"):
            by_comparison[r["comparison"]] = by_comparison.get(r["comparison"], 0) + 1
        if not r["correct"] and r.get("failure_bucket"):
            by_bucket[r["failure_bucket"]] = by_bucket.get(r["failure_bucket"], 0) + 1

    def _stats(hits: List[bool]) -> Dict[str, Any]:
        floats = [1.0 if h else 0.0 for h in hits]
        return {
            "mean": sum(floats) / len(floats) if floats else 0,
            "stddev": statistics.pstdev(floats) if len(floats) > 1 else 0.0,
            "n": len(floats),
        }

    axis_stats = {a: _stats(v) for a, v in by_axis.items() if v}
    per_qid_stats = ({q: _stats(v) for q, v in by_qid.items()}
                     if args.variance > 1 else {})

    report_path = os.path.join(reports_dir, f"run_{run_name}.json")
    with open(report_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "items_count": len(results),
            "correct_count": correct_count,
            "overall_score": overall,
            # Legacy {axis: mean} preserved for downstream consumers (report.py, etc.)
            "axis_scores": {a: s["mean"] for a, s in axis_stats.items()},
            "axis_counts": {a: s["n"] for a, s in axis_stats.items()},
            # Full stats (mean + stddev + n) — variance > 1 makes stddev meaningful.
            "axis_stats": axis_stats,
            "per_qid_stats": per_qid_stats,
            "comparison_counts": by_comparison,
            "failure_buckets": by_bucket,
            "total_ms": total_ms,
            "config": {"items_per_axis": args.items, "seed": args.seed,
                       "variance": args.variance},
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

    # Smoke-test exit logic — non-zero on any pipeline failure (independent
    # of correctness). Correctness rate is information; pipeline integrity
    # is the gate.
    if args.smoke_test:
        smoke = _summarize_smoke(results)
        print(f"\n[smoke] {smoke['total']} items, {len(smoke['failures'])} pipeline failure(s)",
              flush=True)
        if smoke['ok']:
            print("[smoke] PASS — pipeline runs end-to-end on every axis", flush=True)
            sys.exit(0)
        else:
            print("[smoke] FAIL — fix these before launching a long run:", flush=True)
            for f in smoke['failures']:
                print(f"  - {f}", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
