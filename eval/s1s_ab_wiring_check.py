"""S1S v13+muster smoke test — parallel end-to-end wiring verification.

Before committing to the full A/B run, this smoke verifies the new pipeline
actually works:

    1. v13 prompt registers cleanly into an isolated brain
    2. Muster fires and all 4 scouts return envelopes
    3. Scout reports land in the scribe's user_content
    4. Scribe produces actions (remember / revise / connect)
    5. No silent failures in brain_errors

For each (transcript, arm, run_idx) we create a fresh per-job brain,
drive the S1R -> S0 -> S1E loop, and collect:

    - scout trace events (scout_input, scout_findings per cycle)
    - scribe delta trace + captured final_text
    - new nodes + edges created this run
    - any brain_error rows added during the run

Jobs run in parallel via ProcessPoolExecutor (following the longmem
harness pattern -- each worker loads its own embedder). Thread-safe
signalling via explicit muster_enabled kwarg instead of env var, so
parallel jobs don't race the global env.

Usage:
    ./dev python3 eval/s1s_ab_smoke.py                 # full smoke
    ./dev python3 eval/s1s_ab_smoke.py --serial        # sequential
    ./dev python3 eval/s1s_ab_smoke.py --runs 1        # no variance
    ./dev python3 eval/s1s_ab_smoke.py --keep-dbs      # preserve per-job DBs
    ./dev python3 eval/s1s_ab_smoke.py --transcripts conv_001 6f9b354f
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


CORPUS_TRANSCRIPTS = [
    ("conv_001_architecture", "conv_001_architecture.json"),
    ("conv_005_emotions",     "conv_005_emotions.json"),
]

LONGMEM_PICKS = [
    ("longmem_info_ext",    "6f9b354f",       "info_extraction"),
    ("longmem_multisess",   "edced276",       "multi_session"),
    ("longmem_temporal",    "5e1b23de",       "temporal"),
    ("longmem_kupdate",     "10e09553",       "knowledge_update"),
    ("longmem_abstention",  "0862e8bf_abs",   "abstention"),
]

REPORTS_DIR = ROOT / "eval" / "reports" / "s1s_ab_smoke"


def _load_corpus_messages(filename: str) -> List[Dict[str, str]]:
    path = ROOT / "eval" / "corpus" / filename
    conv = json.loads(path.read_text(encoding="utf-8"))
    return [{"role": ex["role"], "content": ex["content"]}
            for ex in conv["exchanges"]]


def _load_longmem_messages(question_id: str) -> Tuple[List[Dict[str, str]], Optional[List[str]]]:
    data = json.loads((ROOT / "eval" / "longmem" / "data" /
                       "longmemeval_oracle.json").read_text(encoding="utf-8"))
    for item in data:
        if item["question_id"] == question_id:
            msgs: List[Dict[str, str]] = []
            dates = item.get("haystack_dates") or []
            for si, sess in enumerate(item["haystack_sessions"]):
                date_tag = (dates[si] if si < len(dates) else None)
                for turn in sess:
                    content = turn["content"]
                    if date_tag and turn["role"] == "user":
                        content = f"[Current date: {date_tag}]\n\n{content}"
                    msgs.append({"role": turn["role"], "content": content})
            return msgs, dates
    raise ValueError(f"longmem question_id {question_id!r} not found")


def load_all_transcripts() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for slug, fname in CORPUS_TRANSCRIPTS:
        out.append({
            "slug": slug,
            "source": "corpus",
            "messages": _load_corpus_messages(fname),
            "axis": slug.split("_", 2)[-1],
        })
    for slug, qid, axis in LONGMEM_PICKS:
        msgs, _dates = _load_longmem_messages(qid)
        out.append({
            "slug": slug,
            "source": "longmem",
            "question_id": qid,
            "axis": axis,
            "messages": msgs,
        })
    return out


def run_job(transcript: Dict[str, Any], arm: str, run_idx: int,
            v13_prompt: Optional[str], keep_db: bool = True,
            brain_base_dir: Optional[str] = None) -> Dict[str, Any]:
    """One (transcript x arm x run_idx) encoding run, end-to-end.

    Each job gets a freshly-seeded brain (via create_fresh_eval_brain)
    rather than a copy of prod — eliminates session_context leak and
    any other prod-state bleed-through that confounds A/B comparison.
    """
    t_start = time.time()
    slug = transcript["slug"]
    job_id = f"{slug}:{arm}:r{run_idx}"
    prefix = f"[{job_id}]"

    _load_env()

    # Per-job brain under run_dir/brains/{job_id}/. Seed-pack fresh — NOT
    # a copy of prod. Reason: copying prod leaks session_context (global
    # config key) and encoding_journal state into every job's scribe
    # prompt. That caused the scribe to skip longmem_info_ext entirely
    # in smoke_full_1: it saw prod's technical session_context and judged
    # a home-decor conversation as "unrelated to the ongoing session."
    # A fresh seed brain eliminates that leak AND all prod-state bleed-
    # through. Every job starts from the same 16-node seed pack.
    if brain_base_dir:
        job_dir = os.path.join(brain_base_dir, job_id.replace(":", "__"))
    else:
        job_dir = tempfile.mkdtemp(prefix=f"s1s_smoke_{slug}_{arm}_r{run_idx}_")

    from eval.longmem.fresh_brain import create_fresh_eval_brain
    brain = create_fresh_eval_brain(path=job_dir, wipe=True)
    # create_fresh_eval_brain sets BRAIN_DB_DIR; no further env munging.

    # Build a local dispatch mirroring the TCP-less path in replay.py.
    # Same COMMAND_TABLE the daemon exposes — call register_interaction
    # via the public write surface rather than the private DAL.
    from servers.daemon_dispatch import COMMAND_TABLE

    def dispatch(cmd, args=None):
        entry = COMMAND_TABLE.get(cmd)
        if not entry:
            return {"ok": False, "error": f"unknown: {cmd}"}
        return entry.handler(brain, args or {}, [])

    registered_version = None
    if arm == "B":
        if not v13_prompt:
            raise RuntimeError(f"{prefix} arm B requires v13_prompt")
        reg = dispatch("register_interaction", {
            "name": "s1e",
            "template": v13_prompt,
            "parameters": "",
            "created_by": "s1s_wiring_check",
        })
        if not reg.get("ok"):
            raise RuntimeError(f"{prefix} register_interaction failed: {reg}")
        registered_version = (reg.get("result") or {}).get("version")
    muster_enabled = (arm == "B")

    session_id = f"smoke-{slug}-{arm}-r{run_idx}-{os.getpid()}"
    err_baseline = _count_errors(brain)
    t_mark = time.time()

    stats = _drive_encoding(brain, transcript["messages"],
                            dispatch=dispatch,
                            session_id=session_id,
                            muster_enabled=muster_enabled,
                            log_prefix=prefix)

    scout_events = _fetch_scout_events(brain, session_id)
    encoding_events = _fetch_encoding_events(brain, session_id)
    new_nodes = _fetch_new_nodes(brain, t_mark)
    new_edges = _fetch_new_edges(brain, t_mark)
    new_errors = _fetch_new_errors(brain, err_baseline)
    prompt_snapshots = _collect_prompt_snapshots(session_id, stats.get("s1e_counters", []))

    # End-to-end step: for longmem transcripts, run query + answer + judge.
    # This is the "recalled with reason well" KPI — does the brain's recall
    # surface the right context AND does the answerer produce a correct
    # response? Abstention axis specifically tests refusal when info absent.
    e2e = None
    if transcript.get("source") == "longmem" and transcript.get("question_id"):
        try:
            from eval.longmem.replay import query_brain
            from eval.longmem.answerer import answer_question
            from eval.longmem.judge import judge_one
            data = json.loads((ROOT / "eval" / "longmem" / "data" /
                               "longmemeval_oracle.json").read_text(encoding="utf-8"))
            item = next((i for i in data
                         if i["question_id"] == transcript["question_id"]), None)
            if item:
                # Drain any deferred embeddings before query (mirrors replay.py)
                try:
                    brain.backfill_vectors(batch_size=50)
                except Exception:
                    pass
                q_res = query_brain(brain, item["question"],
                                    item.get("question_date"))
                a_res = answer_question(item["question"],
                                        q_res["additional_context"],
                                        item.get("question_date"))
                j_res = judge_one(item["question"], item["answer"],
                                  a_res["hypothesis"])
                e2e = {
                    "question": item["question"],
                    "gold_answer": str(item["answer"])[:400],
                    "hypothesis": a_res["hypothesis"][:400],
                    "abstained": a_res["abstained"],
                    "has_context": a_res["has_context"],
                    "correct": j_res["correct"],
                    "judge_raw": j_res["raw"][:200],
                    "query_s1r_ms": q_res["s1r_ms"],
                    "answer_ms": a_res["elapsed_ms"],
                    "answer_tokens_in": a_res.get("tokens_in", 0),
                    "answer_tokens_out": a_res.get("tokens_out", 0),
                    "additional_context_chars": len(q_res.get("additional_context") or ""),
                }
        except Exception as e:
            import traceback
            e2e = {"error": str(e), "traceback": traceback.format_exc()[-1500:]}

    try:
        brain.save()
        brain.close()
    except Exception:
        pass
    # Force GC — see eval/s1s_snapshot_replay.py for rationale.
    # ProcessPoolExecutor isolates jobs in separate processes so this is
    # mostly for the --serial path, but harmless in either case.
    del brain
    import gc
    gc.collect()

    scout_summary = _summarize_scouts(scout_events)

    result = {
        "job_id": job_id,
        "slug": slug,
        "axis": transcript.get("axis"),
        "source": transcript["source"],
        "arm": arm,
        "run_idx": run_idx,
        "muster_enabled": muster_enabled,
        "v13_registered_version": registered_version,
        "elapsed_s": round(time.time() - t_start, 2),
        "n_messages": len(transcript["messages"]),
        "stats": stats,
        "scout_summary": scout_summary,
        "scout_events_count": len(scout_events),
        "encoding_events_count": len(encoding_events),
        "new_nodes": len(new_nodes),
        "new_edges": len(new_edges),
        "new_error_rows": new_errors,
        "prompt_snapshots": prompt_snapshots,
        "e2e": e2e,
        "pass_checks": _evaluate_pass(arm, stats, scout_summary,
                                      prompt_snapshots, new_errors,
                                      len(new_nodes)),
        "job_dir": job_dir if keep_db else None,
    }

    if not keep_db:
        try:
            shutil.rmtree(job_dir)
        except Exception:
            pass

    return result


def _drive_encoding(brain, messages: List[Dict[str, str]], *,
                    dispatch, session_id: str, muster_enabled: bool,
                    log_prefix: str) -> Dict[str, Any]:
    from servers.daemon_hooks import hook_recall, post_response_common
    from servers.scales.s1.encode import run_encoding

    ctx = brain.get_or_create_session(session_id)
    s1e_counters: List[int] = []
    s1e_results: List[Dict[str, Any]] = []
    i = 0
    user_turns = 0
    while i < len(messages):
        turn = messages[i]
        if turn["role"] != "user":
            i += 1
            continue
        user_msg = turn["content"]
        assistant_msg = ""
        if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            assistant_msg = messages[i + 1]["content"]

        try:
            hook_recall(brain, {"prompt": user_msg, "session_id": session_id}, [])
        except Exception as e:
            print(f"{log_prefix} WARN hook_recall failed: {e}", flush=True)

        ctx = post_response_common(brain, session_id, user_msg, assistant_msg)
        user_turns += 1

        if ctx.stop_counter % 5 == 0 and ctx.stop_counter > 0:
            print(f"{log_prefix} s1e firing @ stop={ctx.stop_counter} (muster={muster_enabled})",
                  flush=True)
            t0 = time.time()
            try:
                res = run_encoding(brain, dispatch, ctx.stop_counter, session_id,
                                   muster_enabled=muster_enabled)
            except Exception as e:
                print(f"{log_prefix} S1E ERROR: {e}", flush=True)
                traceback.print_exc()
                res = {"error": str(e)}
            res["_elapsed_ms"] = int((time.time() - t0) * 1000)
            res["_stop_counter"] = ctx.stop_counter
            s1e_counters.append(ctx.stop_counter)
            s1e_results.append({
                "stop": ctx.stop_counter,
                "elapsed_ms": res["_elapsed_ms"],
                "rounds": res.get("rounds"),
                "actions": res.get("actions"),
                "write_actions": res.get("write_actions"),
                "muster": res.get("muster"),
                "error": res.get("error"),
                "tokens_input": res.get("input_tokens"),
                "tokens_output": res.get("output_tokens"),
                "tokens_cache_read": res.get("cache_read_tokens"),
                "tokens_cache_creation": res.get("cache_creation_tokens"),
            })

        i += 2 if assistant_msg else 1

    if ctx.stop_counter % 5 != 0 and ctx.stop_counter > 0:
        print(f"{log_prefix} s1e trailing flush @ stop={ctx.stop_counter}", flush=True)
        t0 = time.time()
        try:
            res = run_encoding(brain, dispatch, ctx.stop_counter, session_id,
                               muster_enabled=muster_enabled)
        except Exception as e:
            print(f"{log_prefix} S1E TRAILING ERROR: {e}", flush=True)
            traceback.print_exc()
            res = {"error": str(e)}
        res["_elapsed_ms"] = int((time.time() - t0) * 1000)
        res["_stop_counter"] = ctx.stop_counter
        s1e_counters.append(ctx.stop_counter)
        s1e_results.append({
            "stop": ctx.stop_counter,
            "elapsed_ms": res["_elapsed_ms"],
            "rounds": res.get("rounds"),
            "actions": res.get("actions"),
            "write_actions": res.get("write_actions"),
            "muster": res.get("muster"),
            "error": res.get("error"),
            "tokens_input": res.get("input_tokens"),
            "tokens_output": res.get("output_tokens"),
            "tokens_cache_read": res.get("cache_read_tokens"),
            "tokens_cache_creation": res.get("cache_creation_tokens"),
        })

    return {
        "user_turns": user_turns,
        "final_stop_counter": ctx.stop_counter,
        "s1e_counters": s1e_counters,
        "s1e_runs": s1e_results,
    }


def _fetch_scout_events(brain, session_id: str) -> List[Dict[str, Any]]:
    try:
        rows = brain.logs_conn.execute(
            "SELECT chain_id, event_type, ref_type, summary, metadata "
            "FROM trace_events WHERE session_id = ? AND "
            "ref_type IN ('scout_input', 'scout_findings') "
            "ORDER BY id ASC",
            (session_id,)).fetchall()
    except Exception as exc:
        print(f'[smoke] _fetch_scout_events error: {type(exc).__name__}: {exc}', flush=True)
        return []
    out = []
    for r in rows:
        try:
            md = json.loads(r[4]) if r[4] else {}
        except Exception:
            md = {}
        # Defensive: trace metadata should be a dict. An older muster bug
        # double-serialized and produced str-on-decode — harden rather than
        # crash the summary.
        if not isinstance(md, dict):
            md = {"_raw": md}
        out.append({
            "chain": r[0], "type": r[1], "ref": r[2],
            "summary": r[3], "metadata": md,
        })
    return out


def _fetch_encoding_events(brain, session_id: str) -> List[Dict[str, Any]]:
    try:
        rows = brain.logs_conn.execute(
            "SELECT chain_id, event_type, ref_type, summary "
            "FROM trace_events WHERE session_id = ? AND "
            "ref_type IN ('encoding_prompt', 'node_catalog', 'encoding_run') "
            "ORDER BY id ASC",
            (session_id,)).fetchall()
    except Exception as exc:
        print(f'[smoke] _fetch_encoding_events error: {type(exc).__name__}: {exc}', flush=True)
        return []
    return [{"chain": r[0], "type": r[1], "ref": r[2], "summary": r[3]}
            for r in rows]


def _fetch_new_nodes(brain, since_ts: float) -> List[Dict[str, Any]]:
    from datetime import datetime, timezone
    iso = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()
    try:
        rows = brain.conn.execute(
            "SELECT id, type, title, created_at, encoding_source "
            "FROM nodes WHERE created_at >= ? AND archived = 0 "
            "ORDER BY created_at ASC",
            (iso,)).fetchall()
    except Exception as exc:
        print(f'[smoke] _fetch_new_nodes error: {type(exc).__name__}: {exc}', flush=True)
        return []
    return [{"id": r[0], "type": r[1], "title": r[2],
             "created_at": r[3], "encoding_source": r[4]} for r in rows]


def _fetch_new_edges(brain, since_ts: float) -> List[Dict[str, Any]]:
    from datetime import datetime, timezone
    iso = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()
    try:
        rows = brain.conn.execute(
            "SELECT edge_id, source_id, target_id, weight, created_at "
            "FROM edges WHERE created_at >= ? "
            "ORDER BY created_at ASC",
            (iso,)).fetchall()
    except Exception as exc:
        print(f'[smoke] _fetch_new_edges error: {type(exc).__name__}: {exc}', flush=True)
        return []
    return [{"edge_id": r[0], "src": r[1], "tgt": r[2],
             "weight": r[3], "created_at": r[4]} for r in rows]


def _count_errors(brain) -> int:
    # Errors land in debug_log with event_type='error' (see brain._log_error).
    # There is no brain_errors table — the "error" events share the debug log.
    try:
        return brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type = 'error'").fetchone()[0]
    except Exception as exc:
        print(f'[smoke] _count_errors error: {type(exc).__name__}: {exc}', flush=True)
        return 0


def _fetch_new_errors(brain, baseline: int) -> List[Dict[str, Any]]:
    current = _count_errors(brain)
    n_new = max(0, current - baseline)
    if n_new == 0:
        return []
    # Errors are stored in debug_log with source in 'source' column and
    # the structured error payload inside the metadata JSON blob.
    try:
        rows = brain.logs_conn.execute(
            "SELECT source, metadata FROM debug_log "
            "WHERE event_type = 'error' ORDER BY id DESC LIMIT ?",
            (n_new,)).fetchall()
    except Exception as exc:
        print(f'[smoke] _fetch_new_errors error: {type(exc).__name__}: {exc}', flush=True)
        return []
    out = []
    for r in rows:
        md = {}
        try:
            md = json.loads(r[1]) if r[1] else {}
        except Exception:
            md = {}
        out.append({"type": r[0], "message": (md.get("error") or "")[:300],
                    "context": (md.get("context") or "")[:200]})
    return out


def _collect_prompt_snapshots(session_id: str, counters: List[int]) -> List[Dict[str, Any]]:
    # Per-session tmp path so parallel jobs don't overwrite each other.
    # Use FULL session_id — 16-char truncation collided between jobs whose
    # session_ids shared a common prefix (e.g. smoke-conv_001_architecture-*).
    sess_safe = (session_id or 'nosession').replace('/', '_').replace(' ', '_')
    # Honor BRAIN_TMP_DIR via the single-source helper (this file already
    # imports from servers, so no new coupling) — must match the WRITER.
    from servers.daemon_config import brain_tmp_dir
    tmp_dir = brain_tmp_dir()
    out = []
    for c in counters:
        path = os.path.join(tmp_dir, f"brain-encoding-prompt-{sess_safe}-{c}.json")
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            out.append({"counter": c, "read_error": str(e), "path": path})
            continue
        uc = data.get("user_content", "")
        out.append({
            "counter": c,
            "user_content_chars": len(uc),
            "has_scout_reports": "## Scout reports" in uc,
            "has_timeline": "### Conversation Timeline" in uc,
            "scout_report_chars": _extract_section_chars(uc, "## Scout reports"),
            "path": path,
        })
    return out


def _extract_section_chars(text: str, header: str) -> int:
    idx = text.find(header)
    if idx < 0:
        return 0
    return len(text) - idx


def _summarize_scouts(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_scout: Dict[str, Dict[str, int]] = {}
    for ev in events:
        md = ev.get("metadata", {})
        name = md.get("scout")
        if not name:
            continue
        bucket = per_scout.setdefault(name, {
            "input_events": 0, "findings_events": 0,
            "candidates_total": 0, "errors": 0})
        if ev["ref"] == "scout_input":
            bucket["input_events"] += 1
        elif ev["ref"] == "scout_findings":
            bucket["findings_events"] += 1
            cands = md.get("candidate_handles") or []
            bucket["candidates_total"] += len(cands)
            if md.get("errors"):
                bucket["errors"] += len(md["errors"])
    return per_scout


def _evaluate_pass(arm: str, stats: Dict[str, Any],
                   scout_summary: Dict[str, Dict[str, int]],
                   prompt_snapshots: List[Dict[str, Any]],
                   new_errors: List[Dict[str, Any]],
                   new_nodes: int) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    s1e_runs = stats.get("s1e_runs", [])
    checks["s1e_ran"] = len(s1e_runs) > 0
    checks["s1e_all_succeeded"] = all(r.get("error") is None for r in s1e_runs)
    checks["scribe_actions_nonempty"] = any(
        (r.get("actions") or 0) > 0 for r in s1e_runs)
    checks["new_nodes_nonempty"] = new_nodes > 0
    checks["no_new_brain_errors"] = len(new_errors) == 0

    if arm == "B":
        expected_cycles = len(s1e_runs)
        expected_scouts = {"quote", "temporal", "facts", "synthesis"}
        seen = set(scout_summary.keys())
        checks["scouts_all_ran"] = seen == expected_scouts
        checks["scout_findings_per_cycle"] = all(
            scout_summary.get(n, {}).get("findings_events", 0) == expected_cycles
            for n in expected_scouts)
        checks["scout_reports_in_prompt"] = all(
            s.get("has_scout_reports") for s in prompt_snapshots)
        checks["no_scout_errors"] = all(
            scout_summary.get(n, {}).get("errors", 0) == 0
            for n in expected_scouts)
    else:
        checks["scout_reports_absent"] = all(
            not s.get("has_scout_reports") for s in prompt_snapshots)

    checks["OVERALL"] = all(v for k, v in checks.items()
                            if k != "OVERALL" and isinstance(v, bool))
    return checks


def _load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _pool_worker(args_tuple):
    (transcript, arm, run_idx, v13_prompt, keep_db, brain_base_dir) = args_tuple
    try:
        return run_job(transcript, arm, run_idx, v13_prompt,
                       keep_db=keep_db, brain_base_dir=brain_base_dir)
    except Exception as e:
        tb = traceback.format_exc()
        return {
            "job_id": f"{transcript['slug']}:{arm}:r{run_idx}",
            "slug": transcript["slug"],
            "arm": arm,
            "run_idx": run_idx,
            "error": str(e),
            "traceback": tb[-2000:],
            "pass_checks": {"OVERALL": False},
        }


def main():
    parser = argparse.ArgumentParser(description="S1S v13+muster smoke test")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--keep-dbs", dest="keep_dbs",
                        action="store_true", default=True,
                        help="Preserve per-job brain dirs (default: True)")
    parser.add_argument("--no-keep-dbs", dest="keep_dbs",
                        action="store_false",
                        help="Delete per-job brain dirs after the run")
    parser.add_argument("--transcripts", nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", default=["A", "B"],
                        choices=["A", "B"])
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    _load_env()

    from eval.s1s_v13_prompt import extract_v13_prompt
    v13_prompt = extract_v13_prompt()
    print(f"[smoke] v13 prompt extracted: {len(v13_prompt)} chars")

    all_transcripts = load_all_transcripts()
    if args.transcripts:
        selected = []
        for t in all_transcripts:
            for needle in args.transcripts:
                if needle in t["slug"]:
                    selected.append(t)
                    break
        all_transcripts = selected
    if not all_transcripts:
        print("[smoke] no transcripts matched selector")
        sys.exit(2)
    for t in all_transcripts:
        print(f"[smoke] transcript {t['slug']:30s} "
              f"source={t['source']:8s} messages={len(t['messages'])}")

    # Set up run_dir FIRST so brain_base_dir can live inside it.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = REPORTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] writing reports to {run_dir}")

    # Brains live under run_dir/brains/ so they survive for post-hoc analysis.
    # Seed-pack brains (fresh), NOT copies of prod — see docstring in run_job.
    brain_base_dir = str(run_dir / "brains")
    os.makedirs(brain_base_dir, exist_ok=True)
    print(f"[smoke] brain dirs will be under: {brain_base_dir}")
    print(f"[smoke] using fresh seed-pack brains (no prod-state leak)")

    jobs: List[Tuple[Any, ...]] = []
    for t in all_transcripts:
        for arm in args.arms:
            for ri in range(args.runs):
                jobs.append((t, arm, ri, v13_prompt, args.keep_dbs,
                             brain_base_dir))
    print(f"[smoke] total jobs: {len(jobs)}  "
          f"(transcripts={len(all_transcripts)} x arms={len(args.arms)} x runs={args.runs})")

    results: List[Dict[str, Any]] = []
    t_run0 = time.time()
    if args.serial or args.workers <= 1:
        for i, job in enumerate(jobs):
            print(f"\n[smoke] job {i+1}/{len(jobs)}: {job[0]['slug']}:{job[1]}:r{job[2]}")
            r = _pool_worker(job)
            results.append(r)
            (run_dir / "results.jsonl").write_text(
                "\n".join(json.dumps(x, default=str) for x in results) + "\n",
                encoding="utf-8")
    else:
        print(f"[smoke] parallel: {args.workers} workers")
        by_idx: Dict[int, Dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_pool_worker, job): i
                       for i, job in enumerate(jobs)}
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    by_idx[i] = fut.result()
                except Exception as e:
                    by_idx[i] = {"job_id": f"idx{i}", "error": str(e),
                                 "pass_checks": {"OVERALL": False}}
                done += 1
                r = by_idx[i]
                mark = "PASS" if r.get("pass_checks", {}).get("OVERALL") else "FAIL"
                print(f"[smoke] {mark} {done}/{len(jobs)}: {r.get('job_id')} "
                      f"({r.get('elapsed_s', '?')}s)", flush=True)
        results = [by_idx[i] for i in range(len(jobs))]

    total_s = round(time.time() - t_run0, 1)

    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in results) + "\n",
        encoding="utf-8")
    (run_dir / "results.json").write_text(
        json.dumps({"run_name": run_name, "elapsed_s": total_s,
                    "jobs": results}, default=str, indent=2),
        encoding="utf-8")

    _print_summary(results, total_s, run_dir)

    all_pass = all(r.get("pass_checks", {}).get("OVERALL") for r in results)
    sys.exit(0 if all_pass else 1)


def _print_summary(results: List[Dict[str, Any]], total_s: float, run_dir: Path):
    print("\n" + "=" * 110)
    print("S1S v13+MUSTER SMOKE RESULTS")
    print("=" * 110)

    n_total = len(results)
    n_pass = sum(1 for r in results if r.get("pass_checks", {}).get("OVERALL"))
    print(f"Total jobs: {n_total}  |  Passed: {n_pass}  |  "
          f"Failed: {n_total - n_pass}  |  Total time: {total_s}s")
    print()

    header = (f"{'JOB':<42} {'ARM':^4} {'ELAPSED':>8} "
              f"{'S1E':>4} {'ACTN':>5} {'NODES':>6} {'EDGES':>6} "
              f"{'SCOUTS (cands: q/t/f/s)':<26} {'RESULT'}")
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x.get("job_id", "")):
        job = r.get("job_id", "?")
        arm = r.get("arm", "?")
        elapsed = r.get("elapsed_s", 0)
        stats = r.get("stats", {}) or {}
        s1e_runs = stats.get("s1e_runs", []) or []
        actions = sum((s.get("actions") or 0) for s in s1e_runs)
        nodes = r.get("new_nodes", 0)
        edges = r.get("new_edges", 0)
        scout_sum = r.get("scout_summary", {}) or {}
        if arm == "B":
            scout_str = "  ".join(
                f"{scout_sum.get(n, {}).get('candidates_total', 0):>2}"
                for n in ("quote", "temporal", "facts", "synthesis")
            )
        else:
            scout_str = " -- arm A --"
        checks = r.get("pass_checks", {})
        mark = "PASS" if checks.get("OVERALL") else "FAIL"
        print(f"{job:<42} {arm:^4} {elapsed:>7}s "
              f"{len(s1e_runs):>4} {actions:>5} {nodes:>6} {edges:>6} "
              f"{scout_str:<26} {mark}")
        if not checks.get("OVERALL"):
            for k, v in checks.items():
                if k == "OVERALL" or v:
                    continue
                print(f"  FAIL: {k}")
            if r.get("error"):
                print(f"  EXCEPTION: {r['error']}")
            if r.get("new_error_rows"):
                for e in r["new_error_rows"][:3]:
                    print(f"  brain_error: {e.get('type')}: {e.get('message', '')[:100]}")

    print()
    print(f"Reports written to: {run_dir}")


if __name__ == "__main__":
    main()
