"""harness — multi-version per-item driver.

Composes the existing eval/longmem pipeline (run_item: fresh brain, replay,
surface, answer, judge) with multi-version arm management and the encoding-
quality probes from `quality_probes.py`.

Per (version, item) cell:
  1. Materialize the production-registered s1e template for `version` into
     a temp file (one-time per arm).
  2. Call eval.longmem.harness.run_item with s1e_override=<temp_path> and
     keep_db=True so the per-item brain DB survives for probe re-opening.
  3. Re-open the per-item brain in read-only mode; run all quality probes.
  4. Seal the cell result (existing run_item output + probes).
  5. Stream the cell row to per_cell.jsonl as soon as it's ready (crash-safe).

Per stage (a batch of items):
  - Aggregate per-version × per-axis × per-probe.
  - Evaluate stop_conditions; if any fires, halt the run for inspection.
  - Write per_stage.json snapshot.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────
# Template materialization (production daemon → temp file)
# ─────────────────────────────────────────────────────────────────

def materialize_s1e_template(version: int) -> str:
    """Fetch the production-registered s1e template for `version` and write
    it to a temp file. Returns the file path. Caller is responsible for
    cleanup (use a try/finally or contextlib.ExitStack).
    """
    from servers.daemon_client import send_command
    r = send_command('get_interaction', {'name': 's1e', 'version': version})
    if not r.get('ok'):
        raise RuntimeError(f"get_interaction failed for s1e v{version}: {r}")
    template = (r.get('result') or {}).get('template')
    if not template:
        raise RuntimeError(f"s1e v{version} has empty template")
    fd, path = tempfile.mkstemp(
        suffix=f'_s1e_v{version}.txt', prefix='encoder_eval_', text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(template)
    return path


# ─────────────────────────────────────────────────────────────────
# Cell driver
# ─────────────────────────────────────────────────────────────────

def run_cell(version: int, item: Dict[str, Any], arm_run_name: str,
             item_idx: int, total: int,
             template_path: str,
             skip_probes: Optional[List[str]] = None,
             ) -> Dict[str, Any]:
    """One (version, item) cell. Returns sealed result dict."""
    from eval.longmem.harness import run_item, _item_axis
    from eval.longmem.fresh_brain import per_item_brain_dir
    from servers.brain import Brain
    from .quality_probes import run_all_probes

    t0 = time.time()
    longmem_result = run_item(
        item=item, item_idx=item_idx, total=total,
        run_name=arm_run_name, keep_db=True,
        s1e_override=template_path)
    longmem_ms = int((time.time() - t0) * 1000)

    # Re-open the per-item brain for read-only probing
    brain_dir = per_item_brain_dir(
        item['question_id'], run_name=arm_run_name)
    probe_brain = None
    probes: Dict[str, Any] = {}
    probe_error: Optional[str] = None
    t_probes = time.time()
    try:
        # Re-open the per-item brain for read-only probing. Env was loaded
        # by the enclosing process when the harness imported daemon modules.
        probe_brain = Brain(
            db_path=os.path.join(brain_dir, 'brain.db'))
        probes = run_all_probes(probe_brain, item, skip=skip_probes)
    except Exception as e:
        probe_error = repr(e)
    finally:
        if probe_brain is not None:
            try:
                probe_brain.conn.close()
            except Exception:
                pass
            try:
                probe_brain.logs_conn.close()
            except Exception:
                pass
    probe_ms = int((time.time() - t_probes) * 1000)

    # Extract turn/round counts — surface from ingest stats so downstream
    # reports don't need to dig into the longmem_result blob.
    ingest = (longmem_result or {}).get('ingest') or {}
    turn_counts = {
        'haystack_turns': ingest.get('turns', 0),
        'user_turns': ingest.get('user_turns', 0),
        's1e_runs': ingest.get('s1e_runs', 0),  # encoder fires per Stop hook
        's2_runs': ingest.get('s2_runs', 0),    # S2 maintenance runs
        # surface calls (Haiku) — per user turn during replay + 1 at query
        'surface_calls_during_replay': ingest.get('user_turns', 0),
        'surface_call_at_query': 1,
    }

    # Artifact + brain paths for offline replay
    brain_db_path = (longmem_result or {}).get('brain_db_path') or brain_dir
    artifact_dir = os.path.join(
        'eval', 'longmem', 'reports', arm_run_name, 'items',
        item['question_id'])

    return {
        'version': version,
        'item_id': item['question_id'],
        'axis': _item_axis(item),
        'longmem_result': longmem_result,
        'probes': probes,
        'probe_error': probe_error,
        'turn_counts': turn_counts,
        'paths': {
            'brain_db': brain_db_path,
            'artifacts_dir': artifact_dir,
            'agent_calls_dir': os.path.join(artifact_dir, 'agent_calls'),
        },
        'timing': {
            'longmem_ms': longmem_ms,
            'probe_ms': probe_ms,
        },
    }


# ─────────────────────────────────────────────────────────────────
# Arm driver — one version across a batch of items
# ─────────────────────────────────────────────────────────────────

def run_arm(version: int, items: List[Dict[str, Any]],
            run_name: str,
            skip_probes: Optional[List[str]] = None,
            cell_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
            parallel_workers: int = 1,
            ) -> List[Dict[str, Any]]:
    """Run all `items` through encoder version `version`. Returns list of
    cell results in input order. `cell_writer` is invoked once per cell as
    soon as it completes — use it to stream to per_cell.jsonl.

    `parallel_workers > 1` runs cells concurrently via ThreadPoolExecutor.
    Threads (not processes) — sqlite3 connections are per-cell anyway
    (each cell's per_item brain has its own connections), and the workload
    is I/O bound (Sonnet/Haiku API calls release the GIL). The daemon
    serializes its own writes via write_lock, so parallel API hits
    naturally backpressure.
    """
    arm_run_name = f"{run_name}-v{version}"
    template_path = materialize_s1e_template(version)
    print(f"\n[encoder_eval] === ARM v{version} === ({len(items)} items, "
          f"workers={parallel_workers})", file=sys.stderr, flush=True)
    cells_by_idx: Dict[int, Dict[str, Any]] = {}
    write_lock = None
    try:
        if parallel_workers <= 1:
            for i, item in enumerate(items):
                cell = run_cell(
                    version=version, item=item, arm_run_name=arm_run_name,
                    item_idx=i, total=len(items),
                    template_path=template_path, skip_probes=skip_probes)
                cells_by_idx[i] = cell
                if cell_writer is not None:
                    try:
                        cell_writer(cell)
                    except Exception as e:
                        print(f"[encoder_eval] cell_writer failed: {e}",
                              file=sys.stderr, flush=True)
        else:
            import threading
            from concurrent.futures import ThreadPoolExecutor, as_completed
            write_lock = threading.Lock()

            def _one(idx, item):
                return idx, run_cell(
                    version=version, item=item, arm_run_name=arm_run_name,
                    item_idx=idx, total=len(items),
                    template_path=template_path, skip_probes=skip_probes)

            with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
                futures = [pool.submit(_one, i, it)
                           for i, it in enumerate(items)]
                for fut in as_completed(futures):
                    try:
                        idx, cell = fut.result()
                    except Exception as e:
                        print(f"[encoder_eval] cell raised: {e!r}",
                              file=sys.stderr, flush=True)
                        continue
                    cells_by_idx[idx] = cell
                    if cell_writer is not None:
                        with write_lock:
                            try:
                                cell_writer(cell)
                            except Exception as e:
                                print(f"[encoder_eval] cell_writer failed: {e}",
                                      file=sys.stderr, flush=True)
    finally:
        try:
            os.unlink(template_path)
        except Exception:
            pass
    return [cells_by_idx[i] for i in sorted(cells_by_idx.keys())]


# ─────────────────────────────────────────────────────────────────
# Stop conditions — predicates over the accumulated cell list
# ─────────────────────────────────────────────────────────────────

def _v22_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in cells if c['version'] == 22]


def stop_if_v22_hex_format_errors_high(
        cells: List[Dict[str, Any]], threshold_pct: float = 5.0
        ) -> Optional[str]:
    v22 = _v22_cells(cells)
    if not v22:
        return None
    failures = sum(
        (c.get('probes', {}).get('source_refs_coverage', {})
          .get('hex_format_failures', 0))
        for c in v22)
    total_refs = sum(
        (c.get('probes', {}).get('source_refs_coverage', {})
          .get('nodes_with_refs', 0))
        for c in v22)
    if total_refs == 0:
        return None
    pct = failures / max(1, total_refs) * 100
    if pct > threshold_pct:
        return (f"v22 hex-format error rate {pct:.1f}% > threshold "
                f"{threshold_pct:.1f}% — encoder regression in source_refs format")
    return None


def stop_if_v22_zero_source_refs(
        cells: List[Dict[str, Any]]) -> Optional[str]:
    v22 = _v22_cells(cells)
    if not v22:
        return None
    refs_total = sum(
        (c.get('probes', {}).get('source_refs_coverage', {})
          .get('nodes_with_refs', 0))
        for c in v22)
    nodes_total = sum(
        (c.get('probes', {}).get('source_refs_coverage', {})
          .get('nodes_encoded', 0))
        for c in v22)
    if nodes_total > 5 and refs_total == 0:
        return (f"v22 produced 0 source_refs across {nodes_total} encoded "
                f"nodes — substrate teaching failed")
    return None


def stop_if_v22_answers_regress(
        cells: List[Dict[str, Any]], baseline_version: int = 19,
        per_axis_threshold_pp: float = 10.0) -> Optional[str]:
    """Compute per-axis pass rate for v22 vs baseline. If any axis is
    >threshold_pp worse, halt."""
    by_va = {}  # (version, axis) -> [correct booleans]
    for c in cells:
        v, a = c['version'], c.get('axis') or 'unknown'
        corr = (c.get('longmem_result') or {}).get('correct', False)
        by_va.setdefault((v, a), []).append(corr)
    axes = sorted({a for (_, a) in by_va.keys()})
    regressions = []
    for a in axes:
        v22_rate = sum(by_va.get((22, a), [])) / max(1, len(by_va.get((22, a), [1])))
        v_base_rate = sum(by_va.get((baseline_version, a), [])) / max(
            1, len(by_va.get((baseline_version, a), [1])))
        delta = (v22_rate - v_base_rate) * 100
        if delta < -per_axis_threshold_pp:
            regressions.append(f"axis={a} v22={v22_rate:.1%} "
                                f"v{baseline_version}={v_base_rate:.1%} "
                                f"Δ={delta:+.1f}pp")
    if regressions:
        return ("v22 answer regression on " + " | ".join(regressions))
    return None


DEFAULT_STOP_CONDITIONS = [
    stop_if_v22_hex_format_errors_high,
    stop_if_v22_zero_source_refs,
    stop_if_v22_answers_regress,
]


def evaluate_stop_conditions(
        cells: List[Dict[str, Any]],
        conditions: List[Callable] = DEFAULT_STOP_CONDITIONS,
        ) -> List[str]:
    """Return list of stop reasons that fired (empty list = continue)."""
    reasons = []
    for cond in conditions:
        try:
            r = cond(cells)
            if r:
                reasons.append(r)
        except Exception as e:
            reasons.append(f"stop-condition {cond.__name__} raised: {e!r}")
    return reasons


# ─────────────────────────────────────────────────────────────────
# Staged run — the top-level entry called by runner.py
# ─────────────────────────────────────────────────────────────────

def run_staged(
        versions: List[int],
        stages: List[Dict[str, Any]],  # [{name, items}]
        run_name: str,
        out_dir: Path,
        skip_probes: Optional[List[str]] = None,
        stop_conditions: List[Callable] = DEFAULT_STOP_CONDITIONS,
        continue_on_stop: bool = False,
        parallel_workers: int = 1,
        ) -> Dict[str, Any]:
    """Run versions × stages with checkpointed halt-on-stop semantics.

    Streams `per_cell.jsonl` as cells complete. After each stage, writes a
    `per_stage_{name}.json` snapshot and evaluates stop_conditions against
    ALL cells accumulated so far (not just this stage). If any stop fires
    and continue_on_stop is False, the run halts and returns the partial
    result.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cell_path = out_dir / 'per_cell.jsonl'
    per_cell_fh = open(per_cell_path, 'a')
    def _cell_writer(cell: Dict[str, Any]):
        per_cell_fh.write(json.dumps(cell, default=str) + '\n')
        per_cell_fh.flush()

    all_cells: List[Dict[str, Any]] = []
    stage_records = []
    halted = False
    halt_reasons: List[str] = []

    try:
        for stage in stages:
            stage_name = stage['name']
            stage_items = stage['items']
            print(f"\n[encoder_eval] ┌─ STAGE {stage_name}: "
                  f"{len(stage_items)} items × {len(versions)} versions",
                  file=sys.stderr, flush=True)
            stage_cells = []
            t_stage = time.time()
            for v in versions:
                cells = run_arm(
                    version=v, items=stage_items,
                    run_name=f"{run_name}-{stage_name}",
                    skip_probes=skip_probes,
                    cell_writer=_cell_writer,
                    parallel_workers=parallel_workers)
                stage_cells.extend(cells)
                all_cells.extend(cells)
            stage_ms = int((time.time() - t_stage) * 1000)

            # Checkpoint
            stage_record = {
                'stage': stage_name,
                'n_items': len(stage_items),
                'n_cells': len(stage_cells),
                'wall_ms': stage_ms,
            }
            reasons = evaluate_stop_conditions(all_cells, stop_conditions)
            if reasons:
                stage_record['stop_reasons'] = reasons
                halted = not continue_on_stop
                halt_reasons = reasons
                print(f"\n[encoder_eval] ⚠️  STOP CONDITIONS FIRED:",
                      file=sys.stderr, flush=True)
                for r in reasons:
                    print(f"  • {r}", file=sys.stderr, flush=True)
            stage_records.append(stage_record)

            stage_snapshot_path = (out_dir
                                    / f"per_stage_{stage_name}.json")
            with open(stage_snapshot_path, 'w') as f:
                json.dump({
                    'stage': stage_name,
                    'cells_this_stage': len(stage_cells),
                    'cells_total': len(all_cells),
                    'wall_ms': stage_ms,
                    'stop_reasons': reasons,
                }, f, indent=2, default=str)

            print(f"[encoder_eval] └─ stage {stage_name} done "
                  f"({stage_ms/1000:.1f}s)", file=sys.stderr, flush=True)

            if halted:
                print(f"\n[encoder_eval] HALTING — re-run with "
                      f"--continue-on-stop to override.",
                      file=sys.stderr, flush=True)
                break
    finally:
        per_cell_fh.close()

    return {
        'run_name': run_name,
        'versions': versions,
        'n_cells': len(all_cells),
        'stage_records': stage_records,
        'halted': halted,
        'halt_reasons': halt_reasons,
        'out_dir': str(out_dir),
    }
