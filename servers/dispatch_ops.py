"""Daemon dispatch — lifecycle / config / admin handlers.

ping, health, save, session, config, plus maintenance one-offs
(backfill / migrate / diagnose / eval).
"""

import json
import os
import time

from .daemon_config import _CODE_FINGERPRINT, _PROCESS_STARTED_AT, REPO_ROOT
from .dispatch_common import caller_session


def _handle_ping(brain, args, graph_changes):
    import threading as _t
    result = {"status": "alive", "pid": os.getpid(),
              "code_fingerprint": _CODE_FINGERPRINT,
              "source_dir": REPO_ROOT,
              # Which brain this daemon is actually writing — lets callers
              # detect db-path divergence (daemon_client._db_dir_changed) the
              # way source_dir lets them detect code staleness.
              "db_dir": os.path.dirname(os.path.abspath(brain.db_path)),
              # How long THIS PROCESS has been up (not the current serving run —
              # see _PROCESS_STARTED_AT). The dashboard health tab renders it.
              "uptime_seconds": int(time.time() - _PROCESS_STARTED_AT),
              "threads": _t.active_count()}
    if args.get("thread_detail"):
        result["thread_list"] = [
            {"name": t.name, "daemon": t.daemon, "alive": t.is_alive()}
            for t in _t.enumerate()
        ]
    return {"ok": True, "result": result}


def _handle_health_check(brain, args, graph_changes):
    return {"ok": True, "result": brain.health_check(
        session_id=args.get("session_id", "daemon"),
        auto_fix=args.get("auto_fix", True))}


def _handle_validate_config(brain, args, graph_changes):
    return {"ok": True, "result": {"warnings": brain.validate_config()}}


def _handle_scan_host(brain, args, graph_changes):
    return {"ok": True, "result": brain.scan_host_environment()}


def _handle_procedure_trigger(brain, args, graph_changes):
    return {"ok": True, "result": brain.procedure_trigger(
        trigger=args.get("trigger", ""),
        context=args.get("context", {}))}


def _handle_get_config(brain, args, graph_changes):
    return {"ok": True, "result": {"value": brain.get_config(
        args.get("key", ""), args.get("default", ""))}}


def _handle_get_debug_status(brain, args, graph_changes):
    return {"ok": True, "result": {"debug": brain.get_debug_status()}}


def _handle_enrichment_coverage(brain, args, graph_changes):
    return {"ok": True, "result": brain.get_enrichment_coverage()}


def _handle_pre_edit(brain, args, graph_changes):
    """Pre-edit handler. The expensive recall under suggest() is
    deduplicated at the recall layer (brain.recall result cache + single
    flight) so concurrent / repeat pre_edit calls collapse there. This
    handler stays simple."""
    file = args.get("file", "")
    tool_name = args.get("tool_name", "Edit")
    sid = caller_session(args)  # identity: per-session pre-edit surfacing
    ctx = brain.get_or_create_session(sid) if sid else None
    data = brain.pre_edit(file=file, tool_name=tool_name, ctx=ctx)
    try:
        data["change_impacts"] = brain.get_change_impact(file)
    except Exception as e:
        brain._log_error("pre_edit_change_impact", e, "fetching change impacts for %s" % file[:60])
        data["change_impacts"] = []
    return {"ok": True, "result": data}


def _handle_save(brain, args, graph_changes):
    brain.save()
    return {"ok": True, "result": {"status": "saved"}}


def _handle_reset_session(brain, args, graph_changes):
    sid = args.get("session_id", "")
    # cwd is fed from the boot hook (Claude side); the daemon never introspects
    # it. reset_session_activity stamps it as session identity + derives branch,
    # and returns True when it RESUMED an existing session (counters preserved).
    is_resume = brain.reset_session_activity(session_id=sid, cwd=args.get("cwd", ""))
    return {"ok": True, "result": {
        "status": "resumed" if is_resume else "reset",
        "session_id": sid,
    }}


def _handle_set_config(brain, args, graph_changes):
    brain.set_config(args.get("key", ""), args.get("value", ""))
    return {"ok": True, "result": {"status": "set"}}


def _handle_promote_staged(brain, args, graph_changes):
    brain.auto_promote_staged(revisit_threshold=args.get("threshold", 3))
    return {"ok": True, "result": {"status": "promoted"}}


def _handle_backfill_summaries(brain, args, graph_changes):
    return {"ok": True, "result": brain.backfill_summaries(
        batch_size=args.get("batch_size", 50))}


def _handle_backfill_vectors(brain, args, graph_changes):
    """Run backfill_vectors over the graph. Computes ALL missing vector
    types per EMBEDDING_GROUPS — `_primary`, `_situation`, `title`,
    `high_meta`, `other_meta`, `edge_context`, `question`, plus the
    field cohort. Returns counts per vector_type."""
    return {"ok": True, "result": brain.backfill_vectors(
        batch_size=args.get("batch_size", 100),
        node_ids=args.get("node_ids"))}


def _handle_diagnose(brain, args, graph_changes):
    """Dump per-thread stacks + vector cache stats + recent error activity.

    In-process introspection. Survives SIP-protected Python (unlike
    py-spy/lldb). Call when the daemon is sluggish or spinning — returns
    enough detail to identify the hot code path without external tools.

    Args (all optional):
        write_file: path to write full dump to (default /tmp/brain-diagnose-{ts}.txt)

    Returns: dict with thread_count, hot_threads (stacks), cache_stats,
             recent_errors, and the file path.
    """
    import os as _os
    import sys as _sys
    import threading as _t
    import time as _time
    import traceback as _tb
    ts = int(_time.time())
    default_path = '/tmp/brain-diagnose-%d.txt' % ts
    write_file = args.get('write_file') or default_path

    # Per-thread stacks via sys._current_frames() — no debugger needed.
    frames = _sys._current_frames()
    alive = {t.ident: t for t in _t.enumerate()}
    threads_out = []
    for tid, frame in frames.items():
        t = alive.get(tid)
        stack = _tb.extract_stack(frame)
        # Deepest (most recent) 12 frames — enough to pinpoint the hot loop.
        frames_short = [
            {'file': _os.path.basename(f.filename), 'line': f.lineno,
             'func': f.name, 'code': (f.line or '').strip()[:120]}
            for f in stack[-12:]
        ]
        threads_out.append({
            'tid': tid,
            'name': (t.name if t else '?'),
            'daemon': (t.daemon if t else None),
            'alive': (t.is_alive() if t else None),
            'frames': frames_short,
        })

    # Vector cache diagnostics (added 2026-04-19 for recall thrash debugging).
    try:
        cache_stats = brain._vec_dal.cache_stats()
    except Exception as e:
        cache_stats = {'error': str(e)}

    # Recent errors — hours defaults to 2. The door degrades to [] on its
    # own internal failure, so no handler-side try/except: a second swallow
    # here would hide which layer failed.
    hours = int(args.get('hours', 2))
    recent_errors = brain.get_recent_errors(hours=hours)

    result = {
        'timestamp': ts,
        'pid': _os.getpid(),
        'thread_count': len(threads_out),
        'cache_stats': cache_stats,
        'recent_errors': recent_errors[:20],
        'file': write_file,
        'threads': threads_out,
    }

    # Write full dump to file so the full stack trace is available even
    # when the TCP response gets truncated by the client.
    try:
        with open(write_file, 'w') as f:
            f.write('brain-diagnose @ %s (pid=%d)\n' % (ts, _os.getpid()))
            f.write('thread_count=%d\n\n' % len(threads_out))
            for t in threads_out:
                f.write('--- Thread %s (id=%s, daemon=%s) ---\n' % (
                    t['name'], t['tid'], t['daemon']))
                for fr in t['frames']:
                    f.write('  %s:%d  %s  |  %s\n' % (
                        fr['file'], fr['line'], fr['func'], fr['code']))
                f.write('\n')
            f.write('\n=== Vector cache stats ===\n%s\n\n' % cache_stats)
            f.write('=== Recent errors (last %dh) ===\n' % hours)
            for e in recent_errors[:20]:
                f.write('  %s\n' % e)
        result['file_written'] = True
    except Exception as e:
        result['file_error'] = str(e)
        result['file_written'] = False

    return {'ok': True, 'result': result}


def _handle_eval(brain, args, graph_changes):
    """Evaluate a Python expression against the live brain — DEBUG ONLY.

    This is effectively arbitrary code execution: the safe_builtins sandbox is
    weak (the `brain` object alone exposes the full DB, and __builtins__
    restriction is bypassable). Acceptable ONLY because the daemon binds to
    127.0.0.1 and is single-user. Do not expose the daemon on a non-loopback
    interface without gating or removing this handler.
    """
    code = args.get("code", "")
    if not code:
        return {"ok": False, "error": "No code provided"}
    safe_builtins = {"str": str, "int": int, "len": len, "list": list, "dict": dict,
                     "bool": bool, "float": float, "round": round, "sorted": sorted,
                     "min": min, "max": max, "sum": sum, "abs": abs, "type": type,
                     "isinstance": isinstance, "range": range, "enumerate": enumerate,
                     "zip": zip, "map": map, "filter": filter, "print": print,
                     "True": True, "False": False, "None": None}
    local_vars = {"brain": brain, "json": json}
    result = eval(code, {"__builtins__": safe_builtins}, local_vars)
    try:
        json.dumps(result)
    except (TypeError, ValueError):
        result = str(result)
    return {"ok": True, "result": result}


def _handle_drop_sys_revision_history(brain, args, graph_changes):
    """Stage 1A migration: drop legacy _sys_revision_history KV blobs.

    Revision history moved to trace events (event_type='delta',
    ref_type='node_revised'). The legacy _sys_revision_history JSON blob in
    node_metadata_kv is no longer written (see brain_remember.py:revise) and
    is never read by anything (audit confirmed).

    Per Stage 1A spec: drop the data, no retroactive trace conversion.

    Args:
        commit: bool — if False (default), reports count without deleting.

    Returns: {commit, count_found, count_deleted}.
    Idempotent — safe to re-run.
    """
    commit = bool(args.get('commit', False))

    count = brain.conn.execute(
        "SELECT COUNT(*) FROM node_metadata_kv WHERE key = '_sys_revision_history'"
    ).fetchone()[0]

    deleted = 0
    if commit and count > 0:
        cursor = brain.conn.execute(
            "DELETE FROM node_metadata_kv WHERE key = '_sys_revision_history'")
        brain.conn.commit()
        deleted = cursor.rowcount

    graph_changes.append("DROP_REVHISTORY: found=%d deleted=%d (commit=%s)" % (
        count, deleted, commit))

    return {"ok": True, "result": {
        'commit': commit,
        'count_found': count,
        'count_deleted': deleted,
    }}
