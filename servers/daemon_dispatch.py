"""
brain — Daemon Command Dispatch

Table-driven command routing. Each command maps to a handler function
with read/write classification for lock management.

Commands are plain functions: handler(brain, args, graph_changes) → dict
The daemon_server.py calls dispatch() which looks up the handler and
reports whether a write lock is needed.
"""

import json
import os
from typing import Any, Dict, List, NamedTuple, Callable, Optional

from .daemon_config import _CODE_FINGERPRINT


def _resolve_id(brain, node_id):
    """Resolve a node ID — exact match first, then prefix match.

    Handles both new 8-char IDs and old 32-char IDs gracefully.
    Returns the full ID if found, or the original input if not.
    """
    if not node_id:
        return node_id
    # Exact match
    row = brain.conn.execute('SELECT id FROM nodes WHERE id = ?', (node_id,)).fetchone()
    if row:
        return row[0]
    # Prefix match (for truncated IDs from tool calls)
    rows = brain.conn.execute('SELECT id FROM nodes WHERE id LIKE ?', (node_id + '%',)).fetchall()
    if len(rows) == 1:
        return rows[0][0]
    return node_id  # Not found or ambiguous — let the caller handle the error


class CmdEntry(NamedTuple):
    handler: Callable
    is_write: bool
    marks_dirty: bool = False


# ═══════════════════════════════════════════════════════════════
# READ handlers — safe for concurrent execution (no lock needed)
# ═══════════════════════════════════════════════════════════════

def _handle_ping(brain, args, graph_changes):
    import threading as _t
    result = {"status": "alive", "pid": os.getpid(),
              "code_fingerprint": _CODE_FINGERPRINT,
              "threads": _t.active_count()}
    if args.get("thread_detail"):
        result["thread_list"] = [
            {"name": t.name, "daemon": t.daemon, "alive": t.is_alive()}
            for t in _t.enumerate()
        ]
    return {"ok": True, "result": result}


def _handle_context_boot(brain, args, graph_changes):
    text = brain.format_boot_context(
        user=args.get("user", "User"),
        project=args.get("project", "default"),
        db_dir=args.get("db_dir", ""))
    return {"ok": True, "result": text}


def _handle_recall(brain, args, graph_changes):
    # By-ID recall: returns single enriched node
    node_id = args.get("node_id")
    if node_id:
        node_id = _resolve_id(brain, node_id)
        result = brain.recall_node(
            node_id, neighbor_limit=args.get("neighbor_limit", 3))
        return {"ok": True, "result": result}

    # By-query recall: semantic search with keyword fallback
    try:
        result = brain.recall(
            query=args.get("query", ""), limit=args.get("limit", 8),
            source='mcp')
        return {"ok": True, "result": result}
    except Exception as e:
        # Degraded: fall back to keyword-only recall
        result = brain.recall(
            query=args.get("query", ""), limit=args.get("limit", 8),
            source='mcp')
        try:
            brain._log_error("recall_degraded", e, "Fell back to keyword-only recall")
        except Exception as e2:
            print('[daemon_dispatch] ERROR logging recall_degraded: %s (original: %s)' % (e2, e), file=__import__('sys').stderr)
        return {"ok": True, "result": result,
                "_degraded": "keyword_fallback", "_reason": str(e)}


def _handle_heartbeat(brain, args, graph_changes):
    nudge = brain.get_encoding_heartbeat(
        nudge_threshold=args.get("threshold", 8))
    return {"ok": True, "result": {"nudge": nudge}}


def _handle_validate_config(brain, args, graph_changes):
    return {"ok": True, "result": {"warnings": brain.validate_config()}}


def _handle_health_check(brain, args, graph_changes):
    return {"ok": True, "result": brain.health_check(
        session_id=args.get("session_id", "daemon"),
        auto_fix=args.get("auto_fix", True))}


def _handle_consciousness(brain, args, graph_changes):
    """Migrated to signal_producers + signal queue. Returns reminders only."""
    return {"ok": True, "result": {"reminders": brain.get_due_reminders()}}


def _handle_dismiss_signal(brain, args, graph_changes):
    """Dismiss a signal from the queue by ID or producer."""
    from .dal_signal_queue import SignalQueueDAL
    sq_dal = SignalQueueDAL(brain.logs_conn)
    signal_id = args.get("signal_id", "")
    producer = args.get("producer", "")
    if signal_id:
        ok = sq_dal.dismiss(signal_id)
        return {"ok": True, "result": {"dismissed": signal_id, "found": ok}}
    elif producer:
        count = sq_dal.dismiss_by_producer(producer)
        return {"ok": True, "result": {"dismissed_producer": producer, "count": count}}
    return {"ok": False, "error": "Provide signal_id or producer"}


def _handle_queue_state(brain, args, graph_changes):
    """Return current signal queue state."""
    from .dal_signal_queue import SignalQueueDAL
    sq_dal = SignalQueueDAL(brain.logs_conn)
    return {"ok": True, "result": sq_dal.get_queue_state()}


def _handle_engineering_context(brain, args, graph_changes):
    return {"ok": True, "result": brain.get_engineering_context(
        project=args.get("project", "default"))}


def _handle_correction_patterns(brain, args, graph_changes):
    return {"ok": True, "result": brain.get_correction_patterns(
        limit=args.get("limit", 5))}


def _handle_last_synthesis(brain, args, graph_changes):
    return {"ok": True, "result": brain.get_last_synthesis()}


def _handle_scan_host(brain, args, graph_changes):
    return {"ok": True, "result": brain.scan_host_environment()}


def _handle_dreams(brain, args, graph_changes):
    return {"ok": True, "result": brain.get_surfaceable_dreams(
        limit=args.get("limit", 2))}


def _handle_staged(brain, args, graph_changes):
    return {"ok": True, "result": brain.list_staged(
        status=args.get("status", "pending"),
        limit=args.get("limit", 10))}


def _handle_suggest_metrics(brain, args, graph_changes):
    return {"ok": True, "result": brain.get_suggest_metrics(
        period_days=args.get("period_days", 7))}


def _handle_procedure_trigger(brain, args, graph_changes):
    return {"ok": True, "result": brain.procedure_trigger(
        trigger=args.get("trigger", ""),
        context=args.get("context", {}))}


def _handle_get_config(brain, args, graph_changes):
    return {"ok": True, "result": {"value": brain.get_config(
        args.get("key", ""), args.get("default", ""))}}


def _handle_get_debug_status(brain, args, graph_changes):
    return {"ok": True, "result": {"debug": brain.get_debug_status()}}


def _handle_get_active_evolutions(brain, args, graph_changes):
    return {"ok": True, "result": brain.get_active_evolutions(
        args.get("types"))}


def _handle_assess_dev_stage(brain, args, graph_changes):
    return {"ok": True, "result": brain.assess_developmental_stage()}


def _handle_instinct_check(brain, args, graph_changes):
    return {"ok": True, "result": {"nudge": brain.get_instinct_check(
        args.get("message", ""))}}


def _handle_prompt_reflection(brain, args, graph_changes):
    return {"ok": True, "result": brain.prompt_reflection()}


def _handle_enrichment_coverage(brain, args, graph_changes):
    return {"ok": True, "result": brain.get_enrichment_coverage()}


def _handle_pre_edit(brain, args, graph_changes):
    data = brain.pre_edit(
        file=args.get("file", ""),
        tool_name=args.get("tool_name", "Edit"))
    try:
        data["change_impacts"] = brain.get_change_impact(args.get("file", ""))
    except Exception as e:
        brain._log_error("pre_edit_change_impact", e, "fetching change impacts for %s" % args.get("file", "")[:60])
        data["change_impacts"] = []
    return {"ok": True, "result": data}


# ═══════════════════════════════════════════════════════════════
# WRITE handlers — require exclusive lock
# ═══════════════════════════════════════════════════════════════

def _handle_save(brain, args, graph_changes):
    brain.save()
    return {"ok": True, "result": {"status": "saved"}}


def _handle_record_message(brain, args, graph_changes):
    brain.record_message()
    nudge = brain.get_encoding_heartbeat()
    return {"ok": True, "result": {"nudge": nudge}}


def _handle_reset_session(brain, args, graph_changes):
    brain.reset_session_activity()
    return {"ok": True, "result": {"status": "reset"}}


def _handle_set_config(brain, args, graph_changes):
    brain.set_config(args.get("key", ""), args.get("value", ""))
    return {"ok": True, "result": {"status": "set"}}


def _handle_log_debug(brain, args, graph_changes):
    brain.log_debug(
        args.get("event", ""), args.get("source", ""),
        metadata=args.get("metadata"))
    return {"ok": True, "result": {"status": "logged"}}


def _handle_self_reflection(brain, args, graph_changes):
    brain.auto_generate_self_reflection()
    return {"ok": True, "result": {"status": "generated"}}


def _handle_promote_staged(brain, args, graph_changes):
    brain.auto_promote_staged(revisit_threshold=args.get("threshold", 3))
    return {"ok": True, "result": {"status": "promoted"}}


def _handle_consolidate(brain, args, graph_changes):
    return {"ok": True, "result": brain.consolidate()}


def _handle_dream(brain, args, graph_changes):
    return {"ok": True, "result": brain.dream()}


def _handle_auto_heal(brain, args, graph_changes):
    return {"ok": True, "result": brain.auto_heal()}


def _handle_auto_tune(brain, args, graph_changes):
    return {"ok": True, "result": brain.auto_tune()}


def _handle_backfill_summaries(brain, args, graph_changes):
    return {"ok": True, "result": brain.backfill_summaries(
        batch_size=args.get("batch_size", 50))}


def _handle_synthesize_session(brain, args, graph_changes):
    return {"ok": True, "result": brain.synthesize_session()}


def _handle_remember(brain, args, graph_changes):
    from .contract import validate_field, get_remember_fields

    # Validate all provided fields against contract
    for field, value in args.items():
        ok, err = validate_field(field, value)
        if not ok:
            return {"ok": False, "error": err}

    # Pass ALL contract fields through to remember() — it handles routing
    # to nodes table, node_metadata, and node_embeddings
    accepted_fields = set(get_remember_fields().keys()) | {'connections'}
    remember_args = {k: v for k, v in args.items() if k in accepted_fields and v is not None}
    result = brain.remember(**remember_args)
    node_id = result.get("id", "?")[:8] if isinstance(result, dict) else "?"
    graph_changes.append(
        "REMEMBER: [%s] %s (%s...)" % (
            args.get("type", "?"), args.get("title", "")[:50], node_id))
    return {"ok": True, "result": result}


def _handle_remember_batch(brain, args, graph_changes):
    from .contract import validate_field, get_remember_fields

    nodes = args.get("nodes", [])
    if not nodes:
        return {"ok": False, "error": "nodes array is required"}

    accepted_fields = set(get_remember_fields().keys())
    # Inherit top-level encoding_source into each node (dispatch wrapper injects this)
    top_encoding_source = args.get("encoding_source")

    cleaned_nodes = []
    for i, spec in enumerate(nodes):
        for field, value in spec.items():
            ok, err = validate_field(field, value)
            if not ok:
                return {"ok": False, "error": "node[%d].%s: %s" % (i, field, err)}
        cleaned = {k: v for k, v in spec.items() if k in accepted_fields and v is not None}
        if top_encoding_source and 'encoding_source' not in cleaned:
            cleaned['encoding_source'] = top_encoding_source
        cleaned_nodes.append(cleaned)

    result = brain.remember_batch(
        nodes=cleaned_nodes,
        connect_to=args.get("connect_to"),
        auto_connect=args.get("auto_connect", True))
    graph_changes.append("REMEMBER_BATCH: %d nodes" % result.get("nodes_created", 0))
    return {"ok": True, "result": result}


def _handle_revise(brain, args, graph_changes):
    """Update any field(s) on an existing node via revise()."""
    from .contract import validate_field, ALL_FIELDS

    node_id = _resolve_id(brain, args.get("node_id", ""))
    reason = args.get("reason", "")
    if not node_id:
        return {"ok": False, "error": "node_id is required"}
    if not reason:
        return {"ok": False, "error": "reason is required"}

    # Validate all update fields against contract
    updates = {k: v for k, v in args.items() if k not in ("node_id", "reason")}
    for field, value in updates.items():
        ok, err = validate_field(field, value)
        if not ok:
            return {"ok": False, "error": err}

    content = updates.pop("content", None)
    result = brain.revise(node_id=node_id, content=content, reason=reason, updates=updates)
    if result.get('error'):
        return {"ok": False, "error": result['error']}

    # Surface verification failures as warnings
    if not result.get('verified', True):
        failures = result.get('verification_failures', [])
        graph_changes.append("VERIFY_FAIL: revise %s — fields not confirmed: %s" % (
            node_id[:12], ', '.join(failures)))
        # Log to brain error system so integrity producer can surface it
        try:
            brain._log_error('write_verification',
                Exception('Revise verification failed for %s: %s' % (node_id[:12], failures)),
                'Fields claimed updated but read-back shows mismatch')
        except Exception as e2:
            print('[daemon_dispatch] ERROR logging write_verification: %s' % e2, file=__import__('sys').stderr)

    graph_changes.append("REVISE: [%s] %s" % (
        result.get("type", "?"), result.get("title", "")[:50]))
    return {"ok": True, "result": result}


def _handle_remember_lesson(brain, args, graph_changes):
    result = brain.remember_lesson(
        title=args.get("title", ""),
        what_happened=args.get("what_happened", ""),
        root_cause=args.get("root_cause", ""),
        fix=args.get("fix", ""),
        preventive_principle=args.get("preventive_principle", ""),
        project=args.get("project"))
    graph_changes.append("REMEMBER_LESSON: %s" % args.get("title", "")[:50])
    return {"ok": True, "result": result}


def _handle_remember_impact(brain, args, graph_changes):
    result = brain.remember_impact(
        title=args.get("title", ""),
        if_changed=args.get("if_changed", ""),
        must_check=args.get("must_check", ""),
        because=args.get("because", ""),
        scope=args.get("scope"),
        severity=args.get("severity"),
        project=args.get("project"))
    graph_changes.append("REMEMBER_IMPACT: %s" % args.get("title", "")[:50])
    return {"ok": True, "result": result}


def _handle_remember_mechanism(brain, args, graph_changes):
    result = brain.remember_mechanism(
        title=args.get("title", ""),
        content=args.get("content", ""),
        scope=args.get("scope"),
        steps=args.get("steps"),
        data_flow=args.get("data_flow"),
        project=args.get("project"))
    graph_changes.append("REMEMBER_MECHANISM: %s" % args.get("title", "")[:50])
    return {"ok": True, "result": result}


def _handle_remember_uncertainty(brain, args, graph_changes):
    result = brain.remember_uncertainty(
        title=args.get("title", ""),
        what_unknown=args.get("what_unknown", ""),
        why_it_matters=args.get("why_it_matters", ""),
        project=args.get("project"))
    graph_changes.append("REMEMBER_UNCERTAINTY: %s" % args.get("title", "")[:50])
    return {"ok": True, "result": result}


def _handle_remember_convention(brain, args, graph_changes):
    result = brain.remember_convention(
        title=args.get("title", ""),
        content=args.get("content", ""),
        scope=args.get("scope"),
        examples=args.get("examples"),
        anti_patterns=args.get("anti_patterns"),
        project=args.get("project"))
    graph_changes.append("REMEMBER_CONVENTION: %s" % args.get("title", "")[:50])
    return {"ok": True, "result": result}


def _handle_remember_mental_model(brain, args, graph_changes):
    result = brain.remember_mental_model(
        title=args.get("title", ""),
        model_description=args.get("model_description", ""),
        applies_to=args.get("applies_to"),
        confidence=args.get("confidence", 0.7),
        project=args.get("project"))
    graph_changes.append("REMEMBER_MENTAL_MODEL: %s" % args.get("title", "")[:50])
    return {"ok": True, "result": result}


def _handle_record_divergence(brain, args, graph_changes):
    result = brain.record_divergence(
        claude_assumed=args.get("claude_assumed", ""),
        reality=args.get("reality", ""),
        underlying_pattern=args.get("underlying_pattern", ""),
        severity=args.get("severity", "medium"),
        original_node_id=args.get("original_node_id"),
        entity=args.get("entity"),
        project=args.get("project"))
    graph_changes.append("RECORD_DIVERGENCE: %s" % args.get("claude_assumed", "")[:50])
    return {"ok": True, "result": result}


def _handle_learn_vocabulary(brain, args, graph_changes):
    result = brain.learn_vocabulary(
        term=args.get("term", ""),
        maps_to=args.get("maps_to", ""),
        context=args.get("context"),
        project=args.get("project"))
    graph_changes.append("LEARN_VOCABULARY: %s" % args.get("term", "")[:50])
    return {"ok": True, "result": result}


def _handle_find_node_by_title(brain, args, graph_changes):
    result = brain.find_node_by_title(
        title_query=args.get("title_query", ""),
        threshold=args.get("threshold", 0.75),
        top_k=args.get("top_k", 1))
    return {"ok": True, "result": result}


def _handle_graph_expand(brain, args, graph_changes):
    """Layer 3: expand from judge-selected seed nodes via structural edges.

    Args:
        node_ids: list of seed node IDs (from judge selection)
        depth: how many hops (default 1)
        limit_per_seed: max neighbors per seed (default 3)

    Returns: list of neighbor nodes with edge info, deduplicated.
    """
    node_ids = args.get("node_ids", [])
    depth = min(args.get("depth", 1), 2)  # Cap at 2 hops
    limit_per = args.get("limit_per_seed", 3)

    if not node_ids:
        return {"ok": True, "result": {"neighbors": []}}

    from .brain_constants import EXCLUDED_EDGE_TYPES
    seen = set(node_ids)
    neighbors = []

    for seed_id in node_ids:
        # Resolve short IDs
        full_id = seed_id
        if len(seed_id) < 16:
            row = brain.conn.execute(
                "SELECT id FROM nodes WHERE id LIKE ?", (seed_id + '%',)).fetchone()
            if row:
                full_id = row[0]

        # Get structural neighbors (both directions — edges are bidirectional)
        rows = brain.conn.execute("""
            SELECT n.id, n.type, n.title, substr(n.content, 1, 300),
                   e.edge_type, e.weight, e.description, n.confidence
            FROM edges e
            JOIN nodes n ON n.id = CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
            WHERE (e.source_id = ? OR e.target_id = ?) AND n.archived = 0
            AND n.id != ?
            AND e.edge_type NOT IN ({})
            ORDER BY e.weight DESC LIMIT ?
        """.format(','.join('?' for _ in EXCLUDED_EDGE_TYPES)),
            [full_id, full_id, full_id, full_id] + list(EXCLUDED_EDGE_TYPES) + [limit_per]).fetchall()

        for r in rows:
            if r[0] not in seen:
                seen.add(r[0])
                neighbors.append({
                    "id": r[0], "type": r[1], "title": r[2],
                    "content": r[3], "edge_type": r[4],
                    "edge_weight": r[5], "edge_description": r[6] or "",
                    "confidence": r[7], "seed_id": full_id,
                })

    return {"ok": True, "result": {"neighbors": neighbors, "seeds": len(node_ids)}}


def _handle_get_node(brain, args, graph_changes):
    node_id = _resolve_id(brain, args.get("node_id", ""))
    if not node_id:
        return {"ok": False, "error": "node_id is required"}
    node = brain.get_node(node_id)
    if not node:
        return {"ok": False, "error": "Node not found: {}".format(node_id)}
    # Add connections
    try:
        edges = brain.conn.execute(
            "SELECT e.target_id, e.relation, e.weight, n.title, n.type "
            "FROM edges e LEFT JOIN nodes n ON n.id = e.target_id "
            "WHERE e.source_id = ? ORDER BY e.weight DESC LIMIT 10",
            (node_id,)
        ).fetchall()
        node["connections"] = [
            {"target_id": e[0], "relation": e[1], "weight": e[2],
             "title": e[3] or "", "type": e[4] or ""}
            for e in edges
        ]
    except Exception as e:
        brain._log_error("get_node_connections", e, "fetching connections for node %s" % node_id[:12])
        node["connections"] = []
    return {"ok": True, "result": node}


def _handle_encode_cluster(brain, args, graph_changes):
    result = brain.encode_cluster(
        nodes=args.get("nodes", []),
        connect_to=args.get("connect_to"),
        auto_connect=args.get("auto_connect", True))
    n = result.get("nodes_created", 0)
    graph_changes.append("ENCODE_CLUSTER: %d nodes" % n)
    return {"ok": True, "result": result}


def _handle_connect(brain, args, graph_changes):
    result = brain.connect(
        source_id=_resolve_id(brain, args.get("source_id", "")),
        target_id=_resolve_id(brain, args.get("target_id", "")),
        relation=args.get("relation", "related_to"),
        weight=args.get("weight", 0.5))
    graph_changes.append(
        "CONNECT: %s -[%s]-> %s" % (
            args.get("source_id", "?")[:8],
            args.get("relation", "related_to"),
            args.get("target_id", "?")[:8]))
    return {"ok": True, "result": result}


def _handle_enrich(brain, args, graph_changes):
    result = brain.store_enrichments(
        node_id=_resolve_id(brain, args.get("node_id", "")),
        question=args.get("question"),
        anchor=args.get("anchor"),
        bridge=args.get("bridge"),
        keywords=args.get("keywords"))
    stored = result.get("enrichments_stored", 0)
    graph_changes.append(
        "ENRICH: %s (+%d vectors)" % (args.get("node_id", "?")[:8], stored))
    return {"ok": True, "result": result}


def _handle_eval(brain, args, graph_changes):
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


# ═══════════════════════════════════════════════════════════════
# COMMAND TABLE — the single source of truth for routing
# ═══════════════════════════════════════════════════════════════

COMMAND_TABLE: Dict[str, CmdEntry] = {
    # ── Reads (no lock needed) ──
    "ping":                     CmdEntry(_handle_ping,                 is_write=False),
    "context_boot":             CmdEntry(_handle_context_boot,         is_write=False),
    "recall":                   CmdEntry(_handle_recall,               is_write=False),
    "heartbeat":                CmdEntry(_handle_heartbeat,            is_write=False),

    "validate_config":          CmdEntry(_handle_validate_config,      is_write=False),
    "health_check":             CmdEntry(_handle_health_check,         is_write=False),
    "consciousness":            CmdEntry(_handle_consciousness,        is_write=False),

    "dismiss_signal":           CmdEntry(_handle_dismiss_signal,       is_write=True),
    "queue_state":              CmdEntry(_handle_queue_state,          is_write=False),
    "engineering_context":      CmdEntry(_handle_engineering_context,   is_write=False),
    "correction_patterns":      CmdEntry(_handle_correction_patterns,  is_write=False),
    "last_synthesis":           CmdEntry(_handle_last_synthesis,       is_write=False),
    "scan_host":                CmdEntry(_handle_scan_host,            is_write=False),
    "dreams":                   CmdEntry(_handle_dreams,               is_write=False),
    "staged":                   CmdEntry(_handle_staged,               is_write=False),
    "suggest_metrics":          CmdEntry(_handle_suggest_metrics,      is_write=False),
    "procedure_trigger":        CmdEntry(_handle_procedure_trigger,    is_write=False),
    "get_config":               CmdEntry(_handle_get_config,           is_write=False),
    "get_debug_status":         CmdEntry(_handle_get_debug_status,     is_write=False),
    "get_active_evolutions":    CmdEntry(_handle_get_active_evolutions, is_write=False),
    "assess_developmental_stage": CmdEntry(_handle_assess_dev_stage,   is_write=False),
    "instinct_check":           CmdEntry(_handle_instinct_check,       is_write=False),
    "prompt_reflection":        CmdEntry(_handle_prompt_reflection,    is_write=False),
    "enrichment_coverage":      CmdEntry(_handle_enrichment_coverage,  is_write=False),
    "pre_edit":                 CmdEntry(_handle_pre_edit,             is_write=False),

    # ── Writes (exclusive lock) ──
    "save":                CmdEntry(_handle_save,               is_write=True, marks_dirty=False),
    "record_message":      CmdEntry(_handle_record_message,     is_write=True, marks_dirty=True),
    "reset_session":       CmdEntry(_handle_reset_session,      is_write=True, marks_dirty=True),
    "set_config":          CmdEntry(_handle_set_config,         is_write=True, marks_dirty=True),
    "log_debug":           CmdEntry(_handle_log_debug,          is_write=True, marks_dirty=True),
    "self_reflection":     CmdEntry(_handle_self_reflection,    is_write=True, marks_dirty=True),
    "promote_staged":      CmdEntry(_handle_promote_staged,     is_write=True, marks_dirty=True),
    "consolidate":         CmdEntry(_handle_consolidate,        is_write=True, marks_dirty=True),
    "dream":               CmdEntry(_handle_dream,              is_write=True, marks_dirty=True),
    "auto_heal":           CmdEntry(_handle_auto_heal,          is_write=True, marks_dirty=True),
    "auto_tune":           CmdEntry(_handle_auto_tune,          is_write=True, marks_dirty=True),
    "backfill_summaries":  CmdEntry(_handle_backfill_summaries, is_write=True, marks_dirty=True),
    "synthesize_session":  CmdEntry(_handle_synthesize_session, is_write=True, marks_dirty=True),
    "remember":              CmdEntry(_handle_remember,             is_write=True, marks_dirty=True),
    "remember_batch":        CmdEntry(_handle_remember_batch,      is_write=True, marks_dirty=True),
    "revise":                CmdEntry(_handle_revise,               is_write=True, marks_dirty=True),
    # Specialized remember tools — kept for interactive MCP use (Anchor calls these).
    # The encoding agent uses generic remember() with promoted fields instead.
    "remember_lesson":       CmdEntry(_handle_remember_lesson,     is_write=True, marks_dirty=True),
    "remember_impact":       CmdEntry(_handle_remember_impact,     is_write=True, marks_dirty=True),
    "remember_mechanism":    CmdEntry(_handle_remember_mechanism,  is_write=True, marks_dirty=True),
    "remember_uncertainty":  CmdEntry(_handle_remember_uncertainty, is_write=True, marks_dirty=True),
    "remember_convention":   CmdEntry(_handle_remember_convention, is_write=True, marks_dirty=True),
    "remember_mental_model": CmdEntry(_handle_remember_mental_model, is_write=True, marks_dirty=True),
    "record_divergence":     CmdEntry(_handle_record_divergence,   is_write=True, marks_dirty=True),
    "learn_vocabulary":      CmdEntry(_handle_learn_vocabulary,    is_write=True, marks_dirty=True),
    "find_node_by_title":    CmdEntry(_handle_find_node_by_title,  is_write=False, marks_dirty=False),
    "get_node":              CmdEntry(_handle_get_node,             is_write=False, marks_dirty=False),
    "graph_expand":          CmdEntry(_handle_graph_expand,         is_write=False, marks_dirty=False),
    # encode_cluster: DEPRECATED — use remember_batch() instead. Handler kept for backward compat.
    # "encode_cluster":        CmdEntry(_handle_encode_cluster,      is_write=True, marks_dirty=True),
    "connect":               CmdEntry(_handle_connect,             is_write=True, marks_dirty=True),
    "enrich":                CmdEntry(_handle_enrich,              is_write=True, marks_dirty=True),
    "eval":                  CmdEntry(_handle_eval,                is_write=True, marks_dirty=True),
}

# "shutdown" is handled directly by daemon_server (needs to set self.running)
# "hook_*" commands are dispatched via HOOK_TABLE in daemon_server
