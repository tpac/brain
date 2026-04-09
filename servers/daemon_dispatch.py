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
    from servers.dal import NodeDAL
    dal = NodeDAL(brain.conn)
    full_id = dal.resolve_id(node_id)
    return full_id if full_id else node_id  # Not found — let the caller handle the error


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


def _enrich_recall_results(brain, result, graph_changes):
    """Enrich recall results via get_rich_node — the shared data atom.

    Anchor's MCP recall gets full enrichment per node:
    metadata, corrections, connections, situation.
    """
    from .pipeline_contract import get_rich_node

    results = result.get("results", [])
    if not results:
        return

    # Enrich each result with get_rich_node data
    for r in results[:8]:
        rich = get_rich_node(brain, r.get("id", ""))
        if rich:
            r["_metadata"] = rich.get("_metadata", {})
            r["_corrections"] = rich.get("_corrections", [])
            r["connections"] = rich.get("connections", [])
            r["situation"] = rich.get("situation", "")


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
            query=args.get("query", ""), filter=args.get("filter"),
            limit=args.get("limit", 8), source='mcp')
    except Exception as e:
        # Degraded: fall back to keyword-only recall
        result = brain.recall(
            query=args.get("query", ""), filter=args.get("filter"),
            limit=args.get("limit", 8), source='mcp')
        try:
            brain._log_error("recall_degraded", e, "Fell back to keyword-only recall")
        except Exception as e2:
            print('[daemon_dispatch] ERROR logging recall_degraded: %s (original: %s)' % (e2, e), file=__import__('sys').stderr)
        result["_degraded"] = "keyword_fallback"
        result["_reason"] = str(e)

    # Enrich with corrections, graph expansion, metadata — same context as hook path
    _enrich_recall_results(brain, result, graph_changes)

    return {"ok": True, "result": result}


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
    brain.reset_session_activity(session_id=args.get("session_id", ""))
    return {"ok": True, "result": {"status": "reset", "session_id": brain.session_id}}


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


def _handle_revise_batch(brain, args, graph_changes):
    """Revise multiple nodes in one call."""
    from .contract import validate_field

    revisions = args.get("revisions", [])
    if not revisions:
        return {"ok": False, "error": "revisions array is required"}

    # Inherit encoding_source from dispatch wrapper
    top_encoding_source = args.get("encoding_source")

    # Validate each revision
    for i, spec in enumerate(revisions):
        if not spec.get("node_id"):
            return {"ok": False, "error": "revisions[%d]: node_id required" % i}
        if not spec.get("reason"):
            return {"ok": False, "error": "revisions[%d]: reason required" % i}
        for field, value in spec.items():
            if field not in ("node_id", "reason"):
                ok, err = validate_field(field, value)
                if not ok:
                    return {"ok": False, "error": "revisions[%d].%s: %s" % (i, field, err)}

    # Resolve short IDs
    resolved = []
    for spec in revisions:
        r = dict(spec)
        r['node_id'] = _resolve_id(brain, r['node_id'])
        if top_encoding_source and 'encoding_source' not in r:
            r['encoding_source'] = top_encoding_source
        resolved.append(r)

    result = brain.revise_batch(resolved)
    graph_changes.append("REVISE_BATCH: %d revised" % result.get("revised", 0))
    return {"ok": True, "result": result}


def _handle_brain_batch(brain, args, graph_changes):
    """Execute multiple brain operations in one call.

    Accepts mixed operations: remember, revise, connect in any order.
    Each operation is validated and executed sequentially.
    Returns results for each operation.

    Args:
        operations: list of {op: "remember"|"revise"|"connect", ...fields}
    """
    operations = args.get("operations", [])
    if not operations:
        return {"ok": False, "error": "operations array is required"}

    top_encoding_source = args.get("encoding_source")
    results = []

    for i, op_spec in enumerate(operations):
        op = op_spec.get("op", "")
        try:
            if op == "remember":
                # Route through existing remember handler
                op_args = {k: v for k, v in op_spec.items() if k != "op"}
                if top_encoding_source and "encoding_source" not in op_args:
                    op_args["encoding_source"] = top_encoding_source
                r = _handle_remember(brain, op_args, graph_changes)
                results.append({"op": "remember", "index": i, **r})

            elif op == "revise":
                op_args = {k: v for k, v in op_spec.items() if k != "op"}
                if top_encoding_source and "encoding_source" not in op_args:
                    op_args["encoding_source"] = top_encoding_source
                r = _handle_revise(brain, op_args, graph_changes)
                results.append({"op": "revise", "index": i, **r})

            elif op == "connect":
                op_args = {k: v for k, v in op_spec.items() if k != "op"}
                r = _handle_connect(brain, op_args, graph_changes)
                results.append({"op": "connect", "index": i, **r})

            else:
                results.append({"op": op, "index": i, "ok": False,
                                "error": "Unknown op: %s (use remember, revise, connect)" % op})
        except Exception as e:
            results.append({"op": op, "index": i, "ok": False, "error": str(e)[:200]})

    succeeded = sum(1 for r in results if r.get("ok"))
    return {"ok": True, "result": {
        "total": len(operations),
        "succeeded": succeeded,
        "failed": len(operations) - succeeded,
        "results": results,
    }}


def _handle_find_node_by_title(brain, args, graph_changes):
    result = brain.find_node_by_title(
        title_query=args.get("title_query", ""),
        threshold=args.get("threshold", 0.75),
        top_k=args.get("top_k", 1))
    return {"ok": True, "result": result}


def _handle_trace_append(brain, args, graph_changes):
    """Append a trace event from any source. Validates against trace_contract."""
    try:
        event_id = brain._trace_dal.append(
            chain_id=args.get("chain_id", ""),
            scale=args.get("scale", "s0"),
            event_type=args.get("event_type", ""),
            ref_type=args.get("ref_type", ""),
            ref_id=args.get("ref_id", ""),
            summary=args.get("summary", ""),
            metadata=args.get("metadata"),
            session_id=args.get("session_id", ""))
        return {"ok": True, "result": {"event_id": event_id}}
    except ValueError as e:
        return {"ok": False, "error": str(e)}


def _handle_query_logs(brain, args, graph_changes):
    """Query brain logs: errors, debug events, and signals."""
    result = brain.query_logs(
        source=args.get("source", "all"),
        hours=args.get("hours", 24),
        level=args.get("level", "all"),
        hook_name=args.get("hook_name", ""),
        limit=args.get("limit", 50))
    return {"ok": True, "result": result}


def _handle_query_traces(brain, args, graph_changes):
    """Query trace events: O/K/Δ/outcome at every scale."""
    result = brain.query_traces(
        scale=args.get("scale", ""),
        hours=args.get("hours", 24),
        event_type=args.get("event_type", ""),
        chain_id=args.get("chain_id", ""),
        session_id=args.get("session_id", ""),
        ref_type=args.get("ref_type", ""),
        grouped=args.get("grouped", False),
        limit=args.get("limit", 100))
    return {"ok": True, "result": result}


def _handle_query_outcomes(brain, args, graph_changes):
    """Query outcome events — the learning signal."""
    result = brain.query_outcomes(
        chain_id=args.get("chain_id", ""),
        scale=args.get("scale", ""),
        hours=args.get("hours", 168))
    return {"ok": True, "result": result}


def _handle_count_traces(brain, args, graph_changes):
    """Count trace events grouped by a field."""
    result = brain.count_traces(
        field=args.get("field", "event_type"),
        scale=args.get("scale", ""),
        hours=args.get("hours", 24))
    return {"ok": True, "result": result}


def _handle_list_interactions(brain, args, graph_changes):
    """List all registered interactions."""
    return {"ok": True, "result": brain.list_interactions()}


def _handle_get_interaction(brain, args, graph_changes):
    """Get a specific interaction by name. version=0 returns latest."""
    result = brain.get_interaction(
        name=args.get("name", ""),
        version=args.get("version", 0))
    if not result:
        return {"ok": False, "error": "Interaction not found: %s" % args.get("name", "")}
    return {"ok": True, "result": result}


def _handle_filter_nodes(brain, args, graph_changes):
    """Structured query: filter nodes by any structural field."""
    result = brain.filter_nodes(
        field=args.get("field", ""),
        include=args.get("include"),
        exclude=args.get("exclude"),
        lt=args.get("lt"),
        gt=args.get("gt"),
        limit=args.get("limit", 50),
        sort_by=args.get("sort_by", "created_at"),
        sort_order=args.get("sort_order", "desc"))
    if "error" in result:
        return {"ok": False, "error": result["error"]}
    return {"ok": True, "result": result}


def _handle_graph_expand(brain, args, graph_changes):
    """Layer 3: expand from surface-selected seed nodes via structural edges.

    Args:
        node_ids: list of seed node IDs (from surface selection)
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
    from .dal import NodeDAL
    dal = NodeDAL(brain.conn)
    seen = set(node_ids)
    neighbors = []

    for seed_id in node_ids:
        # Resolve short IDs
        full_id = seed_id
        if len(seed_id) < 16:
            resolved = dal.resolve_id(seed_id)
            if resolved:
                full_id = resolved

        # Get structural neighbors (single-direction storage, query both directions)
        rows = brain.conn.execute("""
            SELECT n.id, n.type, n.title, substr(n.content, 1, 300),
                   er.relation, e.weight, er.description, n.confidence,
                   CASE WHEN e.source_id = ? THEN 'outgoing' ELSE 'incoming' END as direction
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n ON n.id = CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
            WHERE (e.source_id = ? OR e.target_id = ?) AND n.archived = 0
            AND n.id != ?
            AND er.relation NOT IN ({})
            ORDER BY e.weight DESC LIMIT ?
        """.format(','.join('?' for _ in EXCLUDED_EDGE_TYPES)),
            [full_id, full_id, full_id, full_id, full_id] + list(EXCLUDED_EDGE_TYPES) + [limit_per]).fetchall()

        for r in rows:
            if r[0] not in seen:
                seen.add(r[0])
                neighbors.append({
                    "id": r[0], "type": r[1], "title": r[2],
                    "content": r[3], "edge_type": r[4],
                    "edge_weight": r[5], "edge_description": r[6] or "",
                    "confidence": r[7], "direction": r[8],
                    "seed_id": full_id,
                })

    return {"ok": True, "result": {"neighbors": neighbors, "seeds": len(node_ids)}}


def _handle_get_node(brain, args, graph_changes):
    node_id = _resolve_id(brain, args.get("node_id", ""))
    if not node_id:
        return {"ok": False, "error": "node_id is required"}

    from .pipeline_contract import get_rich_node
    node = get_rich_node(brain, node_id)
    if not node:
        return {"ok": False, "error": "Node not found: {}".format(node_id)}

    return {"ok": True, "result": node}


def _handle_recall_batch(brain, args, graph_changes):
    """Batch recall — multiple queries in one call."""
    queries = args.get("queries", [])
    limit = args.get("limit", 5)
    batch_filter = args.get("filter")
    results = []
    for q in queries[:10]:  # cap at 10 queries
        try:
            result = brain.recall(query=q, filter=batch_filter, limit=limit, source='mcp')
            results.append({"query": q, "results": result.get("results", [])})
        except Exception as e:
            results.append({"query": q, "results": [], "error": str(e)})
    return {"ok": True, "result": results}


def _handle_get_nodes(brain, args, graph_changes):
    """Batch get_node — multiple node IDs in one call."""
    node_ids = args.get("node_ids", [])
    results = []
    for nid in node_ids[:20]:  # cap at 20
        resolved = _resolve_id(brain, nid)
        if not resolved:
            results.append({"id": nid, "error": "not found"})
            continue
        node = brain.get_node(resolved)
        if not node:
            results.append({"id": nid, "error": "not found"})
            continue
        try:
            edges = brain.conn.execute(
                "SELECT e.target_id, e.relation, e.weight, n.title, n.type "
                "FROM edges e LEFT JOIN nodes n ON n.id = e.target_id "
                "WHERE e.source_id = ? ORDER BY e.weight DESC LIMIT 10",
                (resolved,)).fetchall()
            node["connections"] = [
                {"target_id": e[0], "relation": e[1], "weight": e[2],
                 "title": e[3] or "", "type": e[4] or ""}
                for e in edges]
        except Exception:
            node["connections"] = []
        results.append(node)
    return {"ok": True, "result": results}


def _handle_encode_cluster(brain, args, graph_changes):
    result = brain.encode_cluster(
        nodes=args.get("nodes", []),
        connect_to=args.get("connect_to"),
        auto_connect=args.get("auto_connect", True))
    n = result.get("nodes_created", 0)
    graph_changes.append("ENCODE_CLUSTER: %d nodes" % n)
    return {"ok": True, "result": result}


def _handle_connect(brain, args, graph_changes):
    brain.connect_typed(
        source_id=_resolve_id(brain, args.get("source_id", "")),
        target_id=_resolve_id(brain, args.get("target_id", "")),
        relation=args.get("relation", "related_to"),
        weight=args.get("weight", 0.5),
        description=args.get("description", ""))
    graph_changes.append(
        "CONNECT: %s -[%s]-> %s" % (
            args.get("source_id", "?")[:8],
            args.get("relation", "related_to"),
            args.get("target_id", "?")[:8]))
    return {"ok": True, "result": {"connected": True}}


def _handle_connect_batch(brain, args, graph_changes):
    """Create multiple edges in one call."""
    connections = args.get("connections", [])
    if not connections:
        return {"ok": False, "error": "connections array is required"}

    created = 0
    for c in connections:
        try:
            brain.connect_typed(
                source_id=_resolve_id(brain, c.get("source_id", "")),
                target_id=_resolve_id(brain, c.get("target_id", "")),
                relation=c.get("relation", "related_to"),
                weight=c.get("weight", 0.5),
                description=c.get("description", ""))
            created += 1
        except Exception:
            pass
    graph_changes.append("CONNECT_BATCH: %d edges" % created)
    return {"ok": True, "result": {"edges_created": created}}


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
    "revise_batch":          CmdEntry(_handle_revise_batch,         is_write=True, marks_dirty=True),
    # record_divergence, learn_vocabulary REMOVED 2026-04-06 — use remember(type='correction'/'vocabulary')
    "find_node_by_title":    CmdEntry(_handle_find_node_by_title,  is_write=False, marks_dirty=False),
    "filter_nodes":          CmdEntry(_handle_filter_nodes,        is_write=False, marks_dirty=False),
    "query_logs":            CmdEntry(_handle_query_logs,          is_write=False, marks_dirty=False),
    "query_traces":          CmdEntry(_handle_query_traces,        is_write=False, marks_dirty=False),
    "query_outcomes":        CmdEntry(_handle_query_outcomes,      is_write=False, marks_dirty=False),
    "count_traces":          CmdEntry(_handle_count_traces,        is_write=False, marks_dirty=False),
    "list_interactions":     CmdEntry(_handle_list_interactions,   is_write=False, marks_dirty=False),
    "get_interaction":       CmdEntry(_handle_get_interaction,     is_write=False, marks_dirty=False),
    "trace_append":          CmdEntry(_handle_trace_append,        is_write=True,  marks_dirty=False),
    "get_node":              CmdEntry(_handle_get_node,             is_write=False, marks_dirty=False),
    "get_nodes":             CmdEntry(_handle_get_nodes,            is_write=False, marks_dirty=False),
    "recall_batch":          CmdEntry(_handle_recall_batch,         is_write=False, marks_dirty=False),
    "graph_expand":          CmdEntry(_handle_graph_expand,         is_write=False, marks_dirty=False),
    # encode_cluster: DEPRECATED — use remember_batch() instead. Handler kept for backward compat.
    # "encode_cluster":        CmdEntry(_handle_encode_cluster,      is_write=True, marks_dirty=True),
    "connect":               CmdEntry(_handle_connect,             is_write=True, marks_dirty=True),
    "connect_batch":         CmdEntry(_handle_connect_batch,       is_write=True, marks_dirty=True),
    "brain_batch":           CmdEntry(_handle_brain_batch,         is_write=True, marks_dirty=True),
    "enrich":                CmdEntry(_handle_enrich,              is_write=True, marks_dirty=True),
    "eval":                  CmdEntry(_handle_eval,                is_write=True, marks_dirty=True),
}

# "shutdown" is handled directly by daemon_server (needs to set self.running)
# "hook_*" commands are dispatched via HOOK_TABLE in daemon_server
