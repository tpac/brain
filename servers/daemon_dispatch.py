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
    # Optional contract: the set of top-level keys this handler knows how to
    # consume. When set, the dispatcher logs an error for any arg key not in
    # this set — surfaces silent drops (e.g. encoding_source being passed but
    # not forwarded to the brain method). None = no check, opt-in per entry.
    accepts: Optional[frozenset] = None


def check_unknown_keys(cmd: str, entry: 'CmdEntry', args: Dict[str, Any], brain) -> None:
    """Log an error if args contains keys the handler doesn't list as accepted.

    Defensive: any `accepts` contract must allow all keys the handler reads —
    if `accepts` is declared but the handler reads a field not in the set,
    this function will (incorrectly) flag legitimate inputs as unknown. Update
    `accepts` when you change what a handler reads.
    """
    if entry.accepts is None or not args:
        return
    unknown = set(args.keys()) - entry.accepts
    if not unknown:
        return
    try:
        brain._log_error(
            'dispatch_unknown_keys',
            ValueError('cmd=%s dropped keys=%s' % (cmd, sorted(unknown))),
            'accepted=%s' % sorted(entry.accepts))
    except Exception:
        pass


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
        db_dir=args.get("db_dir", ""),
        session_id=args.get("session_id", ""))
    return {"ok": True, "result": text}


def _enrich_recall_results(brain, result, graph_changes):
    """Enrich recall results via brain.get_node() — the shared data atom.

    Anchor's MCP recall gets full enrichment per node:
    metadata, corrections, connections, situation.
    """
    results = result.get("results", [])
    if not results:
        return

    # Enrich each result with brain.get_node() data
    for r in results[:8]:
        rich = brain.get_node(r.get("id", ""))
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
    data = brain.pre_edit(file=file, tool_name=tool_name)
    try:
        data["change_impacts"] = brain.get_change_impact(file)
    except Exception as e:
        brain._log_error("pre_edit_change_impact", e, "fetching change impacts for %s" % file[:60])
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


def _handle_remember(brain, args, graph_changes):
    from .contract import validate_field, get_remember_fields

    # Validate all provided fields against contract
    for field, value in args.items():
        ok, err = validate_field(field, value)
        if not ok:
            return {"ok": False, "error": err}

    # Pass ALL fields through to remember() — contract fields go to nodes table,
    # promoted fields to metadata, everything else to node_metadata_kv as extra_fields.
    # Don't filter — remember() handles routing via **extra_fields kwargs.
    remember_args = {k: v for k, v in args.items() if v is not None}
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
        # Pass ALL fields through — contract fields go to nodes table,
        # promoted fields to metadata, extras to node_metadata_kv
        cleaned = {k: v for k, v in spec.items() if v is not None}
        if top_encoding_source and 'encoding_source' not in cleaned:
            cleaned['encoding_source'] = top_encoding_source
        cleaned_nodes.append(cleaned)

    result = brain.remember_batch(
        nodes=cleaned_nodes,
        connect_to=args.get("connect_to"),
        auto_connect=args.get("auto_connect", True))
    graph_changes.append("REMEMBER_BATCH: %d nodes" % result.get("nodes_created", 0))
    return {"ok": True, "result": result}


def _emit_revise_trace(brain, node_id, reason, encoding_source, deltas,
                       warnings=None, chain_id_override='', session_id=''):
    """Emit a node_revised trace event for a revise() call.

    Trace replaces the legacy _sys_revision_history KV blob as the canonical
    revision history substrate. Emitted when EITHER deltas or warnings is
    non-empty — so audit history captures both successful changes AND
    attempted-but-rejected operations (immutable field passed, archive
    blocked on locked node).

    Scale inference from encoding_source:
      - 's2:*' → 's2' (S2 maintenance units)
      - 'encoder:*' → 's1' (S1 encoder)
      - 'hook:*' → 's0' (lifecycle hooks)
      - else → 's0' (direct MCP, anchor, default)

    chain_id strategy (per Stage 1A spec):
      - Caller-provided `chain_id_override` wins (encoder cycles pass theirs;
        revises join the encoder's chain for grouped querying).
      - Otherwise fall back to a date-based per-scale chain
        (`{scale}-{YYYYMMDD}-revise`) so direct/operator revises group by day.
    """
    warnings = warnings or []
    if not deltas and not warnings:
        return  # nothing to record

    from .trace_contract import build_revise_metadata

    # Infer scale from encoding_source
    if encoding_source.startswith('s2:'):
        scale = 's2'
    elif encoding_source.startswith('encoder:'):
        scale = 's1'
    else:
        scale = 's0'

    if chain_id_override:
        chain_id = chain_id_override
    else:
        from datetime import datetime
        chain_id = '%s-%s-revise' % (
            scale, datetime.utcnow().strftime('%Y%m%d'))

    metadata = build_revise_metadata(
        node_id=node_id, reason=reason,
        encoding_source=encoding_source,
        deltas=deltas, warnings=warnings)

    parts = []
    if deltas:
        parts.append('revised %d field(s): %s' % (
            len(deltas), ', '.join(d['field'] for d in deltas)))
    if warnings:
        parts.append('%d warning(s)' % len(warnings))
    summary = '; '.join(parts) if parts else 'revise no-op'

    try:
        brain._trace_dal.append(
            chain_id=chain_id,
            scale=scale,
            event_type='delta',
            ref_type='node_revised',
            ref_id=node_id,
            summary=summary,
            metadata=metadata,
            session_id=session_id,
        )
    except Exception as e:
        brain._log_error('revise_trace_emit', e,
                         'failed to emit trace for revise of %s' % node_id[:8])


def _emit_edge_revise_trace(brain, edge_id, relation, reason, encoding_source,
                            deltas, warnings=None,
                            chain_id_override='', session_id=''):
    """Emit an edge_relation_revised trace event.

    Mirrors _emit_revise_trace for nodes. Same emit-on-deltas-or-warnings
    behavior, same scale inference, same chain_id strategy. ref_id encodes
    the (edge_id, relation) tuple as f"{edge_id}:{relation}".

    Used by the connect upsert path (deltas show create-via-INSERT or
    field-preserving update) and the disconnect path (deltas show the
    archived flag flip).
    """
    warnings = warnings or []
    if not deltas and not warnings:
        return  # nothing to record

    from .trace_contract import build_edge_revise_metadata

    if encoding_source.startswith('s2:'):
        scale = 's2'
    elif encoding_source.startswith('encoder:'):
        scale = 's1'
    else:
        scale = 's0'

    if chain_id_override:
        chain_id = chain_id_override
    else:
        from datetime import datetime
        chain_id = '%s-%s-revise' % (
            scale, datetime.utcnow().strftime('%Y%m%d'))

    metadata = build_edge_revise_metadata(
        edge_id=edge_id, relation=relation, reason=reason,
        encoding_source=encoding_source,
        deltas=deltas, warnings=warnings)

    parts = []
    if deltas:
        parts.append('%d field(s): %s' % (
            len(deltas), ', '.join(d['field'] for d in deltas)))
    if warnings:
        parts.append('%d warning(s)' % len(warnings))
    summary = '; '.join(parts) if parts else 'edge revise no-op'

    try:
        brain._trace_dal.append(
            chain_id=chain_id,
            scale=scale,
            event_type='delta',
            ref_type='edge_relation_revised',
            ref_id='%s:%s' % (edge_id, relation),
            summary=summary,
            metadata=metadata,
            session_id=session_id,
        )
    except Exception as e:
        brain._log_error('edge_revise_trace_emit', e,
                         'failed to emit trace for edge %s:%s' % (
                             edge_id[:12], relation))


def _handle_revise(brain, args, graph_changes):
    """Update any field(s) on an existing node via revise()."""
    from .contract import validate_field, ALL_FIELDS

    node_id = _resolve_id(brain, args.get("node_id", ""))
    reason = args.get("reason", "")
    if not node_id:
        return {"ok": False, "error": "node_id is required"}
    if not reason:
        return {"ok": False, "error": "reason is required"}

    # Reserve known dispatch keys so they don't get treated as field updates.
    DISPATCH_KEYS = {"node_id", "reason", "encoding_source", "chain_id", "session_id"}
    updates = {k: v for k, v in args.items() if k not in DISPATCH_KEYS}
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

    # Emit node_revised trace event (replaces _sys_revision_history substrate).
    # Includes warnings so audit history captures attempted-but-rejected ops.
    _emit_revise_trace(
        brain, node_id, reason,
        args.get('encoding_source', ''),
        result.get('deltas', []),
        warnings=result.get('warnings', []),
        chain_id_override=args.get('chain_id', ''),
        session_id=args.get('session_id', ''),
    )

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

    # Emit one node_revised trace event per revised row.
    # Includes warnings so audit history captures attempted-but-rejected ops.
    chain_id_override = args.get('chain_id', '')
    session_id = args.get('session_id', '')
    for row, spec in zip(result.get('results', []), resolved):
        if row.get('status') == 'revised':
            _emit_revise_trace(
                brain, row['node_id'], spec.get('reason', ''),
                spec.get('encoding_source', '') or top_encoding_source or '',
                row.get('deltas', []),
                warnings=row.get('warnings', []),
                chain_id_override=chain_id_override,
                session_id=session_id,
            )

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

    # Valid nested op names. Sonnet has been observed inventing structural
    # op names like `consolidate` / `keep` / `evolve` / `skip` — these are
    # prompt-level decisions, not dispatch ops. When invented names land
    # here they silently go to the "Unknown op" branch below; surface them
    # loudly so drift gets noticed immediately. (The historical 25 silent
    # drops across consolidation runs is exactly this pattern.)
    VALID_OPS = {"remember", "revise", "connect", "disconnect", "archive"}

    top_encoding_source = args.get("encoding_source")
    results = []

    # Sibling-aware connect_to: defer per-op connect_to from `remember` ops
    # until ALL ops in this batch have run, so siblings declared in any order
    # can resolve. NEW wins on title collision (sibling beats catalog).
    sibling_map = {}  # lowercased title → new node_id
    deferred_connects = []  # [(src_node_id, connect_to_spec)]

    for i, op_spec in enumerate(operations):
        if not isinstance(op_spec, dict):
            results.append({"op": "?", "index": i, "ok": False,
                            "error": "operation must be a dict, got %s" % type(op_spec).__name__})
            continue
        op = op_spec.get("op", "")
        try:
            if op == "remember":
                # Pop per-op connect_to BEFORE handler so it's not processed
                # eagerly with an empty sibling_map — defer to after the loop.
                ct_spec = op_spec.get("connect_to")
                op_args = {k: v for k, v in op_spec.items()
                           if k not in ("op", "connect_to")}
                if top_encoding_source and "encoding_source" not in op_args:
                    op_args["encoding_source"] = top_encoding_source
                # Same fix as remember_batch: disable inner remember()'s
                # conversation-context auto_connect inside batches so it
                # doesn't create reverse-direction co_accessed edges between
                # siblings before deferred connect_to runs.
                op_args.setdefault("auto_connect", False)
                r = _handle_remember(brain, op_args, graph_changes)
                results.append({"op": "remember", "index": i, **r})
                # Capture for sibling_map + deferred resolution
                if r.get("ok"):
                    inner = r.get("result") or {}
                    new_id = inner.get("id")
                    if new_id:
                        title = (op_args.get("title") or "").lower()
                        if title:
                            sibling_map[title] = new_id
                        if ct_spec:
                            deferred_connects.append((new_id, ct_spec))

            elif op == "revise":
                op_args = {k: v for k, v in op_spec.items() if k != "op"}
                if top_encoding_source and "encoding_source" not in op_args:
                    op_args["encoding_source"] = top_encoding_source
                r = _handle_revise(brain, op_args, graph_changes)
                results.append({"op": "revise", "index": i, **r})

            elif op == "connect":
                op_args = {k: v for k, v in op_spec.items() if k != "op"}
                if top_encoding_source and "encoding_source" not in op_args:
                    op_args["encoding_source"] = top_encoding_source
                r = _handle_connect(brain, op_args, graph_changes)
                results.append({"op": "connect", "index": i, **r})

            elif op == "archive":
                node_id = op_spec.get("node_id")
                if not node_id:
                    results.append({"op": "archive", "index": i, "ok": False,
                                    "error": "node_id is required"})
                else:
                    # Unified archive path — handles guards, edges, vectors, audit.
                    # Fallback chain mirrors disconnect: op-level encoding_source
                    # → op-level archived_by → top-level encoding_source →
                    # 'unknown'. Lets top-level brain_batch tagging cascade to
                    # archive audit without per-op injection.
                    archived_by = op_spec.get('encoding_source') or \
                        op_spec.get('archived_by') or \
                        top_encoding_source or 'unknown'
                    reason = op_spec.get('reason', '')
                    r = brain.archive_node(
                        node_id, archived_by=archived_by, reason=reason)
                    if r.get('ok'):
                        graph_changes.append("ARCHIVE: %s" % node_id[:8])
                    results.append({"op": "archive", "index": i, **r})

            elif op == "disconnect":
                # Soft-archive a specific relation on an edge. Other relations
                # on the same edge survive. v25 — archived row preserved for
                # forensics/recovery; reads filter via JOIN.
                # Lets ABSORB encoders prune survivor edges that don't fit
                # the new framing after revise.
                from .dal import GraphDAL
                source_id = op_spec.get("source_id")
                target_id = op_spec.get("target_id")
                relation = op_spec.get("relation")
                archived_by = op_spec.get('encoding_source') or \
                    op_spec.get('archived_by') or \
                    top_encoding_source or 'unknown'
                if not (source_id and target_id and relation):
                    results.append({"op": "disconnect", "index": i, "ok": False,
                                    "error": "source_id, target_id, relation are required"})
                else:
                    gdal = GraphDAL(brain.conn)
                    edge_id = gdal.get_edge_id(source_id, target_id)
                    gdal.remove_relation(
                        source_id, target_id, relation, archived_by=archived_by)
                    brain.conn.commit()
                    graph_changes.append("DISCONNECT: %s -[%s]-> %s" % (
                        source_id[:8], relation, target_id[:8]))

                    # Emit edge_relation_revised trace event capturing the
                    # archived flag flip. Mirrors connect upsert trace shape.
                    if edge_id:
                        _emit_edge_revise_trace(
                            brain, edge_id, relation,
                            op_spec.get('reason', '') or args.get('reason', ''),
                            archived_by,
                            deltas=[{'field': 'archived',
                                     'old': 0, 'new': 1}],
                            chain_id_override=args.get('chain_id', ''),
                            session_id=args.get('session_id', ''),
                        )

                    results.append({"op": "disconnect", "index": i, "ok": True})

            else:
                # Invalid op name — log loudly. Sonnet sometimes invents
                # structural names (consolidate/keep/evolve/skip) that were
                # never valid ops. Previously this returned ok=False and the
                # caller moved on silently.
                err_msg = ("Unknown op: %s (use remember, revise, connect, disconnect, archive)"
                           % op)
                try:
                    brain._log_error(
                        'brain_batch_invalid_op',
                        ValueError(err_msg),
                        'op_spec=%s' % str(op_spec)[:300])
                except Exception:
                    pass
                results.append({"op": op, "index": i, "ok": False, "error": err_msg})
        except Exception as e:
            results.append({"op": op, "index": i, "ok": False, "error": str(e)[:200]})

    # Pass 2: deferred per-op connect_to resolution. Runs AFTER all ops so
    # siblings declared in any order resolve correctly. _apply_connect_to
    # logs all failures to debug_log; this is sequencing-agnostic and never
    # raises. Returns (edges_created, failures_logged) — both surfaced in
    # the batch result so a cycle with N requested connect_to and 0 edges
    # has a visible "connect_to_failures=N" reason.
    connect_to_edges = 0
    connect_to_failures = 0
    for src_id, ct_spec in deferred_connects:
        edges, fails = brain._apply_connect_to(
            src_id, ct_spec, sibling_map=sibling_map)
        connect_to_edges += edges
        connect_to_failures += fails
    if connect_to_edges:
        graph_changes.append("CONNECT_TO: %d edges" % connect_to_edges)
    if connect_to_failures:
        graph_changes.append("CONNECT_TO_FAILURES: %d" % connect_to_failures)

    succeeded = sum(1 for r in results if r.get("ok"))
    return {"ok": True, "result": {
        "total": len(operations),
        "succeeded": succeeded,
        "failed": len(operations) - succeeded,
        "connect_to_edges": connect_to_edges,
        "connect_to_failures": connect_to_failures,
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


def _handle_clear_errors(brain, args, graph_changes):
    """Clear hook errors and optionally debug log entries."""
    cleared = {}
    hours = args.get("hours")  # None = clear all

    # Hook errors
    if hours:
        c = brain.logs_conn.execute(
            "DELETE FROM hook_errors WHERE created_at < datetime('now', '-%d hours')" % int(hours))
    else:
        c = brain.logs_conn.execute("DELETE FROM hook_errors")
    cleared['hook_errors'] = c.rowcount

    # Debug log (if requested)
    if args.get("debug_log", False):
        if hours:
            c = brain.logs_conn.execute(
                "DELETE FROM debug_log WHERE created_at < datetime('now', '-%d hours')" % int(hours))
        else:
            c = brain.logs_conn.execute("DELETE FROM debug_log")
        cleared['debug_log'] = c.rowcount

    brain.logs_conn.commit()
    return {"ok": True, "result": cleared}


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


def _handle_set_interaction_active(brain, args, graph_changes):
    """Flip the active version pointer for an interaction.

    After this call, the runtime path (brain.get_interaction_prompt /
    get_interaction_config) reads the chosen version. The version must
    already be registered. Use after register_interaction to make a
    newly-registered version live, OR to roll back to a previous version.
    """
    name = args.get("name", "")
    version = args.get("version")
    set_by = args.get("set_by", "anchor")
    if not name or version is None:
        return {"ok": False, "error": "name and version are required"}
    try:
        version = int(version)
    except (TypeError, ValueError):
        return {"ok": False, "error": "version must be an integer"}
    try:
        result = brain._interaction_dal.set_active(name, version, set_by)
        return {"ok": True, "result": result}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def _handle_register_interaction(brain, args, graph_changes):
    """Register a new version of an interaction (prompt + config).

    Creates a new version if the interaction exists, or version 1 if new.
    Used by S2/S3 to evolve learnable boundaries.

    NOTE: Registration does NOT activate the new version (since 2026-05-10).
    Call set_interaction_active to flip the runtime pointer. Exception:
    version 1 (first registration of a name) auto-activates — otherwise
    nothing would be readable for that name.
    """
    name = args.get("name", "")
    template = args.get("template", "")
    parameters = args.get("parameters", "{}")
    created_by = args.get("created_by", "")

    if not name:
        return {"ok": False, "error": "name is required"}

    try:
        result = brain._interaction_dal.register(
            name=name,
            template=template,
            parameters=parameters,
            created_by=created_by)
        # Registration does NOT activate. Return the newly-registered version
        # plus the currently-active version so the caller knows whether it
        # took effect at runtime.
        active = brain._interaction_dal.get_active(name)
        return {"ok": True, "result": {
            "name": name,
            "registered_version": result.get("version"),
            "active_version": active.get("version") if active else None,
            "template_length": len(template),
            "note": (
                "Registration created a new version but did NOT activate it. "
                "Call set_interaction_active to flip the runtime pointer."
                if active and active.get("version") != result.get("version")
                else "First version of this interaction; auto-activated."
            ),
        }}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


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
        sort_order=args.get("sort_order", "desc"),
        rich=args.get("rich", True))
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
    from .dal import NodeDAL, GraphDAL
    node_dal = NodeDAL(brain.conn)
    graph_dal = GraphDAL(brain.conn)
    seen = set(node_ids)
    neighbors = []
    excluded = set(EXCLUDED_EDGE_TYPES)

    for seed_id in node_ids:
        full_id = seed_id
        if len(seed_id) < 16:
            resolved = node_dal.resolve_id(seed_id)
            if resolved:
                full_id = resolved

        rows = graph_dal.get_neighbors(
            full_id,
            limit=limit_per,
            exclude_relations=excluded,
            exclude_node_ids=seen,
            content_preview_chars=300,
        )
        for r in rows:
            if r['id'] not in seen:
                seen.add(r['id'])
                neighbors.append({
                    "id": r['id'], "type": r['type'], "title": r['title'],
                    "content": r.get('content_preview', ''),
                    "edge_type": r['relation'],
                    "edge_weight": r['weight'],
                    "edge_description": r.get('edge_description') or '',
                    "confidence": r['confidence'],
                    "direction": r['direction'],
                    "seed_id": full_id,
                })

    return {"ok": True, "result": {"neighbors": neighbors, "seeds": len(node_ids)}}


def _handle_get_node(brain, args, graph_changes):
    node_id = _resolve_id(brain, args.get("node_id", ""))
    if not node_id:
        return {"ok": False, "error": "node_id is required"}

    node = brain.get_node(node_id)
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
    """Batch get_node — multiple node IDs in one call, same rich shape as get_node."""
    node_ids = args.get("node_ids", [])
    resolved_ids = []
    errors = []
    for nid in node_ids[:20]:  # cap at 20
        resolved = _resolve_id(brain, nid)
        if resolved:
            resolved_ids.append(resolved)
        else:
            errors.append({"id": nid, "error": "not found"})

    rich_map = brain.get_node(resolved_ids) if resolved_ids else {}
    # Batch returns {node_id: rich_dict}. Preserve request order.
    results = [rich_map[nid] for nid in resolved_ids if nid in rich_map]
    results.extend(errors)
    return {"ok": True, "result": results}


def _handle_connect(brain, args, graph_changes):
    # Stage 1B: pass description/encoding_source through only when caller
    # specified them. None preserves existing on update (idempotent upsert).
    relation = args.get("relation", "related_to")
    encoding_source = args.get("encoding_source", "")
    result = brain.connect_typed(
        source_id=_resolve_id(brain, args.get("source_id", "")),
        target_id=_resolve_id(brain, args.get("target_id", "")),
        relation=relation,
        weight=args.get("weight", 0.5),
        description=args.get("description"),
        encoding_source=args.get("encoding_source"))

    # Emit edge_relation_revised trace event capturing create-or-update deltas.
    if result and (result.get('deltas') or result.get('warnings')):
        _emit_edge_revise_trace(
            brain, result['edge_id'], relation,
            args.get('reason', ''),
            encoding_source,
            result.get('deltas', []),
            warnings=result.get('warnings', []),
            chain_id_override=args.get('chain_id', ''),
            session_id=args.get('session_id', ''),
        )

    graph_changes.append(
        "CONNECT: %s -[%s]-> %s" % (
            args.get("source_id", "?")[:8],
            relation,
            args.get("target_id", "?")[:8]))
    return {"ok": True, "result": {"connected": True}}


def _handle_connect_batch(brain, args, graph_changes):
    """Create multiple edges in one call."""
    connections = args.get("connections", [])
    if not connections:
        return {"ok": False, "error": "connections array is required"}

    chain_id_override = args.get('chain_id', '')
    session_id = args.get('session_id', '')
    top_encoding_source = args.get('encoding_source', '')

    created = 0
    for c in connections:
        try:
            # Stage 1B: pass description through only when specified
            # (None → preserve existing on update).
            relation = c.get("relation", "related_to")
            encoding_source = c.get("encoding_source", "") or top_encoding_source
            result = brain.connect_typed(
                source_id=_resolve_id(brain, c.get("source_id", "")),
                target_id=_resolve_id(brain, c.get("target_id", "")),
                relation=relation,
                weight=c.get("weight", 0.5),
                description=c.get("description"))
            created += 1

            # Emit one edge_relation_revised trace per row that actually changed.
            if result and (result.get('deltas') or result.get('warnings')):
                _emit_edge_revise_trace(
                    brain, result['edge_id'], relation,
                    c.get('reason', '') or args.get('reason', ''),
                    encoding_source,
                    result.get('deltas', []),
                    warnings=result.get('warnings', []),
                    chain_id_override=chain_id_override,
                    session_id=session_id,
                )
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
        cache_stats = brain._vec_dal.cache_stats() if hasattr(brain, '_vec_dal') else {}
    except Exception as e:
        cache_stats = {'error': str(e)}

    # Recent errors — hours defaults to 2.
    hours = int(args.get('hours', 2))
    try:
        recent_errors = brain._logs_dal.get_recent_errors(hours=hours) if hasattr(brain, '_logs_dal') else []
    except Exception:
        recent_errors = []

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


def _handle_migrate_to_aspects(brain, args, graph_changes):
    """One-shot migration: seed required aspects + import emergent from legacy.

    Idempotent — safe to re-run. Operator triggers via scripts/migrate_to_aspects.py
    after backing up brain.db. The handler runs in-daemon (single writer
    connection) so it doesn't violate the 'never spawn second Brain()' rule.

    Returns the migration summary.
    """
    from .aspect_migration import migrate_to_aspects
    result = migrate_to_aspects(brain)
    graph_changes.append("MIGRATE_ASPECTS: required=%d created, emergent=%d created, total=%d" % (
        len(result['required']['created']),
        len(result['emergent']['created']),
        result['aspect_node_count']))
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


# ═══════════════════════════════════════════════════════════════
# COMMAND TABLE — the single source of truth for routing
# ═══════════════════════════════════════════════════════════════

COMMAND_TABLE: Dict[str, CmdEntry] = {
    # Removed 2026-04-13: engineering_context, correction_patterns, last_synthesis,
    # dreams, staged, suggest_metrics, get_active_evolutions, assess_developmental_stage,
    # instinct_check, prompt_reflection, self_reflection, consolidate, dream,
    # auto_heal, auto_tune, synthesize_session.
    # Removed 2026-04-06: record_divergence, learn_vocabulary.

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
    "scan_host":                CmdEntry(_handle_scan_host,            is_write=False),
    "procedure_trigger":        CmdEntry(_handle_procedure_trigger,    is_write=False),
    "get_config":               CmdEntry(_handle_get_config,           is_write=False),
    "get_debug_status":         CmdEntry(_handle_get_debug_status,     is_write=False),
    "enrichment_coverage":      CmdEntry(_handle_enrichment_coverage,  is_write=False),
    "pre_edit":                 CmdEntry(_handle_pre_edit,             is_write=False),

    # ── Writes (exclusive lock) ──
    "save":                CmdEntry(_handle_save,               is_write=True, marks_dirty=False),
    "record_message":      CmdEntry(_handle_record_message,     is_write=True, marks_dirty=True),
    "reset_session":       CmdEntry(_handle_reset_session,      is_write=True, marks_dirty=True),
    "set_config":          CmdEntry(_handle_set_config,         is_write=True, marks_dirty=True),
    "log_debug":           CmdEntry(_handle_log_debug,          is_write=True, marks_dirty=True),
    "promote_staged":      CmdEntry(_handle_promote_staged,     is_write=True, marks_dirty=True),
    "backfill_summaries":  CmdEntry(_handle_backfill_summaries, is_write=True, marks_dirty=True),
    "backfill_vectors":    CmdEntry(_handle_backfill_vectors,   is_write=True, marks_dirty=True),
    "remember":              CmdEntry(_handle_remember,             is_write=True, marks_dirty=True),
    "remember_batch":        CmdEntry(_handle_remember_batch,      is_write=True, marks_dirty=True),
    "revise":                CmdEntry(_handle_revise,               is_write=True, marks_dirty=True),
    "revise_batch":          CmdEntry(_handle_revise_batch,         is_write=True, marks_dirty=True),
    "find_node_by_title":    CmdEntry(_handle_find_node_by_title,  is_write=False, marks_dirty=False),
    "filter_nodes":          CmdEntry(_handle_filter_nodes,        is_write=False, marks_dirty=False),
    "query_logs":            CmdEntry(_handle_query_logs,          is_write=False, marks_dirty=False),
    "clear_errors":          CmdEntry(_handle_clear_errors,        is_write=True,  marks_dirty=False),
    "query_traces":          CmdEntry(_handle_query_traces,        is_write=False, marks_dirty=False),
    "query_outcomes":        CmdEntry(_handle_query_outcomes,      is_write=False, marks_dirty=False),
    "count_traces":          CmdEntry(_handle_count_traces,        is_write=False, marks_dirty=False),
    "list_interactions":     CmdEntry(_handle_list_interactions,   is_write=False, marks_dirty=False),
    "get_interaction":       CmdEntry(_handle_get_interaction,     is_write=False, marks_dirty=False),
    "register_interaction":  CmdEntry(_handle_register_interaction,is_write=True,  marks_dirty=False),
    "set_interaction_active": CmdEntry(_handle_set_interaction_active, is_write=True,  marks_dirty=False),
    "trace_append":          CmdEntry(_handle_trace_append,        is_write=True,  marks_dirty=False),
    "get_node":              CmdEntry(_handle_get_node,             is_write=False, marks_dirty=False),
    "get_nodes":             CmdEntry(_handle_get_nodes,            is_write=False, marks_dirty=False),
    "recall_batch":          CmdEntry(_handle_recall_batch,         is_write=False, marks_dirty=False),
    "graph_expand":          CmdEntry(_handle_graph_expand,         is_write=False, marks_dirty=False),
    "connect":               CmdEntry(_handle_connect,             is_write=True, marks_dirty=True,
                                      accepts=frozenset({"source_id", "target_id", "relation",
                                                         "weight", "description", "encoding_source",
                                                         "chain_id", "session_id", "reason"})),
    "connect_batch":         CmdEntry(_handle_connect_batch,       is_write=True, marks_dirty=True,
                                      accepts=frozenset({"connections", "encoding_source",
                                                         "chain_id", "session_id", "reason"})),
    "brain_batch":           CmdEntry(_handle_brain_batch,         is_write=True, marks_dirty=True,
                                      accepts=frozenset({"operations", "encoding_source",
                                                         "chain_id", "session_id", "reason"})),
    "enrich":                CmdEntry(_handle_enrich,              is_write=True, marks_dirty=True),
    "eval":                  CmdEntry(_handle_eval,                is_write=True, marks_dirty=True),
    "diagnose":              CmdEntry(_handle_diagnose,            is_write=False, marks_dirty=False),
    "migrate_to_aspects":    CmdEntry(_handle_migrate_to_aspects,  is_write=True,  marks_dirty=True),
    "drop_sys_revision_history": CmdEntry(_handle_drop_sys_revision_history, is_write=True, marks_dirty=True),
}

# "shutdown" is handled directly by daemon_server (needs to set self.running)
# "hook_*" commands are dispatched via HOOK_TABLE in daemon_server
