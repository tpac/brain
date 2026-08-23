"""Daemon dispatch — observability + learnable-boundary handlers.

Traces, logs, outcomes (the nervous system) and interactions (the K store).
"""

import json

from .clock import iso_cutoff


def _handle_trace_append(brain, args, graph_changes):
    """Append a trace event from any source. Validates against trace_contract.

    Cross-process clients (e.g. PostToolUse hook) typically JSON-encode
    metadata before putting it on the wire because the daemon JSON
    decoder only unwraps one level. Decode it back here so TraceDAL
    stores a clean dict (and identity stamping receives a dict, not a
    string).
    """
    raw_meta = args.get("metadata")
    if isinstance(raw_meta, str) and raw_meta:
        try:
            raw_meta = json.loads(raw_meta)
        except (ValueError, TypeError):
            raw_meta = {"raw": raw_meta}
    # Payload-shape validation lives at the single chokepoint every writer passes
    # — TraceDAL.append/append_batch (loud, never blocking). It used to ALSO run
    # here, which double-validated dispatched traces and split logging across two
    # sinks (errors table here, stderr in the DAL). One point now; the DAL warns
    # to stderr (it can't write the errors table mid-append without risking the
    # brain_batch commit). The json-decode below still normalizes wire metadata.
    try:
        event_id = brain._trace_dal.append(
            chain_id=args.get("chain_id", ""),
            scale=args.get("scale", "s0"),
            event_type=args.get("event_type", ""),
            ref_type=args.get("ref_type", ""),
            ref_id=args.get("ref_id", ""),
            summary=args.get("summary", ""),
            metadata=raw_meta,
            session_id=args.get("session_id", ""),
            interaction_id=args.get("interaction_id"))
        return {"ok": True, "result": {"event_id": event_id}}
    except ValueError as e:
        return {"ok": False, "error": str(e)}


def _handle_get_trace(brain, args, graph_changes):
    """Point-lookup a trace_event by id. v29: id is 8-char hex string.
    Rejects int input loudly — coercion was removed because random hex
    generation made it unsafe (collision with migrated-int range)."""
    trace_id = args.get("trace_id")
    if trace_id is None:
        return {"ok": False, "error": "trace_id is required"}
    if not isinstance(trace_id, str) or not trace_id.strip():
        return {"ok": False,
                "error": "trace_id must be an 8-char hex string (v29), got %r" % (trace_id,)}
    tid = trace_id.strip().lower()
    row = brain.get_trace(tid)
    if not row:
        return {"ok": False, "error": "Trace not found: %s" % tid}
    return {"ok": True, "result": row}


def _handle_get_traces(brain, args, graph_changes):
    """Batch trace_event lookup. v29: ids are 8-char hex strings.
    Accepts up to 50 ids; missing ids skipped. Rejects ints loudly."""
    trace_ids = args.get("trace_ids", [])
    if not isinstance(trace_ids, list):
        return {"ok": False, "error": "trace_ids must be a list of 8-char hex strings"}
    cleaned: list = []
    bad: list = []
    for t in trace_ids[:50]:  # cap to 50 per call
        if isinstance(t, str) and t.strip():
            cleaned.append(t.strip().lower())
        else:
            bad.append(t)
    if bad:
        return {"ok": False,
                "error": "trace_ids must be 8-char hex strings (v29); rejected: %r" % bad[:5]}
    rows = brain.get_traces(cleaned) if cleaned else []
    out = {"ok": True, "result": rows}
    if bad:
        out["invalid_trace_ids"] = bad
    return out


def _handle_query_traces(brain, args, graph_changes):
    """Query trace events: O/K/Δ/outcome at every scale.

    session_id / session_ids handling: brain.query_traces (and ultimately
    TraceDAL.get_recent) refuses to combine the two. We forward what the
    caller sent verbatim and surface any ValueError as a structured error
    rather than crashing the daemon thread.
    """
    sids = args.get("session_ids")
    if sids is not None and not isinstance(sids, list):
        return {"ok": False, "error": "session_ids must be a list of strings"}
    try:
        result = brain.query_traces(
            scale=args.get("scale", ""),
            hours=args.get("hours", 24),
            event_type=args.get("event_type", ""),
            chain_id=args.get("chain_id", ""),
            session_id=args.get("session_id", ""),
            session_ids=sids,
            ref_type=args.get("ref_type", ""),
            grouped=args.get("grouped", False),
            limit=args.get("limit", 100))
        return {"ok": True, "result": result}
    except ValueError as e:
        return {"ok": False, "error": str(e)}


def _handle_recall_episodes(brain, args, graph_changes):
    """Episodic pull over the traces layer — semantic (`query`) and/or lexical
    (`contains`) search over trace_events, returning full episode records.

    session_id / session_ids are mutually exclusive (filter_events raises);
    surface any ValueError as a structured error rather than crashing the
    daemon thread, same as query_traces.
    """
    sids = args.get("session_ids")
    if sids is not None and not isinstance(sids, list):
        return {"ok": False, "error": "session_ids must be a list of strings"}
    try:
        result = brain.recall_episodes(
            query=args.get("query"),
            contains=args.get("contains"),
            session_id=args.get("session_id"),
            session_ids=sids,
            scale=args.get("scale", "s0"),
            event_type=args.get("event_type"),
            ref_type=args.get("ref_type"),
            older_than=args.get("older_than"),
            younger_than=args.get("younger_than"),
            sort_order=args.get("sort_order", "desc"),
            limit=args.get("limit"))
        return {"ok": True, "result": result}
    except ValueError as e:
        return {"ok": False, "error": str(e)}


def _handle_count_traces(brain, args, graph_changes):
    """Count trace events grouped by a field."""
    result = brain.count_traces(
        field=args.get("field", "event_type"),
        scale=args.get("scale", ""),
        hours=args.get("hours", 24))
    return {"ok": True, "result": result}


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
    hours = args.get("hours")  # None = clear all
    cleared = brain._logs_dal.clear_errors(
        cutoff_iso=iso_cutoff(hours=int(hours)) if hours else '',
        include_debug_log=bool(args.get("debug_log", False)))
    return {"ok": True, "result": cleared}


def _handle_log_debug(brain, args, graph_changes):
    brain.log_debug(
        args.get("event", ""), args.get("source", ""),
        metadata=args.get("metadata"))
    return {"ok": True, "result": {"status": "logged"}}


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
    from .dal_logs import SYSTEM_PROVENANCE
    name = args.get("name", "")
    version = args.get("version")
    set_by = args.get("set_by", "anchor")
    if not name or version is None:
        return {"ok": False, "error": "name and version are required"}
    if set_by in SYSTEM_PROVENANCE:
        # Reserved: the shipped-prompt reconcile reads these to tell its own
        # writes apart from a human's. Accepting one here would relabel this
        # deployment decision as an untouched default and let a later bump
        # publish over it.
        return {"ok": False,
                "error": "set_by '%s' is reserved for system provenance"
                         % set_by}
    try:
        version = int(version)
    except (TypeError, ValueError):
        return {"ok": False, "error": "version must be an integer"}
    try:
        result = brain.set_interaction_active(name, version, set_by)
        return {"ok": True, "result": result}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def _handle_clear_interaction_override(brain, args, graph_changes):
    """Delete the active pointer for an interaction — revert to the code default.

    The inverse of set_interaction_active. After this call the runtime path
    (brain.get_interaction_prompt / get_interaction_config) resolves the code
    default, immediately — TTL caches are invalidated. Registered versions
    stay on record for re-activation. No set_by is written (a delete leaves
    no row), so the reserved-provenance check does not apply. A typo'd name
    (nothing deleted, no code default) is refused, never reported as
    "already on the default".
    """
    name = args.get("name", "")
    if not name:
        return {"ok": False, "error": "name is required"}
    try:
        result = brain.clear_interaction_override(name)
        result["note"] = (
            "Override cleared — runtime now resolves the code default."
            if result.get("cleared")
            else "No pointer existed — already running the code default.")
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def _handle_register_interaction(brain, args, graph_changes):
    """Register a new version of an interaction (prompt + config).

    Creates a new version if the interaction exists, or version 1 if new.
    Used by S2/S3 to evolve learnable boundaries.

    NOTE: Registration NEVER activates (a write is not a deployment
    decision) — every name runs on its code default until
    set_interaction_active deploys an override.
    """
    from .dal_logs import SYSTEM_PROVENANCE
    name = args.get("name", "")
    template = args.get("template", "")
    parameters = args.get("parameters", "{}")
    created_by = args.get("created_by", "")

    if not name:
        return {"ok": False, "error": "name is required"}
    if created_by in SYSTEM_PROVENANCE:
        # Reserved — see _handle_set_interaction_active. A version claiming to
        # be reconcile's own would be treated as crash residue and adopted.
        return {"ok": False,
                "error": "created_by '%s' is reserved for system provenance"
                         % created_by}

    try:
        result = brain.register_interaction(
            name=name,
            template=template,
            parameters=parameters,
            created_by=created_by)
        # Registration does NOT activate. Return the newly-registered version
        # plus the currently-active version so the caller knows whether it
        # took effect at runtime.
        active = brain.get_interaction(name)
        return {"ok": True, "result": {
            "name": name,
            "registered_version": result.get("version"),
            "active_version": active.get("version") if active else None,
            "template_length": len(template),
            "note": (
                "Registration created a new version but did NOT activate it. "
                "Call set_interaction_active to deploy it as the override."),
        }}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
