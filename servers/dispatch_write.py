"""Daemon dispatch — write handlers (mutations).

remember / revise / connect / enrich / brain_batch and their helpers:
source_refs validation, and the revise/edge trace emitters. Every write goes
through here.
"""

import re
import sys

from .clock import brain_today
from .dispatch_common import _resolve_id, _pop_session_ctx, caller_session, CALLER_SESSION_KEY
from .scales.dispatch import stamp_project_provenance


_HEX_TRACE_ID_RE = re.compile(r'^[0-9a-f]{8}$')


def _stamp_session_project(brain, cmd, args, ctx=None):
    """Deterministic project provenance at the MCP write boundary.

    Resolves the calling session's derived project (SessionContext.project)
    and applies stamp_project_provenance: node-creating payloads get it
    force-stamped, agent-supplied values elsewhere are dropped. Sessionless
    callers (the in-process encoder dispatch — already stamped at its own
    chokepoint — and direct handler calls) pass through untouched.

    Returns warning strings for the handler to surface in its result.
    """
    if cmd in ('revise', 'revise_batch'):
        # revise policy strips regardless of the project VALUE — only the
        # session's existence matters, so skip the SessionContext load
        project = '' if caller_session(args) else None
    elif ctx is not None:
        project = ctx.project
    else:
        sid = caller_session(args)
        project = (brain.session_env_for(sid).get('project', '')
                   if sid else None)
    warnings = stamp_project_provenance(cmd, args, project)
    for w in warnings:
        # expected agent behavior (the MCP schema still advertises project),
        # surfaced as a warning — not an error-feed entry
        brain._log_warning('project_provenance_stamp', w, 'cmd=%s' % cmd)
    return warnings


def _validate_source_refs(refs, location: str):
    """v29 / Phase B Step 4 — validation for source_refs at the write boundary.

    Returns (ok, error|None). Sparseness + hex-format warnings logged
    separately (non-fatal).

    Rules:
      - Must be a list (or None — handled by caller as "no refs"; empty list
        is legitimate per the unified revise contract — explicit clear).
      - Each item must be a string. Int input rejected loudly (no coercion);
        v29 contract is hex strings end-to-end (reviewer F2).
      - Empty strings rejected.
      - Hex format mismatch (^[0-9a-f]{8}$): soft warn, write proceeds
        (reviewer F6 — catches encoder regressions earlier than S2Healer).
      - >5 refs logged as sparseness warning (decision 13 / §7.5 v22 prompt:
        1-3 typical; second-guess at 5-6; 10+ is "ref everything" anti-pattern).
        Non-fatal — write proceeds, log fires.
    """
    if refs is None:
        return True, None
    if not isinstance(refs, list):
        return False, "source_refs at %s must be a list, got %s" % (
            location, type(refs).__name__)
    for i, r in enumerate(refs):
        if not isinstance(r, str):
            return False, ("source_refs[%d] at %s must be an 8-char hex "
                           "string (v29), got %s (%r)") % (
                i, location, type(r).__name__, r)
        if not r.strip():
            return False, "source_refs[%d] at %s is empty/whitespace" % (i, location)
    return True, None


def _maybe_warn_source_refs_hex_format(brain, refs, location: str):
    """Soft warn when any source_refs entry doesn't match the v29 8-char hex
    shape. Non-fatal — refs persist; S2Healer's invalid_refs_dropped_total
    eventually cleans hallucinated ids but this surfaces them at the write
    boundary (reviewer F6). Most common cause: encoder copied a literal
    `<trace-...>` placeholder from an example instead of substituting real
    hex from the conversation timeline."""
    if not refs or not isinstance(refs, list):
        return
    malformed = [r for r in refs
                 if isinstance(r, str) and not _HEX_TRACE_ID_RE.match(r)]
    if malformed:
        try:
            brain._log_warning(
                'source_refs_hex_format',
                'source_refs at %s has %d entries that do not match the v29 '
                '8-char hex pattern (^[0-9a-f]{8}$). Examples: %r. Likely '
                'literal example-placeholders or pre-v29 integer ids. Refs '
                'persist; recall will degrade gracefully but S2Healer will '
                'archive them.' % (location, len(malformed), malformed[:3]),
                context='location=%s, malformed_count=%d' % (
                    location, len(malformed)))
        except Exception:
            pass  # logging failure must never block the write


def _maybe_warn_source_refs_sparseness(brain, refs, location: str):
    """Log a sparseness warning when source_refs exceeds 5 (decision 13 +
    §7.5 v22 prompt). 1-3 typical, second-guess at 5-6, 10+ is the
    "ref everything" anti-pattern. Non-fatal — the write proceeds.
    Surfaces systemic over-anchoring.

    v22 lowered the threshold from >10 to >5 to align with the prompt's
    sparseness teaching ("when you find yourself wanting to add a 5th or
    6th ref, ask: would that ref actually be the one that surfaces this
    memory next time, or is it just adjacent context?").

    Routes through brain._log_warning for rate-limiting + file-log mirror +
    canonical surface (reviewer F4)."""
    if not refs or not isinstance(refs, list):
        return
    if len(refs) > 5:
        try:
            brain._log_warning(
                'source_refs_sparseness',
                'source_refs at %s has %d entries (decision 13: 1-3 '
                'typical, second-guess at 5-6, 10+ is anti-pattern). '
                'Likely over-anchoring; recall will dilute rather than '
                'focus.' % (location, len(refs)),
                context='count=%d, location=%s' % (len(refs), location))
        except Exception:
            pass  # logging failure must never block the write


def _missing_reason_error(spec, prefix=''):
    """Build the missing-`reason` error for revise, disambiguating `reasoning`.

    `reason` (audit note, recorded in the node_revised trace event, never
    stored on the node) and `reasoning` (a PROMOTED node field — "why this
    was encoded", stored in node_metadata_kv) are near-identical names for
    different concepts, and agents confuse them (observed 2026-06-12: a
    revise op carrying `reasoning` where `reason` was meant). A blind alias
    would be wrong — `reasoning` is a legitimate field update on revise —
    so when it's present the error names both and tells the caller how to
    self-correct.
    """
    base = (prefix + "reason is required — the audit note explaining why "
            "this revision (recorded in the trace event, NOT stored on "
            "the node)")
    if spec.get('reasoning'):
        return (base + ". You passed `reasoning`, which is a node FIELD "
                "(why the node was encoded — updates node metadata), not "
                "the audit note. If you meant the audit note, rename it to "
                "`reason`; if you meant to update the node's reasoning "
                "field, keep `reasoning` and add `reason`.")
    return base


def _infer_scale_and_chain(brain, encoding_source, chain_id_override):
    """Shared by the revise trace emitters.

    Infers scale from encoding_source ('s2:*'→s2, 'encoder:*'→s1, else s0) and
    resolves the chain_id: a caller-provided override wins (encoder cycles pass
    theirs so revises join the encoder's chain); otherwise a date-based
    per-scale chain ('{scale}-{YYYYMMDD}-revise') groups direct/operator
    revises by day.
    """
    if encoding_source.startswith('s2:'):
        scale = 's2'
    elif encoding_source.startswith('encoder:'):
        scale = 's1'
    else:
        scale = 's0'
    if chain_id_override:
        chain_id = chain_id_override
    else:
        # Date-based fallback chain. Route the date through clock.brain_today
        # (operator frame) rather than raw datetime.utcnow() — consistent with
        # the codebase-wide clock alignment and with s2/base's chain-id dates.
        chain_id = '%s-%s-revise' % (scale, brain_today(brain).strftime('%Y%m%d'))
    return scale, chain_id


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

    FAILURE-ISOLATED: the whole body is wrapped (a revise inside a brain_batch
    runs in the batch transaction — a trace error here must log loudly, not
    raise and roll back the revise it was recording).
    """
    warnings = warnings or []
    if not deltas and not warnings:
        return  # nothing to record

    try:
        from .trace_contract import build_revise_metadata
        scale, chain_id = _infer_scale_and_chain(
            brain, encoding_source, chain_id_override)
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
                         'failed to emit trace for revise of %s' % str(node_id)[:8])


def _emit_edge_revise_trace(brain, edge_id, relation, reason, encoding_source,
                            deltas, warnings=None,
                            chain_id_override='', session_id='',
                            source_id='', target_id=''):
    """Emit an edge_relation_revised trace event.

    Mirrors _emit_revise_trace for nodes. Same emit-on-deltas-or-warnings
    behavior, same scale inference, same chain_id strategy. ref_id encodes
    the (edge_id, relation) tuple as f"{edge_id}:{relation}".

    source_id/target_id carry the directional pair into the metadata so the
    edge is reconstructable from the trace alone (edge_id is a one-way hash).

    Used by every edge path: the connect upsert (deltas show create-via-INSERT
    or field-preserving update), connect_batch, revise_edge, the connect_to /
    co_anchored paths (via _emit_edge_traces), and disconnect (archived flip).

    FAILURE-ISOLATED: the ENTIRE body is wrapped — scale/chain inference and
    metadata build included, not just the append — so a trace-layer error logs
    loudly and returns. Trace emission is observability; it must NEVER raise
    into the caller and roll back the graph write that produced the edge.
    """
    warnings = warnings or []
    if not deltas and not warnings:
        return  # nothing to record

    try:
        from .trace_contract import build_edge_revise_metadata
        scale, chain_id = _infer_scale_and_chain(
            brain, encoding_source, chain_id_override)
        metadata = build_edge_revise_metadata(
            edge_id=edge_id, relation=relation, reason=reason,
            encoding_source=encoding_source,
            source_id=source_id, target_id=target_id,
            deltas=deltas, warnings=warnings)
        parts = []
        if deltas:
            parts.append('%d field(s): %s' % (
                len(deltas), ', '.join(d['field'] for d in deltas)))
        if warnings:
            parts.append('%d warning(s)' % len(warnings))
        summary = '; '.join(parts) if parts else 'edge revise no-op'
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
                             str(edge_id)[:12], relation))


def _affected(created=None, revised=None, archived=None):
    """Build the authoritative node-lifecycle split a write handler returns.

    Sits at the TOP LEVEL of a handler's return (sibling to `result`), so it
    reaches the trace substrate — the runner copies it onto each action and
    the daemon forwards it verbatim — without ever entering the agent-facing
    payload (only `result` is formatted back to the model). NODE lifecycle
    only; edges are directional `edge_relation_revised` events, not a flat
    list here. Replaces the runner's tool-name `_split_action_ids` heuristic
    with attribution from the layer that actually knows the op.
    """
    return {'created': list(created or []),
            'revised': list(revised or []),
            'archived': list(archived or [])}


def _emit_edge_traces(brain, made, encoding_source,
                      chain_id_override='', session_id='',
                      reason='connect_to'):
    """Emit directional edge_relation_revised traces for a list of made edges.

    `made` entries are self-contained {src_id, target_id, relation, edge_id,
    deltas} produced by brain._apply_connect_to (connect_to) and remember()'s
    co_anchored path. This is the single seam where remember()-internal edges
    join the same edge-trace substrate the explicit connect ops use — closing
    the gap where those edges emitted nothing.

    ONE append_batch (single commit on the logs DB) for the whole list instead
    of N per-edge appends — scale/chain are loop-invariant here, so they're
    resolved once. Failure-isolated as a unit: a logs-side error logs loudly
    and returns; it never raises into / rolls back the graph write that
    produced these edges (the write already committed by the time we emit).
    Edges with empty deltas (idempotent re-connect — no change) are skipped:
    nothing to record, by design.
    """
    rows = [e for e in (made or [])
            if isinstance(e, dict) and e.get('edge_id') and (e.get('deltas') or [])]
    if not rows:
        return
    try:
        from .trace_contract import build_edge_revise_metadata
        scale, chain_id = _infer_scale_and_chain(
            brain, encoding_source, chain_id_override)
        events = []
        for e in rows:
            deltas = e.get('deltas') or []
            relation = e.get('relation', '')
            events.append({
                'chain_id': chain_id, 'scale': scale, 'event_type': 'delta',
                'ref_type': 'edge_relation_revised',
                'ref_id': '%s:%s' % (e['edge_id'], relation),
                'summary': '%d field(s): %s' % (
                    len(deltas), ', '.join(d['field'] for d in deltas)),
                'metadata': build_edge_revise_metadata(
                    edge_id=e['edge_id'], relation=relation, reason=reason,
                    encoding_source=encoding_source,
                    source_id=e.get('src_id', ''), target_id=e.get('target_id', ''),
                    deltas=deltas),
                'session_id': session_id,
            })
        brain._trace_dal.append_batch(events)
    except Exception as ex:
        brain._log_error('edge_traces_emit', ex,
                         'failed to batch-emit %d edge traces' % len(rows))


def _resolve_archived_by(op_spec, top_encoding_source):
    """Attribution fallback for batch archive/absorb/disconnect ops:
    op-level encoding_source → op-level archived_by → batch encoding_source →
    'unknown'. One source so the three _op_* helpers can't drift."""
    return (op_spec.get('encoding_source') or op_spec.get('archived_by')
            or top_encoding_source or 'unknown')


def _handle_remember(brain, args, graph_changes):
    from .contract import validate_field

    # session_id is a control field — strip BEFORE validation/passthrough so
    # it doesn't land in node_metadata_kv as silent metadata. Loads ctx for
    # per-session record_remember + add_to_segment routing inside remember().
    ctx, args = _pop_session_ctx(brain, args)
    # chain_id is likewise a control field (trace grouping, NOT a node field).
    # remember() routes unknown kwargs to node_metadata_kv, so an un-popped
    # chain_id would leak onto the node — pop it here, use it only for traces.
    chain_id = args.pop('chain_id', '') if isinstance(args, dict) else ''

    # Deterministic project provenance — session-derived, never agent-authored.
    project_warnings = _stamp_session_project(brain, 'remember', args, ctx=ctx)

    # v29 / Phase B Step 4 — validate source_refs shape at the write boundary.
    refs = args.get('source_refs')
    ok, err = _validate_source_refs(refs, 'remember')
    if not ok:
        return {"ok": False, "error": err}
    _maybe_warn_source_refs_sparseness(brain, refs, 'remember')
    _maybe_warn_source_refs_hex_format(brain, refs, 'remember')

    # Validate all provided fields against contract
    for field, value in args.items():
        ok, err = validate_field(field, value)
        if not ok:
            return {"ok": False, "error": err}

    # Mirror of revise's reason/reasoning trap: `reason` is revise's audit
    # param and a remember control field — _store_node_metadata drops it
    # silently. An agent passing it here almost always meant the node field
    # `reasoning`. Semantics unchanged (still dropped); surface the drop.
    reason_warning = None
    if args.get('reason') and not args.get('reasoning'):
        reason_warning = (
            "`reason` is not a node field and was dropped — it is the audit "
            "note for revise ops. Did you mean `reasoning` (why this was "
            "encoded — stored on the node)?")

    # Pass ALL fields through to remember() — contract fields go to nodes table,
    # promoted fields to metadata, everything else to node_metadata_kv as extra_fields.
    # Don't filter — remember() handles routing via **extra_fields kwargs.
    remember_args = {k: v for k, v in args.items() if v is not None}
    result = brain.remember(**remember_args, ctx=ctx)
    if reason_warning and isinstance(result, dict):
        result.setdefault('warnings', []).append(reason_warning)
    if project_warnings and isinstance(result, dict):
        result.setdefault('warnings', []).extend(project_warnings)
    # ctx is cached on Brain; remember's record_remember mutation persists
    # via the autosave loop (no per-call save).
    full_id = result.get("id", "") if isinstance(result, dict) else ""
    node_id = full_id[:8] if full_id else "?"
    graph_changes.append(
        "REMEMBER: [%s] %s (%s...)" % (
            args.get("type", "?"), args.get("title", "")[:50], node_id))
    # Edges remember() materialized emit directional edge traces. Both
    # made-lists are already src-tagged + carry edge_id/deltas (connect_to via
    # _apply_connect_to, co_anchored via remember()), so no re-deriving here.
    enc_src = args.get('encoding_source', '')
    sess = caller_session(args)
    ctr = result.get('connect_to_result') if isinstance(result, dict) else None
    if ctr and ctr.get('created'):
        _emit_edge_traces(brain, ctr['created'], enc_src,
                          chain_id_override=chain_id, session_id=sess,
                          reason='connect_to')
    # pop: co_anchored is an automatic internal edge, not agent-facing like
    # connect_to_result — consume it for tracing, keep it out of the payload.
    co_anchored = result.pop('co_anchored_made', None) if isinstance(result, dict) else None
    if co_anchored:
        _emit_edge_traces(brain, co_anchored, enc_src,
                          chain_id_override=chain_id, session_id=sess,
                          reason='co_anchored (shared episodic anchor)')
    return {"ok": True, "result": result,
            "affected": _affected(created=[full_id] if full_id else None)}


def _handle_remember_batch(brain, args, graph_changes):
    from .contract import validate_field, get_remember_fields

    # Pop session_id BEFORE spec scrubbing so per-node specs don't inherit
    # control fields. Also strip from each spec defensively in case the
    # caller bundled it inside a node spec.
    ctx, args = _pop_session_ctx(brain, args)

    nodes = args.get("nodes", [])
    if not nodes:
        return {"ok": False, "error": "nodes array is required"}

    # Deterministic project provenance — session-derived, never agent-authored.
    project_warnings = _stamp_session_project(
        brain, 'remember_batch', args, ctx=ctx)

    accepted_fields = set(get_remember_fields().keys())
    # Inherit top-level encoding_source into each node (dispatch wrapper injects this)
    top_encoding_source = args.get("encoding_source")

    cleaned_nodes = []
    reason_warnings = []  # reason/reasoning confusion — see _handle_remember
    for i, spec in enumerate(nodes):
        # defensive: neither identity key is a node field. _pop_session_ctx
        # already stripped the top-level args; this guards a spec that bundled
        # either key per-node (so it can't cascade into node_metadata_kv).
        spec.pop('session_id', None)
        spec.pop(CALLER_SESSION_KEY, None)
        if spec.get('reason') and not spec.get('reasoning'):
            reason_warnings.append(
                "nodes[%d]: `reason` is not a node field and was dropped — "
                "it is the audit note for revise ops. Did you mean "
                "`reasoning` (why this was encoded — stored on the node)?" % i)
        # v29 / Phase B Step 4 — per-node source_refs validation
        refs = spec.get('source_refs')
        ok, err = _validate_source_refs(refs, 'remember_batch.nodes[%d]' % i)
        if not ok:
            return {"ok": False, "error": err}
        _maybe_warn_source_refs_sparseness(
            brain, refs, 'remember_batch.nodes[%d]' % i)
        _maybe_warn_source_refs_hex_format(
            brain, refs, 'remember_batch.nodes[%d]' % i)
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

    # remember_batch forwards only nodes/connect_to/ctx. The retired
    # `auto_connect` top-level arg (param removed 2026-06-18; it once triggered
    # pairwise empty-description `related_to` pollution every encoding cycle) is
    # dropped simply by not being forwarded — a stray one never reaches a node.
    result = brain.remember_batch(
        nodes=cleaned_nodes,
        connect_to=args.get("connect_to"),
        ctx=ctx)
    # ctx mutations persist via autosave (no per-call save).
    if reason_warnings and isinstance(result, dict):
        result.setdefault('warnings', []).extend(reason_warnings)
    if project_warnings and isinstance(result, dict):
        result.setdefault('warnings', []).extend(project_warnings)
    graph_changes.append("REMEMBER_BATCH: %d nodes" % result.get("nodes_created", 0))
    enc_src = args.get('encoding_source', '')
    chain = args.get('chain_id', '')
    sess = caller_session(args)
    # `connect_to_made` is the brain method's edge record — consumed here for
    # edge traces, then dropped so it doesn't bloat the agent-facing payload
    # (the edges live in edge_relation_revised events now).
    made = result.pop('connect_to_made', None) if isinstance(result, dict) else None
    _emit_edge_traces(brain, made, enc_src, chain_id_override=chain,
                      session_id=sess, reason='connect_to')
    # co_anchored edges fire inside each per-node remember(); collect + pop them
    # off the per-node results so they're traced but stay out of the payload.
    co_anchored = []
    for node_r in (result.get('results') or []):
        if isinstance(node_r, dict):
            co_anchored.extend(node_r.pop('co_anchored_made', None) or [])
    _emit_edge_traces(brain, co_anchored, enc_src, chain_id_override=chain,
                      session_id=sess, reason='co_anchored (shared episodic anchor)')
    created_ids = [r.get('id') for r in (result.get('results') or [])
                   if isinstance(r, dict) and r.get('id')]
    return {"ok": True, "result": result,
            "affected": _affected(created=created_ids)}


def _handle_revise(brain, args, graph_changes):
    """Update any field(s) on an existing node via revise()."""
    from .contract import validate_field

    node_id = _resolve_id(brain, args.get("node_id", ""))
    reason = args.get("reason", "")
    if not node_id:
        return {"ok": False, "error": "node_id is required"}
    if not reason:
        return {"ok": False, "error": _missing_reason_error(args)}

    # Project is birth provenance — a revise never moves it (migration does).
    project_warnings = _stamp_session_project(brain, 'revise', args)

    # Reserve known dispatch keys so they don't get treated as field updates.
    # CALLER_SESSION_KEY is the ambient identity the proxy stamps — reserve it
    # too so it never lands in `updates` as a bogus node field.
    DISPATCH_KEYS = {"node_id", "reason", "encoding_source", "chain_id",
                     "session_id", CALLER_SESSION_KEY}
    updates = {k: v for k, v in args.items() if k not in DISPATCH_KEYS}

    # v29 / Phase B Step 4 — source_refs validation on revise
    refs = updates.get('source_refs')
    ok, err = _validate_source_refs(refs, 'revise')
    if not ok:
        return {"ok": False, "error": err}
    _maybe_warn_source_refs_sparseness(brain, refs, 'revise')
    _maybe_warn_source_refs_hex_format(brain, refs, 'revise')

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
            print('[daemon_dispatch] ERROR logging write_verification: %s' % e2, file=sys.stderr)

    # Emit node_revised trace event (replaces _sys_revision_history substrate).
    # Includes warnings so audit history captures attempted-but-rejected ops.
    _emit_revise_trace(
        brain, node_id, reason,
        args.get('encoding_source', ''),
        result.get('deltas', []),
        warnings=result.get('warnings', []),
        chain_id_override=args.get('chain_id', ''),
        session_id=caller_session(args),
    )

    graph_changes.append("REVISE: [%s] %s" % (
        result.get("type", "?"), result.get("title", "")[:50]))
    if project_warnings and isinstance(result, dict):
        result.setdefault('warnings', []).extend(project_warnings)
    return {"ok": True, "result": result,
            "affected": _affected(revised=[node_id])}


def _handle_revise_batch(brain, args, graph_changes):
    """Revise multiple nodes in one call."""
    from .contract import validate_field

    revisions = args.get("revisions", [])
    if not revisions:
        return {"ok": False, "error": "revisions array is required"}

    # Project is birth provenance — a revise never moves it (migration does).
    project_warnings = _stamp_session_project(brain, 'revise_batch', args)

    # Inherit encoding_source from dispatch wrapper
    top_encoding_source = args.get("encoding_source")

    # Validate each revision
    for i, spec in enumerate(revisions):
        if not spec.get("node_id"):
            return {"ok": False, "error": "revisions[%d]: node_id required" % i}
        if not spec.get("reason"):
            return {"ok": False,
                    "error": _missing_reason_error(spec, "revisions[%d]: " % i)}
        # v29 / Phase B Step 4 — per-revision source_refs validation
        refs = spec.get('source_refs')
        ok, err = _validate_source_refs(refs, 'revise_batch.revisions[%d]' % i)
        if not ok:
            return {"ok": False, "error": err}
        _maybe_warn_source_refs_sparseness(
            brain, refs, 'revise_batch.revisions[%d]' % i)
        _maybe_warn_source_refs_hex_format(
            brain, refs, 'revise_batch.revisions[%d]' % i)
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
    session_id = caller_session(args)
    revised_ids = []
    for row, spec in zip(result.get('results', []), resolved):
        if row.get('status') == 'revised':
            revised_ids.append(row['node_id'])
            _emit_revise_trace(
                brain, row['node_id'], spec.get('reason', ''),
                spec.get('encoding_source', '') or top_encoding_source or '',
                row.get('deltas', []),
                warnings=row.get('warnings', []),
                chain_id_override=chain_id_override,
                session_id=session_id,
            )

    if project_warnings and isinstance(result, dict):
        result.setdefault('warnings', []).extend(project_warnings)
    return {"ok": True, "result": result,
            "affected": _affected(revised=revised_ids)}


def _op_archive(brain, op_spec, top_encoding_source, graph_changes):
    """brain_batch `archive` op — soft-archive a node (guards, edges, vectors,
    audit all handled by archive_node). Returns archive_node's result plus the
    node-lifecycle `affected` split. Batch-only op (no standalone tool), hence
    a `_op_*` helper rather than a `_handle_*` handler."""
    node_id = op_spec.get("node_id")
    archived_by = _resolve_archived_by(op_spec, top_encoding_source)
    r = brain.archive_node(
        node_id, archived_by=archived_by, reason=op_spec.get('reason', ''))
    if r.get('ok'):
        graph_changes.append("ARCHIVE: %s" % node_id[:8])
        r["affected"] = _affected(archived=[node_id])
    return r


def _op_absorb(brain, op_spec, top_encoding_source, graph_changes):
    """brain_batch `absorb` op — lossless merge: fold absorbed INTO survivor,
    then archive absorbed (transfer-by-default: source_refs, edges,
    access_count, KV — so a merge can't silently drop what the imperative
    revise+connect+archive path did). survivor may be locked; absorbed must
    not be. affected: survivor revised (content rewritten) + absorbed
    archived, so a merge-only run is no longer invisible to S2."""
    survivor_id = op_spec.get("survivor_id")
    absorbed_id = op_spec.get("absorbed_id")
    archived_by = _resolve_archived_by(op_spec, top_encoding_source)
    # Revise-op style: every non-control key is a survivor field override
    # (content, title, confidence, situation, ...), forwarded to absorb()'s
    # updates — same mental model as the revise op.
    _CONTROL = {'op', 'survivor_id', 'absorbed_id', 'prune_edges',
                'drop_fields', 'archived_by', 'encoding_source', 'reason',
                'session_id', 'chain_id'}
    field_updates = {k: v for k, v in op_spec.items() if k not in _CONTROL}
    r = brain.absorb(
        survivor_id, absorbed_id, updates=field_updates or None,
        archived_by=archived_by, reason=op_spec.get('reason', ''),
        prune_edges=op_spec.get('prune_edges'),
        drop_fields=op_spec.get('drop_fields'))
    if r.get('ok'):
        graph_changes.append("ABSORB: %s <- %s" % (
            survivor_id[:8], absorbed_id[:8]))
        r["affected"] = _affected(revised=[survivor_id], archived=[absorbed_id])
    return r


def _op_disconnect(brain, op_spec, top_encoding_source, top_session_id, args,
                   graph_changes):
    """brain_batch `disconnect` op — soft-archive ONE relation on an edge
    (other relations on the same edge survive; archived row kept for
    forensics/recovery). Emits the archived-flag-flip edge trace. Edge-only —
    no node lifecycle, so no `affected`."""
    # Resolve endpoints — mirrors _handle_connect/_handle_connect_batch. Without
    # this, short/title ids passed by an encoder miss get_edge_id (silent no-op)
    # and the edge trace records unresolved ids inconsistent with other edges.
    source_id = _resolve_id(brain, op_spec.get("source_id", ""))
    target_id = _resolve_id(brain, op_spec.get("target_id", ""))
    relation = op_spec.get("relation")
    archived_by = _resolve_archived_by(op_spec, top_encoding_source)
    gdal = brain._graph
    edge_id = gdal.get_edge_id(source_id, target_id)
    # remove_relation gates its own commit on conn.in_batch (True here) →
    # deferred to the batch's single COMMIT.
    gdal.remove_relation(source_id, target_id, relation, archived_by=archived_by)
    graph_changes.append("DISCONNECT: %s -[%s]-> %s" % (
        source_id[:8], relation, target_id[:8]))
    # Emit edge_relation_revised trace capturing the archived flag flip.
    if edge_id:
        _emit_edge_revise_trace(
            brain, edge_id, relation,
            op_spec.get('reason', '') or args.get('reason', ''),
            archived_by,
            deltas=[{'field': 'archived', 'old': 0, 'new': 1}],
            chain_id_override=args.get('chain_id', ''),
            session_id=top_session_id,
            source_id=source_id, target_id=target_id)
    return {"ok": True}


def _handle_brain_batch(brain, args, graph_changes):
    """Execute multiple brain operations in one call.

    Accepts mixed operations: remember, revise, connect in any order.
    Each operation is validated and executed sequentially.
    Returns results for each operation.

    Args:
        operations: list of {op: "remember"|"revise"|"connect", ...fields}
    """
    from .contract import BATCH_OP_SPECS

    operations = args.get("operations", [])
    if not operations:
        return {"ok": False, "error": "operations array is required"}

    # Deterministic project provenance at the BATCH boundary. The remember/
    # revise sub-ops would be covered by their leaf handlers anyway, but ops
    # that don't delegate (absorb — whose field overrides flow into revise()
    # via brain.absorb(updates=...)) are only guarded here; without this an
    # agent-supplied `project` on an absorb op silently moves birth provenance.
    project_warnings = _stamp_session_project(brain, 'brain_batch', args)

    # Valid nested op names live in contract.VALID_BATCH_OPS (single source of
    # truth — see that constant). The dispatcher matches them via the if/elif
    # chain below; the final `else` is the invalid-op guard. Sonnet has been
    # observed inventing structural op names like `consolidate` / `keep` /
    # `evolve` / `skip` / `reject` — these are prompt-level
    # DECISIONS, not dispatch ops. They land in the "Unknown op" branch, get
    # logged loudly (brain_batch_invalid_op), and the S2 rejection_table now
    # detects them so a dropped op isn't mistaken for a clean SKIP.

    top_encoding_source = args.get("encoding_source")
    # The caller's identity, resolved once. Propagated to every sub-op under the
    # RESERVED key (never session_id) so attribution flows to the caller while
    # `session_id == filter` stays true at every layer — a sub-op can't be
    # accidentally scoped by the ambient identity. Sub-handlers read it via
    # caller_session(); remember additionally pops it through _pop_session_ctx.
    top_session_id = caller_session(args)
    # The batch's trace chain — injected into each sub-op so node_revised /
    # edge_relation_revised / co_anchored traces join the encoder's chain
    # instead of scattering to a date-fallback chain. Sub-handlers treat
    # chain_id as control (revise via DISPATCH_KEYS, remember pops it, connect
    # passes explicit kwargs) so it never leaks onto a node.
    top_chain_id = args.get('chain_id', '')
    results = []

    # Sibling-aware connect_to: defer per-op connect_to from `remember` ops
    # until ALL ops in this batch have run, so siblings declared in any order
    # can resolve. NEW wins on title collision (sibling beats catalog).
    sibling_map = {}  # lowercased title → new node_id
    deferred_connects = []  # [(src_node_id, connect_to_spec)]

    # Node-lifecycle Δ aggregated across all sub-ops — the authoritative
    # `affected` the batch returns at top level (the runner reads it for the
    # encoding_run delta; the agent never sees it). Popped off each sub-result
    # so the per-op `results` array stays the agent-facing shape it always was.
    agg = {'created': [], 'revised': [], 'archived': []}

    def _accumulate(aff):
        if aff:
            agg['created'].extend(aff.get('created') or [])
            agg['revised'].extend(aff.get('revised') or [])
            agg['archived'].extend(aff.get('archived') or [])

    # Wrap the whole batch in ONE SQLite transaction. Sub-handlers and DAL
    # writers gate their commits on brain.conn.in_batch (commit_unless_batched);
    # setting it True here makes every per-op commit a no-op so the outer
    # BEGIN IMMEDIATE / COMMIT below owns the single durability point.
    # Pre-change every op committed individually, so a batch of N ops hit
    # the WAL writer slot N times — bad for parallel-session contention,
    # no rollback semantic if op #37 broke. Per-op exceptions are still
    # caught by the inner try/except below and recorded in `results` —
    # that "best-effort" surface is preserved. Only when something
    # escapes the per-op handler entirely (a programmer bug, not a
    # caller-visible op failure) do we rollback the whole batch.
    brain.conn.in_batch = True
    transaction_started = False
    # brain_batch owns its own BEGIN IMMEDIATE / COMMIT envelope. If self.conn
    # is ALREADY mid-transaction here, an upstream op left a deferred auto-BEGIN
    # open without committing (self.conn uses Python's default deferred
    # isolation). Inheriting it would (a) make the explicit BEGIN IMMEDIATE
    # below throw "cannot start a transaction within a transaction" and (b)
    # silently fold that op's uncommitted writes into THIS batch's commit.
    # Flush the orphan cleanly and log loud so the upstream leak stays visible.
    if brain.conn.in_transaction:
        try:
            brain.conn.commit()
        except Exception:
            try:
                brain.conn.rollback()
            except Exception:
                pass
        print('[brain_batch] WARN: flushed a leaked transaction at entry — an '
              'upstream write did not commit (self.conn was mid-transaction)',
              flush=True)
        try:
            brain._log_error(
                'brain_batch_stale_txn',
                RuntimeError('self.conn was mid-transaction at brain_batch entry'),
                'flushed leaked txn before BEGIN IMMEDIATE; upstream op that ran '
                'just before brain_batch did not commit')
        except Exception:
            pass
    try:
        brain.conn.execute('BEGIN IMMEDIATE')
        transaction_started = True

        for i, op_spec in enumerate(operations):
            if not isinstance(op_spec, dict):
                results.append({"op": "?", "index": i, "ok": False,
                                "error": "operation must be a dict, got %s" % type(op_spec).__name__})
                continue
            op = op_spec.get("op", "")

            # Per-op required-field pre-check, derived from the SAME contract
            # the MCP oneOf schema is built from (contract.BATCH_OP_SPECS) —
            # schema signal at generation time, this check at dispatch time,
            # one source. This is the ONLY missing-field gate for batch ops
            # (the old in-branch guards were removed as unreachable;
            # tests/test_brain_batch_op_contract.py pins enforcement per
            # field). Unknown op names fall through to the invalid-op guard
            # below. revise keeps its rich reason/reasoning disambiguation
            # (_missing_reason_error) instead of the generic message.
            _op_contract = BATCH_OP_SPECS.get(op)
            if _op_contract:
                missing = [f for f in _op_contract["required"]
                           if not op_spec.get(f)]
                if missing:
                    if op == "revise" and "reason" in missing:
                        # Keep the reason/reasoning disambiguation whenever
                        # reason is among the missing fields — not only when
                        # it's the sole one (review 2026-06-12 #2).
                        err = _missing_reason_error(op_spec)
                        others = [f for f in missing if f != "reason"]
                        if others:
                            err += " Also missing: %s." % ", ".join(others)
                    else:
                        err = ("%s op missing required field(s): %s — "
                               "per-op required fields are declared in the "
                               "brain_batch schema, one branch per op"
                               % (op, ", ".join(missing)))
                    results.append({"op": op, "index": i, "ok": False,
                                    "error": err})
                    continue

            try:
                if op == "remember":
                    # Pop per-op connect_to BEFORE handler so it's not processed
                    # eagerly with an empty sibling_map — defer to after the loop.
                    ct_spec = op_spec.get("connect_to")
                    op_args = {k: v for k, v in op_spec.items()
                               if k not in ("op", "connect_to")}
                    if top_encoding_source and "encoding_source" not in op_args:
                        op_args["encoding_source"] = top_encoding_source
                    if top_session_id and CALLER_SESSION_KEY not in op_args:
                        op_args[CALLER_SESSION_KEY] = top_session_id
                    if top_chain_id and "chain_id" not in op_args:
                        op_args["chain_id"] = top_chain_id
                    r = _handle_remember(brain, op_args, graph_changes)
                    _accumulate(r.pop("affected", None))
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
                    if top_session_id and CALLER_SESSION_KEY not in op_args:
                        op_args[CALLER_SESSION_KEY] = top_session_id
                    if top_chain_id and "chain_id" not in op_args:
                        op_args["chain_id"] = top_chain_id
                    r = _handle_revise(brain, op_args, graph_changes)
                    _accumulate(r.pop("affected", None))
                    results.append({"op": "revise", "index": i, **r})

                elif op == "connect":
                    op_args = {k: v for k, v in op_spec.items() if k != "op"}
                    if top_encoding_source and "encoding_source" not in op_args:
                        op_args["encoding_source"] = top_encoding_source
                    if top_session_id and CALLER_SESSION_KEY not in op_args:
                        op_args[CALLER_SESSION_KEY] = top_session_id
                    if top_chain_id and "chain_id" not in op_args:
                        op_args["chain_id"] = top_chain_id
                    r = _handle_connect(brain, op_args, graph_changes)
                    _accumulate(r.pop("affected", None))
                    results.append({"op": "connect", "index": i, **r})

                elif op == "archive":
                    # node_id presence guaranteed by the BATCH_OP_SPECS pre-check.
                    r = _op_archive(brain, op_spec, top_encoding_source, graph_changes)
                    _accumulate(r.pop("affected", None))
                    results.append({"op": "archive", "index": i, **r})

                elif op == "absorb":
                    # id presence guaranteed by the BATCH_OP_SPECS pre-check.
                    r = _op_absorb(brain, op_spec, top_encoding_source, graph_changes)
                    _accumulate(r.pop("affected", None))
                    results.append({"op": "absorb", "index": i, **r})

                elif op == "disconnect":
                    # field presence guaranteed by the BATCH_OP_SPECS pre-check.
                    r = _op_disconnect(brain, op_spec, top_encoding_source,
                                       top_session_id, args, graph_changes)
                    _accumulate(r.pop("affected", None))
                    results.append({"op": "disconnect", "index": i, **r})

                else:
                    # Invalid op name — log loudly. Sonnet sometimes invents
                    # structural names (consolidate/keep/evolve/skip) that were
                    # never valid ops. Previously this returned ok=False and the
                    # caller moved on silently.
                    err_msg = ("Unknown op: %s (use remember, revise, connect, disconnect, archive, absorb)"
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
        connect_to_failed = []  # [{title, reason}]
        connect_to_made = []    # [{src_id, target_id, relation, edge_id, deltas}]
        for src_id, ct_spec in deferred_connects:
            # The src node was created by a remember op in this batch, which
            # inherited top_encoding_source (or 'anchor' for direct MCP) — give
            # its connect_to edges the same provenance.
            r = brain._apply_connect_to(
                src_id, ct_spec, sibling_map=sibling_map,
                encoding_source=top_encoding_source or 'anchor')
            connect_to_edges += len(r['created'])
            connect_to_made.extend(r['created'])  # entries src-tagged by _apply_connect_to
            connect_to_failed.extend(r['failed'])
        if connect_to_edges:
            graph_changes.append("CONNECT_TO: %d edges" % connect_to_edges)
        if connect_to_failed:
            graph_changes.append("CONNECT_TO_FAILURES: %d" % len(connect_to_failed))

        # One commit for the whole batch — all per-op writes land here. Edge
        # traces are emitted AFTER this (post-finally), never before: traces
        # live in a different DB, so emitting pre-commit would orphan them if
        # the batch rolled back.
        brain.conn.commit()
    except Exception as e:
        # Per-op exceptions are caught above and recorded in `results` — reaching
        # this handler means something escaped the per-op guard (programmer bug
        # in a sub-handler, lost DB connection, etc.). Roll back so we don't
        # leave half a batch persisted.
        if transaction_started:
            try:
                brain.conn.rollback()
            except Exception as re:
                try:
                    brain._log_error(
                        'brain_batch_rollback_failed', re,
                        'rollback after batch exception failed')
                except Exception:
                    pass
        try:
            brain._log_error(
                'brain_batch_transaction_failed', e,
                'ops=%d ran_before_fail=%d' % (
                    len(operations), len(results)))
        except Exception:
            pass
        # Re-raise so the dispatcher reports the failure to the caller. The
        # daemon's outer dispatch wraps this in its own try/except and turns
        # it into {ok: False, error: ...}.
        raise
    finally:
        brain.conn.in_batch = False

    # POST-COMMIT: the batch is durable (any exception above re-raised past this
    # point). Emit connect_to edge traces now — failure-isolated per edge, so a
    # trace error logs loudly and can neither roll back the committed graph nor
    # orphan a trace for an edge that didn't persist. (co_anchored edges were
    # already traced inline by each remember op's _handle_remember.)
    _emit_edge_traces(
        brain, connect_to_made, top_encoding_source or '',
        chain_id_override=args.get('chain_id', ''),
        session_id=top_session_id, reason='connect_to')

    succeeded = sum(1 for r in results if r.get("ok"))
    _batch_result = {
        "total": len(operations),
        "succeeded": succeeded,
        "failed": len(operations) - succeeded,
        "connect_to_edges": connect_to_edges,
        "connect_to_failures": len(connect_to_failed),
        "connect_to_failed": connect_to_failed,
        "results": results,
    }
    if project_warnings:
        _batch_result["warnings"] = project_warnings
    return {"ok": True, "result": _batch_result, "affected": _affected(
        created=agg['created'], revised=agg['revised'], archived=agg['archived'])}


def _handle_connect(brain, args, graph_changes):
    # Resolve the edge's CREATOR. The encoder pre-stamps args['encoding_source']
    # at its dispatch boundary (apply_encoder_attribution); a direct-MCP connect
    # arrives with none — and an unstamped write reaching this handler IS anchor
    # by definition (every other writer stamps upstream). So `or 'anchor'` mirrors
    # the node birth default (remember(): `encoding_source or 'anchor'`). This is
    # applied at CREATE only — add_relation preserves encoding_source on an
    # active-row update — so re-connecting an existing edge never relabels it.
    relation = args.get("relation", "related_to")
    encoding_source = args.get("encoding_source") or 'anchor'
    src_id = _resolve_id(brain, args.get("source_id", ""))
    tgt_id = _resolve_id(brain, args.get("target_id", ""))
    result = brain.connect_typed(
        source_id=src_id,
        target_id=tgt_id,
        relation=relation,
        weight=args.get("weight", 0.5),
        description=args.get("description"),
        encoding_source=encoding_source)

    # Emit edge_relation_revised trace event capturing create-or-update deltas.
    if result and (result.get('deltas') or result.get('warnings')):
        _emit_edge_revise_trace(
            brain, result['edge_id'], relation,
            args.get('reason', ''),
            encoding_source or '',
            result.get('deltas', []),
            warnings=result.get('warnings', []),
            chain_id_override=args.get('chain_id', ''),
            session_id=caller_session(args),
            source_id=src_id, target_id=tgt_id,
        )

    graph_changes.append(
        "CONNECT: %s -[%s]-> %s" % (
            args.get("source_id", "?")[:8],
            relation,
            args.get("target_id", "?")[:8]))
    return {"ok": True, "result": {"connected": True}}


def _handle_revise_edge(brain, args, graph_changes):
    """Revise an existing edge relation in place — rename and/or update
    description/weight. Identify by (source_id, target_id, relation); omitted
    fields preserve (mirrors revise()). Routes the rename to the in-place
    GraphDAL.rename_relation primitive, not a connect+disconnect pair."""
    source_id = _resolve_id(brain, args.get("source_id", ""))
    target_id = _resolve_id(brain, args.get("target_id", ""))
    relation = args.get("relation")
    if not (source_id and target_id and relation):
        return {"ok": False, "error": "source_id, target_id, relation are required"}

    result = brain.revise_edge(
        source_id=source_id, target_id=target_id, relation=relation,
        new_relation=args.get("new_relation"),
        description=args.get("description"),
        weight=args.get("weight"),
        encoding_source=args.get("encoding_source"),
        reason=args.get("reason", ""))
    if not result.get("ok"):
        return result

    if result.get("deltas"):
        _emit_edge_revise_trace(
            brain, result["edge_id"], result["relation"],
            args.get("reason", ""), args.get("encoding_source") or "",
            result.get("deltas", []),
            chain_id_override=args.get("chain_id", ""),
            session_id=caller_session(args),
            source_id=source_id, target_id=target_id)

    graph_changes.append(
        "REVISE_EDGE: %s -[%s]-> %s" % (
            str(args.get("source_id", "?"))[:8],
            result["relation"],
            str(args.get("target_id", "?"))[:8]))
    return {"ok": True, "result": result}


def _handle_connect_batch(brain, args, graph_changes):
    """Create multiple edges in one call."""
    connections = args.get("connections", [])
    if not connections:
        return {"ok": False, "error": "connections array is required"}

    chain_id_override = args.get('chain_id', '')
    session_id = caller_session(args)
    top_encoding_source = args.get('encoding_source', '')

    # Known per-connection keys — used to build a "did you mean" hint when a
    # caller sends an aliased key (e.g. from_id/to_id) instead of the canonical
    # source_id/target_id. We keep ONE canonical name (no aliasing) but make the
    # error self-correcting.
    KNOWN_CONN_KEYS = {"source_id", "target_id", "relation", "weight",
                       "description", "encoding_source", "reason"}

    created = 0
    failure_details = []  # [{source_id, target_id, relation, reason}]
    for c in connections:
        relation = c.get("relation", "related_to")
        src_raw = c.get("source_id", "")
        tgt_raw = c.get("target_id", "")
        # Self-correcting guard: if an endpoint is missing AND the caller sent
        # keys we don't recognize, the reason names them and suggests the fix.
        if not src_raw or not tgt_raw:
            unknown = [k for k in c if k not in KNOWN_CONN_KEYS]
            hint = ""
            if unknown:
                hint = (" — unrecognized key(s) %s; edges use 'source_id' and "
                        "'target_id'" % unknown)
            failure_details.append({
                "source_id": str(src_raw)[:8], "target_id": str(tgt_raw)[:8],
                "relation": relation,
                "reason": "missing source_id or target_id%s" % hint})
            brain._log_error(
                'connect_batch_missing_endpoint',
                ValueError("missing source_id/target_id%s" % hint),
                'keys=%s' % sorted(c.keys()))
            continue
        try:
            # Resolve the edge's CREATOR (per-row → batch-level → 'anchor'),
            # mirroring _handle_connect. Applied at CREATE only — add_relation
            # preserves encoding_source on an active-row update — so a re-connect
            # never relabels an existing edge.
            encoding_source = (c.get("encoding_source")
                               or top_encoding_source or 'anchor')
            src_id = _resolve_id(brain, src_raw)
            tgt_id = _resolve_id(brain, tgt_raw)
            result = brain.connect_typed(
                source_id=src_id,
                target_id=tgt_id,
                relation=relation,
                weight=c.get("weight", 0.5),
                description=c.get("description"),
                encoding_source=encoding_source)
            created += 1

            # Emit one edge_relation_revised trace per row that actually changed.
            if result and (result.get('deltas') or result.get('warnings')):
                _emit_edge_revise_trace(
                    brain, result['edge_id'], relation,
                    c.get('reason', '') or args.get('reason', ''),
                    encoding_source or '',
                    result.get('deltas', []),
                    warnings=result.get('warnings', []),
                    chain_id_override=chain_id_override,
                    session_id=session_id,
                    source_id=src_id, target_id=tgt_id,
                )
        except Exception as e:
            # No silent drops: a failed edge in a batch must surface, not vanish
            # — and now with its REASON in the response, not just the dashboard.
            failure_details.append({
                "source_id": str(src_raw)[:8], "target_id": str(tgt_raw)[:8],
                "relation": relation, "reason": str(e)[:160]})
            brain._log_error(
                'connect_batch_edge_failed', e,
                '%s -[%s]-> %s' % (str(src_raw)[:8], relation, str(tgt_raw)[:8]))
    graph_changes.append("CONNECT_BATCH: %d edges" % created)
    if failure_details:
        graph_changes.append("CONNECT_BATCH_FAILURES: %d" % len(failure_details))
    return {"ok": True, "result": {
        "edges_created": created,
        "failures": len(failure_details),
        "failure_details": failure_details,
    }}


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
