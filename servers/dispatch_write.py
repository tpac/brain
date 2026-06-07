"""Daemon dispatch — write handlers (mutations).

remember / revise / connect / enrich / brain_batch and their helpers:
source_refs validation, and the revise/edge trace emitters. Every write goes
through here.
"""

import re
import sys

from .clock import brain_today
from .dispatch_common import _resolve_id, _pop_session_ctx


_HEX_TRACE_ID_RE = re.compile(r'^[0-9a-f]{8}$')


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
    """
    warnings = warnings or []
    if not deltas and not warnings:
        return  # nothing to record

    from .trace_contract import build_revise_metadata

    scale, chain_id = _infer_scale_and_chain(brain, encoding_source, chain_id_override)

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

    scale, chain_id = _infer_scale_and_chain(brain, encoding_source, chain_id_override)

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


def _handle_remember(brain, args, graph_changes):
    from .contract import validate_field

    # session_id is a control field — strip BEFORE validation/passthrough so
    # it doesn't land in node_metadata_kv as silent metadata. Loads ctx for
    # per-session record_remember + add_to_segment routing inside remember().
    ctx, args = _pop_session_ctx(brain, args)

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

    # Pass ALL fields through to remember() — contract fields go to nodes table,
    # promoted fields to metadata, everything else to node_metadata_kv as extra_fields.
    # Don't filter — remember() handles routing via **extra_fields kwargs.
    remember_args = {k: v for k, v in args.items() if v is not None}
    result = brain.remember(**remember_args, ctx=ctx)
    # ctx is cached on Brain; remember's record_remember mutation persists
    # via the autosave loop (no per-call save).
    node_id = result.get("id", "?")[:8] if isinstance(result, dict) else "?"
    graph_changes.append(
        "REMEMBER: [%s] %s (%s...)" % (
            args.get("type", "?"), args.get("title", "")[:50], node_id))
    return {"ok": True, "result": result}


def _handle_remember_batch(brain, args, graph_changes):
    from .contract import validate_field, get_remember_fields

    # Pop session_id BEFORE spec scrubbing so per-node specs don't inherit
    # control fields. Also strip from each spec defensively in case the
    # caller bundled it inside a node spec.
    ctx, args = _pop_session_ctx(brain, args)

    nodes = args.get("nodes", [])
    if not nodes:
        return {"ok": False, "error": "nodes array is required"}

    accepted_fields = set(get_remember_fields().keys())
    # Inherit top-level encoding_source into each node (dispatch wrapper injects this)
    top_encoding_source = args.get("encoding_source")

    cleaned_nodes = []
    for i, spec in enumerate(nodes):
        spec.pop('session_id', None)  # defensive: not a node field
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

    # 2026-05-24: auto_connect param removed from remember_batch — the
    # pairwise `related_to` auto-connect it triggered was the source of
    # empty-description `related_to` pollution every encoding cycle. We
    # accept `auto_connect` in args silently (legacy callers may still send
    # it) but drop it before forwarding.
    result = brain.remember_batch(
        nodes=cleaned_nodes,
        connect_to=args.get("connect_to"),
        ctx=ctx)
    # ctx mutations persist via autosave (no per-call save).
    graph_changes.append("REMEMBER_BATCH: %d nodes" % result.get("nodes_created", 0))
    return {"ok": True, "result": result}


def _handle_revise(brain, args, graph_changes):
    """Update any field(s) on an existing node via revise()."""
    from .contract import validate_field

    node_id = _resolve_id(brain, args.get("node_id", ""))
    reason = args.get("reason", "")
    if not node_id:
        return {"ok": False, "error": "node_id is required"}
    if not reason:
        return {"ok": False, "error": "reason is required"}

    # Reserve known dispatch keys so they don't get treated as field updates.
    DISPATCH_KEYS = {"node_id", "reason", "encoding_source", "chain_id", "session_id"}
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

    # Valid nested op names live in contract.VALID_BATCH_OPS (single source of
    # truth — see that constant). The dispatcher matches them via the if/elif
    # chain below; the final `else` is the invalid-op guard. Sonnet has been
    # observed inventing structural op names like `consolidate` / `keep` /
    # `evolve` / `skip` / `reject` — these are prompt-level
    # DECISIONS, not dispatch ops. They land in the "Unknown op" branch, get
    # logged loudly (brain_batch_invalid_op), and the S2 rejection_table now
    # detects them so a dropped op isn't mistaken for a clean SKIP.

    top_encoding_source = args.get("encoding_source")
    # Propagate session_id to each op so sub-handlers can load ctx per op.
    # (Each sub-handler will pop session_id and load its own ctx; cheap
    # session_state reads + writes, acceptable for a batch of 1-10 ops.)
    top_session_id = args.get("session_id", "")
    results = []

    # Sibling-aware connect_to: defer per-op connect_to from `remember` ops
    # until ALL ops in this batch have run, so siblings declared in any order
    # can resolve. NEW wins on title collision (sibling beats catalog).
    sibling_map = {}  # lowercased title → new node_id
    deferred_connects = []  # [(src_node_id, connect_to_spec)]

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
            try:
                if op == "remember":
                    # Pop per-op connect_to BEFORE handler so it's not processed
                    # eagerly with an empty sibling_map — defer to after the loop.
                    ct_spec = op_spec.get("connect_to")
                    op_args = {k: v for k, v in op_spec.items()
                               if k not in ("op", "connect_to")}
                    if top_encoding_source and "encoding_source" not in op_args:
                        op_args["encoding_source"] = top_encoding_source
                    if top_session_id and "session_id" not in op_args:
                        op_args["session_id"] = top_session_id
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

                elif op == "absorb":
                    # Lossless merge: fold absorbed INTO survivor, then archive
                    # absorbed. Transfer-by-default (source_refs, edges,
                    # access_count, KV) so a merge can't silently drop info the
                    # imperative revise+connect+archive path did (node 988de522).
                    # survivor may be locked; absorbed must not be.
                    survivor_id = op_spec.get("survivor_id")
                    absorbed_id = op_spec.get("absorbed_id")
                    if not (survivor_id and absorbed_id):
                        results.append({"op": "absorb", "index": i, "ok": False,
                                        "error": "survivor_id and absorbed_id are required"})
                    else:
                        archived_by = op_spec.get('encoding_source') or \
                            op_spec.get('archived_by') or \
                            top_encoding_source or 'unknown'
                        # Revise-op style: every non-control key is a survivor
                        # field override (content, title, confidence, situation,
                        # ...), forwarded to absorb()'s updates. Same mental
                        # model as the revise op — an agent writes the same shape.
                        _CONTROL = {'op', 'survivor_id', 'absorbed_id',
                                    'prune_edges', 'drop_fields', 'archived_by',
                                    'encoding_source', 'reason',
                                    'session_id', 'chain_id'}
                        field_updates = {k: v for k, v in op_spec.items()
                                         if k not in _CONTROL}
                        r = brain.absorb(
                            survivor_id, absorbed_id,
                            updates=field_updates or None,
                            archived_by=archived_by,
                            reason=op_spec.get('reason', ''),
                            prune_edges=op_spec.get('prune_edges'),
                            drop_fields=op_spec.get('drop_fields'))
                        if r.get('ok'):
                            graph_changes.append("ABSORB: %s <- %s" % (
                                survivor_id[:8], absorbed_id[:8]))
                        results.append({"op": "absorb", "index": i, **r})

                elif op == "disconnect":
                    # Soft-archive a specific relation on an edge. Other relations
                    # on the same edge survive. v25 — archived row preserved for
                    # forensics/recovery; reads filter via JOIN.
                    # Lets ABSORB encoders prune survivor edges that don't fit
                    # the new framing after revise.
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
                        gdal = brain._graph
                        edge_id = gdal.get_edge_id(source_id, target_id)
                        # remove_relation gates its own commit on conn.in_batch
                        # (True here) → deferred to the batch's single COMMIT.
                        gdal.remove_relation(
                            source_id, target_id, relation, archived_by=archived_by)
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

        # One commit for the whole batch — all per-op writes land here.
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

    succeeded = sum(1 for r in results if r.get("ok"))
    return {"ok": True, "result": {
        "total": len(operations),
        "succeeded": succeeded,
        "failed": len(operations) - succeeded,
        "connect_to_edges": connect_to_edges,
        "connect_to_failures": connect_to_failures,
        "results": results,
    }}


def _handle_connect(brain, args, graph_changes):
    # Stage 1B: pass description/encoding_source through only when caller
    # specified them. None preserves existing on update (idempotent upsert).
    relation = args.get("relation", "related_to")
    encoding_source = args.get("encoding_source")  # None = preserve on update
    result = brain.connect_typed(
        source_id=_resolve_id(brain, args.get("source_id", "")),
        target_id=_resolve_id(brain, args.get("target_id", "")),
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
            session_id=args.get('session_id', ''),
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
            session_id=args.get("session_id", ""))

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
    session_id = args.get('session_id', '')
    top_encoding_source = args.get('encoding_source', '')

    created = 0
    failures = 0
    for c in connections:
        try:
            # Stage 1B: pass description/encoding_source through only when
            # specified (None → preserve existing on update).
            relation = c.get("relation", "related_to")
            encoding_source = c.get("encoding_source") or top_encoding_source or None
            result = brain.connect_typed(
                source_id=_resolve_id(brain, c.get("source_id", "")),
                target_id=_resolve_id(brain, c.get("target_id", "")),
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
                )
        except Exception as e:
            # No silent drops: a failed edge in a batch must surface, not vanish.
            failures += 1
            brain._log_error(
                'connect_batch_edge_failed', e,
                '%s -[%s]-> %s' % (
                    str(c.get('source_id', '?'))[:8],
                    c.get('relation', 'related_to'),
                    str(c.get('target_id', '?'))[:8]))
    graph_changes.append("CONNECT_BATCH: %d edges" % created)
    if failures:
        graph_changes.append("CONNECT_BATCH_FAILURES: %d" % failures)
    return {"ok": True, "result": {"edges_created": created, "failures": failures}}


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
