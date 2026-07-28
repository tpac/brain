"""Integration Unit — universal contract for O/K/Δ units at any scale.

Every integration unit declares what it reads (O_SOURCES, K_SOURCES),
implements run() which produces Δ, and writes its own traces.

The contract is what it reads and what it produces, not how it's
structured internally. Units are free to organize their run() however
makes sense for the operation.

## Shared S2 methods

S2 units share common infrastructure for:
- Reading S1 traces (what changed since last run)
- Deciding whether to run (_should_run)
- Calling LLM with learnable prompts from interactions table
- JSON extraction from LLM responses

These live here so every S2 unit (community, dedup, confidence, etc.)
uses the same patterns. S3 can tune any unit's LLM behavior by
revising its interaction entry.
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta


# Single shared Anthropic client timeout — see scales/runner.py.
# read_usage / sum_usage: the canonical token-telemetry contract shared with
# run_llm_loop, so _call_llm and _accumulate_run don't re-derive it. read_usage
# + sum_usage are also re-exported here for the encoder subclasses (e.g. healer).
from ..runner import ANTHROPIC_CLIENT_TIMEOUT, read_usage, sum_usage  # noqa: F401 — re-export for callers
from ..dispatch import ATTRIBUTED_WRITE_COMMANDS, stamp_project_provenance


# Writes whose handlers emit chain-bearing traces (node_revised /
# edge_relation_revised). They must carry the unit's run chain_id so those
# traces join the run's chain instead of dispatch_write's date-based
# '{scale}-{date}-revise' fallback — the phantom "S2 revise" / "S1 revise" unit.
# Derived from the attributed writes + revise_edge (one source, so a new
# attributed write becomes chain-aware for free — the house pattern).
CHAIN_AWARE_WRITES = ATTRIBUTED_WRITE_COMMANDS | {'revise_edge'}


def apply_encoder_attribution(cmd, cmd_args, *, encoding_source, run_chain_id,
                              project=None):
    """Stamp scale-agent attribution onto an outgoing write's args — the single
    chokepoint these facts flow through:

    - ``encoding_source`` on attributed writes (who minted the node/edge),
    - ``run_chain_id`` on chain-bearing writes (which run's chain its trace
      joins),
    - ``project`` provenance policy (stamp_project_provenance semantics):
      the Scribe passes its session's project ('' for a non-repo session) so
      node-creating ops carry deterministic provenance; S2 units pass ''
      (graph-scope work never invents provenance); None leaves args untouched.

    ``setdefault`` for the first two, so an explicitly-supplied value wins.
    ``project`` is the opposite — force/strip — because it is session-derived,
    never agent-authored. Mutates ``cmd_args`` in place; a no-op on non-dict
    args and on reads (get_nodes / recall_batch are in neither set — they
    carry no attribution, and a chain_id on a read is meaningless). The
    in-process encoder dispatch is the one factory S1 Scribe and the S2 units
    share, so the attribution rules live here, once.

    Returns: warning strings from the project stamp (dropped/overridden
    agent-supplied values) for the caller to log.
    """
    if not isinstance(cmd_args, dict):
        return []
    if cmd in ATTRIBUTED_WRITE_COMMANDS:
        cmd_args.setdefault('encoding_source', encoding_source)
    if run_chain_id and cmd in CHAIN_AWARE_WRITES:
        cmd_args.setdefault('chain_id', run_chain_id)
    return stamp_project_provenance(cmd, cmd_args, project)


# Per-batch retry policy for transient API errors. The SDK's built-in
# max_retries handles pre-stream failures (connect refused, 5xx before
# body, rate-limit). It can't retry once a stream has started and stalls
# mid-body (httpx ReadTimeout) — that's what this wrapper covers.
#
# attempts=2 means one retry. First failure => 8s backoff then retry.
# Second failure => give up, batch fails, work moves on. This caps the
# wall-clock cost of a stuck batch at ~2*timeout + backoff and recovers
# the happy-path on transient blips.
RETRY_ATTEMPTS = 2
RETRY_BACKOFF_BASE_S = 8.0


def retry_on_transient_api_error(fn, *, attempts=RETRY_ATTEMPTS,
                                 base_backoff_s=RETRY_BACKOFF_BASE_S,
                                 log_fn=None):
    """Call fn() with retry on transient Anthropic SDK exceptions.

    Retries on: APITimeoutError, APIConnectionError, InternalServerError
    (5xx from Anthropic). Also catches httpx TimeoutException as a safety
    net in case a raw httpx error leaks through streaming.

    Does NOT retry on: BadRequestError, AuthenticationError, PermissionDenied,
    NotFoundError, UnprocessableEntityError, RateLimitError. Those either
    indicate a client bug (retry won't help) or are already handled by the
    SDK's built-in max_retries (rate limit respects Retry-After header).

    Args:
        fn: zero-arg callable that makes the API call
        attempts: total attempts including the first call; 2 = one retry
        base_backoff_s: seconds to wait before first retry; doubles each attempt
        log_fn: optional logger invoked on retry with a one-line message

    Returns: whatever fn() returns

    Raises: the last exception if all attempts fail (transient), or the
        original exception immediately if non-transient
    """
    import anthropic
    try:
        import httpx
        httpx_timeout = (httpx.TimeoutException,)
    except Exception:
        httpx_timeout = ()

    transient = (
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    ) + httpx_timeout

    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except transient as e:
            last_err = e
            if i < attempts - 1:
                sleep_s = base_backoff_s * (2 ** i)
                if log_fn:
                    log_fn('transient API error (%s): %s — retrying in %.0fs '
                           '(attempt %d/%d)' % (
                               type(e).__name__, e, sleep_s, i + 2, attempts))
                time.sleep(sleep_s)
    # Exhausted retries — re-raise the last error so callers can log + skip
    raise last_err


class IntegrationUnit:
    """Base contract for all integration units at any scale."""

    # Subclass must define:
    NAME = ''                    # e.g. 'community_detection'
    SCALE = ''                   # e.g. 's2'
    ENCODING_SOURCE = ''         # e.g. 's2:community_detection'

    # O/K contract — what this unit reads and what shapes its decisions
    O_SOURCES = []               # e.g. ['graph_nodes', 'graph_edges']
    K_SOURCES = []               # e.g. ['leidenalg', 'resolution_param']

    def __init__(self, brain, dispatch_fn=None):
        """Initialize with brain instance and optional dispatch.

        Args:
            brain: Brain instance — the daemon's brain. Units run in-process
                   with direct DB access (writes serialize under
                   brain.write_lock via _make_encoder_dispatch).
            dispatch_fn: Optional pre-built dispatch. None (the norm) means
                         build the in-process encoder dispatch in
                         _make_encoder_dispatch.
        """
        self.brain = brain
        self.dispatch = dispatch_fn

    def run(self):
        """Execute the unit. Returns a result dict describing the delta.

        Must be implemented by subclass. The result dict should include
        at minimum: {actions: int, details: [...]}
        """
        raise NotImplementedError('%s.run() not implemented' % type(self).__name__)

    def chain_id(self):
        """Generate trace chain ID for this run.

        Format: {scale}-{YYYYMMDDHHMMSS}-{name}
        S2 chains are time-based (not session-based) and stamped ONCE per
        run, then cached: every trace() in this run shares one chain_id,
        while two runs of the same unit get distinct ids. Seconds (not just
        date) make notes groupable per-run — date-only collapsed every
        same-day run of a unit onto one id, breaking "last K runs"
        continuity. A unit can't run twice within one second (min-interval
        gating is in minutes), so seconds is collision-free.

        Two parts are LOAD-BEARING — do not split the timestamp into its own
        dash-segment: `_last_run_timestamp` suffix-matches `LIKE '%-{name}'`
        and the dashboard's `_unit_slug_from_chain` reads `split('-', 2)[2]`
        as the unit slug. One combined timestamp segment + trailing `-{name}`
        keeps both intact.
        """
        if not getattr(self, '_chain_id', None):
            self._chain_id = '%s-%s-%s' % (
                self.SCALE,
                datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S'),  # clock-ok — S2 idle-cycle chain id; UTC to align with created_at + avoid DST-rollback second repeats
                self.NAME)
        return self._chain_id

    def project_policy(self):
        """Project-provenance policy for this unit's writes (see
        stamp_project_provenance). S2 units work at graph scope — they never
        invent provenance, so the default is '' (authoritative strip: any
        agent-supplied project is dropped). Session-scoped units (S1 Scribe)
        override with their session's derived project."""
        return ''

    def trace(self, event_type, ref_type, summary, ref_id='', metadata=None):
        """Write a trace event for this unit's current run.

        Uses direct TraceDAL when running inline (dispatch is None).
        Uses dispatch('trace_append', ...) when running in background.
        """
        # Loud-at-the-write-boundary: an LLM-encoder delta that ran + did work
        # but recorded no output tokens means telemetry wasn't threaded into the
        # delta. Every S2 unit writes its delta through here, so a NEW unit that
        # forgets telemetry is caught for free — the recurrence guard for the
        # 2026-06-24 S2 telemetry gap. The detector is the pure contract function
        # (no logger); logging lives here, where the brain is reachable (the
        # other chokepoint, TraceDAL.append, can't write the errors table
        # mid-append without risking the brain_batch commit).
        if event_type == 'delta' and metadata is not None:
            from servers.trace_contract import check_delta_telemetry
            warn = check_delta_telemetry(ref_type, metadata)
            if warn:
                try:
                    self.brain._log_error(
                        's2_%s_telemetry_gap' % self.NAME, ValueError(warn),
                        'delta trace missing LLM telemetry')
                except Exception:
                    pass

        trace_data = {
            'chain_id': self.chain_id(),
            'scale': self.SCALE,
            'event_type': event_type,
            'ref_type': ref_type,
            'ref_id': ref_id,
            'summary': summary[:200] if summary else '',
            'metadata': metadata,
            'session_id': '',
        }

        if self.dispatch:
            self.dispatch('trace_append', trace_data)
        else:
            self.brain._trace_dal.append(**trace_data)

    # ── Shared S2 infrastructure ──

    def _load_journal_notes_prefix(self):
        """Residue continuity (the journal): render the last K note-bearing
        runs' notes into a prompt prefix. The read counterpart to the encoder's
        write_journal_notes(). Scoped by this unit (scale + NAME); K defaults
        from JOURNAL_CONTINUITY_RUNS. Empty string when there are no recent
        notes — a clean history adds nothing.
        """
        from servers.trace_contract import render_journal_notes_prefix
        # Failure-isolated, mirroring write_journal_notes: a transient logs.db
        # read error must never abort an otherwise-valid encode — degrade to
        # no continuity, log loud.
        try:
            notes = self.brain.journal_notes(scale=self.SCALE, unit=self.NAME)
            return render_journal_notes_prefix(notes)
        except Exception as e:
            self.brain._log_error('s2_%s_journal_read' % self.NAME, e,
                                  'residue continuity read failed — encoding without it')
            return ''

    def _inject_edge_aspects(self, system_prompt):
        """Append the edge-relation aspect vocabulary so the encoder picks
        specific relations over generic ones. Shared single source across
        S2/S1 encoders — same inject pattern as _inject_review_block. The block
        (skip set + heading + render) lives in servers.aspects; here we just
        feed it brain.aspects and append. No-op append when there's nothing
        to show.
        """
        from servers.aspects import render_edge_aspects_block
        block = render_edge_aspects_block(self.brain.aspects.all())
        return system_prompt.rstrip() + ('\n\n' + block if block else '')

    def _inject_review_block(self, system_prompt):
        """Append the shared residue-review block — the WRITE-side counterpart
        to _load_journal_notes_prefix. The block is single-sourced in
        trace_contract (`render_journal_review_block`) and never baked into the
        registered prompt, so it iterates in one place and every encoder on the
        note contract gets it live. Concern = the review CONTENT only; the
        terminal-turn/DONE closure is a SEPARATE inject (`_append_closure`), so
        removing or relocating the review never drags the closure with it.
        """
        from servers.trace_contract import render_journal_review_block
        return system_prompt.rstrip() + "\n\n" + render_journal_review_block()

    def _append_closure(self, system_prompt):
        """Append the run's CLOSURE (terminal-turn definition + `## Review`
        placement + DONE) as the LAST block of the prompt. Single-sourced in
        trace_contract (`render_prompt_closure`), independent of the review
        block. Call this AFTER all other prompt assembly so DONE is genuinely
        last.
        """
        from servers.trace_contract import render_prompt_closure
        return system_prompt.rstrip() + "\n\n" + render_prompt_closure()

    def _make_encoder_dispatch(self, archive_guard=None):
        """Build the dispatch function S2 encoders use for brain_batch calls.

        Shared across consolidation/community/future encoders. Sets
        `encoding_source` at the top level (handler cascades to each op).
        Vectors are filled asynchronously by the embed_queue worker, so the
        encoder never embeds inline (the old `skip_embedding` ONNX spin guard
        is gone — embedding moved off the write path).

        Args:
            archive_guard: Optional set of node IDs that archive ops are
                allowed to target this batch. When set, archives of nodes
                outside the set are dropped + logged. Used by consolidation
                to prevent the encoder archiving nodes outside its cluster.
                None → no guard.

        Returns: dispatch(cmd, cmd_args) → handler result.
        """
        if self.dispatch:
            return self.dispatch

        from servers.daemon_dispatch import COMMAND_TABLE, check_unknown_keys

        brain = self.brain
        encoding_source = self.ENCODING_SOURCE
        unit_name = self.NAME
        # The run chain — s1e-{session}-{stop} for the Scribe, s2-{ts}-{unit}
        # for S2 units — stamped on writes so their revise/edge traces join THIS
        # run instead of the date-fallback chain. Computed once; chain_id() is
        # cached on the unit.
        run_chain_id = self.chain_id()
        # Project provenance policy — the Scribe stamps its session's project,
        # S2 units strip ('' — graph-scope work never invents provenance).
        # Computed once per dispatch build.
        project_policy = self.project_policy()

        def dispatch(cmd, cmd_args):
            _proj_warnings = apply_encoder_attribution(
                cmd, cmd_args,
                encoding_source=encoding_source, run_chain_id=run_chain_id,
                project=project_policy)
            for _w in _proj_warnings:
                # encoder drift, not a failure — warning severity so the
                # error feed stays real errors only
                brain._log_warning('project_provenance_stamp', _w,
                                   'unit=%s' % unit_name)

            if cmd == 'brain_batch' and isinstance(cmd_args, dict):
                if archive_guard is not None:
                    surviving_ops = []
                    for op in cmd_args.get('operations', []):
                        if isinstance(op, dict) and op.get('op') == 'archive':
                            nid = op.get('node_id') or op.get('id')
                            if nid and nid not in archive_guard:
                                brain._log_error(
                                    's2_%s_out_of_scope_archive' % unit_name,
                                    ValueError(
                                        'archive op for %s not in allowed set'
                                        % nid),
                                    'allowed=%d ids; rejected archive (encoder drift)'
                                    % len(archive_guard))
                                continue
                        surviving_ops.append(op)
                    cmd_args['operations'] = surviving_ops

            entry = COMMAND_TABLE.get(cmd)
            if entry:
                check_unknown_keys(cmd, entry, cmd_args, brain)
                # Serialize against daemon dispatch + autosave + embed_queue.
                # S2 runs in-process but on a pool worker thread; without
                # this, encoder writes can interleave with concurrent
                # client writes (and with each other across S2 units).
                with brain.write_lock:
                    return entry.handler(brain, cmd_args, [])
            return {'ok': False, 'error': 'Unknown command: %s' % cmd}

        return dispatch

    def _last_run_timestamp(self):
        """Find this unit's most recent completed run timestamp.

        Returns ISO timestamp string, or '' if never run. Counts only REAL
        integration deltas — residue (journal_note) is excluded, so a run that
        only journaled doesn't read as a completed integration. Goes through the
        trace API (query_traces), not raw SQL; chain_suffix scopes to this unit's
        chains and now escapes the '_' in names like community_detection.
        """
        from ...trace_contract import RESIDUE_REF_TYPES
        try:
            res = self.brain.query_traces(
                scale=self.SCALE, event_type='delta', chain_suffix=self.NAME,
                exclude_ref_types=list(RESIDUE_REF_TYPES), hours=None, limit=1)
            events = res.get('events', [])
            return events[0]['created_at'] if events else ''
        except Exception as e:
            # Log to brain errors (not just stderr) so repeated failures
            # surface via consciousness. A broken _last_run_timestamp makes
            # every cold-start decision wrong — keep it visible.
            try:
                self.brain._log_error(
                    's2_%s_last_run_timestamp' % self.NAME, e,
                    'scale=%s name=%s' % (self.SCALE, self.NAME))
            except Exception:
                import sys
                print('[%s] _last_run_timestamp error: %s' % (self.NAME, e),
                      file=sys.stderr)
            return ''

    def _has_new_traces(self, scale, ref_type=None):
        """Check if there are traces newer than this unit's last run.

        Generic: any S2 unit can check any scale's traces.
        Returns True on first run (no prior traces).

        Args:
            scale: Scale to check (e.g. 's1', 's2')
            ref_type: Optional filter (e.g. 'encoding_run')
        """
        last_ts = self._last_run_timestamp()
        if not last_ts:
            return True  # Never run — cold start

        try:
            if ref_type:
                traces = self.brain.query_traces(
                    ref_type=ref_type, scale=scale, hours=168, limit=5)['events']
            else:
                traces = self.brain.query_traces(
                    scale=scale, event_type='delta', limit=5)['events']
            for t in traces:
                if t.get('created_at', '') > last_ts:
                    return True
        except Exception as e:
            # Fail-open to "run" is safer than stuck-idle, but log the
            # underlying cause so the operator sees the trace-read outage
            # instead of a silently-retriggering unit.
            try:
                self.brain._log_error(
                    's2_%s_has_new_traces' % self.NAME, e,
                    'trace read failed; falling back to run=True')
            except Exception:
                pass
            return True

        return False

    def _read_traces_since(self, scale, since_ts='', hours=168, ref_types=None):
        """Read trace events from a scale since a timestamp.

        Generic trace reader — each unit interprets the results
        according to what it needs.

        Args:
            scale: Scale to read ('s1', 's2', etc.)
            since_ts: ISO timestamp. If empty, reads all available (cold start).
            hours: Lookback window. Ignored if since_ts is empty (uses max).
            ref_types: Optional list of ref_types to filter. If None, reads all.

        Returns:
            List of trace event dicts, filtered to after since_ts.
        """
        if not since_ts:
            hours = 8760  # ~1 year for cold start

        results = []
        if ref_types:
            for rt in ref_types:
                results.extend(self.brain.query_traces(
                    ref_type=rt, scale=scale, hours=hours,
                    limit=500)['events'])
        else:
            results = self.brain.query_traces(
                scale=scale, hours=hours, limit=500)['events']

        if since_ts:
            results = [t for t in results if t.get('created_at', '') > since_ts]

        return results

    def _get_interaction_config(self, name):
        """Get config dict from interactions table.

        Returns {} if not found. Config is the 'parameters' JSON field.
        """
        return self.brain.get_interaction_config(name)

    def _accumulate_run(self, total, result):
        """Fold one run_llm_loop result batch into `total` (in place): the loop
        counts (rounds / actions / write_actions), the per-tool records
        (action_details + read_calls — both carry per-tool latency/result_count),
        and token telemetry (via the shared sum_usage). The single multi-batch
        accumulator, shared by the consolidation and community encoders so the
        batch-fold loop lives in one place.

        Per-unit fields (final_text formatting, unit-specific counters) stay in
        each encoder. elapsed_ms is NOT summed — each encoder wall-clocks its
        whole batch loop once. `total` must pre-init action_details/read_calls
        to []."""
        total['rounds'] += result.get('rounds', 0)
        total['actions'] += result.get('actions', 0)
        total['write_actions'] += result.get('write_actions', 0)
        total['action_details'].extend(result.get('action_details') or [])
        total['read_calls'].extend(result.get('read_calls') or [])
        sum_usage(total, result)

    def _fold_batch_result(self, total, result, batch_num, trunc_source,
                           trunc_detail='tool call likely corrupted'):
        """Process one batch's run_llm_loop result for a multi-batch encoder:
        accumulate it into `total` (via _accumulate_run), append this batch's
        final_text, persist its review notes as journal_note rows PER BATCH, and
        log any max_tokens truncation. The shared per-batch body for the
        consolidation + community encoders, so the loop logic lives in one place;
        per-unit steps (state refresh, progress traces) stay in each loop.

        Per-batch journal write (not post-loop): extract_review_block keys on the
        FIRST `## Review` fence, so a single post-loop write over the accumulated
        final_text would drop every batch's notes but the first. Writing per
        batch, all sharing this run's chain_id, groups them as one run's notes.
        The journal write is failure-isolated by the caller's per-batch
        try/except (a journal hiccup never aborts the run)."""
        self._accumulate_run(total, result)
        batch_text = result.get('final_text', '')
        if batch_text:
            total['final_text'] += '\n--- batch %d ---\n%s' % (batch_num, batch_text)
            self.brain.write_journal_notes(
                final_text=batch_text, chain_id=self.chain_id(),
                scale=self.SCALE, session_id='')
        for trunc in result.get('truncations', []):
            self.brain._log_error(
                trunc_source,
                'max_tokens truncation: round %d used %s/%s output tokens' % (
                    trunc['round'], trunc['output_tokens'], trunc['max_tokens']),
                'batch %d — %s' % (batch_num, trunc_detail))

    def _call_llm(self, interaction_name, user_content):
        """Call LLM with a learnable prompt from interactions table.

        Loads system prompt from interaction template.
        Loads config (model, max_tokens) from interaction parameters.
        Handles JSON extraction from response.

        Args:
            interaction_name: Key in interactions table (e.g. 's2_community_enrichment')
            user_content: String content for the user message

        Returns:
            (parsed_json, telemetry): parsed_json is the JSON parsed from the
            LLM response (None on failure); telemetry is a dict
            {elapsed_ms, input_tokens, output_tokens, cache_read_tokens,
            cache_creation_tokens} for the call — zeros when the call never ran
            or failed before a usage report. Single-shot encoders (healer,
            aspect) thread this into their delta trace via build_delta_metadata,
            so their production deltas stop recording elapsed_ms=0/output_tokens=0
            (this is run_llm_loop's per-call telemetry, hand-built here because
            this path uses a plain messages.create, not the loop).
        """
        import anthropic
        from ..dispatch import load_env

        # Load learnable prompt and config
        system_prompt = self.brain.get_interaction_prompt(interaction_name)
        config = self._get_interaction_config(interaction_name)
        model = config.get('model', 'claude-haiku-4-5')
        max_tokens = config.get('max_tokens', 4096)

        # read_usage(None) is the all-zero token baseline — reused on the
        # no-prompt early return and the pre-usage failure path.
        telemetry = {'elapsed_ms': 0, **read_usage(None)}

        if not system_prompt:
            print('[%s] WARNING: no interaction prompt for %s' % (
                self.NAME, interaction_name), flush=True)
            # Loud: a missing prompt means this unit silently does nothing
            # every cycle — surface it in the errors table, not just stdout.
            self.brain._log_error(
                '%s_missing_prompt' % self.NAME,
                RuntimeError('no interaction prompt for %s' % interaction_name),
                'unit cannot run its LLM step')
            return None, telemetry

        # Ensure API key
        if not os.environ.get('ANTHROPIC_API_KEY'):
            load_env()

        t0 = time.time()
        try:
            client = anthropic.Anthropic(timeout=ANTHROPIC_CLIENT_TIMEOUT)
            # 1h cache on the (stable, byte-identical) system prompt so repeat
            # _call_llm calls within a run/hour read it from cache instead of
            # re-billing full input — mirrors run_llm_loop's BP1. No-op below
            # the model's cacheable floor: s2_healer (~2.5K tok) sits under
            # Haiku 4.5's 4096 floor, so this only engages once a _call_llm
            # prompt clears the floor (s2_aspects on Sonnet already does).
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system_prompt,
                         "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                messages=[{"role": "user", "content": user_content}])

            telemetry = {'elapsed_ms': int((time.time() - t0) * 1000),
                         **read_usage(response)}

            raw = response.content[0].text.strip()
            return self._extract_json(raw), telemetry

        except Exception as e:
            print('[%s] LLM call failed: %s' % (self.NAME, e), flush=True)
            self.brain._log_error(self.NAME, e, 'LLM call for %s' % interaction_name)
            telemetry['elapsed_ms'] = int((time.time() - t0) * 1000)
            return None, telemetry

    @staticmethod
    def _extract_json(text):
        """Extract JSON array or object from LLM response text.

        Handles markdown code fences, leading/trailing text.
        Returns parsed JSON (list or dict), or None on failure.
        """
        # Strip markdown fences
        if '```' in text:
            parts = text.split('```')
            if len(parts) >= 3:
                text = parts[1]
                if text.startswith('json'):
                    text = text[4:]
                text = text.strip()

        # Find JSON array or object
        # Try array first (most common for batched proposals)
        start = text.find('[')
        if start >= 0:
            end = text.rfind(']') + 1
            if end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

        # Try object
        start = text.find('{')
        if start >= 0:
            end = text.rfind('}') + 1
            if end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

        return None
