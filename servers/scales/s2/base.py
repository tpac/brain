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
# USAGE_FIELDS / read_usage: the canonical token-telemetry contract shared with
# run_llm_loop, so _call_llm and _sum_telemetry don't re-derive it.
from ..runner import ANTHROPIC_CLIENT_TIMEOUT, USAGE_FIELDS, read_usage  # noqa: F401 — re-export for callers
from ..dispatch import ATTRIBUTED_WRITE_COMMANDS


# Writes whose handlers emit chain-bearing traces (node_revised /
# edge_relation_revised). They must carry the unit's run chain_id so those
# traces join the run's chain instead of dispatch_write's date-based
# '{scale}-{date}-revise' fallback — the phantom "S2 revise" / "S1 revise" unit.
# Derived from the attributed writes + revise_edge (one source, so a new
# attributed write becomes chain-aware for free — the house pattern).
CHAIN_AWARE_WRITES = ATTRIBUTED_WRITE_COMMANDS | {'revise_edge'}


def apply_encoder_attribution(cmd, cmd_args, *, encoding_source, run_chain_id):
    """Stamp scale-agent attribution onto an outgoing write's args — the single
    chokepoint both facts flow through:

    - ``encoding_source`` on attributed writes (who minted the node/edge),
    - ``run_chain_id`` on chain-bearing writes (which run's chain its trace
      joins).

    ``setdefault``, so an explicitly-supplied value wins. Mutates ``cmd_args``
    in place; a no-op on non-dict args and on reads (get_nodes / recall_batch
    are in neither set — they carry no attribution, and a chain_id on a read is
    meaningless). This is make_scale_dispatch's rule, now that the in-process
    encoder dispatch is the one factory S1 Scribe and the S2 units share.
    """
    if not isinstance(cmd_args, dict):
        return
    if cmd in ATTRIBUTED_WRITE_COMMANDS:
        cmd_args.setdefault('encoding_source', encoding_source)
    if run_chain_id and cmd in CHAIN_AWARE_WRITES:
        cmd_args.setdefault('chain_id', run_chain_id)


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
            brain: Brain instance. When running inline (idle hook),
                   this is the daemon's brain with direct DB access.
                   When running via run_in_background, this is a
                   read-only copy.
            dispatch_fn: Optional dispatch function for TCP writes.
                         None when running inline with direct DB access.
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

    def _inject_review_block(self, system_prompt, legacy_heading=None, relabels=None):
        """Inject the shared residue-review block into a registered prompt at
        runtime — the WRITE-side counterpart to _load_journal_notes_prefix, and
        the same pattern encoders use for `## Edge Families`. The block is
        single-sourced in trace_contract and never baked into the registered
        prompt, so it iterates in one place and every unit on the note contract
        gets it live (consolidation now; community/S1E later).

        - legacy_heading: if the registered prompt still carries a pre-redesign
          journal section, its heading — everything from there is stripped.
          Expected-but-absent is logged LOUD (drift: the registered prompt
          changed shape, so the legacy section would survive and the review
          block would double up).
        - relabels: [(old, new), ...] per-encoder body fixups (e.g. relabel a
          continuity-read line). A no-match here is a benign cosmetic no-op.
        """
        from servers.trace_contract import render_journal_review_block
        if legacy_heading:
            cut = system_prompt.find(legacy_heading)
            if cut != -1:
                system_prompt = system_prompt[:cut].rstrip()
            else:
                self.brain._log_error(
                    's2_%s_prompt_transform' % self.NAME,
                    ValueError('legacy journal heading %r absent from registered '
                               'prompt — not stripped; review block will double '
                               'up' % legacy_heading),
                    'prompt drift — re-check the registered %s prompt' % self.NAME)
        for old, new in (relabels or []):
            system_prompt = system_prompt.replace(old, new)
        return (system_prompt.rstrip() + "\n\n## When you're done\n\n"
                + render_journal_review_block() + '\n\nThen write "DONE".')

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

        def dispatch(cmd, cmd_args):
            apply_encoder_attribution(
                cmd, cmd_args,
                encoding_source=encoding_source, run_chain_id=run_chain_id)

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
                traces = self.brain._trace_dal.get_by_ref_type(
                    ref_type, scale=scale, hours=168, limit=5)
            else:
                traces = self.brain._trace_dal.get_recent(
                    scale=scale, event_type='delta', limit=5)
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
                results.extend(self.brain._trace_dal.get_by_ref_type(
                    rt, scale=scale, hours=hours, limit=500))
        else:
            results = self.brain._trace_dal.get_recent(
                scale=scale, hours=hours, limit=500)

        if since_ts:
            results = [t for t in results if t.get('created_at', '') > since_ts]

        return results

    def _get_interaction_config(self, name):
        """Get config dict from interactions table.

        Returns {} if not found. Config is the 'parameters' JSON field.
        """
        return self.brain.get_interaction_config(name)

    def _sum_telemetry(self, total, result):
        """Accumulate one batch's token telemetry from `result` into `total`
        (in place), over the canonical USAGE_FIELDS. Shared by the multi-batch
        S2 encoders (consolidation / community / healer) so the per-batch
        summation lives in one place. elapsed_ms is NOT summed — each encoder
        wall-clocks its whole batch loop once."""
        for f in USAGE_FIELDS:
            total[f] = total.get(f, 0) + (result.get(f) or 0)

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
            return None, telemetry

        # Ensure API key
        if not os.environ.get('ANTHROPIC_API_KEY'):
            load_env()

        t0 = time.time()
        try:
            client = anthropic.Anthropic(timeout=ANTHROPIC_CLIENT_TIMEOUT)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
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
