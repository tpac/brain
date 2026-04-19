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
from datetime import date, datetime, timezone, timedelta


# Per-request timeout on the Anthropic SDK client used by S2 encoders.
# The SDK default (600s/request) let a hung API call tie up consolidation
# for 45 minutes on 2026-04-19. 180s is generous for a 10-proposal batch
# on Sonnet and caps the blast radius of any single stuck request.
ANTHROPIC_CLIENT_TIMEOUT = 180.0


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

        Format: {scale}-{YYYYMMDD}-{name}
        S2 chains are date-based, not session-based.
        """
        return '%s-%s-%s' % (self.SCALE, date.today().strftime('%Y%m%d'), self.NAME)

    def trace(self, event_type, ref_type, summary, ref_id='', metadata=None):
        """Write a trace event for this unit's current run.

        Uses direct TraceDAL when running inline (dispatch is None).
        Uses dispatch('trace_append', ...) when running in background.
        """
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

    def _last_run_timestamp(self):
        """Find this unit's most recent completed run timestamp.

        Returns ISO timestamp string, or '' if never run.
        Only counts runs that wrote a delta trace — incomplete runs
        (encoder hung, timed out) don't have deltas and are skipped.
        """
        try:
            row = self.brain.logs_conn.execute(
                "SELECT created_at FROM trace_events "
                "WHERE scale = ? AND event_type = 'delta' AND chain_id LIKE ? "
                "ORDER BY created_at DESC LIMIT 1",
                (self.SCALE, '%%-' + self.NAME)).fetchone()
            return row[0] if row else ''
        except Exception as e:
            import sys
            print('[%s] _last_run_timestamp error: %s' % (self.NAME, e), file=sys.stderr)
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
        except Exception:
            return True  # On error, run to be safe

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

    def _call_llm(self, interaction_name, user_content):
        """Call LLM with a learnable prompt from interactions table.

        Loads system prompt from interaction template.
        Loads config (model, max_tokens) from interaction parameters.
        Handles JSON extraction from response.

        Args:
            interaction_name: Key in interactions table (e.g. 's2_community_enrichment')
            user_content: String content for the user message

        Returns:
            Parsed JSON from the LLM response, or None on failure.
        """
        import anthropic
        from ..dispatch import load_env

        # Load learnable prompt and config
        system_prompt = self.brain.get_interaction_prompt(interaction_name)
        config = self._get_interaction_config(interaction_name)
        model = config.get('model', 'claude-haiku-4-5')
        max_tokens = config.get('max_tokens', 4096)

        if not system_prompt:
            print('[%s] WARNING: no interaction prompt for %s' % (
                self.NAME, interaction_name), flush=True)
            return None

        # Ensure API key
        if not os.environ.get('ANTHROPIC_API_KEY'):
            load_env()

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}])

            raw = response.content[0].text.strip()
            return self._extract_json(raw)

        except Exception as e:
            print('[%s] LLM call failed: %s' % (self.NAME, e), flush=True)
            self.brain._log_error(self.NAME, e, 'LLM call for %s' % interaction_name)
            return None

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
