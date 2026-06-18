"""S2 Healer Encoder — generates missing fields via Haiku, stores via revise().

Takes proposals from decoder, calls Haiku to generate question/situation/reasoning,
parses JSON response, writes fields through brain's standard write path.

Uses the s2_healer interaction (learnable prompt in interactions table).
"""

import json

from servers.trace_contract import build_delta_metadata

from .base import IntegrationUnit
from .healer_contract import HEALER


class HealerEncoder(IntegrationUnit):
    NAME = 'healer'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:healer'

    O_SOURCES = ['healer_proposals']
    K_SOURCES = ['llm_healer']

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or HEALER

    def run(self, proposals):
        """Generate missing fields for proposed nodes.

        Args:
            proposals: List of proposal dicts from decoder.

        Returns: {nodes_healed, fields_written, skipped, errors, journal}
        """
        if not proposals:
            self.trace('delta', 'healer_generated', 'No proposals to process')
            return {'nodes_healed': 0, 'fields_written': 0, 'skipped': 0,
                    'errors': [], 'journal': ''}

        nodes_healed = 0
        fields_written = 0
        skipped = 0
        errors = []
        healed_ids = []       # for journal_entry: which nodes got filled (short)
        revised_ids = []      # full ids → delta `revised` (structured Δ, was empty)
        field_counter = {}    # for outcomes: which fields got written most
        batches = 0

        batch_size = self.config['max_nodes_per_call']
        for batch_start in range(0, len(proposals), batch_size):
            batches += 1
            batch = proposals[batch_start:batch_start + batch_size]
            # Lookup table: match Haiku's node_id back to the proposal
            # that triggered it (so we can enforce needs_* flags and reject
            # any unsolicited fields Haiku regenerates).
            by_full = {p['node_id']: p for p in batch}
            by_short = {p['node_id'][:8]: p for p in batch}

            user_content = self._format_batch(batch)
            result = self._call_llm('s2_healer', user_content)

            if result is None:
                errors.append('LLM call failed for batch %d' % batches)
                continue

            if isinstance(result, list):
                healings = result
            elif isinstance(result, dict):
                healings = [result]
            else:
                errors.append('Unexpected response type: %s' % type(result).__name__)
                continue

            for healing in healings:
                if not isinstance(healing, dict):
                    continue
                nid = healing.get('node_id', '')
                if not nid:
                    continue

                proposal = by_full.get(nid) or by_short.get(nid[:8])
                written, full_id = self._store_fields(nid, healing, proposal)
                if written > 0:
                    nodes_healed += 1
                    fields_written += written
                    healed_ids.append(nid[:8])
                    if full_id:
                        revised_ids.append(full_id)
                    for field in ('question', 'situation', 'reasoning'):
                        if healing.get(field):
                            field_counter[field] = field_counter.get(field, 0) + 1
                else:
                    skipped += 1

        # Synthesize a journal entry from the numeric result. Healer
        # doesn't produce free-form agent text, so we build the per-run
        # narrative here — short, scannable, honest.
        journal_entry = ''
        if healed_ids or errors:
            parts = []
            parts.append('Healed %d node(s), wrote %d field(s), skipped %d.' % (
                nodes_healed, fields_written, skipped))
            if field_counter:
                parts.append('Fields: %s' % ', '.join(
                    '%s=%d' % kv for kv in sorted(field_counter.items())))
            if healed_ids:
                parts.append('Nodes: %s' % ', '.join(healed_ids[:10]))
            if errors:
                parts.append('Errors: %d (first: %s)' % (len(errors), errors[0][:80]))
            journal_entry = ' '.join(parts)

        outcomes = {
            'filled': nodes_healed,
            'skipped': skipped,
        }
        outcomes.update(field_counter)  # question/situation/reasoning counts

        self.trace('delta', 'healer_generated',
                   '%d nodes healed, %d fields written, %d skipped' % (
                       nodes_healed, fields_written, skipped),
                   metadata=build_delta_metadata(
                       actions=nodes_healed + skipped,
                       write_actions=fields_written,
                       rounds=batches,
                       inputs_processed=len(proposals),
                       outcomes=outcomes,
                       journal_entry=journal_entry,
                       errors=errors,
                       # Healer writes via dispatch('revise') in a hand-rolled
                       # loop (not run_llm_loop), so it passes the revised ids
                       # explicitly — without this the structured Δ was empty
                       # and every heal lived only in the journal counter-string.
                       revised=revised_ids,
                       nodes_healed=nodes_healed,
                       fields_written=fields_written,
                   ))

        return {
            'nodes_healed': nodes_healed,
            'fields_written': fields_written,
            'skipped': skipped,
            'errors': errors,
            'journal': journal_entry,
        }

    def _format_batch(self, proposals):
        """Format proposals for the Haiku prompt.

        Rich presentation — full node context, conversation exchanges,
        healing flags. The stateless agent needs everything in one view.
        """
        lines = []
        for p in proposals:
            lines.append('')
            lines.append('=' * 50)
            lines.append('NODE_ID: %s' % p['node_id'][:8])

            # Rich node data from get_rich_node
            node = p.get('rich_node', {})
            lines.append('TITLE: "%s"' % node.get('title', p.get('title', '')))
            lines.append('TYPE: %s | CONFIDENCE: %s | LOCKED: %s' % (
                node.get('type', ''), node.get('confidence', ''),
                'yes' if node.get('locked') else 'no'))

            # What needs healing
            needs = []
            if p.get('needs_question'):
                needs.append('question')
            if p.get('needs_situation'):
                needs.append('situation')
            if p.get('needs_reasoning'):
                needs.append('reasoning')
            lines.append('NEEDS: %s' % ', '.join(needs))

            # Content
            lines.append('')
            lines.append('CONTENT:')
            lines.append(node.get('content', ''))

            # Existing fields (so Haiku doesn't regenerate what exists)
            if node.get('situation'):
                lines.append('EXISTING SITUATION: %s' % node['situation'])

            metadata = node.get('_metadata', {})
            if metadata.get('reasoning'):
                lines.append('EXISTING REASONING: %s' % metadata['reasoning'])
            if metadata.get('user_raw_quote'):
                lines.append('OPERATOR SAID: "%s"' % metadata['user_raw_quote'])
            if metadata.get('anchor_raw_quote'):
                lines.append('ASSISTANT SAID: "%s"' % metadata['anchor_raw_quote'])
            if metadata.get('question'):
                lines.append('EXISTING QUESTION: %s' % metadata['question'])

            # Corrections — heavy slice via the unified render_corrections().
            # Single source of truth (contract.py:render_corrections) — same
            # function render_rich_node uses, just called directly here because
            # HealerEncoder builds its own framing (NEEDS, NODE_ID, etc.) and
            # doesn't go through render_rich_node.
            corrections = node.get('_corrections', [])
            if corrections:
                from servers.contract import render_corrections
                corr_lines = render_corrections(
                    corrections[:3], mode='heavy',
                    content_limit_heavy=500,
                    meta_limit_heavy=400,
                    indent='  ')
                if corr_lines:
                    lines.append('')
                    lines.append('CORRECTIONS:')
                    lines.extend(corr_lines)

            # Connections
            connections = node.get('connections', [])
            if connections:
                lines.append('')
                lines.append('CONNECTIONS (%d):' % len(connections))
                for conn in connections[:8]:
                    rel = conn.get('relation', 'related')
                    desc = conn.get('description', '')
                    title = conn.get('title', '')[:50]
                    ntype = conn.get('type', '')
                    desc_str = ' — %s' % desc[:60] if desc else ''
                    lines.append('  [%s] "%s" (%s)%s' % (rel, title, ntype, desc_str))

            # Conversation context
            conversation = p.get('conversation', [])
            if conversation:
                lines.append('')
                lines.append('CONVERSATION (around when this node was encoded):')

                encoding_ts = p.get('encoding_timestamp', '')
                for turn in conversation:
                    role = 'operator' if turn['role'] == 'user' else 'assistant'
                    content = (turn.get('content') or '')[:200]
                    ts = turn.get('timestamp', '')

                    marker = ''
                    if encoding_ts and ts and ts >= encoding_ts:
                        if not hasattr(self, '_marked_encoding'):
                            self._marked_encoding = True
                            marker = '  ← ENCODED AROUND HERE'

                    lines.append('  [%s] %s%s' % (role, content, marker))

                # Reset marker for next proposal
                if hasattr(self, '_marked_encoding'):
                    del self._marked_encoding
            else:
                lines.append('')
                lines.append('CONVERSATION: (not available — pre-trace node)')

        return '\n'.join(lines)

    def _store_fields(self, node_id, enrichment, proposal=None):
        """Store healed fields via brain's standard write path (revise).

        Always uses revise() — handles metadata, situation embedding,
        group vectors, FTS5, everything in one call. Goes through dispatch
        (TCP to daemon) when available, direct brain.revise() when inline.

        When `proposal` is provided, only fields flagged in the proposal's
        needs_* set get written — Haiku-returned fields for slots that
        already had content are rejected and logged. Prevents the renegade-
        healer pattern (overwriting good data with freshly-generated text).

        Returns (count of fields written, full_id) — full_id lets the caller
        record the revised node in the delta's structured Δ. (0, None) on
        no-op or failure.
        """
        fields_to_write = {}

        for field in ['question', 'situation', 'reasoning']:
            value = enrichment.get(field, '').strip()
            if not (value and len(value) > 5):
                continue
            # Reject unsolicited fields — node already had this slot filled.
            if proposal is not None and not proposal.get('needs_' + field, False):
                self.brain._log_error(
                    'healer_unsolicited_field',
                    Exception('Haiku returned %s for %s but needs_%s=False' % (
                        field, node_id[:8], field)),
                    'rejecting — node already had %s, not overwriting' % field)
                continue
            fields_to_write[field] = value

        if not fields_to_write:
            return 0, None

        # Resolve short node ID
        ndal = self.brain._nodes
        full_id = ndal.resolve_id(node_id) if len(node_id) < 16 else node_id
        if not full_id:
            return 0, None

        try:
            revise_args = {
                'node_id': full_id,
                'reason': 'healer: %s' % ', '.join(fields_to_write.keys()),
                **fields_to_write,
            }

            if self.dispatch:
                self.dispatch('revise', revise_args)
            else:
                self.brain.revise(**revise_args)

            return len(fields_to_write), full_id

        except Exception as e:
            self.brain._log_error(self.NAME, e,
                                  'storing fields for %s' % node_id[:8])
            return 0, None
