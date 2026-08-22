"""S2 Healer Encoder — generates missing fields via Haiku, stores via revise().

Takes proposals from decoder, calls Haiku to generate question/situation/reasoning,
parses JSON response, writes fields through brain's standard write path.

Uses the s2_healer interaction (learnable prompt in interactions table).
"""

import json
import time

from servers.trace_contract import build_delta_metadata

from .base import IntegrationUnit, read_usage, sum_usage
from .healer_contract import HEALER


class HealerEncoder(IntegrationUnit):
    NAME = 'healer'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:healer'

    O_SOURCES = ['healer_proposals']
    K_SOURCES = ['llm_healer', 'journal_notes']

    # Residue flows to journal_note trace rows via the journal binding on
    # _call_llm (decorate → harvest), read back via continuity() — the note
    # contract, same as the loop encoders. The synthesized journal_entry
    # counter-string below is the TRACE record (what happened), orthogonal
    # to residue (what the actions don't capture).

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or HEALER
        self._dispatch_fn = None

    def _make_dispatch(self):
        """Healer dispatch — inherits base `_make_encoder_dispatch` (no
        archive guard: the healer never archives; its writes are field-fill
        revises). Built HERE, on the encoder instance: the closure stamps
        `run_chain_id` from `chain_id()`, which caches per instance at
        seconds resolution — a closure built on the ORCHESTRATOR would put
        node_revised rows and this instance's `healer_generated` delta on
        two different chains, one pass rendering as two phantom runs
        (plan step 10). Caveat: a dispatch_fn injected at construction
        (tests, evals) is honored verbatim by the base and bypasses this
        attribution — production constructs with None."""
        if self._dispatch_fn is None:
            self._dispatch_fn = self._make_encoder_dispatch()
        return self._dispatch_fn

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
        # Cost/latency telemetry — tokens summed across per-batch LLM calls via
        # the shared sum_usage, elapsed by a wall-clock timer around the loop.
        # Healer uses _call_llm (a plain messages.create), not run_llm_loop, so
        # the per-call telemetry comes from _call_llm's tuple return (just
        # elapsed + tokens — no run loop to fold, so sum_usage not _accumulate_run).
        # Before this, the healer_generated delta omitted them (elapsed_ms=0/
        # output_tokens=0).
        tel_totals = read_usage(None)
        _t0 = time.time()

        # Residue continuity — read ONCE before the loop (a per-batch read
        # would echo batch 1's just-written notes into batch 2), prepended to
        # every batch's content, mirroring the loop encoders.
        journal_prefix = self.journal.continuity()

        batch_size = self.config['max_nodes_per_call']
        for batch_start in range(0, len(proposals), batch_size):
            batches += 1
            batch = proposals[batch_start:batch_start + batch_size]
            # Lookup table: match Haiku's node_id back to the proposal
            # that triggered it (so we can enforce needs_* flags and reject
            # any unsolicited fields Haiku regenerates).
            by_full = {p['node_id']: p for p in batch}
            by_short = {p['node_id'][:8]: p for p in batch}

            user_content = journal_prefix + self._format_batch(batch)
            result, call_tel = self._call_llm('s2_healer', user_content,
                                              journal=True)
            sum_usage(tel_totals, call_tel)

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
                if proposal is None:
                    # Out-of-batch healing: the journal continuity prefix
                    # shows node ids from past runs, so the model can return
                    # one — but without a proposal there are no needs_* flags
                    # and the unsolicited-field guard cannot run. Ids in
                    # continuity are context, never work orders. Drop loud.
                    self.brain._log_error(
                        'healer_out_of_batch_healing',
                        Exception('healing for %s matches no proposal in '
                                  'this batch — dropped' % nid[:8]),
                        'model healed an id outside the batch (likely from '
                        'the continuity prefix); no needs_* flags to guard '
                        'against overwrites')
                    skipped += 1
                    continue
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

        elapsed_ms = int((time.time() - _t0) * 1000)

        self.trace('delta', 'healer_generated',
                   '%d nodes healed, %d fields written, %d skipped, '
                   '%dms, %d→%d tok' % (
                       nodes_healed, fields_written, skipped, elapsed_ms,
                       tel_totals.get('input_tokens', 0),
                       tel_totals.get('output_tokens', 0)),
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
                       elapsed_ms=elapsed_ms,
                       input_tokens=tel_totals.get('input_tokens', 0),
                       output_tokens=tel_totals.get('output_tokens', 0),
                       cache_read_tokens=tel_totals.get('cache_read_tokens', 0),
                       cache_creation_tokens=tel_totals.get('cache_creation_tokens', 0),
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

            metadata = node.get('_metadata', {})

            # What needs healing (the empty slots)
            needs = []
            if p.get('needs_question'):
                needs.append('question')
            if p.get('needs_situation'):
                needs.append('situation')
            if p.get('needs_reasoning'):
                needs.append('reasoning')
            lines.append('NEEDS: %s' % ', '.join(needs))

            # Explicit per-field filled/empty status. NEEDS (above) lists the
            # empty slots; this shows EVERY healable field's state in one place
            # with the FILLED content inline, so Haiku can't mistake a filled
            # field for an empty one and regenerate it. (The unsolicited-field
            # guard in _store_fields rejects an overwrite either way; showing
            # the state plainly prevents the wasted generation up front.)
            # Sourced identically to the decoder's needs_* computation —
            # situation top-level, question/reasoning from _metadata, all
            # canonical node_metadata_kv values — so status and content agree.
            field_values = {
                'question': metadata.get('question', ''),
                'situation': node.get('situation', ''),
                'reasoning': metadata.get('reasoning', ''),
            }
            lines.append('FIELD STATUS (regenerate ONLY the EMPTY ones; '
                         'never overwrite a FILLED one):')
            for field in ('question', 'situation', 'reasoning'):
                if p.get('needs_' + field):
                    lines.append('  %s: EMPTY' % field)
                else:
                    lines.append('  %s: FILLED — %s' % (field, field_values[field]))

            # Content
            lines.append('')
            lines.append('CONTENT:')
            lines.append(node.get('content', ''))

            # Voice quotes — grounding for the generated fields.
            if metadata.get('their_raw_quote'):
                lines.append('OPERATOR SAID: "%s"' % metadata['their_raw_quote'])
            if metadata.get('my_raw_quote'):
                lines.append('ASSISTANT SAID: "%s"' % metadata['my_raw_quote'])

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
        group vectors, FTS5, everything in one call. Always through the
        encoder dispatch (built on THIS instance, see _make_dispatch), so
        every field-fill emits an attributed node_revised trace on the same
        chain as the healer_generated delta.

        Only fields flagged in the proposal's needs_* set get written —
        model-returned fields for slots that already had content are rejected
        and logged. Prevents the renegade-healer pattern (overwriting good
        data with freshly-generated text). A missing proposal rejects every
        field — the caller drops out-of-batch healings before reaching here,
        and this guard holds even if a new caller doesn't.

        Returns (count of fields written, full_id) — full_id lets the caller
        record the revised node in the delta's structured Δ. (0, None) on
        no-op or failure.
        """
        fields_to_write = {}

        for field in ['question', 'situation', 'reasoning']:
            value = enrichment.get(field, '').strip()
            if not (value and len(value) > 5):
                continue
            # Reject unsolicited fields — node already had this slot filled
            # (no proposal at all = nothing was solicited; reject everything).
            if proposal is None or not proposal.get('needs_' + field, False):
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

            # Always through dispatch — the direct brain.revise() fallback
            # was the last healer write bypassing the chokepoint: no
            # node_revised trace, no run-chain attribution (plan step 10).
            r = self._make_dispatch()('revise', revise_args)

            # _handle_revise returns ok=False rather than raising — without
            # this check a rejected revise reported fields_written=N.
            if not (r or {}).get('ok'):
                self.brain._log_error(
                    self.NAME,
                    Exception((r or {}).get('error', 'revise returned no result')),
                    'revise rejected for %s' % node_id[:8])
                return 0, None

            return len(fields_to_write), full_id

        except Exception as e:
            self.brain._log_error(self.NAME, e,
                                  'storing fields for %s' % node_id[:8])
            return 0, None
