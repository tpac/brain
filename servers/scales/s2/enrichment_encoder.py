"""S2 Healer Encoder — generates missing fields via Haiku, stores via revise().

Takes proposals from decoder, calls Haiku to generate question/situation/reasoning,
parses JSON response, writes fields through brain's standard write path.

Uses the s2_enrichment interaction (learnable prompt in interactions table).
"""

import json

from .base import IntegrationUnit
from .enrichment_contract import ENRICHMENT


class EnrichmentEncoder(IntegrationUnit):
    NAME = 'enrichment'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:enrichment'

    O_SOURCES = ['enrichment_proposals']
    K_SOURCES = ['llm_enrichment']

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or ENRICHMENT

    def run(self, proposals):
        """Generate missing fields for proposed nodes.

        Args:
            proposals: List of proposal dicts from decoder.

        Returns: {nodes_healed, fields_written, skipped, errors, journal}
        """
        if not proposals:
            self.trace('delta', 'enrichment_generated', 'No proposals to process')
            return {'nodes_healed': 0, 'fields_written': 0, 'skipped': 0,
                    'errors': [], 'journal': ''}

        nodes_healed = 0
        fields_written = 0
        skipped = 0
        errors = []
        journal = ''

        # Process in batches
        batch_size = self.config['max_nodes_per_call']
        for batch_start in range(0, len(proposals), batch_size):
            batch = proposals[batch_start:batch_start + batch_size]

            # Format batch for Haiku
            user_content = self._format_batch(batch)

            # Call Haiku via learnable prompt
            result = self._call_llm('s2_enrichment', user_content)

            if result is None:
                errors.append('LLM call failed for batch %d' % (batch_start // batch_size + 1))
                continue

            # Parse: result should be a JSON array
            # _call_llm already extracts JSON, but response may have journal after it
            if isinstance(result, list):
                enrichments = result
            elif isinstance(result, dict):
                enrichments = [result]
            else:
                errors.append('Unexpected response type: %s' % type(result).__name__)
                continue

            # Store each enrichment via revise()
            for enrichment in enrichments:
                if not isinstance(enrichment, dict):
                    continue
                nid = enrichment.get('node_id', '')
                if not nid:
                    continue

                written = self._store_fields(nid, enrichment)
                if written > 0:
                    nodes_healed += 1
                    fields_written += written
                else:
                    skipped += 1

        # Delta trace
        self.trace('delta', 'enrichment_generated',
                   '%d nodes healed, %d fields written, %d skipped' % (
                       nodes_healed, fields_written, skipped),
                   metadata={
                       'nodes_healed': nodes_healed,
                       'fields_written': fields_written,
                       'skipped': skipped,
                       'errors': errors[:5],
                   })

        return {
            'nodes_healed': nodes_healed,
            'fields_written': fields_written,
            'skipped': skipped,
            'errors': errors,
            'journal': journal,
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

            # Corrections
            corrections = node.get('_corrections', [])
            if corrections:
                lines.append('')
                lines.append('CORRECTIONS:')
                for c in corrections[:3]:
                    direction = c.get('direction', '?')
                    lines.append('  [%s] %s "%s"' % (
                        direction, c.get('id', '')[:8],
                        c.get('title', c.get('content', ''))[:60]))

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

    def _store_fields(self, node_id, enrichment):
        """Store healed fields via brain's standard write path (revise).

        Always uses revise() — handles metadata, situation embedding,
        group vectors, FTS5, everything in one call. Goes through dispatch
        (TCP to daemon) when available, direct brain.revise() when inline.

        Returns count of fields written.
        """
        fields_to_write = {}

        for field in ['question', 'situation', 'reasoning']:
            value = enrichment.get(field, '').strip()
            if value and len(value) > 5:
                fields_to_write[field] = value

        if not fields_to_write:
            return 0

        # Resolve short node ID
        from servers.dal import NodeDAL
        ndal = NodeDAL(self.brain.conn)
        full_id = ndal.resolve_id(node_id) if len(node_id) < 16 else node_id
        if not full_id:
            return 0

        try:
            revise_args = {
                'node_id': full_id,
                'reason': 'healer: %s' % ', '.join(fields_to_write.keys()),
                **fields_to_write,
            }

            # skip_embedding: prevent ONNX multi-thread spin during S2
            revise_args['skip_embedding'] = True

            if self.dispatch:
                self.dispatch('revise', revise_args)
            else:
                self.brain.revise(**revise_args)

            return len(fields_to_write)

        except Exception as e:
            self.brain._log_error(self.NAME, e,
                                  'storing fields for %s' % node_id[:8])
            return 0
