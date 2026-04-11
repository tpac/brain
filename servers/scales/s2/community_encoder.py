"""S2CE — Community Encoder.

Takes decoder proposals → runs agentic Sonnet → writes community nodes.
Uses brain_batch tool through run_llm_loop (same pattern as S1E).

The encoder sees proposals formatted as text: timeline, edge signatures,
representative nodes, sample edges. It creates community nodes with
narratives, situations, and metadata via tool calls.
"""

import os

from .base import IntegrationUnit
from .community_contract import (
    COMMUNITY_DETECTION, S2CE_NODE_FORMAT,
)
from .community_decoder import read_community_meta


class CommunityEncoder(IntegrationUnit):
    NAME = 'community_detection'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:community_detection'

    O_SOURCES = ['community_proposals']
    K_SOURCES = ['llm_enrichment', 'community_journal']

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or COMMUNITY_DETECTION

    def run(self, proposals, community_state):
        """Encode proposals into community nodes.

        Args:
            proposals: List of proposal dicts from decoder.
                       Only actionable types are processed:
                       new_community, add_to_existing, drift, health_update,
                       merge_communities.
            community_state: Current community state from decoder.

        Returns:
            dict with keys: actions, write_actions, rounds,
                            action_details, final_text
            None on failure.
        """
        # Filter to actionable proposals
        encoder_proposals = [p for p in proposals
                             if p['type'] in ('new_community', 'add_to_existing',
                                              'drift', 'health_update',
                                              'merge_communities')]
        if not encoder_proposals:
            self.trace('delta', 'community_enriched', 'No actionable proposals')
            return {'actions': 0, 'write_actions': 0, 'rounds': 0,
                    'action_details': [], 'final_text': ''}

        result = self._encode(encoder_proposals, community_state)
        if not result:
            self.trace('delta', 'community_enriched', 'Enrichment failed')
            return None

        actions = result.get('actions', 0)
        write_actions = result.get('write_actions', 0)
        rounds = result.get('rounds', 0)

        self.trace('delta', 'community_enriched',
                   '%d actions (%d writes) in %d rounds' % (
                       actions, write_actions, rounds),
                   metadata={
                       'actions': actions,
                       'write_actions': write_actions,
                       'rounds': rounds,
                       'action_details': result.get('action_details', []),
                       'final_text': result.get('final_text', '')[:2000],
                   })

        return result

    # ══════════════════════════════════════════════════════════
    # _encode — agentic Sonnet via run_llm_loop
    # ══════════════════════════════════════════════════════════

    def _encode(self, proposals, community_state):
        """Run S2CE in batches — processes ALL proposals.

        Proposals are split into chunks of max_proposals_per_call.
        Each chunk gets one Sonnet call. Between chunks, community_state
        is refreshed so later chunks see what earlier ones created.
        """
        import anthropic
        from ..dispatch import load_env
        from ..runner import run_llm_loop
        from .community_decoder import CommunityDecoder

        system_prompt = self.brain.get_interaction_prompt(
            's2_community_enrichment')
        config = self._get_interaction_config('s2_community_enrichment') or {}
        model = config.get('model', self.config.get(
            'model', 'claude-sonnet-4-20250514'))
        max_tokens = config.get('max_tokens', self.config.get(
            'max_tokens', 8192))

        if not system_prompt:
            print('[s2ce] WARNING: no enrichment prompt', flush=True)
            return None

        if not os.environ.get('ANTHROPIC_API_KEY'):
            load_env()

        tools = self._get_tool_schemas()
        dispatch_fn = self._make_dispatch()
        client = anthropic.Anthropic()

        # Journal text
        journal = self.brain.get_config('s2_community_journal') or ''
        journal_prefix = ("COMMUNITY JOURNAL:\n%s\n\n" % journal[
            -self.config.get('journal_max_chars', 14000):]) if journal \
            else "COMMUNITY JOURNAL: First run — no previous encoding.\n\n"

        # Batch proposals
        batch_size = self.config.get('max_proposals_per_call', 15)
        total_result = {
            'rounds': 0, 'actions': 0, 'write_actions': 0,
            'action_details': [], 'final_text': '',
        }
        current_state = list(community_state)

        # Need a decoder instance to refresh community state between batches
        decoder = CommunityDecoder(self.brain, self.dispatch, self.config)

        for batch_idx in range(0, len(proposals), batch_size):
            batch = proposals[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(proposals) + batch_size - 1) // batch_size

            print('[s2ce] Batch %d/%d (%d proposals)' % (
                batch_num, total_batches, len(batch)), flush=True)

            # Format this batch
            user_content = self._format_proposals(batch)

            # Existing communities (refreshed between batches)
            if current_state:
                existing = ["EXISTING COMMUNITIES:"]
                for comm in current_state:
                    maturity = read_community_meta(
                        self.brain.conn, comm['id'],
                        'community_maturity', type='str')
                    existing.append('  "%s" (%d members, %s)' % (
                        comm['title'], len(comm['members']), maturity or '?'))
                user_content = '\n'.join(existing) + '\n\n' + user_content
            else:
                user_content = "EXISTING COMMUNITIES: None.\n\n" + user_content

            user_content = journal_prefix + user_content

            try:
                result = run_llm_loop(
                    client=client,
                    model=model,
                    max_tokens=max_tokens,
                    max_rounds=self.config.get('max_rounds', 4),
                    system_prompt=system_prompt,
                    user_content=user_content,
                    tools=tools,
                    dispatch_fn=dispatch_fn,
                    log_fn=lambda msg: print('[s2ce] %s' % msg, flush=True))

                total_result['rounds'] += result.get('rounds', 0)
                total_result['actions'] += result.get('actions', 0)
                total_result['write_actions'] += result.get('write_actions', 0)
                total_result['action_details'].extend(
                    result.get('action_details', []))
                total_result['final_text'] = result.get('final_text', '')

                # Log truncation errors to brain errors table
                for trunc in result.get('truncations', []):
                    self.brain._log_error(
                        's2ce_truncation',
                        'max_tokens truncation: round %d used %s/%s output tokens' % (
                            trunc['round'], trunc['output_tokens'], trunc['max_tokens']),
                        'batch %d — tool call likely corrupted, community data may be lost' % batch_num)

                # Refresh community state for next batch
                current_state = decoder._read_community_state()

            except Exception as e:
                print('[s2ce] BATCH %d FAILED: %s' % (batch_num, e), flush=True)
                self.brain._log_error(self.NAME, e,
                                      'encode batch %d' % batch_num)

        # Save journal from last batch's final text
        if total_result['final_text']:
            self._save_journal(total_result['final_text'])

        return total_result

    # ══════════════════════════════════════════════════════════
    # Tool infrastructure
    # ══════════════════════════════════════════════════════════

    def _get_tool_schemas(self):
        """S2CE tools: brain_batch (primary) + get_nodes (one read call)."""
        from servers import brain_mcp
        S2CE_TOOLS = {
            'brain_batch',
            'get_nodes',
        }
        return [{"name": t["name"], "description": t["description"],
                 "input_schema": t["inputSchema"]}
                for t in brain_mcp.TOOLS if t["name"] in S2CE_TOOLS]

    def _make_dispatch(self):
        """Create dispatch function for S2CE tool calls."""
        if self.dispatch:
            return self.dispatch

        from servers.daemon_dispatch import COMMAND_TABLE

        brain = self.brain
        encoding_source = self.ENCODING_SOURCE

        def dispatch(cmd, cmd_args):
            if cmd in ('remember', 'remember_batch', 'revise', 'brain_batch'):
                if isinstance(cmd_args, dict):
                    cmd_args.setdefault('encoding_source', encoding_source)
                    for op in cmd_args.get('operations', []):
                        if isinstance(op, dict) and op.get('op') in ('remember',):
                            op.setdefault('encoding_source', encoding_source)

            entry = COMMAND_TABLE.get(cmd)
            if entry:
                return entry.handler(brain, cmd_args, [])
            return {"ok": False, "error": "Unknown command: %s" % cmd}

        return dispatch

    # ══════════════════════════════════════════════════════════
    # Journal
    # ══════════════════════════════════════════════════════════

    def _save_journal(self, final_text):
        """Extract and save journal from encoder's final text."""
        journal_entry = ''
        if '---' in final_text:
            _, journal_part = final_text.split('---', 1)
            journal_entry = journal_part.strip()
        elif 'ACCEPTED:' in final_text or 'OBSERVATIONS:' in final_text:
            for marker in ['ACCEPTED:', 'REJECTED:', 'CORRIDORS:', 'OBSERVATIONS:']:
                idx = final_text.find(marker)
                if idx >= 0:
                    journal_entry = final_text[idx:].strip()
                    break

        if not journal_entry:
            return

        existing = self.brain.get_config('s2_community_journal') or ''
        run_header = '--- S2C Run %s ---' % self.brain.now()[:10]
        new_journal = existing + '\n' + run_header + '\n' + journal_entry

        max_chars = self.config.get('journal_max_chars', 14000)
        if len(new_journal) > max_chars:
            cutpoint = new_journal.find('--- S2C Run', len(new_journal) - max_chars)
            if cutpoint > 0:
                new_journal = new_journal[cutpoint:]

        self.brain.set_config('s2_community_journal', new_journal.strip())

    # ══════════════════════════════════════════════════════════
    # Proposal formatting (text rendering for Sonnet)
    # ══════════════════════════════════════════════════════════

    def _format_proposals(self, proposals):
        from servers.contract import render_rich_node
        from servers.pipeline_contract import get_rich_node

        lines = ['PROPOSALS:\n']

        for i, prop in enumerate(proposals):
            ptype = prop['type'].upper().replace('_', ' ')
            lines.append('[%d] %s' % (i + 1, ptype))

            if prop['type'] == 'new_community':
                corridor = ' (CORRIDOR)' if prop.get('is_corridor') else ''
                lines.append('    %d members, int_frac=%.0f%%%s' % (
                    prop['member_count'],
                    prop['internal_fraction'] * 100,
                    corridor))

                # Timeline
                tl = prop.get('timeline', {})
                if tl:
                    lines.append('    Timeline: %s' % tl.get('date_range', '?'))
                    origin = tl.get('origin', {})
                    latest = tl.get('latest', {})
                    if origin:
                        lines.append('    Origin (%s): [%s] "%s"' % (
                            origin.get('date', '?'),
                            origin.get('type', '?'),
                            origin.get('title', '?')))
                    for t in tl.get('transitions', [])[:3]:
                        lines.append('    Transition (%s): "%s" %s "%s"' % (
                            t.get('date', '?'),
                            t.get('corrector_title', '?')[:40],
                            t.get('relation', '?'),
                            t.get('corrected_title', '?')[:40]))
                    if latest and latest.get('id') != (origin or {}).get('id'):
                        lines.append('    Latest (%s): [%s] "%s"' % (
                            latest.get('date', '?'),
                            latest.get('type', '?'),
                            latest.get('title', '?')))

                # Edge signature
                sig = prop.get('edge_signature', {})
                if sig:
                    parts = ['%s(%.0f%%)' % (f, p * 100)
                             for f, p in sorted(sig.items(),
                                                key=lambda x: -x[1])[:5]]
                    lines.append('    Relational signature: %s' %
                                 ', '.join(parts))

                # ALL members with titles (chronological)
                all_members = prop.get('all_members', [])
                if all_members:
                    lines.append('    All members (chronological):')
                    for m in all_members:
                        lines.append('      %s [%s] "%s" (id:%s)' % (
                            m.get('date', '?'),
                            m.get('type', '?'),
                            m.get('title', '?'),
                            m.get('id', '?')[:8]))

                # Structural hubs (richly rendered)
                reps = prop.get('representatives', [])
                if reps:
                    lines.append('    Structural hubs (richly rendered):')
                    for rep in reps[:3]:
                        try:
                            rich = get_rich_node(
                                self.brain.conn, rep['id'])
                            if rich:
                                rendered = render_rich_node(
                                    rich, S2CE_NODE_FORMAT)
                                lines.append('      ' + rendered.replace(
                                    '\n', '\n      '))
                            else:
                                lines.append(
                                    '      - "%s" (%s)' % (
                                        rep['title'], rep['type']))
                        except Exception:
                            lines.append(
                                '      - "%s" (%s)' % (
                                    rep['title'], rep['type']))

                # Render latest member if not already a hub
                if prop.get('render_latest'):
                    latest_id = tl.get('latest', {}).get('id')
                    if latest_id:
                        try:
                            rich = get_rich_node(
                                self.brain.conn, latest_id)
                            if rich:
                                rendered = render_rich_node(
                                    rich, S2CE_NODE_FORMAT)
                                lines.append('    Latest member (current state):')
                                lines.append('      ' + rendered.replace(
                                    '\n', '\n      '))
                        except Exception:
                            pass

                # Sample internal edges
                edges = prop.get('sample_edges', [])
                if edges:
                    lines.append('    Sample edges:')
                    for e in edges:
                        lines.append(
                            '      - "%s" %s "%s" — %s' % (
                                e['source'], e['relation'],
                                e['target'], e['description']))

            elif prop['type'] == 'node_affinities':
                lines.append('    Node: [%s] "%s" (via %s)' % (
                    prop.get('node_type', '?'),
                    prop.get('node_title', '?'),
                    prop.get('method', '?')))
                for aff in prop.get('affinities', [])[:3]:
                    lines.append('    → cluster %d (%.0f%%)' % (
                        aff['cluster_id'], aff['affinity'] * 100))

            elif prop['type'] == 'cross_cutting':
                lines.append('    [%s] "%s" — spreads across %d clusters, top=%.0f%%' % (
                    prop.get('node_type', '?'),
                    prop.get('node_title', '?'),
                    prop.get('cluster_count', 0),
                    prop.get('top_affinity', 0) * 100))

            elif prop['type'] == 'add_to_existing':
                lines.append('    Node: [%s] "%s"' % (
                    prop.get('node_type', '?'),
                    prop.get('node_title', '?')))
                for comm in prop.get('communities', []):
                    lines.append('    → existing "%s" (affinity: %.0f%%)' % (
                        comm.get('title', '?'), comm.get('affinity', 0) * 100))

            elif prop['type'] == 'drift':
                lines.append('    Node: [%s] "%s"' % (
                    prop.get('node_type', '?'),
                    prop.get('node_title', '?')))
                lines.append('    Home: "%s" (affinity: %.0f%%)' % (
                    prop.get('home_community', '?'),
                    prop.get('home_affinity', 0) * 100))
                for f in prop.get('foreign', []):
                    lines.append('    Drifting toward: "%s" (affinity: %.0f%%)' % (
                        f.get('title', '?'), f.get('affinity', 0) * 100))

            elif prop['type'] == 'health_update':
                lines.append('    Community: "%s"' % prop.get('community_title', '?'))
                lines.append('    Signal: %s (int_frac %.2f → %.2f)' % (
                    prop.get('signal', '?'),
                    prop.get('old_fraction', 0),
                    prop.get('new_fraction', 0)))

            elif prop['type'] == 'merge_communities':
                lines.append('    Larger: "%s" (%d members, id:%s)' % (
                    prop.get('larger_title', '?'),
                    prop.get('larger_size', 0),
                    prop.get('larger_id', '?')[:8]))
                lines.append('    Smaller: "%s" (%d members, id:%s)' % (
                    prop.get('smaller_title', '?'),
                    prop.get('smaller_size', 0),
                    prop.get('smaller_id', '?')[:8]))
                lines.append('    Overlap: %d shared (%.0f%% of smaller), %d unique in smaller' % (
                    prop.get('shared_count', 0),
                    prop.get('overlap_pct', 0) * 100,
                    prop.get('unique_in_smaller', 0)))

            lines.append('')

        return '\n'.join(lines)
