"""S2CE — Community Encoder.

Takes decoder proposals → runs the S2 Community Encoder agent → writes
community nodes. Uses brain_batch tool through run_llm_loop (same pattern
as S1E). The encoder agent (currently Haiku 4.5) is configured in the
s2_community_enrichment interaction — model and prompt are both learnable.

The agent sees proposals formatted as text: timeline, edge signatures,
representative nodes, sample edges. It creates community nodes with
narratives, situations, and metadata via tool calls.
"""

import os

from .base import IntegrationUnit
from .community_contract import COMMUNITY_DETECTION
from .community_decoder import read_community_meta
from servers.trace_contract import build_delta_metadata

from .rejection_table import (
    match_proposals_to_actions,
    record_rejections,
    sort_proposals_by_priority,
    node_ids_touched_by_invalid_ops,
    get_proposed_ids,
)


class CommunityEncoder(IntegrationUnit):
    NAME = 'community_detection'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:community_detection'

    O_SOURCES = ['community_proposals']
    K_SOURCES = ['llm_enrichment', 'community_journal']

    # Journal contract (see IntegrationUnit.JOURNAL_* for semantics).
    # JOURNAL_KEY is pinned explicitly to preserve the existing brain_meta
    # entry that predates this unit's rename ('community' → 'community_detection').
    JOURNAL_MARKERS = ('ACCEPTED:', 'REJECTED:', 'CORRIDORS:', 'OBSERVATIONS:')
    JOURNAL_LABEL = 'COMMUNITY JOURNAL'
    JOURNAL_RUN_HEADER = 'S2C Run'
    JOURNAL_KEY = 's2_community_journal'

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

        # Filter corridors — track in traces, don't send to encoder.
        # Corridors (int_frac < 20%) rarely form coherent communities.
        # If they mature, health_update will catch them.
        corridors = [p for p in encoder_proposals
                     if p['type'] == 'new_community' and p.get('is_corridor')]
        if corridors:
            encoder_proposals = [p for p in encoder_proposals
                                 if not (p['type'] == 'new_community' and p.get('is_corridor'))]
            print('[s2ce] Filtered %d corridor proposals (tracked in traces)' % len(corridors),
                  flush=True)
            self.trace('K', 'community_proposals',
                       'Filtered %d corridors: %s' % (
                           len(corridors),
                           ', '.join('%d members' % c.get('member_count', 0)
                                     for c in corridors[:5])))

        # Honest skip: if corridor filtering emptied the actionable set, there
        # is genuinely nothing to encode. Report it as a skip — NOT as a
        # "COMPLETE: 0 actions in 0 rounds", which reads like a successful run
        # on the dashboard and masks that the encoder never had work to do.
        if not encoder_proposals:
            self.trace('delta', 'community_enriched',
                       'SKIPPED: %d corridor(s) filtered, no other actionable proposals'
                       % len(corridors))
            return {'actions': 0, 'write_actions': 0, 'rounds': 0,
                    'action_details': [], 'final_text': '',
                    'skipped': 'corridors_only'}

        # Cap with per-type quotas (process backlog over multiple idle cycles).
        # Sort within each type by confidence (highest first) so strong
        # proposals get encoder time before borderline ones.
        max_per_run = self.config.get('max_actionable_per_run', 60)
        quotas = self.config.get('type_quotas', {})
        if quotas and len(encoder_proposals) > max_per_run:
            sorted_proposals = sort_proposals_by_priority(encoder_proposals)
            capped = []
            by_type = {}
            for p in sorted_proposals:
                by_type.setdefault(p['type'], []).append(p)
            for ptype, quota in quotas.items():
                available = by_type.get(ptype, [])
                capped.extend(available[:quota])
            skipped = len(encoder_proposals) - len(capped)
            # Re-sort capped set by priority so encoder sees structural-impact
            # proposals first (merge → new_community → add_to_existing → ...)
            encoder_proposals = sort_proposals_by_priority(capped[:max_per_run])
            print('[s2ce] Capped to %d proposals (%d deferred): %s' % (
                len(encoder_proposals), skipped,
                ', '.join('%s=%d' % (t, sum(1 for p in encoder_proposals if p['type'] == t))
                          for t in quotas if any(p['type'] == t for p in encoder_proposals))),
                flush=True)
        else:
            # Even without quota cap, sort by priority for encoder batching
            encoder_proposals = sort_proposals_by_priority(encoder_proposals)

        result = self._encode(encoder_proposals, community_state)
        if not result:
            self.trace('delta', 'community_enriched', 'Enrichment failed')
            return None

        actions = result.get('actions', 0)
        write_actions = result.get('write_actions', 0)
        rounds = result.get('rounds', 0)

        # Precise rejection stamping: walk brain_batch operations to identify
        # which proposals the encoder actually acted on. Only stamp skipped
        # ones. Accepted proposals auto-invalidate on the next decode because
        # the graph state changed (new community_member edges, etc.).
        # Safety: if encoder failed (0 rounds or error), don't stamp anything.
        action_details = result.get('action_details', [])
        final_text = result.get('final_text', '') or ''
        encoder_failed = (
            rounds == 0 or
            bool(result.get('error')) or
            'FAILED' in final_text or
            'ERROR' in final_text[:200]
        )

        if encoder_failed:
            print('[s2ce] Encoder failed or incomplete - NOT stamping rejections',
                  flush=True)
            result['rejection_skipped_count'] = 0
            acted_on = []
        else:
            acted_on, skipped_proposals = match_proposals_to_actions(
                encoder_proposals, action_details)
            # A proposal the encoder TRIED to act on with an invalid op (e.g.
            # `op: reject` instead of `revise`+_sys_drift_threshold) lands in
            # skipped because the matcher sees no valid action — but it's an
            # encoder FAILURE, not a clean SKIP. Pull it out so we don't stamp
            # a fingerprint that abandons the drift-rejection forever; it
            # retries next cycle.
            invalid_touched = node_ids_touched_by_invalid_ops(action_details)
            invalid_op_failures = 0
            if invalid_touched and skipped_proposals:
                retry = [p for p in skipped_proposals
                         if set(get_proposed_ids(p)) & invalid_touched]
                if retry:
                    invalid_op_failures = len(retry)
                    skipped_proposals = [p for p in skipped_proposals
                                         if p not in retry]
                    self.brain._log_warning(
                        's2_community_invalid_op_retry',
                        '%d proposal(s) hit invalid brain_batch ops — '
                        'retrying next cycle, NOT suppressed' % invalid_op_failures)
                    print('[s2ce] %d proposal(s) hit invalid ops — retry, NOT suppressed'
                          % invalid_op_failures, flush=True)
            if skipped_proposals:
                record_rejections(self.brain, skipped_proposals)
                print('[s2ce] Stamped %d rejected proposals (fingerprints)' % len(skipped_proposals),
                      flush=True)
            result['rejection_skipped_count'] = len(skipped_proposals)
            result['invalid_op_failures'] = invalid_op_failures

        # Outcome vocab — count acted-on proposals by type.
        outcomes = {}
        for p in acted_on:
            ptype = p.get('type', 'unknown')
            outcomes[ptype] = outcomes.get(ptype, 0) + 1

        self.trace('delta', 'community_enriched',
                   'COMPLETE: %d actions (%d writes) in %d rounds, %d stamped' % (
                       actions, write_actions, rounds,
                       result.get('rejection_skipped_count', 0)),
                   metadata=build_delta_metadata(
                       actions=actions,
                       write_actions=write_actions,
                       rounds=rounds,
                       inputs_processed=len(encoder_proposals),
                       outcomes=outcomes,
                       rejection_skipped=result.get('rejection_skipped_count', 0),
                       invalid_op_failures=result.get('invalid_op_failures', 0),
                       journal_entry=result.get('journal_entry', ''),
                       action_details=action_details,
                       final_text=final_text,
                       corridors_filtered=len(corridors),
                   ))

        return result

    # ══════════════════════════════════════════════════════════
    # _encode — S2 Community Encoder agent via run_llm_loop
    # ══════════════════════════════════════════════════════════

    def _encode(self, proposals, community_state):
        """Run S2CE in batches — processes ALL proposals.

        Proposals are split into chunks of max_proposals_per_call.
        Each chunk gets one encoder-agent call. Between chunks, community_state
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
            'model', 'claude-haiku-4-5-20251001'))
        max_tokens = config.get('max_tokens', self.config.get(
            'max_tokens', 32768))

        if not system_prompt:
            print('[s2ce] WARNING: no enrichment prompt', flush=True)
            return None

        # Inject edge aspects (Step 12 of unified-aspects). Same skip pattern
        # as consolidation_encoder + surface — generic_relation/noise excluded
        # from the prompt, node-only aspects skipped (no edge_relations).
        family_lines = []
        for name, aspect in sorted(self.brain.aspects.all().items()):
            if name in ('generic_relation', 'noise'):
                continue
            if not aspect.edge_relations:
                continue
            family_lines.append('- **%s**: %s' % (
                name, ', '.join(list(aspect.edge_relations[:8]))))
        if family_lines:
            families_section = ('\n\n## Edge Families (from brain.aspects — %d aspects)\n\n%s\n\n'
                                'Avoid `related_to`. Pick specific types.' % (
                                    len(family_lines), '\n'.join(family_lines)))
            system_prompt = system_prompt + families_section

        if not os.environ.get('ANTHROPIC_API_KEY'):
            load_env()

        tools = self._get_tool_schemas()
        dispatch_fn = self._make_dispatch()
        # Single shared S2 timeout — see base.ANTHROPIC_CLIENT_TIMEOUT.
        from .base import ANTHROPIC_CLIENT_TIMEOUT
        client = anthropic.Anthropic(timeout=ANTHROPIC_CLIENT_TIMEOUT)

        # Journal text — continuity between stateless runs, rendered by base.
        journal_prefix = self._load_journal_prefix()

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

            # Relevant existing communities (via recall, not full listing)
            relevant_comms = self._find_relevant_communities(batch, current_state)
            if relevant_comms:
                user_content = relevant_comms + '\n\n' + user_content
            else:
                user_content = "EXISTING COMMUNITIES: None.\n\n" + user_content

            user_content = journal_prefix + user_content

            try:
                from .base import retry_on_transient_api_error
                result = retry_on_transient_api_error(
                    lambda: run_llm_loop(
                        client=client,
                        model=model,
                        max_tokens=max_tokens,
                        max_rounds=self.config.get('max_rounds', 4),
                        system_prompt=system_prompt,
                        user_content=user_content,
                        tools=tools,
                        dispatch_fn=dispatch_fn,
                        log_fn=lambda msg: print('[s2ce] %s' % msg, flush=True)),
                    log_fn=lambda msg: print('[s2ce] %s' % msg, flush=True))

                total_result['rounds'] += result.get('rounds', 0)
                total_result['actions'] += result.get('actions', 0)
                total_result['write_actions'] += result.get('write_actions', 0)
                total_result['action_details'].extend(
                    result.get('action_details', []))
                # Append journal from each batch (don't overwrite previous batches)
                batch_text = result.get('final_text', '')
                if batch_text:
                    total_result['final_text'] += '\n--- batch %d ---\n%s' % (
                        batch_num, batch_text)

                # Log truncation errors to brain errors table
                for trunc in result.get('truncations', []):
                    self.brain._log_error(
                        's2ce_truncation',
                        'max_tokens truncation: round %d used %s/%s output tokens' % (
                            trunc['round'], trunc['output_tokens'], trunc['max_tokens']),
                        'batch %d — tool call likely corrupted, community data may be lost' % batch_num)

                # Refresh community state for next batch
                current_state = decoder._read_community_state()

                # Per-batch progress trace — visible in dashboard immediately
                self.trace('delta', 'community_enriched',
                           'batch %d/%d: %d actions (%d writes), %d total' % (
                               batch_num, total_batches,
                               result.get('actions', 0), result.get('write_actions', 0),
                               total_result['actions']))

            except Exception as e:
                print('[s2ce] BATCH %d FAILED: %s' % (batch_num, e), flush=True)
                self.brain._log_error(self.NAME, e,
                                      'encode batch %d' % batch_num)
                self.trace('delta', 'community_enriched',
                           'batch %d/%d FAILED: %s' % (batch_num, total_batches, str(e)[:80]))

        # Save journal from last batch's final text. Also return the
        # extracted entry in the result so the delta trace carries it.
        total_result['journal_entry'] = ''
        if total_result['final_text']:
            total_result['journal_entry'] = self._save_journal(total_result['final_text']) or ''

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
        """S2CE dispatch — inherits base `_make_encoder_dispatch` (no archive
        guard; community encoder archives whole communities, not individual
        members from a tracked batch).
        """
        return self._make_encoder_dispatch()

    # Journal save/load is inherited from IntegrationUnit.
    # Class attributes JOURNAL_MARKERS / JOURNAL_LABEL / JOURNAL_RUN_HEADER /
    # JOURNAL_KEY configure the pattern; base owns the logic.

    # ══════════════════════════════════════════════════════════
    # Relevant community lookup
    # ══════════════════════════════════════════════════════════

    def _find_relevant_communities(self, batch_proposals, community_state):
        """Find existing communities relevant to this batch via recall.

        Instead of listing all 100+ communities, recall the 15 most
        semantically relevant based on proposal member titles.
        Uses brain.get_node() + render_rich_node() with S2CE_COMMUNITY_FORMAT.
        """
        from servers.contract import render_rich_node
        from .community_contract import S2CE_COMMUNITY_FORMAT

        # Build a query from member titles in this batch
        titles = []
        for p in batch_proposals:
            if p['type'] == 'new_community':
                for m in p.get('all_members', [])[:5]:
                    titles.append(m.get('title', ''))
            elif p.get('node_title'):
                titles.append(p['node_title'])
        query = ' '.join(titles)[:500]
        if not query.strip():
            return None

        try:
            recall_result = self.brain.recall(
                query, limit=15,
                filter={'type': {'in': ['community']}})
            results = recall_result.get('results', []) if isinstance(recall_result, dict) else []
            if not results:
                return None

            # Batch-fetch full nodes for rendering
            result_ids = [r['id'] for r in results if r.get('id')]
            rich_nodes = self.brain.get_node(result_ids)
            if not rich_nodes:
                return None

            lines = ['RELEVANT EXISTING COMMUNITIES (%d of %d total):' % (
                len(rich_nodes), len(community_state))]
            for nid, node in rich_nodes.items():
                rendered = render_rich_node(node, S2CE_COMMUNITY_FORMAT)
                lines.append(rendered)
                lines.append('')
            return '\n'.join(lines)
        except Exception as e:
            # Fallback: compact one-liner listing
            print('[s2ce] Community recall failed: %s — using compact listing' % e,
                  flush=True)
            lines = ['EXISTING COMMUNITIES (%d total):' % len(community_state)]
            for comm in community_state[:20]:
                lines.append('  "%s" (%d members)' % (
                    comm['title'][:60], len(comm['members'])))
            if len(community_state) > 20:
                lines.append('  ... +%d more' % (len(community_state) - 20))
            return '\n'.join(lines)

    # ══════════════════════════════════════════════════════════
    # Proposal formatting (text rendering for the encoder agent)
    # ══════════════════════════════════════════════════════════

    def _format_proposals(self, proposals):
        from servers.scales.s1.surface_contract import _relative_time

        lines = ['PROPOSALS:\n']

        for i, prop in enumerate(proposals):
            ptype = prop['type'].upper().replace('_', ' ')
            lines.append('[%d] %s' % (i + 1, ptype))

            if prop['type'] == 'new_community':
                lines.append('    %d members, int_frac=%.0f%%' % (
                    prop['member_count'],
                    prop['internal_fraction'] * 100))

                # Edge signature — what kind of story
                sig = prop.get('edge_signature', {})
                if sig:
                    parts = ['%s(%.0f%%)' % (f, p * 100)
                             for f, p in sorted(sig.items(),
                                                key=lambda x: -x[1])[:4]]
                    lines.append('    Signature: %s' % ', '.join(parts))

                # Members — one line each with relative time.
                # Encoder can call get_nodes() to inspect specific ones.
                all_members = prop.get('all_members', [])
                if all_members:
                    lines.append('    Members:')
                    for m in all_members:
                        age = _relative_time(m.get('date', '')) or m.get('date', '?')
                        lines.append('      [%s] "%s" (id:%s, %s)' % (
                            m.get('type', '?'),
                            m.get('title', '?'),
                            m.get('id', '?')[:8],
                            age))

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
                node_id = prop.get('node_id', '?')
                lines.append('    Node: [%s] "%s" (node_id: %s)' % (
                    prop.get('node_type', '?'),
                    prop.get('node_title', '?'),
                    node_id[:8] if node_id else '?'))
                if prop.get('source') == 'overlap_check':
                    lines.append('    (Algorithmic placement — not agent-reviewed)')
                for comm in prop.get('communities', []):
                    lines.append('    → connect to "%s" (community_id: %s, affinity: %.0f%%)' % (
                        comm.get('title', '?'),
                        comm.get('id', '?')[:8],
                        comm.get('affinity', 0) * 100))

            elif prop['type'] == 'drift':
                node_id = prop.get('node_id', '?')
                lines.append('    Node: [%s] "%s" (node_id: %s)' % (
                    prop.get('node_type', '?'),
                    prop.get('node_title', '?'),
                    node_id[:8] if node_id else '?'))
                lines.append('    Home: "%s" (affinity: %.0f%%)' % (
                    prop.get('home_community', '?'),
                    prop.get('home_affinity', 0) * 100))
                lines.append('    Current threshold: %.1fx (reject to raise by 0.1)' % (
                    prop.get('current_drift_threshold', 1.5)))
                for f in prop.get('foreign', []):
                    lines.append('    Drifting toward: "%s" (community_id: %s, affinity: %.0f%%)' % (
                        f.get('title', '?'),
                        f.get('id', '?')[:8],
                        f.get('affinity', 0) * 100))

            elif prop['type'] == 'health_update':
                lines.append('    Community: "%s" (community_id: %s)' % (
                    prop.get('community_title', '?'),
                    prop.get('community_id', '?')[:8]))
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
