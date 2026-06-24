"""S2 Consolidation Encoder.

Takes decoder clusters → runs agentic Sonnet → writes consolidation actions.
Uses brain_batch tool through run_llm_loop (same pattern as S2CE community encoder).

The encoder sees clusters formatted as text: pre-classification, similarity scores,
full node content, behavioral evidence, graph context. It decides CONSOLIDATE,
EVOLVE, KEEP, or SKIP for each cluster via tool calls.
"""

import os

from servers.trace_contract import build_delta_metadata
from servers.daemon_config import brain_tmp_dir

from .base import IntegrationUnit
from .consolidation_contract import CONSOLIDATION


class ConsolidationEncoder(IntegrationUnit):
    NAME = 'consolidation'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:consolidation'

    O_SOURCES = ['consolidation_proposals']
    K_SOURCES = ['llm_enrichment', 'journal_notes']

    # Legacy journal RETIRED (2026-06-23 journal redesign Phase 3): residue now
    # flows to journal_note trace rows via brain.write_journal_notes, read back
    # via _load_journal_notes_prefix. Empty JOURNAL_MARKERS makes the inherited
    # _load_journal_prefix / _save_journal no-ops — we use the note contract.
    JOURNAL_MARKERS = ()

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or CONSOLIDATION

    def run(self, clusters):
        """Encode clusters into consolidation actions.

        Args:
            clusters: List of enriched cluster dicts from decoder.
                      Each has: nodes, node_details, pre_class,
                      similarity scores, behavioral evidence.

        Returns:
            dict with keys: actions, write_actions, rounds,
                            action_details, final_text
            None on failure.
        """
        if not clusters:
            self.trace('delta', 'consolidated', 'No clusters to process')
            return {'actions': 0, 'write_actions': 0, 'rounds': 0,
                    'action_details': [], 'final_text': ''}

        result = self._encode(clusters)
        if not result:
            self.trace('delta', 'consolidated', 'Encoding failed')
            return None

        actions = result.get('actions', 0)
        write_actions = result.get('write_actions', 0)
        rounds = result.get('rounds', 0)
        final_text = result.get('final_text', '') or ''

        # No marker-count `outcomes`: the CONSOLIDATED:/EVOLVED:/KEPT:/SKIPPED:
        # markers retired with the legacy journal, so counting them would report
        # all-zeros (a silent-lie the dashboard would render). The accurate
        # per-op effect is the delta's created/revised/archived (auto-aggregated
        # from action_details); the orchestrator's edge-diff is authoritative
        # for suppression. journal_entry is likewise gone — residue is its own
        # journal_note rows now.
        self.trace('delta', 'consolidated',
                   '%d actions (%d writes) in %d rounds for %d clusters' % (
                       actions, write_actions, rounds, len(clusters)),
                   metadata=build_delta_metadata(
                       actions=actions,
                       write_actions=write_actions,
                       rounds=rounds,
                       inputs_processed=len(clusters),
                       action_details=result.get('action_details', []),
                       final_text=final_text,
                       clusters_processed=len(clusters),
                   ))

        return result

    # ══════════════════════════════════════════════════════════
    # _encode — agentic Sonnet via run_llm_loop
    # ══════════════════════════════════════════════════════════

    def _encode(self, clusters):
        """Run consolidation encoder in batches.

        Clusters are split into chunks of max_proposals_per_call.
        Each chunk gets one Sonnet call.
        """
        import anthropic
        from ..dispatch import load_env
        from ..runner import run_llm_loop

        system_prompt = self.brain.get_interaction_prompt(
            's2_consolidation_enrichment')
        config = self._get_interaction_config('s2_consolidation_enrichment') or {}
        model = config.get('model', self.config.get(
            'model', 'claude-sonnet-4-6'))
        max_tokens = config.get('max_tokens', self.config.get(
            'max_tokens', 32768))

        if not system_prompt:
            print('[s2-consolidation] WARNING: no enrichment prompt', flush=True)
            return None

        # Inject edge aspects (Step 8 of unified-aspects). Same data the old
        # s2_edge_families interaction held, now read from brain.aspects.
        # Skip the two no-signal aspects (generic_relation, noise) and any
        # aspect with empty edge_relations (node-only — irrelevant to a
        # cluster-merge prompt).
        family_lines = []
        for name, aspect in sorted(self.brain.aspects.all().items()):
            if name in ('generic_relation', 'noise'):
                continue
            if not aspect.edge_relations:
                continue
            family_lines.append('- **%s**: %s' % (
                name, ', '.join(list(aspect.edge_relations[:8]))))
        if family_lines:
            system_prompt = system_prompt.replace(
                '## Edge Families',
                '## Edge Families (loaded from brain.aspects — %d aspects)\n\n%s' % (
                    len(family_lines), '\n'.join(family_lines)),
                1) if '## Edge Families' in system_prompt else (
                system_prompt + '\n\n## Edge Families\n\n' + '\n'.join(family_lines))

        # Journal section is contract-owned, injected at runtime (like ## Edge
        # Families above): strip the legacy section, relabel the continuity-read
        # line, append the shared review block. The mechanism lives in base so
        # community/S1E inherit it — only the per-encoder anchors live here.
        system_prompt = self._inject_review_block(
            system_prompt,
            legacy_heading='## Encoding Journal',
            relabels=[
                ('- **CONSOLIDATION JOURNAL** — what previous runs decided. Your continuity.',
                 '- **RECENT REVIEW NOTES** — residue your recent runs flagged (doubt, '
                 'friction, surprise); empty if they were clean. Your continuity.'),
                ('Round 2: journal + DONE.', 'Round 2: review + DONE.'),
            ])

        if not os.environ.get('ANTHROPIC_API_KEY'):
            load_env()

        tools = self._get_tool_schemas()
        # Single shared S2 timeout — see base.ANTHROPIC_CLIENT_TIMEOUT.
        from .base import ANTHROPIC_CLIENT_TIMEOUT
        client = anthropic.Anthropic(timeout=ANTHROPIC_CLIENT_TIMEOUT)

        # Residue continuity — the last few runs' review notes, rendered by base.
        journal_prefix = self._load_journal_notes_prefix()

        # Batch clusters
        batch_size = self.config.get('max_proposals_per_call', 10)
        total_result = {
            'rounds': 0, 'actions': 0, 'write_actions': 0,
            'action_details': [], 'final_text': '',
        }

        for batch_idx in range(0, len(clusters), batch_size):
            batch = clusters[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(clusters) + batch_size - 1) // batch_size

            print('[s2-consolidation] Batch %d/%d (%d clusters)' % (
                batch_num, total_batches, len(batch)), flush=True)

            # Build the set of node IDs legal to archive in this batch.
            # Defense against encoder drift: if Sonnet emits an archive op for
            # a node NOT in the current batch's clusters, the dispatch closure
            # rejects it. This guards against the class of bug where the
            # encoder archived nodes from unrelated clusters (observed in
            # pre-survive-and-absorb runs).
            valid_archive_ids = {nid for c in batch for nid in c['nodes']}
            dispatch_fn = self._make_dispatch(valid_archive_ids=valid_archive_ids)

            # Format this batch
            user_content = journal_prefix + self._format_clusters(batch)

            # Write prompt to tmp file (passive observer for dashboard)
            try:
                import json as _json
                prompt_path = os.path.join(
                    brain_tmp_dir(), 'brain-consolidation-prompt-%d.json' % batch_num)
                with open(prompt_path, 'w') as _pf:
                    _json.dump({"batch": batch_num, "clusters": len(batch),
                                "user_content": user_content[:50000]}, _pf)
            except Exception:
                pass

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
                        log_fn=lambda msg: print('[s2-consolidation] %s' % msg, flush=True)),
                    log_fn=lambda msg: print('[s2-consolidation] %s' % msg, flush=True))

                total_result['rounds'] += result.get('rounds', 0)
                total_result['actions'] += result.get('actions', 0)
                total_result['write_actions'] += result.get('write_actions', 0)
                total_result['action_details'].extend(
                    result.get('action_details', []))
                total_result['final_text'] = result.get('final_text', '')

                # Log truncation errors
                for trunc in result.get('truncations', []):
                    self.brain._log_error(
                        's2_consolidation_truncation',
                        'max_tokens truncation: round %d used %s/%s output tokens' % (
                            trunc['round'], trunc['output_tokens'], trunc['max_tokens']),
                        'batch %d — tool call likely corrupted' % batch_num)

            except Exception as e:
                print('[s2-consolidation] BATCH %d FAILED: %s' % (batch_num, e), flush=True)
                self.brain._log_error(self.NAME, e,
                                      'encode batch %d' % batch_num)

        # Residue review → journal_note trace rows (the new journal). Shares
        # this run's chain_id so the notes group with the run's ops-delta.
        # write_journal_notes is failure-isolated — a journal write never breaks
        # the run. Legacy brain_meta journal is retired (JOURNAL_MARKERS=()).
        if total_result['final_text']:
            self.brain.write_journal_notes(
                final_text=total_result['final_text'],
                chain_id=self.chain_id(), scale=self.SCALE, session_id='')

        return total_result

    # ══════════════════════════════════════════════════════════
    # Tool infrastructure
    # ══════════════════════════════════════════════════════════

    def _get_tool_schemas(self):
        """Consolidation tools: brain_batch + get_nodes only.

        Don't expose remember_batch/revise_batch/connect_batch as separate
        tools — their schemas leak fields (like encoding_source) that Sonnet
        then fills in, overriding the dispatch's forced values.
        The prompt examples teach brain_batch's operation shapes.
        """
        from servers import brain_mcp
        TOOLS = {'brain_batch', 'get_nodes'}
        return [{"name": t["name"], "description": t["description"],
                 "input_schema": t["inputSchema"]}
                for t in brain_mcp.TOOLS if t["name"] in TOOLS]

    def _make_dispatch(self, valid_archive_ids=None):
        """Consolidation dispatch — delegates to base `_make_encoder_dispatch`
        with an archive guard restricting archives to current-batch cluster
        members (prevents encoder drift archiving unrelated nodes).
        """
        return self._make_encoder_dispatch(archive_guard=valid_archive_ids)

    # Journal save/load is inherited from IntegrationUnit.
    # Class attributes JOURNAL_MARKERS / JOURNAL_LABEL / JOURNAL_RUN_HEADER
    # configure the pattern; base owns the logic.

    # ══════════════════════════════════════════════════════════
    # Cluster formatting (text rendering for Sonnet)
    # ══════════════════════════════════════════════════════════

    def _format_clusters(self, clusters):
        from servers.contract import render_rich_node
        from .consolidation_contract import CONSOLIDATION_NODE_FORMAT, CLUSTER_REQUIRED_FIELDS

        # Validate cluster shape against contract
        for i, c in enumerate(clusters):
            missing = CLUSTER_REQUIRED_FIELDS - set(c.keys())
            if missing:
                print('[s2-consolidation] WARNING: cluster %d missing fields: %s' % (
                    i, missing), flush=True)

        lines = ['CLUSTERS:\n']

        for i, cluster in enumerate(clusters):
            pre = cluster.get('pre_class', 'needs_judgment')
            size = cluster['size']
            c_max = cluster['content_cosine_max']
            t_max = cluster['title_cosine_max']

            lines.append('[%d] %s (size=%d, content_cosine=%.3f, title_cosine=%.3f)' % (
                i + 1, pre.upper(), size, c_max, t_max))

            # Behavioral evidence summary
            co_recall = cluster.get('co_recall_count', 0)
            shared_e = cluster.get('shared_edge_count', 0)
            same_comm = cluster.get('same_community', False)
            has_corr = cluster.get('has_correction_edge', False)
            blind_any = any(cluster.get('catalog_blind', {}).values())

            has_tension = cluster.get('has_tension_edge', False)

            evidence = []
            if co_recall > 0:
                evidence.append('co_recall=%d' % co_recall)
            if shared_e > 0:
                evidence.append('shared_edges=%d' % shared_e)
            if same_comm:
                evidence.append('same_community')
            if has_corr:
                evidence.append('CORRECTION_EDGE')
            if has_tension:
                evidence.append('⚠ TENSION_EDGE — these nodes CONTRADICT or CHALLENGE each other')
            if blind_any:
                evidence.append('CATALOG_BLIND')
            if evidence:
                lines.append('    Evidence: %s' % ', '.join(evidence))

            # Judge preference
            judge = cluster.get('judge_preference', {})
            if any(v > 0 for v in judge.values()):
                parts = ['%s=%dx' % (nid[:8], v) for nid, v in judge.items() if v > 0]
                lines.append('    Judge preference: %s' % ', '.join(parts))

            # Query coverage
            qc = cluster.get('query_coverage', {})
            all_queries = set()
            for queries in qc.values():
                all_queries.update(queries)
            if all_queries:
                lines.append('    Queries that find these: %s' %
                             ', '.join(list(all_queries)[:5]))

            # Community membership
            comms = cluster.get('communities', {})
            comm_names = set()
            for node_comms in comms.values():
                for c in node_comms:
                    comm_names.add(c.get('title', '?'))
            if comm_names:
                lines.append('    Communities: %s' % ', '.join(list(comm_names)[:3]))

            # Render each node richly
            lines.append('    Nodes:')
            for nid in cluster['nodes']:
                nd = cluster.get('node_details', {}).get(nid, {})
                locked = nd.get('locked', False)
                critical = nd.get('critical', False)
                recall_count = cluster.get('recall_counts', {}).get(nid, 0)
                judge_count = cluster.get('judge_preference', {}).get(nid, 0)

                flags = []
                if locked:
                    flags.append('LOCKED')
                if critical:
                    flags.append('CRITICAL')
                flag_str = ' [%s]' % ', '.join(flags) if flags else ''

                lines.append('    --- %s ---' % nid[:8])
                lines.append('      [%s] "%s"%s' % (
                    nd.get('type', '?'), nd.get('title', '?'), flag_str))
                lines.append('      recalled=%dx  judged=%dx  src=%s  created=%s' % (
                    recall_count, judge_count,
                    nd.get('encoding_source', '?')[:15],
                    nd.get('created_at', '?')[:10]))

                # Catalog blindness per node
                if cluster.get('catalog_blind', {}).get(nid, False):
                    lines.append('      ⚠ CATALOG BLIND — created without seeing other cluster members')

                # Rich node content (using consolidation format — more depth than community)
                try:
                    rich = self.brain.get_node(nid)
                    if rich:
                        rendered = render_rich_node(rich, CONSOLIDATION_NODE_FORMAT)
                        lines.append('      ' + rendered.replace('\n', '\n      '))
                    else:
                        content = nd.get('content', '')
                        if content:
                            lines.append('      Content: %s' % content[:600])
                except Exception:
                    content = nd.get('content', '')
                    if content:
                        lines.append('      Content: %s' % content[:600])

                # Surface ALL metadata KV — emergent fields must survive consolidation.
                # Don't hardcode keys — any brain may have domain-specific fields
                # (a poet's emotional_tone, a lawyer's jurisdiction, etc.)
                try:
                    kv_rows = self.brain.conn.execute(
                        "SELECT key, value FROM node_metadata_kv "
                        "WHERE node_id = ? AND value IS NOT NULL AND value != ''",
                        (nid,)).fetchall()
                    kv_fields = [(k, v) for k, v in kv_rows
                                 if not k.startswith('_sys_')]  # system fields hidden
                    if kv_fields:
                        lines.append('      Metadata:')
                        for k, v in kv_fields:
                            lines.append('        %s: %s' % (k, v[:200]))
                except Exception as e:
                    # Metadata rendering is best-effort (decoder already has
                    # the core cluster data), but a failing KV read means the
                    # encoder sees the node stripped of emergent fields —
                    # degrades consolidation quality silently. Log once.
                    self.brain._log_error(
                        's2_consolidation_metadata_kv_render', e,
                        'node %s metadata skipped in cluster rendering' % nid[:8])

                # External edges — every edge to a non-cluster-member
                # neighbor, with direction, relation, and description.
                # The encoder reads these to reason about ABSORB migration:
                # survivor keeps its own, the peer's outgoing edges migrate
                # via the survivor's connections list, and incoming edges
                # migrate via separate `connect` ops from the neighbor.
                edge_details = cluster.get('edge_details', {}).get(nid, {})
                external = {nbr: es for nbr, es in edge_details.items()
                            if nbr not in cluster['nodes']}
                if external:
                    total = sum(len(es) for es in external.values())
                    lines.append('      External edges (%d):' % total)
                    for nbr_id, edges_list in external.items():
                        for e in edges_list:
                            arrow = '→' if e.get('direction') == 'outgoing' else '←'
                            desc = e.get('description', '')
                            desc_str = ' — %s' % desc if desc else ''
                            lines.append('        %s %s [%s] "%s" (%s)%s' % (
                                arrow, nbr_id[:8],
                                e.get('type', '?'),
                                e.get('title', '?')[:50],
                                e.get('relation', '?'),
                                desc_str))

            lines.append('')

        return '\n'.join(lines)
