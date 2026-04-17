"""S2 Consolidation Encoder.

Takes decoder clusters → runs agentic Sonnet → writes consolidation actions.
Uses brain_batch tool through run_llm_loop (same pattern as S2CE community encoder).

The encoder sees clusters formatted as text: pre-classification, similarity scores,
full node content, behavioral evidence, graph context. It decides CONSOLIDATE,
EVOLVE, KEEP, or SKIP for each cluster via tool calls.
"""

import os

from .base import IntegrationUnit
from .consolidation_contract import CONSOLIDATION


class ConsolidationEncoder(IntegrationUnit):
    NAME = 'consolidation'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:consolidation'

    O_SOURCES = ['consolidation_proposals']
    K_SOURCES = ['llm_enrichment', 'consolidation_journal']

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

        self.trace('delta', 'consolidated',
                   '%d actions (%d writes) in %d rounds for %d clusters' % (
                       actions, write_actions, rounds, len(clusters)),
                   metadata={
                       'actions': actions,
                       'write_actions': write_actions,
                       'rounds': rounds,
                       'clusters_processed': len(clusters),
                       'action_details': result.get('action_details', []),
                       'final_text': result.get('final_text', '')[:2000],
                   })

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
            'model', 'claude-sonnet-4-20250514'))
        max_tokens = config.get('max_tokens', self.config.get(
            'max_tokens', 16384))

        if not system_prompt:
            print('[s2-consolidation] WARNING: no enrichment prompt', flush=True)
            return None

        # Inject edge families from DB (latest classification)
        edge_families_config = self._get_interaction_config('s2_edge_families')
        if edge_families_config:
            # Build compact family reference for the prompt
            family_lines = []
            for family, types in sorted(edge_families_config.items()):
                if isinstance(types, list) and family not in ('generic_relation', 'noise'):
                    family_lines.append('- **%s**: %s' % (
                        family, ', '.join(types[:8])))
            if family_lines:
                system_prompt = system_prompt.replace(
                    '## Edge Families',
                    '## Edge Families (loaded from brain — %d families)\n\n%s' % (
                        len(family_lines), '\n'.join(family_lines)),
                    1) if '## Edge Families' in system_prompt else (
                    system_prompt + '\n\n## Edge Families\n\n' + '\n'.join(family_lines))

        if not os.environ.get('ANTHROPIC_API_KEY'):
            load_env()

        tools = self._get_tool_schemas()
        dispatch_fn = self._make_dispatch()
        client = anthropic.Anthropic()

        # Journal text
        journal = self.brain.get_config('s2_consolidation_journal') or ''
        journal_prefix = ("CONSOLIDATION JOURNAL:\n%s\n\n" % journal[
            -self.config.get('journal_max_chars', 14000):]) if journal \
            else "CONSOLIDATION JOURNAL: First run — no previous encoding.\n\n"

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

            # Format this batch
            user_content = journal_prefix + self._format_clusters(batch)

            # Write prompt to tmp file (passive observer for dashboard)
            try:
                import json as _json
                prompt_path = '/tmp/brain-consolidation-prompt-%d.json' % batch_num
                with open(prompt_path, 'w') as _pf:
                    _json.dump({"batch": batch_num, "clusters": len(batch),
                                "user_content": user_content[:50000]}, _pf)
            except Exception:
                pass

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

        # Save journal from last batch's final text
        if total_result['final_text']:
            self._save_journal(total_result['final_text'])

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

    def _make_dispatch(self):
        """Create dispatch function for consolidation tool calls."""
        if self.dispatch:
            return self.dispatch

        from servers.daemon_dispatch import COMMAND_TABLE

        brain = self.brain
        encoding_source = self.ENCODING_SOURCE

        def dispatch(cmd, cmd_args):
            # Force encoding_source + skip_embedding on S2 writes.
            # skip_embedding prevents ONNX multi-thread spin — vectors
            # computed by backfill_vectors() after S2 finishes.
            # Revise ops should NOT get encoding_source — don't change who originally
            # created a node just because S2 touched it.
            if cmd in ('remember', 'remember_batch'):
                if isinstance(cmd_args, dict):
                    cmd_args['encoding_source'] = encoding_source
                    cmd_args['skip_embedding'] = True
            if cmd in ('revise', 'revise_batch'):
                if isinstance(cmd_args, dict):
                    cmd_args['skip_embedding'] = True
            if cmd == 'brain_batch' and isinstance(cmd_args, dict):
                for op in cmd_args.get('operations', []):
                    if isinstance(op, dict):
                        if op.get('op') == 'remember':
                            op['encoding_source'] = encoding_source
                        if op.get('op') in ('remember', 'revise'):
                            op['skip_embedding'] = True
                        if op.get('op') == 'archive':
                            op['archived_by'] = encoding_source

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
        elif 'CONSOLIDATED:' in final_text or 'EVOLVED:' in final_text:
            for marker in ['CONSOLIDATED:', 'EVOLVED:', 'KEPT:', 'SKIPPED:',
                           'OBSERVATIONS:']:
                idx = final_text.find(marker)
                if idx >= 0:
                    journal_entry = final_text[idx:].strip()
                    break

        if not journal_entry:
            return

        existing = self.brain.get_config('s2_consolidation_journal') or ''
        run_header = '--- Consolidation Run %s ---' % self.brain.now()[:10]
        new_journal = existing + '\n' + run_header + '\n' + journal_entry

        max_chars = self.config.get('journal_max_chars', 14000)
        if len(new_journal) > max_chars:
            cutpoint = new_journal.find('--- Consolidation Run',
                                        len(new_journal) - max_chars)
            if cutpoint > 0:
                new_journal = new_journal[cutpoint:]

        self.brain.set_config('s2_consolidation_journal', new_journal.strip())

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
                lines.append('      conf=%.2f  recalled=%dx  judged=%dx  src=%s  created=%s' % (
                    nd.get('confidence', 0), recall_count, judge_count,
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
                except Exception:
                    pass

                # Unique edges (not shared with cluster mates)
                unique_e = cluster.get('unique_edges', {}).get(nid, 0)
                if unique_e > 0:
                    lines.append('      %d unique edges (not shared with cluster)' % unique_e)

                    # Show a few edge details
                    edge_details = cluster.get('edge_details', {}).get(nid, {})
                    shown = 0
                    for nbr_id, edges in edge_details.items():
                        if nbr_id not in cluster['nodes'] and shown < 3:
                            for e in edges[:1]:
                                lines.append('        → [%s] "%s" (%s)' % (
                                    e.get('type', '?'),
                                    e.get('title', '?')[:50],
                                    e.get('relation', '?')))
                            shown += 1

            lines.append('')

        return '\n'.join(lines)
