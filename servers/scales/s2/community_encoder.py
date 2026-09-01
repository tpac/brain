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
import time

from .base import IntegrationUnit
from .community_contract import S2CE_NODE_FORMAT
from servers.trace_contract import build_delta_metadata

from .rejection_table import (
    match_proposals_to_actions,
    record_rejections,
    sort_proposals_by_priority,
    node_ids_touched_by_invalid_ops,
    had_rejected_batch_call,
    get_proposed_ids,
    THWARTED_RETRY_LIMIT,
)


class CommunityEncoder(IntegrationUnit):
    NAME = 'community_detection'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:community_detection'

    O_SOURCES = ['community_proposals']
    K_SOURCES = ['llm_enrichment', 'journal_notes']

    # Consecutive runs whose thwarted brain_batch attempts (invalid ops or a
    # rejected call) shielded proposals from fingerprinting — the give-up
    # bound reads/resets it across cycles.
    THWARTED_STREAK_KEY = 's2_community_thwarted_streak'

    # Residue flows to journal_note trace rows via brain.write_journal_notes,
    # read back via the journal binding's continuity() (the note contract). The old
    # s2_community_journal brain_meta blob is orphaned and no longer read.

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        # Same resolver read as CommunityDecoder — the pipeline passes its
        # config down, so this fires only for standalone construction.
        self.config = config or brain.get_interaction_config('s2_community')

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
        # Community ids that existed BEFORE the agent ran — used after, to
        # find the ones it created (live now, absent here) for the structural
        # stamp.
        pre_community_ids = {c['id'] for c in community_state}
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
            # Thwarted proposals: the encoder TRIED to act and dispatch
            # dropped the attempt — an invalid op naming the proposal's nodes
            # (e.g. `op: reject` instead of `revise`+_sys_drift_threshold),
            # or a rejected brain_batch call (empty/malformed operations),
            # which names no node ids and so taints every un-acted-on
            # proposal. Thwarted-not-decided means retry, not stamp. Bounded:
            # after THWARTED_RETRY_LIMIT consecutive shielded runs, stamp
            # anyway — fingerprints are community's only suppression, so a
            # persistently thwarted encoder would otherwise re-feed the same
            # proposals forever.
            invalid_touched = node_ids_touched_by_invalid_ops(action_details)
            rejected_call = had_rejected_batch_call(action_details)
            invalid_op_failures = 0
            retry, kept = [], []
            for p in skipped_proposals:
                if rejected_call or (invalid_touched
                                     and set(get_proposed_ids(p)) & invalid_touched):
                    retry.append(p)
                else:
                    kept.append(p)
            streak = int(self.brain.get_config(self.THWARTED_STREAK_KEY) or 0)
            if retry and streak < THWARTED_RETRY_LIMIT:
                invalid_op_failures = len(retry)
                skipped_proposals = kept
                self.brain.set_config(self.THWARTED_STREAK_KEY, str(streak + 1))
                cause = ('a rejected brain_batch call' if rejected_call
                         else 'invalid brain_batch ops')
                self.brain._log_warning(
                    's2_community_thwarted_retry',
                    '%d proposal(s) thwarted by %s — retrying next cycle, '
                    'NOT suppressed' % (invalid_op_failures, cause))
                print('[s2ce] %d proposal(s) thwarted by %s — retry, NOT suppressed'
                      % (invalid_op_failures, cause), flush=True)
            elif retry:
                self.brain.set_config(self.THWARTED_STREAK_KEY, '0')
                self.brain._log_warning(
                    's2_community_thwarted_giveup',
                    'thwarted brain_batch attempts in %d shielded runs — '
                    'stamping %d proposal(s) anyway to unpin the unit'
                    % (streak, len(retry)))
            elif int(self.brain.get_config(self.THWARTED_STREAK_KEY) or 0):
                self.brain.set_config(self.THWARTED_STREAK_KEY, '0')
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

        # Membership restorer runs BEFORE the delta write (moved 2026-08-07,
        # plan step 10) so its backfill lands in THIS run's delta: the
        # restorer is direct-DAL — the mutation emitter can't see it — so the
        # community unit's own delta is where the S2 story records it
        # (count + edge ids, ruled 2026-08-04). See the block comment below.
        recon = {'communities_healed': 0, 'edges_backfilled': 0,
                 'details': [], 'edge_ids': []}
        reconciled_ids = []
        try:
            with self.brain.write_lock:
                recon = self.brain._graph.reconcile_community_membership()
            reconciled_ids = [cid for cid, _ in recon.get('details', [])]
            if recon['edges_backfilled']:
                # Auto-heal event → _log_warning (not _log_error): it's a
                # repaired inconsistency, not a failure, and warning-level
                # keeps the error-triage view clean. Still loud/surfaced.
                self.brain._log_warning(
                    'community_membership_backfilled',
                    'community declared members but held 0 edges — back-filled',
                    'back-filled %d member edge(s) across %d orphaned '
                    'community(ies): %s' % (
                        recon['edges_backfilled'], recon['communities_healed'],
                        ', '.join('%s+%d' % (c[:8], n)
                                  for c, n in recon['details'][:10])))
                print('[s2ce] membership reconcile: +%d edge(s) across %d '
                      'community(ies)' % (recon['edges_backfilled'],
                                          recon['communities_healed']), flush=True)
        except Exception as e:
            self.brain._log_error('community_membership_reconcile', e,
                                  'membership restorer failed')

        self.trace('delta', 'community_enriched',
                   'COMPLETE: %d actions (%d writes) in %d rounds, %d stamped, '
                   '%dms, %d→%d tok' % (
                       actions, write_actions, rounds,
                       result.get('rejection_skipped_count', 0),
                       result.get('elapsed_ms', 0),
                       result.get('input_tokens', 0),
                       result.get('output_tokens', 0)),
                   metadata=build_delta_metadata(
                       actions=actions,
                       write_actions=write_actions,
                       rounds=rounds,
                       inputs_processed=len(encoder_proposals),
                       outcomes=outcomes,
                       rejection_skipped=result.get('rejection_skipped_count', 0),
                       invalid_op_failures=result.get('invalid_op_failures', 0),
                       action_details=action_details,
                       read_calls=result.get('read_calls', []),
                       final_text=final_text,
                       corridors_filtered=len(corridors),
                       elapsed_ms=result.get('elapsed_ms', 0),
                       input_tokens=result.get('input_tokens', 0),
                       output_tokens=result.get('output_tokens', 0),
                       cache_read_tokens=result.get('cache_read_tokens', 0),
                       cache_creation_tokens=result.get('cache_creation_tokens', 0),
                       membership_reconciled={
                           'communities_healed': recon.get('communities_healed', 0),
                           'edges_backfilled': recon.get('edges_backfilled', 0),
                           'edge_ids': recon.get('edge_ids', []),
                       },
                   ))

        # (Membership restorer: see the block above the delta write — a
        # community declaring N members with ZERO edges gets its edges
        # back-filled from the declaration. Direct-DAL, recorded in the
        # delta's membership_reconciled; idempotent, self-quiets.)

        # Second, ALGORITHMIC Δ: derive + stamp the structural fields from the
        # now-final edge state (post-agent, post-reconcile). These are pure
        # counts over community_member edges — the agent no longer writes them
        # (it guessed blind and they drifted). Runs here precisely so it reads
        # the agent's new edges + reconcile's backfills. Failure-isolated — a
        # stamp failure never breaks the run.
        try:
            self._stamp_structural_fields(
                encoder_proposals, pre_community_ids, reconciled_ids)
        except Exception as e:
            self.brain._log_error('community_structural_stamp', e,
                                  'structural stamp failed')

        return result

    def _stamp_structural_fields(self, encoder_proposals,
                                 pre_community_ids, reconciled_ids):
        """Per-encode Δ: stamp the structural fields for every community this
        run touched, derived from the final member edges.

        Touched = created (live now, absent before) + added-to / merged /
        health-updated / drifted-into (from the proposals) + reconcile-healed.
        Existing wrong values self-heal as their community is next touched; the
        one-time fill corrects the rest. Gate-safe: community-node metadata
        revises are ignored by the idle gate (type='community') and create no
        edges. See docs/COMMUNITY-METADATA-DENORMALIZATION.md.
        """
        touched = set(reconciled_ids)
        for p in encoder_proposals:
            t = p.get('type')
            if t == 'health_update' and p.get('community_id'):
                touched.add(p['community_id'])
            elif t == 'merge_communities' and p.get('larger_id'):
                touched.add(p['larger_id'])
            elif t == 'add_to_existing':
                touched.update(c['id'] for c in p.get('communities', [])
                               if c.get('id'))
            elif t == 'drift':
                touched.update(f['id'] for f in p.get('foreign', [])
                               if f.get('id'))

        # Newly-created communities OF THIS UNIT: live now, absent before the
        # run. Source-filtered to match pre_community_ids (the decoder's
        # _read_community_state reads only this unit's communities) — else a
        # community authored elsewhere is absent from pre and would read as
        # "new" (and get re-stamped) every cycle.
        new_ids = {r[0] for r in self.brain.conn.execute(
            "SELECT id FROM nodes WHERE type = 'community' AND archived = 0 "
            "AND encoding_source = ?", (self.ENCODING_SOURCE,)).fetchall()}
        touched.update(new_ids - pre_community_ids)
        return self._stamp_ids(touched)

    def _stamp_ids(self, community_ids):
        """Derive + stamp the structural fields for the given communities (the
        ones still live), via one brain_batch of revise ops. Shared by the
        per-encode Δ and the one-time backfill. Returns the count stamped.
        """
        from .community_structural import compute_community_structural

        ids = sorted({c for c in community_ids if c})
        if not ids:
            return 0
        # Keep only currently-live communities — a merge's smaller / a health
        # archive may have removed some, and we never stamp an archived node.
        placeholders = ','.join('?' * len(ids))
        live = {r[0] for r in self.brain.conn.execute(
            "SELECT id FROM nodes WHERE type = 'community' AND archived = 0 "
            "AND id IN (%s)" % placeholders, ids).fetchall()}
        ids = [c for c in ids if c in live]
        if not ids:
            return 0

        derived = compute_community_structural(self.brain, ids)
        ops = []
        for cid, fields in derived.items():
            op = {
                'op': 'revise',
                'node_id': cid,
                'reason': 'structural fields derived from member edges',
                'community_size': str(fields['community_size']),
                'community_internal_fraction':
                    str(fields['community_internal_fraction']),
                'community_is_corridor':
                    'true' if fields['community_is_corridor'] else 'false',
            }
            if fields['community_dominant_type']:
                op['community_dominant_type'] = fields['community_dominant_type']
            ops.append(op)
        if not ops:
            return 0

        dispatch_fn = self._make_dispatch()
        dispatch_fn('brain_batch', {
            'operations': ops,
            'encoding_source': self.ENCODING_SOURCE,
        })
        # Bare marker (no build_delta_metadata payload) — the contract lets
        # delta ref_types double as markers; a partial dict would trip the
        # required-keys guard. The count lives in the summary string.
        self.trace('delta', 'community_enriched',
                   'STRUCTURAL STAMP: %d communities (size/int_frac/is_corridor/'
                   'dominant_type derived from edges)' % len(ops))
        print('[s2ce] structural stamp: %d communities' % len(ops), flush=True)
        return len(ops)

    def backfill_all_communities(self, chunk=100):
        """One-time fill: stamp the structural fields for EVERY live community
        from its edges. The per-encode Δ only touches communities a run acted
        on; this corrects the existing backlog of Haiku-authored values in one
        pass. Chunked so each brain_batch transaction stays modest. Idempotent —
        re-running just re-derives the same values. Returns the count stamped.

        OFFLINE MAINTENANCE ENTRY POINT — invoked by hand under the maintenance
        lock, so it has no production caller by design (the `brain.backfill_
        community_structural()` one-shot wrapper was retired 2026-08-07 once its
        fill completed). Reads as TEST_ONLY to a dead-code scan; it is not dead.
        """
        live_ids = [r[0] for r in self.brain.conn.execute(
            "SELECT id FROM nodes WHERE type = 'community' "
            "AND archived = 0").fetchall()]
        total = 0
        for i in range(0, len(live_ids), chunk):
            total += self._stamp_ids(live_ids[i:i + chunk])
        print('[s2ce] backfill complete: %d communities stamped' % total,
              flush=True)
        return total

    # ══════════════════════════════════════════════════════════
    # _encode — S2 Community Encoder agent via run_llm_loop
    # ══════════════════════════════════════════════════════════

    def _encode(self, proposals, community_state):
        """Run S2CE in batches — processes ALL proposals.

        Proposals are split into chunks of max_proposals_per_call.
        Each chunk gets one encoder-agent call. Between chunks, community_state
        is refreshed so later chunks see what earlier ones created.
        """
        from ..dispatch import load_env
        from ..runner import run_llm_loop, make_client, retry_on_transient_api_error
        from .community_decoder import CommunityDecoder

        system_prompt = self.brain.get_interaction_prompt(
            's2_community_enrichment')
        config = self.brain.get_interaction_config('s2_community_enrichment')
        model = config['model']
        max_tokens = config['max_tokens']

        # Prompt closers — the edge-aspect vocabulary, then the journal
        # component's system-tail decoration (review block + closure, DONE
        # genuinely last). Single-sourced: aspects in servers/aspects.py, the
        # journal blocks in trace_contract via scales/journal.py.
        system_prompt = self._inject_edge_aspects(system_prompt)
        system_prompt = self.journal.decorate_system(system_prompt)

        if not os.environ.get('ANTHROPIC_API_KEY'):
            load_env()

        tools = self._get_tool_schemas()
        dispatch_fn = self._make_dispatch()
        client = make_client()

        # Residue continuity — the last few runs' review notes, rendered by base.
        journal_prefix = self.journal.continuity()

        # Batch proposals
        batch_size = self.config.get('max_proposals_per_call', 15)
        total_result = {
            'rounds': 0, 'actions': 0, 'write_actions': 0,
            'action_details': [], 'read_calls': [], 'final_text': '',
        }
        # Cost/latency telemetry — loop counts, per-tool records, and tokens are
        # folded per batch by the shared _accumulate_run; elapsed by a wall-clock
        # timer around the whole batch loop. Mirrors S1 Scribe / consolidation;
        # before this, run()'s build_delta_metadata omitted them and every
        # production `community_enriched` delta read elapsed_ms=0,
        # output_tokens=0 (the gap).
        _t0 = time.time()
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
                        get_nodes_config=S2CE_NODE_FORMAT,
                        log_fn=lambda msg: print('[s2ce] %s' % msg, flush=True),
                        record_round_fn=self.brain.round_recorder(
                            self.chain_id(), seq_base=batch_num * 100)),
                    log_fn=lambda msg: print('[s2ce] %s' % msg, flush=True))

                # Accumulate + per-batch journal + truncation logging — shared
                # multi-batch body (see IntegrationUnit._fold_batch_result).
                self._fold_batch_result(
                    total_result, result, batch_num, 's2ce_truncation',
                    trunc_detail='tool call likely corrupted, community data may be lost')

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

        total_result['elapsed_ms'] = int((time.time() - _t0) * 1000)
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

                ov = prop.get('overlaps_existing')
                if ov:
                    lines.append(
                        '    ⚠ %.0f%% of members connect into existing '
                        'community "%s" (community_id: %s) — judge: '
                        'genuinely new story, or an extension of that one?' % (
                            ov.get('connected_frac', 0) * 100,
                            ov.get('title', '?'),
                            (ov.get('id') or '?')[:8]))

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
