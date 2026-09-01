"""S2 Consolidation — orchestrator.

Wires ConsolidationDecoder (algorithmic) → ConsolidationEncoder (agentic Sonnet).
This file is the public API — callers import Consolidation from here.

Decoder: servers/scales/s2/consolidation_decoder.py
Encoder: servers/scales/s2/consolidation_encoder.py
Contract: servers/scales/s2/consolidation_contract.py
Prompt: servers/scales/s2/consolidation_enrichment_prompt.py
"""

from .consolidation_decoder import ConsolidationDecoder
from .consolidation_encoder import ConsolidationEncoder
from .rejection_table import (
    filter_rejected, record_rejections, node_ids_touched_by_invalid_ops,
    node_ids_touched_by_valid_ops, had_rejected_batch_call,
    REJECTED_BATCH_RETRY_LIMIT)


class Consolidation(ConsolidationDecoder):
    """Full S2 consolidation pipeline — decode clusters then encode via Sonnet.

    Inherits from ConsolidationDecoder so all decoder methods are available.
    Callers that only need the decoder can use ConsolidationDecoder directly.
    """

    # Consecutive runs whose rejected brain_batch call shielded clusters from
    # fingerprinting — the give-up bound reads/resets it across cycles.
    REJECTED_STREAK_KEY = 's2_consolidation_rejected_call_streak'

    def run(self):
        """Run full decode → encode pipeline.

        Returns:
            dict with: actions, clusters, stats, details, skipped, error
        """
        # Decode
        decode_result = super().run()

        if decode_result.get('skipped'):
            return {'actions': 0, 'skipped': decode_result['skipped']}

        clusters = decode_result['clusters']
        stats = decode_result.get('stats', {})

        if not clusters:
            self._record_scan_baseline(decode_result)
            return {'actions': 0, 'clusters': 0, 'stats': stats}

        # Fingerprint-based rejection filter — SKIP decisions from prior
        # runs populate s2_rejections; those clusters don't resurface here.
        # `member_updated_at` folds into the fingerprint so any revise()
        # on a member invalidates the rejection automatically (cluster IDs
        # unchanged + edited content ⇒ the encoder's judgment may differ
        # from its prior SKIP and must re-evaluate).
        # Deliberately updated_at, NOT revised_at: this is the cheap
        # invalidate-on-any-change tripwire, and re-proposal still requires
        # the pair to re-form in a scan — which the change set gates on
        # revised_at (claim changes only). A metadata-only revise
        # invalidates the fingerprint but never re-forms the pair, so
        # nothing re-proposes; switching this key to revised_at would
        # orphan every stored fingerprint for no behavioral gain.
        proposals = [{'type': 'consolidation_cluster',
                      'members': c.get('nodes', []),
                      'member_updated_at': {
                          nid: c.get('node_details', {}).get(nid, {}).get('updated_at', '')
                          for nid in c.get('nodes', [])
                      },
                      '_cluster': c}
                     for c in clusters]
        surviving, fp_suppressed = filter_rejected(self.brain, proposals)
        clusters = [p['_cluster'] for p in surviving]
        stats['fingerprint_suppressed'] = fp_suppressed
        if fp_suppressed:
            print('[s2-consolidation] Fingerprint-suppressed %d clusters' % fp_suppressed,
                  flush=True)

        if not clusters:
            self._record_scan_baseline(decode_result)
            return {'actions': 0, 'clusters': 0, 'stats': stats}

        # Cap clusters per run (graduated cold start).
        # Process easiest first — likely_consolidate before needs_judgment.
        max_per_run = self.config.get('max_clusters_per_run', 30)
        if len(clusters) > max_per_run:
            priority = {'likely_consolidate': 0, 'likely_evolve': 1,
                        'likely_keep': 2, 'needs_judgment': 3}
            clusters.sort(key=lambda c: (
                priority.get(c.get('pre_class', 'needs_judgment'), 3),
                -c.get('content_cosine_max', 0)))
            clusters = clusters[:max_per_run]
            print('[s2-consolidation] Capped to %d clusters (of %d total)' % (
                max_per_run, stats.get('clusters_formed', '?')), flush=True)

        # Snapshot which cluster members are already archived. An ABSORB
        # (whether via the `absorb` op or the legacy revise+archive dance)
        # archives the absorbed peer but writes NO similar_to/consolidated_into
        # edge — so edge-detection alone reads a *successful* merge as a SKIP
        # and stamps a false rejection fingerprint. A member flipping
        # archived 0→1 across the encoder run is the merge's signature.
        all_member_ids = sorted(
            {nid for c in clusters for nid in c.get('nodes', [])})

        def _snapshot_archived(ids):
            archived = set()
            for k in range(0, len(ids), 900):
                chunk = ids[k:k + 900]
                if not chunk:
                    continue
                ph = ','.join('?' * len(chunk))
                for row in self.brain.conn.execute(
                    "SELECT id FROM nodes WHERE id IN (%s) AND archived = 1" % ph,
                        chunk):
                    archived.add(row[0])
            return archived

        pre_archived = _snapshot_archived(all_member_ids)

        # Encode
        encoder = ConsolidationEncoder(
            self.brain, self.dispatch, self.config)
        encode_result = encoder.run(clusters)

        if not encode_result:
            return {'actions': 0, 'clusters': len(clusters),
                    'stats': stats, 'error': 'encoding failed'}

        # Fingerprint EVERY processed cluster (2026-07-27) — suppression
        # follows the encoder's decision itself, not its edge vocabulary.
        # Previously only no-new-edge clusters were fingerprinted and edges
        # did the rest — but the decoder's skip-list is a closed verb set
        # (the settlement aspect) while the encoder resolves pairs in open
        # text (`addresses` ×240, `reframes` ×101 live edges, plus the
        # prompt's own "link survivors with a REAL relation"), so
        # verb-mismatched resolutions re-proposed every cycle (journal
        # finding #4, id:2ace76f3).
        # A fingerprint invalidates when a member's updated_at changes —
        # a real write, now that access marks no longer touch it — so
        # legitimate re-review after content change is preserved.
        # Excluded from fingerprinting:
        # - ABSORBed clusters (member archived this run): the scan's
        #   archived=0 filter already suppresses them; stamping would
        #   pollute s2_rejections.
        # - Invalid-op clusters: the merge was thwarted, not decided —
        #   never stamp, force a retry next cycle. Checked BEFORE the
        #   archived exclusion: a cluster can carry both a landed absorb and
        #   a thwarted op, and treating it as resolved would advance the
        #   baseline past the thwarted half.
        # - A rejected brain_batch call (empty/malformed operations) names no
        #   node ids, so it can't be attributed — treat every cluster no
        #   valid op touched as thwarted (retry), not decided. Bounded: after
        #   REJECTED_BATCH_RETRY_LIMIT consecutive shielded runs, stamp
        #   anyway — an encoder that persistently rejects otherwise pins the
        #   unit forever (no fingerprints, no baseline, full re-encode every
        #   cycle).
        details = encode_result.get('action_details', [])
        newly_archived = _snapshot_archived(all_member_ids) - pre_archived
        invalid_touched = node_ids_touched_by_invalid_ops(details)
        rejected_call = had_rejected_batch_call(details)
        valid_touched = node_ids_touched_by_valid_ops(details) if rejected_call else set()

        streak = int(self.brain.get_config(self.REJECTED_STREAK_KEY) or 0)
        shield_active = rejected_call and streak < REJECTED_BATCH_RETRY_LIMIT
        if rejected_call and not shield_active:
            self.brain._log_warning(
                's2_consolidation_rejected_call_giveup',
                'rejected brain_batch call in %d consecutive runs — '
                'stamping fingerprints anyway to unpin the unit' % (streak + 1))

        processed_proposals = []
        fingerprint_members = set()
        invalid_op_clusters = 0
        rejected_call_clusters = 0
        for c in clusters:
            members = c.get('nodes', [])
            if invalid_touched and (set(members) & invalid_touched):
                invalid_op_clusters += 1
                continue
            if set(members) & newly_archived:
                continue
            if shield_active and not (set(members) & valid_touched):
                rejected_call_clusters += 1
                continue
            if len(members) >= 2:
                processed_proposals.append(c)
                fingerprint_members.update(members)

        # Fingerprint on POST-encode updated_at: the encoder may have revised
        # members this run (CONTRADICTION revises the corrector); decode-time
        # timestamps would mismatch on the next cycle's re-derivation and the
        # fingerprint would never suppress. One fresh read covers all members.
        fresh_updated = {}
        member_list = sorted(fingerprint_members)
        for k in range(0, len(member_list), 900):
            chunk = member_list[k:k + 900]
            ph = ','.join('?' * len(chunk))
            for row in self.brain.conn.execute(
                    "SELECT id, updated_at FROM nodes WHERE id IN (%s)" % ph,
                    chunk):
                fresh_updated[row[0]] = row[1] or ''

        recorded = record_rejections(
            self.brain,
            [{
                'type': 'consolidation_cluster',
                'members': sorted(c.get('nodes', [])),
                'member_updated_at': {
                    nid: fresh_updated.get(nid, '')
                    for nid in c.get('nodes', [])
                },
            } for c in processed_proposals],
            integration_unit='s2:consolidation') if processed_proposals else 0
        if recorded:
            print('[s2-consolidation] Recorded %d cluster fingerprints' % recorded,
                  flush=True)
        if invalid_op_clusters:
            self.brain._log_warning(
                's2_consolidation_invalid_op_retry',
                '%d cluster(s) hit invalid brain_batch ops (merge thwarted) — '
                'retrying next cycle, NOT suppressed' % invalid_op_clusters)
            print('[s2-consolidation] %d cluster(s) hit invalid ops — retry, NOT suppressed'
                  % invalid_op_clusters, flush=True)
        if rejected_call_clusters:
            self.brain._log_warning(
                's2_consolidation_rejected_call_retry',
                '%d cluster(s) left un-acted after a rejected brain_batch '
                'call — retrying next cycle, NOT suppressed'
                % rejected_call_clusters)
            print('[s2-consolidation] %d cluster(s) un-acted after rejected '
                  'brain_batch call — retry, NOT suppressed'
                  % rejected_call_clusters, flush=True)

        # The give-up bound counts consecutive runs that actually shielded
        # clusters; a rejected call that pinned nothing doesn't advance it.
        if shield_active and rejected_call_clusters:
            self.brain.set_config(self.REJECTED_STREAK_KEY, str(streak + 1))
        elif streak:
            self.brain.set_config(self.REJECTED_STREAK_KEY, '0')

        # Advance the cutoff only when EVERY cluster was resolved. The
        # encode-failure path returns above without stamping (forcing retry);
        # invalid-op and rejected-call clusters need the same treatment —
        # they were neither edge-handled nor SKIP-stamped (thwarted), and
        # advancing the baseline would hide them from the incremental decoder
        # forever (their member timestamps never changed), silently losing
        # them — the exact failure this guard prevents, just relocated from
        # fingerprint to cutoff. Leaving the baseline forces a re-scan next
        # cycle; clusters that WERE resolved are already suppressed
        # (edges/fingerprints), so only the thwarted ones return.
        if not invalid_op_clusters and not rejected_call_clusters:
            self._record_scan_baseline(decode_result)

        return {
            'actions': encode_result.get('write_actions', 0),
            'clusters': len(clusters),
            'skipped_recorded': recorded,
            'invalid_op_clusters': invalid_op_clusters,
            'rejected_call_clusters': rejected_call_clusters,
            'stats': stats,
            'details': {
                'rounds': encode_result.get('rounds', 0),
                'actions': encode_result.get('actions', 0),
                'write_actions': encode_result.get('write_actions', 0),
            },
        }

    def _record_scan_baseline(self, decode_result):
        """Advance the last-run cutoff — ONLY after a run fully completes.

        Stamps the timestamp + threshold the decoder captured at scan start.
        Called on the completed paths (nothing to encode, or encoder
        succeeded), never on a skip or a mid-run encoder failure — so failed
        work is retried next cycle instead of being silently skipped past.
        """
        stamp = decode_result.get('_stamp')
        if stamp:
            self.brain.set_config(self.LAST_RUN_TS_KEY, str(stamp['ts']))
            self.brain.set_config(self.LAST_THRESHOLD_KEY, stamp['threshold'])
