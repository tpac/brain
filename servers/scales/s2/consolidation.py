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
from .consolidation_contract import CONSOLIDATION
from .rejection_table import filter_rejected, record_rejections


class Consolidation(ConsolidationDecoder):
    """Full S2 consolidation pipeline — decode clusters then encode via Sonnet.

    Inherits from ConsolidationDecoder so all decoder methods are available.
    Callers that only need the decoder can use ConsolidationDecoder directly.
    """

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
            return {'actions': 0, 'clusters': 0, 'stats': stats}

        # Fingerprint-based rejection filter — SKIP decisions from prior
        # runs populate s2_rejections; those clusters don't resurface here.
        # `member_updated_at` folds into the fingerprint so any revise()
        # on a member invalidates the rejection automatically (cluster IDs
        # unchanged + edited content ⇒ the encoder's judgment may differ
        # from its prior SKIP and must re-evaluate).
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

        # Snapshot suppression edges before encoder runs so we can detect
        # which input clusters the encoder actually acted on. Any cluster
        # with no new intra-cluster similar_to/consolidated_into edge after
        # the run is recorded as a rejection (SKIP path) — no edge written
        # on the graph, fingerprint in s2_rejections prevents re-proposal.
        def _snapshot_suppression_pairs():
            # Active suppression edges only — archived rows don't count
            # as current suppression. TODO(v25-dal): could migrate to a
            # GraphDAL.get_pairs_with_relations helper if a second caller
            # appears.
            pairs = set()
            for row in self.brain.conn.execute(
                "SELECT e.source_id, e.target_id FROM edges e "
                "JOIN edge_relations er ON er.edge_id = e.edge_id "
                "WHERE er.relation IN ('similar_to','consolidated_into') "
                "AND er.archived = 0"):
                pairs.add((min(row[0], row[1]), max(row[0], row[1])))
            return pairs

        pre_edges = _snapshot_suppression_pairs()

        # Encode
        encoder = ConsolidationEncoder(
            self.brain, self.dispatch, self.config)
        encode_result = encoder.run(clusters)

        if not encode_result:
            return {'actions': 0, 'clusters': len(clusters),
                    'stats': stats, 'error': 'encoding failed'}

        # Detect clusters that got no suppression edge this run → SKIPPED →
        # record a rejection fingerprint so the decoder doesn't resurface them.
        post_edges = _snapshot_suppression_pairs()
        new_edges = post_edges - pre_edges
        skipped_proposals = []
        for c in clusters:
            members = c.get('nodes', [])
            handled = False
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    key = (min(members[i], members[j]),
                           max(members[i], members[j]))
                    if key in new_edges:
                        handled = True
                        break
                if handled:
                    break
            if not handled and len(members) >= 2:
                skipped_proposals.append({
                    'type': 'consolidation_cluster',
                    'members': sorted(members),
                    'member_updated_at': {
                        nid: c.get('node_details', {}).get(nid, {}).get('updated_at', '')
                        for nid in members
                    },
                })
        recorded = record_rejections(
            self.brain, skipped_proposals,
            integration_unit='s2:consolidation') if skipped_proposals else 0
        if recorded:
            print('[s2-consolidation] Recorded %d SKIP rejections' % recorded,
                  flush=True)

        return {
            'actions': encode_result.get('write_actions', 0),
            'clusters': len(clusters),
            'skipped_recorded': recorded,
            'stats': stats,
            'details': {
                'rounds': encode_result.get('rounds', 0),
                'actions': encode_result.get('actions', 0),
                'write_actions': encode_result.get('write_actions', 0),
            },
        }
