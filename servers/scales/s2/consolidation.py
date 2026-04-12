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

        # Encode
        encoder = ConsolidationEncoder(
            self.brain, self.dispatch, self.config)
        encode_result = encoder.run(clusters)

        if not encode_result:
            return {'actions': 0, 'clusters': len(clusters),
                    'stats': stats, 'error': 'encoding failed'}

        return {
            'actions': encode_result.get('write_actions', 0),
            'clusters': len(clusters),
            'stats': stats,
            'details': {
                'rounds': encode_result.get('rounds', 0),
                'actions': encode_result.get('actions', 0),
                'write_actions': encode_result.get('write_actions', 0),
            },
        }
