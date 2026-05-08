"""S2 Aspect Integration — orchestrator: decoder → encoder pipeline.

Thin wrapper following the same pattern as Healer / CommunityDetection /
Consolidation. Decoder finds unclassified strings, encoder routes them
into the 14 aspects, applier merges into aspects_v1.json. All file I/O,
no brain mutations.
"""

from .aspect_decoder import AspectDecoder
from .aspect_encoder import AspectEncoder


class AspectIntegration(AspectDecoder):
    """Orchestrator: decoder → encoder pipeline for aspect classification."""

    def run(self):
        # Decode: find unclassified, build proposals
        decode_result = super().run()

        if decode_result.get('skipped'):
            return {'actions': 0, 'skipped': decode_result['skipped'],
                    'stats': decode_result.get('stats', {})}

        proposals = decode_result.get('proposals', [])
        if not proposals:
            return {'actions': 0, 'proposals': 0,
                    'stats': decode_result.get('stats', {})}

        # Encode: classify via Sonnet, merge into aspects_v1.json
        encoder = AspectEncoder(self.brain, self.dispatch, self.config)
        encode_result = encoder.run(proposals)

        if not encode_result:
            return {'actions': 0, 'error': 'aspect classification failed'}

        return {
            'actions': encode_result.get('classified', 0),
            'proposals': len(proposals),
            'classified': encode_result.get('classified', 0),
            'rejected': encode_result.get('rejected', 0),
            'remaining': decode_result.get('remaining', 0),
            'stats': decode_result.get('stats', {}),
            'per_aspect': encode_result.get('per_aspect', {}),
            'errors': encode_result.get('errors', []),
        }
