"""S2 Healer — orchestrator for node healing.

Thin wrapper: decoder scans for gaps → encoder generates via Haiku → done.
Follows the same pattern as CommunityDetection and Consolidation.
"""

from .healer_decoder import HealerDecoder
from .healer_encoder import HealerEncoder


class Healer(HealerDecoder):
    """Orchestrator: decoder → encoder pipeline for node healing."""

    def run(self):
        # Decode: find gaps and build proposals
        decode_result = super().run()

        if decode_result.get('skipped'):
            return {'actions': 0, 'skipped': decode_result['skipped']}

        proposals = decode_result.get('proposals', [])
        if not proposals:
            return {'actions': 0, 'proposals': 0,
                    'stats': decode_result.get('stats', {})}

        # Encode: generate missing fields via Haiku
        encoder = HealerEncoder(self.brain, self.dispatch, self.config)
        encode_result = encoder.run(proposals)

        if not encode_result:
            return {'actions': 0, 'error': 'healing failed'}

        return {
            'actions': encode_result.get('fields_written', 0),
            'proposals': len(proposals),
            'nodes_healed': encode_result.get('nodes_healed', 0),
            'fields_written': encode_result.get('fields_written', 0),
            'skipped': encode_result.get('skipped', 0),
            'stats': decode_result.get('stats', {}),
            'errors': encode_result.get('errors', []),
        }
