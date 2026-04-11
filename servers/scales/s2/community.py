"""S2 Community Detection — orchestrator.

Wires CommunityDecoder (algorithmic) → CommunityEncoder (agentic Sonnet).
This file is the public API — callers import CommunityDetection from here.

Decoder: servers/scales/s2/community_decoder.py
Encoder: servers/scales/s2/community_encoder.py
Contract: servers/scales/s2/community_contract.py
Prompt: servers/scales/s2/community_enrichment_prompt.py
"""

from .community_decoder import CommunityDecoder
from .community_encoder import CommunityEncoder
from .community_contract import COMMUNITY_DETECTION


class CommunityDetection(CommunityDecoder):
    """Full S2CD+S2CE pipeline — decode proposals then encode via Sonnet.

    Inherits from CommunityDecoder so all decoder methods are available.
    Callers that only need the decoder can use CommunityDecoder directly.
    """

    def run(self):
        """Run full decode → encode pipeline.

        Returns:
            dict with: actions, proposals, communities, details, skipped, error
        """
        # Decode
        decode_result = super().run()

        if decode_result.get('skipped'):
            return {'actions': 0, 'skipped': decode_result['skipped']}

        proposals = decode_result['proposals']
        community_state = decode_result['community_state']

        if not proposals:
            return {'actions': 0, 'proposals': 0,
                    'communities': len(community_state)}

        # Encode
        encoder = CommunityEncoder(
            self.brain, self.dispatch, self.config)
        encode_result = encoder.run(proposals, community_state)

        if not encode_result:
            return {'actions': 0, 'proposals': len(proposals),
                    'error': 'enrichment failed'}

        return {
            'actions': encode_result.get('write_actions', 0),
            'proposals': len(proposals),
            'communities': len(community_state),
            'details': {
                'rounds': encode_result.get('rounds', 0),
                'actions': encode_result.get('actions', 0),
                'write_actions': encode_result.get('write_actions', 0),
            },
        }
