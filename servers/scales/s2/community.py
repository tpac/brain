"""S2 Community Detection — orchestrator.

Wires CommunityDecoder (algorithmic) → rejection filter → CommunityEncoder.
This file is the public API — callers import CommunityDetection from here.

Decoder: servers/scales/s2/community_decoder.py
Encoder: servers/scales/s2/community_encoder.py
Rejection suppression: servers/scales/s2/rejection_table.py
Contract: servers/scales/s2/community_contract.py
Prompt: s2_community_enrichment interaction (v12+)
"""

from .community_decoder import CommunityDecoder
from .community_encoder import CommunityEncoder
from .community_contract import COMMUNITY_DETECTION
from .rejection_table import filter_rejected


class CommunityDetection(CommunityDecoder):
    """Full S2CD+S2CE pipeline — decode → filter rejections → encode.

    Inherits from CommunityDecoder so all decoder methods are available.
    Callers that only need the decoder can use CommunityDecoder directly.
    """

    def run(self):
        """Run full decode → filter → encode pipeline.

        Rejection filter runs between decoder and encoder. Proposals whose
        fingerprints match previous rejections are suppressed before the
        encoder sees them — prevents the re-proposal loop where the encoder
        wastes tokens judging identical proposals every idle cycle.

        Returns:
            dict with: actions, proposals, communities, details, skipped, error
        """
        # s2_rejections table is created by ensure_schema() at Brain startup
        # (see servers/schema.py). No per-run creation needed.

        # Decode
        decode_result = super().run()

        if decode_result.get('skipped'):
            return {'actions': 0, 'skipped': decode_result['skipped']}

        raw_proposals = decode_result['proposals']
        community_state = decode_result['community_state']

        if not raw_proposals:
            return {'actions': 0, 'proposals': 0,
                    'communities': len(community_state)}

        # Filter through rejection table — suppress re-proposals
        proposals, suppressed_count = filter_rejected(self.brain, raw_proposals)

        if suppressed_count:
            self.trace('K', 'rejection_suppression',
                       '%d proposals suppressed by prior rejections (of %d raw)' % (
                           suppressed_count, len(raw_proposals)),
                       metadata={'suppressed': suppressed_count,
                                 'raw_total': len(raw_proposals),
                                 'surviving': len(proposals)})

        if not proposals:
            return {'actions': 0, 'proposals': 0,
                    'raw_proposals': len(raw_proposals),
                    'suppressed': suppressed_count,
                    'communities': len(community_state),
                    'skipped': 'all proposals suppressed'}

        # Encode
        encoder = CommunityEncoder(
            self.brain, self.dispatch, self.config)
        encode_result = encoder.run(proposals, community_state)

        if not encode_result:
            return {'actions': 0, 'proposals': len(proposals),
                    'raw_proposals': len(raw_proposals),
                    'suppressed': suppressed_count,
                    'error': 'enrichment failed'}

        return {
            'actions': encode_result.get('write_actions', 0),
            'proposals': len(proposals),
            'raw_proposals': len(raw_proposals),
            'suppressed': suppressed_count,
            'communities': len(community_state),
            'details': {
                'rounds': encode_result.get('rounds', 0),
                'actions': encode_result.get('actions', 0),
                'write_actions': encode_result.get('write_actions', 0),
            },
        }
