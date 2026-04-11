"""S2 Consolidation — orchestrator.

Phase 1: decoder only. Finds convergent clusters and writes traces.
Phase 2 (future): wire in ConsolidationEncoder for Sonnet-driven merges.

Callers import Consolidation from here.
"""

from .consolidation_decoder import ConsolidationDecoder


class Consolidation(ConsolidationDecoder):
    """Full consolidation pipeline. Currently decoder-only.

    Phase 2 will add encoder after decoder.run(),
    same pattern as CommunityDetection.
    """

    def run(self):
        """Run consolidation. Phase 1: decode only."""
        return super().run()
