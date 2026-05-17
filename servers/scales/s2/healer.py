"""S2 Healer — orchestrator for node healing.

Two responsibilities, both run on each healer pass:
  1. Algorithmic invariant restoration (no LLM): archive any active edge
     that points at an archived node. Pre-April-2026 archive_node bugs and
     mid-migration races leave dangling edges; this scrubs them every cycle.
  2. LLM-based gap filling: decoder scans for nodes missing
     question/situation/reasoning → encoder generates via Haiku.

Follows the same pattern as CommunityDetection and Consolidation.
"""

from .healer_decoder import HealerDecoder
from .healer_encoder import HealerEncoder
from servers.dal import GraphDAL


class Healer(HealerDecoder):
    """Orchestrator: invariant restorer → decoder → encoder."""

    def run(self):
        # Invariant restorer (cheap, no LLM): archive edges to archived
        # nodes. Runs every cycle so historical leaks self-heal over time.
        archive_error = None
        try:
            edges_archived = GraphDAL(self.brain.conn).archive_dangling_edges(
                archived_by='s2:healer')
        except Exception as e:
            edges_archived = 0
            # Capture the error for the result dict so the caller can
            # distinguish "0 to archive" from "archive blew up". Also log
            # to brain errors so it surfaces at next boot via the signal
            # queue. Both paths run — visibility is layered.
            archive_error = '%s: %s' % (type(e).__name__, e)
            try:
                self.brain._log_error('healer_archive_dangling', e,
                                      'archive_dangling_edges failed')
            except Exception as log_err:
                # _log_error itself failing is rare; surface to stderr so
                # daemon.log captures it rather than cascade-silencing.
                import sys
                print('[healer] archive_dangling_edges failed + '
                      '_log_error failed: archive=%r log=%r' % (e, log_err),
                      file=sys.stderr, flush=True)

        def _result(base):
            """Build healer result with archive visibility baked in."""
            base['edges_archived'] = edges_archived
            if archive_error is not None:
                base['archive_error'] = archive_error
            return base

        # Decode: find gaps and build proposals
        decode_result = super().run()

        if decode_result.get('skipped'):
            return _result({'actions': edges_archived,
                            'skipped': decode_result['skipped']})

        proposals = decode_result.get('proposals', [])
        if not proposals:
            return _result({'actions': edges_archived,
                            'proposals': 0,
                            'stats': decode_result.get('stats', {})})

        # Encode: generate missing fields via Haiku
        encoder = HealerEncoder(self.brain, self.dispatch, self.config)
        encode_result = encoder.run(proposals)

        if not encode_result:
            return _result({'actions': edges_archived,
                            'error': 'healing failed'})

        return _result({
            'actions': encode_result.get('fields_written', 0) + edges_archived,
            'proposals': len(proposals),
            'nodes_healed': encode_result.get('nodes_healed', 0),
            'fields_written': encode_result.get('fields_written', 0),
            'skipped': encode_result.get('skipped', 0),
            'stats': decode_result.get('stats', {}),
            'errors': encode_result.get('errors', []),
        })
