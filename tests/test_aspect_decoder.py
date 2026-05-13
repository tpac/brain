"""AspectDecoder run() behavior — empty-batch must NOT write an O trace.

The decoder's O trace ('aspect_scan') marks "I observed unclassified work."
Writing it on empty batches caused a runaway S2 cascade in production
(rolled back 2026-05-08, see coordinator.py). The fix is an early-out
BEFORE the trace write. This test locks the contract so the regression
can't sneak back in.
"""

from tests.brain_test_base import BrainTestBase
from servers.scales.s2.aspect_decoder import AspectDecoder


class TestAspectDecoderEmptyBatch(BrainTestBase):
    """Empty-batch decoder is a true no-op — no proposals, no trace."""

    needs_embedder = False

    def _count_aspect_scan_traces(self):
        rows = self.brain._trace_dal.conn.execute(
            "SELECT COUNT(*) FROM trace_events "
            "WHERE scale='s2' AND ref_type='aspect_scan'"
        ).fetchone()
        return rows[0] if rows else 0

    def test_empty_batch_returns_skipped(self):
        # Fresh BrainTestBase has only the 14 seeded aspect-nodes; their
        # types ('aspect') are classified in aspects_v1.json. So the
        # decoder finds nothing unclassified.
        decoder = AspectDecoder(self.brain)
        result = decoder.run()

        self.assertEqual(result.get('proposals'), [])
        self.assertEqual(result.get('skipped'), 'nothing unclassified')

    def test_empty_batch_writes_no_trace(self):
        # The original bug: O trace fired even when batch was empty,
        # downstream units treated it as fresh work.
        before = self._count_aspect_scan_traces()
        AspectDecoder(self.brain).run()
        after = self._count_aspect_scan_traces()

        self.assertEqual(after, before,
                         'aspect_scan trace must NOT be written on empty batch '
                         '— this is the cascade-prevention contract')

    def test_unclassified_string_DOES_write_trace(self):
        # Symmetric proof: when there IS work, the O trace fires as expected.
        # Add a node with an unclassified type, above min_count_threshold.
        threshold = AspectDecoder(self.brain).config['min_count_threshold']
        for i in range(threshold + 1):
            self.brain.remember(
                type='zzz_novel_unclassified_type',
                title='novel-type node %d' % i,
                content='content %d' % i,
                source='test',
            )

        before = self._count_aspect_scan_traces()
        result = AspectDecoder(self.brain).run()
        after = self._count_aspect_scan_traces()

        self.assertGreater(len(result.get('proposals', [])), 0,
                           'unclassified type should produce a proposal')
        self.assertEqual(after, before + 1,
                         'aspect_scan trace fires when batch is non-empty')
