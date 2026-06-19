"""Community membership restorer — GraphDAL.reconcile_community_membership.

Catches the silent-omission bug: the community encoder (Haiku) creates a
community node + its `community_members` metadata but omits the edge field
entirely (or used the retired `connections=` param, dropped by remember()'s
guard). The node then declares N members while holding ZERO edges — an
inconsistency nothing else catches, because the declared member list is the
only diffable intent. The restorer back-fills the gap, scoped to the
zero-edge case so it never resurrects an intentionally drifted member.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.brain_test_base import BrainTestBase


class TestCommunityMembershipReconcile(BrainTestBase):
    needs_embedder = False

    def _member_edges(self, cid):
        """Active community_member targets of a community."""
        return {r[0] for r in self.brain.conn.execute(
            "SELECT e.target_id FROM edges e "
            "JOIN edge_relations er ON er.edge_id = e.edge_id "
            "WHERE e.source_id = ? AND er.relation = 'community_member' "
            "AND er.archived = 0", (cid,)).fetchall()}

    def _member(self, title):
        return self.brain.remember(type='finding', title=title, content='c',
                                   encoding_source='test')['id']

    def _community(self, title, member_ids):
        """Community node declaring `member_ids` in metadata, but no edges."""
        cid = self.brain.remember(
            type='community', title=title, content='c',
            encoding_source='s2:community_detection')['id']
        members_str = ', '.join('%s: %s member' % (m, m) for m in member_ids)
        self.brain._meta_kv.set(cid, 'community_members', members_str)
        return cid

    def _reconcile(self):
        with self.brain.write_lock:
            return self.brain._graph.reconcile_community_membership()

    def test_backfills_orphaned_community(self):
        m1, m2, m3 = self._member('one'), self._member('two'), self._member('three')
        cid = self._community('Orphaned', [m1, m2, m3])
        self.assertEqual(self._member_edges(cid), set())  # the bug state

        recon = self._reconcile()

        self.assertEqual(self._member_edges(cid), {m1, m2, m3})
        self.assertEqual(recon['communities_healed'], 1)
        self.assertEqual(recon['edges_backfilled'], 3)

    def test_skips_archived_declared_members(self):
        m_live = self._member('live')
        m_arch = self._member('archived')
        self.brain.conn.execute(
            "UPDATE nodes SET archived = 1 WHERE id = ?", (m_arch,))
        self.brain.conn.commit()
        cid = self._community('HasArchived', [m_live, m_arch])

        recon = self._reconcile()

        # Archived declared member is drift, not omission — never re-added.
        self.assertEqual(self._member_edges(cid), {m_live})
        self.assertEqual(recon['edges_backfilled'], 1)

    def test_skips_partial_community_no_drift_resurrection(self):
        m_kept = self._member('kept')
        m_drifted = self._member('drifted')
        cid = self._community('Partial', [m_kept, m_drifted])
        # One edge already present; m_drifted was intentionally disconnected.
        self.brain.connect(cid, m_kept, relation='community_member', weight=0.6)

        recon = self._reconcile()

        # A community with ANY edge is left alone — re-adding m_drifted would
        # resurrect an intentional removal.
        self.assertEqual(self._member_edges(cid), {m_kept})
        self.assertEqual(recon['communities_healed'], 0)

    def test_idempotent(self):
        m1, m2 = self._member('a'), self._member('b')
        self._community('C', [m1, m2])

        first = self._reconcile()
        self.assertEqual(first['edges_backfilled'], 2)

        second = self._reconcile()
        self.assertEqual(second['edges_backfilled'], 0)
        self.assertEqual(second['communities_healed'], 0)


if __name__ == '__main__':
    unittest.main()
