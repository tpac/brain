"""Integration tests for source_refs through the full brain.remember()
and brain.revise() write paths (v29 / Phase B Steps 2-4).

These tests exercise the MCP-schema → dispatch validation → brain.remember
kwarg → GraphDAL persistence path end-to-end via an isolated Brain
instance. Pure-DAL tests live in test_episodic_refs_dal.py; this file
locks the integration contract.
"""

import os
import shutil
import tempfile
import unittest

from servers.brain import Brain
from servers.dal import GraphDAL
from servers.daemon_dispatch import _validate_source_refs


class RememberSourceRefsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix='brain_source_refs_test_')
        os.environ['BRAIN_DB_DIR'] = cls.tmpdir + '/'
        os.environ['BRAIN_DEV_MODE'] = '1'
        cls.brain = Brain(db_path=os.path.join(cls.tmpdir, 'brain.db'))

    @classmethod
    def tearDownClass(cls):
        try:
            cls.brain.conn.close()
            cls.brain.logs_conn.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        # Seed a couple of trace_event rows so we have real ids to anchor against
        self.brain.logs_conn.execute(
            "INSERT OR REPLACE INTO trace_events "
            "(id, chain_id, scale, event_type, ref_type, summary, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ('a3f5e2b1', 'chain-test', 's0', 'K', 'user_message',
             'tom probe', 'sess-test', '2026-05-25T00:00:00Z'))
        self.brain.logs_conn.execute(
            "INSERT OR REPLACE INTO trace_events "
            "(id, chain_id, scale, event_type, ref_type, summary, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ('b8c9d0e1', 'chain-test', 's0', 'delta', 'assistant_message',
             'anchor reply', 'sess-test', '2026-05-25T00:00:01Z'))
        self.brain.logs_conn.commit()
        self.created_node_ids = []

    def tearDown(self):
        gd = GraphDAL(self.brain.conn)
        for nid in self.created_node_ids:
            try:
                gd.replace_source_refs(nid, [])
                self.brain.conn.execute('DELETE FROM nodes WHERE id = ?', (nid,))
                self.brain.conn.commit()
            except Exception:
                pass

    def _create_node(self, **kwargs):
        result = self.brain.remember(
            type=kwargs.pop('type', 'principle'),
            title=kwargs.pop('title', 'test'),
            content=kwargs.pop('content', 'test content'),
            encoding_source='anchor',
            **kwargs,
        )
        nid = result['id']
        self.created_node_ids.append(nid)
        return nid, result

    # ── remember() ────────────────────────────────────────────

    def test_remember_persists_source_refs(self):
        nid, _ = self._create_node(source_refs=['a3f5e2b1', 'b8c9d0e1'])
        gd = GraphDAL(self.brain.conn)
        self.assertEqual(gd.get_source_refs(nid), ['a3f5e2b1', 'b8c9d0e1'])

    def test_remember_without_source_refs_leaves_table_empty(self):
        nid, _ = self._create_node()
        gd = GraphDAL(self.brain.conn)
        self.assertEqual(gd.get_source_refs(nid), [])

    def test_remember_empty_list_is_noop(self):
        nid, _ = self._create_node(source_refs=[])
        gd = GraphDAL(self.brain.conn)
        self.assertEqual(gd.get_source_refs(nid), [])

    def test_engram_cohort_via_shared_ref(self):
        """Two nodes anchored to the same trace_id form a cohort retrievable
        via get_nodes_referencing — the substrate for the future co_anchored
        edge (Step 7)."""
        nid1, _ = self._create_node(title='node 1', source_refs=['a3f5e2b1'])
        nid2, _ = self._create_node(title='node 2', source_refs=['a3f5e2b1'])
        gd = GraphDAL(self.brain.conn)
        cohort = sorted(gd.get_nodes_referencing('a3f5e2b1'))
        self.assertEqual(cohort, sorted([nid1, nid2]))

    # ── revise() — field-level REPLACE per decision 995ffeb1 ──

    def test_revise_with_source_refs_replaces_not_appends(self):
        """When source_refs IS in the revise payload → REPLACE entire list."""
        nid, _ = self._create_node(source_refs=['a3f5e2b1'])
        gd = GraphDAL(self.brain.conn)
        self.assertEqual(gd.get_source_refs(nid), ['a3f5e2b1'])

        result = self.brain.revise(nid, reason='swap anchor', source_refs=['b8c9d0e1'])
        self.assertIn('source_refs', result.get('fields_updated', []))
        self.assertEqual(gd.get_source_refs(nid), ['b8c9d0e1'])

    def test_revise_without_source_refs_preserves_existing(self):
        """When source_refs is ABSENT from the revise payload → preserve."""
        nid, _ = self._create_node(source_refs=['a3f5e2b1', 'b8c9d0e1'])
        gd = GraphDAL(self.brain.conn)
        self.brain.revise(nid, reason='content only',
                          content='new content (refs untouched)')
        self.assertEqual(gd.get_source_refs(nid), ['a3f5e2b1', 'b8c9d0e1'])

    def test_revise_with_empty_list_explicitly_clears(self):
        """Empty list is the explicit clear signal (per unified contract:
        present-field replaces, even with empty value)."""
        nid, _ = self._create_node(source_refs=['a3f5e2b1', 'b8c9d0e1'])
        gd = GraphDAL(self.brain.conn)
        result = self.brain.revise(nid, reason='clear refs', source_refs=[])
        self.assertIn('source_refs', result.get('fields_updated', []))
        self.assertEqual(gd.get_source_refs(nid), [])
        self.assertEqual(result.get('source_refs_replaced'), 0)

    # ── dispatch validator ────────────────────────────────────

    def test_validator_accepts_well_formed_list(self):
        ok, err = _validate_source_refs(['a3f5e2b1', 'b8c9d0e1'], 'test')
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_validator_accepts_none(self):
        ok, err = _validate_source_refs(None, 'test')
        self.assertTrue(ok)

    def test_validator_accepts_empty_list(self):
        """Empty list is a legitimate shape (explicit clear on revise;
        pure-synthesis no-anchor pattern on remember)."""
        ok, err = _validate_source_refs([], 'test')
        self.assertTrue(ok)

    def test_validator_rejects_non_list(self):
        ok, err = _validate_source_refs('a3f5e2b1', 'test')
        self.assertFalse(ok)
        self.assertIn('must be a list', err)

    def test_validator_rejects_int_in_list(self):
        """v29: ints rejected loudly (reviewer F2)."""
        ok, err = _validate_source_refs([42], 'test')
        self.assertFalse(ok)
        self.assertIn('hex string', err)

    def test_validator_rejects_non_string_element(self):
        ok, err = _validate_source_refs(['valid', None], 'test')
        self.assertFalse(ok)

    def test_validator_rejects_empty_string_element(self):
        ok, err = _validate_source_refs(['valid', '  '], 'test')
        self.assertFalse(ok)
        self.assertIn('empty', err)


if __name__ == '__main__':
    unittest.main()
