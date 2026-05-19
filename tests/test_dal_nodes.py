"""Tests for NodeDAL."""

from tests.brain_test_base import BrainTestBase
from servers.dal import NodeDAL


class TestNodeDALResolveId(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.dal = NodeDAL(self.brain.conn)
        self.node = self.brain.remember(
            type='rule', title='Test node for resolve',
            content='Content for resolve_id tests')
        self.node_id = self.node['id']

    def test_prefix_match(self):
        result = self.dal.resolve_id(self.node_id[:8])
        self.assertEqual(result, self.node_id)

    def test_exact_match(self):
        result = self.dal.resolve_id(self.node_id)
        self.assertEqual(result, self.node_id)

    def test_no_match_returns_none(self):
        result = self.dal.resolve_id('zzz_nonexistent')
        self.assertIsNone(result)

    def test_empty_string_returns_none(self):
        result = self.dal.resolve_id('')
        self.assertIsNone(result)


class TestNodeDALGetTitle(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.dal = NodeDAL(self.brain.conn)
        self.node = self.brain.remember(
            type='rule', title='Unique title for get_title',
            content='Content for get_title tests')
        self.node_id = self.node['id']

    def test_prefix_match(self):
        result = self.dal.get_title(self.node_id[:8])
        self.assertEqual(result, 'Unique title for get_title')

    def test_exact_match(self):
        result = self.dal.get_title(self.node_id)
        self.assertEqual(result, 'Unique title for get_title')

    def test_no_match_returns_none(self):
        result = self.dal.get_title('zzz_nonexistent')
        self.assertIsNone(result)
