"""Tests for NodeDAL and SignalQueueDAL."""

import time
from datetime import datetime, timezone, timedelta
from tests.brain_test_base import BrainTestBase
from servers.dal import NodeDAL
from servers.dal_signal_queue import SignalQueueDAL


# ── NodeDAL Tests ──────────────────────────────────────────────────────


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


# ── SignalQueueDAL Tests ───────────────────────────────────────────────


class TestSignalQueueEnqueue(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.sq = SignalQueueDAL(self.brain.logs_conn)

    def test_creates_signal(self):
        self.sq.enqueue(id='sig-1', producer='test', signal_type='reminder',
                        priority=0.8, content='Do the thing')
        state = self.sq.get_queue_state()
        self.assertEqual(len(state), 1)
        self.assertEqual(state[0]['id'], 'sig-1')
        self.assertEqual(state[0]['content'], 'Do the thing')

    def test_deduplicates_by_id(self):
        self.sq.enqueue(id='sig-dup', producer='test', signal_type='reminder',
                        priority=0.5, content='Version 1')
        self.sq.enqueue(id='sig-dup', producer='test', signal_type='reminder',
                        priority=0.9, content='Version 2')
        state = self.sq.get_queue_state()
        self.assertEqual(len(state), 1)
        self.assertEqual(state[0]['priority'], 0.9)
        self.assertEqual(state[0]['content'], 'Version 2')


class TestSignalQueuePull(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.sq = SignalQueueDAL(self.brain.logs_conn)

    def test_returns_by_priority_desc(self):
        self.sq.enqueue(id='low', producer='test', signal_type='info',
                        priority=0.2, content='Low')
        self.sq.enqueue(id='high', producer='test', signal_type='info',
                        priority=0.9, content='High')
        self.sq.enqueue(id='mid', producer='test', signal_type='info',
                        priority=0.5, content='Mid')
        results = self.sq.pull(budget_chars=10000, limit=10)
        self.assertEqual([r['id'] for r in results], ['high', 'mid', 'low'])

    def test_respects_cooldown(self):
        self.sq.enqueue(id='cool', producer='test', signal_type='info',
                        priority=0.8, content='Cooled', cooldown_seconds=3600)
        # First pull succeeds
        results = self.sq.pull(budget_chars=10000)
        self.assertEqual(len(results), 1)
        # Second pull within cooldown returns nothing
        results = self.sq.pull(budget_chars=10000)
        self.assertEqual(len(results), 0)


class TestSignalQueueDismiss(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.sq = SignalQueueDAL(self.brain.logs_conn)

    def test_dismiss_by_signal_id(self):
        self.sq.enqueue(id='to-dismiss', producer='test', signal_type='info',
                        priority=0.5, content='Dismiss me')
        result = self.sq.dismiss('to-dismiss')
        self.assertTrue(result)
        state = self.sq.get_queue_state()
        self.assertEqual(len(state), 0)

    def test_dismiss_nonexistent_returns_false(self):
        result = self.sq.dismiss('no-such-signal')
        self.assertFalse(result)

    def test_dismiss_by_producer(self):
        self.sq.enqueue(id='p1', producer='noisy', signal_type='info',
                        priority=0.5, content='A')
        self.sq.enqueue(id='p2', producer='noisy', signal_type='info',
                        priority=0.6, content='B')
        self.sq.enqueue(id='p3', producer='quiet', signal_type='info',
                        priority=0.7, content='C')
        count = self.sq.dismiss_by_producer('noisy')
        self.assertEqual(count, 2)
        state = self.sq.get_queue_state()
        self.assertEqual(len(state), 1)
        self.assertEqual(state[0]['producer'], 'quiet')


class TestSignalQueueExpireStale(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.sq = SignalQueueDAL(self.brain.logs_conn)

    def test_expires_over_surfaced(self):
        self.sq.enqueue(id='over', producer='test', signal_type='info',
                        priority=0.5, content='Over surfaced', max_surfaces=1)
        # Pull once to hit max_surfaces
        self.sq.pull(budget_chars=10000)
        expired = self.sq.expire_stale()
        self.assertEqual(expired, 1)
        state = self.sq.get_queue_state()
        self.assertEqual(len(state), 0)

    def test_expires_ttl(self):
        self.sq.enqueue(id='old', producer='test', signal_type='info',
                        priority=0.5, content='Old signal', ttl_seconds=1)
        # Backdate the created_at to force expiry
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        self.brain.logs_conn.execute(
            'UPDATE signal_queue SET created_at = ? WHERE id = ?', (past, 'old'))
        self.brain.logs_conn.commit()
        expired = self.sq.expire_stale()
        self.assertEqual(expired, 1)
        state = self.sq.get_queue_state()
        self.assertEqual(len(state), 0)


class TestSignalQueueGetQueueState(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.sq = SignalQueueDAL(self.brain.logs_conn)

    def test_returns_all_pending(self):
        self.sq.enqueue(id='a', producer='p1', signal_type='info',
                        priority=0.3, content='A')
        self.sq.enqueue(id='b', producer='p2', signal_type='reminder',
                        priority=0.8, content='B')
        self.sq.dismiss('a')
        state = self.sq.get_queue_state()
        self.assertEqual(len(state), 1)
        self.assertEqual(state[0]['id'], 'b')

    def test_empty_queue(self):
        state = self.sq.get_queue_state()
        self.assertEqual(state, [])
