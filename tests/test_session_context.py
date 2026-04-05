"""Tests for SessionContext — per-request session identity.

SessionContext flows with every brain call. The brain doesn't own sessions.
Every hook, MCP call, and encoding run receives a SessionContext.

Run: python3 -m pytest tests/test_session_context.py -v
"""
import pytest
from tests.isolated_brain import IsolatedBrain


# ═══════════════════════════════════════════════════════
# SessionContext object
# ═══════════════════════════════════════════════════════

class TestSessionContextCreation:
    """Verify SessionContext can be created and carries identity."""

    def test_create_with_id(self):
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='abc123')
        assert ctx.session_id == 'abc123'

    def test_create_from_hook_args(self):
        """Create from Claude Code hook JSON input."""
        from servers.session_context import SessionContext
        args = {'session_id': 'hook-session-1', 'cwd': '/home/user', 'hook_event_name': 'Stop'}
        ctx = SessionContext.from_hook_args(args)
        assert ctx.session_id == 'hook-session-1'

    def test_missing_session_id_uses_fallback(self):
        """If hook args don't have session_id, generates a fallback."""
        from servers.session_context import SessionContext
        ctx = SessionContext.from_hook_args({})
        assert ctx.session_id  # not empty
        assert len(ctx.session_id) > 0

    def test_session_id_short(self):
        """Short form for chain IDs — first 8 chars."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='abcdef1234567890')
        assert ctx.session_short == 'abcdef12'

    def test_stop_counter(self):
        """SessionContext carries stop counter."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='abc', stop_counter=42)
        assert ctx.stop_counter == 42

    def test_stop_counter_default(self):
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='abc')
        assert ctx.stop_counter == 0

    def test_increment_stop(self):
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='abc', stop_counter=5)
        ctx.increment_stop()
        assert ctx.stop_counter == 6


# ═══════════════════════════════════════════════════════
# SessionContext persistence
# ═══════════════════════════════════════════════════════

class TestSessionContextPersistence:
    """Verify SessionContext can be saved to and loaded from DB."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            yield

    def test_save_and_load(self):
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='persist-test', stop_counter=10)
        ctx.save(self.brain.logs_conn)

        loaded = SessionContext.load(self.brain.logs_conn, 'persist-test')
        assert loaded is not None
        assert loaded.session_id == 'persist-test'
        assert loaded.stop_counter == 10

    def test_load_nonexistent_returns_none(self):
        from servers.session_context import SessionContext
        loaded = SessionContext.load(self.brain.logs_conn, 'nonexistent')
        assert loaded is None

    def test_save_updates_existing(self):
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='update-test', stop_counter=5)
        ctx.save(self.brain.logs_conn)

        ctx.stop_counter = 15
        ctx.save(self.brain.logs_conn)

        loaded = SessionContext.load(self.brain.logs_conn, 'update-test')
        assert loaded.stop_counter == 15

    def test_multiple_sessions_isolated(self):
        """Two sessions don't interfere with each other."""
        from servers.session_context import SessionContext
        ctx1 = SessionContext(session_id='sess-1', stop_counter=10)
        ctx2 = SessionContext(session_id='sess-2', stop_counter=20)
        ctx1.save(self.brain.logs_conn)
        ctx2.save(self.brain.logs_conn)

        loaded1 = SessionContext.load(self.brain.logs_conn, 'sess-1')
        loaded2 = SessionContext.load(self.brain.logs_conn, 'sess-2')
        assert loaded1.stop_counter == 10
        assert loaded2.stop_counter == 20


# ═══════════════════════════════════════════════════════
# SessionContext in trace writes
# ═══════════════════════════════════════════════════════

class TestSessionContextInTraces:
    """Verify traces use SessionContext for chain IDs and session tagging."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            env.brain.logs_conn.execute('DELETE FROM trace_events')
            env.brain.logs_conn.commit()
            yield

    def test_chain_id_from_context(self):
        """Chain IDs use session_short and stop from context."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='abcdef1234567890', stop_counter=42)

        s0_chain = ctx.s0_chain()
        assert s0_chain == 's0-abcdef12-42'

        s1r_chain = ctx.s1r_chain()
        assert s1r_chain == 's1r-abcdef12-42'

        s1e_chain = ctx.s1e_chain()
        assert s1e_chain == 's1e-abcdef12-42'

    def test_trace_write_with_context(self):
        """Trace events use session_id from context."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='trace-ctx-test', stop_counter=5)

        self.dal.append(
            chain_id=ctx.s0_chain(), scale='s0', event_type='K',
            ref_type='user_message', summary='test',
            session_id=ctx.session_id)

        events = self.dal.get_chain(ctx.s0_chain())
        assert len(events) == 1
        chain = self.dal.get_chains(session_id='trace-ctx-test', scale='s0')
        assert len(chain) == 1

    def test_consistent_across_stops(self):
        """Same session, different stops — different chains, same session_id."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='consist-test', stop_counter=1)

        self.dal.append(chain_id=ctx.s0_chain(), scale='s0', event_type='K',
                        ref_type='user_message', summary='stop 1',
                        session_id=ctx.session_id)
        ctx.increment_stop()
        self.dal.append(chain_id=ctx.s0_chain(), scale='s0', event_type='K',
                        ref_type='user_message', summary='stop 2',
                        session_id=ctx.session_id)

        chains = self.dal.get_chains(session_id='consist-test', scale='s0')
        assert len(chains) == 2
        assert chains[0]['chain_id'] != chains[1]['chain_id']


# ═══════════════════════════════════════════════════════
# SessionContext replaces brain.session_id
# ═══════════════════════════════════════════════════════

class TestSessionContextReplacesSingleton:
    """Verify SessionContext works where brain.session_id used to."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            yield

    def test_daemon_restart_preserves_session(self):
        """If daemon restarts, loading the saved context preserves session_id."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='stable-session', stop_counter=50)
        ctx.save(self.brain.logs_conn)

        # Simulate daemon restart — new Brain instance, same DB
        from servers.brain import Brain
        brain2 = Brain(self.brain.db_path)

        loaded = SessionContext.load(brain2.logs_conn, 'stable-session')
        assert loaded is not None
        assert loaded.session_id == 'stable-session'
        assert loaded.stop_counter == 50
        brain2.close()

    def test_parallel_sessions_no_conflict(self):
        """Two sessions writing traces simultaneously don't interfere."""
        from servers.session_context import SessionContext
        ctx_a = SessionContext(session_id='parallel-a', stop_counter=1)
        ctx_b = SessionContext(session_id='parallel-b', stop_counter=1)

        dal = self.brain._trace_dal
        self.brain.logs_conn.execute('DELETE FROM trace_events')
        self.brain.logs_conn.commit()

        dal.append(chain_id=ctx_a.s0_chain(), scale='s0', event_type='K',
                   ref_type='user_message', summary='from A',
                   session_id=ctx_a.session_id)
        dal.append(chain_id=ctx_b.s0_chain(), scale='s0', event_type='K',
                   ref_type='user_message', summary='from B',
                   session_id=ctx_b.session_id)

        chains_a = dal.get_chains(session_id='parallel-a', scale='s0')
        chains_b = dal.get_chains(session_id='parallel-b', scale='s0')
        assert len(chains_a) == 1
        assert len(chains_b) == 1
        assert chains_a[0]['events'][0]['summary'] == 'from A'
        assert chains_b[0]['events'][0]['summary'] == 'from B'
