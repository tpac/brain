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

    def test_load_corrupt_blob_raises_typed_signal(self):
        # A corrupt session_state blob is SIGNALED (SessionContextCorrupt), not
        # silently None — the brain-holding caller catches it and logs via the
        # canonical _log_error. Absent rows still return None (test above).
        from servers.session_context import SessionContext, SessionContextCorrupt
        from servers.dal import SessionStateDAL
        SessionStateDAL(self.brain.logs_conn).set('corrupt-sess', '_session_context', 'not json{')
        with pytest.raises(SessionContextCorrupt):
            SessionContext.load(self.brain.logs_conn, 'corrupt-sess')
        # the brain caller catches the signal and degrades gracefully (no crash)
        assert self.brain.session_env_for('corrupt-sess') == {'cwd': '', 'branch': ''}

    def test_save_updates_existing(self):
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='update-test', stop_counter=5)
        ctx.save(self.brain.logs_conn)

        ctx.stop_counter = 15
        ctx.save(self.brain.logs_conn)

        loaded = SessionContext.load(self.brain.logs_conn, 'update-test')
        assert loaded.stop_counter == 15

    def test_save_and_load_cwd_branch(self):
        # cwd/branch are session identity (fed from the boot hook) — they must
        # round-trip through the JSON blob like the other fields.
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='env-test')
        ctx.cwd = '/work/tree/x'
        ctx.branch = 'claude/x'
        ctx.save(self.brain.logs_conn)
        loaded = SessionContext.load(self.brain.logs_conn, 'env-test')
        assert loaded.cwd == '/work/tree/x'
        assert loaded.branch == 'claude/x'

    def test_reset_session_activity_stamps_cwd(self):
        # boot feeds cwd → a NEW session's reset stamps cwd + derived branch.
        from servers.session_context import SessionContext
        is_resume = self.brain.reset_session_activity(session_id='env-sess', cwd='/work/tree/y')
        assert is_resume is False                    # never booted → new session
        after = SessionContext.load(self.brain.logs_conn, 'env-sess')
        assert after.cwd == '/work/tree/y'          # stamped from the boot feed
        assert after.branch                          # derived (real branch or 'unknown')

    def test_reset_session_activity_resume_preserves_state(self):
        # REGRESSION: a re-boot of an already-booted session is a RESUME — it must
        # CONTINUE accumulated state, not zero it. Pre-fix, every boot built a
        # fresh ctx, so under parallel sessions the global resume-guard misfired
        # and reset live sessions: stop_counter→0 (duplicate chain IDs) and segment
        # state lost. Resume detection now lives in the session object
        # (ctx.boot_time). (The Scribe cadence itself is trace-derived now, so it's
        # immune regardless — see test_turns_since_last_encode_trace_pull.)
        from servers.session_context import SessionContext
        assert self.brain.reset_session_activity(session_id='resume-sess', cwd='/w/a') is False
        ctx = self.brain._session_contexts['resume-sess']
        ctx.stop_counter = 7
        ctx.message_count = 4
        ctx.segment_id = 2
        # Re-boot the SAME session (resume) — different cwd to prove identity still
        # refreshes while accumulated state is preserved.
        assert self.brain.reset_session_activity(session_id='resume-sess', cwd='/w/b') is True
        after = SessionContext.load(self.brain.logs_conn, 'resume-sess')
        assert after.stop_counter == 7               # chain-id sequence preserved (no dup chains)
        assert after.message_count == 4
        assert after.segment_id == 2
        assert after.cwd == '/w/b'                    # identity refreshed on resume

    def test_resume_after_daemon_restart_loads_from_db(self):
        # A daemon restart empties the in-memory cache but the persisted row
        # survives. The next boot must still detect a resume (from the DB row) and
        # preserve state — the suspend-restart path that was resetting live
        # sessions in production.
        from servers.session_context import SessionContext
        self.brain.reset_session_activity(session_id='restart-sess', cwd='/w/a')
        ctx = self.brain._session_contexts['restart-sess']
        ctx.stop_counter = 9
        ctx.save(self.brain.logs_conn)               # persist before the "restart"
        self.brain._session_contexts.clear()         # simulate daemon restart
        assert self.brain.reset_session_activity(session_id='restart-sess', cwd='/w/a') is True
        after = SessionContext.load(self.brain.logs_conn, 'restart-sess')
        assert after.stop_counter == 9               # survived the restart + re-boot

    def test_turns_since_last_encode_trace_pull(self):
        # The S1 Scribe cadence is derived LIVE from traces — no stored counter.
        # Pins the definition: a turn == one s0 user_message; <task-notification>
        # ignitions don't count; only turns AFTER the last s1 encoding_prompt count.
        from servers.scales.s0.conversation import turns_since_last_encode
        sid = 'cadence-sess'
        c = self.brain.logs_conn
        _n = [0]
        def ins(scale, etype, ref_type, ts, summary='hi'):
            _n[0] += 1
            c.execute(
                "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, "
                "ref_id, summary, metadata, session_id, interaction_id, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ('t%07d' % _n[0], '%s-x-%d' % (scale, _n[0]), scale, etype, ref_type,
                 '', summary, None, sid, None, ts))
        # 3 turns, no encode yet → counts all 3
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:01+00:00')
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:02+00:00')
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:03+00:00')
        c.commit()
        assert turns_since_last_encode(self.brain, sid) == 3
        # encode runs, then 2 real turns + 1 task-notification ignition after it
        ins('s1', 'O', 'encoding_prompt', '2026-06-13T10:00:04+00:00')
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:05+00:00')
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:06+00:00')
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:07+00:00', summary='<task-notification> wake')
        c.commit()
        assert turns_since_last_encode(self.brain, sid) == 2   # ignition excluded; only real turns since encode


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


class TestScribeStarvationThreshold:
    """The Scribe-starvation alarm is a pure threshold decision (level trigger,
    rate-limited). The gate (daemon_hooks) logs a loud brain error when it trips —
    the monitor that would have caught the 20h encode-drought on hour one."""

    def test_below_threshold_not_starved(self):
        from servers.scales.s1.encode_contract import (
            scribe_is_starved, SCRIBE_STARVATION_TURNS, ENCODE_EVERY)
        assert SCRIBE_STARVATION_TURNS == 4 * ENCODE_EVERY
        assert not scribe_is_starved(0)
        assert not scribe_is_starved(ENCODE_EVERY)               # normal: gate fires, no alarm
        assert not scribe_is_starved(SCRIBE_STARVATION_TURNS - 1)

    def test_at_threshold_then_rate_limited(self):
        from servers.scales.s1.encode_contract import (
            scribe_is_starved, SCRIBE_STARVATION_TURNS, ENCODE_EVERY)
        assert scribe_is_starved(SCRIBE_STARVATION_TURNS)                  # first alert at threshold
        assert not scribe_is_starved(SCRIBE_STARVATION_TURNS + 1)          # rate-limited between cadences
        assert scribe_is_starved(SCRIBE_STARVATION_TURNS + ENCODE_EVERY)   # next alert one cadence later


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
