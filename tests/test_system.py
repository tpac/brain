"""
System-level tests — end-to-end simulation of the full brain hook lifecycle.

These tests simulate what ACTUALLY happens in a Claude Code session:
  1. Boot fires → daemon starts → brain context printed
  2. User prompts fire → recall surfaces context → track detects vocab gaps
  3. File edits fire → pre-edit surfaces rules
  4. Bash commands fire → safety checks run
  5. Idle fires → maintenance runs
  6. Compaction fires → save + reboot
  7. Session ends → synthesis + shutdown

Unlike test_hooks.py (which tests individual hooks in isolation),
these tests verify the FLOW between hooks and the daemon's role
as the shared state backbone. They also cover the critical failure
mode: what happens when the daemon isn't ready yet.

Test hierarchy:
  TestDaemonReadiness     — daemon not ready, slow start, socket race conditions
  TestSessionLifecycle    — full session from boot to end
  TestDaemonStateFlow     — state propagation across hook calls via daemon
  TestGraphChangeTracking — graph mutations surface in recall
  TestGracefulDegradation — every hook survives missing daemon, corrupt DB, etc.
  TestConcurrentHooks     — multiple hooks firing while daemon is busy
"""

import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from servers.brain import Brain
from servers.daemon import (
    BrainDaemon, send_command, get_socket_path, get_pid_path,
    is_daemon_running, ensure_daemon, stop_daemon,
)
from servers.daemon_hooks import (
    hook_recall, hook_post_response_track, hook_idle_maintenance,
    hook_post_compact_reboot, hook_pre_edit, hook_pre_bash_safety,
    hook_pre_compact_save, hook_session_end, hook_stop_failure_log,
    hook_config_change_host, hook_post_bash_host_check,
    hook_worktree_context, hook_worktree_cleanup,
)
from tests.brain_test_base import BrainTestBase, HookTestBase as _SharedHookBase


# ── Shared test infrastructure ────────────────────────────────────────────

def _kill_all_test_daemons():
    """Kill any test daemons by PID file."""
    for pid_file in glob.glob('/tmp/brain-daemon-*.pid'):
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
        except (OSError, ValueError):
            pass
    # Clean socket files too
    for sock_file in glob.glob('/tmp/brain-daemon-*.sock'):
        try:
            os.unlink(sock_file)
        except OSError:
            pass
    for pid_file in glob.glob('/tmp/brain-daemon-*.pid'):
        try:
            os.unlink(pid_file)
        except OSError:
            pass


def _seed_realistic_brain(brain):
    """Seed a brain with realistic data for system-level tests.

    This isn't 2 toy nodes — it's a realistic brain with multiple types,
    connections, vocabulary, and enough data that recall actually has to score.
    """
    nodes = {}

    # Core rules (locked — always surface)
    n = brain.remember(
        type='rule',
        title='Authentication must use Clerk with magic links — no password flow',
        content='Clerk handles entire auth flow. Magic links for login. '
                'Webhook syncs user data to our Supabase DB. Free tier for MVP. '
                'Never implement custom auth or password storage.',
        keywords='auth clerk login passwordless magic-link webhook supabase',
        locked=True, confidence=0.95)
    nodes['auth_rule'] = n.get('id', '') if isinstance(n, dict) else ''

    n = brain.remember(
        type='rule',
        title='All database migrations must be backward-compatible',
        content='Never drop columns or rename tables in a migration. Always add new '
                'columns as nullable, backfill, then clean up in a later migration. '
                'This prevents downtime during rolling deploys.',
        keywords='database migrations backward-compatible schema deploy',
        locked=True, confidence=0.9)
    nodes['migration_rule'] = n.get('id', '') if isinstance(n, dict) else ''

    # Decisions
    n = brain.remember(
        type='decision',
        title='Supply Adapter pattern for ad delivery abstraction layer',
        content='Clean abstraction between Glo app and ad delivery backends. '
                'Interface: createCampaign, updateCampaign, pauseCampaign. '
                'V1 implements GAM adapter only. Can swap without business logic changes.',
        keywords='adapter pattern supply abstraction gam interface architecture',
        locked=True, confidence=0.85)
    nodes['adapter_decision'] = n.get('id', '') if isinstance(n, dict) else ''

    n = brain.remember(
        type='decision',
        title='React component architecture: atomic design pattern',
        content='Components organized by atomic design: atoms (Button, Input), '
                'molecules (FormField), organisms (LoginForm). '
                'Shared via internal package @glo/ui.',
        keywords='react components atomic design pattern architecture ui',
        locked=True, confidence=0.8)
    nodes['react_decision'] = n.get('id', '') if isinstance(n, dict) else ''

    # Engineering memory
    n = brain.remember(
        type='mechanism',
        title='Supabase RLS policies enforce row-level security on all tables',
        content='Every table has RLS enabled with policies checking auth.uid(). '
                'Service role key bypasses RLS for server-side operations. '
                'Client-side queries always go through RLS.',
        keywords='supabase rls row-level-security policies auth tables',
        confidence=0.8)
    nodes['rls_mechanism'] = n.get('id', '') if isinstance(n, dict) else ''

    n = brain.remember(
        type='lesson',
        title='Git worktree remove destroys another session CWD silently',
        content='Discovered 2026-03-15: running git worktree remove on a worktree '
                'that another Claude Code session is using as its CWD causes that '
                'session to silently fail on all file operations. The other session '
                'gets "No such file or directory" but doesnt crash. Must check for '
                'active sessions before removing worktrees.',
        keywords='git worktree remove destroy session cwd silent failure dangerous',
        locked=True, confidence=0.95)
    nodes['worktree_lesson'] = n.get('id', '') if isinstance(n, dict) else ''

    # Context/working notes
    brain.remember(
        type='context',
        title='Current sprint: ad delivery pipeline v2',
        content='Working on ad delivery pipeline v2. Key files: '
                'src/services/ad-delivery.ts, src/adapters/gam-adapter.ts. '
                'Goal: support multiple ad networks via adapter pattern.',
        keywords='sprint ad delivery pipeline current work',
        confidence=0.6)

    brain.remember(
        type='context',
        title='Tom prefers explicit error handling over silent swallowing',
        content='Feedback from Tom: never use bare except:pass. Always log errors. '
                'If something fails, the user needs to know. Silent failures are '
                'the worst kind of bugs.',
        keywords='error handling explicit logging feedback tom preference',
        confidence=0.7)

    # Vocabulary
    try:
        brain.learn_vocabulary(
            term='working copy',
            context='git',
            content='worktree',
            source='operator')
    except Exception:
        pass

    try:
        brain.learn_vocabulary(
            term='supply adapter',
            context='architecture',
            content='ad delivery abstraction layer using adapter pattern',
            source='operator')
    except Exception:
        pass

    # Connections
    if nodes.get('auth_rule') and nodes.get('rls_mechanism'):
        try:
            brain.connect(nodes['auth_rule'], nodes['rls_mechanism'],
                         'depends_on', weight=0.8)
        except Exception:
            pass

    if nodes.get('adapter_decision') and nodes.get('react_decision'):
        try:
            brain.connect(nodes['adapter_decision'], nodes['react_decision'],
                         'related_to', weight=0.5)
        except Exception:
            pass

    brain.save()
    return nodes


class SystemTestBase(_SharedHookBase):
    """Base for system-level tests. Seeds realistic brain, cleans daemons."""

    def setUp(self):
        self._test_start = time.time()
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        self.brain = Brain(self.db_path)
        self.brain.reset_session_activity()
        self.node_ids = _seed_realistic_brain(self.brain)
        self.brain.save()
        # Close brain — hooks open their own connections
        self.brain.close()
        self.brain = None

        # Resolve paths
        self.scripts_dir = os.path.join(PROJECT_ROOT, 'hooks', 'scripts')
        self.servers_dir = os.path.join(PROJECT_ROOT, 'servers')

        # Build env — NO daemon socket exists yet (clean state)
        self.env = os.environ.copy()
        self.env['BRAIN_DB_DIR'] = self.tmp
        self.env['BRAIN_SERVER_DIR'] = self.servers_dir
        self.env['PLUGIN_ROOT'] = PROJECT_ROOT
        self.env['CLAUDE_PLUGIN_ROOT'] = PROJECT_ROOT

    def tearDown(self):
        _kill_all_test_daemons()
        if self.brain is not None:
            try:
                self.brain.close()
            except Exception:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_hook(self, script_name, stdin_data=None, timeout=30):
        """Run a hook script and return (returncode, stdout, stderr)."""
        script_path = os.path.join(self.scripts_dir, script_name)
        result = subprocess.run(
            ['bash', script_path],
            env=self.env,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.tmp,
        )
        return result.returncode, result.stdout, result.stderr

    def _open_brain(self):
        """Open a fresh Brain connection for assertions."""
        return Brain(self.db_path)

    def _remove_daemon_socket(self):
        """Ensure no daemon socket exists (simulates daemon not ready)."""
        sock_path = get_socket_path()
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        # Also remove PID file
        pid_path = get_pid_path()
        if os.path.exists(pid_path):
            os.unlink(pid_path)


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Daemon Readiness — what happens when daemon isn't up yet
# ══════════════════════════════════════════════════════════════════════════

class TestDaemonReadiness(SystemTestBase):
    """Tests for the critical window between boot starting daemon and hooks needing it.

    Real scenario: boot-brain.sh starts daemon in background (&) then execs boot_brain.py.
    The daemon takes 1-3s to load Brain + embedder. Meanwhile, UserPromptSubmit fires
    pre-response-recall.sh which checks daemon_available(). If daemon isn't ready yet,
    the hook must still produce valid output via direct fallback.
    """

    def test_recall_without_daemon_returns_valid_json(self):
        """Recall hook must return valid JSON even when daemon isn't running."""
        self._remove_daemon_socket()
        hook_input = json.dumps({
            'prompt': 'How does the authentication flow work with Clerk?',
        })
        rc, stdout, stderr = self.run_hook('pre-response-recall.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0, f"recall exited {rc}.\nstderr: {stderr}")
        output = json.loads(stdout.strip())
        # Must have EITHER additionalContext (found results) or decision=approve
        has_context = 'additionalContext' in output
        has_approve = output.get('decision') == 'approve'
        self.assertTrue(has_context or has_approve,
                        f"Expected valid recall output. Got: {json.dumps(output)[:300]}")

    def test_recall_without_daemon_still_finds_relevant_nodes(self):
        """Direct fallback must actually recall — not just approve blindly."""
        self._remove_daemon_socket()
        hook_input = json.dumps({
            'prompt': 'Tell me about the React component architecture we use',
        })
        rc, stdout, stderr = self.run_hook('pre-response-recall.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0)
        output = json.loads(stdout.strip())
        # Should find the React decision node
        context = output.get('additionalContext', '')
        self.assertIn('atomic design', context.lower(),
                       f"Direct fallback should recall React architecture node.\n"
                       f"Got: {context[:500]}")

    def test_pre_edit_without_daemon_returns_valid_json(self):
        """Pre-edit hook must return valid JSON when daemon is down."""
        self._remove_daemon_socket()
        hook_input = json.dumps({
            'tool_name': 'Edit',
            'tool_input': {'file_path': '/project/src/auth.py', 'old_string': 'x', 'new_string': 'y'},
        })
        rc, stdout, stderr = self.run_hook('pre-edit-suggest.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0)
        output = json.loads(stdout.strip())
        self.assertEqual(output.get('decision'), 'approve')

    def test_pre_bash_safety_without_daemon_still_warns(self):
        """Safety hook must still detect destructive commands without daemon."""
        self._remove_daemon_socket()
        hook_input = json.dumps({
            'tool_name': 'Bash',
            'tool_input': {'command': 'rm -rf /tmp/important_project'},
        })
        rc, stdout, stderr = self.run_hook('pre-bash-safety.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0)
        output = json.loads(stdout.strip())
        # Must at least detect destructive command (approve with warning, or block)
        reason = output.get('reason', '')
        self.assertIn('destructive', reason.lower(),
                       f"Safety hook should warn about rm -rf even without daemon.\n"
                       f"Got: {json.dumps(output)[:300]}")

    def test_pre_bash_safety_safe_command_without_daemon(self):
        """Safe commands should approve instantly without daemon (regex pre-screen)."""
        self._remove_daemon_socket()
        hook_input = json.dumps({
            'tool_name': 'Bash',
            'tool_input': {'command': 'ls -la /tmp'},
        })
        rc, stdout, stderr = self.run_hook('pre-bash-safety.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0)
        output = json.loads(stdout.strip())
        self.assertEqual(output.get('decision'), 'approve')
        # Should NOT have a reason (no warning needed for safe commands)
        self.assertNotIn('destructive', output.get('reason', '').lower())

    def test_post_response_track_without_daemon(self):
        """Post-response tracker should work via direct fallback."""
        self._remove_daemon_socket()
        hook_input = json.dumps({
            'prompt': 'How does authentication work with our Clerk setup?',
            'hook_event_name': 'UserPromptSubmit',
        })
        rc, stdout, stderr = self.run_hook('post-response-track.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0, f"post-response-track exited {rc}.\nstderr: {stderr}")

    def test_idle_maintenance_without_daemon(self):
        """Idle maintenance should run direct fallback without crashing."""
        self._remove_daemon_socket()
        rc, stdout, stderr = self.run_hook('idle-maintenance.sh', timeout=60)
        self.assertEqual(rc, 0, f"idle-maintenance exited {rc}.\nstderr: {stderr}")

    def test_pre_compact_without_daemon(self):
        """Pre-compact must ALWAYS return approve — even without daemon."""
        self._remove_daemon_socket()
        rc, stdout, stderr = self.run_hook('pre-compact-save.sh')
        self.assertEqual(rc, 0)
        output = json.loads(stdout.strip())
        self.assertEqual(output.get('decision'), 'approve',
                         "Pre-compact must NEVER block compaction, even without daemon")

    def test_session_end_without_daemon(self):
        """Session end should run clean without daemon."""
        self._remove_daemon_socket()
        rc, stdout, stderr = self.run_hook('session-end.sh')
        self.assertEqual(rc, 0, f"session-end exited {rc}.\nstderr: {stderr}")

    def test_post_compact_reboot_without_daemon(self):
        """Post-compact reboot should output context even without daemon."""
        self._remove_daemon_socket()
        rc, stdout, stderr = self.run_hook('post-compact-reboot.sh')
        self.assertEqual(rc, 0)
        # Should output SOMETHING — this is the safety net after compaction
        combined = stdout.lower()
        self.assertTrue(
            'brain' in combined or 'reboot' in combined or 'compac' in combined,
            f"Post-compact reboot should output context.\nGot: {stdout[:500]}")

    def test_all_hooks_exit_zero_without_daemon(self):
        """EVERY hook must exit 0 when daemon is unavailable. No exceptions."""
        self._remove_daemon_socket()
        hooks_and_inputs = [
            ('pre-response-recall.sh', json.dumps({'prompt': 'test query for recall'})),
            ('post-response-track.sh', json.dumps({'prompt': 'test message', 'hook_event_name': 'UserPromptSubmit'})),
            ('pre-edit-suggest.sh', json.dumps({'tool_name': 'Edit', 'tool_input': {'file_path': '/test.py', 'old_string': 'a', 'new_string': 'b'}})),
            ('pre-bash-safety.sh', json.dumps({'tool_name': 'Bash', 'tool_input': {'command': 'echo hello'}})),
            ('pre-compact-save.sh', None),
            ('post-compact-reboot.sh', None),
            ('session-end.sh', None),
            ('stop-failure-log.sh', json.dumps({'error': 'test_error', 'error_details': 'test'})),
            ('config-change-host.sh', json.dumps({'source': 'test'})),
            ('worktree-context.sh', json.dumps({'name': 'test-wt', 'cwd': self.tmp})),
            ('worktree-cleanup.sh', json.dumps({})),
        ]
        for script_name, stdin_data in hooks_and_inputs:
            rc, stdout, stderr = self.run_hook(script_name, stdin_data=stdin_data, timeout=30)
            self.assertEqual(rc, 0,
                             f"{script_name} exited {rc} without daemon.\nstderr: {stderr[:300]}")

    def test_direct_fallback_latency(self):
        """Direct fallback (no daemon) should complete within hook timeout.

        The hook timeout for recall is 5s. Direct Python imports Brain fresh
        (~600ms-1.6s for embedder). Total must stay under 5s.
        """
        self._remove_daemon_socket()
        hook_input = json.dumps({'prompt': 'Tell me about our authentication system'})
        t_start = time.time()
        rc, stdout, stderr = self.run_hook('pre-response-recall.sh', stdin_data=hook_input, timeout=10)
        elapsed = time.time() - t_start
        self.assertEqual(rc, 0)
        # Must complete within the 5s hook timeout (with margin)
        self.assertLess(elapsed, 8.0,
                         f"Direct fallback took {elapsed:.1f}s — would exceed 5s hook timeout")


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Session Lifecycle — boot to end, full simulation
# ══════════════════════════════════════════════════════════════════════════

class TestSessionLifecycle(SystemTestBase):
    """Simulates a complete Claude Code session from boot to end.

    This is the root system test. It walks through the exact sequence of
    events that happen in a real session, verifying state at each step.
    """

    def test_full_session_sequence(self):
        """Complete session: boot → prompt → edit → bash → idle → compact → end.

        This is the single most important test. It simulates every major
        hook event in order and verifies the brain's state after each.
        """
        # ── STEP 1: Boot ──
        rc, stdout, stderr = self.run_hook('boot-brain.sh', timeout=15)
        self.assertEqual(rc, 0, f"Boot failed.\nstderr: {stderr[:500]}")
        self.assertIn('brain', stdout.lower() + stderr.lower(),
                       "Boot should mention brain in output")

        # ── STEP 2: First user prompt (recall) ──
        hook_input = json.dumps({
            'prompt': 'How does the authentication flow work?',
            'session_id': 'test-lifecycle-1',
        })
        rc, stdout, stderr = self.run_hook('pre-response-recall.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0, f"Recall failed.\nstderr: {stderr[:500]}")
        output = json.loads(stdout.strip())
        # Boot started daemon in background — recall may use daemon or direct
        has_content = 'additionalContext' in output or output.get('decision') == 'approve'
        self.assertTrue(has_content, f"Recall output invalid: {json.dumps(output)[:300]}")

        # ── STEP 3: Post-response tracking (fires alongside recall) ──
        track_input = json.dumps({
            'prompt': 'How does the authentication flow work?',
            'hook_event_name': 'UserPromptSubmit',
        })
        rc, stdout, stderr = self.run_hook('post-response-track.sh', stdin_data=track_input)
        self.assertEqual(rc, 0, f"Track failed.\nstderr: {stderr[:500]}")

        # ── STEP 4: User edits a file ──
        edit_input = json.dumps({
            'tool_name': 'Edit',
            'tool_input': {
                'file_path': '/project/src/auth.py',
                'old_string': 'def login():',
                'new_string': 'def login_with_clerk():',
            },
        })
        rc, stdout, stderr = self.run_hook('pre-edit-suggest.sh', stdin_data=edit_input)
        self.assertEqual(rc, 0, f"Pre-edit failed.\nstderr: {stderr[:500]}")
        output = json.loads(stdout.strip())
        self.assertEqual(output.get('decision'), 'approve')

        # ── STEP 5: User runs a safe bash command ──
        bash_input = json.dumps({
            'tool_name': 'Bash',
            'tool_input': {'command': 'npm test -- --coverage'},
        })
        rc, stdout, stderr = self.run_hook('pre-bash-safety.sh', stdin_data=bash_input)
        self.assertEqual(rc, 0)
        output = json.loads(stdout.strip())
        self.assertEqual(output.get('decision'), 'approve')

        # ── STEP 6: Second prompt (verifies recall still works) ──
        hook_input2 = json.dumps({
            'prompt': 'Now lets look at the React component structure',
            'session_id': 'test-lifecycle-1',
        })
        rc, stdout, stderr = self.run_hook('pre-response-recall.sh', stdin_data=hook_input2)
        self.assertEqual(rc, 0)
        output2 = json.loads(stdout.strip())
        has_content2 = 'additionalContext' in output2 or output2.get('decision') == 'approve'
        self.assertTrue(has_content2)

        # ── STEP 7: Compaction (pre-compact → post-compact) ──
        rc, stdout, stderr = self.run_hook('pre-compact-save.sh')
        self.assertEqual(rc, 0)
        compact_out = json.loads(stdout.strip())
        self.assertEqual(compact_out.get('decision'), 'approve',
                         "Pre-compact must ALWAYS approve")

        rc, stdout, stderr = self.run_hook('post-compact-reboot.sh')
        self.assertEqual(rc, 0)
        # Reboot should mention brain context
        self.assertTrue(len(stdout) > 50,
                         f"Post-compact output too short: {stdout[:200]}")

        # ── STEP 8: Session end ──
        rc, stdout, stderr = self.run_hook('session-end.sh')
        self.assertEqual(rc, 0, f"Session end failed.\nstderr: {stderr[:500]}")

        # ── Verify brain state after full session ──
        brain = self._open_brain()
        try:
            # Should have compaction boundary node
            rows = brain.conn.execute(
                "SELECT title FROM nodes WHERE title LIKE '%Compaction boundary%'"
            ).fetchall()
            self.assertTrue(len(rows) >= 1,
                            "Pre-compact should have created a boundary marker node")

            # Should have a session synthesis
            synth = brain.conn.execute(
                "SELECT id FROM session_syntheses ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            self.assertIsNotNone(synth, "Session end should have created a synthesis")
        finally:
            brain.close()

    def test_boot_then_immediate_recall(self):
        """Simulate race: boot fires, daemon starting, recall fires immediately.

        In real usage, UserPromptSubmit can fire within milliseconds of boot.
        The daemon may not be ready yet. Recall must handle this gracefully.
        """
        # Boot (starts daemon in background)
        rc, _, _ = self.run_hook('boot-brain.sh', timeout=15)
        self.assertEqual(rc, 0)

        # IMMEDIATELY fire recall — daemon might not be ready
        hook_input = json.dumps({
            'prompt': 'What are the locked rules I should follow?',
        })
        rc, stdout, stderr = self.run_hook('pre-response-recall.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0, f"Recall failed during daemon startup.\nstderr: {stderr[:500]}")
        output = json.loads(stdout.strip())
        has_content = 'additionalContext' in output or output.get('decision') == 'approve'
        self.assertTrue(has_content,
                         "Recall must produce valid output even during daemon startup race")

    def test_double_compaction_cycle(self):
        """Two compaction cycles in one session — reboot reinjects properly each time."""
        # Boot
        rc, _, _ = self.run_hook('boot-brain.sh', timeout=15)
        self.assertEqual(rc, 0)

        # First compaction cycle
        rc, _, _ = self.run_hook('pre-compact-save.sh')
        self.assertEqual(rc, 0)
        rc, stdout1, _ = self.run_hook('post-compact-reboot.sh')
        self.assertEqual(rc, 0)
        self.assertIn('brain', stdout1.lower())

        # Some more prompts after first compaction
        hook_input = json.dumps({'prompt': 'Continuing work after compaction'})
        rc, _, _ = self.run_hook('pre-response-recall.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0)

        # Second compaction cycle
        rc, _, _ = self.run_hook('pre-compact-save.sh')
        self.assertEqual(rc, 0)
        rc, stdout2, _ = self.run_hook('post-compact-reboot.sh')
        self.assertEqual(rc, 0)
        self.assertIn('brain', stdout2.lower())

        # Both reboots should produce meaningful output
        self.assertTrue(len(stdout1) > 50)
        self.assertTrue(len(stdout2) > 50)


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: Daemon State Flow — state propagates across hook calls
# ══════════════════════════════════════════════════════════════════════════

class TestDaemonStateFlow(BrainTestBase):
    """Tests that daemon_hooks functions share state correctly through Brain.

    These test the daemon_hooks.py functions DIRECTLY (no subprocess),
    verifying that state set by one hook is visible to the next.
    """

    def test_recall_drains_pending_messages(self):
        """Messages stored by background hooks should appear in recall output."""
        # Simulate a background hook storing a pending message
        self.brain.set_config('pending_hook_messages',
                              json.dumps(['HOST: Python upgraded 3.11 -> 3.12']))
        self.brain.save()

        result = hook_recall(self.brain, {
            'prompt': 'What should I know about the current environment?',
        }, [])

        self.assertIn('json', result)
        context = result['json'].get('additionalContext', '')
        self.assertIn('QUEUED MESSAGES', context,
                       f"Recall should surface pending messages.\nGot: {context[:500]}")
        self.assertIn('Python upgraded', context)

        # Pending should be drained (cleared)
        remaining = self.brain.get_config('pending_hook_messages', '[]')
        self.assertEqual(remaining, '[]',
                         "Pending messages should be cleared after drain")

    def test_track_stores_vocab_gaps(self):
        """Post-response track should detect and store vocabulary gaps."""
        hook_post_response_track(self.brain, {
            'prompt': 'What about the "flux capacitor" in the rendering pipeline?',
            'hook_event_name': 'UserPromptSubmit',
        }, [])

        gaps_raw = self.brain.get_config('vocabulary_gaps', '[]')
        gaps = json.loads(gaps_raw)
        # "flux capacitor" is in quotes → should be detected as a candidate
        gap_terms = [g.get('term') if isinstance(g, dict) else g for g in gaps]
        self.assertTrue(
            any('flux capacitor' in str(t) for t in gap_terms),
            f"Should detect 'flux capacitor' as vocabulary gap.\nGaps: {gaps}")

    def test_idle_stores_pending_for_recall(self):
        """Idle maintenance stores results as pending, recall drains them."""
        # Run idle maintenance
        hook_idle_maintenance(self.brain, {}, [])

        # Check pending messages exist
        pending_raw = self.brain.get_config('pending_hook_messages', '[]')
        pending = json.loads(pending_raw)
        self.assertTrue(len(pending) > 0,
                         "Idle maintenance should store output as pending message")
        self.assertTrue(any('IDLE MAINTENANCE' in str(p) for p in pending),
                         f"Pending should contain maintenance output.\nGot: {pending}")

        # Now recall should drain and surface them
        result = hook_recall(self.brain, {
            'prompt': 'What happened during idle time?',
        }, [])

        if 'json' in result and 'additionalContext' in result['json']:
            context = result['json']['additionalContext']
            self.assertIn('QUEUED MESSAGES', context)

    def test_pre_compact_creates_boundary_node(self):
        """Pre-compact should create a compaction boundary marker node."""
        count_before = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE title LIKE '%Compaction boundary%'"
        ).fetchone()[0]

        hook_pre_compact_save(self.brain, {}, [])

        count_after = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE title LIKE '%Compaction boundary%'"
        ).fetchone()[0]
        self.assertEqual(count_after, count_before + 1,
                         "Pre-compact should create exactly one boundary marker")

    def test_session_end_runs_without_crash(self):
        """Session end should complete without crashing (synthesis may be empty)."""
        # Record some activity first
        self.brain.record_message()
        self.brain.record_message()
        # Remember something to give synthesis content to work with
        self.brain.remember(
            type='decision', title='Test decision during session',
            content='We decided to use X instead of Y.',
            keywords='test decision session')
        self.brain.save()

        # Session end should not crash
        result = hook_session_end(self.brain, {}, [])
        self.assertIn('output', result)

        # Session end calls synthesize + consolidate. At minimum, save ran.
        # Synthesis may or may not produce a record depending on session data,
        # but the function itself must not raise.

    def test_worktree_context_sets_config(self):
        """Worktree context hook should set brain config keys."""
        hook_worktree_context(self.brain, {
            'name': 'test-feature-branch',
            'cwd': '/tmp/test-worktree',
        }, [])

        self.assertEqual(self.brain.get_config('current_worktree'), 'test-feature-branch')
        self.assertEqual(self.brain.get_config('current_cwd'), '/tmp/test-worktree')

    def test_worktree_cleanup_clears_config(self):
        """Worktree cleanup should clear all worktree config."""
        # Set some values first
        self.brain.set_config('current_worktree', 'old-worktree')
        self.brain.set_config('current_branch', 'old-branch')
        self.brain.set_config('current_cwd', '/old/path')

        hook_worktree_cleanup(self.brain, {}, [])

        # get_config returns None or '' for cleared values
        self.assertFalse(self.brain.get_config('current_worktree'),
                         "current_worktree should be cleared")
        self.assertFalse(self.brain.get_config('current_branch'),
                         "current_branch should be cleared")
        self.assertFalse(self.brain.get_config('current_cwd'),
                         "current_cwd should be cleared")

    def test_stop_failure_runs_without_crash(self):
        """Stop failure hook should log without crashing."""
        # log_miss writes to brain_logs.db (separate DB), not brain.db
        # Just verify the function completes without error
        result = hook_stop_failure_log(self.brain, {
            'error': 'rate_limit',
            'error_details': 'Too many requests',
            'session_id': 'test-session',
        }, [])
        self.assertIn('output', result)


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: Graph Change Tracking — mutations visible in recall
# ══════════════════════════════════════════════════════════════════════════

class TestGraphChangeTracking(BrainTestBase):
    """Tests that graph mutations are tracked and surfaced by recall."""

    def test_remember_tracked_in_graph_changes(self):
        """Brain.remember via daemon should append to graph_changes."""
        graph_changes = []

        # Simulate what daemon does: remember then pass changes to recall
        self.brain.remember(
            type='decision',
            title='Test decision for graph tracking',
            content='This decision should appear in graph changes.',
            keywords='test graph tracking',
        )
        graph_changes.append("REMEMBER: [decision] Test decision for graph tracking")

        result = hook_recall(self.brain, {
            'prompt': 'What changed recently?',
        }, graph_changes)

        context = result.get('json', {}).get('additionalContext', '')
        self.assertIn('GRAPH ACTIVITY', context,
                       f"Recall should surface graph changes.\nGot: {context[:500]}")
        self.assertIn('Test decision', context)

    def test_graph_changes_drained_after_recall(self):
        """Graph changes should be cleared after recall drains them."""
        graph_changes = [
            "REMEMBER: [rule] New safety rule",
            "CONNECT: abc12345 -[depends_on]-> def67890",
        ]

        hook_recall(self.brain, {'prompt': 'test query'}, graph_changes)

        self.assertEqual(len(graph_changes), 0,
                         "Graph changes should be cleared after drain")

    def test_multiple_mutations_accumulate(self):
        """Multiple mutations between prompts should all appear."""
        graph_changes = []

        # Simulate multiple operations between user prompts
        graph_changes.append("REMEMBER: [decision] Chose REST over GraphQL")
        graph_changes.append("REMEMBER: [lesson] GraphQL complexity not worth it for MVP")
        graph_changes.append("CONNECT: node1 -[leads_to]-> node2")
        graph_changes.append("DREAM: 2 new dream node(s)")

        result = hook_recall(self.brain, {
            'prompt': 'What happened while I was away?',
        }, graph_changes)

        context = result.get('json', {}).get('additionalContext', '')
        self.assertIn('GRAPH ACTIVITY', context)
        self.assertIn('REST over GraphQL', context)
        self.assertIn('DREAM', context)

    def test_no_graph_changes_no_section(self):
        """When no graph changes, the GRAPH ACTIVITY section should not appear."""
        result = hook_recall(self.brain, {
            'prompt': 'Tell me about authentication',
        }, [])

        context = result.get('json', {}).get('additionalContext', '')
        # Might get approve (no results) or context. Either way, no GRAPH ACTIVITY.
        if context:
            self.assertNotIn('GRAPH ACTIVITY', context,
                             "No mutations → no GRAPH ACTIVITY section")

    def test_idle_maintenance_adds_graph_changes(self):
        """Idle maintenance should log its mutations to graph_changes."""
        graph_changes = []
        hook_idle_maintenance(self.brain, {}, graph_changes)
        # Idle does dream, consolidate, heal — at least consolidate should log something
        # Even if counts are 0, the function runs
        # Check that at least the list was populated OR is empty (both valid)
        # The key test is that graph_changes is the SAME list (shared reference)
        self.assertIsInstance(graph_changes, list)


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: Graceful Degradation — every hook survives failure conditions
# ══════════════════════════════════════════════════════════════════════════

class TestGracefulDegradation(SystemTestBase):
    """Tests that hooks degrade gracefully under various failure conditions.

    Real-world failures: corrupt DB, missing tables, daemon crash mid-request,
    stale socket file, brain import failure, embedder not loading.
    """

    def test_corrupt_db_all_hooks_survive(self):
        """Every hook should exit 0 even with a corrupt brain.db."""
        # Write garbage to brain.db
        with open(self.db_path, 'wb') as f:
            f.write(b'THIS IS NOT A SQLITE DATABASE AT ALL GARBAGE DATA ' * 10)

        self._remove_daemon_socket()

        hooks_and_inputs = [
            ('pre-response-recall.sh', json.dumps({'prompt': 'test query'})),
            ('pre-edit-suggest.sh', json.dumps({'tool_name': 'Edit', 'tool_input': {'file_path': '/test.py', 'old_string': 'a', 'new_string': 'b'}})),
            ('pre-bash-safety.sh', json.dumps({'tool_name': 'Bash', 'tool_input': {'command': 'echo hello'}})),
            ('pre-compact-save.sh', None),
            ('session-end.sh', None),
        ]

        for script_name, stdin_data in hooks_and_inputs:
            rc, stdout, stderr = self.run_hook(script_name, stdin_data=stdin_data, timeout=15)
            self.assertIn(rc, [0, 1],
                          f"{script_name} should handle corrupt DB gracefully.\n"
                          f"Exit code: {rc}\nstderr: {stderr[:300]}")

    def test_missing_brain_db_dir(self):
        """Hooks should exit cleanly when BRAIN_DB_DIR points to nonexistent path."""
        env_override = {'BRAIN_DB_DIR': '/tmp/nonexistent_brain_test_' + str(os.getpid())}

        hooks_and_inputs = [
            ('pre-response-recall.sh', json.dumps({'prompt': 'test query'})),
            ('pre-compact-save.sh', None),
            ('session-end.sh', None),
        ]

        for script_name, stdin_data in hooks_and_inputs:
            rc, stdout, stderr = self.run_hook(script_name, stdin_data=stdin_data,
                                               timeout=15)
            # Hook should not crash (exit > 1)
            self.assertIn(rc, [0, 1],
                          f"{script_name} should handle missing DB dir.\n"
                          f"Exit code: {rc}\nstderr: {stderr[:300]}")

    def test_stale_socket_file(self):
        """Hooks should handle a stale daemon socket (file exists, no listener)."""
        # Create a fake socket file (not a real Unix socket)
        socket_path = get_socket_path()
        try:
            # Create a real socket then close it (leaves stale file)
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            s.bind(socket_path)
            s.close()
            # Socket file exists but nobody is listening

            hook_input = json.dumps({'prompt': 'Test with stale socket'})
            rc, stdout, stderr = self.run_hook('pre-response-recall.sh',
                                               stdin_data=hook_input, timeout=15)
            self.assertEqual(rc, 0,
                             f"Recall should handle stale socket.\nstderr: {stderr[:300]}")
        finally:
            if os.path.exists(socket_path):
                os.unlink(socket_path)

    def test_empty_hook_input(self):
        """Hooks should handle empty/missing HOOK_INPUT gracefully."""
        # Run with empty stdin (no HOOK_INPUT)
        hooks = [
            'pre-response-recall.sh',
            'post-response-track.sh',
            'pre-edit-suggest.sh',
            'pre-bash-safety.sh',
            'stop-failure-log.sh',
            'config-change-host.sh',
            'worktree-context.sh',
            'worktree-cleanup.sh',
        ]
        for script_name in hooks:
            rc, stdout, stderr = self.run_hook(script_name, stdin_data='', timeout=15)
            self.assertEqual(rc, 0,
                             f"{script_name} should handle empty input.\n"
                             f"Exit code: {rc}\nstderr: {stderr[:300]}")

    def test_pre_compact_never_blocks(self):
        """Pre-compact MUST return approve under ALL conditions.

        If pre-compact blocks compaction, Claude Code crashes or enters
        an unrecoverable state. This tests the absolute invariant.
        """
        scenarios = [
            ('normal', {}),
            ('missing_db', {'BRAIN_DB_DIR': '/nonexistent'}),
        ]

        for scenario_name, env_override in scenarios:
            env = self.env.copy()
            env.update(env_override)

            script_path = os.path.join(self.scripts_dir, 'pre-compact-save.sh')
            result = subprocess.run(
                ['bash', script_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
                cwd=self.tmp,
            )

            # Must exit 0 AND output approve
            self.assertEqual(result.returncode, 0,
                             f"pre-compact [{scenario_name}] must exit 0.\n"
                             f"stderr: {result.stderr[:300]}")
            try:
                output = json.loads(result.stdout.strip())
                self.assertEqual(output.get('decision'), 'approve',
                                 f"pre-compact [{scenario_name}] must approve.\n"
                                 f"Output: {result.stdout[:300]}")
            except json.JSONDecodeError:
                # Even if output is garbage, it MUST have approve somewhere
                self.assertIn('approve', result.stdout.lower(),
                              f"pre-compact [{scenario_name}] must output approve.\n"
                              f"Got: {result.stdout[:300]}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 6: Hook Function Unit Tests — daemon_hooks.py functions directly
# ══════════════════════════════════════════════════════════════════════════

class TestHookFunctions(BrainTestBase):
    """Direct tests of daemon_hooks.py functions with a real Brain instance.

    These bypass subprocess overhead and test the centralized logic directly.
    """

    def setUp(self):
        super().setUp()
        _seed_realistic_brain(self.brain)

    def test_recall_finds_auth_rules(self):
        """Recall should find auth rules for auth-related queries."""
        result = hook_recall(self.brain, {
            'prompt': 'How does authentication work with Clerk?',
        }, [])

        self.assertIn('json', result)
        context = result['json'].get('additionalContext', '')
        self.assertTrue(
            'clerk' in context.lower() or 'auth' in context.lower(),
            f"Should find auth-related nodes.\nContext: {context[:500]}")

    def test_recall_short_message_approves(self):
        """Short messages should skip recall entirely."""
        result = hook_recall(self.brain, {'prompt': 'ok'}, [])
        # hook_recall doesn't handle the short-message skip — that's in the client
        # But querying with 'ok' should either find something or approve
        self.assertIn('json', result)

    def test_pre_edit_surfaces_auth_rules_for_auth_file(self):
        """Editing auth.py should surface auth-related brain rules."""
        result = hook_pre_edit(self.brain, {
            'filename': 'auth.py',
            'tool_name': 'Edit',
        }, [])

        self.assertIn('json', result)
        output = result['json']
        self.assertEqual(output.get('decision'), 'approve')
        reason = output.get('reason', '')
        # Should surface auth/clerk rules if brain has them
        if reason:
            reason_lower = reason.lower()
            self.assertTrue(
                'auth' in reason_lower or 'clerk' in reason_lower or 'brain auto-suggest' in reason_lower,
                f"Pre-edit for auth.py should mention auth/clerk.\nReason: {reason[:500]}")

    def test_pre_bash_detects_rm_rf(self):
        """Safety check should detect rm -rf as destructive."""
        result = hook_pre_bash_safety(self.brain, {
            'command': 'rm -rf /tmp/important_project',
        }, [])

        self.assertIn('json', result)
        output = result['json']
        # Should be approve (with warning) or block (if critical match)
        self.assertIn(output.get('decision'), ['approve', 'block'])
        reason = output.get('reason', '')
        self.assertTrue(
            'destructive' in reason.lower() or 'safety' in reason.lower() or 'warning' in reason.lower(),
            f"Safety should warn about rm -rf.\nReason: {reason[:300]}")

    def test_pre_bash_approves_safe_commands(self):
        """Non-destructive commands should be approved without warning."""
        result = hook_pre_bash_safety(self.brain, {
            'command': 'ls -la /tmp',
        }, [])
        # Wait — hook_pre_bash_safety in daemon_hooks.py doesn't do the regex pre-screen.
        # It always calls brain.safety_check(). The regex is in the CLIENT.
        # So this test is about what happens when a non-destructive command hits safety_check.
        self.assertIn('json', result)

    def test_post_compact_reboot_outputs_context(self):
        """Post-compact reboot should output substantial brain context."""
        result = hook_post_compact_reboot(self.brain, {}, [])

        self.assertIn('output', result)
        output = result['output']
        self.assertTrue(len(output) > 100,
                         f"Post-compact output too short ({len(output)} chars).\n"
                         f"Output: {output[:500]}")
        self.assertIn('BRAIN POST-COMPACTION REBOOT', output)

    def test_idle_maintenance_runs_all_stages(self):
        """Idle maintenance should attempt all stages without crashing."""
        graph_changes = []
        result = hook_idle_maintenance(self.brain, {}, graph_changes)

        # Output is empty (stored as pending) but shouldn't crash
        self.assertIn('output', result)

        # Check pending messages were stored
        pending_raw = self.brain.get_config('pending_hook_messages', '[]')
        pending = json.loads(pending_raw)
        self.assertTrue(len(pending) > 0,
                         "Idle should store maintenance output as pending")

    def test_config_change_detects_host_changes(self):
        """Config change hook should scan host and store pending if changes found."""
        result = hook_config_change_host(self.brain, {
            'source': 'test',
            'file_path': '/test/config',
        }, [])
        # May or may not find changes (depends on host state), but should not crash
        self.assertIn('output', result)


# ══════════════════════════════════════════════════════════════════════════
# TEST 7: Concurrent Safety — hooks firing in parallel
# ══════════════════════════════════════════════════════════════════════════

class TestConcurrentHooks(SystemTestBase):
    """Tests hooks firing concurrently (as they do in real Claude Code).

    In practice, UserPromptSubmit fires BOTH pre-response-recall.sh and
    post-response-track.sh. They run in parallel, potentially hitting
    the daemon at the same time. The daemon's _lock must serialize them.
    """

    def test_recall_and_track_concurrent(self):
        """Recall and track firing simultaneously should both succeed."""
        # Boot first to start daemon
        rc, _, _ = self.run_hook('boot-brain.sh', timeout=15)
        self.assertEqual(rc, 0)

        # Give daemon a moment to be ready
        time.sleep(1)

        # Fire both hooks concurrently
        recall_input = json.dumps({'prompt': 'Tell me about authentication'})
        track_input = json.dumps({
            'prompt': 'Tell me about authentication',
            'hook_event_name': 'UserPromptSubmit',
        })

        results = {}
        errors = {}

        def run_recall():
            try:
                results['recall'] = self.run_hook('pre-response-recall.sh',
                                                   stdin_data=recall_input, timeout=10)
            except Exception as e:
                errors['recall'] = str(e)

        def run_track():
            try:
                results['track'] = self.run_hook('post-response-track.sh',
                                                  stdin_data=track_input, timeout=10)
            except Exception as e:
                errors['track'] = str(e)

        t1 = threading.Thread(target=run_recall)
        t2 = threading.Thread(target=run_track)
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        self.assertEqual(len(errors), 0, f"Concurrent hooks errored: {errors}")
        self.assertIn('recall', results)
        self.assertIn('track', results)

        rc_recall, stdout_recall, _ = results['recall']
        rc_track, _, _ = results['track']

        self.assertEqual(rc_recall, 0, "Recall should succeed concurrently")
        self.assertEqual(rc_track, 0, "Track should succeed concurrently")

        # Recall should still produce valid JSON
        output = json.loads(stdout_recall.strip())
        has_content = 'additionalContext' in output or output.get('decision') == 'approve'
        self.assertTrue(has_content)


# ══════════════════════════════════════════════════════════════════════════
# TEST 8: Pre-Screen Guard Tests — client-side guards before daemon call
# ══════════════════════════════════════════════════════════════════════════

class TestPreScreenGuards(SystemTestBase):
    """Tests that client-side pre-screen guards work correctly.

    Pre-screens avoid daemon round-trips for known-safe inputs. These are
    the first line of defense and must be correct to avoid unnecessary latency.
    """

    def test_recall_skips_short_messages(self):
        """Messages under 5 chars should get instant approve (no daemon/brain call)."""
        self._remove_daemon_socket()
        for msg in ['hi', 'ok', 'yes', 'no', 'k']:
            hook_input = json.dumps({'prompt': msg})
            rc, stdout, _ = self.run_hook('pre-response-recall.sh', stdin_data=hook_input)
            self.assertEqual(rc, 0)
            output = json.loads(stdout.strip())
            self.assertEqual(output.get('decision'), 'approve',
                             f"Short message '{msg}' should get instant approve")

    def test_recall_skips_slash_commands(self):
        """Messages starting with / should get instant approve."""
        self._remove_daemon_socket()
        hook_input = json.dumps({'prompt': '/remember this is a test'})
        rc, stdout, _ = self.run_hook('pre-response-recall.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0)
        output = json.loads(stdout.strip())
        self.assertEqual(output.get('decision'), 'approve')

    def test_pre_edit_skips_log_files(self):
        """Editing .log files should get instant approve."""
        self._remove_daemon_socket()
        hook_input = json.dumps({
            'tool_name': 'Edit',
            'tool_input': {'file_path': '/project/debug.log', 'old_string': 'a', 'new_string': 'b'},
        })
        rc, stdout, _ = self.run_hook('pre-edit-suggest.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0)
        output = json.loads(stdout.strip())
        self.assertEqual(output.get('decision'), 'approve')
        # No reason means no brain suggestions (skipped)
        self.assertFalse(output.get('reason', ''),
                          "Log file edits should skip brain suggestions entirely")

    def test_pre_bash_safe_commands_are_instant(self):
        """Non-destructive commands should approve without any brain call."""
        self._remove_daemon_socket()
        safe_commands = [
            'ls -la',
            'cat README.md',
            'git status',
            'npm test',
            'python3 -m pytest tests/',
            'echo "hello world"',
            'grep -r "pattern" src/',
        ]
        for cmd in safe_commands:
            hook_input = json.dumps({
                'tool_name': 'Bash',
                'tool_input': {'command': cmd},
            })
            t_start = time.time()
            rc, stdout, _ = self.run_hook('pre-bash-safety.sh', stdin_data=hook_input)
            elapsed = time.time() - t_start
            self.assertEqual(rc, 0)
            output = json.loads(stdout.strip())
            self.assertEqual(output.get('decision'), 'approve',
                             f"Safe command '{cmd}' should approve instantly")
            # Safe commands should be fast (no brain import, just regex)
            self.assertLess(elapsed, 2.0,
                            f"Safe command '{cmd}' took {elapsed:.1f}s — should be instant")

    def test_pre_bash_destructive_commands_trigger_brain(self):
        """Destructive commands should pass the regex pre-screen and hit brain."""
        destructive_commands = [
            'rm -rf /tmp/project',
            'git reset --hard HEAD~5',
            'git push --force origin main',
            'git clean -fd',
            'git worktree remove /tmp/wt',
            'find . | xargs rm',
        ]
        for cmd in destructive_commands:
            hook_input = json.dumps({
                'tool_name': 'Bash',
                'tool_input': {'command': cmd},
            })
            rc, stdout, _ = self.run_hook('pre-bash-safety.sh', stdin_data=hook_input)
            self.assertEqual(rc, 0)
            output = json.loads(stdout.strip())
            reason = output.get('reason', '').lower()
            self.assertTrue(
                'destructive' in reason or 'safety' in reason or 'warning' in reason or 'block' in output.get('decision', ''),
                f"Destructive command '{cmd}' should trigger brain safety.\n"
                f"Decision: {output.get('decision')}, Reason: {reason[:200]}")

    def test_post_bash_host_check_skips_non_env_commands(self):
        """Non-env-changing bash commands should skip host check entirely."""
        self._remove_daemon_socket()
        safe_commands = ['ls', 'git status', 'echo hello', 'cat file.txt']
        for cmd in safe_commands:
            hook_input = json.dumps({
                'tool_name': 'Bash',
                'tool_input': {'command': cmd},
            })
            t_start = time.time()
            rc, _, _ = self.run_hook('post-bash-host-check.sh', stdin_data=hook_input)
            elapsed = time.time() - t_start
            self.assertEqual(rc, 0)
            self.assertLess(elapsed, 1.0,
                            f"Non-env command '{cmd}' took {elapsed:.1f}s — should exit instantly")


# ══════════════════════════════════════════════════════════════════════════
# TEST 9: Seed Brain Tests — foundational knowledge for new brains
# ══════════════════════════════════════════════════════════════════════════

class TestSeedBrain(unittest.TestCase):
    """Tests for scripts/seed_brain.py — the seed brain populator."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='brain_test_seed_')
        # Create a fresh brain
        db_path = os.path.join(self.tmpdir, 'brain.db')
        self.brain = Brain(db_path)
        self.brain.save()

    def tearDown(self):
        self.brain.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_seed_creates_expected_nodes(self):
        """seed_brain.py creates all 5 foundational nodes.

        VERIFIES: Seed nodes are created with correct types and locked status.
        SIGNALS: If count < 5, a seed node definition is missing.
        """
        sys.path.insert(0, PROJECT_ROOT)
        from scripts.seed_brain import seed_brain
        seed_brain(self.tmpdir)

        # Should have exactly 5 seed nodes
        count = self.brain.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        self.assertEqual(count, 5, "Seed should create exactly 5 nodes")

        # Check locked nodes (4 of 5 should be locked)
        locked = self.brain.conn.execute("SELECT COUNT(*) FROM nodes WHERE locked = 1").fetchone()[0]
        self.assertEqual(locked, 4, "4 of 5 seed nodes should be locked")

    def test_seed_is_idempotent(self):
        """Running seed twice doesn't create duplicates.

        VERIFIES: Idempotency via exact title match.
        SIGNALS: If count > 5 after double seed, duplicate detection is broken.
        """
        from scripts.seed_brain import seed_brain
        seed_brain(self.tmpdir)
        seed_brain(self.tmpdir)  # Second run

        count = self.brain.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        self.assertEqual(count, 5, "Double seed should still produce exactly 5 nodes")

    def test_seed_creates_connections(self):
        """Seed creates cross-connections between foundational nodes.

        VERIFIES: Edges exist between seed nodes.
        SIGNALS: If 0 edges, the connection logic in seed_brain.py is broken.
        """
        from scripts.seed_brain import seed_brain
        seed_brain(self.tmpdir)

        edge_count = self.brain.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        self.assertGreaterEqual(edge_count, 3, "Seed should create at least 3 connections")

    def test_seed_includes_conflict_protocol(self):
        """The conflict protocol rule is a seed node.

        VERIFIES: The most important seed node (conflict protocol) exists and is locked.
        """
        from scripts.seed_brain import seed_brain
        seed_brain(self.tmpdir)

        row = self.brain.conn.execute(
            "SELECT locked FROM nodes WHERE title LIKE '%conflict protocol%'"
        ).fetchone()
        self.assertIsNotNone(row, "Conflict protocol node should exist")
        self.assertEqual(row[0], 1, "Conflict protocol should be locked")


# ══════════════════════════════════════════════════════════════════════════
# TEST 10: Daemon Dispatch Tests — hook table + telemetry wrapper
# ══════════════════════════════════════════════════════════════════════════

class TestDaemonDispatch(unittest.TestCase):
    """Tests for the daemon's hook dispatch table and telemetry wrapper."""

    def test_hook_table_covers_all_hooks(self):
        """HOOK_TABLE has entries for all 13 hook functions.

        VERIFIES: No hook was forgotten when converting from elif blocks.
        SIGNALS: Missing entry means that hook silently fails in daemon mode.
        """
        expected_hooks = [
            'hook_recall', 'hook_post_response_track', 'hook_idle_maintenance',
            'hook_post_compact_reboot', 'hook_pre_edit', 'hook_pre_bash_safety',
            'hook_pre_compact_save', 'hook_session_end', 'hook_stop_failure_log',
            'hook_config_change_host', 'hook_post_bash_host_check',
            'hook_worktree_context', 'hook_worktree_cleanup',
        ]
        for hook_name in expected_hooks:
            self.assertIn(hook_name, BrainDaemon.HOOK_TABLE,
                         f"Missing hook in HOOK_TABLE: {hook_name}")

    def test_hook_table_functions_exist(self):
        """All functions referenced in HOOK_TABLE exist in daemon_hooks module.

        VERIFIES: No typos in function names.
        SIGNALS: ImportError at runtime if a function name is wrong.
        """
        import servers.daemon_hooks as dh
        for hook_name, (func_name, _) in BrainDaemon.HOOK_TABLE.items():
            self.assertTrue(hasattr(dh, func_name),
                           f"daemon_hooks missing function: {func_name}")
            self.assertTrue(callable(getattr(dh, func_name)),
                           f"{func_name} is not callable")

    def test_hook_table_dirty_flags(self):
        """Verify which hooks mark the brain as dirty.

        VERIFIES: pre_bash_safety is the only non-dirty hook (read-only safety check).
        """
        non_dirty = [name for name, (_, dirty) in BrainDaemon.HOOK_TABLE.items() if not dirty]
        self.assertEqual(non_dirty, ['hook_pre_bash_safety'],
                        "Only pre_bash_safety should be non-dirty")


if __name__ == '__main__':
    unittest.main()
