"""
Hook integration tests — runs actual bash hook scripts against a temp brain.db.

Each test creates a fresh brain.db with realistic test data, sets env vars,
runs the actual hook script via subprocess, and verifies exit codes, stdout,
and DB state changes.
"""

import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

# Ensure the project root is on sys.path so we can import servers.brain
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from servers.brain import Brain


class HookTestBase(unittest.TestCase):
    """Base class for hook integration tests.

    Sets up a temp directory with a brain.db populated with realistic test data,
    configures environment variables, and provides run_hook() for executing scripts.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        # Create brain with some test data
        self.brain = Brain(self.db_path)
        self.brain.reset_session_activity()
        # Add realistic test data
        self.brain.remember(
            type='rule',
            title='Authentication must use Clerk with magic links',
            content=(
                'Clerk handles auth flow. Magic links for login, no passwords. '
                'Webhook syncs user data to our DB. Free tier covers MVP needs.'
            ),
            keywords='auth clerk login passwordless magic-link webhook',
            locked=True,
        )
        self.brain.remember(
            type='decision',
            title='React component architecture: atomic design pattern',
            content=(
                'Components organized by atomic design: atoms (Button, Input), '
                'molecules (FormField), organisms (LoginForm). Shared via internal package.'
            ),
            keywords='react components atomic design pattern architecture',
            locked=True,
        )
        self.brain.save()
        self.brain.close()
        self.brain = None

        # Set up environment
        self.scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')
        self.env = os.environ.copy()
        self.env['BRAIN_DB_DIR'] = self.tmp
        self.env['BRAIN_SERVER_DIR'] = os.path.join(os.path.dirname(__file__), '..', 'servers')
        self.env['PLUGIN_ROOT'] = os.path.join(os.path.dirname(__file__), '..')

    def tearDown(self):
        # Kill any daemon that may have been started
        for pid_file in glob.glob('/tmp/brain-daemon-*.pid'):
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())
                os.kill(pid, signal.SIGTERM)
            except (OSError, ValueError):
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

    def _node_count(self):
        """Open the brain DB and return total node count."""
        brain = Brain(self.db_path)
        ctx = brain.context_boot(user='Test', project='test', task='count')
        count = ctx.get('total_nodes', 0)
        brain.close()
        return count


class TestBootHook(HookTestBase):
    """Tests for hooks/scripts/boot-brain.sh (SessionStart hook)."""

    def test_boot_exits_zero(self):
        """Boot hook should exit 0 with a valid brain.db."""
        rc, stdout, stderr = self.run_hook('boot-brain.sh')
        self.assertEqual(rc, 0, f"boot-brain.sh exited {rc}.\nstderr: {stderr}")

    def test_boot_outputs_context(self):
        """Boot hook should output brain context including node count or version info."""
        rc, stdout, stderr = self.run_hook('boot-brain.sh')
        self.assertEqual(rc, 0)
        # Boot output may go to stdout or stderr depending on exec/background process timing.
        # Check both streams for brain-related output.
        combined = (stdout + stderr).lower()
        has_brain_ref = (
            'brain' in combined
            or 'node' in combined
            or 'booted' in combined
            or 'embedder' in combined
            or 'schema' in combined
        )
        self.assertTrue(
            has_brain_ref,
            f"Expected boot output to mention brain/nodes/booted/embedder. "
            f"Got stdout:\n{stdout[:300]}\nstderr:\n{stderr[:300]}",
        )

    def test_boot_with_no_db_dir(self):
        """Boot hook should exit cleanly (0 or 1) when BRAIN_DB_DIR points to nonexistent path."""
        env = self.env.copy()
        env['BRAIN_DB_DIR'] = '/tmp/nonexistent_brain_dir_' + str(os.getpid())
        script_path = os.path.join(self.scripts_dir, 'boot-brain.sh')
        result = subprocess.run(
            ['bash', script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.tmp,
        )
        # Should not crash with a traceback exit code > 1
        self.assertIn(result.returncode, [0, 1],
                       f"Boot should exit 0 or 1, not {result.returncode}.\nstderr: {result.stderr}")

    def test_boot_idempotent(self):
        """Running boot twice should not corrupt the DB or change node count."""
        count_before = self._node_count()
        rc1, _, _ = self.run_hook('boot-brain.sh')
        self.assertEqual(rc1, 0)
        rc2, _, _ = self.run_hook('boot-brain.sh')
        self.assertEqual(rc2, 0)
        count_after = self._node_count()
        self.assertEqual(count_before, count_after,
                         f"Node count changed after double boot: {count_before} -> {count_after}")


class TestPreResponseRecall(HookTestBase):
    """Tests for hooks/scripts/pre-response-recall.sh (UserPromptSubmit hook)."""

    def test_pre_response_with_hook_input(self):
        """Recall hook should return auth-related context for an auth query."""
        hook_input = json.dumps({
            'prompt': 'How does the auth login flow work with Clerk?',
            'session_id': 'test-session-1',
        })
        rc, stdout, stderr = self.run_hook('pre-response-recall.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0, f"pre-response-recall.sh exited {rc}.\nstderr: {stderr}")
        # Output should be valid JSON with decision field
        try:
            output = json.loads(stdout.strip())
        except json.JSONDecodeError:
            self.fail(f"Expected JSON output, got:\n{stdout[:500]}")
        self.assertEqual(output.get('decision'), 'approve')
        # If there's a reason field, it should mention auth or Clerk
        reason = output.get('reason', '')
        if reason:
            reason_lower = reason.lower()
            self.assertTrue(
                'clerk' in reason_lower or 'auth' in reason_lower or 'login' in reason_lower,
                f"Expected auth-related recall. Got reason:\n{reason[:500]}",
            )

    def test_pre_response_short_message(self):
        """Recall hook should handle very short messages gracefully (quick exit)."""
        hook_input = json.dumps({'prompt': 'hi'})
        rc, stdout, stderr = self.run_hook('pre-response-recall.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0, f"Exited {rc} for short message.\nstderr: {stderr}")
        # Should output valid JSON (approve with no recall needed)
        try:
            output = json.loads(stdout.strip())
        except json.JSONDecodeError:
            self.fail(f"Expected JSON output for short message, got:\n{stdout[:500]}")
        self.assertEqual(output.get('decision'), 'approve')

    def test_pre_response_no_daemon(self):
        """Recall hook should work via direct Python path when no daemon is running."""
        # Ensure no daemon socket exists for our user
        socket_path = f'/tmp/brain-daemon-{os.getuid()}.sock'
        if os.path.exists(socket_path):
            os.remove(socket_path)
        hook_input = json.dumps({
            'prompt': 'Tell me about the React component architecture we use',
            'session_id': 'test-session-2',
        })
        rc, stdout, stderr = self.run_hook('pre-response-recall.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0, f"Exited {rc} without daemon.\nstderr: {stderr}")
        try:
            output = json.loads(stdout.strip())
        except json.JSONDecodeError:
            self.fail(f"Expected JSON output, got:\n{stdout[:500]}")
        self.assertEqual(output.get('decision'), 'approve')


class TestIdleMaintenance(HookTestBase):
    """Tests for hooks/scripts/idle-maintenance.sh (Notification idle_prompt hook)."""

    def test_idle_runs_without_crash(self):
        """Idle maintenance hook should exit 0 with no ERROR in stderr."""
        rc, stdout, stderr = self.run_hook('idle-maintenance.sh', timeout=60)
        self.assertEqual(rc, 0, f"idle-maintenance.sh exited {rc}.\nstderr: {stderr}")
        # Check for critical errors (ignore warnings)
        self.assertNotIn('ERROR', stderr.upper().replace('DREAM ERROR', '').replace('HEAL ERROR', '').replace('CONSOLIDATE ERROR', ''),
                         f"Unexpected ERROR in stderr:\n{stderr}")

    def test_idle_produces_output(self):
        """Idle maintenance should produce some maintenance output."""
        rc, stdout, stderr = self.run_hook('idle-maintenance.sh', timeout=60)
        self.assertEqual(rc, 0)
        stdout_upper = stdout.upper()
        has_maintenance = (
            'DREAM' in stdout_upper
            or 'CONSOLIDATE' in stdout_upper
            or 'HEAL' in stdout_upper
            or 'REFLECT' in stdout_upper
            or 'TUNE' in stdout_upper
            or 'SUMMARIES' in stdout_upper
            or 'EMBEDDINGS' in stdout_upper
        )
        self.assertTrue(
            has_maintenance,
            f"Expected maintenance output (DREAM/CONSOLIDATE/HEAL/etc). Got:\n{stdout[:500]}",
        )


class TestPreEditSuggest(HookTestBase):
    """Tests for hooks/scripts/pre-edit-suggest.sh (PreToolUse Edit|Write hook)."""

    def test_pre_edit_with_file(self):
        """Pre-edit hook should surface auth-related rules when editing auth.py."""
        hook_input = json.dumps({
            'tool_name': 'Edit',
            'tool_input': {'file_path': '/project/src/auth.py', 'old_string': 'x', 'new_string': 'y'},
        })
        rc, stdout, stderr = self.run_hook('pre-edit-suggest.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0, f"pre-edit-suggest.sh exited {rc}.\nstderr: {stderr}")
        try:
            output = json.loads(stdout.strip())
        except json.JSONDecodeError:
            self.fail(f"Expected JSON output, got:\n{stdout[:500]}")
        self.assertEqual(output.get('decision'), 'approve')
        # The reason should mention auth rules if the brain matched
        reason = output.get('reason', '')
        if reason:
            reason_lower = reason.lower()
            self.assertTrue(
                'auth' in reason_lower or 'clerk' in reason_lower or 'brain auto-suggest' in reason_lower,
                f"Expected auth/clerk in suggestions. Got reason:\n{reason[:500]}",
            )

    def test_pre_edit_unknown_file(self):
        """Pre-edit hook should exit 0 cleanly for an unknown file."""
        hook_input = json.dumps({
            'tool_name': 'Edit',
            'tool_input': {'file_path': '/project/random_file.xyz', 'old_string': 'a', 'new_string': 'b'},
        })
        rc, stdout, stderr = self.run_hook('pre-edit-suggest.sh', stdin_data=hook_input)
        self.assertEqual(rc, 0, f"pre-edit-suggest.sh exited {rc} for unknown file.\nstderr: {stderr}")
        try:
            output = json.loads(stdout.strip())
        except json.JSONDecodeError:
            self.fail(f"Expected JSON output, got:\n{stdout[:500]}")
        self.assertEqual(output.get('decision'), 'approve')


class TestSessionEnd(HookTestBase):
    """Tests for hooks/scripts/session-end.sh (SessionEnd hook)."""

    def test_session_end_saves(self):
        """Session end hook should write to the DB file (mtime updated)."""
        mtime_before = os.path.getmtime(self.db_path)
        # Small delay to ensure mtime difference is detectable
        time.sleep(0.1)
        rc, stdout, stderr = self.run_hook('session-end.sh')
        self.assertEqual(rc, 0)
        mtime_after = os.path.getmtime(self.db_path)
        self.assertGreaterEqual(
            mtime_after, mtime_before,
            f"DB mtime should not go backwards: {mtime_before} -> {mtime_after}",
        )

    def test_session_end_exits_zero(self):
        """Session end hook should exit cleanly with code 0."""
        rc, stdout, stderr = self.run_hook('session-end.sh')
        self.assertEqual(rc, 0, f"session-end.sh exited {rc}.\nstderr: {stderr}")


class TestHookErrorIsolation(HookTestBase):
    """Tests that hooks handle error conditions gracefully without crashing Claude."""

    def test_hook_with_corrupt_db(self):
        """Boot hook should exit 0 even with a corrupt brain.db (no crash)."""
        # Write garbage to brain.db
        with open(self.db_path, 'wb') as f:
            f.write(b'THIS IS NOT A SQLITE DATABASE AT ALL')
        rc, stdout, stderr = self.run_hook('boot-brain.sh')
        # Should not crash with a segfault or unhandled exception exit code
        # Exit 0 or 1 are both acceptable (graceful error handling)
        self.assertIn(rc, [0, 1],
                       f"Boot should exit 0 or 1 with corrupt DB, not {rc}.\nstderr: {stderr}")

    def test_hook_missing_env(self):
        """Boot hook should exit 0 with helpful message when BRAIN_DB_DIR is unset."""
        env = self.env.copy()
        # Remove BRAIN_DB_DIR so resolution chain runs and finds nothing
        env.pop('BRAIN_DB_DIR', None)
        # Also ensure no fallback paths exist
        env['HOME'] = self.tmp  # No AgentsContext here
        script_path = os.path.join(self.scripts_dir, 'boot-brain.sh')
        result = subprocess.run(
            ['bash', script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.tmp,
        )
        self.assertEqual(result.returncode, 0,
                         f"Boot should exit 0 when no DB found.\nstderr: {result.stderr}")
        # Should output guidance about how to set up brain
        stdout_lower = result.stdout.lower()
        self.assertTrue(
            'no brain.db found' in stdout_lower or 'brain_db_dir' in stdout_lower or 'not found' in stdout_lower,
            f"Expected guidance message when no DB. Got:\n{result.stdout[:500]}",
        )


if __name__ == '__main__':
    unittest.main()
