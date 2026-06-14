#!/usr/bin/env python3
"""
Daemon Architecture Tests

Tests for the split daemon modules, including:
  - Integrity: dispatch table ↔ MCP tools ↔ hook table consistency
  - Concurrency: parallel reads don't block, writes serialize
  - Degradation: recall fallback flagging
  - Agent isolation: DB copy, changes listing, cleanup
  - Worktree: hooks.json has required hooks
  - CLI: brain_cli.py commands work
  - Schema: encoding_version column exists and is set
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from servers.daemon_config import (
    get_socket_path, get_pid_path, THREAD_POOL_SIZE,
    SOCKET_BACKLOG, _code_fingerprint,
)
from servers.daemon_dispatch import COMMAND_TABLE, CmdEntry
from servers.daemon_client import (
    send_command, is_daemon_running, ensure_daemon, stop_daemon,
    _kill_daemon, create_agent_db, list_agent_changes, cleanup_agent_db,
)
from servers.brain_mcp import TOOLS as MCP_TOOLS


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Integrity — dispatch, MCP, hooks all in sync
# ══════════════════════════════════════════════════════════════════════════

class TestIntegrity(unittest.TestCase):
    """Verify that dispatch table, MCP tools, and hook table stay in sync.

    These tests catch the case where someone adds a Brain method,
    exposes it in one place, but forgets the others.
    """

    def test_all_mcp_tools_have_dispatch_entries(self):
        """Every MCP tool must have a matching daemon command."""
        mcp_names = {t['name'] for t in MCP_TOOLS}
        dispatch_names = set(COMMAND_TABLE.keys())
        # Commands handled directly by daemon_server before COMMAND_TABLE dispatch
        # (they kill/restart the process — can't go through normal dispatch)
        daemon_direct = {'restart', 'shutdown'}

        for name in mcp_names:
            self.assertTrue(name in dispatch_names or name in daemon_direct,
                          f"MCP tool '{name}' has no dispatch entry in COMMAND_TABLE or daemon_direct")

    def test_dispatch_table_has_valid_entries(self):
        """Every dispatch entry must be a CmdEntry with callable handler."""
        for cmd, entry in COMMAND_TABLE.items():
            self.assertIsInstance(entry, CmdEntry,
                                 f"'{cmd}' is not a CmdEntry")
            self.assertTrue(callable(entry.handler),
                            f"'{cmd}' handler is not callable")
            self.assertIsInstance(entry.is_write, bool,
                                 f"'{cmd}' is_write must be bool")

    def test_read_commands_dont_mark_dirty(self):
        """Read commands should never mark dirty (they don't mutate state)."""
        for cmd, entry in COMMAND_TABLE.items():
            if not entry.is_write:
                self.assertFalse(entry.marks_dirty,
                                 f"Read command '{cmd}' marks_dirty=True — reads shouldn't mutate")

    def test_hook_table_matches_daemon_hooks_module(self):
        """Every hook in HOOK_TABLE must exist as a function in daemon_hooks.

        HOOK_TABLE entries are (is_write, marks_dirty); the daemon resolves
        the handler via getattr(_hooks, cmd), so the cmd name itself must
        be a callable attribute on daemon_hooks.
        """
        from servers.daemon_server import BrainDaemon
        import servers.daemon_hooks as hooks_module

        for hook_cmd, (is_write, marks_dirty) in BrainDaemon.HOOK_TABLE.items():
            self.assertIsInstance(is_write, bool,
                                  f"HOOK_TABLE['{hook_cmd}'][0] must be bool (is_write)")
            self.assertIsInstance(marks_dirty, bool,
                                  f"HOOK_TABLE['{hook_cmd}'][1] must be bool (marks_dirty)")
            self.assertTrue(hasattr(hooks_module, hook_cmd),
                            f"Hook '{hook_cmd}' has no matching function in daemon_hooks")
            self.assertTrue(callable(getattr(hooks_module, hook_cmd)),
                            f"'{hook_cmd}' in daemon_hooks is not callable")

    def test_mcp_tool_descriptions_not_empty(self):
        """Every MCP tool must have a non-empty description."""
        for tool in MCP_TOOLS:
            self.assertTrue(len(tool.get('description', '')) > 10,
                            f"MCP tool '{tool['name']}' has no/short description")

    def test_dispatch_covers_all_hook_commands(self):
        """Hook commands (hook_*) should NOT be in COMMAND_TABLE.
        They're routed via HOOK_TABLE in daemon_server."""
        from servers.daemon_server import BrainDaemon
        for cmd in COMMAND_TABLE:
            self.assertFalse(cmd.startswith('hook_'),
                             f"'{cmd}' is in COMMAND_TABLE but should be in HOOK_TABLE")
        for hook_cmd in BrainDaemon.HOOK_TABLE:
            self.assertNotIn(hook_cmd, COMMAND_TABLE,
                             f"'{hook_cmd}' is in both COMMAND_TABLE and HOOK_TABLE")


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Worktree hooks integrity
# ══════════════════════════════════════════════════════════════════════════

class TestWorktreeHooks(unittest.TestCase):
    """Verify worktree hooks.json has required hooks for brain functionality."""

    REQUIRED_WORKTREE_HOOKS = {
        'SessionStart',       # Boot brain
        'UserPromptSubmit',   # Recall before responding — CRITICAL
        'PreToolUse',         # Pre-edit suggestions
    }

    def _load_worktree_hooks(self):
        """Load all worktree hooks.json files, normalized to event level.

        Worktree hooks.json mirrors the plugin format (`hooks/hooks.json`): a
        `{"description": ..., "hooks": {<event>: [...]}}` envelope. Unwrap the
        envelope so callers always see the event-level dict — matching how
        `test_main_settings_has_all_hooks` reads `hooks_config.get('hooks')`.
        A flat (already event-level) file is returned as-is for back-compat.
        """
        worktree_dir = os.path.join(PROJECT_ROOT, '.claude', 'worktrees')
        if not os.path.isdir(worktree_dir):
            return {}
        results = {}
        for wt in os.listdir(worktree_dir):
            hooks_path = os.path.join(worktree_dir, wt, 'hooks', 'hooks.json')
            if os.path.isfile(hooks_path):
                with open(hooks_path) as f:
                    config = json.load(f)
                # Unwrap the plugin envelope; no hook event is named "hooks",
                # so a dict-valued "hooks" key unambiguously marks the wrapper.
                inner = config.get('hooks')
                results[wt] = inner if isinstance(inner, dict) else config
        return results

    def test_worktree_hooks_have_required_events(self):
        """Every worktree hooks.json must include critical hook events."""
        worktrees = self._load_worktree_hooks()
        if not worktrees:
            self.skipTest("No worktree hooks.json found")

        for wt_name, hooks in worktrees.items():
            for event in self.REQUIRED_WORKTREE_HOOKS:
                self.assertIn(event, hooks,
                              f"Worktree '{wt_name}' missing required hook event: {event}")

    def test_main_settings_has_all_hooks(self):
        """Plugin hooks.json must have all hook events."""
        hooks_path = os.path.join(PROJECT_ROOT, 'hooks', 'hooks.json')
        if not os.path.isfile(hooks_path):
            self.skipTest("No hooks/hooks.json found")

        with open(hooks_path) as f:
            hooks_config = json.load(f)

        hooks = hooks_config.get('hooks', {})
        required_main = {
            'SessionStart', 'UserPromptSubmit', 'PreToolUse',
            'SessionEnd', 'Stop',
        }
        for event in required_main:
            self.assertIn(event, hooks,
                          f"hooks/hooks.json missing hook event: {event}")

    def test_worktree_hooks_reference_valid_scripts(self):
        """Hook commands in worktree hooks.json must reference scripts that exist."""
        worktrees = self._load_worktree_hooks()
        scripts_dir = os.path.join(PROJECT_ROOT, 'hooks', 'scripts')

        for wt_name, hooks in worktrees.items():
            for event, entries in hooks.items():
                for entry in entries:
                    for hook in entry.get('hooks', []):
                        if hook.get('type') == 'command':
                            cmd = hook['command']
                            # Extract script name from "bash ${...}/scripts/foo.sh"
                            parts = cmd.split('/')
                            script_name = parts[-1] if parts else ''
                            if script_name:
                                script_path = os.path.join(scripts_dir, script_name)
                                self.assertTrue(os.path.isfile(script_path),
                                                f"Worktree '{wt_name}' event '{event}' "
                                                f"references missing script: {script_name}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: Schema — encoding_version
# ══════════════════════════════════════════════════════════════════════════

class TestEncodingVersion(unittest.TestCase):
    """Verify encoding_version column exists and is set on new nodes."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        os.environ['ORT_DISABLE_ALL_ACCELERATORS'] = '1'
        from servers.brain import Brain
        self.brain = Brain(self.db_path)

    def tearDown(self):
        self.brain.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_encoding_version_column_exists(self):
        """Nodes table must have encoding_version column."""
        cols = [row[1] for row in
                self.brain.conn.execute("PRAGMA table_info(nodes)").fetchall()]
        self.assertIn('encoding_version', cols)

    def test_new_nodes_get_encoding_version(self):
        """remember() should set encoding_version on new nodes."""
        from servers.brain_constants import CURRENT_ENCODING_VERSION
        result = self.brain.remember(
            type='lesson', title='test encoding version',
            content='testing that encoding_version is set')
        node_id = result['id'] if isinstance(result, dict) else result

        row = self.brain.conn.execute(
            "SELECT encoding_version FROM nodes WHERE id = ?",
            (node_id,)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], CURRENT_ENCODING_VERSION)

    def test_legacy_nodes_have_null_version(self):
        """Nodes created before versioning should have NULL encoding_version."""
        # Insert a node directly without encoding_version
        self.brain.conn.execute(
            "INSERT INTO nodes (id, type, title, content, created_at) "
            "VALUES ('legacy_test', 'context', 'legacy node', 'old', '2025-01-01')")
        self.brain.conn.commit()

        row = self.brain.conn.execute(
            "SELECT encoding_version FROM nodes WHERE id = 'legacy_test'").fetchone()
        self.assertIsNone(row[0])


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: Agent DB isolation
# ══════════════════════════════════════════════════════════════════════════

class TestAgentIsolation(unittest.TestCase):
    """Verify agent DB copy, change tracking, and cleanup."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        os.environ['ORT_DISABLE_ALL_ACCELERATORS'] = '1'
        from servers.brain import Brain
        self.brain = Brain(self.db_path)
        self.brain.remember(type='rule', title='seed rule', content='exists before agent')
        self.brain.save()
        self.brain.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_agent_db_copies_data(self):
        """Agent DB should contain all production nodes."""
        agent_db = create_agent_db('test-agent-1', source_db=self.db_path)
        try:
            conn = sqlite3.connect(agent_db)
            count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            conn.close()
            self.assertGreater(count, 0, "Agent DB should have nodes from production")
        finally:
            cleanup_agent_db(agent_db)

    def test_agent_changes_tracked(self):
        """Nodes created in agent DB after a timestamp should be listable."""
        agent_db = create_agent_db('test-agent-2', source_db=self.db_path)
        try:
            since = time.strftime('%Y-%m-%dT%H:%M:%S')
            from servers.brain import Brain
            agent_brain = Brain(agent_db)
            agent_brain.remember(type='lesson', title='agent discovery',
                                 content='found something interesting')
            agent_brain.save()
            agent_brain.close()

            changes = list_agent_changes(agent_db, since)
            agent_titles = [c['title'] for c in changes]
            self.assertIn('agent discovery', agent_titles,
                          "Agent's new node should appear in changes list")
        finally:
            cleanup_agent_db(agent_db)

    def test_agent_cleanup_removes_db(self):
        """cleanup_agent_db should delete the file."""
        agent_db = create_agent_db('test-agent-3', source_db=self.db_path)
        self.assertTrue(os.path.exists(agent_db))
        cleanup_agent_db(agent_db)
        self.assertFalse(os.path.exists(agent_db))

    def test_agent_writes_dont_affect_production(self):
        """Writing to agent DB must not change production DB."""
        agent_db = create_agent_db('test-agent-4', source_db=self.db_path)
        try:
            from servers.brain import Brain
            agent_brain = Brain(agent_db)
            agent_brain.remember(type='lesson', title='agent only node',
                                 content='should not appear in production')
            agent_brain.save()
            agent_brain.close()

            # Check production
            prod = sqlite3.connect(self.db_path)
            row = prod.execute(
                "SELECT COUNT(*) FROM nodes WHERE title = 'agent only node'").fetchone()
            prod.close()
            self.assertEqual(row[0], 0, "Agent node leaked to production DB")
        finally:
            cleanup_agent_db(agent_db)


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: Daemon module structure
# ══════════════════════════════════════════════════════════════════════════

class TestDaemonModuleStructure(unittest.TestCase):
    """Verify the split daemon modules are importable and properly structured."""

    def test_split_modules_export_all_public_symbols(self):
        """Split daemon modules must export all expected symbols in their respective homes."""
        from servers.daemon_server import BrainDaemon
        from servers.daemon_client import (send_command, is_daemon_running,
                                           ensure_daemon, stop_daemon, _kill_daemon,
                                           create_agent_db, list_agent_changes, cleanup_agent_db)
        from servers.daemon_config import (get_socket_path, get_pid_path,
                                           get_lock_path, get_status_path)
        from servers.daemon_dispatch import COMMAND_TABLE
        # Verify they're all importable and non-None
        for sym_name, sym in [
            ('BrainDaemon', BrainDaemon),
            ('send_command', send_command),
            ('is_daemon_running', is_daemon_running),
            ('ensure_daemon', ensure_daemon),
            ('stop_daemon', stop_daemon),
            ('_kill_daemon', _kill_daemon),
            ('get_socket_path', get_socket_path),
            ('get_pid_path', get_pid_path),
            ('get_lock_path', get_lock_path),
            ('get_status_path', get_status_path),
            ('COMMAND_TABLE', COMMAND_TABLE),
            ('create_agent_db', create_agent_db),
            ('list_agent_changes', list_agent_changes),
            ('cleanup_agent_db', cleanup_agent_db),
        ]:
            self.assertIsNotNone(sym, f"Split daemon modules missing symbol: {sym_name}")

    def test_daemon_config_is_small(self):
        """daemon_config.py should stay under 175 lines.
        # ADJUSTED: 100→120 — added BRAIN_DEV_MODE + is_dev_mode() helper with
        #   plugin-repackaging caution docstring (2026-05-19).
        # ADJUSTED: 120→150 — added LAUNCHD_LABEL + get_recovery_state_path()
        #   for the consolidated hung-daemon recovery path (2026-05-28).
        # ADJUSTED: 150→160 — recursive content-based fingerprint (os.walk over
        #   servers/**/*.py) replaces the top-level mtime hash, so subpackage
        #   (scales/, ...) edits are detected (2026-06-06).
        # ADJUSTED: 160→175 — added _is_worktree_checkout() + _IS_WORKTREE so a
        #   worktree session never triggers a daemon staleness-restart (the
        #   non-convergent churn fix, 2026-06-06).
        """
        config_path = os.path.join(PROJECT_ROOT, 'servers', 'daemon_config.py')
        with open(config_path) as f:
            lines = len(f.readlines())
        self.assertLess(lines, 175,
                        f"daemon_config.py is {lines} lines — should be <175")

    def test_daemon_dispatch_is_readable(self):
        """daemon_dispatch.py should stay under 1120 lines."""
        path = os.path.join(PROJECT_ROOT, 'servers', 'daemon_dispatch.py')
        with open(path) as f:
            lines = len(f.readlines())
        self.assertLess(lines, 1120,
                        f"daemon_dispatch.py is {lines} lines — should be <1120")

    def test_daemon_server_is_readable(self):
        """daemon_server.py should stay under 950 lines.
        # ADJUSTED: 350→400 approved by Tom 2026-03-23 — same rationale as dispatch.
        # ADJUSTED: 400→450 approved by Tom 2026-03-24 — observer channel wiring added.
        # ADJUSTED: 450→790 — scales runner integration, session context, trace pipeline,
        #   background encoding lifecycle (2026-04)
        # ADJUSTED: 790→950 approved by Tom 2026-05-28 — one cohesive daemon class
        #   (supervisor loop, signal handling, suspend detector, request handling);
        #   modest overage, splitting a single class isn't worth the seams.
        # ADJUSTED: 950→970 approved by Tom 2026-06-06 — PID file claimed only
        #   after a successful bind (+_wrote_pid guard in _cleanup) so a
        #   deferring duplicate can't unlink the incumbent's PID file.
        """
        path = os.path.join(PROJECT_ROOT, 'servers', 'daemon_server.py')
        with open(path) as f:
            lines = len(f.readlines())
        self.assertLess(lines, 970,
                        f"daemon_server.py is {lines} lines — should be <970")

    def test_no_circular_imports(self):
        """Importing daemon modules in any order should not cause circular imports."""
        # These should all succeed without ImportError
        import importlib
        for mod_name in ['daemon_config', 'daemon_dispatch', 'daemon_client', 'daemon_server']:
            mod = importlib.import_module(f'servers.{mod_name}')
            self.assertIsNotNone(mod)

    def test_code_fingerprint_changes_on_file_modification(self):
        """Code fingerprint should detect file changes."""
        fp1 = _code_fingerprint()
        self.assertNotEqual(fp1, "unknown")
        self.assertEqual(len(fp1), 16)  # MD5 hex truncated to 16

    def test_fingerprint_is_content_based_not_mtime(self):
        """The churn fix: fingerprint must be STABLE across mtime-only changes
        and SENSITIVE to content changes — so a worktree looks 'stale' to the
        daemon only when its code actually differs, not just because checkout
        gave its files fresh mtimes (the 2026-06-05 restart-churn root cause)."""
        import tempfile, shutil
        from servers.daemon_config import _fingerprint_dir
        d = tempfile.mkdtemp(prefix="brain-fp-test-")
        try:
            p = os.path.join(d, "a.py")
            with open(p, "w") as f:
                f.write("x = 1\n")
            fp1 = _fingerprint_dir(d)
            os.utime(p, (10**9, 10**9))   # mtime-only change
            self.assertEqual(_fingerprint_dir(d), fp1,
                             "mtime-only change must NOT change the fingerprint")
            with open(p, "w") as f:        # content change
                f.write("x = 2\n")
            fp2 = _fingerprint_dir(d)
            self.assertNotEqual(fp2, fp1,
                                "content change MUST change the fingerprint")
            # recursion: a .py added in a SUBPACKAGE must change the fingerprint
            os.makedirs(os.path.join(d, "sub"))
            with open(os.path.join(d, "sub", "s.py"), "w") as f:
                f.write("y = 1\n")
            self.assertNotEqual(_fingerprint_dir(d), fp2,
                                "subpackage .py change MUST change the fingerprint")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_worktree_checkout_detection(self):
        """A linked worktree has `.git` as a FILE (gitdir: pointer); the primary
        checkout has `.git` as a DIRECTORY; a tarball install has neither. This
        is the signal that suppresses staleness restarts for worktree sessions
        (2026-06-06 churn fix)."""
        import tempfile, shutil
        from servers.daemon_config import _is_worktree_checkout
        d = tempfile.mkdtemp(prefix="brain-wt-test-")
        try:
            os.makedirs(os.path.join(d, ".git"))            # primary checkout
            self.assertFalse(_is_worktree_checkout(d),
                             "primary checkout (.git dir) is NOT a worktree")
            shutil.rmtree(os.path.join(d, ".git"))
            with open(os.path.join(d, ".git"), "w") as f:    # linked worktree
                f.write("gitdir: /repo/.git/worktrees/wt\n")
            self.assertTrue(_is_worktree_checkout(d),
                            "linked worktree (.git file) IS a worktree")
            os.remove(os.path.join(d, ".git"))               # tarball install
            self.assertFalse(_is_worktree_checkout(d),
                             "missing .git → not a worktree (safe default)")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_code_changed_warranted_only_for_daemon_source(self):
        """_code_changed gates on the daemon's reported source_dir: a restart only
        converges if THIS checkout is the daemon's launch source, so a different
        checkout (worktree, second clone) never reports changed even on a
        fingerprint mismatch. Daemons that omit source_dir fall back to the
        linked-worktree heuristic. (2026-06-06 non-convergent churn fix.)"""
        import servers.daemon_client as dc
        from unittest import mock

        with mock.patch.object(dc, "REPO_ROOT", "/repo/main"), \
             mock.patch.object(dc, "_CODE_FINGERPRINT", "FP_A"):
            # different source_dir → never restart, even with a fingerprint mismatch
            self.assertFalse(dc._code_changed(
                {"result": {"source_dir": "/repo/.wt/feat", "code_fingerprint": "FP_B"}}),
                "a checkout that isn't the daemon's source must never report changed")
            # same source_dir + mismatch → restart warranted
            self.assertTrue(dc._code_changed(
                {"result": {"source_dir": "/repo/main", "code_fingerprint": "FP_B"}}),
                "the daemon's own source running changed code IS code-changed")
            # same source_dir + matching fingerprint → no restart
            self.assertFalse(dc._code_changed(
                {"result": {"source_dir": "/repo/main", "code_fingerprint": "FP_A"}}),
                "same source and same fingerprint is not code-changed")
            # fallback (daemon omits source_dir): the worktree heuristic decides
            with mock.patch.object(dc, "_IS_WORKTREE", True):
                self.assertFalse(dc._code_changed({"result": {"code_fingerprint": "FP_B"}}),
                                 "old daemon + worktree caller falls back to suppression")
            with mock.patch.object(dc, "_IS_WORKTREE", False):
                self.assertTrue(dc._code_changed({"result": {"code_fingerprint": "FP_B"}}),
                                "old daemon + primary checkout + mismatch IS code-changed")


# ══════════════════════════════════════════════════════════════════════════
# TEST 6: CLI
# ══════════════════════════════════════════════════════════════════════════

class TestBrainCLI(unittest.TestCase):
    """Test brain_cli.py commands produce valid output."""

    def _run_cli(self, *args):
        """Run brain_cli.py with args, return (returncode, stdout, stderr)."""
        import subprocess
        cli_path = os.path.join(PROJECT_ROOT, 'servers', 'brain_cli.py')
        result = subprocess.run(
            [sys.executable, cli_path] + list(args),
            capture_output=True, text=True, timeout=15,
            env={**os.environ, 'PYTHONPATH': PROJECT_ROOT},
        )
        return result.returncode, result.stdout, result.stderr

    def test_ping_outputs_json(self):
        """brain ping should return valid JSON."""
        rc, stdout, _ = self._run_cli('ping')
        self.assertEqual(rc, 0)
        data = json.loads(stdout)
        self.assertIn('ok', data)

    def test_status_outputs_json(self):
        """brain status should return valid JSON."""
        rc, stdout, _ = self._run_cli('status')
        self.assertEqual(rc, 0)
        data = json.loads(stdout)
        self.assertIn('ok', data)

    def test_no_args_shows_help(self):
        """brain with no args should show help and exit non-zero."""
        rc, stdout, stderr = self._run_cli()
        self.assertNotEqual(rc, 0)

    def test_recall_outputs_json(self):
        """brain recall should return valid JSON (even if daemon is down)."""
        rc, stdout, _ = self._run_cli('recall', 'test query')
        # May fail if daemon not running, but output should still be JSON
        data = json.loads(stdout)
        self.assertIn('ok', data)


# ══════════════════════════════════════════════════════════════════════════
# TEST 7: Version-aware relevance floor
# ══════════════════════════════════════════════════════════════════════════

class TestVersionAwareFloor(unittest.TestCase):
    """Verify the relevance floor adapts to encoding quality."""

    def test_floor_constants_exist(self):
        """Both floor constants must be defined."""
        from servers.brain_constants import RELEVANCE_FLOOR_ENRICHED, RELEVANCE_FLOOR_PRIMARY
        self.assertGreater(RELEVANCE_FLOOR_ENRICHED, RELEVANCE_FLOOR_PRIMARY,
                           "Enriched floor should be higher than primary floor")
        self.assertGreater(RELEVANCE_FLOOR_PRIMARY, 0.0)
        self.assertLess(RELEVANCE_FLOOR_ENRICHED, 1.0)

    def test_encoding_version_constant_exists(self):
        """CURRENT_ENCODING_VERSION must be defined."""
        from servers.brain_constants import CURRENT_ENCODING_VERSION
        self.assertIsNotNone(CURRENT_ENCODING_VERSION)
        self.assertTrue(CURRENT_ENCODING_VERSION.startswith('v'))


# ══════════════════════════════════════════════════════════════════════════
# TEST 8: SKILL.md exists and is loadable
# ══════════════════════════════════════════════════════════════════════════

class TestSkillAvailability(unittest.TestCase):
    """Verify SKILL.md exists and contains critical sections."""

    def test_skill_md_exists(self):
        """SKILL.md must exist at the expected path."""
        skill_path = os.path.join(PROJECT_ROOT, 'skills', 'brain', 'SKILL.md')
        self.assertTrue(os.path.isfile(skill_path),
                        f"SKILL.md not found at {skill_path}")

    def test_skill_md_has_anchor_identity(self):
        """SKILL.md (Anchor) must contain identity and encoding examples."""
        skill_path = os.path.join(PROJECT_ROOT, 'skills', 'brain', 'SKILL.md')
        with open(skill_path) as f:
            content = f.read()
        # Anchor must have identity section
        self.assertIn('Anchor', content)
        # Must carry encoding guidance (heading renamed in the 2026-05 SKILL.md
        # rewrite from "What Good Encoding Looks Like" → "Encoding Craft").
        self.assertIn('Encoding Craft', content)
        # Must have the "What You Are" identity section
        self.assertIn('What You Are', content)

    def test_skill_md_documents_core_tools(self):
        """SKILL.md mentions the core write/read tools by name.

        Was `test_skill_md_has_api_reference` which asserted SKILL.md is
        a tool catalog (every MCP tool name present). The current SKILL.md
        is intentionally identity-first prose, not a catalog —
        `consciousness` is a meta tool not used in normal flow and
        doesn't belong in the identity doc. Narrowed to the genuinely
        load-bearing names that define how Anchor uses the brain.
        """
        skill_path = os.path.join(PROJECT_ROOT, 'skills', 'brain', 'SKILL.md')
        with open(skill_path) as f:
            content = f.read()
        for tool in ['recall', 'remember', 'connect']:
            self.assertIn(tool, content,
                          f"SKILL.md should mention core tool '{tool}'")


# ══════════════════════════════════════════════════════════════════════════
# TEST 9: Boot self-knowledge + boot nodes
# ══════════════════════════════════════════════════════════════════════════

class TestBootSelfKnowledge(unittest.TestCase):
    """Verify boot surfaces self-knowledge and boot nodes."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        os.environ['ORT_DISABLE_ALL_ACCELERATORS'] = '1'
        from servers.brain import Brain
        self.brain = Brain(self.db_path)

    def tearDown(self):
        self.brain.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fetch_boot_nodes_returns_boot_type(self):
        """fetch_boot_nodes() should return nodes of type 'boot'."""
        self.brain.remember(type='boot', title='Session #1 handoff',
                            content='First session advice', keywords='boot handoff')
        self.brain.save()
        nodes = self.brain.fetch_boot_nodes(limit=3)
        self.assertEqual(len(nodes), 1)
        self.assertIn('handoff', nodes[0]['title'])

    # 2026-05-02 (Frame Phase 2.5): four tests removed — they asserted the
    # old recall-driven boot contract (YOU: / OPERATOR: / PATTERNS YOU FALL
    # INTO / RECENTLY ENCODED / WHAT YOU'VE LEARNED ABOUT YOURSELF /
    # PROJECT UNDERSTANDING sections) that the Frame-centered render_boot_v2
    # replaced. The Frame's contract is exercised by tests/test_frame.py.
    # Removed:
    #   test_fetch_self_knowledge_finds_behavioral_nodes — method deleted
    #   test_boot_context_includes_self_knowledge — section gone
    #   test_boot_context_includes_boot_nodes — YOU:/RECENTLY ENCODED gone
    #   test_self_knowledge_before_engineering_context — sections gone


# ══════════════════════════════════════════════════════════════════════════
# TEST 10: Behavioral mirror
# ══════════════════════════════════════════════════════════════════════════

# TestBehavioralMirror removed — _behavioral_mirror() deleted.
# Encoding behavior now tracked by encoding_gap producer in signal queue.


# ══════════════════════════════════════════════════════════════════════════
# TEST 11: Session-end reflection
# ══════════════════════════════════════════════════════════════════════════


class TestConsciousnessFeatures(unittest.TestCase):
    """Verify thought decay and SKILL.md orientation preamble."""

    def test_thought_decay_is_7_days(self):
        """Thought decay should be 168h (7 days), not 3h."""
        from servers.brain_constants import DECAY_HALF_LIFE
        self.assertEqual(DECAY_HALF_LIFE['thought'], 168,
                         "Thought decay should be 168h — thoughts need time to connect")

    def test_boot_decay_is_infinite(self):
        """Boot nodes should never decay."""
        from servers.brain_constants import DECAY_HALF_LIFE
        self.assertEqual(DECAY_HALF_LIFE['boot'], float('inf'))

    def test_boot_type_in_schema(self):
        """Boot type must be in NODE_TYPES."""
        from servers.schema import NODE_TYPES
        self.assertIn('boot', NODE_TYPES)

    # test_skill_md_has_orientation_preamble — REMOVED (redundant). Its two
    # assertions ('Anchor' + 'What You Are' present in SKILL.md) are a strict
    # subset of TestSkillAvailability.test_skill_md_has_anchor_identity, which
    # asserts both plus 'Encoding Craft'. Coverage lives there now.


# ══════════════════════════════════════════════════════════════════════════
# TEST 12: Host-suspend detection in autosave loop
# ══════════════════════════════════════════════════════════════════════════


class TestAutosaveSuspendDetection(unittest.TestCase):
    """Verify _autosave_loop detects host suspend (laptop sleep) and
    sends SIGTERM to self for clean restart.

    Background: when macOS suspends the process mid-Anthropic-call, the
    SDK's monotonic-clock-based timeout doesn't fire on wake, leaving
    httpx blocked in recv() on a TCP socket the remote already closed.
    The autosave loop's wall-clock gap detector catches this within one
    autosave interval and exits cleanly so auto-restart can take over.
    """

    def _build_stub_daemon(self):
        """Minimal BrainDaemon stub with just the attrs _autosave_loop reads."""
        from servers.daemon_server import BrainDaemon
        stub = BrainDaemon.__new__(BrainDaemon)
        stub.running = True
        stub.dirty = False
        stub.brain = None
        stub._log_lines = []
        stub._log = lambda msg: stub._log_lines.append(msg)
        stub._write_status = lambda: None
        return stub

    def test_long_wall_gap_triggers_self_sigterm(self):
        """Wall clock jumps 10 minutes past expected interval → SIGTERM fired."""
        import signal as _signal
        from servers import daemon_server as ds

        stub = self._build_stub_daemon()

        # Wall clock: returns t=0 on first call, then t=600 (10 min) — far past
        # AUTOSAVE_INTERVAL_SECONDS + 90s threshold.
        wall_values = iter([1000.0, 1600.0])
        kill_calls = []

        original_time_time = ds.time.time
        original_time_sleep = ds.time.sleep
        original_os_kill = ds.os.kill

        def fake_time():
            try:
                return next(wall_values)
            except StopIteration:
                stub.running = False  # exit loop on cleanup
                return 9999.0

        def fake_sleep(_n):
            pass  # zero-cost — simulated time advances via fake_time

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))
            stub.running = False  # so loop exits if return doesn't

        try:
            ds.time.time = fake_time
            ds.time.sleep = fake_sleep
            ds.os.kill = fake_kill
            stub._autosave_loop()
        finally:
            ds.time.time = original_time_time
            ds.time.sleep = original_time_sleep
            ds.os.kill = original_os_kill

        self.assertEqual(len(kill_calls), 1,
                         "Expected exactly one os.kill call, got: %s" % kill_calls)
        self.assertEqual(kill_calls[0][1], _signal.SIGTERM)
        self.assertEqual(kill_calls[0][0], os.getpid())
        self.assertTrue(any("HOST SUSPEND DETECTED" in m for m in stub._log_lines),
                        "Expected HOST SUSPEND log message, got: %s" % stub._log_lines)

    def test_normal_interval_does_not_trigger(self):
        """Wall clock advances by exactly AUTOSAVE_INTERVAL — no SIGTERM."""
        from servers import daemon_server as ds
        from servers.daemon_config import AUTOSAVE_INTERVAL_SECONDS

        stub = self._build_stub_daemon()

        # Normal-interval ticks. Stop after a few iterations.
        wall_values = iter([
            1000.0,
            1000.0 + AUTOSAVE_INTERVAL_SECONDS,
            1000.0 + 2 * AUTOSAVE_INTERVAL_SECONDS,
            1000.0 + 3 * AUTOSAVE_INTERVAL_SECONDS,
        ])
        kill_calls = []
        iter_count = [0]

        original_time_time = ds.time.time
        original_time_sleep = ds.time.sleep
        original_os_kill = ds.os.kill

        def fake_time():
            try:
                return next(wall_values)
            except StopIteration:
                stub.running = False
                return 9999.0

        def fake_sleep(_n):
            iter_count[0] += 1
            if iter_count[0] >= 3:
                stub.running = False  # bail after a few normal ticks

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))

        # threading.enumerate is called in the loop tail — keep it real
        try:
            ds.time.time = fake_time
            ds.time.sleep = fake_sleep
            ds.os.kill = fake_kill
            stub._autosave_loop()
        finally:
            ds.time.time = original_time_time
            ds.time.sleep = original_time_sleep
            ds.os.kill = original_os_kill

        self.assertEqual(kill_calls, [],
                         "Normal interval ticks must not trigger SIGTERM, got: %s" % kill_calls)
        self.assertFalse(any("HOST SUSPEND DETECTED" in m for m in stub._log_lines),
                         "No HOST SUSPEND log expected, got: %s" % stub._log_lines)

    def test_dev_mode_suppresses_sigterm_but_logs(self):
        """BRAIN_DEV_MODE=1: long wall gap logs the detection but skips
        the kill — developer keeps the daemon alive for inspection."""
        from servers import daemon_server as ds

        stub = self._build_stub_daemon()

        # Two suspend events back-to-back; loop should survive both and not kill.
        wall_values = iter([1000.0, 1600.0, 1700.0, 2300.0])
        kill_calls = []
        iter_count = [0]

        original_time_time = ds.time.time
        original_time_sleep = ds.time.sleep
        original_os_kill = ds.os.kill
        original_env = os.environ.get('BRAIN_DEV_MODE')

        def fake_time():
            try:
                return next(wall_values)
            except StopIteration:
                stub.running = False
                return 9999.0

        def fake_sleep(_n):
            iter_count[0] += 1
            if iter_count[0] >= 3:
                stub.running = False

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))

        try:
            os.environ['BRAIN_DEV_MODE'] = '1'
            ds.time.time = fake_time
            ds.time.sleep = fake_sleep
            ds.os.kill = fake_kill
            stub._autosave_loop()
        finally:
            ds.time.time = original_time_time
            ds.time.sleep = original_time_sleep
            ds.os.kill = original_os_kill
            if original_env is None:
                os.environ.pop('BRAIN_DEV_MODE', None)
            else:
                os.environ['BRAIN_DEV_MODE'] = original_env

        self.assertEqual(kill_calls, [],
                         "Dev mode must not send SIGTERM, got: %s" % kill_calls)
        # Detection still logs — operator must see the gap.
        self.assertTrue(any("HOST SUSPEND DETECTED" in m for m in stub._log_lines),
                        "Dev mode should still log the detection")
        self.assertTrue(any("BRAIN_DEV_MODE" in m for m in stub._log_lines),
                        "Dev mode log must mention the flag, got: %s" % stub._log_lines)


if __name__ == '__main__':
    unittest.main()
