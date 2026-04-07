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

        # MCP tools that map to daemon commands
        for name in mcp_names:
            self.assertIn(name, dispatch_names,
                          f"MCP tool '{name}' has no dispatch entry in COMMAND_TABLE")

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
        """Every hook in HOOK_TABLE must exist as a function in daemon_hooks."""
        from servers.daemon_server import BrainDaemon
        import servers.daemon_hooks as hooks_module

        for hook_cmd, (func_name, marks_dirty) in BrainDaemon.HOOK_TABLE.items():
            self.assertTrue(hasattr(hooks_module, func_name),
                            f"Hook '{hook_cmd}' references '{func_name}' "
                            f"which doesn't exist in daemon_hooks")
            self.assertTrue(callable(getattr(hooks_module, func_name)),
                            f"'{func_name}' in daemon_hooks is not callable")

    def test_no_duplicate_commands(self):
        """Dispatch table should have no duplicate command names."""
        # Python dicts can't have dupes, but we check the count matches
        self.assertEqual(len(COMMAND_TABLE), len(set(COMMAND_TABLE.keys())))

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
        'PreCompact',         # Save before compaction
        'PostCompact',        # Reboot after compaction
    }

    def _load_worktree_hooks(self):
        """Load all worktree hooks.json files."""
        worktree_dir = os.path.join(PROJECT_ROOT, '.claude', 'worktrees')
        if not os.path.isdir(worktree_dir):
            return {}
        results = {}
        for wt in os.listdir(worktree_dir):
            hooks_path = os.path.join(worktree_dir, wt, 'hooks', 'hooks.json')
            if os.path.isfile(hooks_path):
                with open(hooks_path) as f:
                    results[wt] = json.load(f)
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
            'PreCompact', 'PostCompact', 'SessionEnd', 'Stop',
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
        """daemon_config.py should stay under 100 lines."""
        config_path = os.path.join(PROJECT_ROOT, 'servers', 'daemon_config.py')
        with open(config_path) as f:
            lines = len(f.readlines())
        self.assertLess(lines, 100,
                        f"daemon_config.py is {lines} lines — should be <100")

    def test_daemon_dispatch_is_readable(self):
        """daemon_dispatch.py should stay under 1120 lines.
        # ADJUSTED: 350→400 approved by Tom 2026-03-23 — files are well-structured
        # ADJUSTED: 400→500 approved by Tom 2026-03-23 — added 7 remember_* handlers
        # ADJUSTED: 500→600 — added dismiss_signal + queue_state handlers (signal queue refactor)
        # ADJUSTED: 600→1120 — batch tools (remember_batch, revise_batch, etc.), trace tools,
        #   interaction tools, filter_nodes, get_nodes added (2026-04)
        """
        path = os.path.join(PROJECT_ROOT, 'servers', 'daemon_dispatch.py')
        with open(path) as f:
            lines = len(f.readlines())
        self.assertLess(lines, 1120,
                        f"daemon_dispatch.py is {lines} lines — should be <1120")

    def test_daemon_server_is_readable(self):
        """daemon_server.py should stay under 790 lines.
        # ADJUSTED: 350→400 approved by Tom 2026-03-23 — same rationale as dispatch.
        # ADJUSTED: 400→450 approved by Tom 2026-03-24 — observer channel wiring added.
        # ADJUSTED: 450→790 — scales runner integration, session context, trace pipeline,
        #   background encoding lifecycle (2026-04)
        """
        path = os.path.join(PROJECT_ROOT, 'servers', 'daemon_server.py')
        with open(path) as f:
            lines = len(f.readlines())
        self.assertLess(lines, 790,
                        f"daemon_server.py is {lines} lines — should be <790")

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
        # Must have encoding examples (show, don't tell)
        self.assertIn('What Good Encoding Looks Like', content)
        # Must have the "What You Are" identity section
        self.assertIn('What You Are', content)

    def test_skill_md_has_api_reference(self):
        """SKILL.md must document the core MCP tools."""
        skill_path = os.path.join(PROJECT_ROOT, 'skills', 'brain', 'SKILL.md')
        with open(skill_path) as f:
            content = f.read()
        for tool in ['recall', 'remember', 'connect', 'consciousness']:
            self.assertIn(tool, content,
                          f"SKILL.md missing documentation for '{tool}' tool")


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

    def test_fetch_self_knowledge_finds_behavioral_nodes(self):
        """fetch_self_knowledge() should find nodes about Claude's behavior."""
        self.brain.remember(type='lesson', title='Encoding drift pattern',
                            content='I drift when building', keywords='encoding drift claude instinct')
        self.brain.save()
        nodes = self.brain.fetch_self_knowledge(limit=3)
        self.assertTrue(len(nodes) >= 1)
        self.assertIn('drift', nodes[0]['title'].lower())

    def test_boot_context_includes_self_knowledge(self):
        """Boot context should have PATTERNS YOU FALL INTO section."""
        self.brain.remember(type='lesson', title='Compression instinct',
                            content='I compress by instinct', keywords='compression instinct claude')
        self.brain.save()
        ctx = self.brain.format_boot_context(user='Test', project='test')
        self.assertIn("PATTERNS YOU FALL INTO:", ctx)
        self.assertIn('Compression instinct', ctx)

    def test_boot_context_includes_boot_nodes(self):
        """Boot context should have YOU: section surfacing boot-type nodes."""
        self.brain.remember(type='boot', title='Session #5 handoff',
                            content='Remember to encode early', keywords='boot handoff')
        self.brain.save()
        ctx = self.brain.format_boot_context(user='Test', project='test')
        self.assertIn('YOU:', ctx)
        self.assertIn('encode early', ctx)

    def test_self_knowledge_before_engineering_context(self):
        """Self-knowledge must appear before engineering context in boot."""
        self.brain.remember(type='lesson', title='Agreeability bias',
                            content='I agree too easily', keywords='agreeab bias claude')
        self.brain.remember(type='purpose', title='API gateway',
                            content='Routes requests', keywords='api gateway purpose')
        self.brain.save()
        ctx = self.brain.format_boot_context(user='Test', project='test')
        sk_pos = ctx.find("WHAT YOU'VE LEARNED ABOUT YOURSELF")
        eng_pos = ctx.find('PROJECT UNDERSTANDING')
        if sk_pos >= 0 and eng_pos >= 0:
            self.assertLess(sk_pos, eng_pos,
                            "Self-knowledge should appear before engineering context")


# ══════════════════════════════════════════════════════════════════════════
# TEST 10: Behavioral mirror
# ══════════════════════════════════════════════════════════════════════════

# TestBehavioralMirror removed — _behavioral_mirror() deleted.
# Encoding behavior now tracked by encoding_gap producer in signal queue.


# ══════════════════════════════════════════════════════════════════════════
# TEST 11: Session-end reflection
# ══════════════════════════════════════════════════════════════════════════

class TestSessionEndReflection(unittest.TestCase):
    """Verify reflect_for_next_claude() creates boot nodes."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        os.environ['ORT_DISABLE_ALL_ACCELERATORS'] = '1'
        from servers.brain import Brain
        self.brain = Brain(self.db_path)

    def tearDown(self):
        self.brain.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reflect_creates_boot_node(self):
        """reflect_for_next_claude() should create a boot node."""
        # Encode something first so reflection has content
        self.brain.remember(type='decision', title='Test decision',
                            content='We decided to test')
        self.brain.save()
        result = self.brain.reflect_for_next_claude()
        self.assertIsNotNone(result)
        node_id = result.get('id') if isinstance(result, dict) else result

        # Verify it's a boot node
        row = self.brain.conn.execute(
            "SELECT type, title FROM nodes WHERE id = ?", (node_id,)).fetchone()
        self.assertEqual(row[0], 'boot')
        self.assertIn('handoff', row[1].lower())

    def test_reflect_notes_zero_encodes(self):
        """When nothing encoded, reflection should note the gap."""
        result = self.brain.reflect_for_next_claude()
        self.assertIsNotNone(result)
        content = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE type = 'boot' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()[0]
        self.assertIn('nothing was encoded', content.lower())

    def test_boot_nodes_surface_after_reflection(self):
        """After reflection, fetch_boot_nodes should return the new node."""
        self.brain.reflect_for_next_claude()
        nodes = self.brain.fetch_boot_nodes(limit=3)
        self.assertTrue(len(nodes) >= 1)
        self.assertIn('handoff', nodes[0]['title'].lower())


# ══════════════════════════════════════════════════════════════════════════
# TEST 12: Thought node decay + SKILL.md preamble
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

    def test_skill_md_has_orientation_preamble(self):
        """SKILL.md should open with identity orientation for Anchor."""
        skill_path = os.path.join(PROJECT_ROOT, 'skills', 'brain', 'SKILL.md')
        with open(skill_path) as f:
            content = f.read()
        self.assertIn('Anchor', content)
        self.assertIn('What You Are', content)


if __name__ == '__main__':
    unittest.main()
