"""
Contract Sync Tests — ensures all brain layers stay in sync.

The brain has 6 layers that must agree on method names, parameters, and defaults:
  1. Brain methods (servers/brain_*.py) — the source of truth
  2. Daemon dispatch (servers/daemon_dispatch.py) — command routing
  3. MCP server (servers/brain_mcp.py) — tool definitions for Claude
  4. SKILL.md (skills/brain/SKILL.md) — API reference Claude reads
  5. Eval fake tools (eval/skill_eval.py) — test harness
  6. Hook scripts (hooks/scripts/*.py) — automated triggers

When ANY of these change, this test catches drift.

Run: python3 -m pytest tests/test_contract_sync.py -v
"""

import inspect
import json
import os
import re
import sys
import unittest
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "servers"))


class TestBrainMethodsExist(unittest.TestCase):
    """Layer 1: Brain methods are the source of truth. Verify they exist and have correct signatures."""

    @classmethod
    def setUpClass(cls):
        """Introspect all brain modules for public encoding/connection methods."""
        from servers import brain_engineering, brain_vocabulary, brain_connections, brain_remember, brain_evolution

        cls.methods = {}
        for mod_name, mod in [
            ('brain_engineering', brain_engineering),
            ('brain_vocabulary', brain_vocabulary),
            ('brain_connections', brain_connections),
            ('brain_remember', brain_remember),
            ('brain_evolution', brain_evolution),
        ]:
            for name in dir(mod):
                cls_obj = getattr(mod, name)
                if not inspect.isclass(cls_obj):
                    continue
                for method_name, method in inspect.getmembers(cls_obj, predicate=inspect.isfunction):
                    if method_name.startswith('_'):
                        continue
                    # Only track encoding/connection/recall methods
                    if any(method_name.startswith(p) for p in
                           ('remember', 'create_', 'connect', 'record_', 'learn_', 'recall')):
                        sig = inspect.signature(method)
                        params = [k for k in sig.parameters.keys() if k != 'self']
                        defaults = {
                            k: v.default for k, v in sig.parameters.items()
                            if k != 'self' and v.default is not inspect.Parameter.empty
                        }
                        cls.methods[method_name] = {
                            'module': mod_name,
                            'params': params,
                            'defaults': defaults,
                        }

    def test_core_methods_exist(self):
        """Critical encoding methods must exist in brain."""
        required = [
            'remember', 'remember_lesson', 'remember_impact', 'remember_mechanism',
            'remember_uncertainty', 'remember_convention', 'remember_purpose',
            'remember_mental_model', 'remember_constraint',
            'connect', 'learn_vocabulary', 'record_divergence',
        ]
        for method in required:
            self.assertIn(method, self.methods,
                          f"Brain missing core method: {method}")

    def test_remember_has_required_params(self):
        """remember() must accept type, title, content, keywords, locked."""
        params = self.methods['remember']['params']
        for p in ['type', 'title', 'content', 'keywords', 'locked']:
            self.assertIn(p, params, f"remember() missing param: {p}")

    def test_remember_lesson_has_required_params(self):
        """remember_lesson() must have the full lesson structure."""
        params = self.methods['remember_lesson']['params']
        for p in ['title', 'what_happened', 'root_cause', 'fix', 'preventive_principle']:
            self.assertIn(p, params, f"remember_lesson() missing param: {p}")

    def test_connect_has_required_params(self):
        """connect() must accept source_id, target_id, relation, weight."""
        params = self.methods['connect']['params']
        for p in ['source_id', 'target_id', 'relation', 'weight']:
            self.assertIn(p, params, f"connect() missing param: {p}")


class TestDaemonDispatchSync(unittest.TestCase):
    """Layer 2: Every core brain method must have a daemon dispatch command."""

    @classmethod
    def setUpClass(cls):
        from servers.daemon_dispatch import COMMAND_TABLE
        cls.commands = set(COMMAND_TABLE.keys())

    def test_core_encoding_commands_exist(self):
        """All remember_* methods must be in daemon dispatch — not just accessible via eval."""
        required_commands = [
            'remember', 'remember_lesson', 'remember_impact', 'remember_mechanism',
            'remember_uncertainty', 'remember_convention', 'record_divergence',
            'learn_vocabulary',
        ]
        missing = [cmd for cmd in required_commands if cmd not in self.commands]
        self.assertEqual(missing, [],
                         f"Daemon dispatch missing commands (forces eval workaround): {missing}")

    def test_connection_commands_exist(self):
        """connect must be in daemon dispatch."""
        self.assertIn('connect', self.commands)

    def test_core_read_commands_exist(self):
        """Essential read commands must exist."""
        for cmd in ['recall', 'ping', 'consciousness', 'context_boot',
                     'health_check', 'save', 'eval']:
            self.assertIn(cmd, self.commands, f"Missing core command: {cmd}")


class TestMCPToolSync(unittest.TestCase):
    """Layer 3: MCP tools must match daemon commands and brain signatures."""

    @classmethod
    def setUpClass(cls):
        """Parse MCP tool definitions from brain_mcp.py."""
        mcp_path = ROOT / "servers" / "brain_mcp.py"
        content = mcp_path.read_text()

        # Extract tool names from the TOOLS array
        cls.mcp_tools = set(re.findall(r'"name":\s*"(\w+)"', content))

        # Extract tool parameter names per tool (rough parse)
        cls.mcp_params = {}
        # Find each tool block and extract properties
        tool_blocks = re.split(r'\{\s*"name":', content)
        for block in tool_blocks[1:]:  # skip pre-first
            name_match = re.match(r'\s*"(\w+)"', block)
            if name_match:
                name = name_match.group(1)
                props = re.findall(r'"(\w+)":\s*\{[^}]*"type"', block)
                # Filter to actual params (not schema keys)
                cls.mcp_params[name] = [p for p in props if p not in (
                    'type', 'properties', 'required', 'inputSchema', 'input_schema')]

    def test_mcp_tools_have_daemon_commands(self):
        """Every MCP tool must have a matching daemon command."""
        from servers.daemon_dispatch import COMMAND_TABLE
        daemon_cmds = set(COMMAND_TABLE.keys())

        for tool in self.mcp_tools:
            self.assertIn(tool, daemon_cmds,
                          f"MCP tool '{tool}' has no daemon command — will fail at runtime")

    def test_core_encoding_tools_exposed(self):
        """Core encoding tools should be MCP-native, not eval-only.
        Note: ping, save, health_check, set_config, get_config removed in Session #14 —
        daemon self-manages. Specialized remember_* promoted to first-class tools."""
        should_be_mcp = [
            'remember', 'connect', 'recall', 'consciousness', 'eval',
            'remember_lesson', 'remember_impact', 'remember_mechanism',
            'remember_convention', 'remember_uncertainty', 'remember_mental_model',
            'record_divergence', 'learn_vocabulary',
        ]
        for tool in should_be_mcp:
            self.assertIn(tool, self.mcp_tools,
                          f"Core tool '{tool}' not in MCP — Claude can't use it directly")

    def test_remember_tool_params_match_brain(self):
        """MCP remember tool params must be a subset of brain.remember() params."""
        if 'remember' not in self.mcp_params:
            self.skipTest("remember not found in MCP params")

        from servers import brain_remember
        for name in dir(brain_remember):
            cls = getattr(brain_remember, name)
            if inspect.isclass(cls):
                for mname, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                    if mname == 'remember':
                        sig = inspect.signature(method)
                        brain_params = {k for k in sig.parameters.keys() if k != 'self'}
                        mcp_params = set(self.mcp_params.get('remember', []))
                        invalid = mcp_params - brain_params
                        self.assertEqual(invalid, set(),
                                         f"MCP remember has params not in brain: {invalid}")

    def test_connect_uses_ids_not_titles(self):
        """MCP connect tool must use source_id/target_id, not source_title/target_title."""
        if 'connect' in self.mcp_params:
            params = self.mcp_params['connect']
            self.assertNotIn('source_title', params,
                             "MCP connect uses source_title but brain needs source_id")
            self.assertNotIn('target_title', params,
                             "MCP connect uses target_title but brain needs target_id")


class TestEvalFakeToolSync(unittest.TestCase):
    """Layer 6: Eval fake tools must match real MCP tools."""

    @classmethod
    def setUpClass(cls):
        """Load eval fake tools."""
        eval_path = ROOT / "eval" / "skill_eval.py"
        if not eval_path.exists():
            cls.fake_tools = {}
            return

        # Import FAKE_BRAIN_TOOLS
        import importlib.util
        spec = importlib.util.spec_from_file_location("skill_eval", eval_path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            cls.fake_tools = {t['name']: t for t in mod.FAKE_BRAIN_TOOLS}
        except Exception:
            cls.fake_tools = {}

    def test_eval_connect_uses_correct_params(self):
        """Eval brain_connect must use source_id/target_id, not source_title/target_title."""
        if 'brain_connect' not in self.fake_tools:
            self.skipTest("brain_connect not in eval fake tools")
        props = self.fake_tools['brain_connect']['input_schema'].get('properties', {})
        self.assertNotIn('source_title', props,
                         "Eval brain_connect uses source_title but real API needs source_id — eval results invalid")
        self.assertNotIn('target_title', props,
                         "Eval brain_connect uses target_title but real API needs target_id — eval results invalid")

    def test_eval_vocabulary_maps_to_type(self):
        """Eval brain_learn_vocabulary maps_to should note it can be a list."""
        if 'brain_learn_vocabulary' not in self.fake_tools:
            self.skipTest("brain_learn_vocabulary not in eval")
        # This is informational — brain handles string→list conversion
        # but eval should ideally document the correct type

    def test_eval_tools_match_brain_methods(self):
        """Every eval fake tool should correspond to a real brain method."""
        from servers import brain_engineering, brain_vocabulary, brain_connections, brain_remember

        # Map eval tool names to brain method names
        tool_to_method = {
            'brain_remember': 'remember',
            'brain_remember_impact': 'remember_impact',
            'brain_remember_mechanism': 'remember_mechanism',
            'brain_remember_uncertainty': 'remember_uncertainty',
            'brain_remember_lesson': 'remember_lesson',
            'brain_connect': 'connect',
            'brain_record_divergence': 'record_divergence',
            'brain_learn_vocabulary': 'learn_vocabulary',
            'brain_remember_convention': 'remember_convention',
        }

        for tool_name, method_name in tool_to_method.items():
            if tool_name in self.fake_tools:
                # Verify the method exists somewhere in brain
                found = False
                for mod in [brain_engineering, brain_vocabulary, brain_connections, brain_remember]:
                    for name in dir(mod):
                        cls = getattr(mod, name)
                        if inspect.isclass(cls) and hasattr(cls, method_name):
                            found = True
                            break
                self.assertTrue(found,
                                f"Eval tool '{tool_name}' maps to '{method_name}' but method not found in brain")


class TestSKILLMDSync(unittest.TestCase):
    """Layer 4: SKILL.md API reference must document real methods."""

    @classmethod
    def setUpClass(cls):
        skill_path = ROOT / "skills" / "brain" / "SKILL.md"
        cls.content = skill_path.read_text() if skill_path.exists() else ""

    def test_core_methods_documented(self):
        """Core encoding methods must appear in SKILL.md."""
        required = [
            'remember', 'remember_lesson', 'remember_impact', 'remember_mechanism',
            'remember_uncertainty', 'remember_convention', 'connect',
            'record_divergence', 'learn_vocabulary',
        ]
        for method in required:
            self.assertIn(method, self.content,
                          f"SKILL.md doesn't mention '{method}' — Claude won't know it exists")

    def test_no_phantom_methods(self):
        """SKILL.md shouldn't document methods that don't exist."""
        # Extract method-like references from SKILL.md
        method_refs = set(re.findall(r'brain\.(\w+)\(', self.content))

        from servers import brain_engineering, brain_vocabulary, brain_connections, brain_remember, brain_evolution

        all_methods = set()
        for mod in [brain_engineering, brain_vocabulary, brain_connections, brain_remember, brain_evolution]:
            for name in dir(mod):
                cls = getattr(mod, name)
                if inspect.isclass(cls):
                    for mname, _ in inspect.getmembers(cls, predicate=inspect.isfunction):
                        if not mname.startswith('_'):
                            all_methods.add(mname)

        # Also include non-class methods and common brain methods
        all_methods.update(['recall', 'save', 'health_check',
                            'format_boot_context', 'set_config', 'get_config',
                            'recall', 'get_engineering_context',
                            'synthesize_session',
                            'record_message', 'get_encoding_heartbeat'])

        phantom = method_refs - all_methods
        # Filter out common false positives
        phantom = {p for p in phantom if not p.startswith('_') and p not in ('method', 'function')}
        self.assertEqual(phantom, set(),
                         f"SKILL.md references methods that don't exist: {phantom}")


class TestDefaultsSync(unittest.TestCase):
    """Cross-layer: parameter defaults must agree."""

    def test_confidence_default_consistent(self):
        """remember() confidence default must match across brain, daemon, and MCP."""
        # Brain default
        from servers import brain_remember
        brain_default = None
        for name in dir(brain_remember):
            cls = getattr(brain_remember, name)
            if inspect.isclass(cls):
                for mname, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                    if mname == 'remember':
                        sig = inspect.signature(method)
                        if 'confidence' in sig.parameters:
                            brain_default = sig.parameters['confidence'].default

        # Daemon default
        dispatch_path = ROOT / "servers" / "daemon_dispatch.py"
        dispatch_content = dispatch_path.read_text()
        daemon_match = re.search(r'confidence.*?args\.get\(["\']confidence["\'],?\s*([\d.]+)', dispatch_content)
        daemon_default = float(daemon_match.group(1)) if daemon_match else None

        # MCP default
        mcp_path = ROOT / "servers" / "brain_mcp.py"
        mcp_content = mcp_path.read_text()
        # Find confidence default in MCP
        mcp_match = re.search(r'"confidence".*?"default":\s*([\d.]+)', mcp_content, re.DOTALL)
        mcp_default = float(mcp_match.group(1)) if mcp_match else None

        if brain_default is not None and daemon_default is not None:
            self.assertEqual(brain_default, daemon_default,
                             f"confidence default: brain={brain_default} vs daemon={daemon_default}")
        if brain_default is not None and mcp_default is not None:
            self.assertEqual(brain_default, mcp_default,
                             f"confidence default: brain={brain_default} vs MCP={mcp_default}")


class TestHookCommandSync(unittest.TestCase):
    """Layer 5: Hook scripts must only call commands that exist in daemon."""

    @classmethod
    def setUpClass(cls):
        """Extract daemon command calls from hook scripts."""
        hooks_dir = ROOT / "hooks" / "scripts"
        cls.hook_commands = {}
        if hooks_dir.exists():
            for script in hooks_dir.glob("*.py"):
                content = script.read_text()
                # Find daemon_call_raw("command", ...) patterns
                calls = re.findall(r'daemon_call_raw\(["\'](\w+)["\']', content)
                # Also find send_command("command", ...) patterns
                calls += re.findall(r'send_command\(["\'](\w+)["\']', content)
                if calls:
                    cls.hook_commands[script.name] = calls

    def test_hook_commands_exist_in_daemon(self):
        """Every command a hook calls must exist in daemon dispatch or hook table."""
        from servers.daemon_dispatch import COMMAND_TABLE
        daemon_cmds = set(COMMAND_TABLE.keys())

        # Hook commands are prefixed with hook_ and handled separately
        for script, commands in self.hook_commands.items():
            for cmd in commands:
                valid = cmd in daemon_cmds or cmd.startswith('hook_')
                self.assertTrue(valid,
                                f"Hook {script} calls '{cmd}' which doesn't exist in daemon")


if __name__ == '__main__':
    unittest.main()
