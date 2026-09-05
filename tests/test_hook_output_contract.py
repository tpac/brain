"""Contract: hook stdout is host-portable (Claude Code and Codex) and never gates.

Both hosts parse a hook's stdout against strict per-event JSON schemas. Codex
marks a run FAILED for `{"decision":"approve"}` on UserPromptSubmit / PreToolUse
and for plain text on Stop; on Claude Code that approve meant "allow, skip the
permission prompt" — a decision a memory plugin must not make. So "nothing to
say" is silence, a block is honored only on Stop (self-message delivery),
`hook_common.emit_hook_output` is the single writer, and this locks (1) its
event-by-event mapping, (2) that it stays loud on shapes it does not recognise,
and (3) that no hook entrypoint — Python OR shell shim — writes a decision shape
around it. Pure unit + file inspection — no brain, no daemon.
"""
import contextlib
import glob
import io
import json
import os
import re
import sys
import unittest
from unittest import mock

_HOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "hooks", "scripts")
sys.path.insert(0, _HOOKS_DIR)
import hook_common  # noqa: E402

# Matches the approve decision in any spacing/quoting a script could print it:
# `{"decision":"approve"}`, `"decision": "approve"`, `'decision': 'approve'`.
_APPROVE_SHAPE = re.compile(r"""decision["']?\s*:\s*["']?approve""")


def _emit(event, payload):
    """Run the emitter with its sinks stubbed; return (stdout, logged_warnings)."""
    buf = io.StringIO()
    logged = []
    with mock.patch.object(hook_common, "brain_debug", lambda *_a, **_k: None), \
            mock.patch.object(hook_common, "log_hook_error",
                              lambda *a, **k: logged.append((a, k))), \
            contextlib.redirect_stdout(buf):
        hook_common.emit_hook_output(event, payload)
    return buf.getvalue(), logged


def _out(event, payload):
    return _emit(event, payload)[0]


def _context(out):
    return json.loads(out)["hookSpecificOutput"]["additionalContext"]


class TestEmitMapping(unittest.TestCase):
    def test_approve_is_silence(self):
        for event in ("UserPromptSubmit", "PreToolUse", "Stop", "SessionStart"):
            self.assertEqual(_out(event, {"decision": "approve"}), "", event)
            self.assertEqual(_out(event, {}), "", event)
            self.assertEqual(_out(event, None), "", event)

    def test_approve_with_reason_becomes_context(self):
        out = json.loads(_out("PreToolUse", {"decision": "approve", "reason": "mind the rule"}))
        self.assertEqual(out, {"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "additionalContext": "mind the rule"}})

    def test_additional_context_payload(self):
        out = json.loads(_out("UserPromptSubmit", {"additionalContext": "recalled"}))
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit")
        self.assertEqual(out["hookSpecificOutput"]["additionalContext"], "recalled")
        self.assertNotIn("decision", out)

    def test_stop_block_is_the_delivery_form(self):
        out = json.loads(_out("Stop", {"decision": "block", "reason": "wait"}))
        self.assertEqual(out, {"decision": "block", "reason": "wait"})

    def test_stop_block_without_reason_still_carries_one(self):
        # Both hosts reject a block with an empty reason — the delivery must not
        # silently degrade into a no-op.
        out = json.loads(_out("Stop", {"decision": "block"}))
        self.assertEqual(out["decision"], "block")
        self.assertTrue(out["reason"].strip())

    def test_block_outside_stop_is_downgraded_and_logged(self):
        # The brain informs, it never gates: a block on a tool or prompt event
        # is a daemon-side mistake to surface, and its text still reaches the
        # model as context.
        for event in ("PreToolUse", "UserPromptSubmit"):
            out, logged = _emit(event, {"decision": "block", "reason": "careful"})
            self.assertEqual(len(logged), 1, event)
            self.assertIn("never gates", logged[0][0][1])
            self.assertEqual(_context(out), "careful", event)
            self.assertNotIn("permissionDecision", out)
            self.assertNotIn('"decision"', out)

    def test_stop_drops_context_loudly(self):
        # Codex's Stop schema has no additionalContext channel; the two-host
        # contract drops it — and says so in hook_errors.
        for payload in ({"decision": "approve", "reason": "(stored)"}, {"additionalContext": "x"}):
            out, logged = _emit("Stop", payload)
            self.assertEqual(out, "")
            self.assertEqual(len(logged), 1, payload)
            self.assertEqual(logged[0][1].get("level"), "warning")

    def test_unknown_decision_is_logged_and_informational(self):
        for bad in ("deny", "block ", "ALLOW"):
            out, logged = _emit("PreToolUse", {"decision": bad, "reason": "careful"})
            self.assertEqual(len(logged), 1, bad)
            self.assertIn("unknown hook decision", logged[0][0][1])
            self.assertEqual(_context(out), "careful", bad)

    def test_non_dict_payload_is_logged_not_swallowed(self):
        for bad in ("not a dict", ["a", "b"], 7):
            out, logged = _emit("UserPromptSubmit", bad)
            self.assertEqual(out, "")
            self.assertEqual(len(logged), 1, bad)
            self.assertIn("non-dict", logged[0][0][1])

    def test_non_string_fields_are_rendered_not_fatal(self):
        out = json.loads(_out("Stop", {"decision": "block", "reason": ["line one", "line two"]}))
        self.assertIn("line one", out["reason"])
        self.assertIn('"k"', _context(_out("PreToolUse", {"additionalContext": {"k": 1}})))

    def test_never_emits_approve_or_deny(self):
        for event in ("UserPromptSubmit", "PreToolUse", "Stop"):
            for payload in ({"decision": "approve"}, {"decision": "approve", "reason": "r"},
                            {"additionalContext": "c"}, {"decision": "block", "reason": "b"},
                            {"decision": "deny", "reason": "d"}):
                out = _out(event, payload)
                self.assertIsNone(_APPROVE_SHAPE.search(out), (event, payload))
                self.assertNotIn("permissionDecision", out, (event, payload))


class TestToolTargetFile(unittest.TestCase):
    def test_claude_code_shape(self):
        self.assertEqual(hook_common.tool_target_file({"file_path": "a/b.py"}), "a/b.py")

    def test_codex_apply_patch_shape(self):
        patch = "*** Begin Patch\n*** Update File: servers/brain.py\n@@\n-a\n+b\n*** End Patch"
        self.assertEqual(hook_common.tool_target_file({"command": patch}), "servers/brain.py")
        self.assertEqual(hook_common.tool_target_file({"command": "*** Add File: new.txt\n+x"}), "new.txt")

    def test_no_target(self):
        self.assertEqual(hook_common.tool_target_file({"command": "ls -la"}), "")
        self.assertEqual(hook_common.tool_target_file("nope"), "")


class TestScriptsRouteThroughEmitter(unittest.TestCase):
    # Every hook entrypoint — the bash shims Claude Code / Codex actually run
    # AND the Python bodies they exec — is in scope. Only the shared module that
    # implements the emitter may spell the shapes.
    _SHARED = {"hook_common.py"}

    @classmethod
    def setUpClass(cls):
        paths = glob.glob(os.path.join(_HOOKS_DIR, "*.py")) + glob.glob(os.path.join(_HOOKS_DIR, "*.sh"))
        cls.sources = {}
        for p in sorted(paths):
            name = os.path.basename(p)
            if name in cls._SHARED:
                continue
            with open(p) as f:
                cls.sources[name] = f.read()

    def test_scan_sees_both_shims_and_bodies(self):
        # Guard against a glob typo making the scan vacuous.
        self.assertIn("pre-bash-safety.sh", self.sources)
        self.assertIn("pre_bash_safety.py", self.sources)

    def test_no_script_prints_approve(self):
        for name, src in self.sources.items():
            self.assertIsNone(_APPROVE_SHAPE.search(src),
                              "%s emits an approve decision — exit 0 with no output instead" % name)
            self.assertNotIn("APPROVE", src, "%s keeps an APPROVE constant" % name)

    def test_no_script_hand_rolls_hook_json(self):
        # The hookSpecificOutput envelope is composed in exactly one place. The
        # needle is the quoted JSON key, so a docstring may still NAME the field.
        for name, src in self.sources.items():
            self.assertNotIn('"hookSpecificOutput"', src,
                             "%s builds hook JSON itself — route it through emit_hook_output" % name)


if __name__ == "__main__":
    unittest.main()
