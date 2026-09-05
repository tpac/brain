"""Contract: the two hook manifests are one set of scripts, projected per host.

hooks/hooks.json is the Claude Code manifest; hooks/hooks.codex.json is the
Codex projection of it — same commands, only the event set and per-host handler
fields differ. Two files that share seven identical handlers drift silently, so
this locks BOTH directions: every Codex handler is backed by a CC handler (same
command, a subset of its tool names), and every CC handler on a shared event is
projected into the Codex file (minus the tool names that only exist on Claude
Code). Plus the two Codex-specific constraints — SessionEnd within Codex's 3 s
cap, and the injecting hooks lifting Codex's ~2,500-token spill limit — and
timeout parity everywhere else. Pure file inspection.
"""
import json
import os
import unittest

_HOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "hooks")

# Events the Codex hook engine dispatches (learn.chatgpt.com/docs/hooks).
CODEX_EVENTS = {
    "SessionStart", "SessionEnd", "UserPromptSubmit", "PreToolUse", "PostToolUse",
    "PermissionRequest", "PreCompact", "PostCompact", "SubagentStart", "SubagentStop",
    "Stop", "Interrupt",
}
# Claude Code events with no Codex counterpart — must not appear in the projection.
CC_ONLY_EVENTS = {"WorktreeCreate", "WorktreeRemove", "ConfigChange", "StopFailure"}
# Claude Code local-tool names that never fire under Codex: its file tool is
# apply_patch (matched via the Edit|Write aliases), sub-agents are spawn_agent
# (matched via the Agent alias), and search/fetch are hosted tools that skip the
# hook path entirely.
CC_ONLY_TOOL_NAMES = ("Read", "Glob", "Grep", "WebSearch", "WebFetch", "NotebookEdit")

# Codex caps SessionEnd (and Interrupt) handlers at 3 seconds.
CODEX_SESSION_END_MAX_TIMEOUT = 3
# Events whose additionalContext Codex spills to disk above ~2,500 tokens unless
# the handler sets additionalContextLimit (0 = pass everything through).
INJECTING_EVENTS = ("SessionStart", "UserPromptSubmit")


def _load(name):
    with open(os.path.join(_HOOKS_DIR, name)) as f:
        doc = json.load(f)
    assert isinstance(doc.get("hooks"), dict), "%s must have the {\"hooks\": {...}} wrapper" % name
    return doc["hooks"]


def _handlers(events, event):
    """(matcher, command, timeout) triples registered for `event`."""
    return {
        (group.get("matcher", "") or "", handler["command"], handler.get("timeout"))
        for group in events.get(event, [])
        for handler in group["hooks"]
    }


def _alternatives(matcher):
    """The tool names a matcher selects; None means every tool ("" or "*")."""
    if matcher in ("", "*"):
        return None
    return set(matcher.split("|"))


def _covers(wide, narrow):
    """True when matcher `wide` selects every tool matcher `narrow` selects."""
    w, n = _alternatives(wide), _alternatives(narrow)
    if w is None:
        return True
    if n is None:
        return False
    return n <= w


def _projected(cc_matcher):
    """What a CC matcher should look like on Codex: the same tools minus the
    Claude-Code-only names. None = universal; empty set = nothing to project."""
    alts = _alternatives(cc_matcher)
    if alts is None:
        return None
    return alts - set(CC_ONLY_TOOL_NAMES)


class TestHooksManifestSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cc = _load("hooks.json")
        cls.cx = _load("hooks.codex.json")
        cls.shared_events = set(cls.cc) - CC_ONLY_EVENTS

    def test_codex_events_are_supported(self):
        unknown = set(self.cx) - CODEX_EVENTS
        self.assertFalse(unknown, "hooks.codex.json registers events Codex has no engine for: %s" % sorted(unknown))

    def test_cc_only_events_absent(self):
        leaked = set(self.cx) & CC_ONLY_EVENTS
        self.assertFalse(leaked, "Claude-Code-only events in the Codex projection: %s" % sorted(leaked))

    def test_every_shared_event_is_projected(self):
        missing = self.shared_events - set(self.cx)
        self.assertFalse(missing, "events in hooks.json with no Codex projection: %s" % sorted(missing))

    def test_codex_handlers_are_backed_by_cc_handlers(self):
        # Codex ⊆ CC: a Codex handler must exist in hooks.json with the same
        # command and a matcher that selects at least the same tools.
        for event in self.cx:
            cc = _handlers(self.cc, event)
            for matcher, command, _timeout in _handlers(self.cx, event):
                self.assertTrue(
                    any(c == command and _covers(m, matcher) for m, c, _t in cc),
                    "%s handler %r under matcher %r has no backing handler in hooks.json — "
                    "the Codex manifest is a projection (same command, a subset of the tool names)"
                    % (event, command, matcher))

    def test_cc_handlers_are_projected_to_codex(self):
        # CC ⊆ Codex (minus CC-only tools): every handler hooks.json registers on
        # a shared event must reach the Codex file, or the Codex host silently
        # loses it the day someone adds a handler on the Claude Code side.
        for event in self.shared_events:
            cx = _handlers(self.cx, event)
            for matcher, command, _timeout in _handlers(self.cc, event):
                wanted = _projected(matcher)
                if wanted is not None and not wanted:
                    continue  # only Claude-Code-only tools: nothing to project
                narrow = "" if wanted is None else "|".join(sorted(wanted))
                self.assertTrue(
                    any(c == command and _covers(m, narrow) for m, c, _t in cx),
                    "%s handler %r (matcher %r) is not projected into hooks.codex.json — "
                    "add it there (Codex tools: %s)" % (event, command, matcher, narrow or "all"))

    def test_no_cc_only_tool_names_in_codex_matchers(self):
        for event in ("PreToolUse", "PostToolUse"):
            for matcher, _c, _t in _handlers(self.cx, event):
                for name in CC_ONLY_TOOL_NAMES:
                    self.assertNotIn(
                        name, matcher.split("|"),
                        "%s matcher %r names %s, which never fires under Codex" % (event, matcher, name))

    def test_timeouts_match_except_session_end(self):
        # The handler tuning is one value per script; only SessionEnd differs,
        # because Codex caps it (checked below).
        for event in self.cx:
            if event == "SessionEnd":
                continue
            cc = {c: t for _m, c, t in _handlers(self.cc, event)}
            for _matcher, command, timeout in _handlers(self.cx, event):
                self.assertEqual(timeout, cc.get(command),
                                 "%s handler %r: Codex timeout %r != Claude Code timeout %r"
                                 % (event, command, timeout, cc.get(command)))

    def test_session_end_within_codex_cap(self):
        for _m, _c, timeout in _handlers(self.cx, "SessionEnd"):
            self.assertLessEqual(
                timeout if timeout is not None else 600, CODEX_SESSION_END_MAX_TIMEOUT,
                "Codex kills SessionEnd handlers after %d s" % CODEX_SESSION_END_MAX_TIMEOUT)

    def test_injecting_hooks_lift_spill_limit(self):
        for event in INJECTING_EVENTS:
            for group in self.cx.get(event, []):
                for handler in group["hooks"]:
                    self.assertIn(
                        "additionalContextLimit", handler,
                        "%s handler must set additionalContextLimit — boot and recall "
                        "injections sit at Codex's default spill threshold" % event)


if __name__ == "__main__":
    unittest.main()
