"""PreCompact — synthesize session + compaction boundary + save.
Must always output {"decision":"approve"} — never block compaction.
Thin client: sends hook_pre_compact_save to daemon, falls back to direct Python.
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error

hook_input = get_hook_input()
APPROVE = json.dumps({"decision": "approve"})

try:
    if daemon_available():
        resp = daemon_call_raw("hook_pre_compact_save", {"session_id": hook_input.get("session_id", "")}, timeout=8.0)
        # Always approve regardless of result
    else:
        daemon_unavailable_error("pre_compact_save")
except Exception:
    pass

print(APPROVE)
