"""SessionEnd — session synthesis + consolidation.
Thin client: sends hook_session_end to daemon, falls back to direct Python.

Does NOT shut the daemon down. The daemon is shared across all concurrent
sessions for this user and is launchd-managed (KeepAlive). Ending one session
must not tear it down under the others — doing so killed in-flight recalls in
sibling sessions (closed-DB / empty-reply / connection-reset) and forced the
next recall onto a cold, slow daemon. Lifecycle is owned by launchd + the
4h idle-timeout + the maintenance lock, not by per-session hooks. (Root cause
of the daemon restart-churn diagnosed 2026-06-06; was a stale assumption from
the original single-session daemon consolidation, cf4d140.)
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, run_hook

hook_input = get_hook_input()

def main():
    if daemon_available():
        daemon_call_raw("hook_session_end", {"session_id": hook_input.get("session_id", "")}, timeout=30.0)
    else:
        daemon_unavailable_error("session_end")

run_hook("session_end", main)
