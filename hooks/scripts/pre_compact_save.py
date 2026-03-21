"""PreCompact — synthesize session + compaction boundary + save.
Must always output {"decision":"approve"} — never block compaction.
Thin client: sends hook_pre_compact_save to daemon, falls back to direct Python.
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import daemon_available, daemon_call_raw, get_brain, close_brain

APPROVE = json.dumps({"decision": "approve"})

try:
    if daemon_available():
        resp = daemon_call_raw("hook_pre_compact_save", {}, timeout=8.0)
        # Always approve regardless of result
    else:
        # Direct fallback
        parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from servers.daemon_hooks import hook_pre_compact_save
        brain = get_brain()
        if brain:
            try:
                hook_pre_compact_save(brain, {}, [])
            except Exception:
                pass
            finally:
                close_brain(brain)
except Exception:
    pass

print(APPROVE)
