"""PostCompact — re-inject brain context after compaction.
PostCompact stdout IS visible. This is the safety net.
Thin client: sends hook_post_compact_reboot to daemon, falls back to direct Python.
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import daemon_available, daemon_call_raw, get_brain, close_brain

def _format_output(result):
    """Extract merged output (both channels already merged via wrap_for_hook)."""
    return result.get("output", "")

try:
    if daemon_available():
        resp = daemon_call_raw("hook_post_compact_reboot", {}, timeout=10.0)
        if resp.get("ok"):
            result = resp.get("result", {})
            formatted = _format_output(result)
            if formatted:
                print(formatted)
            else:
                print("BRAIN POST-COMPACTION REBOOT: daemon returned no output")
        else:
            print("BRAIN POST-COMPACTION REBOOT: daemon error — %s" % resp.get("error", "unknown"))
    else:
        # Direct fallback
        parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from servers.daemon_hooks import hook_post_compact_reboot
        brain = get_brain()
        if brain:
            try:
                result = hook_post_compact_reboot(brain, {}, [])
                formatted = _format_output(result)
                if formatted:
                    print(formatted)
            except Exception as e:
                print("brain: post-compact error: %s" % e, file=sys.stderr)
            finally:
                close_brain(brain)
        else:
            print("brain: post-compact import failed", file=sys.stderr)
except Exception as e:
    print("brain: post-compact fatal: %s" % e, file=sys.stderr)
