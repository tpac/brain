#!/bin/bash
# brain — PreToolUse hook for Bash
# Catches destructive bash commands BEFORE execution and surfaces safety warnings.
# Direct Python brain module call. No HTTP server needed.
#
# Input: JSON on stdin with tool_name, tool_input.command
# Output: JSON with decision + reason containing brain safety warnings

source "$(dirname "$0")/resolve-brain-db.sh"

if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  echo '{"decision":"approve"}'
  exit 0
fi

# Read hook input from stdin
export HOOK_INPUT=$(cat)

# ── Direct Python path ──
exec python3 -c '
import json, sys, os, re, time

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

# Add servers dir to Python path
if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

# Parse hook input
try:
    hook_input = json.loads(os.environ.get("HOOK_INPUT", "{}"))
except Exception:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

tool_input = hook_input.get("tool_input", {})
command = tool_input.get("command", "")

if not command:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

# ── Fast regex pre-screen ──
# If command does not match any destructive pattern, approve immediately (no Brain init)
DESTRUCTIVE_REGEXES = [
    r"rm\s+(-[rf]+\s+|.*--force)",
    r"git\s+worktree\s+remove",
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-[fd]",
    r"git\s+checkout\s+--\s",
    r"git\s+push\s+.*--force",
    r"DROP\s+TABLE",
    r"DELETE\s+FROM",
    r"TRUNCATE",
    r"\brmdir\b",
    r"xargs\s+rm",
]

is_destructive = False
for pattern in DESTRUCTIVE_REGEXES:
    if re.search(pattern, command, re.IGNORECASE):
        is_destructive = True
        break

if not is_destructive:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

# ── Destructive command detected — initialize Brain for safety check ──
try:
    from servers.brain import Brain
    brain = Brain(db_path)
except Exception:
    # Graceful degradation: approve with generic warning
    print(json.dumps({"decision": "approve", "reason": "\u26a0\ufe0f Destructive command detected. Brain unavailable for safety check — proceed carefully."}))
    sys.exit(0)

try:
    result = brain.safety_check(command)
except Exception:
    brain.close()
    print(json.dumps({"decision": "approve", "reason": "\u26a0\ufe0f Destructive command detected. Safety check failed — proceed carefully."}))
    sys.exit(0)

critical_matches = result.get("critical_matches", [])
warnings = result.get("warnings", [])

if critical_matches:
    # Block — critical safety nodes matched
    lines = ["\u26a0\ufe0f BRAIN SAFETY: This command may affect critical brain-tracked resources:"]
    lines.append("")
    for cm in critical_matches[:5]:
        title = cm.get("title", "")[:80]
        content = cm.get("content", "")
        if len(content) > 200:
            content = content[:200] + "..."
        lines.append("  [%s] %s" % (cm.get("type", "?"), title))
        lines.append("    %s" % content)
        lines.append("")
    lines.append("Review the above before proceeding. This command has been BLOCKED.")
    reason = "\n".join(lines)
    brain.close()
    print(json.dumps({"decision": "block", "reason": reason}))

elif warnings:
    # Approve with warning — safety-related nodes matched
    lines = ["\u26a0\ufe0f BRAIN WARNING: Destructive command detected. Relevant brain context:"]
    lines.append("")
    for w in warnings[:5]:
        title = w.get("title", "")[:80]
        content = w.get("content", "")
        if len(content) > 200:
            content = content[:200] + "..."
        lines.append("  [%s] %s" % (w.get("type", "?"), title))
        lines.append("    %s" % content)
        lines.append("")
    lines.append("Proceed carefully — verify this command is intentional.")
    reason = "\n".join(lines)
    brain.close()
    print(json.dumps({"decision": "approve", "reason": reason}))

else:
    # Destructive but no brain matches
    brain.close()
    print(json.dumps({"decision": "approve", "reason": "\u26a0\ufe0f Destructive command detected. No brain safety rules match, but proceed carefully."}))
'
