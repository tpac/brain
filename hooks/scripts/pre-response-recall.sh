#!/bin/bash
# tmemory v14 (serverless) — Layer C: Pre-response recall trigger
# Fires before Claude responds to ANY user message (not just file edits).
# This is the key architectural improvement: brain context is available
# during reasoning, not just during file operations.
#
# Input: JSON on stdin with tool_name (should be empty or "UserMessage")
# Output: JSON with decision + reason containing recalled context

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SERVER_DIR="$PLUGIN_ROOT/servers"

# ── Resolve brain DB ──
DB_DIR=""
if [ -n "$TMEMORY_DB_DIR" ] && [ -d "$TMEMORY_DB_DIR" ]; then
  DB_DIR="$TMEMORY_DB_DIR"
fi
if [ -z "$DB_DIR" ]; then
  for candidate in /sessions/*/mnt/AgentsContext/tmemory; do
    if [ -f "$candidate/brain.db" ]; then
      DB_DIR="$candidate"
      break
    fi
  done
fi
if [ -z "$DB_DIR" ] && [ -f "$HOME/AgentsContext/tmemory/brain.db" ]; then
  DB_DIR="$HOME/AgentsContext/tmemory"
fi
if [ -z "$DB_DIR" ] || [ ! -f "$DB_DIR/brain.db" ]; then
  echo '{"decision":"approve"}'
  exit 0
fi

export TMEMORY_DB_DIR="$DB_DIR"
export TMEMORY_SERVER_DIR="$SERVER_DIR"
export HOOK_INPUT=$(cat)

exec python3 -c '
import json, sys, os, time

server_dir = os.environ.get("TMEMORY_SERVER_DIR", "")
db_dir = os.environ.get("TMEMORY_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

# Parse hook input — extract user message content
try:
    hook_input = json.loads(os.environ.get("HOOK_INPUT", "{}"))
except Exception:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

# The user message content comes from the hook input
# For Notification hooks, the content is in the message field
user_message = hook_input.get("message", "")
if not user_message:
    # Try extracting from tool_input for compatibility
    user_message = hook_input.get("tool_input", {}).get("content", "")

if not user_message or len(user_message) < 5:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

# Skip very short or system-like messages
if user_message.startswith("/") or user_message.startswith("!"):
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

try:
    from servers.brain import Brain
    brain = Brain(db_path)
except Exception:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

try:
    # Recall with embeddings if available, fall back to TF-IDF
    try:
        result = brain.recall_with_embeddings(
            query=user_message[:500],  # Cap query length
            limit=8
        )
    except Exception:
        result = brain.recall(query=user_message[:500], limit=8)

    results = result.get("results", [])

    if not results:
        brain.close()
        print(json.dumps({"decision": "approve"}))
        sys.exit(0)

    # Format recalled context — keep it concise for pre-response
    lines = ["BRAIN RECALL (auto-surfaced for this conversation):", ""]

    for r in results[:5]:  # Show top 5
        typ = r.get("type", "?")
        title = r.get("title", "")[:60]
        content = r.get("content", "")
        locked = "LOCKED " if r.get("locked") else ""
        score = r.get("effective_activation", 0)

        if len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"  [{typ}] {locked}{title} (score: {score:.2f})")
        lines.append(f"    {content}")
        lines.append("")

    lines.append("Use this context to inform your response. Call /remember for new decisions.")
    context = "\n".join(lines)

    brain.save()
    brain.close()
    print(json.dumps({"decision": "approve", "reason": context}))

except Exception:
    try:
        brain.close()
    except Exception:
        pass
    print(json.dumps({"decision": "approve"}))
'
