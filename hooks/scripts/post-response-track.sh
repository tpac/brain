#!/bin/bash
# brain v4 — Stop hook (post-response tracking)
# Hook: Stop — fires when Claude finishes responding (before waiting for user).
# Tracks consciousness engagement and increments encoding gap counter.
#
# Input: JSON on stdin with { session_id, last_assistant_message, stop_hook_active }
# Output: none (tracking only, no context injection needed)

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SERVER_DIR="$PLUGIN_ROOT/servers"

# ── Resolve brain DB ──
DB_DIR=""
if [ -n "$BRAIN_DB_DIR" ] && [ -d "$BRAIN_DB_DIR" ]; then
  DB_DIR="$BRAIN_DB_DIR"
fi
if [ -z "$DB_DIR" ]; then
  for candidate in /sessions/*/mnt/AgentsContext/brain; do
    if [ -f "$candidate/brain.db" ]; then
      DB_DIR="$candidate"
      break
    fi
  done
fi
if [ -z "$DB_DIR" ] && [ -f "$HOME/AgentsContext/brain/brain.db" ]; then
  DB_DIR="$HOME/AgentsContext/brain"
fi
if [ -z "$DB_DIR" ] || [ ! -f "$DB_DIR/brain.db" ]; then
  exit 0
fi

export BRAIN_DB_DIR="$DB_DIR"
export BRAIN_SERVER_DIR="$SERVER_DIR"
export HOOK_INPUT=$(cat)

python3 -c '
import sys, os, json

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

try:
    hook_input = json.loads(os.environ.get("HOOK_INPUT", "{}"))
except Exception:
    sys.exit(0)

# Prevent infinite loops
if hook_input.get("stop_hook_active"):
    sys.exit(0)

try:
    from servers.brain import Brain
    brain = Brain(db_path)
except Exception:
    sys.exit(0)

try:
    last_message = hook_input.get("last_assistant_message", "")

    # Detect consciousness engagement: did Claude reference brain signals?
    engagement_keywords = [
        "brain recall", "brain dream", "evolution", "hypothesis",
        "aspiration", "tension", "fading", "consciousness",
        "brain.remember", "brain.recall", "brain.confirm_evolution",
        "brain.dismiss_evolution", "brain.create_"
    ]

    message_lower = last_message.lower() if last_message else ""
    engaged_signals = []
    for kw in engagement_keywords:
        if kw.lower() in message_lower:
            engaged_signals.append(kw)

    # Log engagement for consciousness adaptation
    if engaged_signals:
        for signal in engaged_signals[:3]:
            try:
                brain.log_consciousness_response(signal_type=signal, responded=True)
            except Exception:
                pass

    # Increment turn counter for encoding gap detection
    try:
        turns = brain.get_config("session_turns", 0)
        brain.set_config("session_turns", int(turns) + 1)
    except Exception:
        pass

    # Check if brain.remember() was called this turn — acknowledge what was stored
    try:
        session_start = brain.get_config("session_start_at")
        if session_start and "brain.remember" in message_lower:
            cursor = brain.conn.execute(
                """SELECT type, title, locked FROM nodes
                   WHERE created_at >= ? AND archived = 0
                     AND type NOT IN ('context', 'thought', 'intuition')
                   ORDER BY created_at DESC LIMIT 3""",
                (session_start,)
            )
            just_stored = cursor.fetchall()
            if just_stored:
                lines = ["BRAIN ACKNOWLEDGED — just stored:"]
                for ntype, title, locked in just_stored:
                    lock_icon = " LOCKED" if locked else ""
                    lines.append("  [%s]%s %s" % (ntype, lock_icon, title[:70]))
                lines.append("  Correct me if anything is wrong.")
                print(json.dumps({"systemMessage": "\\n".join(lines)}))
    except Exception:
        pass

    brain.save()
    brain.close()

except Exception:
    try:
        brain.close()
    except Exception:
        pass
' 2>/dev/null

exit 0
