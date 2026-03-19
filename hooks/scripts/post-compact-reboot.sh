#!/bin/bash
# brain v4 — PostCompact hook
# Hook: PostCompact — fires after context compaction completes.
# Re-injects critical brain context into the fresh context window.
# Without this, brain goes dark after compaction until next SessionStart.
#
# Input: JSON on stdin with { session_id, trigger, compact_summary }
# Output: stdout with locked rules + consciousness signals (injected into context)

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

python3 -c '
import sys, os, json

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

try:
    from servers.brain import Brain
    brain = Brain(db_path)
except Exception:
    sys.exit(0)

output_lines = ["BRAIN POST-COMPACTION REBOOT (context was compacted, re-injecting critical state):", ""]

try:
    # Re-run lightweight boot: locked rules + consciousness
    boot = brain.context_boot()

    # Locked rules (these MUST survive compaction)
    locked_rules = boot.get("locked_rules", [])
    if locked_rules:
        output_lines.append(f"LOCKED RULES ({len(locked_rules)} active):")
        for rule in locked_rules[:15]:
            title = rule.get("title", "")[:80]
            output_lines.append(f"  🔒 {title}")
        output_lines.append("")

    # Active consciousness signals (abbreviated)
    signals = brain.get_consciousness_signals()

    reminders = signals.get("reminders", [])
    if reminders:
        output_lines.append("ACTIVE REMINDERS:")
        for r in reminders[:5]:
            title = r.get("title", "")[:80]
            output_lines.append(f"  🔔 {title}")
        output_lines.append("")

    evolutions = signals.get("evolutions", [])
    if evolutions:
        output_lines.append("ACTIVE EVOLUTIONS:")
        for e in evolutions[:5]:
            title = e.get("title", "")[:80]
            etype = e.get("type", "")
            output_lines.append(f"  {title} [{etype}]")
        output_lines.append("")

    encoding_gap = signals.get("encoding_gap")
    if encoding_gap:
        output_lines.append(f"⚠️ ENCODING GAP: {encoding_gap}")
        output_lines.append("")

    output_lines.append("Brain is live. Use brain.recall_with_embeddings() and brain.remember() as normal.")

    brain.close()
    print("\n".join(output_lines))

except Exception as e:
    try:
        brain.close()
    except Exception:
        pass
' 2>/dev/null

exit 0
