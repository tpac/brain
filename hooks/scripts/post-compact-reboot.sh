#!/bin/bash
# brain v4 — PostCompact hook
# Hook: PostCompact — fires after context compaction completes.
# Re-injects critical brain context into the fresh context window.
# Without this, brain goes dark after compaction until next SessionStart.
#
# Input: JSON on stdin with { session_id, trigger, compact_summary }
# Output: stdout with locked rules + consciousness signals (injected into context)

source "$(dirname "$0")/resolve-brain-db.sh"

if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  exit 0
fi

# ── Try daemon first (fast path) ──
source "$(dirname "$0")/daemon-client.sh"
if daemon_available; then
  python3 -c '
import json, sys, socket

sock_path = "'"$BRAIN_SOCKET_PATH"'"

def daemon_call(cmd, args=None):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(15.0)
    sock.connect(sock_path)
    msg = json.dumps({"cmd": cmd, "args": args or {}}) + "\n"
    sock.sendall(msg.encode())
    data = b""
    while b"\n" not in data:
        chunk = sock.recv(65536)
        if not chunk: break
        data += chunk
    sock.close()
    return json.loads(data.decode().strip())

try:
    # Get locked rules via context_boot
    boot_resp = daemon_call("context_boot", {"user": "", "project": "", "task": "post-compaction reboot"})
    consciousness_resp = daemon_call("consciousness")

    output = ["BRAIN POST-COMPACTION REBOOT (context was compacted, re-injecting critical state):", ""]

    if boot_resp.get("ok"):
        locked_nodes = boot_resp.get("result", {}).get("locked", [])
        rules = [n for n in locked_nodes if n.get("type") == "rule"]
        if rules:
            output.append("LOCKED RULES (%d active):" % len(rules))
            for r in rules[:15]:
                output.append("  " + r.get("title", "")[:80])
            output.append("")

    if consciousness_resp.get("ok"):
        signals = consciousness_resp.get("result", {})
        reminders = signals.get("reminders", [])
        if reminders:
            output.append("ACTIVE REMINDERS:")
            for r in reminders[:5]:
                output.append("  " + r.get("title", "")[:80])
            output.append("")

        evolutions = signals.get("evolutions", [])
        if evolutions:
            output.append("ACTIVE EVOLUTIONS:")
            for e in evolutions[:5]:
                output.append("  %s [%s]" % (e.get("title", "")[:80], e.get("type", "")))
            output.append("")

        encoding_gap = signals.get("encoding_gap")
        if encoding_gap:
            output.append("ENCODING GAP: %s" % encoding_gap.get("warning", ""))
            output.append("")

    output.append("Brain is live. Use brain.recall_with_embeddings() and brain.remember() as normal.")
    print("\n".join(output))

except Exception:
    # Fall through handled by exit code
    sys.exit(1)
' 2>/dev/null
  if [ $? -eq 0 ]; then
    exit 0
  fi
  # Fall through to direct Python
fi

# ── Direct Python fallback ──
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
    # Resolve user/project from brain config (same as boot-brain.sh)
    user = brain.get_config("default_user", "User")
    project = brain.get_config("default_project", "default")
    boot = brain.context_boot(user=user, project=project, task="post-compaction reboot")

    # Locked rules (these MUST survive compaction)
    locked_nodes = boot.get("locked", [])
    locked_rules = [n for n in locked_nodes if n.get("type") == "rule"]
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
