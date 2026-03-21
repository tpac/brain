#!/bin/bash
# brain v5.1 — PostCompact hook
# Fires after context compaction. This is the SAFETY NET.
#
# Re-injects critical brain context into the fresh context window.
# Also checks if pre-compact synthesis succeeded — if not, runs it now.
# Surfaces the latest synthesis and any transcript rehydration hints.
#
# Input: JSON on stdin with { session_id, trigger, compact_summary }
# Output: stdout with brain state re-injection

source "$(dirname "$0")/resolve-brain-db.sh"

if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  exit 0
fi

# ── Try daemon first (fast path) ──
source "$(dirname "$0")/daemon-client.sh"
if daemon_available; then
  # Daemon path: quick re-injection
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
    boot_resp = daemon_call("context_boot", {"user": "", "project": "", "task": "post-compaction reboot"})
    consciousness_resp = daemon_call("consciousness")

    output = ["BRAIN POST-COMPACTION REBOOT:", ""]

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
        for sig_key, sig_label in [("reminders", "REMINDERS"), ("evolutions", "EVOLUTIONS")]:
            items = signals.get(sig_key, [])
            if items:
                output.append("%s:" % sig_label)
                for item in items[:5]:
                    output.append("  %s" % item.get("title", "")[:80])
                output.append("")

    # v5.3: Recall recent session context — what was Claude working on before compaction?
    # Use last_synthesis + recall to recover working context
    try:
        synth_resp = daemon_call("last_synthesis")
        recall_query = ""
        if synth_resp.get("ok"):
            synth = synth_resp.get("result", {})
            # Build recall query from synthesis
            parts = []
            for key in ("decisions_made", "corrections_received", "open_questions"):
                val = synth.get(key, "")
                if val and val != "[]":
                    parts.append(str(val)[:150])
            if parts:
                recall_query = " ".join(parts)[:500]

        # If no synthesis, try recalling recent config hints
        if not recall_query:
            try:
                task_resp = daemon_call("get_config", {"key": "current_task", "default": ""})
                if task_resp.get("ok") and task_resp.get("result"):
                    recall_query = str(task_resp["result"])[:500]
            except Exception:
                pass

        if recall_query:
            recall_resp = daemon_call("recall", {"query": recall_query, "limit": 8})
            if recall_resp.get("ok"):
                recall_results = recall_resp.get("result", {}).get("results", [])
                if recall_results:
                    output.append("RECALLED CONTEXT (related to recent work):")
                    for r in recall_results[:6]:
                        typ = r.get("type", "?")
                        title = r.get("title", "")[:70]
                        content = r.get("content", "")
                        if len(content) > 200: content = content[:200] + "..."
                        locked = "LOCKED " if r.get("locked") else ""
                        output.append("  [%s] %s%s" % (typ, locked, title))
                        output.append("    %s" % content)
                        output.append("")
    except Exception:
        pass

    output.append("Brain is live. Context was compacted — you lost conversation history.")
    output.append("The brain persists. Use brain.recall_with_embeddings() to recover context.")
    print("\n".join(output))

except Exception as e:
    import sys as _sys
    print("brain: post-compact daemon error: %s" % e, file=_sys.stderr)
    sys.exit(1)
'
  if [ $? -eq 0 ]; then
    exit 0
  fi
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
except Exception as e:
    print("brain: post-compact import failed: %s" % e, file=sys.stderr)
    sys.exit(0)

output = ["BRAIN POST-COMPACTION REBOOT:", ""]

try:
    user = brain.get_config("default_user", "User")
    project = brain.get_config("default_project", "default")

    # Safety net: check if pre-compact synthesis ran
    import sqlite3
    last_synth = brain.conn.execute(
        "SELECT created_at FROM session_syntheses ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    session_start = brain.get_config("session_start_at", "")

    synth_ran = False
    if last_synth and session_start:
        synth_ran = last_synth[0] >= session_start

    if not synth_ran:
        output.append("NOTE: Pre-compact synthesis did not run. Running now...")
        try:
            synthesis = brain.synthesize_session()
            parts = []
            for key in ("decisions", "corrections", "open_questions"):
                val = synthesis.get(key)
                if val:
                    parts.append("%s %s" % (val, key))
            if parts:
                output.append("  Synthesis: " + ", ".join(parts))
            else:
                output.append("  Synthesis: no notable events captured")
            brain.save()
        except Exception as e:
            output.append("  Synthesis failed: %s" % e)
        output.append("")

    # Re-run lightweight boot
    boot = brain.context_boot(user=user, project=project, task="post-compaction reboot")

    # Locked rules
    locked_nodes = boot.get("locked", [])
    locked_rules = [n for n in locked_nodes if n.get("type") == "rule"]
    if locked_rules:
        output.append("LOCKED RULES (%d active):" % len(locked_rules))
        for rule in locked_rules[:15]:
            output.append("  %s" % rule.get("title", "")[:80])
        output.append("")

    # Last synthesis — but only if recent (stale synthesis misleads more than helps)
    synth_row = brain.conn.execute(
        """SELECT open_questions, decisions_made, corrections_received, created_at
           FROM session_syntheses ORDER BY created_at DESC LIMIT 1"""
    ).fetchone()
    if synth_row and synth_row[3]:
        from datetime import datetime, timezone
        try:
            synth_time = datetime.fromisoformat(synth_row[3].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_minutes = (now - synth_time).total_seconds() / 60

            if age_minutes < 30:
                # Fresh synthesis — show open questions
                oq = synth_row[0]
                if oq:
                    try:
                        questions = json.loads(oq)
                        if questions:
                            output.append("OPEN QUESTIONS (from synthesis %d min ago):" % int(age_minutes))
                            for q in questions[:5]:
                                output.append("  ? %s" % str(q)[:100])
                            output.append("")
                    except Exception:
                        pass
            else:
                # Stale synthesis - just note it exists, skip old questions
                output.append("NOTE: Last synthesis was %.0f hours ago - open questions may be resolved." % (age_minutes / 60))
                output.append("  Use brain.recall_with_embeddings() for current context instead.")
                output.append("")
        except Exception:
            pass

    # Consciousness signals (abbreviated)
    signals = brain.get_consciousness_signals()
    for sig_key, sig_label in [("reminders", "REMINDERS"), ("evolutions", "EVOLUTIONS")]:
        items = signals.get(sig_key, [])
        if items:
            output.append("%s:" % sig_label)
            for item in items[:5]:
                output.append("  %s" % item.get("title", "")[:80])
            output.append("")

    # Developmental stage (brief)
    try:
        dev = brain.assess_developmental_stage()
        if dev:
            output.append("STAGE: %s (%.0f%%)" % (dev.get("stage_name", "?"), dev.get("maturity_score", 0) * 100))
            output.append("")
    except Exception:
        pass

    # v5.3: Recall context related to recent work — what was Claude doing before compaction?
    try:
        # Build recall query from recent nodes + synthesis
        recall_query_parts = []

        # Recent node titles (last 2 hours)
        recent_rows = brain.conn.execute(
            "SELECT title FROM nodes WHERE created_at > datetime('now', '-2 hours') ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
        for row in recent_rows:
            if row[0]:
                recall_query_parts.append(row[0])

        # Synthesis decisions/corrections
        if synth_row:
            for field_idx in (1, 2):  # decisions_made, corrections_received
                val = synth_row[field_idx]
                if val and val != "[]":
                    recall_query_parts.append(str(val)[:150])

        if recall_query_parts:
            recall_query = " ".join(recall_query_parts)[:500]
            try:
                result = brain.recall_with_embeddings(query=recall_query, limit=8)
            except Exception:
                result = brain.recall(query=recall_query, limit=8)

            recall_results = result.get("results", [])
            # Filter out very recent nodes (already captured in synthesis)
            recent_ids = {r[0] if len(r) > 0 else None for r in brain.conn.execute(
                "SELECT id FROM nodes WHERE created_at > datetime('now', '-2 hours')"
            ).fetchall()}
            new_recall = [r for r in recall_results if r.get("id") not in recent_ids]

            if new_recall:
                output.append("RECALLED CONTEXT (related to recent work):")
                for r in new_recall[:6]:
                    typ = r.get("type", "?")
                    title = r.get("title", "")[:70]
                    content = r.get("content", "")
                    if len(content) > 200: content = content[:200] + "..."
                    locked = "LOCKED " if r.get("locked") else ""
                    output.append("  [%s] %s%s" % (typ, locked, title))
                    output.append("    %s" % content)
                    output.append("")
    except Exception:
        pass

    # Find transcript for rehydration hint
    transcript_path = None
    try:
        # Find the most recent JSONL transcript
        home = os.path.expanduser("~")
        claude_projects = os.path.join(home, ".claude", "projects")
        if os.path.isdir(claude_projects):
            candidates = []
            for pdir in os.listdir(claude_projects):
                ppath = os.path.join(claude_projects, pdir)
                if not os.path.isdir(ppath):
                    continue
                for fname in os.listdir(ppath):
                    if fname.endswith(".jsonl"):
                        fpath = os.path.join(ppath, fname)
                        candidates.append(fpath)
            if candidates:
                transcript_path = max(candidates, key=os.path.getmtime)
    except Exception:
        pass

    output.append("Brain is live. Context was compacted — you lost conversation history.")
    output.append("The brain persists. Use brain.recall_with_embeddings() to recover context.")
    if transcript_path:
        output.append("")
        output.append("TRANSCRIPT AVAILABLE FOR REHYDRATION:")
        output.append("  Path: %s" % transcript_path)
        output.append("  To recover lost context, run:")
        output.append("    BRAIN_DB_DIR=%s python3 %s/hooks/scripts/extract-session-log.py --last-n-hours 4" % (
            db_dir, os.environ.get("CLAUDE_PLUGIN_ROOT", ".")))
        output.append("  Or read the transcript directly to find what you lost.")

    brain.close()
    print("\n".join(output))

except Exception as e:
    print("brain: post-compact error: %s" % e, file=sys.stderr)
    try:
        brain.close()
    except Exception:
        pass
'

exit 0
