#!/bin/bash
# tmemory v14 (serverless) — SessionStart hook
# No HTTP server to start. Imports Python brain module directly.
#
# Brain resolution order:
# 1. TMEMORY_DB_DIR env var (explicit override)
# 2. AgentsContext/tmemory/ in mounted paths (Cowork sessions)
# 3. $HOME/AgentsContext/tmemory/ (local Claude Code)
# 4. Create in first available AgentsContext

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SERVER_DIR="$PLUGIN_ROOT/servers"

# ── Resolve brain location ──
DB_DIR=""

if [ -n "$TMEMORY_DB_DIR" ] && [ -d "$TMEMORY_DB_DIR" ]; then
  DB_DIR="$TMEMORY_DB_DIR"
fi

# Cowork: search mounted AgentsContext directories
if [ -z "$DB_DIR" ]; then
  for candidate in /sessions/*/mnt/AgentsContext/tmemory; do
    if [ -f "$candidate/brain.db" ]; then
      DB_DIR="$candidate"
      break
    fi
  done
fi

# Local Claude Code
if [ -z "$DB_DIR" ] && [ -f "$HOME/AgentsContext/tmemory/brain.db" ]; then
  DB_DIR="$HOME/AgentsContext/tmemory"
fi

# Last resort: create in AgentsContext if it exists
if [ -z "$DB_DIR" ]; then
  for ac_dir in /sessions/*/mnt/AgentsContext; do
    if [ -d "$ac_dir" ]; then
      DB_DIR="$ac_dir/tmemory"
      mkdir -p "$DB_DIR" 2>/dev/null
      break
    fi
  done
fi

# Final fallback — writable temp location
if [ -z "$DB_DIR" ]; then
  DB_DIR="/tmp/tmemory-brain"
  mkdir -p "$DB_DIR"
fi

export TMEMORY_DB_DIR="$DB_DIR"
export TMEMORY_SERVER_DIR="$SERVER_DIR"

# ── Direct Python brain call (no HTTP server needed) ──
SCRIPT_DIR="$(dirname "$0")"

exec python3 -c '
import sys, os, json

server_dir = os.environ.get("TMEMORY_SERVER_DIR", "")
db_dir = os.environ.get("TMEMORY_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

# Add servers dir to Python path for imports
if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

try:
    from servers.brain import Brain, BRAIN_VERSION
except ImportError as e:
    print(f"tmemory: Failed to import brain module: {e}", file=sys.stderr)
    sys.exit(1)

# Initialize brain (creates DB if needed, runs schema migration)
try:
    brain = Brain(db_path)
except Exception as e:
    print(f"tmemory: Failed to initialize brain: {e}", file=sys.stderr)
    sys.exit(1)

# Reset session activity for new session
brain.reset_session_activity()

# Boot context
user = os.environ.get("TMEMORY_USER", "User")
project = os.environ.get("TMEMORY_PROJECT", "default")

# Read user/project from brain config if not set
if user == "User":
    stored_user = brain.get_config("default_user", "User")
    if stored_user and stored_user != "User":
        user = stored_user
if project == "default":
    stored_project = brain.get_config("default_project", "default")
    if stored_project and stored_project != "default":
        project = stored_project

ctx = brain.context_boot(user=user, project=project, task="session start")

# Health check with auto-fix
health = brain.health_check(session_id="session_boot", auto_fix=True)

# Staged learnings
staged = brain.list_staged(status="pending", limit=10)

# Auto-promote staged with enough revisits
brain.auto_promote_staged(revisit_threshold=3)

# Suggest metrics
metrics = brain.get_suggest_metrics(period_days=7)

# Procedures for session start
procs = brain.procedure_trigger("session_start", {"session_count": ctx.get("reset_count", 0)})

# Save after all operations
brain.save()

# ── Output context for Claude ──
# NOTE: Avoid f-strings with dict access containing quotes (breaks in bash -c)
print("tmemory brain booted from: " + db_dir)
print()
reset_count = ctx.get("reset_count", 0)
print("Session #" + str(reset_count + 1))
print()

# Last session note
note = ctx.get("last_session_note") or {}
if note:
    ntitle = note.get("title", "")
    print("Last session note: " + ntitle)
    ncontent = note.get("content", "")
    if ncontent:
        print(ncontent[:500])
    print()

# Health alerts
issues = health.get("issues", [])
actions = health.get("actions", [])
high = [i for i in issues if i.get("severity") == "high"]
if high:
    print("HEALTH ALERTS:")
    for i in high:
        itype = i.get("type", "?")
        imsg = i.get("message", "")
        print("  [" + itype + "] " + imsg)
    print()
medium = [i for i in issues if i.get("severity") == "medium"]
if medium:
    print("Health warnings:")
    for i in medium:
        itype = i.get("type", "?")
        imsg = i.get("message", "")
        print("  [" + itype + "] " + imsg)
    print()
if actions:
    print("Auto-maintenance: " + "  ".join(actions))
    print()

# Session-start procedures
matched = procs.get("matched", [])
if matched:
    print("Procedures to run this session:")
    for p in matched:
        pcat = p.get("category", "")
        ptitle = p.get("title", "")
        print("  [" + pcat + "] " + ptitle)
        psteps = p.get("steps", "")
        if psteps:
            print("    " + psteps[:200])
    print()

# Locked rules
locked_nodes = ctx.get("locked", [])
rules = [n for n in locked_nodes if n.get("type") == "rule"][:10]
if rules:
    print("Key locked rules:")
    for r in rules:
        print("  - " + r.get("title", ""))
    print()

# Staged learnings
pending = staged.get("staged", [])
if pending:
    print("STAGED LEARNINGS (" + str(len(pending)) + " pending review):")
    print("Ask the user to confirm or dismiss these before they influence future sessions.")
    for s in pending[:5]:
        stitle = str(s.get("title", "")).replace("[staged] ", "")
        sconf = s.get("confidence", 0.2)
        srev = s.get("times_revisited", 0)
        snid = s.get("node_id", "")
        print("  [%.1f] %s (revisited %dx) [%s]" % (sconf, stitle, srev, snid))
        scontent = str(s.get("content", ""))[:150]
        if scontent:
            print("       " + scontent)
    if len(pending) > 5:
        print("  ... and " + str(len(pending) - 5) + " more")
    print()

tn = ctx.get("total_nodes", "?")
te = ctx.get("total_edges", "?")
tl = ctx.get("total_locked", "?")
print("Brain status: %s nodes, %s edges, %s locked" % (tn, te, tl))

# Embedder status — alert user if running degraded
from servers import embedder as _emb
if _emb.is_ready():
    es = _emb.get_stats()
    print("Embeddings: ACTIVE (%s, %dd, loaded in %dms)" % (es["model_name"], es["embedding_dim"], es["load_time_ms"]))
else:
    print("WARNING: Embeddings UNAVAILABLE — running TF-IDF only (degraded recall quality).")
    print("  The brain works but semantic recall is weaker without embeddings.")
    print("  To fix: pip install fastembed sqlite-vec --break-system-packages")
print()

# Suggest metrics
total_suggests = metrics.get("total_suggests", 0)
if total_suggests > 0:
    pd = metrics.get("period_days", 7)
    al = metrics.get("avg_locked_per_suggest", 0)
    ap = metrics.get("avg_promoted_per_suggest", 0)
    print("Suggest metrics (%dd): %d calls, avg %.0f locked/call, avg %.1f promoted/call" % (pd, total_suggests, al, ap))
    print()

# Debug mode
debug_on = brain.get_debug_status()
if debug_on:
    brain.log_debug("session_start", "boot_hook", metadata=json.dumps({"db_dir": db_dir}))
    print("DEBUG MODE ON - all brain interactions are being logged.")
    print()

print("IMPORTANT: PreToolUse hook auto-surfaces memories before file edits. Use /remember for decisions.")

brain.close()
' 2>/dev/null

# ── v14: Post-compaction log reader ──
SCRIPT_DIR="$(dirname "$0")"
SESSION_LOG=""
if [ -f "$SCRIPT_DIR/extract-session-log.py" ]; then
  SESSION_LOG=$(TMEMORY_DB_DIR="$DB_DIR" python3 "$SCRIPT_DIR/extract-session-log.py" 2>/dev/null)
fi

if [ -n "$SESSION_LOG" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════"
  echo "POST-COMPACTION LOG: Unencoded session history detected."
  echo "Review this narrative and encode important learnings into"
  echo "the brain. Both your reasoning and the user's decisions"
  echo "matter — this is a shared brain."
  echo "═══════════════════════════════════════════════════════════"
  echo ""
  echo "$SESSION_LOG"
  echo ""
  echo "═══════════════════════════════════════════════════════════"
  echo "ACTION: Review the above and call /remember for any"
  echo "decisions, corrections, learnings, or insights that are"
  echo "not already in the brain. Then proceed with the user's task."
  echo "═══════════════════════════════════════════════════════════"
fi
