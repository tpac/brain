#!/bin/bash
# brain v14 (serverless) — SessionStart hook
# No HTTP server to start. Imports Python brain module directly.
#
# Brain resolution order:
# 1. BRAIN_DB_DIR env var (explicit override)
# 2. AgentsContext/brain/ in mounted paths (Cowork sessions)
# 3. $HOME/AgentsContext/brain/ (local Claude Code)
# 4. Create in first available AgentsContext

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SERVER_DIR="$PLUGIN_ROOT/servers"

# ── Resolve brain location ──
DB_DIR=""

if [ -n "$BRAIN_DB_DIR" ] && [ -d "$BRAIN_DB_DIR" ]; then
  DB_DIR="$BRAIN_DB_DIR"
fi

# Cowork: search mounted AgentsContext directories
if [ -z "$DB_DIR" ]; then
  for candidate in /sessions/*/mnt/AgentsContext/brain; do
    if [ -f "$candidate/brain.db" ]; then
      DB_DIR="$candidate"
      break
    fi
  done
fi

# Local Claude Code
if [ -z "$DB_DIR" ] && [ -f "$HOME/AgentsContext/brain/brain.db" ]; then
  DB_DIR="$HOME/AgentsContext/brain"
fi

# Last resort: create in AgentsContext if it exists
if [ -z "$DB_DIR" ]; then
  for ac_dir in /sessions/*/mnt/AgentsContext; do
    if [ -d "$ac_dir" ]; then
      DB_DIR="$ac_dir/brain"
      mkdir -p "$DB_DIR" 2>/dev/null
      break
    fi
  done
fi

# Final fallback — writable temp location
if [ -z "$DB_DIR" ]; then
  DB_DIR="/tmp/brain-data"
  mkdir -p "$DB_DIR"
fi

export BRAIN_DB_DIR="$DB_DIR"
export BRAIN_SERVER_DIR="$SERVER_DIR"

# ── Auto-install embedding dependencies ──
# fastembed auto-downloads models from HuggingFace on first use and caches them.
# On normal machines: just works. No user action needed.
# On restricted environments (Cowork/proxy): if HuggingFace is blocked,
# fall back to brain-embedding pip package which bundles the model files.
python3 -c "import fastembed" 2>/dev/null || \
  pip install fastembed --break-system-packages --quiet 2>/dev/null || true

# ── Direct Python brain call (no HTTP server needed) ──
SCRIPT_DIR="$(dirname "$0")"

exec python3 -c '
import sys, os, json

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

# Add servers dir to Python path for imports
if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

try:
    from servers.brain import Brain, BRAIN_VERSION
except ImportError as e:
    print(f"brain: Failed to import brain module: {e}", file=sys.stderr)
    sys.exit(1)

# Initialize brain (creates DB if needed, runs schema migration)
try:
    brain = Brain(db_path)
except Exception as e:
    print(f"brain: Failed to initialize brain: {e}", file=sys.stderr)
    sys.exit(1)

# Reset session activity for new session
brain.reset_session_activity()

# Boot context
user = os.environ.get("BRAIN_USER", "User")
project = os.environ.get("BRAIN_PROJECT", "default")

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

# v4: Gather ALL consciousness signals
consciousness = {}
try:
    consciousness = brain.get_consciousness_signals()
except Exception:
    pass  # Method may not exist on older schema

active_evolutions = consciousness.get("evolutions", [])
fluid_personal = consciousness.get("fluid_personal", [])
reminders = consciousness.get("reminders", [])
fading = consciousness.get("fading", [])
stale_count = consciousness.get("stale_context_count", 0)
failure_modes = consciousness.get("failure_modes", [])
performance = consciousness.get("performance", [])
capabilities = consciousness.get("capabilities", [])
interactions = consciousness.get("interactions", [])
meta_learning = consciousness.get("meta_learning", [])
novelty = consciousness.get("novelty", [])
miss_trends = consciousness.get("miss_trends", [])
encoding_gap = consciousness.get("encoding_gap")
density_shift = consciousness.get("density_shift")
emotional_trajectory = consciousness.get("emotional_trajectory")
rule_contradictions = consciousness.get("rule_contradictions", [])

# v4: Host environment scan
host_info = {}
host_diff = {}
host_research = []
try:
    host_result = brain.scan_host_environment()
    host_info = host_result.get("environment", {})
    host_diff = host_result.get("diff", {})
    host_research = host_result.get("research_needed", [])
except Exception:
    pass

# v4: Surfaceable dreams (high surprise, cross-cluster)
dreams = []
try:
    dreams = brain.get_surfaceable_dreams(limit=2)
except Exception:
    pass

# v4: Auto self-reflection
try:
    brain.auto_generate_self_reflection()
except Exception:
    pass

# Save after all operations
brain.save()

# ── Output context for Claude ──
# NOTE: Avoid f-strings with dict access containing quotes (breaks in bash -c)
print("brain booted from: " + db_dir)
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

# ═══════════════════════════════════════════════════════════
# v4: BRAIN CONSCIOUSNESS — everything the brain wants to share
# ═══════════════════════════════════════════════════════════
has_conscious = reminders or active_evolutions or fluid_personal or fading or failure_modes or stale_count > 10 or performance or capabilities or interactions or meta_learning or novelty or miss_trends or encoding_gap or density_shift or emotional_trajectory or rule_contradictions

if has_conscious:
    print("BRAIN CONSCIOUSNESS")
    print()

# Reminders due — surface FIRST, before everything else
if reminders:
    for rem in reminders:
        rtitle = rem.get("title", "")
        rdue = str(rem.get("due_date", ""))[:10]
        rcreated = str(rem.get("created_at", ""))[:10]
        print("  " + rtitle[:80])
        print("    due: " + rdue + " — set: " + rcreated)
    print()

# Active evolution nodes
if active_evolutions:
    for ev in active_evolutions[:6]:
        etitle = ev.get("title", "")
        econf = ev.get("confidence")
        ecreated = str(ev.get("created_at", ""))[:10]
        eline = "  " + etitle[:80]
        if econf is not None:
            eline += " (confidence: %.1f)" % econf
        eline += " — since " + ecreated
        print(eline)
        econtent = str(ev.get("content", ""))[:120]
        if econtent:
            print("    " + econtent)
        print()
    if len(active_evolutions) > 6:
        print("  ... and %d more active" % (len(active_evolutions) - 6))
    print()

# Failure modes — always visible
if failure_modes:
    for fm in failure_modes:
        print("  " + fm.get("title", "")[:80])
        fmc = str(fm.get("content", ""))[:100]
        if fmc:
            print("    " + fmc)
    print()

# Fading knowledge — important nodes about to decay
if fading:
    print("  FADING KNOWLEDGE (accessed 3+ times but untouched for 14+ days):")
    for f in fading:
        ftitle = f.get("title", "")[:70]
        flast = str(f.get("last_accessed", ""))[:10]
        print("    ⏳ " + ftitle + " — last: " + flast)
    print("  Still relevant? Access them to refresh, or let them decay.")
    print()

# Stale context cleanup suggestion
if stale_count > 10:
    print("  ⏳ STALE — %d context nodes older than 7 days. Consider archiving." % stale_count)
    print()

# Fluid personal nodes — "still true?" check
if fluid_personal:
    print("  FLUID PERSONAL KNOWLEDGE — confirm or update:")
    for fp in fluid_personal[:5]:
        fptitle = fp.get("title", "")
        print("    ? " + fptitle[:80] + " — still true?")
    print()

# Performance observations
if performance:
    for p in performance[:2]:
        ptitle = p.get("title", "")[:80]
        print("  " + ptitle)
    print()

# Capabilities
if capabilities:
    for c in capabilities[:2]:
        ctitle = c.get("title", "")[:80]
        print("  " + ctitle)
    print()

# Interaction observations
if interactions:
    for i in interactions[:2]:
        ititle = i.get("title", "")[:80]
        print("  " + ititle)
    print()

# Meta-learning methods
if meta_learning:
    for m in meta_learning[:2]:
        mtitle = m.get("title", "")[:80]
        print("  " + mtitle)
    print()

# Novelty — new terms or concepts introduced recently
if novelty:
    for n in novelty[:2]:
        ntitle = n.get("title", "")[:60]
        print("  " + ntitle + " — introduced this session")
    print()

# Miss trends — queries that keep failing
if miss_trends:
    for mt in miss_trends[:2]:
        mq = mt.get("query", "")[:50]
        mc = mt.get("count", 0)
        print("  Recall keeps missing on: " + mq + " (%dx this week)" % mc)
    print()

# Encoding gap warning
if encoding_gap:
    egw = encoding_gap.get("warning", "")
    print("  " + egw)
    print()

# Connection density shift
if density_shift:
    dsw = density_shift.get("warning", "")
    print("  " + dsw)
    print()

# Emotional trajectory
if emotional_trajectory:
    etrend = emotional_trajectory.get("trend", "")
    eavg = emotional_trajectory.get("recent_avg", 0)
    if etrend == "increasing":
        print("  Emotional intensity trending UP (avg %.2f recent)" % eavg)
    elif etrend == "decreasing":
        print("  Emotional intensity trending DOWN (avg %.2f recent)" % eavg)
    print()

# Rule contradictions
if rule_contradictions:
    print("  POTENTIAL RULE CONTRADICTIONS:")
    for rc in rule_contradictions[:3]:
        rn = rc.get("recent_node", "")[:50]
        rl = rc.get("locked_rule", "")[:50]
        rs = rc.get("similarity", 0)
        print("    " + rn + " may conflict with LOCKED: " + rl + " (sim %.2f)" % rs)
    print()

# Surfaceable dreams — high surprise, cross-cluster connections
if dreams:
    for dr in dreams:
        dtitle = dr.get("title", "")[:70]
        dhops = dr.get("total_hops", 0)
        print("  " + dtitle + " — %d hops apart" % dhops)
        dcontent = str(dr.get("content", ""))[:120]
        if dcontent:
            print("    " + dcontent)
    print()

# Host environment changes
if host_diff:
    print("  HOST CHANGES since last session:")
    for hkey in list(host_diff.keys())[:5]:
        hd = host_diff[hkey]
        was = str(hd.get("was", "?"))[:30]
        now = str(hd.get("now", "?"))[:30]
        print("    " + hkey + ": " + was + " -> " + now)
    print()

if host_research:
    print("  HOST RESEARCH NEEDED:")
    for hr in host_research[:3]:
        print("    " + hr)
    print("  Search online for release notes and changes that may affect the brain.")
    print()

if has_conscious or dreams or host_diff:
    print("Weave relevant items into conversation naturally. Do not dump all at once.")
    print()

tn = ctx.get("total_nodes", "?")
te = ctx.get("total_edges", "?")
tl = ctx.get("total_locked", "?")
print("Brain status: %s nodes, %s edges, %s locked" % (tn, te, tl))

# Embedder status — alert user if running degraded
from servers import embedder as _emb
if _emb.is_ready():
    es = _emb.get_stats()
    ename = es["model_name"]
    edim = es["embedding_dim"]
    etime = es["load_time_ms"]
    print("Embeddings: ACTIVE (" + str(ename) + ", " + str(edim) + "d, loaded in " + str(etime) + "ms)")
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
  SESSION_LOG=$(BRAIN_DB_DIR="$DB_DIR" python3 "$SCRIPT_DIR/extract-session-log.py" 2>/dev/null)
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
