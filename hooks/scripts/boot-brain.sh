#!/bin/bash
# brain v4 (serverless) — SessionStart hook
# Imports Python brain module directly. No HTTP server.
#
# Brain DB resolution order:
# 1. BRAIN_DB_DIR env var (explicit override)
# 2. /sessions/*/mnt/AgentsContext/brain/ (Cowork mounted paths)
# 3. $HOME/AgentsContext/brain/ (local Claude Code via symlink)
# 4. Create in first available Cowork AgentsContext mount
# If none found, boot fails cleanly (no /tmp fallback — silent data loss is worse).
#
# TODO(pypi): Once PyPI storage limit is increased:
#   - Re-add auto-install: pip install brain-embedding fastembed onnxruntime
#   - Remove the manual pip install brain-embedding fallback in embedder.py (lines 136-149)
#   - The brain-embedding wheel (~416MB) bundles the ONNX model so Cowork
#     environments (where HuggingFace is proxy-blocked) can get embeddings.
#   - Currently: fastembed+onnxruntime installed manually on local Mac,
#     Cowork has them pre-installed. No auto-install in boot.

source "$(dirname "$0")/resolve-brain-db.sh"

# No DB found — fail cleanly
if [ -z "$BRAIN_DB_DIR" ]; then
  echo "brain: No brain.db found and no AgentsContext available. Set BRAIN_DB_DIR env var." >&2
  exit 0
fi

# ── Direct Python brain call ──
exec python3 -c '
import sys, os, json

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

try:
    from servers.brain import Brain, BRAIN_VERSION
except ImportError as e:
    print("brain: Failed to import brain module: " + str(e), file=sys.stderr)
    sys.exit(1)

try:
    brain = Brain(db_path)
except Exception as e:
    print("brain: Failed to initialize: " + str(e), file=sys.stderr)
    sys.exit(1)

brain.reset_session_activity()

# Resolve user/project from env or brain config
user = os.environ.get("BRAIN_USER", "User")
project = os.environ.get("BRAIN_PROJECT", "default")
if user == "User":
    stored_user = brain.get_config("default_user", "User")
    if stored_user and stored_user != "User":
        user = stored_user
if project == "default":
    stored_project = brain.get_config("default_project", "default")
    if stored_project and stored_project != "default":
        project = stored_project

ctx = brain.context_boot(user=user, project=project, task="session start")
health = brain.health_check(session_id="session_boot", auto_fix=True)
staged = brain.list_staged(status="pending", limit=10)
brain.auto_promote_staged(revisit_threshold=3)
metrics = brain.get_suggest_metrics(period_days=7)
procs = brain.procedure_trigger("session_start", {"session_count": ctx.get("reset_count", 0)})

# Consciousness signals
consciousness = brain.get_consciousness_signals()
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

# Host environment scan
host_info = {}
host_diff = {}
host_research = []
host_result = brain.scan_host_environment()
host_info = host_result.get("environment", {})
host_diff = host_result.get("diff", {})
host_research = host_result.get("research_needed", [])

# Dreams + self-reflection
dreams = brain.get_surfaceable_dreams(limit=2)
brain.auto_generate_self_reflection()

brain.save()

# ── Output context for Claude ──
print("brain v" + str(BRAIN_VERSION) + " booted from: " + db_dir)
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

# ── BRAIN CONSCIOUSNESS ──
has_conscious = reminders or active_evolutions or fluid_personal or fading or failure_modes or stale_count > 10 or performance or capabilities or interactions or meta_learning or novelty or miss_trends or encoding_gap or density_shift or emotional_trajectory or rule_contradictions

if has_conscious:
    print("BRAIN CONSCIOUSNESS")
    print()

if reminders:
    for rem in reminders:
        rtitle = rem.get("title", "")
        rdue = str(rem.get("due_date", ""))[:10]
        rcreated = str(rem.get("created_at", ""))[:10]
        print("  " + rtitle[:80])
        print("    due: " + rdue + " — set: " + rcreated)
    print()

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

if failure_modes:
    for fm in failure_modes:
        print("  " + fm.get("title", "")[:80])
        fmc = str(fm.get("content", ""))[:100]
        if fmc:
            print("    " + fmc)
    print()

if fading:
    print("  FADING KNOWLEDGE (accessed 3+ times but untouched for 14+ days):")
    for f in fading:
        ftitle = f.get("title", "")[:70]
        flast = str(f.get("last_accessed", ""))[:10]
        print("    " + ftitle + " — last: " + flast)
    print("  Still relevant? Access them to refresh, or let them decay.")
    print()

if stale_count > 10:
    print("  STALE — %d context nodes older than 7 days. Consider archiving." % stale_count)
    print()

if fluid_personal:
    print("  FLUID PERSONAL KNOWLEDGE — confirm or update:")
    for fp in fluid_personal[:5]:
        fptitle = fp.get("title", "")
        print("    ? " + fptitle[:80] + " — still true?")
    print()

if performance:
    for p in performance[:2]:
        ptitle = p.get("title", "")[:80]
        print("  " + ptitle)
    print()

if capabilities:
    for c in capabilities[:2]:
        ctitle = c.get("title", "")[:80]
        print("  " + ctitle)
    print()

if interactions:
    for i in interactions[:2]:
        ititle = i.get("title", "")[:80]
        print("  " + ititle)
    print()

if meta_learning:
    for m in meta_learning[:2]:
        mtitle = m.get("title", "")[:80]
        print("  " + mtitle)
    print()

if novelty:
    for n in novelty[:2]:
        ntitle = n.get("title", "")[:60]
        print("  " + ntitle + " — introduced this session")
    print()

if miss_trends:
    for mt in miss_trends[:2]:
        mq = mt.get("query", "")[:50]
        mc = mt.get("count", 0)
        print("  Recall keeps missing on: " + mq + " (%dx this week)" % mc)
    print()

if encoding_gap:
    egw = encoding_gap.get("warning", "")
    print("  " + egw)
    print()

if density_shift:
    dsw = density_shift.get("warning", "")
    print("  " + dsw)
    print()

if emotional_trajectory:
    etrend = emotional_trajectory.get("trend", "")
    eavg = emotional_trajectory.get("recent_avg", 0)
    if etrend == "increasing":
        print("  Emotional intensity trending UP (avg %.2f recent)" % eavg)
    elif etrend == "decreasing":
        print("  Emotional intensity trending DOWN (avg %.2f recent)" % eavg)
    print()

if rule_contradictions:
    print("  POTENTIAL RULE CONTRADICTIONS:")
    for rc in rule_contradictions[:3]:
        rn = rc.get("recent_node", "")[:50]
        rl = rc.get("locked_rule", "")[:50]
        rs = rc.get("similarity", 0)
        print("    " + rn + " may conflict with LOCKED: " + rl + " (sim %.2f)" % rs)
    print()

if dreams:
    for dr in dreams:
        dtitle = dr.get("title", "")[:70]
        dhops = dr.get("total_hops", 0)
        print("  " + dtitle + " — %d hops apart" % dhops)
        dcontent = str(dr.get("content", ""))[:120]
        if dcontent:
            print("    " + dcontent)
    print()

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

# Embedder status
from servers import embedder as _emb
if _emb.is_ready():
    es = _emb.get_stats()
    ename = es["model_name"]
    edim = es["embedding_dim"]
    etime = es["load_time_ms"]
    print("Embeddings: ACTIVE (" + str(ename) + ", " + str(edim) + "d, loaded in " + str(etime) + "ms)")
else:
    print("WARNING: Embeddings UNAVAILABLE — running TF-IDF only (degraded recall quality).")
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

print("PreToolUse hook auto-surfaces memories before file edits. Use brain.remember() for decisions.")

brain.close()
'

# ── Post-compaction log reader ──
SCRIPT_DIR="$(dirname "$0")"
SESSION_LOG=""
if [ -f "$SCRIPT_DIR/extract-session-log.py" ]; then
  SESSION_LOG=$(BRAIN_DB_DIR="$DB_DIR" python3 "$SCRIPT_DIR/extract-session-log.py" 2>/dev/null)
fi

if [ -n "$SESSION_LOG" ]; then
  echo ""
  echo "POST-COMPACTION LOG: Unencoded session history detected."
  echo "Review and encode important learnings via brain.remember()."
  echo ""
  echo "$SESSION_LOG"
  echo ""
  echo "ACTION: Review the above and call brain.remember() for any"
  echo "decisions, corrections, or insights not already in the brain."
fi
