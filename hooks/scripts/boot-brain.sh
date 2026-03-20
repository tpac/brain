#!/bin/bash
# brain v15 (serverless) — SessionStart hook
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

# v5: Engineering context — the warm-up killer
eng_ctx = {}
try:
    eng_ctx = brain.get_engineering_context(project=project)
except Exception:
    pass

# v5: Correction patterns — shape Claude's behavior
correction_patterns = []
try:
    correction_patterns = brain.get_correction_patterns(limit=5)
except Exception:
    pass

# v5: Last session synthesis
last_synthesis = None
try:
    last_synthesis = brain.get_last_synthesis()
except Exception:
    pass

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
stale_reasoning = consciousness.get("stale_reasoning", [])
uncharted_code = consciousness.get("uncharted_code", [])
stale_file_inv = consciousness.get("stale_file_inventory", [])
vocabulary_gap = consciousness.get("vocabulary_gap", [])
recurring_divergence = consciousness.get("recurring_divergence", [])
validated_approaches = consciousness.get("validated_approaches", [])
uncertain_areas = consciousness.get("uncertain_areas", [])
mental_model_drift = consciousness.get("mental_model_drift", [])
silent_errors = consciousness.get("silent_errors", [])

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

# ═══════════════════════════════════════════════════════════
# v5: ENGINEERING CONTEXT — warm-up killer boot sequence
# ═══════════════════════════════════════════════════════════

# Layer 1: System purpose
sys_purpose = eng_ctx.get("system_purpose")
if sys_purpose:
    sp_purpose = sys_purpose.get("purpose", "")
    if sp_purpose:
        print("SYSTEM PURPOSE: " + sp_purpose[:300])
        sp_arch = sys_purpose.get("architecture", "")
        if sp_arch:
            print("Architecture: " + sp_arch[:200])
        sp_decisions = sys_purpose.get("key_decisions")
        if sp_decisions:
            try:
                decs = json.loads(sp_decisions) if isinstance(sp_decisions, str) else sp_decisions
                if decs:
                    print("Key decisions: " + "; ".join(str(d)[:60] for d in decs[:3]))
            except (json.JSONDecodeError, TypeError):
                pass
        print()

# Layer 1b: Purpose nodes (system/module/file scope)
purposes = eng_ctx.get("purposes", [])
if purposes:
    print("PROJECT UNDERSTANDING:")
    for p in purposes[:8]:
        pscope = p.get("scope", "?")
        ptitle = p.get("title", "")[:70]
        pcontent = p.get("content", "")
        if len(pcontent) > 150:
            pcontent = pcontent[:150] + "..."
        print("  [" + pscope + "] " + ptitle)
        if pcontent:
            print("    " + pcontent)
    print()

# Layer 2: Last session synthesis
if last_synthesis:
    ls_date = str(last_synthesis.get("created_at", ""))[:10]
    ls_dur = last_synthesis.get("duration_minutes")
    ls_header = "LAST SESSION"
    if ls_date:
        ls_header += " (" + ls_date + ")"
    if ls_dur:
        ls_header += " — " + str(ls_dur) + " min"
    print(ls_header + ":")
    ls_decisions = last_synthesis.get("decisions_made", [])
    if ls_decisions:
        print("  Decisions:")
        for d in ls_decisions[:3]:
            dtitle = d.get("title", "") if isinstance(d, dict) else str(d)
            print("    - " + dtitle[:80])
    ls_corrections = last_synthesis.get("corrections_received", [])
    if ls_corrections:
        print("  Corrections received: " + str(len(ls_corrections)))
        for c in ls_corrections[:2]:
            assumed = c.get("assumed", "") if isinstance(c, dict) else str(c)
            print("    - " + assumed[:80])
    ls_arcs = last_synthesis.get("teaching_arcs", [])
    if ls_arcs:
        print("  Teaching arcs:")
        for a in ls_arcs[:2]:
            apattern = a.get("pattern", "") if isinstance(a, dict) else str(a)
            print("    - " + apattern[:80])
    ls_open = last_synthesis.get("open_questions", [])
    if ls_open:
        print("  Open questions:")
        for q in ls_open[:3]:
            qtext = q.get("text", q) if isinstance(q, dict) else str(q)
            print("    ? " + str(qtext)[:80])
    print()

# Layer 3: File changes since last session
file_changes = eng_ctx.get("file_changes", [])
if file_changes:
    print("FILES CHANGED since last session:")
    for fc in file_changes[:10]:
        fpath = fc.get("file_path", "")
        fpurpose = fc.get("purpose", "")
        print("  " + fpath)
        if fpurpose:
            print("    was: " + fpurpose[:80])
    print("  Re-read changed files to update your understanding.")
    print()

# Layer 4: Correction patterns — Claude's known failure modes
if correction_patterns:
    print("CORRECTION PATTERNS (known divergence tendencies):")
    for cp in correction_patterns[:3]:
        cpat = cp.get("pattern", "")[:70]
        ccnt = cp.get("count", 0)
        csev = cp.get("max_severity", "minor")
        print("  [" + csev + " x" + str(ccnt) + "] " + cpat)
        cex = cp.get("examples", "")[:120]
        if cex:
            print("    e.g.: " + cex)
    print("  Be aware of these patterns. They are where you tend to diverge from reality.")
    print()

# Layer 5: Active evolutions — already handled in consciousness section below

# Layer 6: Operator vocabulary
vocab = eng_ctx.get("vocabulary", [])
if vocab:
    print("OPERATOR VOCABULARY:")
    for v in vocab[:10]:
        vtitle = v.get("title", "")[:40]
        vcontent = v.get("content", "")[:80]
        print("  " + vtitle + " => " + vcontent)
    print()

# Vocabulary gaps — terms the user used that have no mapping
if vocabulary_gap:
    print("VOCABULARY GAPS (unmapped operator terms — consider learning these):")
    for vg in vocabulary_gap[:5]:
        if isinstance(vg, dict):
            vgterm = vg.get("term", "")
            vgpreview = vg.get("message_preview", "")[:60]
            print("  ? " + vgterm)
            if vgpreview:
                print("    from: " + vgpreview + "...")
        else:
            print("  ? " + str(vg))
    print("  ACTION: If you know what these map to, use brain.learn_vocabulary(term, maps_to, context).")
    print("  If unsure, ASK the user: 'You mentioned X — what does that refer to in this context?'")
    print("  Note: the same term can mean different things in different contexts. Include context when learning.")
    print()

# Layer 7: Constraints + Conventions (architecture overview)
constraints = eng_ctx.get("constraints", [])
if constraints:
    print("CONSTRAINTS:")
    for c in constraints[:5]:
        ctitle = c.get("title", "")[:70]
        print("  " + ctitle)
        ccontent = c.get("content", "")
        if ccontent and len(ccontent) > 120:
            ccontent = ccontent[:120] + "..."
        if ccontent:
            print("    " + ccontent)
    print()

conventions = eng_ctx.get("conventions", [])
if conventions:
    print("CONVENTIONS:")
    for cv in conventions[:3]:
        cvtitle = cv.get("title", "")[:70]
        print("  " + cvtitle)
        cvcontent = cv.get("content", "")
        if cvcontent and len(cvcontent) > 120:
            cvcontent = cvcontent[:120] + "..."
        if cvcontent:
            print("    " + cvcontent)
    print()

# Impact links — surfaced at boot for awareness
impacts = eng_ctx.get("impacts", [])
if impacts:
    print("CHANGE IMPACT MAP:")
    for imp in impacts[:5]:
        imptitle = imp.get("title", "")[:80]
        print("  " + imptitle)
    print()

# File inventory summary
file_inv = eng_ctx.get("file_inventory", {}).get("files", [])
if file_inv:
    print("FILE INVENTORY (" + str(len(file_inv)) + " tracked):")
    for fi in file_inv[:8]:
        fipath = fi.get("file_path", "")
        fipurpose = fi.get("purpose", "")[:60]
        print("  " + fipath + " — " + fipurpose)
    if len(file_inv) > 8:
        print("  ... and " + str(len(file_inv) - 8) + " more")
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
has_conscious = reminders or active_evolutions or fluid_personal or fading or failure_modes or stale_count > 10 or performance or capabilities or interactions or meta_learning or novelty or miss_trends or encoding_gap or density_shift or emotional_trajectory or rule_contradictions or stale_reasoning or uncharted_code or stale_file_inv or recurring_divergence or validated_approaches or uncertain_areas or mental_model_drift or silent_errors

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

# v5: Stale reasoning — rich nodes that need revalidation
if stale_reasoning:
    print("  STALE REASONING (detailed rationale that may be outdated):")
    for sr in stale_reasoning:
        srt = sr.get("title", "")[:60]
        srlv = sr.get("last_validated")
        srage = "never validated" if not srlv else "validated: " + str(srlv)[:10]
        print("    " + srt + " — " + srage)
        srp = sr.get("reasoning_preview", "")
        if srp:
            print("      " + srp[:100])
    print("  Still accurate? Use brain.validate_node(id) to confirm.")
    print()

# v5: Recurring divergence patterns — surfaced as consciousness
if recurring_divergence:
    print("  RECURRING DIVERGENCE PATTERNS:")
    for rd in recurring_divergence[:3]:
        rdpat = rd.get("pattern", "")[:70]
        rdcnt = rd.get("count", 0)
        print("    [x" + str(rdcnt) + "] " + rdpat)
    print("  These are persistent failure modes. Be extra careful in these areas.")
    print()

# v5: Recently validated approaches
if validated_approaches:
    print("  RECENTLY VALIDATED:")
    for va in validated_approaches[:3]:
        vatitle = va.get("title", "")[:70]
        vacount = va.get("count", 0)
        print("    " + vatitle + " (validated " + str(vacount) + "x)")
    print()

# v5: Silent errors — operations that failed without surfacing
if silent_errors:
    print("  SILENT ERRORS (last 24h — these failed without you knowing):")
    for se in silent_errors[:5]:
        se_source = se.get("source", "unknown")
        se_error = se.get("error", "")[:100]
        se_context = se.get("context", "")[:60]
        se_time = se.get("created_at", "")[:19]
        print("    [%s] %s: %s" % (se_time, se_source, se_error))
        if se_context:
            print("      context: %s" % se_context)
    print("  ACTION: These may indicate broken features. Investigate and fix.")
    print("  Use brain.get_recent_errors() for full details + tracebacks.")
    print()

# v5: Uncertain areas — things Claude knows it doesn't understand
if uncertain_areas:
    print("  UNCERTAIN AREAS (unresolved):")
    for ua in uncertain_areas[:3]:
        uatitle = ua.get("title", "")[:70]
        uapreview = ua.get("preview", "")[:100]
        print("    ? " + uatitle)
        if uapreview:
            print("      " + uapreview)
    print("  Consider investigating these if relevant to current work.")
    print()

# v5: Mental model drift — models that may be outdated or contradicted
if mental_model_drift:
    print("  MENTAL MODEL DRIFT (may need revision):")
    for mm in mental_model_drift[:3]:
        mmtitle = mm.get("title", "")[:70]
        mmconf = mm.get("confidence")
        mmchecked = mm.get("last_checked", "")[:10]
        label = mmtitle
        if mmconf is not None:
            label += " (confidence: " + str(round(mmconf, 2)) + ")"
        if mmchecked:
            label += " [last checked: " + mmchecked + "]"
        print("    ~ " + label)
    print("  These mental models may be stale. Validate before relying on them.")
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

print("IMPORTANT: PreToolUse hook auto-surfaces memories before file edits.")
print("  /remember — store decisions, learnings")
print("  /remember_rich — detailed encoding with reasoning, alternatives, user quotes")
print("  brain.remember_purpose/mechanism/impact/constraint/convention/lesson — engineering memory")
print("  brain.learn_vocabulary — map operator terms to code")
print("  brain.remember_mental_model/remember_uncertainty — Claude cognitive layer")

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
