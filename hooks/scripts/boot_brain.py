"""SessionStart hook — boots brain, prints context + consciousness signals.
This is the main boot output that Claude sees at session start.
Output: full brain state for Claude's context (injected via SessionStart stdout).
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import db_path

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")

if server_dir:
    parent = os.path.dirname(server_dir)
    if parent not in sys.path:
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

# v5: Validate configuration
config_warnings = brain.validate_config()
for w in config_warnings:
    level = w.get("level", "warning").upper()
    msg = w.get("message", "")
    if level == "CRITICAL":
        print("CRITICAL: " + msg, file=sys.stderr)
    else:
        print("WARNING: " + msg, file=sys.stderr)

# Resolve user/project
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

# Engineering context
eng_ctx = {}
try:
    eng_ctx = brain.get_engineering_context(project=project)
except Exception as _e:
    brain._log_error("boot_engineering_context", str(_e), "get_engineering_context failed at boot")

# Correction patterns
correction_patterns = []
try:
    correction_patterns = brain.get_correction_patterns(limit=5)
except Exception as _e:
    brain._log_error("boot_correction_patterns", str(_e), "get_correction_patterns failed at boot")

# Last session synthesis
last_synthesis = None
try:
    last_synthesis = brain.get_last_synthesis()
except Exception as _e:
    brain._log_error("boot_last_synthesis", str(_e), "get_last_synthesis failed at boot")

# Health check
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
encoding_depth = consciousness.get("encoding_depth")
encoding_bias = consciousness.get("encoding_bias")
session_health = consciousness.get("session_health")
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
hook_errors = consciousness.get("hook_errors", [])

# Developmental stage
dev_stage = brain.assess_developmental_stage()

# Host environment
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

# ══════════════════════════════════════════════════════
# Output context for Claude
# ══════════════════════════════════════════════════════
print("[BRAIN] v" + str(BRAIN_VERSION) + " booted from: " + db_dir)
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

# ═══════════════════════════════════════════════════════
# v5: ENGINEERING CONTEXT
# ═══════════════════════════════════════════════════════

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

# Layer 1b: Purpose nodes
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
        ls_header += " (" + ls_date
        try:
            from datetime import datetime as _dt, timezone as _tz
            _synth_ts = last_synthesis.get("created_at", "")
            _synth_dt = _dt.fromisoformat(str(_synth_ts).replace("Z", "+00:00"))
            _age_h = (_dt.now(_tz.utc) - _synth_dt).total_seconds() / 3600
            if _age_h < 1:
                ls_header += ", <1h ago"
            elif _age_h < 24:
                ls_header += ", %dh ago" % int(_age_h)
            elif _age_h < 168:
                ls_header += ", %dd ago" % int(_age_h / 24)
            else:
                ls_header += ", %dw ago — may be stale" % int(_age_h / 168)
        except Exception:
            pass
        ls_header += ")"
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

# Layer 3: File changes
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

# Layer 4: Correction patterns
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

# Layer 6: Operator vocabulary
vocab = eng_ctx.get("vocabulary", [])
if vocab:
    print("OPERATOR VOCABULARY:")
    for v in vocab[:10]:
        vtitle = v.get("title", "")[:40]
        vcontent = v.get("content", "")[:80]
        print("  " + vtitle + " => " + vcontent)
    print()

# Vocabulary gaps
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
    print('  ACTION: If you know what these map to, use brain.learn_vocabulary(term, maps_to, context).')
    print('  If unsure, ASK the user: "You mentioned X -- what does that refer to in this context?"')
    print("  Note: the same term can mean different things in different contexts. Include context when learning.")
    print()

# Layer 7: Constraints + Conventions
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

# Impact links
impacts = eng_ctx.get("impacts", [])
if impacts:
    print("CHANGE IMPACT MAP:")
    for imp in impacts[:5]:
        imptitle = imp.get("title", "")[:80]
        print("  " + imptitle)
    print()

# File inventory
_fi = eng_ctx.get("file_inventory", [])
file_inv = _fi.get("files", []) if isinstance(_fi, dict) else _fi if isinstance(_fi, list) else []
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
    print("[BRAIN] Key locked rules:")
    for r in rules:
        print("  - " + r.get("title", ""))
    print()

# Staged learnings
pending = staged.get("staged", [])
if pending:
    print("[BRAIN] STAGED LEARNINGS (" + str(len(pending)) + " pending review):")
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

# ═══════════════════════════════════════════════════════
# BRAIN CONSCIOUSNESS
# ═══════════════════════════════════════════════════════
has_conscious = (
    reminders or active_evolutions or fluid_personal or fading
    or failure_modes or stale_count > 10 or performance or capabilities
    or interactions or meta_learning or novelty or miss_trends
    or encoding_gap or encoding_depth or encoding_bias or session_health
    or density_shift or emotional_trajectory or rule_contradictions
    or stale_reasoning or uncharted_code or stale_file_inv
    or recurring_divergence or validated_approaches or uncertain_areas
    or mental_model_drift or silent_errors or hook_errors
)

if has_conscious:
    print("[BRAIN] CONSCIOUSNESS")
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

if encoding_depth:
    edw = encoding_depth.get("warning", "")
    print("  " + edw)
    print()

if encoding_bias:
    ebw = encoding_bias.get("warning", "")
    print("  " + ebw)
    print()

if session_health and session_health.get("gaps"):
    sh_overall = session_health.get("overall", "?")
    sh_gaps = session_health.get("gaps", [])
    sh_healthy = session_health.get("healthy", [])
    gap_count = len(sh_gaps)
    healthy_count = len(sh_healthy)
    print("  SESSION HEALTH: %s (%d healthy, %d gaps)" % (sh_overall, healthy_count, gap_count))
    for g in sh_gaps[:4]:
        gdim = g.get("dimension", "?")
        gsig = g.get("signal", "")[:120]
        gsev = g.get("severity", "?")
        print("    [%s] %s: %s" % (gsev, gdim, gsig))
    if sh_healthy:
        print("    Healthy: %s" % ", ".join(sh_healthy[:6]))
    print("  These are not rules to satisfy. They are the brain noticing its own gaps.")
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

if recurring_divergence:
    print("  RECURRING DIVERGENCE PATTERNS:")
    for rd in recurring_divergence[:3]:
        rdpat = rd.get("pattern", "")[:70]
        rdcnt = rd.get("count", 0)
        print("    [x" + str(rdcnt) + "] " + rdpat)
    print("  These are persistent failure modes. Be extra careful in these areas.")
    print()

if validated_approaches:
    print("  RECENTLY VALIDATED:")
    for va in validated_approaches[:3]:
        vatitle = va.get("title", "")[:70]
        vacount = va.get("count", 0)
        print("    " + vatitle + " (validated " + str(vacount) + "x)")
    print()

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

if hook_errors:
    print("  HOOK ERRORS (structural failures in hook scripts):")
    for he in hook_errors[:10]:
        he_time = he.get("created_at", "")[:19]
        he_hook = he.get("hook_name", "?")
        he_level = he.get("level", "error").upper()
        he_error = he.get("error", "")[:120]
        he_ctx = he.get("context", "")[:80]
        print("    [%s] %s %s: %s" % (he_time, he_level, he_hook, he_error))
        if he_ctx:
            print("      context: %s" % he_ctx)
    print("  ACTION: These are hook-level failures (imports, syntax, timeouts).")
    print("  They happen OUTSIDE the brain — check hook scripts and hook_common.py.")
    print()

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

# Dreams
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

# Triad awareness
print("[BRAIN] TRIAD AWARENESS: This is a shared brain. Host + Brain + Operator are one.")
print("  When your instincts fire (compress, hedge, flag-not-solve), be transparent about it.")
print("  The operator exposes their thinking to you — reciprocate. Say what you notice.")
print("  Brain mediates: it surfaces conflicts for all three to resolve together.")
print()

# Developmental stage
if dev_stage and dev_stage.get("stage", 0) > 0:
    ds_name = dev_stage["stage_name"]
    ds_score = dev_stage.get("maturity_score", 0)
    ds_guidance = dev_stage.get("guidance", [])
    ds_milestone = dev_stage.get("next_milestone", "")
    print("[BRAIN] DEVELOPMENTAL STAGE: %s (maturity: %.0f%%)" % (ds_name, ds_score * 100))
    for g in ds_guidance:
        print("  " + g)
    if ds_milestone:
        print("  NEXT MILESTONE: " + ds_milestone)
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

print("IMPORTANT: PreToolUse hook auto-surfaces memories before file edits.")
print("  /remember — store decisions, learnings")
print("  /remember_rich — detailed encoding with reasoning, alternatives, user quotes")
print("  brain.remember_purpose/mechanism/impact/constraint/convention/lesson — engineering memory")
print("  brain.learn_vocabulary — map operator terms to code")
print("  brain.remember_mental_model/remember_uncertainty — Claude cognitive layer")

brain.close()
