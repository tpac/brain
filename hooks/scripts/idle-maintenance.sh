#!/bin/bash
# brain — idle maintenance hook
# Runs during idle prompts: dream, consolidate (includes auto-discovery), self-reflect.
# Surfaces any auto-discovered evolutions for consciousness.

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

if [ -z "$DB_DIR" ]; then
  exit 0
fi

export BRAIN_DB_DIR="$DB_DIR"
export BRAIN_SERVER_DIR="$SERVER_DIR"

exec python3 -c '
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

    output_lines = []

    # 1. Dream — random walks, surprise scoring, intuition creation
    try:
        dream_result = brain.dream()
        dream_count = dream_result.get("count", 0)
        if dream_count > 0:
            output_lines.append("DREAM: %d dream(s) generated" % dream_count)
            for d in dream_result.get("dreams", [])[:2]:
                dtitle = d.get("title", "untitled")
                output_lines.append("  - " + dtitle)
    except Exception as e:
        output_lines.append("DREAM ERROR: %s" % e)

    # 2. Consolidate — stability boosts, bridging, and auto-evolution discovery
    try:
        consolidate_result = brain.consolidate()
        cons_count = consolidate_result.get("consolidated", 0)
        output_lines.append("CONSOLIDATE: %d nodes boosted" % cons_count)

        # Surface auto-discovered evolutions
        discoveries = consolidate_result.get("discoveries", {})
        total = discoveries.get("total", 0)
        if total > 0:
            output_lines.append("\nBRAIN DISCOVERED (%d evolution(s)):" % total)

            # Get the actual discovery details from a fresh call
            # (consolidate already created them, just fetch active ones)
            active = brain.get_active_evolutions()
            auto_discovered = [e for e in active if "auto-discovered" in (e.get("content") or "")]

            for evo in auto_discovered[:5]:
                etype = evo["type"].upper()
                title = evo["title"]
                eid = evo["id"]
                # Extract action from content
                content = evo.get("content", "")
                action = ""
                if "ACTION:" in content:
                    action = content.split("ACTION:")[-1].strip()

                output_lines.append("  %s: %s" % (etype, title))
                if action:
                    output_lines.append("    -> " + action)
                eid_short = eid[:8]
                output_lines.append("    [confirm: brain.confirm_evolution(\"%s...\")]" % eid_short)
                output_lines.append("    [dismiss: brain.dismiss_evolution(\"%s...\")]" % eid_short)
    except Exception as e:
        output_lines.append("CONSOLIDATE ERROR: %s" % e)

    # 3. Self-healing — resolve discoveries, tune parameters, clean graph
    try:
        heal_result = brain.auto_heal()
        resolved = heal_result.get("resolved", [])
        tuned = heal_result.get("tuned", [])
        cleaned = heal_result.get("cleaned", {})

        if resolved:
            output_lines.append("\nBRAIN HEALED (%d action(s)):" % len(resolved))
            for r in resolved[:5]:
                action = r.get("action", "unknown")
                if action == "merge_duplicate":
                    archived_name = r.get("archived", "")
                    kept_name = r.get("kept", "")
                    sim_val = r.get("sim", "")
                    output_lines.append("  MERGED: \"%s\" into \"%s\" (sim %s)" % (archived_name, kept_name, sim_val))
                elif action == "auto_lock":
                    lock_title = r.get("title", "")
                    ac = r.get("access_count", 0)
                    output_lines.append("  LOCKED: \"%s\" (%d accesses)" % (lock_title, ac))
                else:
                    output_lines.append("  %s: %s" % (action, r))

        if tuned:
            output_lines.append("\nBRAIN TUNED (%d parameter(s)):" % len(tuned))
            for t in tuned[:5]:
                tparam = t.get("param", "")
                treason = t.get("reason", "")
                output_lines.append("  %s: %s" % (tparam, treason))

        archived = cleaned.get("archived", 0)
        edges_created = cleaned.get("edges_created", 0)
        edges_normalized = cleaned.get("edges_normalized", 0)
        merged = cleaned.get("merged", 0)
        locked = cleaned.get("locked", 0)
        if any([archived, edges_created, edges_normalized, merged, locked]):
            parts = []
            if merged: parts.append("%d merged" % merged)
            if locked: parts.append("%d locked" % locked)
            if archived: parts.append("%d archived" % archived)
            if edges_created: parts.append("%d edges created" % edges_created)
            if edges_normalized: parts.append("%d edges normalized" % edges_normalized)
            output_lines.append("  HYGIENE: " + ", ".join(parts))
    except Exception as e:
        output_lines.append("HEAL ERROR: %s" % e)

    # 3b. v5: Standalone auto-tune (v5-specific parameter adjustments)
    try:
        tune_result = brain.auto_tune()
        tuned = tune_result.get("tuned", [])
        if tuned:
            output_lines.append("\nBRAIN AUTO-TUNED (%d parameter(s)):" % len(tuned))
            for t in tuned[:5]:
                param = t.get("param", "")
                reason = t.get("reason", t.get("note", ""))
                output_lines.append("  %s: %s" % (param, reason))
    except Exception:
        pass

    # 4. Reflection prompts — surface questions for the host to encode learnings
    try:
        reflections = brain.prompt_reflection()
        if reflections:
            output_lines.append("")
            output_lines.append("REFLECT (transferable insights from this session?):")
            for r in reflections[:3]:
                output_lines.append("  " + r)
            output_lines.append("")
    except Exception:
        pass

    # 5. Self-reflection — performance/capability/interaction nodes
    try:
        reflection = brain.auto_generate_self_reflection()
        ref_count = sum(1 for v in reflection.values() if v)
        if ref_count > 0:
            output_lines.append("SELF-REFLECTION: %d reflection(s) generated" % ref_count)
    except Exception as e:
        pass  # self-reflection is optional

    # 5. v5: Backfill content summaries for nodes missing them
    try:
        backfill = brain.backfill_summaries(batch_size=50)
        bf_count = backfill.get("updated", 0)
        if bf_count > 0:
            output_lines.append("SUMMARIES: backfilled %d nodes" % bf_count)
    except Exception:
        pass

    brain.save()

    if output_lines:
        print("\n".join(output_lines))

except Exception as e:
    print("IDLE MAINTENANCE ERROR: %s" % e, file=sys.stderr)
'
