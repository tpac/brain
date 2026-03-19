#!/bin/bash
# brain v4 — Idle maintenance + brain voice
# Hook: Notification/idle_prompt — fires when Claude is idle (waiting for user).
# This is when the brain thinks out loud: run maintenance AND surface insights.
#
# Operations:
#   1. dream() — random-walk association discovery
#   2. smart_prune() — decay weak edges by half-life
#   3. backfill_embeddings() — ensure all nodes have embeddings
#   4. consolidate() — strengthen frequent memories, detect bridges
#   5. get_consciousness_signals() — surface brain state to user
#
# Input: JSON on stdin with { session_id, notification_type, message }
# Output: stdout text surfaced to user (brain's voice during downtime)

source "$(dirname "$0")/resolve-brain-db.sh"

if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  exit 0
fi

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

output_lines = []

try:
    # ── 1. Dream: random-walk association discovery ──
    try:
        dream_result = brain.dream(session_id="idle")
        insights = dream_result.get("insights", [])
        if insights:
            output_lines.append("BRAIN DREAM (idle associations):")
            for insight in insights[:3]:
                title = insight.get("title", "")[:80]
                content = str(insight.get("content", ""))[:120]
                output_lines.append(f"  💭 {title}")
                if content:
                    output_lines.append(f"    {content}")
            output_lines.append("")
    except Exception:
        pass

    # ── 2. Smart prune: decay weak edges ──
    try:
        brain.smart_prune()
    except Exception:
        pass

    # ── 3. Backfill embeddings ──
    try:
        backfill_count = brain.backfill_embeddings(batch_size=20)
        if backfill_count and backfill_count > 0:
            output_lines.append(f"BRAIN MAINTENANCE: Backfilled {backfill_count} embeddings")
            output_lines.append("")
    except Exception:
        pass

    # ── 4. Consolidate: strengthen frequent, detect bridges ──
    try:
        consolidation = brain.consolidate()
        bridges = consolidation.get("bridges_created", 0) if consolidation else 0
        if bridges > 0:
            output_lines.append(f"BRAIN CONSOLIDATION: Discovered {bridges} new bridges between knowledge clusters")
            output_lines.append("")
    except Exception:
        pass

    # ── 5. Surface consciousness signals ──
    try:
        signals = brain.get_consciousness_signals()

        # Fading knowledge warnings
        fading = signals.get("fading", [])
        if fading:
            output_lines.append("FADING KNOWLEDGE (not accessed in 25+ days):")
            for node in fading[:3]:
                title = node.get("title", "")[:80]
                output_lines.append(f"  ⏳ {title}")
            output_lines.append("")

        # Active evolutions
        evolutions = signals.get("evolutions", [])
        if evolutions:
            output_lines.append("ACTIVE EVOLUTIONS:")
            for evo in evolutions[:3]:
                title = evo.get("title", "")[:80]
                etype = evo.get("type", "")
                output_lines.append(f"  {title} [{etype}]")
            output_lines.append("")

        # Fluid personal nodes ("still true?")
        fluid = signals.get("fluid_personal", [])
        if fluid:
            output_lines.append("FLUID PERSONAL (still true?):")
            for node in fluid[:2]:
                title = node.get("title", "")[:80]
                output_lines.append(f"  🔄 {title}")
            output_lines.append("")

        # Encoding gap warning
        encoding_gap = signals.get("encoding_gap")
        if encoding_gap:
            output_lines.append(f"ENCODING GAP: {encoding_gap}")
            output_lines.append("")

        # Recent encodings — what the brain logged this session
        recent = signals.get("recent_encodings", [])
        if recent:
            output_lines.append(f"BRAIN LOGGED THIS SESSION ({len(recent)} nodes):")
            for node in recent[:6]:
                nid = node.get("id", "")[:12]
                ntype = node.get("type", "")
                title = node.get("title", "")[:70]
                locked = " LOCKED" if node.get("locked") else ""
                output_lines.append(f"  [{ntype}]{locked} {title}  (id: {nid})")
            if len(recent) > 6:
                output_lines.append(f"  ... and {len(recent) - 6} more")
            output_lines.append("")
            output_lines.append("  Anything wrong? Use brain.update(node_id, ...) to fix or brain.conn.execute('DELETE FROM nodes WHERE id=?', (id,)) to remove.")
            output_lines.append("")

    except Exception:
        pass

    brain.save()
    brain.close()

    if output_lines:
        print("\n".join(output_lines))

except Exception as e:
    try:
        brain.close()
    except Exception:
        pass
' 2>/dev/null

exit 0
