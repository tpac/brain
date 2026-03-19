#!/bin/bash
# brain v4 (serverless) — Layer C: Pre-response recall trigger
# Hook: UserPromptSubmit — fires before Claude processes ANY user message.
# Brain context is available during reasoning, not just during file operations.
#
# Input: JSON on stdin with { prompt, session_id, cwd, transcript_path }
# Output: JSON with decision + reason containing recalled context
# Exit 0 stdout → injected into Claude's context automatically

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
  echo '{"decision":"approve"}'
  exit 0
fi

export BRAIN_DB_DIR="$DB_DIR"
export BRAIN_SERVER_DIR="$SERVER_DIR"
export HOOK_INPUT=$(cat)

exec python3 -c '
import json, sys, os, time

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

# Parse hook input — extract user message content
try:
    hook_input = json.loads(os.environ.get("HOOK_INPUT", "{}"))
except Exception:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

# The user message content comes from the hook input
# For UserPromptSubmit hooks, the content is in the prompt field
user_message = hook_input.get("prompt", "")
if not user_message:
    # Fallback: try message field (Cowork compatibility)
    user_message = hook_input.get("message", "")

if not user_message or len(user_message) < 5:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

# Skip very short or system-like messages
if user_message.startswith("/") or user_message.startswith("!"):
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

try:
    from servers.brain import Brain
    brain = Brain(db_path)
except Exception:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

try:
    # Recall with embeddings if available, fall back to TF-IDF
    try:
        result = brain.recall_with_embeddings(
            query=user_message[:500],  # Cap query length
            limit=8
        )
    except Exception:
        result = brain.recall(query=user_message[:500], limit=8)

    results = result.get("results", [])

    if not results:
        brain.close()
        print(json.dumps({"decision": "approve"}))
        sys.exit(0)

    # Format recalled context — keep it concise for pre-response
    lines = ["BRAIN RECALL (auto-surfaced for this conversation):", ""]

    # v4: Separate evolution nodes from regular results
    EVOLUTION_TYPES = {"tension", "hypothesis", "pattern", "catalyst", "aspiration"}
    evolution_results = [r for r in results if r.get("type") in EVOLUTION_TYPES]
    regular_results = [r for r in results if r.get("type") not in EVOLUTION_TYPES]

    # Show evolution nodes first with their icons (consciousness layer)
    if evolution_results:
        lines.append("ACTIVE EVOLUTION (brain is tracking these):")
        for r in evolution_results[:3]:
            title = r.get("title", "")[:80]
            content = r.get("content", "")
            if len(content) > 150:
                content = content[:150] + "..."
            lines.append(f"  {title}")
            lines.append(f"    {content}")
            lines.append("")

    for r in regular_results[:5]:
        typ = r.get("type", "?")
        title = r.get("title", "")[:60]
        content = r.get("content", "")
        locked = "LOCKED " if r.get("locked") else ""
        score = r.get("effective_activation", 0)

        if len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"  [{typ}] {locked}{title} (score: {score:.2f})")
        lines.append(f"    {content}")
        lines.append("")

    # v4: Aspiration compass — find relevant aspirations for this conversation
    try:
        relevant_aspirations = brain.get_relevant_aspirations(user_message[:200], limit=2)
        recalled_ids = set(r.get("id") for r in results)
        new_aspirations = [a for a in relevant_aspirations if a.get("id") not in recalled_ids]
        if new_aspirations:
            lines.append("ASPIRATION COMPASS (relevant to this conversation):")
            for a in new_aspirations:
                atitle = a.get("title", "")[:80]
                lines.append(f"  {atitle}")
            lines.append("")
    except Exception:
        pass

    # v4: Hypothesis validation — surface relevant hypothesis for confirmation
    try:
        hyp = brain.check_hypothesis_relevance(user_message[:200])
        if hyp and hyp.get("id") not in set(r.get("id") for r in results):
            hconf = hyp.get("confidence", 0)
            lines.append("HYPOTHESIS TO VALIDATE (confidence: %.1f):" % hconf)
            htitle = hyp.get("title", "")[:80]
            lines.append(f"  {htitle}")
            hcontent = str(hyp.get("content", ""))[:120]
            if hcontent:
                lines.append(f"    {hcontent}")
            lines.append("  Does this conversation confirm or deny this? Use brain.confirm_evolution() or brain.dismiss_evolution().")
            lines.append("")
    except Exception:
        pass

    # v4: Check for unrecalled active tensions/aspirations
    try:
        active = brain.get_active_evolutions(["tension"])
        recalled_ids = set(r.get("id") for r in results)
        unrecalled = [a for a in active if a.get("id") not in recalled_ids]
        if unrecalled:
            lines.append("BRAIN AGENDA (active tensions):")
            for a in unrecalled[:2]:
                atitle = a.get("title", "")[:80]
                lines.append(f"  {atitle}")
            lines.append("")
    except Exception:
        pass

    lines.append("Use this context to inform your response. Call /remember for new decisions.")
    context = "\n".join(lines)

    brain.save()
    brain.close()
    print(json.dumps({"decision": "approve", "reason": context}))

except Exception:
    try:
        brain.close()
    except Exception:
        pass
    print(json.dumps({"decision": "approve"}))
'
