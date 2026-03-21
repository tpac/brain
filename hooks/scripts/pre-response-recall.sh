#!/bin/bash
# brain v15 (serverless) — Layer C: Pre-response recall trigger
# Fires before Claude responds to ANY user message (not just file edits).
# This is the key architectural improvement: brain context is available
# during reasoning, not just during file operations.
#
# Input: JSON on stdin with { prompt, session_id, cwd, transcript_path }
# Output: JSON with decision + reason containing recalled context
# Exit 0 stdout → injected into Claude's context automatically

source "$(dirname "$0")/resolve-brain-db.sh"

if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  echo '{"decision":"approve"}'
  exit 0
fi
export HOOK_INPUT=$(cat)

# ── Try daemon first (fast path: ~70ms vs ~2000ms) ──
SOCKET_PATH="/tmp/brain-daemon-$(id -u).sock"
if [ -S "$SOCKET_PATH" ]; then
  DAEMON_RESULT=$(python3 -c '
import json, sys, os, socket, re

hook_input = json.loads(os.environ.get("HOOK_INPUT", "{}"))
user_message = hook_input.get("prompt", "") or hook_input.get("message", "")
if not user_message or len(user_message) < 5 or user_message.startswith("/") or user_message.startswith("!"):
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(10.0)
try:
    sock.connect("'"$SOCKET_PATH"'")

    # v5.2: Store last user message for operator voice capture
    try:
        store_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        store_sock.settimeout(3.0)
        store_sock.connect("'"$SOCKET_PATH"'")
        store_msg = json.dumps({"cmd": "set_config", "args": {"key": "last_user_message", "value": user_message[:500]}}) + "\n"
        store_sock.sendall(store_msg.encode())
        store_sock.recv(4096)
        store_sock.close()
    except Exception:
        pass

    # Vocab expansion
    vocab_msg = json.dumps({"cmd": "vocab_check", "args": {"message": user_message[:500]}}) + "\n"
    sock.sendall(vocab_msg.encode())
    data = b""
    while b"\n" not in data:
        chunk = sock.recv(65536)
        if not chunk: break
        data += chunk
    vocab_resp = json.loads(data.decode().strip())
    sock.close()

    expansions = vocab_resp.get("result", {}).get("expansions", []) if vocab_resp.get("ok") else []
    enriched = user_message[:500]
    if expansions:
        enriched += " " + " ".join(expansions)[:200]

    # Recall
    sock2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock2.settimeout(10.0)
    sock2.connect("'"$SOCKET_PATH"'")
    recall_msg = json.dumps({"cmd": "recall", "args": {"query": enriched, "limit": 8}}) + "\n"
    sock2.sendall(recall_msg.encode())
    data = b""
    while b"\n" not in data:
        chunk = sock2.recv(65536)
        if not chunk: break
        data += chunk
    recall_resp = json.loads(data.decode().strip())
    sock2.close()

    if not recall_resp.get("ok"):
        raise Exception("Daemon recall failed")

    results = recall_resp.get("result", {}).get("results", [])
    if not results:
        print(json.dumps({"decision": "approve"}))
        sys.exit(0)

    lines = ["BRAIN RECALL (auto-surfaced for this conversation):", ""]
    EVOLUTION_TYPES = {"tension", "hypothesis", "pattern", "catalyst", "aspiration"}
    evolution_results = [r for r in results if r.get("type") in EVOLUTION_TYPES]
    regular_results = [r for r in results if r.get("type") not in EVOLUTION_TYPES]

    if evolution_results:
        lines.append("ACTIVE EVOLUTION (brain is tracking these):")
        for r in evolution_results[:3]:
            title = r.get("title", "")[:80]
            content = r.get("content", "")
            if len(content) > 150: content = content[:150] + "..."
            lines.append("  " + title)
            lines.append("    " + content)
            lines.append("")

    for r in regular_results[:5]:
        typ = r.get("type", "?")
        title = r.get("title", "")[:60]
        content = r.get("content", "")
        locked = "LOCKED " if r.get("locked") else ""
        score = r.get("effective_activation", 0)
        if len(content) > 300: content = content[:300] + "..."
        lines.append("  [%s] %s%s (score: %.2f)" % (typ, locked, title, score))
        lines.append("    " + content)
        lines.append("")

    # v6: Host instinct awareness — the brain as prefrontal cortex
    try:
        sock3 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock3.settimeout(5.0)
        sock3.connect("'"$SOCKET_PATH"'")
        instinct_msg = json.dumps({"cmd": "instinct_check", "args": {"message": user_message[:500]}}) + "\n"
        sock3.sendall(instinct_msg.encode())
        data = b""
        while b"\n" not in data:
            chunk = sock3.recv(65536)
            if not chunk: break
            data += chunk
        instinct_resp = json.loads(data.decode().strip())
        sock3.close()
        nudge = instinct_resp.get("result", {}).get("nudge") if instinct_resp.get("ok") else None
        if nudge:
            lines.insert(0, nudge)
            lines.insert(1, "")
    except Exception:
        pass

    lines.append("Use this context to inform your response. Call /remember for new decisions.")
    context = "\n".join(lines)
    print(json.dumps({"additionalContext": context}))

except Exception as e:
    # Daemon failed — signal to fall through to direct Python
    print("DAEMON_FALLBACK")
    sys.exit(1)
' 2>/dev/null)

  if [ $? -eq 0 ] && [ -n "$DAEMON_RESULT" ] && [ "$DAEMON_RESULT" != "DAEMON_FALLBACK" ]; then
    echo "$DAEMON_RESULT"
    exit 0
  fi
  # Fall through to direct Python if daemon failed
fi

# ── Direct Python fallback (cold path: ~2000ms) ──
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

# v5.2: Store last user message for automatic operator voice capture.
# remember_rich() can auto-populate user_raw_quote from this.
try:
    brain.set_config("last_user_message", user_message[:500])
except Exception:
    pass

try:
    # v5: Resolve vocabulary in user message to enrich recall query
    # If user says "fix the recall hook", expand to include actual file/function names
    vocab_expansions = []
    try:
        import re as _re
        # Extract candidate terms (2-4 word phrases, hyphenated terms)
        candidates = set()
        candidates.update(t.strip().lower() for t in _re.findall(r"\bthe\s+([\w][\w\s-]{2,25})\b", user_message, _re.IGNORECASE))
        candidates.update(t.strip().lower() for t in _re.findall(r"\b([\w]+-[\w]+(?:-[\w]+)?)\b", user_message) if len(t) > 4)
        for term in candidates:
            resolved = brain.resolve_vocabulary(term)
            if resolved:
                if resolved.get("ambiguous"):
                    # Multiple mappings — include all targets
                    for m in resolved.get("mappings", []):
                        content = m.get("content", "")
                        vocab_expansions.append(content)
                else:
                    content = resolved.get("content", "")
                    vocab_expansions.append(content)
    except Exception:
        pass

    # Build enriched query: original message + vocabulary expansions
    enriched_query = user_message[:500]
    if vocab_expansions:
        expansion_text = " ".join(vocab_expansions)[:200]
        enriched_query = enriched_query + " " + expansion_text

    # Recall with embeddings if available, fall back to TF-IDF
    try:
        result = brain.recall_with_embeddings(
            query=enriched_query,
            limit=8
        )
    except Exception:
        result = brain.recall(query=enriched_query, limit=8)

    results = result.get("results", [])

    # v5.1: Segment boundary detection — uses the query embedding already computed
    segment_note = None
    try:
        query_emb = result.get("_query_embedding")
        if query_emb:
            seg = brain.check_segment_boundary(query_emb)
            if seg.get("is_boundary"):
                segment_note = "--- CONTEXT SHIFT (segment %d, sim=%.2f) ---" % (
                    seg["segment_id"], seg["similarity"])
            # Track recalled nodes in current segment
            for r in results:
                brain.add_to_segment(r.get("id", ""))
    except Exception:
        pass

    if not results:
        brain.save()
        brain.close()
        print(json.dumps({"decision": "approve"}))
        sys.exit(0)

    # v6: Priming — check if this query touches an active concern
    priming_note = None
    try:
        primes = brain.get_active_primes()
        if primes:
            match = brain.check_priming(user_message[:500], primes)
            if match:
                priming_note = "🎯 PRIMED TOPIC: \"%s\" (source: %s, sim: %.2f) — this conversation touches an active concern." % (
                    match["topic"][:80], match["source"], match["similarity"])
    except Exception:
        pass

    # Format recalled context — keep it concise for pre-response
    lines = ["BRAIN RECALL (auto-surfaced for this conversation):", ""]

    # Show segment boundary if detected
    if segment_note:
        lines.append(segment_note)
        lines.append("")

    # Show priming match first if present
    if priming_note:
        lines.append(priming_note)
        lines.append("")

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
        tiered = r.get("_tiered", "full")

        if tiered == "full" and len(content) > 300:
            content = content[:300] + "..."
        elif tiered != "full" and len(content) > 200:
            content = content[:200] + "..."

        tier_tag = "" if tiered == "full" else " [summary]"
        # v6: Show confidence qualifier for uncertain nodes
        conf = r.get("confidence") or r.get("_brain_to_host", {}).get("confidence", 1.0)
        conf_tag = ""
        if conf < 0.4:
            conf_tag = " ⚠️ LOW CONFIDENCE"
        elif conf < 0.6:
            conf_tag = " [uncertain]"
        # v5.1: Temporal freshness — how old is this info?
        freshness = r.get("_brain_to_host", {}).get("freshness", "")
        fresh_tag = ""
        if freshness == "this_month":
            fresh_tag = " [~weeks old]"
        elif freshness == "older":
            fresh_tag = " [old — verify still valid]"
        lines.append(f"  [{typ}] {locked}{title} (score: {score:.2f}){tier_tag}{conf_tag}{fresh_tag}")
        lines.append(f"    {content}")

        # v5: Show metadata for top results (reasoning, user quotes)
        meta = r.get("_metadata")
        if meta:
            if meta.get("reasoning"):
                reasoning_preview = meta["reasoning"][:120]
                if len(meta["reasoning"]) > 120:
                    reasoning_preview += "..."
                lines.append(f"    Reasoning: {reasoning_preview}")
            if meta.get("user_raw_quote"):
                lines.append(f"    User said: \"{meta['user_raw_quote'][:100]}\"")

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

    # v6: Host instinct awareness — the brain as prefrontal cortex
    try:
        nudge = brain.get_instinct_check(user_message[:500])
        if nudge:
            lines.insert(0, nudge)
            lines.insert(1, "")
    except Exception:
        pass

    lines.append("Use this context to inform your response. Call /remember for new decisions.")
    context = "\n".join(lines)

    brain.save()
    brain.close()
    print(json.dumps({"additionalContext": context}))

except Exception:
    try:
        brain.close()
    except Exception:
        pass
    print(json.dumps({"decision": "approve"}))
'
