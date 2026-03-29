"""Pre-response recall — surfaces brain context before Claude responds.
Fires on UserPromptSubmit. Output: JSON {"additionalContext":"..."}.

Flow:
1. Command hook calls daemon → gets raw candidates
2. If candidates found, calls Anthropic API (Haiku) → distills to focused context
3. Returns distilled context as additionalContext
"""
import sys, os, json, time

# Load .env from project root (API keys)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_env_path = os.path.join(_project_root, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _key, _val = _k.strip(), _v.strip()
                if _val:  # Only set if value is non-empty
                    os.environ[_key] = _val

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import (get_hook_input, daemon_available, daemon_call_raw,
                         daemon_unavailable_error, brain_debug, log_hook_output)

APPROVE = json.dumps({"decision": "approve"})

hook_input = get_hook_input()
user_message = hook_input.get("prompt", "") or hook_input.get("message", "")

# Skip short, slash, or bang messages
if not user_message or len(user_message) < 5 or user_message.startswith("/") or user_message.startswith("!"):
    brain_debug("recall: skipped (short/slash/bang)")
    print(APPROVE)
    sys.exit(0)

t0 = time.time()
try:
    if not daemon_available():
        err = daemon_unavailable_error("recall")
        log_hook_output("recall", output_text="(daemon unavailable)", user_prompt=user_message)
        print(json.dumps({"additionalContext": err}))
        sys.exit(0)

    # Step 1: Get raw candidates from daemon
    resp = daemon_call_raw("hook_recall", {
        "prompt": hook_input.get("prompt", ""),
        "message": hook_input.get("message", ""),
    }, timeout=12.0)

    if not resp.get("ok"):
        err_msg = resp.get("error", "unknown error")
        log_hook_output("recall", output_text="(daemon error: %s)" % err_msg, user_prompt=user_message)
        print(json.dumps({"additionalContext":
            "[BRAIN]\n⚠️ RECALL FAILED: %s\nThe brain could not search for relevant memories. Operating without context.\n[/BRAIN]" % err_msg}))
        sys.exit(0)

    # Extract candidates from the file the daemon wrote
    result = resp.get("result", {})
    session_id = hook_input.get("session_id", "unknown")

    # Read candidates file written by daemon
    # The daemon returns its session_id in the response so we find the right file
    daemon_session = result.get("session_id", "")
    if not daemon_session:
        # Fallback: try Claude Code's session_id
        daemon_session = session_id
    candidates_path = "/tmp/brain-%s-recall-candidates.json" % daemon_session
    candidates_data = None
    if os.path.exists(candidates_path) and os.path.getsize(candidates_path) > 0:
        try:
            with open(candidates_path) as f:
                candidates_data = json.load(f)
        except Exception:
            pass

    if not candidates_data or not candidates_data.get("candidates"):
        # No candidates — nothing to distill
        log_hook_output("recall", output_text="(no candidates)", user_prompt=user_message)
        print(APPROVE)
        sys.exit(0)

    candidates = candidates_data["candidates"]
    latency_fetch = (time.time() - t0) * 1000

    # Step 2: Distill candidates via Anthropic API
    t1 = time.time()
    try:
        import anthropic
        client = anthropic.Anthropic()

        # Build rich candidate text with graph neighborhoods
        candidates_text = ""
        relevant_count = 0
        for c in candidates[:8]:
            # Core node info
            locked = "LOCKED " if c.get("locked") else ""
            candidates_text += "[%s] %s%s (id:%s, conf:%.2f, revised:%s, created:%s)\n" % (
                c.get("type", "?"), locked, c.get("title", "?"),
                c.get("id", "")[:16], c.get("confidence") or 0,
                c.get("revised_at") or "never",
                str(c.get("created_at") or "")[:10])
            candidates_text += "  %s\n" % (c.get("content") or "")[:500]

            # Graph neighborhood (degree 1 — rich fields)
            graph = c.get("_graph", {})
            d1 = graph.get("degree_1", [])
            for nb in d1[:3]:
                locked_nb = "LOCKED " if nb.get("locked") else ""
                summary_nb = nb.get("content_summary") or ""
                candidates_text += "  → %s: %s\"%s\" (%s, id:%s, conf:%.2f, revised:%s)" % (
                    nb.get("relation", "related"), locked_nb,
                    nb.get("title", "")[:60],
                    nb.get("type", "?"), nb.get("id", "")[:12],
                    nb.get("confidence") or 0,
                    nb.get("revised_at") or "never")
                if summary_nb:
                    candidates_text += "\n      %s" % summary_nb[:150]
                candidates_text += "\n"

            # Degree 2 as breadcrumbs with type
            d2 = graph.get("degree_2", [])
            if d2:
                d2_items = ", ".join("\"%s\" (%s, id:%s)" % (
                    n.get("title", "")[:35], n.get("type", "?"), n.get("id", "")[:8]) for n in d2[:3])
                candidates_text += "  →→ %s\n" % d2_items

            candidates_text += "\n"
            if (c.get("confidence") or 0) > 0.3:
                relevant_count += 1

        # Dynamic budget based on query complexity
        query_len = len(user_message)
        budget = 400 + min(800, relevant_count * 100 + (100 if query_len > 100 else 0))
        max_tokens = min(500, budget // 2)

        distill_prompt = """You are the awareness layer of a persistent AI brain.
Distill these memory candidates into focused context for the main AI.

USER MESSAGE: %s

CANDIDATES:
%s

Rules:
- Only include what's DIRECTLY relevant to the user's message
- Preserve node IDs like (id:abc123) so the AI can pull full details
- Include graph connections when they add context (→ related nodes)
- If a correction or rule applies, lead with it
- If nothing is relevant, return just the word EMPTY. No explanation.
- Max %d characters. Be surgical, like a colleague whispering context.
- If this seems like the start of a conversation, be more generous.
- NEVER add your own opinions or analysis. You are a filter, not an advisor.""" % (user_message[:500], candidates_text, budget)

        api_resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": distill_prompt}]
        )
        distilled = api_resp.content[0].text.strip()
        latency_distill = (time.time() - t1) * 1000

        # Store full prompt details for dashboard debugging
        _prompt_details = json.dumps({
            "distill_prompt": distill_prompt,
            "model": "claude-haiku-4-5",
            "candidates_count": len(candidates),
            "latency_fetch_ms": round(latency_fetch),
            "latency_distill_ms": round(latency_distill),
        })

        # Strip Haiku's opinions — if it starts with EMPTY, treat as empty
        if not distilled or distilled.startswith("EMPTY") or distilled.upper().startswith("EMPTY"):
            log_hook_output("recall", output_text="(distilled: empty)", user_prompt=user_message, metadata=_prompt_details)
            print(APPROVE)
        else:
            context = "[BRAIN]\n%s\n[/BRAIN]" % distilled
            log_hook_output("recall", output_text=context, user_prompt=user_message, metadata=_prompt_details)
            brain_debug("recall: distilled %d candidates → %d chars in %dms (fetch: %dms)" % (
                len(candidates), len(context), latency_distill, latency_fetch))
            print(json.dumps({"additionalContext": context}))

    except ImportError:
        # anthropic SDK not installed — fall back to raw candidates summary
        brain_debug("recall: anthropic SDK not available, returning raw summary")
        summary = "Brain found %d candidates but LLM distillation unavailable." % len(candidates)
        for c in candidates[:3]:
            summary += "\n- [%s] %s (id:%s)" % (c.get("type", "?"), c.get("title", "?")[:60], c.get("id", "")[:12])
        context = "[BRAIN]\n%s\n[/BRAIN]" % summary
        log_hook_output("recall", output_text=context, user_prompt=user_message)
        print(json.dumps({"additionalContext": context}))

    except Exception as e:
        # API call failed — log error, return raw summary as fallback
        brain_debug("recall: distill failed: %s" % e)
        log_hook_output("recall", output_text="(distill error: %s)" % e, user_prompt=user_message)
        summary = "Brain recall found %d candidates (distillation failed: %s)" % (len(candidates), str(e)[:50])
        for c in candidates[:3]:
            summary += "\n- [%s] %s (id:%s)" % (c.get("type", "?"), c.get("title", "?")[:60], c.get("id", "")[:12])
        print(json.dumps({"additionalContext": "[BRAIN]\n%s\n[/BRAIN]" % summary}))

except Exception as e:
    log_hook_output("recall", output_text="(exception) %s" % e, user_prompt=user_message)
    print(APPROVE)
