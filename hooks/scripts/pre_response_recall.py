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
            "[BRAIN ERROR] Recall failed: %s" % err_msg}))
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

        # Build the distillation prompt
        candidates_text = ""
        for c in candidates[:8]:
            neighbors_text = ""
            if c.get("neighbors"):
                neighbors_text = " | neighbors: " + ", ".join(
                    n.get("title", "")[:40] for n in c["neighbors"][:3])
            candidates_text += "[%s] %s (id:%s, conf:%.2f)%s\n  %s\n\n" % (
                c.get("type", "?"), c.get("title", "?"),
                c.get("id", "")[:12], c.get("confidence", 0),
                neighbors_text,
                (c.get("content") or "")[:300])

        distill_prompt = """You are the awareness layer of a persistent AI brain.
Distill these memory candidates into focused context for the main AI.

USER MESSAGE: %s

CANDIDATES:
%s

Rules:
- Only include what's DIRECTLY relevant to the user's message
- Preserve node IDs like (id:abc123) so the AI can reference them
- If a correction or rule applies, lead with it
- If nothing is relevant, return just the word EMPTY — no explanation, no commentary, no suggestions about what you'd need. Just EMPTY.
- Max 600 characters. Be surgical, like a colleague whispering context.
- If this seems like the start of a conversation, be more generous with context.
- NEVER add your own opinions or analysis. You are a filter, not an advisor.""" % (user_message[:500], candidates_text)

        api_resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
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

        if distilled == "EMPTY" or not distilled:
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
