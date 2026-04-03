"""Encoding Agent v3 — LLM-powered brain encoding via Sonnet API.

Called by the daemon's Stop hook on every 5th stop event.
Gathers conversation timeline with pre-attached recall,
calls Sonnet with brain tools, dispatches tool calls directly.

v3 changes from v2:
  - Timeline format with pre-attached recall per turn (not flat messages)
  - Session-scoped encoding journal (not global overwritten state)
  - Reduced tool set: 8 tools (remember, remember_batch, revise, connect,
    recall, find_node_by_title, get_node, record_divergence, learn_vocabulary)
  - remember() returns related_nodes (recall-on-create)
  - Prompt optimized for 2-3 rounds, focused nodes, batch operations

Uses:
  - brain_mcp.TOOLS for tool schemas (single source of truth)
  - pipeline_contract.ENCODING_AGENT for limits
  - contract.generate_field_summary for field documentation
"""

import os
import json
import time


def run_encoding(brain, dispatch_fn, counter, log_fn=None):
    """Run the encoding agent.

    Args:
        brain: Brain instance (direct access, not via TCP)
        dispatch_fn: function(cmd, args) for tool calls
        counter: Current stop counter value
        log_fn: Optional logging function

    Returns:
        dict with encoding results summary
    """
    def _log(msg):
        print("[encoding-agent] %s" % msg, flush=True)
        if log_fn:
            log_fn("Encoding agent: %s" % msg)

    import anthropic
    from .pipeline_contract import ENCODING_AGENT

    _t0 = time.time()
    profile = []

    def _step(name):
        profile.append((name, int((time.time() - _t0) * 1000)))

    _load_env()
    _step("env_loaded")

    try:
        client = anthropic.Anthropic()
    except Exception as e:
        brain._log_error('encoding_agent_api', e, 'Cannot create Anthropic client')
        return {"error": str(e)}
    _step("api_client")

    session_id = brain.get_config('session_id', 'unknown')

    # 1. Gather messages with pre-attached recall
    messages = _gather_messages(brain, session_id)
    _step("messages(%d)" % len(messages))
    if not messages:
        _log("no messages, skipping")
        return {"skipped": True, "reason": "no messages"}

    # 2. Build prompt (no independent recall — timeline has pre-attached recall)
    system_prompt = _build_system_prompt()
    user_content = _build_user_content(brain, messages, counter, session_id)
    _step("prompt(%d chars)" % len(user_content))

    # 3. Get tool schemas
    tools = _get_tool_schemas()
    _step("tools(%d)" % len(tools))

    # 4. Call Sonnet
    _log("calling Sonnet with %d tools, %d chars context..." % (len(tools), len(user_content)))
    _log("PROFILE so far: %s" % " → ".join("%s:%dms" % (n, t) for n, t in profile))

    try:
        api_messages = [{"role": "user", "content": user_content}]
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=ENCODING_AGENT['max_tokens'],
            system=system_prompt, messages=api_messages, tools=tools)

        _step("sonnet_r0")
        actions = []
        rounds = 0
        max_rounds = ENCODING_AGENT.get('max_rounds', 5)

        for rounds in range(max_rounds):
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            tool_results = []
            for tu in tool_uses:
                result = dispatch_fn(tu.name, tu.input)
                from . import brain_mcp
                if result.get("ok"):
                    result_text = brain_mcp._format_result(tu.name, result.get("result", {}))
                else:
                    result_text = "ERROR: %s" % result.get("error", "Unknown")
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tu.id,
                    "content": result_text,
                })
                action_summary = tu.input.get("title", tu.input.get("query",
                    tu.input.get("node_id", "")))[:60]
                actions.append({"tool": tu.name, "summary": action_summary})
                _log("  [%s] %s" % (tu.name, action_summary))

            api_messages.append({"role": "assistant", "content": [
                {"type": b.type, **({"text": b.text} if b.type == "text" else
                                    {"id": b.id, "name": b.name, "input": b.input})}
                for b in response.content]})
            api_messages.append({"role": "user", "content": tool_results})
            response = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=ENCODING_AGENT['max_tokens'],
                system=system_prompt, messages=api_messages, tools=tools)
            _step("sonnet_r%d" % (rounds + 1))

        # Save encoding journal (session-scoped, cumulative)
        final_text = "".join(b.text for b in response.content if b.type == "text")
        _save_journal(brain, session_id, counter, final_text)

        # Extract and store session context for recall judge (Layer 2)
        _save_session_context(brain, final_text)

        # Surface questions to operator via signal queue
        if final_text and '?' in final_text:
            try:
                from .dal_signal_queue import SignalQueueDAL
                sq = SignalQueueDAL(brain.logs_conn)
                sq.produce(
                    producer='encoding_agent',
                    signal_type='encoding_question',
                    priority=0.7,
                    content=final_text[:500],
                    ttl_seconds=86400,
                )
                brain.logs_conn.commit()
            except Exception as _e:
                print('[encoding-agent] ERROR surfacing question to signal queue: %s' % _e, flush=True)

        brain.save()
        _step("saved")
        profile_str = " → ".join("%s:%dms" % (n, t) for n, t in profile)
        _log("done. %d rounds, %d actions. PROFILE: %s" % (rounds + 1, len(actions), profile_str))
        return {"rounds": rounds + 1, "actions": len(actions), "profile": profile}

    except Exception as e:
        _step("FAILED")
        profile_str = " → ".join("%s:%dms" % (n, t) for n, t in profile)
        brain._log_error('encoding_agent_sonnet', e, 'Sonnet API call failed. PROFILE: %s' % profile_str)
        _log("FAILED: %s PROFILE: %s" % (e, profile_str))
        return {"error": str(e), "profile": profile}


# ── Helpers ──

def _load_env():
    """Load .env file for API key."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())


def _gather_messages(brain, session_id):
    """Fetch recent messages from message_stream with pre-attached recall."""
    from .pipeline_contract import ENCODING_AGENT
    try:
        rows = brain.logs_conn.execute(
            "SELECT id, role, content, signal_type, timestamp, recalled_raw, judge_output "
            "FROM message_stream WHERE session_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (session_id, ENCODING_AGENT['max_messages'])
        ).fetchall()
        return [{"id": r[0], "role": r[1],
                 "content": (r[2] or "")[:ENCODING_AGENT['message_content_limit']],
                 "signal": r[3], "timestamp": r[4],
                 "recalled_raw": r[5], "judge_output": r[6]}
                for r in reversed(rows)]
    except Exception as e:
        brain._log_error('encoding_agent_messages', e, 'Failed to fetch messages')
        return []


def _build_system_prompt():
    """Load v3 encoding agent prompt + contract field summary."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(project_dir, 'hooks', 'prompts', 'encoding-agent-v3.md')
    try:
        with open(prompt_path) as f:
            prompt = f.read()
    except Exception:
        prompt = "You are the encoding agent. Encode focused nodes. Batch operations. 2-3 rounds."
    try:
        from .contract import generate_field_summary
        prompt += "\n\n## Available Fields (from contract)\n\n" + generate_field_summary()
    except Exception as _e:
        print('[encoding-agent] WARNING: could not load field summary: %s' % _e, flush=True)
    return prompt


def _build_user_content(brain, messages, counter, session_id):
    """Assemble the v3.2 encoding prompt: node catalog + timeline with references.

    Structure:
    1. Encoding journal (what previous runs did)
    2. Session context (running summary)
    3. Node catalog: all judge-surfaced nodes, full rich metadata, deduplicated
    4. Timeline: conversation turns with node references (IDs only, not repeated)
    """
    from .pipeline_contract import ENCODING_AGENT, build_encoder_node_catalog
    import re

    # Encoding journal (session-scoped, cumulative)
    journal_key = 'encoding_journal_%s' % session_id
    journal = brain.get_config(journal_key, '') or 'First run — no previous encoding in this session.'

    # Build node catalog from all judge outputs (deduplicated, full metadata)
    judge_outputs = [m.get("judge_output") for m in messages if m.get("role") == "user"]
    node_catalog, cataloged_ids = build_encoder_node_catalog(judge_outputs, brain.conn)

    # Build conversation timeline with node references (not full nodes)
    timeline = ""
    turn_num = 0
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "user":
            turn_num += 1
            user_content = (m.get("content") or "")[:ENCODING_AGENT['message_display_limit']]
            turn_id = m.get("id", "")

            timeline += "[TURN %d]\n" % turn_num
            timeline += "USER: \"%s\" (turn_id: %s)\n" % (user_content, turn_id)

            # Reference surfaced nodes by ID (full data in catalog above)
            judge_output = m.get("judge_output")
            if judge_output and judge_output != '(no selection)':
                ref_ids = re.findall(r'id:([a-f0-9]{8})', judge_output)
                if ref_ids:
                    # Get titles for readable references
                    refs = []
                    for rid in ref_ids:
                        title_row = brain.conn.execute(
                            "SELECT title FROM nodes WHERE id LIKE ?",
                            (rid + '%',)).fetchone()
                        title = title_row[0][:50] if title_row else rid
                        refs.append('%s ("%s")' % (rid, title))
                    timeline += "BRAIN SURFACED: %s\n" % ", ".join(refs)
                else:
                    timeline += "BRAIN SURFACED: (judge selected but no IDs parsed)\n"
            elif judge_output == '(no selection)':
                timeline += "BRAIN SURFACED: (none relevant)\n"
            else:
                # Fallback: raw candidates (judge didn't complete or old data)
                recalled_raw = m.get("recalled_raw")
                if recalled_raw:
                    try:
                        recalled = json.loads(recalled_raw) if isinstance(recalled_raw, str) else recalled_raw
                        if recalled:
                            timeline += "BRAIN SURFACED (%d candidates, no judge):\n" % len(recalled)
                            for r in recalled[:5]:
                                timeline += "  [%s] %s (id:%s)\n" % (
                                    r.get("type", "?"), r.get("title", "?"),
                                    r.get("id", "?")[:8])
                    except (json.JSONDecodeError, TypeError) as _e:
                        print('[encoding-agent] WARNING: bad recalled_raw JSON: %s' % _e, flush=True)
                else:
                    timeline += "BRAIN SURFACED: (no recall data)\n"

            # Include assistant response — both roles matter equally for encoding
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                asst = (messages[i + 1].get("content") or "")[:ENCODING_AGENT['message_display_limit']]
                timeline += "ASSISTANT: \"%s\"\n" % asst
                i += 1

            timeline += "\n"
        i += 1

    # Previous session context (encoder evolves this each run)
    prev_context = brain.get_config('session_context', '') or ''

    content = "## ENCODING RUN #%d\n\n" % counter
    content += "### Encoding Journal\n%s\n\n" % journal
    if prev_context:
        content += "### Session Context\n%s\n\n" % prev_context
    if node_catalog:
        content += "### %s\n\n" % node_catalog
    content += "### Conversation Timeline\n\n%s\n" % timeline
    return content


def _save_journal(brain, session_id, counter, final_text):
    """Append encoding run to session-scoped journal."""
    from .pipeline_contract import ENCODING_AGENT
    journal_key = 'encoding_journal_%s' % session_id
    existing = brain.get_config(journal_key, '') or ''
    max_chars = ENCODING_AGENT.get('journal_max_chars', 8000)

    new_entry = "--- Run #%d ---\n%s" % (counter, final_text[:ENCODING_AGENT['journal_entry_limit']])
    updated = (existing + '\n' + new_entry).strip()

    # Truncate from the beginning to keep recent runs
    if len(updated) > max_chars:
        updated = updated[-max_chars:]

    brain.set_config(journal_key, updated)

    # Also keep old key for backward compat during transition
    brain.set_config('encoding_agent_state', final_text[:2000])


def _save_session_context(brain, final_text):
    """Extract SESSION_CONTEXT from encoder output and store for recall judge."""
    from .pipeline_contract import JUDGE
    limit = JUDGE.get('session_context_limit', 200)
    for line in final_text.split('\n'):
        stripped = line.strip()
        if stripped.upper().startswith('SESSION_CONTEXT:'):
            context = stripped[len('SESSION_CONTEXT:'):].strip()
            if context:
                brain.set_config('session_context', context[:limit])
                return
    # No SESSION_CONTEXT line found — don't clear existing (previous run's context is still valid)


def _get_tool_schemas():
    """Get v3 encoding tool schemas from brain_mcp (single source of truth)."""
    from . import brain_mcp
    ENCODING_TOOLS = {
        'recall', 'find_node_by_title', 'get_node',
        'remember_batch', 'revise',
        'record_divergence', 'learn_vocabulary',
    }
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["inputSchema"]}
            for t in brain_mcp.TOOLS if t["name"] in ENCODING_TOOLS]
