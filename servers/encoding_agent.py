"""Encoding Agent — LLM-powered brain encoding via Sonnet API.

Called by the daemon's HTTP encoding hook on every 5th Stop event.
Gathers conversation + brain context, calls Sonnet with brain tools,
dispatches tool calls directly against the brain.

Uses:
  - brain_mcp.TOOLS for tool schemas (single source of truth)
  - pipeline_contract.ENCODING_AGENT for limits
  - brain_voice.BrainVoice for formatting
  - contract.generate_field_summary for field documentation

The daemon passes its brain reference and dispatch function.
No TCP relay — direct brain access.
"""

import os
import json
import time


def run_encoding(brain, dispatch_fn, counter, log_fn=None):
    """Run the encoding agent.

    Args:
        brain: Brain instance (direct access, not via TCP)
        dispatch_fn: daemon._dispatch function for tool calls
        counter: Current stop counter value
        log_fn: Optional logging function (daemon._log)

    Returns:
        dict with encoding results summary
    """
    def _log(msg):
        print("[encoding-agent] %s" % msg, flush=True)
        if log_fn:
            log_fn("Encoding agent: %s" % msg)

    import anthropic
    from .pipeline_contract import ENCODING_AGENT
    from .brain_voice import BrainVoice

    _t0 = time.time()
    profile = []  # (step, ms)

    def _step(name):
        profile.append((name, int((time.time() - _t0) * 1000)))

    # Load API key from .env
    _load_env()
    _step("env_loaded")

    try:
        client = anthropic.Anthropic()
    except Exception as e:
        brain._log_error('encoding_agent_api', e, 'Cannot create Anthropic client')
        _log("PROFILE FAILED at api_init: %s" % e)
        return {"error": str(e)}
    _step("api_client")

    session_id = brain.get_config('session_id', 'unknown')

    # 1. Gather messages from DB
    messages = _gather_messages(brain, session_id)
    _step("messages(%d)" % len(messages))
    if not messages:
        _log("no messages, skipping. PROFILE: %s" % profile)
        return {"skipped": True, "reason": "no messages"}

    # 2. Independent recall
    recall_context = _gather_recall_context(brain, messages)
    _step("recall(%d chars)" % len(recall_context))

    # 3. Build prompt
    system_prompt = _build_system_prompt()
    user_content = _build_user_content(brain, messages, recall_context, counter)
    _step("prompt(%d chars)" % len(user_content))

    # 4. Get tool schemas
    tools = _get_tool_schemas()
    _step("tools(%d)" % len(tools))

    # 5. Call Sonnet with tool use loop
    _log("calling Sonnet with %d tools, %d chars context..." % (len(tools), len(user_content)))
    _log("PROFILE so far: %s" % " → ".join("%s:%dms" % (n, t) for n, t in profile))

    try:
        api_messages = [{"role": "user", "content": user_content}]
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=4096,
            system=system_prompt, messages=api_messages, tools=tools)

        _step("sonnet_r0")
        actions = []
        rounds = 0
        for rounds in range(8):
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
                model="claude-sonnet-4-6", max_tokens=4096,
                system=system_prompt, messages=api_messages, tools=tools)
            _step("sonnet_r%d" % (rounds + 1))

        # Save agent state + surface questions to operator
        final_text = "".join(b.text for b in response.content if b.type == "text")
        if final_text:
            brain.set_config('encoding_agent_state', final_text[:2000])
            # Surface questions to Tom via signal queue
            if '?' in final_text:
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
                except Exception:
                    pass

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
    """Fetch recent messages from message_stream."""
    from .pipeline_contract import ENCODING_AGENT
    try:
        rows = brain.logs_conn.execute(
            "SELECT role, content, signal_type, timestamp "
            "FROM message_stream WHERE session_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (session_id, ENCODING_AGENT['max_messages'])
        ).fetchall()
        return [{"role": r[0], "content": (r[1] or "")[:ENCODING_AGENT['message_content_limit']],
                 "signal": r[2], "timestamp": r[3]}
                for r in reversed(rows)]
    except Exception as e:
        brain._log_error('encoding_agent_messages', e, 'Failed to fetch messages')
        return []


def _gather_recall_context(brain, messages):
    """Do independent recall based on conversation topics."""
    from .pipeline_contract import ENCODING_AGENT
    from .brain_voice import BrainVoice
    try:
        user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return ""
        recall_query = " ".join(msg[:200] for msg in user_msgs[-3:])
        result = brain.recall(query=recall_query, limit=ENCODING_AGENT['recall_candidates_limit'])
        results = result.get("results", [])
        if not results:
            return ""
        lines = []
        for r in results:
            c = {"id": r.get("id", ""), "type": r.get("type", ""),
                 "title": r.get("title", ""), "content": r.get("content", ""),
                 "confidence": r.get("confidence", 0), "locked": r.get("locked", False),
                 "revised_at": r.get("revised_at"), "created_at": r.get("created_at"),
                 "_graph": r.get("_graph", {})}
            BrainVoice.format_node_deep(c, lines, conn=brain.conn,
                max_d1=ENCODING_AGENT['max_d1'],
                max_d2=ENCODING_AGENT['max_d2'],
                max_d3=ENCODING_AGENT['max_d3'])
        return "\n".join(lines)
    except Exception as e:
        brain._log_error('encoding_agent_recall', e, 'Failed independent recall')
        return ""


def _build_system_prompt():
    """Load encoding agent prompt + contract field summary."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(project_dir, 'hooks', 'prompts', 'encoding-agent.md')
    try:
        with open(prompt_path) as f:
            prompt = f.read()
    except Exception:
        prompt = "You are the encoding agent. Search before encoding. Revise stale nodes."
    try:
        from .contract import generate_field_summary
        prompt += "\n\n## Available Fields (from contract)\n\n" + generate_field_summary()
    except Exception:
        pass
    return prompt


def _build_user_content(brain, messages, recall_context, counter):
    """Assemble the full encoding prompt."""
    from .pipeline_contract import ENCODING_AGENT

    previous_state = brain.get_config('encoding_agent_state', '') or 'First run.'

    msg_text = ""
    for m in messages:
        role = (m.get("role") or "?").upper()
        content = (m.get("content") or "")[:ENCODING_AGENT['message_display_limit']]
        msg_text += "[%s]: %s\n\n" % (role, content)

    content = "## ENCODING RUN #%d\n\n" % counter
    content += "### Previous State\n%s\n\n" % previous_state
    content += "### Conversation (last %d exchanges)\n\n%s\n" % (len(messages), msg_text)
    if recall_context:
        content += "### Brain Context\n\n%s\n" % recall_context
    else:
        content += "### Brain Context\nNo recall data available.\n\n"
    return content


def _get_tool_schemas():
    """Get encoding-relevant tool schemas from brain_mcp (single source of truth)."""
    from . import brain_mcp
    ENCODING_TOOLS = {
        'recall', 'find_node_by_title', 'get_node',
        'remember', 'revise', 'connect',
        'record_divergence', 'learn_vocabulary',
        'remember_lesson', 'remember_mechanism',
        'remember_mental_model', 'remember_impact',
        'remember_convention',
    }
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["inputSchema"]}
            for t in brain_mcp.TOOLS if t["name"] in ENCODING_TOOLS]
