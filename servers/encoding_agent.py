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
        brain: Brain instance (READ-ONLY — for recall, get_node, session_id)
        dispatch_fn: function(cmd, args) for ALL writes (routes through daemon TCP)
        counter: Current stop counter value
        log_fn: Optional logging function

    All writes (remember, revise, connect, set_config, trace_append) go through
    dispatch_fn which routes to daemon via TCP. brain is read-only to avoid
    DB lock contention with the main daemon thread.

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
        print('[encoding-agent] ERROR: Cannot create Anthropic client: %s' % e, flush=True)
        return {"error": str(e)}
    _step("api_client")

    # session_id passed via dispatch context, fallback to brain property (deprecated)
    session_id = brain.session_id  # TODO: receive as parameter once all callers pass it

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

    # 4. Write prompt to tmp file for dashboard (passive observer pattern)
    try:
        _prompt_path = "/tmp/brain-encoding-prompt-%d.json" % counter
        import json as _pjson
        with open(_prompt_path, 'w') as _pf:
            _pjson.dump({
                "counter": counter,
                "system_prompt_chars": len(system_prompt),
                "user_content": user_content,
                "tools_count": len(tools),
            }, _pf)
    except Exception as _pe:
        print('[encoding-agent] WARNING: could not write prompt file: %s' % _pe, flush=True)

    # Trace S1 encode: O and K via dispatch (routes through daemon for writes)
    try:
        from .session_context import SessionContext
        _ctx = SessionContext(session_id=session_id, stop_counter=counter)
        _enc_chain = _ctx.s1e_chain()
        _turn_count = len(messages) if messages else 0

        # K: extract node IDs from recalled_raw in messages
        _node_ids_in_catalog = set()
        for m in (messages or []):
            _raw = m.get('recalled_raw') or ''
            if _raw:
                try:
                    for c in json.loads(_raw):
                        _cid = c.get('id', '') if isinstance(c, dict) else ''
                        if _cid:
                            _node_ids_in_catalog.add(_cid[:8])
                except (ValueError, TypeError):
                    pass

        # Write traces via dispatch (goes through daemon TCP for writes)
        dispatch_fn('trace_append', {
            'chain_id': _enc_chain, 'scale': 's1', 'event_type': 'O',
            'ref_type': 'encoding_prompt',
            'ref_id': '/tmp/brain-encoding-prompt-%d.json' % counter,
            'summary': '%d turns, %d chars context, interaction: encoding-agent-v3' % (
                _turn_count, len(user_content)),
            'session_id': session_id})
        dispatch_fn('trace_append', {
            'chain_id': _enc_chain, 'scale': 's1', 'event_type': 'K',
            'ref_type': 'node_catalog',
            'ref_id': ','.join(sorted(_node_ids_in_catalog)[:20]),
            'summary': '%d unique nodes in catalog from %d turns' % (
                len(_node_ids_in_catalog), _turn_count),
            'session_id': session_id})
    except Exception as _te:
        print('[encoding-agent] TRACE ERROR: %s' % _te, flush=True)

    # Call Sonnet
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
        _save_journal(brain, dispatch_fn, session_id, counter, final_text)

        # Extract and store session context for recall judge (Layer 2)
        _save_session_context(brain, dispatch_fn, final_text)

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

        # No brain.save() needed — writes go through daemon TCP, not enc_brain
        _step("done")
        profile_str = " → ".join("%s:%dms" % (n, t) for n, t in profile)
        _log("done. %d rounds, %d actions. PROFILE: %s" % (rounds + 1, len(actions), profile_str))
        _write_actions = [a for a in actions if a['tool'] in (
            'remember', 'remember_batch', 'revise', 'revise_batch',
            'connect', 'record_divergence', 'learn_vocabulary')]
        return {"rounds": rounds + 1, "actions": len(actions),
                "write_actions": len(_write_actions),
                "action_details": _write_actions,
                "final_text": final_text[:2000] if final_text else '',
                "profile": profile}

    except Exception as e:
        _step("FAILED")
        profile_str = " → ".join("%s:%dms" % (n, t) for n, t in profile)
        print('[encoding-agent] ERROR: Sonnet API call failed: %s PROFILE: %s' % (e, profile_str), flush=True)
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
    """Fetch recent messages from traces (primary) with message_stream fallback.

    Returns: [{id, role, content, signal, timestamp, recalled_raw, judge_output}]
    Reads from trace_events via get_session_turns(). Falls back to message_stream
    if traces are empty (transition period).
    """
    from .pipeline_contract import ENCODING_AGENT
    limit = ENCODING_AGENT['max_messages']
    content_limit = ENCODING_AGENT['message_content_limit']

    # Primary: read from traces
    try:
        turns = brain._trace_dal.get_session_turns(session_id, limit=limit)
        if turns:
            for i, t in enumerate(turns):
                t['id'] = 'turn-%d' % i  # synthetic ID for timeline references
                t['content'] = (t.get('content', '') or '')[:content_limit]
            return turns
    except Exception as e:
        print('[encoding-agent] TRACE READ ERROR: %s' % e, flush=True)

    # Fallback: read from message_stream (legacy, for sessions without traces)
    try:
        rows = brain.logs_conn.execute(
            "SELECT id, role, content, signal_type, timestamp, recalled_raw, judge_output "
            "FROM message_stream WHERE session_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
        return [{"id": r[0], "role": r[1],
                 "content": (r[2] or "")[:content_limit],
                 "signal": r[3], "timestamp": r[4],
                 "recalled_raw": r[5], "judge_output": r[6]}
                for r in reversed(rows)]
    except Exception as e:
        print('[encoding-agent] ERROR: Failed to fetch messages: %s' % e, flush=True)
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
    try:
        node_catalog, cataloged_ids = build_encoder_node_catalog(judge_outputs, brain.conn)
    except Exception as _e:
        print('[encoding-agent] ERROR building node catalog: %s' % _e, flush=True)
        node_catalog, cataloged_ids = '', set()

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
    prev_context = brain.session_context

    content = "## ENCODING RUN #%d\n\n" % counter
    content += "### Encoding Journal\n%s\n\n" % journal
    if prev_context:
        content += "### Session Context\n%s\n\n" % prev_context
    if node_catalog:
        content += "### %s\n\n" % node_catalog
    content += "### Conversation Timeline\n\n%s\n" % timeline
    return content


def _save_journal(brain, dispatch_fn, session_id, counter, final_text):
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

    dispatch_fn('set_config', {'key': journal_key, 'value': updated})

    # Also keep old key for backward compat during transition
    from .pipeline_contract import PIPELINE as _PL
    dispatch_fn('set_config', {'key': 'encoding_agent_state', 'value': final_text[:_PL['encoding_state_compat']]})


def _save_session_context(brain, dispatch_fn, final_text):
    """Extract SESSION_CONTEXT from encoder output and APPEND to session journey.

    Each encoding run adds its context to the running summary, building
    a journey: "dashboard fix | judge moved to daemon | encoder cleanup".
    Truncates from the beginning to keep recent context within limit.
    Both the recall judge and the encoder read this accumulated context.
    """
    from .pipeline_contract import ENCODING_AGENT
    limit = ENCODING_AGENT.get('session_context_limit', 800)
    for line in final_text.split('\n'):
        stripped = line.strip()
        if stripped.upper().startswith('SESSION_CONTEXT:'):
            new_context = stripped[len('SESSION_CONTEXT:'):].strip()
            if new_context:
                existing = brain.session_context
                if existing:
                    combined = existing + ' | ' + new_context
                else:
                    combined = new_context
                # Truncate from beginning to keep recent context
                if len(combined) > limit:
                    combined = combined[len(combined) - limit:]
                    # Clean up — don't start mid-word
                    pipe_idx = combined.find(' | ')
                    if pipe_idx >= 0 and pipe_idx < 50:
                        combined = combined[pipe_idx + 3:]
                dispatch_fn('set_config', {'key': 'session_context', 'value': combined})
                return
    # No SESSION_CONTEXT line found — don't clear existing (previous run's context is still valid)


def _get_tool_schemas():
    """Get v3 encoding tool schemas from brain_mcp (single source of truth)."""
    from . import brain_mcp
    ENCODING_TOOLS = {
        'recall', 'find_node_by_title', 'get_node',
        'remember_batch', 'revise', 'revise_batch',
        'record_divergence', 'learn_vocabulary',
    }
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["inputSchema"]}
            for t in brain_mcp.TOOLS if t["name"] in ENCODING_TOOLS]
