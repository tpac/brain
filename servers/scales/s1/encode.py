"""S1 Turn Encoder — LLM-powered brain encoding via Sonnet API.

Scale: S1 (Turn integration, every 5th stop)
Chain: s1e (encode)
Interaction: 'encoding_agent' in interactions table (learnable boundary)

Triggered by: encoding gate in hook_post_response_track (daemon_hooks.py)
Reads: traces (conversation turns), brain nodes (catalog), interactions table
Writes: nodes/edges via dispatch, traces (O/K), journal + session context via config
"""

import os
import json
import time

from servers.scales.dispatch import load_env
from servers.scales.runner import run_llm_loop


def run_encoding(brain, dispatch_fn, counter, session_id, log_fn=None):
    """S1 turn encoder: gather → prompt → trace O/K → LLM loop → post-process.

    Args:
        brain: Brain instance (READ-ONLY)
        dispatch_fn: function(cmd, args) for writes (routes through daemon TCP)
        counter: Stop counter value
        session_id: Session ID (required)
        log_fn: Optional logging function

    Returns:
        dict with encoding results summary
    """
    def _log(msg):
        print("[s1e] %s" % msg, flush=True)
        if log_fn:
            log_fn("S1 encode: %s" % msg)

    from servers.scales.s1.encode_contract import ENCODING_AGENT

    t0 = time.time()
    profile = []

    def _step(name):
        profile.append((name, int((time.time() - t0) * 1000)))

    load_env()
    _step("env_loaded")

    import anthropic
    try:
        client = anthropic.Anthropic()
    except Exception as e:
        print('[s1e] ERROR: Cannot create Anthropic client: %s' % e, flush=True)
        return {"error": str(e)}
    _step("api_client")

    if not session_id:
        raise ValueError("session_id is required")

    # 1. Gather messages (S1-specific: reads from traces)
    messages = _gather_messages(brain, session_id)
    _step("messages(%d)" % len(messages))
    if not messages:
        _log("no messages, skipping")
        return {"skipped": True, "reason": "no messages"}

    # 2. Build prompt (from interactions table — learnable boundary)
    enc_interaction = brain.get_interaction('encoding_agent')
    enc_instructions = enc_interaction.get('template', '') if enc_interaction else ''
    system_prompt = _build_system_prompt(prompt_instructions=enc_instructions or None)
    user_content = _build_user_content(brain, messages, counter, session_id)
    _step("prompt(%d chars)" % len(user_content))

    # 3. Get tools (S1-specific subset)
    tools = _get_tool_schemas()
    _step("tools(%d)" % len(tools))

    # 4. Write prompt to tmp file (passive observer for dashboard)
    try:
        with open("/tmp/brain-encoding-prompt-%d.json" % counter, 'w') as f:
            json.dump({
                "counter": counter,
                "system_prompt_chars": len(system_prompt),
                "user_content": user_content,
                "tools_count": len(tools),
            }, f)
    except Exception as e:
        print('[s1e] WARNING: could not write prompt file: %s' % e, flush=True)

    # 5. Write S1 traces: O (observation) and K (knowledge)
    _write_pre_traces(brain, dispatch_fn, messages, user_content, counter, session_id)

    # 6. Run generic LLM loop (shared with S2+)
    _log("calling Sonnet with %d tools, %d chars context..." % (len(tools), len(user_content)))
    _log("PROFILE so far: %s" % " → ".join("%s:%dms" % (n, t) for n, t in profile))

    try:
        result = run_llm_loop(
            client=client,
            model="claude-sonnet-4-6",
            max_tokens=ENCODING_AGENT['max_tokens'],
            max_rounds=ENCODING_AGENT.get('max_rounds', 5),
            system_prompt=system_prompt,
            user_content=user_content,
            tools=tools,
            dispatch_fn=dispatch_fn,
            log_fn=_log)

        _step("done")
        profile_str = " → ".join("%s:%dms" % (n, t) for n, t in profile)
        _log("done. %d rounds, %d actions. PROFILE: %s" % (
            result['rounds'], result['actions'], profile_str))

        # 7. Post-process (S1-specific: journal, session context, signals)
        final_text = result.get('final_text', '')
        _save_journal(brain, dispatch_fn, session_id, counter, final_text)
        _save_session_context(brain, dispatch_fn, final_text)
        _surface_questions(brain, final_text)

        result['profile'] = profile
        return result

    except Exception as e:
        _step("FAILED")
        profile_str = " → ".join("%s:%dms" % (n, t) for n, t in profile)
        print('[s1e] ERROR: Sonnet API call failed: %s PROFILE: %s' % (e, profile_str), flush=True)
        return {"error": str(e), "profile": profile}


# ── S1-Specific Helpers ──


def _gather_messages(brain, session_id):
    """Fetch recent messages from S0 traces.

    Returns: [{id, role, content, signal, timestamp, recalled_raw, judge_output}]
    Reads from trace_events via get_session_turns().
    """
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    limit = ENCODING_AGENT['max_messages']
    content_limit = ENCODING_AGENT['message_content_limit']

    try:
        turns = brain._trace_dal.get_session_turns(session_id, limit=limit)
        if turns:
            for i, t in enumerate(turns):
                t['id'] = 'turn-%d' % i
                t['content'] = (t.get('content', '') or '')[:content_limit]
            return turns
    except Exception as e:
        print('[s1e] TRACE READ ERROR: %s' % e, flush=True)

    # No traces found — empty (message_stream fallback removed 2026-04-05)
    return []


def _build_system_prompt(prompt_instructions=None):
    """Build encoding agent system prompt.

    If prompt_instructions provided (from interactions table), uses it.
    Otherwise falls back to encoding-agent-v3.md file.
    Appends contract field summary in both cases.
    """
    if prompt_instructions:
        prompt = prompt_instructions
    else:
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        prompt_path = os.path.join(project_dir, 'hooks', 'prompts', 'encoding-agent-v3.md')
        try:
            with open(prompt_path) as f:
                prompt = f.read()
        except Exception:
            prompt = "You are the encoding agent. Encode focused nodes. Batch operations. 2-3 rounds."
    try:
        from servers.contract import generate_field_summary
        prompt += "\n\n## Available Fields (from contract)\n\n" + generate_field_summary()
    except Exception as e:
        print('[s1e] WARNING: could not load field summary: %s' % e, flush=True)
    return prompt


def _build_user_content(brain, messages, counter, session_id):
    """Assemble S1 encoding prompt: node catalog + timeline with references."""
    from servers.scales.s1.encode_contract import ENCODING_AGENT, build_node_catalog
    import re

    # Encoding journal (session-scoped, cumulative)
    journal_key = 'encoding_journal_%s' % session_id
    journal = brain.get_config(journal_key, '') or 'First run — no previous encoding in this session.'

    # Build node catalog from judge outputs in the visible window
    judge_outputs = [m.get("judge_output") for m in messages if m.get("role") == "user"]
    try:
        node_catalog, cataloged_ids = build_node_catalog(judge_outputs, brain.conn)
    except Exception as e:
        print('[s1e] ERROR building node catalog: %s' % e, flush=True)
        node_catalog, cataloged_ids = '', set()

    # Build conversation timeline with node references
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
            # Only show when there are actual node IDs — skip noise lines
            judge_output = m.get("judge_output")
            if judge_output and judge_output != '(no selection)':
                ref_ids = re.findall(r'id:([a-f0-9]{8})', judge_output)
                if ref_ids:
                    refs = []
                    for rid in ref_ids:
                        title_row = brain.conn.execute(
                            "SELECT title FROM nodes WHERE id LIKE ?",
                            (rid + '%',)).fetchone()
                        title = title_row[0][:50] if title_row else rid
                        refs.append('%s ("%s")' % (rid, title))
                    timeline += "SURFACED: %s\n" % ", ".join(refs)

            # Include assistant response
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                asst = (messages[i + 1].get("content") or "")[:ENCODING_AGENT['message_display_limit']]
                timeline += "ASSISTANT: \"%s\"\n" % asst
                i += 1

            timeline += "\n"
        i += 1

    # Previous session context
    prev_context = brain.session_context

    # Compute run sequence from journal
    run_seq = journal.count('--- Run ') + 1 if journal != 'First run — no previous encoding in this session.' else 1
    content = "## ENCODING RUN %d (stop #%d)\n\n" % (run_seq, counter)
    content += "### Encoding Journal\n%s\n\n" % journal
    if prev_context:
        content += "### Session Context\n%s\n\n" % prev_context
    if node_catalog:
        content += "### %s\n" % node_catalog
    content += "### Conversation Timeline\n\n%s\n" % timeline
    content += "---\nYou have all the nodes you need in the catalog above. Read what you got before calling any tools. Put ALL operations (creates + revises + connects) in ONE tool call. Target: 2 rounds — one tool call, then journal.\n"
    return content


def _write_pre_traces(brain, dispatch_fn, messages, user_content, counter, session_id):
    """Write S1 encode traces: O (encoding prompt) and K (node catalog)."""
    try:
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id=session_id, stop_counter=counter)
        enc_chain = ctx.s1e_chain()
        turn_count = len(messages) if messages else 0

        # K: extract node IDs from judge outputs
        node_ids = set()
        for m in (messages or []):
            raw = m.get('recalled_raw') or ''
            if raw:
                try:
                    for c in json.loads(raw):
                        cid = c.get('id', '') if isinstance(c, dict) else ''
                        if cid:
                            node_ids.add(cid[:8])
                except (ValueError, TypeError):
                    pass

        dispatch_fn('trace_append', {
            'chain_id': enc_chain, 'scale': 's1', 'event_type': 'O',
            'ref_type': 'encoding_prompt',
            'ref_id': '/tmp/brain-encoding-prompt-%d.json' % counter,
            'summary': '%d turns, %d chars context, interaction: encoding-agent-v3' % (
                turn_count, len(user_content)),
            'session_id': session_id})
        dispatch_fn('trace_append', {
            'chain_id': enc_chain, 'scale': 's1', 'event_type': 'K',
            'ref_type': 'node_catalog',
            'ref_id': ','.join(sorted(node_ids)[:20]),
            'summary': '%d unique nodes in catalog from %d turns' % (
                len(node_ids), turn_count),
            'session_id': session_id})
    except Exception as e:
        print('[s1e] TRACE ERROR: %s' % e, flush=True)


def _save_journal(brain, dispatch_fn, session_id, counter, final_text):
    """Append encoding run to session-scoped journal."""
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    journal_key = 'encoding_journal_%s' % session_id
    existing = brain.get_config(journal_key, '') or ''
    max_chars = ENCODING_AGENT.get('journal_max_chars', 8000)

    # Count previous runs in journal to get sequence number
    run_seq = existing.count('--- Run ') + 1
    new_entry = "--- Run %d (stop #%d) ---\n%s" % (run_seq, counter, final_text[:ENCODING_AGENT['journal_entry_limit']])
    updated = (existing + '\n' + new_entry).strip()

    if len(updated) > max_chars:
        # Truncate at entry boundaries, not mid-character
        truncated = updated[-max_chars:]
        marker = '--- Run '
        idx = truncated.find(marker)
        if idx > 0:
            truncated = truncated[idx:]
        updated = truncated

    dispatch_fn('set_config', {'key': journal_key, 'value': updated})

    # Backward compat key
    from servers.pipeline_contract import PIPELINE
    dispatch_fn('set_config', {'key': 'encoding_agent_state',
                               'value': final_text[:PIPELINE['encoding_state_compat']]})


def _save_session_context(brain, dispatch_fn, final_text):
    """Extract SESSION_CONTEXT from encoder output and append to session journey."""
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    limit = ENCODING_AGENT.get('session_context_limit', 800)
    for line in final_text.split('\n'):
        stripped = line.strip()
        if stripped.upper().startswith('SESSION_CONTEXT:'):
            new_context = stripped[len('SESSION_CONTEXT:'):].strip()
            if new_context:
                existing = brain.session_context
                # Newline-separated entries instead of pipe noise
                combined = (existing + '\n' + new_context) if existing else new_context
                if len(combined) > limit:
                    # Truncate at line boundaries from the front
                    truncated = combined[len(combined) - limit:]
                    nl_idx = truncated.find('\n')
                    if nl_idx >= 0 and nl_idx < 60:
                        truncated = truncated[nl_idx + 1:]
                    combined = truncated
                dispatch_fn('set_config', {'key': 'session_context', 'value': combined})
                return


def _surface_questions(brain, final_text):
    """Surface encoding agent questions to operator via signal queue."""
    if final_text and '?' in final_text:
        try:
            from servers.dal_signal_queue import SignalQueueDAL
            sq = SignalQueueDAL(brain.logs_conn)
            sq.produce(
                producer='encoding_agent',
                signal_type='encoding_question',
                priority=0.7,
                content=final_text[:500],
                ttl_seconds=86400,
            )
            brain.logs_conn.commit()
        except Exception as e:
            print('[s1e] ERROR surfacing question: %s' % e, flush=True)


def _get_tool_schemas():
    """Get S1 encoding tool schemas from brain_mcp (single source of truth)."""
    from servers import brain_mcp
    ENCODING_TOOLS = {
        'remember_batch', 'revise_batch',
        'brain_batch', 'connect_batch',
        'recall_batch', 'get_nodes',
    }
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["inputSchema"]}
            for t in brain_mcp.TOOLS if t["name"] in ENCODING_TOOLS]
