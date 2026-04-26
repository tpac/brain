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
from servers.trace_contract import build_delta_metadata


def run_encoding(brain, dispatch_fn, counter, session_id, log_fn=None,
                 muster_enabled=None):
    """S1 turn encoder: gather → prompt → trace O/K → LLM loop → post-process.

    Args:
        brain: Brain instance (READ-ONLY)
        dispatch_fn: function(cmd, args) for writes (routes through daemon TCP)
        counter: Stop counter value
        session_id: Session ID (required)
        log_fn: Optional logging function
        muster_enabled: Explicit override for the Phase-1 scouts muster.
            When None (default), muster runs — the v13 prompt is built
            around scout reports. Passing False is for tests / ablation
            harnesses that want to measure no-scout behavior.

    Returns:
        dict with encoding results summary. When muster runs, also includes
        'muster' key with per-scout metrics and any scout errors.
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
    enc_interaction = brain.get_interaction('s1e') or brain.get_interaction('encoding_agent')
    enc_instructions = enc_interaction.get('template', '') if enc_interaction else ''
    system_prompt = _build_system_prompt(prompt_instructions=enc_instructions or None)
    user_preamble, user_content, catalog_text, catalog_ids = _build_user_content(
        brain, messages, counter, session_id)
    _step("prompt(preamble=%d chars, body=%d chars)" % (
        len(user_preamble), len(user_content)))

    # 2b. Muster phase — Phase-1 scouts (quote / temporal / facts / synthesis)
    # fan out in parallel, emit O/K traces on the s1e chain, and produce a
    # report block appended to user_content. Architectural default: ON.
    # The v13 prompt ships with `## Scout reports` as part of its expected
    # input; running without scouts leaves a structural hole. The explicit
    # `muster_enabled=False` kwarg remains for tests that need to toggle.
    if muster_enabled is None:
        muster_enabled = True

    muster_summary = None
    if muster_enabled:
        try:
            from servers.scales.s1.scouts.muster import (
                build_muster_context, run_muster)
            muster_ctx = build_muster_context(
                brain=brain, messages=messages, session_id=session_id,
                counter=counter,
                catalog_rendered=catalog_text,
                catalog_node_ids=catalog_ids,
                session_context=(brain.session_context or ''),
                log_fn=log_fn,
            )
            _step("muster_ctx")
            scout_report, scout_outputs, muster_metrics = run_muster(muster_ctx)
            _step("muster_done(%dms,%dc)" % (
                muster_metrics.get('elapsed_ms', 0),
                muster_metrics.get('total_candidates', 0)))
            if scout_report.strip():
                user_content = user_content + "\n\n## Scout reports\n\n" + scout_report
            muster_summary = {
                'enabled': True,
                'metrics': muster_metrics,
                'scout_names': list(scout_outputs.keys()),
            }
        except Exception as muster_exc:
            # Scouts are advisory — never block encoding. Log loud, proceed.
            print('[s1e] MUSTER ERROR (falling back to no scouts): %s' %
                  muster_exc, flush=True)
            try:
                # _log_error expects an Exception so its traceback formatter
                # works — passing the caught exception directly.
                brain._log_error('s1e_muster_fallback', muster_exc,
                                 'muster raised; encoding continues without scout reports')
            except Exception:
                pass
            muster_summary = {'enabled': True, 'error': str(muster_exc)}
    else:
        muster_summary = {'enabled': False}

    # 3. Get tools (S1-specific subset)
    tools = _get_tool_schemas()
    _step("tools(%d)" % len(tools))

    # 4. Write prompt to tmp file (passive observer for dashboard + post-hoc
    # eval inspection). Path includes FULL session_id + pid so parallel
    # jobs don't clobber each other's files. 16-char prefix collided
    # across jobs whose session_ids shared the same leading bytes — a real
    # bug caught in the smoke_seed_2 run where arm-A jobs read arm-B's
    # prompt files and wrongly flagged "scout_reports_absent" as failed.
    _session_safe = (session_id or 'nosession').replace('/', '_').replace(' ', '_')
    prompt_path = "/tmp/brain-encoding-prompt-%s-%d.json" % (
        _session_safe, counter)
    try:
        with open(prompt_path, 'w') as f:
            json.dump({
                "counter": counter,
                "session_id": session_id,
                "system_prompt_chars": len(system_prompt),
                "user_preamble": user_preamble,
                "user_content": user_content,
                "tools_count": len(tools),
            }, f)
        # Legacy counter-only path for dashboards that still expect it —
        # overwrite is acceptable for dashboards, not for parallel eval.
        try:
            with open("/tmp/brain-encoding-prompt-%d.json" % counter, 'w') as f:
                json.dump({
                    "counter": counter,
                    "session_id": session_id,
                    "system_prompt_chars": len(system_prompt),
                    "user_preamble": user_preamble,
                "user_content": user_content,
                    "tools_count": len(tools),
                }, f)
        except Exception:
            pass  # best-effort dashboard compat
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
            user_preamble=user_preamble,
            tools=tools,
            dispatch_fn=dispatch_fn,
            log_fn=_log)

        _step("done")
        profile_str = " → ".join("%s:%dms" % (n, t) for n, t in profile)
        _log("done. %d rounds, %d actions. PROFILE: %s" % (
            result['rounds'], result['actions'], profile_str))

        # Log truncation errors to brain errors table
        for trunc in result.get('truncations', []):
            brain._log_error(
                's1e_truncation',
                'max_tokens truncation: round %d used %s/%s output tokens' % (
                    trunc['round'], trunc['output_tokens'], trunc['max_tokens']),
                'S1E tool call likely corrupted, encoding data may be lost')

        # 7. Post-process (S1-specific: journal, session context)
        final_text = result.get('final_text', '')
        journal_entry = _save_journal(brain, dispatch_fn, session_id, counter, final_text) or ''
        _save_session_context(brain, dispatch_fn, final_text)

        # 8. Delta trace — unified shape across S1E + S2 encoders.
        # Outcomes: count write actions by tool (remember / revise / connect / …).
        action_details = result.get('action_details', [])
        outcomes = {}
        for a in action_details:
            tool = a.get('tool', 'unknown')
            outcomes[tool] = outcomes.get(tool, 0) + 1

        enc_chain = 's1e-%s-%d' % (session_id[:8], counter)
        dispatch_fn('trace_append', {
            'chain_id': enc_chain, 'scale': 's1', 'event_type': 'delta',
            'ref_type': 'encoding_run',
            'summary': '%d actions (%d writes) in %d rounds' % (
                result.get('actions', 0),
                result.get('write_actions', 0),
                result.get('rounds', 0)),
            'metadata': build_delta_metadata(
                actions=result.get('actions', 0),
                write_actions=result.get('write_actions', 0),
                rounds=result.get('rounds', 0),
                inputs_processed=len(messages),
                outcomes=outcomes,
                journal_entry=journal_entry,
                action_details=action_details,
                read_calls=result.get('read_calls', []),
                final_text=final_text,
                stop_counter=counter,
            ),
            'session_id': session_id,
        })

        result['profile'] = profile
        if muster_summary is not None:
            result['muster'] = muster_summary
        return result

    except Exception as e:
        _step("FAILED")
        profile_str = " → ".join("%s:%dms" % (n, t) for n, t in profile)
        print('[s1e] ERROR: Sonnet API call failed: %s PROFILE: %s' % (e, profile_str), flush=True)
        return {"error": str(e), "profile": profile}


# ── S1-Specific Helpers ──


def _gather_messages(brain, session_id):
    """Fetch recent messages for the current session via S0 API.

    Returns: [{id, role, content, signal, timestamp, recalled_raw, surface_output}]
    Uses S0 layer's get_conversation() — single source of truth.
    """
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    from servers.scales.s0.conversation import get_conversation
    limit = ENCODING_AGENT['max_messages']
    content_limit = ENCODING_AGENT['message_content_limit']

    try:
        turns = get_conversation(brain, session_id, limit=limit)
        if turns:
            for i, t in enumerate(turns):
                t['id'] = 'turn-%d' % i
                t['content'] = (t.get('content', '') or '')[:content_limit]
            return turns
    except Exception as e:
        print('[s1e] S0 CONVERSATION ERROR: %s' % e, flush=True)

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
    """Assemble S1 encoding prompt: stable preamble + dynamic body.

    The split is deliberate for caching. The stable preamble (instructions
    + format expectations + section legend) is byte-identical across every
    encoding cycle and gets a 1h cache breakpoint via run_llm_loop's
    `user_preamble` arg. The dynamic body (journal, catalog, timeline)
    gets the 5m breakpoint.

    Returns:
        (user_preamble, user_body, catalog_text, catalog_ids)
        - user_preamble: stable instructions; safe to cache 1h.
        - user_body: dynamic content for this cycle (5m cache).
        - catalog_text: rendered catalog block (reused by muster).
        - catalog_ids: set of node ids in the catalog (reused by temporal scout).
    """
    from servers.scales.s1.encode_contract import ENCODING_AGENT, build_node_catalog
    import re

    # Encoding journal (session-scoped, cumulative)
    journal_key = 'encoding_journal_%s' % session_id
    journal = brain.get_config(journal_key, '') or 'First run — no previous encoding in this session.'

    # Build node catalog from surface outputs in the visible window
    judge_outputs = [m.get("judge_output") for m in messages if m.get("role") == "user"]
    try:
        node_catalog, cataloged_ids = build_node_catalog(judge_outputs, brain)
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
                ref_ids = re.findall(r'id:([a-z0-9_]{6,8})', judge_output)
                if ref_ids:
                    from servers.dal import NodeDAL
                    dal = NodeDAL(brain.conn)
                    refs = []
                    for rid in ref_ids:
                        title = (dal.get_title(rid) or rid)[:50]
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

    # ── Stable preamble — byte-identical across encoding cycles.
    # Cached at 1h TTL via run_llm_loop's user_preamble parameter. The
    # only constraint: nothing here may vary per cycle. Section legend
    # + format expectations live here. The "ENCODING RUN N" header
    # was removed (was metadata; not load-bearing for the encoder).
    preamble = (
        "You are encoding what you've just observed. The sections below give you, "
        "in order: prior encoding work this session (Encoding Journal), what the "
        "session is about (Session Context), nodes the brain already knows pre-"
        "loaded for this window (Node Catalog), and the actual turns with "
        "references to surfaced nodes (Conversation Timeline).\n\n"
        "Read what you got before calling any tools. Put ALL operations "
        "(remember + revise + connect) in ONE tool call. Target: 2 rounds — "
        "one tool call, then the journal.\n"
    )

    # ── Dynamic body — varies per cycle, 5m cache.
    body = ""
    body += "### Encoding Journal\n%s\n\n" % journal
    if prev_context:
        body += "### Session Context\n%s\n\n" % prev_context
    if node_catalog:
        body += "### %s\n" % node_catalog
    body += "### Conversation Timeline\n\n%s\n" % timeline
    return preamble, body, node_catalog, cataloged_ids


def _write_pre_traces(brain, dispatch_fn, messages, user_content, counter, session_id):
    """Write S1 encode traces: O (encoding prompt) and K (node catalog)."""
    try:
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id=session_id, stop_counter=counter)
        enc_chain = ctx.s1e_chain()
        turn_count = len(messages) if messages else 0

        # K: extract node IDs from surface outputs
        node_ids = set()
        for m in (messages or []):
            raw = m.get('recalled_raw') or ''
            if raw:
                try:
                    for c in json.loads(raw):
                        cid = c.get('id', '') if isinstance(c, dict) else ''
                        if cid:
                            node_ids.add(cid[:8])
                except (ValueError, TypeError) as _e:
                    # Corrupt recalled_raw JSON silently dropped node IDs
                    # from the encoding-prompt trace. Surface so we can spot
                    # whether the producer side (recall result serialization)
                    # is emitting malformed content.
                    try:
                        brain._log_error(
                            'encoding_recall_parse', _e,
                            'malformed recalled_raw — node refs missing from O-trace; sample=%r'
                            % str(raw)[:160])
                    except Exception:
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
    """Append encoding run to session-scoped journal. Returns the entry text.

    S1E's entry is just the truncated final_text — encoder output is
    already the narrative. If final_text is empty, returns ''; logs a
    brain error since that's an agent-drift signal.
    """
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    journal_key = 'encoding_journal_%s' % session_id
    existing = brain.get_config(journal_key, '') or ''
    max_chars = ENCODING_AGENT.get('journal_max_chars', 8000)

    entry_body = final_text[:ENCODING_AGENT['journal_entry_limit']]
    if not entry_body.strip():
        brain._log_error(
            's1e_journal_extraction',
            'empty final_text from S1E (stop #%d)' % counter,
            'encoder produced no narrative — check prompt + LLM output')
        return ''

    # Count previous runs in journal to get sequence number
    run_seq = existing.count('--- Run ') + 1
    new_entry = "--- Run %d (stop #%d) ---\n%s" % (run_seq, counter, entry_body)
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

    return entry_body


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
