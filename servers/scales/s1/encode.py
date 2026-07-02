"""S1 Turn Encoder — LLM-powered brain encoding via Sonnet API.

Scale: S1 (Turn integration, every 5th stop)
Chain: s1e (encode)
Interaction: 's1e' in interactions table (learnable boundary; 'encoding_agent' was the legacy name)

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
from servers.daemon_config import brain_tmp_dir


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
    from ..runner import ANTHROPIC_CLIENT_TIMEOUT
    try:
        client = anthropic.Anthropic(timeout=ANTHROPIC_CLIENT_TIMEOUT)
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

    # Resolve the A/B arm ONCE per run (env read is non-atomic across call sites;
    # resolving here and threading down guarantees the system prompt, body, and
    # post-process write all agree — no torn arm if the flag is toggled mid-run).
    lived = _lived_sequence_enabled()

    # 2. Build prompt (from interactions table — learnable boundary)
    enc_interaction = brain.get_interaction('s1e')
    enc_instructions = enc_interaction.get('template', '') if enc_interaction else ''
    system_prompt = _build_system_prompt(
        prompt_instructions=enc_instructions or None, lived=lived)

    # 2a. Catalog FIRST (both arms) — muster needs the rendered catalog +
    # catalog ids, and on the lived arm the body needs the muster's findings
    # (inlined into the timeline), so the assembly order is catalog → scouts →
    # body. Control arm output is unchanged: same catalog, same body, the scout
    # report appended after — only the internal build order moved.
    catalog_text, catalog_ids, streams = _build_catalog(
        brain, messages, session_id, lived)
    _step("catalog(%d ids)" % len(catalog_ids))

    # 2b. Muster phase — Phase-1 scouts fan out in parallel, emit O/K traces on
    # the s1e chain. Architectural default: ON. The lived arm runs WITHOUT the
    # quote scout (episodes recall preserves verbatim substrate — Tom 2026-07-02)
    # and consumes findings as per-turn timeline annotations; the control arm
    # runs the full set and appends the classic `## Scout reports` block.
    if muster_enabled is None:
        muster_enabled = True

    scout_report, scout_outputs, muster_summary = '', None, {'enabled': False}
    if muster_enabled:
        scout_report, scout_outputs, muster_summary = _run_muster_phase(
            brain, messages, session_id, counter, catalog_text, catalog_ids,
            log_fn, _step, exclude_scouts=(('quote',) if lived else ()))

    # 2c. Body assembly. Lived: scout findings ride INSIDE the timeline
    # (<scout_notes> per turn + <scout_legend>); control: legacy body + the
    # appended report.
    user_preamble, user_content, _cat_text2, _cat_ids2 = _build_user_content(
        brain, messages, counter, session_id, lived_sequence=lived,
        precomputed=(catalog_text, catalog_ids, streams),
        scout_outputs=(scout_outputs if lived else None))
    if not lived and scout_report.strip():
        user_content = user_content + "\n\n## Scout reports\n\n" + scout_report
    _step("prompt(preamble=%d chars, body=%d chars)" % (
        len(user_preamble), len(user_content)))

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
    prompt_path = os.path.join(brain_tmp_dir(), "brain-encoding-prompt-%s-%d.json" % (
        _session_safe, counter))
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
            with open(os.path.join(brain_tmp_dir(), "brain-encoding-prompt-%d.json" % counter), 'w') as f:
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

    # Full-prompt capture label (eval/observability) — only computed when
    # BRAIN_PROMPT_CAPTURE_DIR is set, so production stays a no-op. Keys the
    # dump by arm + session + stop so control vs new (and each turn) never
    # collide; runner.py adds round + a monotonic seq.
    capture_label = None
    if os.environ.get('BRAIN_PROMPT_CAPTURE_DIR'):
        arm = 'new' if lived else 'control'
        safe_sid = (session_id or 'nosession').replace('/', '_').replace(' ', '_')
        capture_label = "%s__%s__stop%d" % (arm, safe_sid, counter)

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
            log_fn=_log,
            capture_label=capture_label)

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

        # 7. Post-process (S1-specific: journal/residue, session context)
        final_text = result.get('final_text', '')
        enc_chain = 's1e-%s-%d' % (session_id[:8], counter)
        if lived:
            # New residue (Piece 4): the `## Review` note contract, SESSION-BOUND.
            # write_journal_notes extracts the fence + writes one journal_note
            # trace per note, all sharing enc_chain (this run); session_id walls
            # continuity to this conversation. Replaces the legacy blob; the arc
            # (_save_session_context) is a SEPARATE object and stays untouched.
            #
            # INTENTIONAL flag-on side effect: not writing the blob leaves the
            # Frame's `## Recent moves` (which reads the legacy encoding_journal
            # blob) empty. That's the deferred Frame-slot cut previewing — the cut
            # lands at activation (replacement-before-removal). No eval confound:
            # the Frozen-Corpus sweep queries with a FRESH session_id, so Recent
            # moves is empty in BOTH arms there regardless of this flag.
            try:
                brain.write_journal_notes(final_text=final_text, chain_id=enc_chain,
                                          scale='s1', session_id=session_id)
            except Exception as e:
                brain._log_error('s1e_journal_notes_write', e,
                                 'residue note write failed — run otherwise intact')
            journal_entry = ''
        else:
            journal_entry = _save_journal(brain, dispatch_fn, session_id, counter, final_text) or ''
        _save_session_context(brain, dispatch_fn, session_id, final_text)

        # 8. Delta trace — unified shape across S1E + S2 encoders.
        # Outcomes: count write actions by tool (remember / revise / connect / …).
        action_details = result.get('action_details', [])
        outcomes = {}
        for a in action_details:
            tool = a.get('tool', 'unknown')
            outcomes[tool] = outcomes.get(tool, 0) + 1

        # enc_chain computed above (post-process) — reused here for the delta.
        # Which K version produced this Δ — the FK (interaction_id) is stamped
        # on the trace row for joins; the version number rides in metadata for
        # human-readable scanning. Lets higher scales A/B prompt versions from
        # production traces (the whole point of interactions-as-K-store).
        enc_iid = (enc_interaction or {}).get('id')
        enc_ver = (enc_interaction or {}).get('version', 0)
        enc_metadata = build_delta_metadata(
            actions=result.get('actions', 0),
            write_actions=result.get('write_actions', 0),
            rounds=result.get('rounds', 0),
            inputs_processed=len(messages),
            outcomes=outcomes,
            journal_entry=journal_entry,
            action_details=action_details,
            read_calls=result.get('read_calls', []),
            final_text=final_text,
            elapsed_ms=result.get('elapsed_ms', 0),
            input_tokens=result.get('input_tokens', 0),
            output_tokens=result.get('output_tokens', 0),
            cache_read_tokens=result.get('cache_read_tokens', 0),
            cache_creation_tokens=result.get('cache_creation_tokens', 0),
            truncated=len(result.get('truncations', []) or []),
            interaction_version=enc_ver,
            stop_counter=counter,
        )
        dispatch_fn('trace_append', {
            'chain_id': enc_chain, 'scale': 's1', 'event_type': 'delta',
            'ref_type': 'encoding_run',
            'interaction_id': enc_iid,
            'summary': '%d actions (%d writes) in %d rounds, %dms, %d→%d tok' % (
                result.get('actions', 0),
                result.get('write_actions', 0),
                result.get('rounds', 0),
                result.get('elapsed_ms', 0),
                result.get('input_tokens', 0),
                result.get('output_tokens', 0)),
            'metadata': enc_metadata,
            'session_id': session_id,
        })
        # Loud-at-the-write-boundary: same telemetry guard IntegrationUnit.trace
        # applies to the S2 encoders — but encoding_run is S1E's delta, written
        # via dispatch (not IntegrationUnit), so the check lives here too. brain
        # is in scope; logs to the errors view. S1E already threads telemetry, so
        # this is a regression guard (it stays silent on a healthy run).
        from servers.trace_contract import check_delta_telemetry
        _tel_warn = check_delta_telemetry('encoding_run', enc_metadata)
        if _tel_warn:
            brain._log_error('s1e_telemetry_gap', ValueError(_tel_warn),
                             'delta trace missing LLM telemetry')

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
                # `id` was a synthetic display label (turn-N). Keep it as
                # `turn_label` for human-facing render. `trace_id` (hex)
                # comes from S0 layer and is the load-bearing reference
                # the encoder copies into source_refs.
                t['turn_label'] = 'turn-%d' % i
                t['id'] = 'turn-%d' % i  # backward-compat for downstream readers
                t['content'] = (t.get('content', '') or '')[:content_limit]
            return turns
    except Exception as e:
        print('[s1e] S0 CONVERSATION ERROR: %s' % e, flush=True)

    return []


def _build_system_prompt(prompt_instructions=None, lived=None):
    """Build encoding agent system prompt.

    If prompt_instructions provided (from interactions table), uses it.
    Otherwise falls back to encoding-agent-v3.md file.
    Appends contract field summary in both cases.

    `lived` is the resolved A/B arm (run_encoding resolves it once and threads it
    in); None → read the env flag directly (tests / standalone callers).
    """
    if lived is None:
        lived = _lived_sequence_enabled()
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

    # Residue contract (Piece 4, flag-gated): the WRITE-side instructions — tell
    # the encoder to emit a `## Review` fence, then the closure (terminal-turn +
    # DONE) as the LAST block. Two SEPARATE injects (the closure must not be
    # entangled with the review block), placed at the end by design: writing the
    # review is the encoder's final act, so its instruction sits where the action
    # lands (recency). Contract-owned text (single source across all encoders) —
    # never hardcoded in the s1e template, so it can't drift. Flag-off keeps the
    # legacy blob path, so this stays absent from the control arm.
    if lived:
        try:
            from servers.trace_contract import (render_journal_review_block,
                                                 render_prompt_closure)
            prompt = prompt.rstrip() + "\n\n" + render_journal_review_block()
            prompt = prompt.rstrip() + "\n\n" + render_prompt_closure()
        except Exception as e:
            print('[s1e] WARNING: could not inject review block/closure: %s' % e, flush=True)
    return prompt


def _run_muster_phase(brain, messages, session_id, counter, catalog_text,
                      catalog_ids, log_fn, _step, exclude_scouts=()):
    """The muster try/except envelope, extracted from run_encoding so the two
    arms can sequence it differently (lived: before body assembly, findings
    inlined; control: report appended after). Scouts are advisory — any failure
    logs loud and encoding proceeds without them.

    Returns (scout_report, scout_outputs, muster_summary)."""
    try:
        from servers.scales.s1.scouts.muster import (
            build_muster_context, run_muster)
        # conversation_now resolves the date THIS conversation thinks
        # it's happening: eval replay reads [Current date: ...] prefix;
        # production reads SessionContext.created_at; falls back to
        # operator wall-clock. Critical for temporal scout — without it,
        # "today/yesterday" in historical conversations resolve to NOW.
        # See servers/clock.py + brain memory 6d5b789e.
        from servers.clock import conversation_now
        session_ctx_obj = brain.get_or_create_session(session_id)
        conv_started = getattr(session_ctx_obj, 'started_at', None)
        conv_dt = conversation_now(
            messages=messages,
            session_started_at=conv_started,
            brain=brain)
        muster_ctx = build_muster_context(
            brain=brain, messages=messages, session_id=session_id,
            counter=counter,
            catalog_rendered=catalog_text,
            catalog_node_ids=catalog_ids,
            session_context=brain.session_context_for(session_id),
            current_date=conv_dt.date().isoformat(),
            log_fn=log_fn,
        )
        _step("muster_ctx")
        scout_report, scout_outputs, muster_metrics = run_muster(
            muster_ctx, exclude_scouts=exclude_scouts)
        _step("muster_done(%dms,%dc)" % (
            muster_metrics.get('elapsed_ms', 0),
            muster_metrics.get('total_candidates', 0)))
        return scout_report, scout_outputs, {
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
        return '', None, {'enabled': True, 'error': str(muster_exc)}


def _build_catalog(brain, messages, session_id, lived):
    """The catalog half of the encoding input — extracted from
    _build_user_content so muster (which consumes the rendered catalog) can run
    BEFORE body assembly on the lived arm. Gathers the trace streams once
    (lived); the same tuple threads into the timeline's <provenance>.

    Returns (node_catalog, cataloged_ids, streams)."""
    from servers.scales.s1.encode_contract import build_node_catalog

    judge_outputs = [m.get("judge_output") for m in messages if m.get("role") == "user"]

    streams, extra_ids = None, None
    if lived:
        try:
            from servers.scales.s1.trace_links import gather, session_node_ids
            streams = gather(brain, session_id)           # {'surface','encode','touched'}
            extra_ids = session_node_ids(streams['encode'], streams['touched'])
        except AttributeError:
            # Stub brain (tests) without query_traces — expected, quiet. Degrade
            # to surfaced-only catalog + self-gathering timeline.
            streams, extra_ids = None, None
        except Exception as e:
            # A real failure (trace-contract drift, malformed record) — LOUD via
            # brain.errors (the monitored surface), then degrade. Mirrors the
            # _turn_links guard; a bare print in the daemon log is not the surface.
            try:
                brain._log_error('s1e_catalog_streams', e,
                                 'trace_links gather/union failed; catalog stays surfaced-only')
            except Exception:
                pass
            streams, extra_ids = None, None

    try:
        node_catalog, cataloged_ids = build_node_catalog(
            judge_outputs, brain, extra_ids=extra_ids)
    except Exception as e:
        print('[s1e] ERROR building node catalog: %s' % e, flush=True)
        node_catalog, cataloged_ids = '', set()
    return node_catalog, cataloged_ids, streams


def _scout_note_line(scout, cand):
    """One compact <scout_notes> line for a candidate: scout + handle + a
    trimmed detail (event description / evidence quote), whitespace-collapsed
    and XML-escaped so it can't malform the timeline."""
    handle = str(cand.get('handle') or '').strip()
    detail = str(cand.get('event_description') or cand.get('evidence_quote') or '').strip()
    extras = [str(b) for b in (cand.get('precision'), cand.get('relational_marker')) if b]
    line = '%s: %s' % (scout, handle)
    if detail:
        line += ' — %s' % detail[:140]
    if extras:
        line += ' (%s)' % ', '.join(extras)
    return _xml_escape(' '.join(line.split()))


def _map_scout_notes(scout_outputs, messages):
    """Join scout candidates to the timeline's turns (Tom #5 — findings live
    where they happened, not in a trailing report).

    The join: candidates cite muster turn ids (`messages[i]['id']`, mirroring
    build_muster_context's `m.get('id') or 'turn-{i}'`); each cited message maps
    to its OWNING user turn (walk back to the nearest user message), whose
    `trace_id` is exactly the episode id the lived timeline keys its turns on.
    First mappable evidence turn wins; candidates citing nothing mappable land
    in the window-level `unmapped` list (rendered in the legend, not dropped).

    Returns (per_turn {user_trace_id: [line, ...]}, unmapped [line, ...],
    legend_lines [scout category statements])."""
    idx_by_mid = {}
    for i, m in enumerate(messages or []):
        idx_by_mid[m.get('id') or 'turn-%d' % i] = i
    owner_trace, last_user = [], None
    for m in (messages or []):
        if m.get('role') == 'user':
            last_user = m.get('trace_id')
        owner_trace.append(last_user)

    per_turn, unmapped, legend = {}, [], []
    for name in ('temporal', 'facts'):
        env = (scout_outputs or {}).get(name) or {}
        if env.get('_errors'):
            continue                      # stub (disabled / timed out) — nothing to inline
        cs = ' '.join(str(env.get('category_statement') or '').split())
        if cs:
            legend.append('%s — %s' % (name, _xml_escape(cs)))
        for cand in (env.get('candidates') or ()):
            line = _scout_note_line(name, cand)
            owner = None
            for et in (cand.get('evidence_turns') or ()):
                i = idx_by_mid.get(et)
                if i is not None and owner_trace[i]:
                    owner = owner_trace[i]
                    break
            if owner:
                per_turn.setdefault(owner, []).append(line)
            else:
                unmapped.append(line)
    return per_turn, unmapped, legend


def _render_scout_legend(legend_lines, unmapped):
    """The <scout_legend> block (Tom's spec): explains that the inline
    `scout:` notes came from OUTSIDE this read — what the scouts are and how
    their findings got into the timeline — plus each scout's category statement
    and any window-level findings no single turn owns."""
    out = "<scout_legend>\n"
    out += ("Notes inside <scout_notes> came from outside this read: focused "
            "scouts that scanned this same window in parallel before this "
            "encode — temporal (date anchors + event descriptions, "
            "algorithmic) and facts (entity-feature-value triples, Haiku) — "
            "and their findings are attached to the turns they cite. Each was "
            "primed for one kind of atomization: hints, not the map. They "
            "propose; I compose — I read the window myself.\n")
    for ln in legend_lines:
        out += "- %s\n" % ln
    if unmapped:
        out += "Window-level findings no single turn owns:\n"
        for ln in unmapped:
            out += "- %s\n" % ln
    out += "</scout_legend>\n"
    return out


def _build_user_content(brain, messages, counter, session_id, lived_sequence=None,
                        precomputed=None, scout_outputs=None):
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

    `precomputed` — the (node_catalog, cataloged_ids, streams) tuple from
    _build_catalog, when run_encoding already built it (it runs muster between
    the catalog and the body). None → build here (tests / standalone callers).
    `scout_outputs` — muster envelopes (lived arm only): temporal+facts
    candidates inline into the timeline as per-turn <scout_notes>, with a
    <scout_legend> explaining where they came from; the trailing
    `## Scout reports` block is retired on this arm.
    """
    # A/B flag (piece 1, docs/S1-SCRIBE-REDESIGN.md §10.3.1): OFF (default) =
    # markdown messages-only timeline + surfaced-only catalog (the long-standing
    # path, untouched). ON = the new input as ONE unit: the XML lived-sequence
    # timeline (messages + tool actions + per-turn <provenance> + <scout_notes>)
    # AND the widened catalog (surfaced ∪ encoded ∪ authored ∪ recalled, tagged).
    # Both consume the same trace streams, gathered ONCE and threaded into both.
    lived = _lived_sequence_enabled() if lived_sequence is None else lived_sequence
    if precomputed is not None:
        node_catalog, cataloged_ids, streams = precomputed
    else:
        node_catalog, cataloged_ids, streams = _build_catalog(
            brain, messages, session_id, lived)

    scout_legend = ''
    if lived:
        scout_notes = None
        if scout_outputs:
            per_turn, unmapped, legend_lines = _map_scout_notes(
                scout_outputs, messages)
            scout_notes = per_turn
            if legend_lines or unmapped or per_turn:
                scout_legend = _render_scout_legend(legend_lines, unmapped)
        timeline = _render_lived_sequence_timeline(
            brain, session_id, messages, streams=streams,
            scout_notes=scout_notes)
    else:
        timeline = _render_markdown_timeline(brain, messages)

    # Previous session context (per-session — no global leak across parallel sessions)
    prev_context = brain.session_context_for(session_id)

    # Continuity — the encoder's prior residue this session (the READ side; the
    # encoder's own prompt framing, NOT the identity Frame). New arm = the
    # `## Review` note contract, SESSION-BOUND (scale='s1', session_id → last K=5
    # runs in THIS conversation; never carries across sessions). Old arm = the
    # legacy `### Encoding Journal` blob. self-labeled block either way.
    if lived:
        try:
            from servers.trace_contract import render_journal_notes_prefix
            journal_block = render_journal_notes_prefix(
                brain.journal_notes(scale='s1', session_id=session_id))
        except Exception as e:
            try:
                brain._log_error('s1e_journal_notes_read', e,
                                 'residue continuity read failed — encoding without it')
            except Exception:
                pass
            journal_block = ''
    else:
        blob = (brain.get_config('encoding_journal_%s' % session_id, '')
                or 'First run — no previous encoding in this session.')
        journal_block = "### Encoding Journal\n%s" % blob

    # ── Stable preamble — byte-identical across encoding cycles (1h cache via
    # run_llm_loop's user_preamble). Branches on the arm. The NEW arm's section
    # legend + reading order live in the registered v-next system prompt (it names
    # <continuity>/<node_catalog>/<timeline>), so the preamble here drops the legend
    # and keeps only the operational anchor — two voices describing the layout would
    # confound the A/B. The control arm keeps the legacy legend verbatim.
    if lived:
        # First person — matches the v-next system prompt's register (the encoder
        # speaks as itself: "I am Anchor, and this is me encoding my own memory").
        preamble = (
            "I'm encoding what I've just observed. I read everything below before "
            "calling any tools, then put all operations (remember + revise + "
            "connect) in one tool call.\n"
        )
    else:
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

    # ── Dynamic body — varies per cycle, 5m cache. The NEW arm wraps each section
    # in the XML label the v-next prompt names (<continuity> folds residue + arc;
    # <node_catalog>; <timeline>). The control arm keeps the legacy ### markdown
    # headers — byte-identical to the long-standing path.
    if lived:
        continuity = ""
        if journal_block:   # self-labeled ("RECENT REVIEW NOTES …"); empty on a fresh session
            continuity += journal_block
        if prev_context:
            continuity += "Session arc: %s\n" % prev_context
        body = ""
        if continuity:
            body += "<continuity>\n%s</continuity>\n\n" % continuity
        if node_catalog:
            body += "<node_catalog>\n%s\n</node_catalog>\n\n" % node_catalog
        if scout_legend:    # explains the <scout_notes> inside the timeline
            body += "%s\n" % scout_legend
        body += "<timeline>\n%s</timeline>\n" % timeline
    else:
        body = ""
        if journal_block:   # self-labeled
            body += "%s\n\n" % journal_block
        if prev_context:
            body += "### Session Context\n%s\n\n" % prev_context
        if node_catalog:
            body += "### %s\n" % node_catalog
        body += "### Conversation Timeline\n\n%s\n" % timeline
    return preamble, body, node_catalog, cataloged_ids


def _xml_escape(s):
    """Escape the three XML-significant chars so message/tool text can't malform
    the lived-sequence timeline or forge tags (e.g. a '</user>' or '<turn>'
    substring in a user prompt, or 'Bash: a > b' / 'Grep: <svg>' in a tool cue)."""
    return (s or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _lived_sequence_enabled():
    """A/B flag for the lived-sequence timeline (piece 1, S1E code-half).

    OFF by default. The eval flips it per arm via the env var; tests pass the
    `lived_sequence` param to `_build_user_content` explicitly. See
    docs/S1-SCRIBE-REDESIGN.md §10.3.1.
    """
    return os.environ.get('BRAIN_S1E_LIVED_SEQUENCE', '') in ('1', 'true', 'True')


def _render_markdown_timeline(brain, messages):
    """The long-standing messages-only timeline (the A/B control arm).

    Extracted verbatim from `_build_user_content` so the lived-sequence path can
    branch beside it; output is byte-identical to the pre-piece-1 builder.
    Per-turn `[trace:<hex>]` markers (v29): each USER/ASSISTANT line carries the
    trace_event.id; the encoder copies these into `source_refs` (sparse, 1-3).
    """
    import re
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    timeline = ""
    turn_num = 0
    i = 0

    def _fmt_trace(tid):
        return "[trace:%s]" % tid if tid else "[trace:?]"

    while i < len(messages):
        m = messages[i]
        if m.get("role") == "user":
            turn_num += 1
            user_content = (m.get("content") or "")[:ENCODING_AGENT['message_display_limit']]
            turn_id = m.get("id", "")
            user_trace = m.get("trace_id")

            timeline += "[TURN %d]\n" % turn_num
            timeline += "USER %s: \"%s\" (turn_id: %s)\n" % (
                _fmt_trace(user_trace), user_content, turn_id)

            judge_output = m.get("judge_output")
            if judge_output and judge_output != '(no selection)':
                ref_ids = re.findall(r'id:([a-z0-9_]{6,8})', judge_output)
                if ref_ids:
                    dal = brain._nodes
                    refs = []
                    for rid in ref_ids:
                        title = (dal.get_title(rid) or rid)[:50]
                        refs.append('%s ("%s")' % (rid, title))
                    timeline += "SURFACED: %s\n" % ", ".join(refs)

            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                asst_msg = messages[i + 1]
                asst = (asst_msg.get("content") or "")[:ENCODING_AGENT['message_display_limit']]
                asst_trace = asst_msg.get("trace_id")
                timeline += "ASSISTANT %s: \"%s\"\n" % (_fmt_trace(asst_trace), asst)
                i += 1

            timeline += "\n"
        i += 1
    return timeline


def _render_lived_sequence_timeline(brain, session_id, messages, streams=None,
                                    scout_notes=None):
    """The XML lived sequence — messages + tool actions interleaved (piece 1).

    `scout_notes` (optional): {user_trace_id: [line, ...]} from _map_scout_notes
    — scout findings rendered inside the turn they cite (<scout_notes>), after
    the actions, before the provenance. None → no annotation blocks.

    Reads through the existing `recall_episodes` door (the conversational lens
    over s0 traces), NOT a bespoke DAL query — docs/S1-SCRIBE-REDESIGN.md §10.2.
    Tool actions arrive as `tool_result` episodes whose `summary` already carries
    the per-tool cue ("Edit: foo.py", "Bash: …"). Window-matched to the control
    arm (trimmed to the same number of user turns as `messages`).

    `streams` (optional) is the pre-gathered {'surface','encode','touched'} dict from
    one trace_links.gather call, threaded in so the catalog and the <provenance>
    block share a single pull. None → _turn_links gathers its own.

    surfaced/judge_output is deliberately NOT rendered here — it belongs to the
    <provenance> block (piece 2). Piece 1 is a pure timeline read.
    """
    from servers.scales.s1.encode_contract import ENCODING_AGENT, LIVED_SEQUENCE_PULL
    lim = ENCODING_AGENT['message_display_limit']
    n_turns = sum(1 for m in (messages or []) if m.get('role') == 'user') or 20

    try:
        episodes = brain.recall_episodes(
            session_id=session_id,
            ref_type=['user_message', 'assistant_message', 'tool_result'],
            sort_order='desc', limit=LIVED_SEQUENCE_PULL)['episodes']
    except Exception as e:
        print('[s1e] ERROR reading lived sequence, falling back to markdown: %s' % e, flush=True)
        return _render_markdown_timeline(brain, messages)

    # Most-recent-N pulled desc; sort chronological so turns read in order.
    episodes = sorted(episodes, key=lambda e: e.get('created_at') or '')

    # Group into turns: a user_message opens a turn; the assistant_message + the
    # tool_results before the next user_message belong to it. A leading non-user
    # event (rare) opens an orphan turn so nothing is dropped.
    turns = []
    cur = None
    for e in episodes:
        rt = e.get('ref_type')
        if rt == 'user_message' or cur is None:
            cur = {'user': None, 'assistant': None, 'actions': []}
            turns.append(cur)
        if rt == 'user_message':
            cur['user'] = e
        elif rt == 'assistant_message':
            cur['assistant'] = e
        elif rt == 'tool_result':
            cur['actions'].append(e)

    turns = turns[-n_turns:]  # window-match the control arm

    def _text(ep, cap=None):
        # Full message body lives in metadata['content'] (≤4000); `summary` is the
        # 200-char display truncation. Mirror get_session_turns (dal.py): prefer
        # content, fall back to summary. Truncate to `cap` (default: the display
        # limit; already-encoded turns pass the trim cap), mark the cut with an
        # ellipsis, then escape.
        meta = ep.get('metadata')
        body = (meta.get('content') if isinstance(meta, dict) else None) or ep.get('summary') or ''
        cap = cap or lim
        cut = body[:cap] + ('…' if len(body) > cap else '')
        return _xml_escape(cut)

    # Piece 2: per-turn <provenance> — the trace↔node links (what recall surfaced
    # / what prior runs encoded, joined by stop). Guarded: any failure degrades to
    # the piece-1 timeline (no provenance), never breaks the lived arm.
    links, frontier = _turn_links(brain, session_id, turns, streams=streams)

    # «tag» locality (fork #2): each provenance id ref carries its 1-line title so
    # the encoder reads WHERE a node came up without scanning the catalog. Titles
    # are fetched NAKED (cheap; the catalog already holds the rich bodies — this is
    # a title-only lookup) in one batched get_bulk over the window's referenced ids.
    # Guarded: a stub brain (tests, no _nodes) or any failure → bare refs, never a
    # broken render.
    titles = {}
    ref_ids = set()
    for lk in links.values():
        ref_ids.update(lk.get('surfaced') or ())
        ref_ids.update(lk.get('encoded') or ())
        ref_ids.update(lk.get('authored') or ())
    if ref_ids:
        try:
            titles = {nid: (row.get('title') or '')
                      for nid, row in brain._nodes.get_bulk(list(ref_ids)).items()}
        except Exception:
            titles = {}

    # Already-encoded turns render as trimmed context stubs (Tom's 3.2): the
    # scribe's read is the unencoded tail; covered turns are there for cross-turn
    # pattern/contradiction catching, with their full substance living in the
    # catalog (the encoded nodes). The `encoded` attr states coverage on the turn
    # itself — it replaces the old provenance `✓` marker. No attr when the
    # trace-link join is unavailable (degraded piece-1 path — coverage unknown).
    trim = ENCODING_AGENT.get('encoded_turn_trim', 300)

    out = ""
    for n, t in enumerate(turns, 1):
        uid = (t['user'] or {}).get('id') if t['user'] else None
        link = links.get(uid) if uid else None
        enc_attr, cap = '', None
        if link is not None:
            is_enc = bool(link.get('encoded_by'))
            enc_attr = ' encoded="%s"' % ('true' if is_enc else 'false')
            if is_enc:
                cap = trim
        out += '<turn n="%d"%s>\n' % (n, enc_attr)
        if t['user']:
            out += '  <user trace="%s">%s</user>\n' % (
                t['user'].get('id', ''), _text(t['user'], cap))
        if t['assistant']:
            out += '  <assistant trace="%s">%s</assistant>\n' % (
                t['assistant'].get('id', ''), _text(t['assistant'], cap))
        if t['actions']:
            out += '  <actions>\n'
            for a in t['actions']:
                # tool cues have no metadata['content'] — the summary IS the cue
                out += '    %s\n' % _xml_escape(a.get('summary') or '')
            out += '  </actions>\n'
        notes = scout_notes.get(uid) if (scout_notes and uid) else None
        if notes:
            out += '  <scout_notes>\n'
            for ln in notes:
                out += '    %s\n' % ln     # lines pre-escaped by _scout_note_line
            out += '  </scout_notes>\n'
        prov = _render_provenance(links, frontier, t, n - 1, titles)
        if prov:
            out += '  <provenance>%s</provenance>\n' % prov
        out += '</turn>\n\n'
    return out


def _short_refs(ids, titles=None):
    """Node ids as 8-char `id:` refs, each carrying its 1-line «tag» (title) when
    known — the locality affordance (fork #2): WHERE a node came up in the
    timeline, complementary to the catalog's full WHAT-it-is. The 8-char short
    still matches the catalog so a ref dereferences there; the «tag» saves the
    scan. Falls back to a bare `id:` when no title is mapped (stub brains, or an
    id the title fetch missed)."""
    titles = titles or {}
    out = []
    for i in ids:
        short = str(i)[:8]
        # Escape the title + collapse whitespace: it lands inside the strict
        # per-turn timeline XML, next to message/action text that _xml_escape()
        # already guards — a title with '<'/'>'/'&' (or a forged '</provenance>')
        # must not malform it, and a stray newline must not break the one-line
        # provenance.
        t = _xml_escape(' '.join((titles.get(i) or '').split()))
        out.append('id:%s «%s»' % (short, t) if t else 'id:%s' % short)
    return ' '.join(out)


def _turn_links(brain, session_id, turns, streams=None):
    """Compute the trace↔node link map + per-run frontier index for the window.

    Returns (links, frontier):
      links    — {user_trace_id: {surfaced, encoded, encoded_by}} from trace_links.
      frontier — {encoded_by_run_id: window-index of the LAST turn that run covers},
                 so the render shows a run's full encoded id-list once (at its
                 frontier turn, adjacent to the unencoded boundary) and a light ✓
                 on the earlier covered turns — no 5× repetition (which the design
                 warns would nudge dense source_refs).

    `streams` (optional): the pre-gathered {'surface','encode','touched'} dict, so
    the catalog and this share one gather. None → gather here.

    Guarded: any failure returns ({}, {}) → the timeline renders exactly as piece
    1 did. Also the path the piece-1 stub brains take (no query_traces).
    """
    try:
        from servers.scales.s1.trace_links import gather, nodes_for_traces
        targets = [t['user'] for t in turns if t['user']]
        st = streams if streams is not None else gather(brain, session_id)
        links = nodes_for_traces(st['surface'], st['encode'], targets,
                                 touched_traces=st['touched'])
    except AttributeError:
        # Expected, quiet: a stub brain (tests) without query_traces. Degrade to
        # the piece-1 timeline (no provenance) — production brains always have it.
        return {}, {}
    except Exception as e:
        # A real failure (trace-contract drift, malformed record) — LOUD, then
        # degrade (mirrors the muster fallback above; brain.errors is the surface,
        # a bare print is not). Provenance is advisory; never block the timeline.
        try:
            brain._log_error('s1e_provenance', e,
                             'trace-link provenance failed; timeline renders without it')
        except Exception:
            pass
        return {}, {}

    frontier = {}
    for idx, t in enumerate(turns):
        uid = (t['user'] or {}).get('id')
        eb = (links.get(uid) or {}).get('encoded_by') if uid else None
        if eb:
            frontier[eb] = idx  # ascending walk → last index wins
    return links, frontier


def _render_provenance(links, frontier, turn, idx, titles=None):
    """One <provenance> line for a turn — REAL refs only, no coverage markers.

    Coverage lives on the turn itself (the `encoded="true|false"` attribute the
    timeline render stamps), so provenance never says "covered, nothing to show"
    — the old `✓` marker is gone. What renders: `surfaced:` refs (per-turn, 1:1),
    the owning run's full id-list ONCE at the run's frontier turn (an edge-only
    run has an empty id set → nothing), and `encoded(Anchor):` — the turn-local
    set Anchor wrote mid-turn (link['authored'] = created ∪ revised, joined by
    stop; empty in replayed eval corpora). `titles` maps id→title for the «tag»
    locality. Returns '' when there's nothing real to show.
    """
    uid = (turn['user'] or {}).get('id') if turn['user'] else None
    link = links.get(uid) if uid else None
    if not link:
        return ''
    parts = []
    if link['surfaced']:
        parts.append('surfaced: %s' % _short_refs(link['surfaced'], titles))
    eb = link['encoded_by']
    if eb and frontier.get(eb) == idx and link['encoded']:
        parts.append('encoded(S1S): %s' % _short_refs(link['encoded'], titles))
    if link.get('authored'):
        parts.append('encoded(Anchor): %s' % _short_refs(link['authored'], titles))
    return ' | '.join(parts)


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
            'ref_id': os.path.join(brain_tmp_dir(), 'brain-encoding-prompt-%d.json' % counter),
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
    return entry_body


def _save_session_context(brain, dispatch_fn, session_id, final_text):
    """Extract SESSION_CONTEXT from encoder output, append to per-session journey.

    2026-05-02 (Frame Phase 2.5): writes per-session key
    `session_context_{session_id}` instead of the global `session_context`.
    The previous global was a parallel-session leak — encoder writes from
    session A and session B clobbered each other. Per-session keys mirror
    the existing `encoding_journal_{session_id}` pattern.
    """
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    if not session_id:
        return  # nowhere to write — should not happen in production paths
    limit = ENCODING_AGENT.get('session_context_limit', 800)
    key = 'session_context_' + session_id
    for line in final_text.split('\n'):
        stripped = line.strip()
        if stripped.upper().startswith('SESSION_CONTEXT:'):
            new_context = stripped[len('SESSION_CONTEXT:'):].strip()
            if new_context:
                existing = brain.session_context_for(session_id)
                # Newline-separated entries instead of pipe noise
                combined = (existing + '\n' + new_context) if existing else new_context
                if len(combined) > limit:
                    # Truncate at line boundaries from the front
                    truncated = combined[len(combined) - limit:]
                    nl_idx = truncated.find('\n')
                    if nl_idx >= 0 and nl_idx < 60:
                        truncated = truncated[nl_idx + 1:]
                    combined = truncated
                dispatch_fn('set_config', {'key': key, 'value': combined})
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
