"""S1 Turn Encoder — LLM-powered brain encoding via Sonnet API.

Scale: S1 (Turn integration, every 5th stop)
Chain: s1e (encode)
Interaction: 's1e' in interactions table (learnable boundary; 'encoding_agent' was the legacy name)

Triggered by: encoding gate in hook_post_response_track (daemon_hooks.py)
Reads: traces (conversation turns), brain nodes (catalog), interactions table
Writes: nodes/edges via dispatch, traces (O/K), journal + session context via config
"""

import os
import re
import time

from servers.scales.dispatch import load_env
from servers.scales.runner import run_llm_loop
from servers.trace_contract import build_delta_metadata


def scribe_journal_kwargs(session_id):
    """The Scribe's binding shape beyond scale/source: session-walled residue
    plus the `## Arc` opt-in. One spelling, used by S1Scribe.journal (the
    production binding, which adds the source) and by _journal below."""
    return {'session_id': session_id, 'arc': True}


def _journal(brain, session_id=''):
    """A SOURCE-LESS Scribe-shaped binding for the standalone callers (evals,
    tests) that carry no unit — tell/ask lines stay plain notes there. The
    production binding is `S1Scribe.journal`, handed in as `journal=`.
    Decorate-only sites may pass brain=None."""
    from servers.scales.journal import JournalBinding
    return JournalBinding(brain, scale='s1', **scribe_journal_kwargs(session_id))


def run_encoding(brain, dispatch_fn, counter, session_id, log_fn=None,
                 journal=None):
    """S1 turn encoder: gather → prompt → trace O/K → LLM loop → post-process.

    Args:
        brain: Brain instance (READ-ONLY)
        dispatch_fn: function(cmd, args) for writes (routes through daemon TCP)
        counter: Stop counter value
        session_id: Session ID (required)
        log_fn: Optional logging function

    Returns:
        dict with encoding results summary.
    """
    def _log(msg):
        print("[s1e] %s" % msg, flush=True)
        if log_fn:
            log_fn("S1 encode: %s" % msg)

    from servers.scales.s1.encode_contract import (
        ENCODING_AGENT, SCRIBE_RUN_DEADLINE_SECONDS)

    t0 = time.time()
    profile = []

    def _step(name):
        profile.append((name, int((time.time() - t0) * 1000)))

    # ONE binding per run — decoration, continuity and harvest all read it.
    # The production caller (S1Scribe) hands in its sourced binding; the
    # standalone callers get the source-less Scribe shape.
    journal = journal or _journal(brain, session_id)

    load_env()
    _step("env_loaded")

    from ..runner import make_client
    try:
        client = make_client()
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
    # View policy (encoder_view.py): same once-per-run resolution. Lived-arm
    # only by construction — every policy surface (catalog aging, provenance
    # verbs, action stubs) lives in the lived render; the markdown arm has none.
    from servers.scales.s1.encoder_view import view_policy_enabled
    view = lived and view_policy_enabled()
    # Conversation time, resolved ONCE per run (like the arm flags above) —
    # the catalog's relative headers and the timeline's now/age stamps must
    # agree on the instant.
    view_now = _conversation_now_safe(brain, session_id, messages) if view else None

    # 2. Build prompt (learnable boundary — DB override overlaid on the code
    # default by the resolver)
    enc_instructions = brain.get_interaction_prompt('s1e')
    # K-provenance stamp, resolved at the same moment as the template so the
    # delta trace records the K this run actually used.
    enc_stamp = brain.get_interaction_stamp('s1e')
    # The gist is its own K (`s1e_gist`, read in _build_user_content); its
    # stamp rides the same delta so a run's provenance names both texts.
    gist_stamp = brain.get_interaction_stamp('s1e_gist')
    # Per-version config rides in the interaction's parameters JSON (the
    # K-store): `effort` maps to the API's output_config.effort; `model`
    # picks the encoder model. Lets an effort or model change ship as a
    # prompt version (A/B-able via ab_encode's parameters injection), not a
    # code edit. Resolved config is total — the defaults live in
    # encode_contract.S1E_INTERACTION_DEFAULT, nowhere else.
    enc_cfg = brain.get_interaction_config('s1e')
    enc_effort = enc_cfg['effort']
    enc_model = enc_cfg['model']
    system_prompt = _build_system_prompt(
        prompt_instructions=enc_instructions or None, lived=lived,
        journal=journal)

    # 2a. Catalog first (both arms), then the body — the two halves profile
    # separately and the body reuses the catalog's trace streams.
    catalog_text, catalog_ids, streams = _build_catalog(
        brain, messages, session_id, lived, view_policy=view, now=view_now)
    _step("catalog(%d ids)" % len(catalog_ids))

    # 2b. Body assembly. Lived: the XML lived-sequence timeline; control: the
    # legacy markdown body.
    user_preamble, user_content, _cat_text2, _cat_ids2 = _build_user_content(
        brain, messages, counter, session_id, lived_sequence=lived,
        precomputed=(catalog_text, catalog_ids, streams), view_policy=view,
        view_now=view_now, journal=journal)
    _step("prompt(preamble=%d chars, body=%d chars)" % (
        len(user_preamble), len(user_content)))

    # 3. Get tools (S1-specific subset)
    tools = _get_tool_schemas()
    _step("tools(%d)" % len(tools))

    # 4. Record the assembled prompt via the payload recorder (dashboard +
    # post-hoc eval read it back by pointer/chain; per-arm eval brains land
    # it inside their own db_dir by construction). The O trace below carries
    # the returned db_dir-relative pointer as its ref_id.
    enc_chain = 's1e-%s-%d' % (session_id[:8], counter)
    prompt_pointer = brain.record_payload(
        enc_chain, 'prompt', user_preamble + "\n\n" + user_content)

    # 5. Write S1 traces: O (observation) and K (knowledge)
    _write_pre_traces(brain, dispatch_fn, messages, user_content, counter,
                      session_id, catalog_ids=catalog_ids,
                      prompt_pointer=prompt_pointer)

    # 6. Run generic LLM loop (shared with S2+)
    _log("calling %s with %d tools, %d chars context, effort=%s..." % (
        enc_model, len(tools), len(user_content), enc_effort or 'default(high)'))
    _log("PROFILE so far: %s" % " → ".join("%s:%dms" % (n, t) for n, t in profile))

    try:
        result = run_llm_loop(
            client=client,
            model=enc_model,
            effort=enc_effort,
            max_tokens=ENCODING_AGENT['max_tokens'],
            max_rounds=ENCODING_AGENT.get('max_rounds', 5),
            system_prompt=system_prompt,
            user_content=user_content,
            user_preamble=user_preamble,
            tools=tools,
            dispatch_fn=dispatch_fn,
            log_fn=_log,
            record_round_fn=brain.round_recorder(enc_chain),
            deadline_seconds=SCRIBE_RUN_DEADLINE_SECONDS)
    except Exception as e:
        # The LLM loop itself died — no writes happened this run. Loud by
        # default: errors table + a `encoding_run_failed` delta so the run is
        # visible (dashboard, forensics) WITHOUT claiming turn coverage —
        # trace_links joins on `encoding_run` only, so a failed run never
        # marks turns encoded. The print stays for daemon.log grep-ability.
        _step("FAILED")
        profile_str = " → ".join("%s:%dms" % (n, t) for n, t in profile)
        print('[s1e] ERROR: Sonnet API call failed: %s PROFILE: %s' % (e, profile_str), flush=True)
        brain._log_error('s1e_run_failed', e,
                         'session=%s stop=%d — LLM loop failed; writes from '
                         'completed rounds (if any) are kept, turns stay '
                         'unencoded and will retry' % (session_id[:8], counter))
        # Salvage the actions that ran before the failure (RunLoopError
        # carries them). Bounding + record shape live in the traces layer
        # (build_failed_run_metadata) — this consumer just hands them over.
        # The full conversation at failure time (RunLoopError.msgs, tool
        # results already capped) is recorded as the failed_run payload —
        # the 2AM story: the next 6M-char-class incident is diagnosable
        # from the trace row + one file, in normal mode. The traces layer
        # owns the payload shape and caps; round-0 failures arrive unwrapped
        # (no msgs) and record nothing — the prompt kind is that half.
        from servers.trace_contract import (build_failed_run_metadata,
                                            build_journal_note_metadata)
        failed_ptr = brain.record_failed_run(enc_chain, e)
        with brain.loud('s1e_failed_trace_write', 'recording encoding_run_failed delta'):
            dispatch_fn('trace_append', {
                'chain_id': enc_chain, 'scale': 's1', 'event_type': 'delta',
                'ref_type': 'encoding_run_failed',
                'summary': 'FAILED: %s' % str(e)[:200],
                'metadata': build_failed_run_metadata(
                    error=e, stop_counter=counter,
                    inputs_processed=len(messages),
                    partial_actions=getattr(e, 'partial_actions', None),
                    payload_pointer=failed_ptr),
                'session_id': session_id,
            })
        # Reflective residue (docs/TRACE-MODES-DESIGN.md §Failed-run residue):
        # a dead run must leave a journal note so the retry's continuity
        # window shows the failure instead of amnesia. Same journal_note
        # shape write_journal_notes emits — the read door needs no change.
        with brain.loud('s1e_failed_journal_note',
                        'recording failure journal note'):
            dispatch_fn('trace_append', {
                'chain_id': enc_chain, 'scale': 's1', 'event_type': 'delta',
                'ref_type': 'journal_note', 'ref_id': 'encoding-run-failure',
                'summary': 'run FAILED before finishing',
                'metadata': build_journal_note_metadata(
                    note='stop %d run FAILED: %s — turns stay unencoded and '
                         'retry; writes from completed rounds (if any) were '
                         'kept' % (counter, str(e)[:200]),
                    tag='failure'),
                'session_id': session_id,
            })
        return {"error": str(e), "profile": profile}

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

    # Oversized-tool-result scan — the runner truncates and marks the action
    # (it has no brain reference); the loud errors-table entry lands here.
    for a in (result.get('action_details', []) + result.get('read_calls', [])):
        if a.get('result_truncated'):
            brain._log_error(
                's1e_oversized_tool_result',
                RuntimeError('%s returned %d chars — truncated to cap' % (
                    a.get('tool'), a.get('result_chars', -1))),
                'head: %s; session=%s stop=%d' % (
                    (a.get('result_head') or '')[:200], session_id[:8], counter))

    # 7. Post-process (S1-specific: journal/residue, session context).
    # Guarded as a whole: the loop's writes already landed — a post-process
    # failure (e.g. `database is locked`) must not swallow step 8's delta
    # trace, which coverage attribution and the dashboard both read.
    final_text = result.get('final_text', '')
    journal_entry = ''
    with brain.loud('s1e_postprocess',
                    'journal/arc post-process failed — writes intact, '
                    'continuing to delta trace; session=%s stop=%d'
                    % (session_id[:8], counter)):
        if lived:
            # New residue (Piece 4): the `## Review` note contract, SESSION-BOUND.
            # write_journal_notes extracts the fence + writes one journal_note
            # trace per note, all sharing enc_chain (this run); session_id walls
            # continuity to this conversation. Replaces the legacy blob. The arc
            # is a SEPARATE object with its own journal component: the encoder
            # emits a `## Arc` fenced one-liner (render_journal_arc_block) and
            # write_session_arc accumulates it into session_context_{sid} —
            # replacing the legacy SESSION_CONTEXT:-line parse, which v26's
            # prompt no longer emits (the A/B arc-regression fix). Both write
            # doors are failure-isolated internally.
            #
            # INTENTIONAL flag-on side effect: not writing the blob leaves the
            # Frame's `## Recent moves` (which reads the legacy encoding_journal
            # blob) empty. That's the deferred Frame-slot cut previewing — the cut
            # lands at activation (replacement-before-removal). No eval confound:
            # the Frozen-Corpus sweep queries with a FRESH session_id, so Recent
            # moves is empty in BOTH arms there regardless of this flag.
            journal.harvest(
                final_text, enc_chain,
                arc_limit=ENCODING_AGENT.get('session_context_limit', 800))
            journal_entry = ''
        else:
            journal_entry = _save_journal(brain, dispatch_fn, session_id, counter, final_text) or ''
            _save_session_context(brain, dispatch_fn, session_id, final_text)

    # 8. Delta trace — unified shape across S1E + S2 encoders. Own guard:
    # losing this trace makes the run invisible (dashboard "0 actions") and
    # pushes coverage attribution onto the next run — so a failure here is
    # loud, and post-process failures above can't reach it.
    # Outcomes: count write actions by tool (remember / revise / connect / …).
    with brain.loud('s1e_delta_write',
                    'encoding_run delta lost — run shows 0 actions on '
                    'dashboard, coverage attribution falls to next run; '
                    'session=%s stop=%d' % (session_id[:8], counter)):
        action_details = result.get('action_details', [])
        outcomes = {}
        for a in action_details:
            tool = a.get('tool', 'unknown')
            outcomes[tool] = outcomes.get(tool, 0) + 1

        # enc_chain computed above (post-process) — reused here for the delta.
        # Which K produced this Δ — the stamp (fingerprint + source + version,
        # resolved with the template at run start) rides the metadata; the
        # install-local rowid rides the trace row for display. Lets higher
        # scales A/B prompt versions from production traces (the whole point
        # of interactions-as-K-store).
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
            model=result.get('model', ''),
            interaction_version=enc_stamp['version'],
            interaction_fingerprint=enc_stamp['fingerprint'],
            interaction_source=enc_stamp['source'],
            gist_interaction_version=gist_stamp['version'],
            gist_interaction_fingerprint=gist_stamp['fingerprint'],
            gist_interaction_source=gist_stamp['source'],
            stop_counter=counter,
        )
        dispatch_fn('trace_append', {
            'chain_id': enc_chain, 'scale': 's1', 'event_type': 'delta',
            'ref_type': 'encoding_run',
            'interaction_id': enc_stamp['id'],
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
    return result


# ── S1-Specific Helpers ──


def _gather_messages(brain, session_id):
    """Fetch recent messages for the current session via the traces layer.

    Returns: [{id, turn_label, role, trace_id, content, timestamp, judge_output}]
    Uses brain.get_conversation() — single source of truth.
    """
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    limit = ENCODING_AGENT['max_messages']
    content_limit = ENCODING_AGENT['message_content_limit']

    try:
        turns = brain.get_conversation(session_id, limit=limit)
        if turns:
            for i, t in enumerate(turns):
                # `id` was a synthetic display label (turn-N). Keep it as
                # `turn_label` for human-facing render. `trace_id` (hex)
                # comes from the traces layer and is the load-bearing
                # reference the encoder copies into source_refs.
                t['turn_label'] = 'turn-%d' % i
                t['id'] = 'turn-%d' % i  # backward-compat for downstream readers
                t['content'] = (t.get('content', '') or '')[:content_limit]
            return turns
    except Exception as e:
        print('[s1e] CONVERSATION READ ERROR: %s' % e, flush=True)

    return []


def _build_system_prompt(prompt_instructions=None, lived=None, journal=None):
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
        except Exception as e:
            print('[s1e] WARNING: could not read seed prompt %s — using stub: %s'
                  % (prompt_path, e), flush=True)
            prompt = "You are the encoding agent. Encode focused nodes. Batch operations. 2-3 rounds."
    try:
        from servers.contract import generate_field_summary
        prompt += "\n\n## Available Fields (from contract)\n\n" + generate_field_summary()
    except Exception as e:
        print('[s1e] WARNING: could not load field summary: %s' % e, flush=True)

    # Residue contract (Piece 4, flag-gated): the WRITE-side instructions —
    # closing acts in §7.2 order (Encode → Arc → Review), then the closure
    # (terminal-turn + DONE) as the LAST block. THREE SEPARATE injects (the
    # closure must not be entangled with either block), placed at the end by
    # design: these are the encoder's final acts, so their instructions sit
    # where the action lands (recency). The arc block is the journal
    # mechanism's per-encoder opt-in second component — S1E opts in because it
    # owns the session arc (`session_context_{sid}` → Frame 'Current focus');
    # its absence was the v26 A/B gate-blocker (arc went dark on the lived
    # arm). Contract-owned text (single source across all encoders) — never
    # hardcoded in the s1e template, so it can't drift. Flag-off keeps the
    # legacy SESSION_CONTEXT:-line path, so this stays absent from the
    # control arm.
    if lived:
        try:
            prompt = (journal or _journal(None)).decorate_system(prompt)
        except Exception as e:
            print('[s1e] WARNING: could not inject arc/review block/closure: %s' % e, flush=True)
    return prompt


def _build_catalog(brain, messages, session_id, lived, view_policy=False,
                   now=None):
    """The catalog half of the encoding input, built before the body so
    run_encoding can profile the two halves separately. Gathers the trace
    streams once (lived); the same tuple threads into the timeline's
    <provenance>.
    `view_policy` (resolved once in run_encoding) turns on catalog aging in
    build_node_catalog.

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

    if view_policy and now is None:
        now = _conversation_now_safe(brain, session_id, messages)

    from servers.scales.s1.encoder_view import associated_stubs_enabled
    stubs_on = lived and associated_stubs_enabled()

    # One window pull shared by the catalog cutoff and the stub seeds — the
    # timeline render keeps its own (the pre-existing pair); this block must
    # not add a third. Quiet degrade mirrors window_first_turn's contract:
    # no readable window → no cutoff, no seeds.
    turns = None
    if view_policy or stubs_on:
        try:
            turns = _lived_turns(brain, session_id, _window_n_turns(messages))
        except Exception:
            turns = None

    # Window-aligned cutoff (arm D): the catalog ages against the same turn
    # window the timeline renders — widen the chat window and the full-depth
    # catalog widens with it. None (no datable head, or flag off) falls back
    # to round-based aging inside build_node_catalog.
    head = (window_first_turn(brain, session_id, messages, turns=turns)
            if view_policy else None)

    # Associated stubs (the subconscious): production
    # recall over the window's unencoded messages, catalog excluded, rendered
    # as the catalog's LAST entries tagged [associated]. Lived arm only (the
    # tag grammar lives on the widened catalog), flag-gated for the A/B.
    # Guarded loud: retrieval is advisory — never block the catalog.
    associated = None
    if stubs_on and turns is not None:
        try:
            from servers.scales.s1.encode_contract import (
                ASSOCIATED_TAG, PROVENANCE_TAGS, surfaced_ids_of)
            # Loud when the flag runs ahead of the prompt: the encoder can
            # only read the tag correctly once the active s1e prompt teaches
            # it — a comment can't enforce that ordering, this check can.
            # Advisory: logs, never blocks (the A/B arm decides).
            try:
                if ASSOCIATED_TAG not in (brain.get_interaction_prompt('s1e') or ''):
                    brain._log_error(
                        's1e_associated_stubs_untaught',
                        'BRAIN_S1E_ASSOCIATED_STUBS is on but the active s1e '
                        'prompt never mentions %s' % ASSOCIATED_TAG,
                        'stubs render untaught — activate the prompt version '
                        'that teaches the tag')
            except Exception:
                pass
            # Exactly the id set build_node_catalog will catalog (surfaced ∪
            # the provenance categories) — derived from the same parse + the
            # same PROVENANCE_TAGS keys, so exclusion can't drift from the
            # render. Empty ⇔ the catalog will be empty: no home for stubs.
            exclude = surfaced_ids_of(judge_outputs)
            for key, _tag in PROVENANCE_TAGS:
                exclude |= set((extra_ids or {}).get(key) or ())
            if exclude:
                associated = _associated_stub_ids(
                    brain, messages, session_id, exclude,
                    streams=streams, turns=turns)
        except Exception as e:
            try:
                brain._log_error('s1e_associated_stubs', e,
                                 'associated-stub recall failed; catalog renders without stubs')
            except Exception:
                pass
            associated = None

    try:
        node_catalog, cataloged_ids = build_node_catalog(
            judge_outputs, brain, extra_ids=extra_ids,
            scope=brain.session_scope(session_id), view_policy=view_policy,
            now=now, window_first_turn=head, associated_ids=associated)
    except Exception as e:
        print('[s1e] ERROR building node catalog: %s' % e, flush=True)
        node_catalog, cataloged_ids = '', set()
    return node_catalog, cataloged_ids, streams


def _conversation_now_safe(brain, session_id, messages):
    """Conversation time for the view policy's relative-time renders — replays
    inject historical [Current date:] prefixes; wall-clock would corrupt them
    (the S1 rule — tests/test_time_window_contract). Guarded: a stub brain
    without session machinery degrades to None (wall-clock render)."""
    try:
        from servers.clock import conversation_now
        ctx = brain.get_or_create_session(session_id)
        return conversation_now(messages=messages,
                                session_started_at=getattr(ctx, 'started_at', None),
                                brain=brain)
    except Exception:
        return None


def _associated_stub_ids(brain, messages, session_id, exclude_ids,
                         streams=None, turns=None):
    """The subconscious: ids of the K nodes production recall ranks nearest
    this window that the catalog does NOT already show: the encoder receives
    not just the conscious catalog but the associative penumbra recall ranked
    below the surface cut.

    Measured shape (eval/encode_moment_recall_probe.py, [thread:encoder-
    staleness]): one recall per UNENCODED window message, both voices — never
    a whole-window blob (needle absent from its top-200); turnmax aggregation
    (a node's score is its best single-message match — turnsum buried the
    needle at rank 18-20). Production recall is deliberate: LAF, edge walks,
    correction chains and the scope veil all apply — no parallel ranker.

    Time: NO `as_of`, deliberately. Every candidate value breaks something
    (code-review verified, 2026-08-21): a live conversation_now is
    operator-TZ and LAF's masks compare lexicographically against uniform
    UTC, skewing the field by the host offset; a replay's [Current date:]
    prefix predates every corpus node's wall-clock created_at, emptying the
    masked universe (ValueError each round); and any as_of couples the
    feature to BRAIN_RECALL_VARIANT=laf_v1 (the champion path raises). Live,
    as_of≈now is a no-op mask anyway. A harness replaying against a brain
    that holds FUTURE nodes relative to the replayed moment must pin the
    field itself (the probe passes the run's stored UTC trace timestamp).

    Reads are pure: mark_accessed=False — assembly is machinery, and marking
    would inflate access counts + fatigue with the encoder's own looking
    (the dashboard-observer precedent). Community nodes never render:
    recall's default NODE_TYPE_EXCLUSIONS drops them here, and
    build_node_catalog's community scan is the render chokepoint.

    `turns` (optional): the pre-pulled _lived_turns list threaded from
    _build_catalog so one window pull serves the cutoff and the seeds
    (None → pull here; standalone/test callers). Coverage joins through
    _turn_links on the pre-gathered `streams`; a turn with no link is real
    (degraded join / orphan) and counts as unencoded — the probe's
    convention.

    Returns rank-ordered id list (may be empty). Raises to the caller's
    guard on read failure — advisory, never blocks the catalog.
    """
    from servers.scales.s1.encoder_view import (
        ASSOCIATED_QUERY_CAP, ASSOCIATED_RECALL_LIMIT, ASSOCIATED_SEED_CAP,
        ASSOCIATED_STUBS_K)
    from servers.scales.s1.surface_contract import recall_score

    if turns is None:
        turns = _lived_turns(brain, session_id, _window_n_turns(messages))
    links, _frontier = _turn_links(brain, session_id, turns, streams=streams)

    def _body(ep):
        meta = ep.get('metadata')
        body = ((meta.get('content') if isinstance(meta, dict) else None)
                or ep.get('summary') or '')
        return body[:ASSOCIATED_QUERY_CAP].strip()

    seeds = []
    for t in turns:
        uid = (t['user'] or {}).get('id') if t['user'] else None
        link = links.get(uid) if uid else None
        if link is not None and link.get('encoded_by'):
            continue    # covered turn — a prior run already encoded it
        for ep in (t['user'], t['assistant']):
            text = _body(ep) if ep else ''
            if text:
                seeds.append(text)
    # Dedup (repeated 'ok'/'continue' turns), then keep the NEWEST seeds —
    # the tail turns are the encoder's working material. The cap bounds the
    # sequential recall loop when every turn reads unencoded (first encode
    # of a long session, or a degraded trace join).
    seeds = list(dict.fromkeys(seeds))[-ASSOCIATED_SEED_CAP:]
    if not seeds:
        return []

    exclude = {(i or '')[:8] for i in exclude_ids if i}
    best = {}
    for text in seeds:
        r = brain.recall(query=text, limit=ASSOCIATED_RECALL_LIMIT,
                         session_id=session_id, source='s1e_associated',
                         mark_accessed=False)
        for h in (r.get('results') or ()):
            nid = str(h.get('id') or '')
            if not nid or nid[:8] in exclude:
                continue
            score = recall_score(h)
            if score > best.get(nid, -1.0):
                best[nid] = score    # turnmax, never sum

    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return [nid for nid, _s in ranked[:ASSOCIATED_STUBS_K]]


def _build_user_content(brain, messages, counter, session_id, lived_sequence=None,
                        precomputed=None, view_policy=None,
                        view_now=None, journal=None):
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
        - catalog_text: rendered catalog block.
        - catalog_ids: set of node ids in the catalog.

    `precomputed` — the (node_catalog, cataloged_ids, streams) tuple from
    _build_catalog, when run_encoding already built it. None → build here
    (tests / standalone callers).
    `view_policy` — the encoder_view flag, resolved once in run_encoding and
    threaded in; None → read the env (tests / standalone callers). Lived-arm
    only; shapes the timeline render (aging rides the precomputed catalog).
    `view_now` — the run's conversation-time instant (resolved once alongside
    the flag); None → resolve here (standalone callers).
    """
    # A/B flag (piece 1, docs/S1-SCRIBE-REDESIGN.md §10.3.1): OFF (default) =
    # markdown messages-only timeline + surfaced-only catalog (the long-standing
    # path, untouched). ON = the new input as ONE unit: the XML lived-sequence
    # timeline (messages + tool actions + per-turn <provenance>)
    # AND the widened catalog (surfaced ∪ encoded ∪ authored ∪ recalled, tagged).
    # Both consume the same trace streams, gathered ONCE and threaded into both.
    lived = _lived_sequence_enabled() if lived_sequence is None else lived_sequence
    if view_policy is None:
        from servers.scales.s1.encoder_view import view_policy_enabled
        view_policy = lived and view_policy_enabled()
    if precomputed is not None:
        node_catalog, cataloged_ids, streams = precomputed
    else:
        node_catalog, cataloged_ids, streams = _build_catalog(
            brain, messages, session_id, lived, view_policy=view_policy)

    if lived:
        conv_now = view_now
        if view_policy and conv_now is None:
            conv_now = _conversation_now_safe(brain, session_id, messages)
        timeline = _render_lived_sequence_timeline(
            brain, session_id, messages, streams=streams,
            view_policy=view_policy, now=conv_now)
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
        journal_block = (journal or _journal(brain, session_id)).continuity()
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
    if lived and os.environ.get('BRAIN_S1E_LISTS_PREAMBLE', '0') in ('1', 'true', 'True'):
        # The operating-guide arm (flag-gated input change, like the view policy
        # and associated stubs): the encoder's first reply carries its four lists
        # as text, then the tool call they call for, in the same reply.
        preamble = (
            "I'm encoding what I've just observed. I read everything below; my "
            "first reply opens with my four lists — changes, targets, fetch, new — "
            "and ends in the tool call they call for.\n"
        )
    elif lived:
        # First person — matches the v-next system prompt's register (the encoder
        # speaks as itself: "This is me encoding my own memory").
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
    # Failed-run residue (docs/TRACE-MODES-DESIGN.md §Failed-run residue):
    # encoding_run_failed traces newer than the last successful run mean the
    # turns below were ALREADY attempted and died — the retry must encode
    # with knowledge of the failure, not amnesia.
    failed_block = _render_failed_encodes(brain, session_id)

    if lived:
        continuity = ""
        if journal_block:   # self-labeled ("RECENT REVIEW NOTES …"); empty on a fresh session
            continuity += journal_block
        if prev_context:
            continuity += "Session arc: %s\n" % prev_context
        body = ""
        if continuity:
            # Escaped like every other free-form string in the lived prompt:
            # the producer view carries operator-authored answer text, which
            # must not be able to close this block or forge a sibling tag.
            body += "<continuity>\n%s</continuity>\n\n" % _xml_escape(continuity)
        if failed_block:
            body += "<failed_encodes>\n%s</failed_encodes>\n\n" % failed_block
        if node_catalog:
            body += "<node_catalog>\n%s\n</node_catalog>\n\n" % node_catalog
        # The gist — the last instruction before the timeline, its own K
        # (`s1e_gist`): read through the resolver so an override can edit it
        # or turn it off (`enabled`), and its fingerprint says what the
        # encoder actually read. Config is total by construction — subscript.
        if brain.get_interaction_config('s1e_gist')['enabled']:
            gist = brain.get_interaction_prompt('s1e_gist')
            if gist:
                body += gist.rstrip('\n') + "\n\n"
        # `now=` stamp (view policy): the absolute anchor that makes every
        # relative label below invertible — and the current-time declaration
        # the encoder's date resolution never had before the view policy.
        now_attr = ''
        if view_policy and conv_now is not None:
            from servers.scales.s1.encoder_view import timeline_now_attr
            now_attr = timeline_now_attr(conv_now)
        body += "<timeline%s>\n%s</timeline>\n" % (now_attr, timeline)
    else:
        body = ""
        if journal_block:   # self-labeled
            body += "%s\n\n" % journal_block
        if failed_block:
            body += "### Previous encode attempts FAILED\n%s\n" % failed_block
        if prev_context:
            body += "### Session Context\n%s\n\n" % prev_context
        if node_catalog:
            body += "### %s\n" % node_catalog
        body += "### Conversation Timeline\n\n%s\n" % timeline
    return preamble, body, node_catalog, cataloged_ids


def _render_failed_encodes(brain, session_id, cap=3):
    """Render this session's encoding_run_failed traces newer than the last
    successful encoding_run — a short block (≤`cap` newest) so the retry sees
    what its predecessor attempted and how it died. Reads go through the
    query_traces door (session-scoped pulls ignore wall-clock windows —
    eval-replay safe); created_at comparison is transaction-time on both
    sides. Returns '' when there's nothing to show; never raises."""
    try:
        # get_by_ref_type orders created_at DESC — limit bounds the pull:
        # newest failures only, and one row is enough for the success mark.
        failed = (brain.query_traces(ref_type='encoding_run_failed',
                                     session_id=session_id, hours=None,
                                     limit=cap * 2)
                  or {}).get('events') or []
        if not failed:
            return ''
        ok = (brain.query_traces(ref_type='encoding_run',
                                 session_id=session_id, hours=None, limit=1)
              or {}).get('events') or []
        last_ok = max((e.get('created_at') or '' for e in ok), default='')
        fresh = sorted((e for e in failed
                        if (e.get('created_at') or '') > last_ok),
                       key=lambda e: e.get('created_at') or '')
        if not fresh:
            return ''
        lines = []
        for e in fresh[-cap:]:
            md = e.get('metadata') or {}
            # Error text lands inside the XML-structured lived prompt —
            # escape it so an API error echoing request content can't close
            # the block or forge timeline tags (same guard as message text).
            parts = ['stop %s FAILED: %s' % (
                md.get('stop_counter', '?'),
                _xml_escape(str(md.get('error') or
                                e.get('summary') or '?')))]
            n_partial = len(md.get('partial_actions') or [])
            if n_partial:
                parts.append('%d action(s) completed before death (their '
                             'writes are already in the graph — do not '
                             're-create them)' % n_partial)
            if md.get('payload_pointer'):
                parts.append('full conversation at failure: %s'
                             % md['payload_pointer'])
            lines.append('- ' + '; '.join(parts))
        return '\n'.join(lines) + '\n'
    except Exception as e:
        brain._log_error('s1e_failed_encodes_block', e,
                         'session=%s — omitting block' % (session_id or '')[:8])
        return ''


def _xml_escape(s):
    """Escape the three XML-significant chars so message/tool text can't malform
    the lived-sequence timeline or forge tags (e.g. a '</other>' or '<turn>'
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
                from .encode_contract import SURFACED_ID_PATTERN
                ref_ids = re.findall(SURFACED_ID_PATTERN, judge_output)
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


def _lived_turns(brain, session_id, n_turns):
    """The windowed turn list the lived timeline renders: episodes grouped into
    turns, trimmed to the last `n_turns`. Raises on read failure — the renderer
    keeps its markdown fallback.

    One owner for the window: the timeline renders these turns and
    window_first_turn dates the catalog against them, so the two can never
    disagree about where the conversation window opens.
    """
    from servers.scales.s1.encode_contract import LIVED_SEQUENCE_PULL
    from servers.trace_contract import SAID_AND_DID_REF_TYPES
    episodes = brain.recall_episodes(
        session_id=session_id, ref_type=list(SAID_AND_DID_REF_TYPES),
        sort_order='desc', limit=LIVED_SEQUENCE_PULL)['episodes']
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

    return turns[-n_turns:]  # window-match the control arm


def _window_n_turns(messages):
    """The window-size rule — the ONE derivation of how many turns the chat
    window spans, shared by the timeline render, the catalog cutoff
    (window_first_turn) and the associated-stub seeds so the three can never
    disagree about the window. Falls back to WINDOW_TURNS_FALLBACK when the
    hook window carries no user rows."""
    from servers.scales.s1.encode_contract import WINDOW_TURNS_FALLBACK
    return (sum(1 for m in (messages or []) if m.get('role') == 'user')
            or WINDOW_TURNS_FALLBACK)


def window_first_turn(brain, session_id, messages, turns=None):
    """The 1-based turn number the rendered timeline opens on — the
    window-aligned catalog cutoff (encode_contract.build_node_catalog's
    `window_first_turn`). None when the window has no datable head, which the
    caller reads as "no window cutoff available", falling back to round-based
    aging rather than ageing everything.

    `turns` (optional): the pre-pulled _lived_turns list, threaded from
    _build_catalog so one window pull serves the cutoff and the stub seeds
    (None → pull here; eval callers)."""
    from servers.scales.s1.trace_links import display_turn
    if turns is None:
        try:
            turns = _lived_turns(brain, session_id, _window_n_turns(messages))
        except Exception:
            return None
    for t in turns:
        if t.get('user'):
            return display_turn(t['user'].get('chain_id'))
    return None


def _render_lived_sequence_timeline(brain, session_id, messages, streams=None,
                                    view_policy=False, now=None):
    """The XML lived sequence — messages + tool actions interleaved (piece 1).

    `view_policy` (encoder_view, resolved once in run_encoding): ON, the
    render policies apply — already-encoded turns get a stubbed <actions>
    element; node-op/lookup tool lines provenance already shows are dropped
    from every <actions>; <provenance> renders the verbs split
    (created/revised/recalled/archived (me)) instead of the merged
    encoded(me). OFF: the pre-policy render (message text is full in both
    arms — the encoded-turn trim is retired).

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
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    lim = ENCODING_AGENT['message_display_limit']

    try:
        turns = _lived_turns(brain, session_id, _window_n_turns(messages))
    except Exception as e:
        print('[s1e] ERROR reading lived sequence, falling back to markdown: %s' % e, flush=True)
        return _render_markdown_timeline(brain, messages)

    def _text(ep):
        # The store-capped message body lives in metadata['content']; `summary`
        # is the short display excerpt. Preserve the body and any store-limit
        # marker when the timeline limit is None, then escape for XML.
        meta = ep.get('metadata')
        body = (meta.get('content') if isinstance(meta, dict) else None) or ep.get('summary') or ''
        cut = body[:lim] + ('…' if lim is not None and len(body) > lim else '')
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
        if view_policy:
            # the verb split also renders recalled/looked_up/archived refs —
            # fetch their titles too (authored covers created ∪ revised)
            ref_ids.update(lk.get('recalled') or ())
            ref_ids.update(lk.get('looked_up') or ())
            ref_ids.update(lk.get('archived') or ())
    if ref_ids:
        try:
            titles = {nid: (row.get('title') or '')
                      for nid, row in brain._nodes.get_bulk(list(ref_ids)).items()}
        except Exception:
            titles = {}

    # Covered turns keep their FULL message text — trimming an encoded turn
    # would defeat the purpose; what a covered turn drops is its <actions>,
    # stubbed below. The `encoded` attr states coverage on the turn itself.
    # No attr when the trace-link join is unavailable (degraded piece-1 path —
    # coverage unknown).
    if view_policy:
        from servers.scales.s1.encoder_view import actions_stub_line
        from servers.scales.s1.encoder_actions import condense_actions

    if view_policy:
        from servers.scales.s1.trace_links import display_turn

    out = ""
    for n, t in enumerate(turns, 1):
        uid = (t['user'] or {}).get('id') if t['user'] else None
        link = links.get(uid) if uid else None
        n_disp = n   # n stays the window ordinal — the frontier index below
        if view_policy:
            # the real turn number — session-global, 1-based (trace_links
            # owns the axis; run attribution 'encoded(me, turn 5)' shares it
            # with no shift). Orphan turns keep the ordinal.
            real = display_turn((t['user'] or {}).get('chain_id')) if t['user'] else None
            n_disp = real if real is not None else n
        enc_attr, is_enc = '', False
        if link is not None:
            is_enc = bool(link.get('encoded_by'))
            enc_attr = ' encoded="%s"' % ('true' if is_enc else 'false')
        age_attr = ''
        if view_policy and now is not None:
            # per-turn recency against conversation time — with the <timeline
            # now=…> stamp this gives full orientation (relative labels are
            # invertible). Caveat: replayed corpora stamp turns at replay
            # wall-clock, so ages there degrade to 'just now' — both A/B arms
            # share the artifact.
            from servers.pipeline_contract import _relative_time
            first = t['user'] or t['assistant'] or (t['actions'][0] if t['actions'] else None)
            age = _relative_time((first or {}).get('created_at'), now=now, fine=True)
            if age:
                age_attr = ' age="%s"' % age
        out += '<turn n="%d"%s%s>\n' % (n_disp, age_attr, enc_attr)
        # Tag vocabulary is identity-native, not role-native:
        # <me> = my side of the exchange; <other> = whoever is on the other side
        # this session (usually the operator, sometimes an agent — an identity
        # attr comes later). Presentation-layer only: the substrate keeps
        # user_message/assistant_message; the join is by trace id.
        if t['user']:
            out += '  <other trace="%s">%s</other>\n' % (
                t['user'].get('id', ''), _text(t['user']))
        # <provenance> sits right after <other>: chronologically, recall/surface
        # fires on the user prompt — before my reply exists.
        prov = _render_provenance(links, frontier, t, n - 1, titles,
                                  view_policy=view_policy)
        if prov:
            out += '  <provenance>%s</provenance>\n' % prov
        if t['assistant']:
            out += '  <me trace="%s">%s</me>\n' % (
                t['assistant'].get('id', ''), _text(t['assistant']))
        if t['actions']:
            if view_policy and is_enc:
                # Covered turn: the previous run already read this churn. The
                # element stays — a stub can't be misread as "nothing happened".
                out += '  <actions>%s</actions>\n' % actions_stub_line(
                    len(t['actions']))
            else:
                if view_policy:
                    # The condenser owns everything actions-shaped under the
                    # policy: drop/stub (existing action_mode semantics),
                    # exact dedup, and the per-turn budget with the rollup
                    # accounting line. The tail turn gets the larger budget —
                    # it is the encoder's actual working material.
                    lines = [_xml_escape(ln) for ln in condense_actions(
                        t['actions'], is_tail=(n == len(turns)))]
                else:
                    # Flag off: the legacy render — one verbatim summary per
                    # action (tool cues have no metadata['content']).
                    lines = [_xml_escape(a.get('summary') or '')
                             for a in t['actions']]
                if lines:
                    out += '  <actions>\n'
                    for ln in lines:
                        out += '    %s\n' % ln
                    out += '  </actions>\n'
        out += '</turn>\n\n'
    return out


def _short_refs(ids, titles=None, title_first=False):
    """Node ids as 8-char `id:` refs, each carrying its 1-line «tag» (title) when
    known — the locality affordance (fork #2): WHERE a node came up in the
    timeline, complementary to the catalog's full WHAT-it-is. The 8-char short
    still matches the catalog so a ref dereferences there; the «tag» saves the
    scan. Falls back to a bare `id:` when no title is mapped (stub brains, or an
    id the title fetch missed).

    `title_first` (view policy): `"title" id:x` — title-leading AND
    double-quoted like every other render in the system (catalog headers,
    edges); the id-first «tag» form is the legacy arm's."""
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
        if not t:
            out.append('id:%s' % short)
        elif title_first:
            out.append('"%s" id:%s' % (t, short))
        else:
            out.append('id:%s «%s»' % (short, t))
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
        # degrade (brain.errors is the surface, a bare print is not).
        # Provenance is advisory; never block the timeline.
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


def _render_provenance(links, frontier, turn, idx, titles=None,
                       view_policy=False):
    """One <provenance> line for a turn — REAL refs only, no coverage markers.

    Coverage lives on the turn itself (the `encoded="true|false"` attribute the
    timeline render stamps), so provenance never says "covered, nothing to show"
    — the old `✓` marker is gone. What renders: `surfaced:` refs (per-turn, 1:1),
    the owning run's full id-list ONCE at the run's frontier turn (an edge-only
    run has an empty id set → nothing), and the session's own node ops (joined by
    stop; empty in replayed eval corpora) — legacy: the merged
    `encoded(me):` (link['authored'] = created ∪ revised); view policy ON:
    the verb split (encoder_view.PROVENANCE_SPLIT — created / revised /
    recalled / archived), which is what lets the node-op action lines drop
    without losing the information. `titles` maps id→title for the «tag»
    locality. Returns '' when there's nothing real to show.
    """
    uid = (turn['user'] or {}).get('id') if turn['user'] else None
    link = links.get(uid) if uid else None
    if not link:
        return ''
    parts = []
    tf = bool(view_policy)   # title-first refs under the policy (system-wide shape)
    if link['surfaced']:
        parts.append('surfaced: %s' % _short_refs(link['surfaced'], titles, tf))
    eb = link['encoded_by']
    if eb and frontier.get(eb) == idx and link['encoded']:
        # policy: turn-coordinate attribution ('encoded(me, turn 36)'); the
        # legacy arm keeps its 'encoded(S1S)' string byte-for-byte
        run_label = 'encoded(S1S)'
        if view_policy:
            from servers.scales.s1.encoder_view import encoded_run_label
            run_label = encoded_run_label(link.get('encoded_by_stop'))
        parts.append('%s: %s' % (run_label, _short_refs(link['encoded'], titles, tf)))
    if view_policy:
        from servers.scales.s1.encoder_view import PROVENANCE_SPLIT
        for label, keys in PROVENANCE_SPLIT:
            ids = [i for k in keys for i in (link.get(k) or ())]
            if ids:
                # cross-key dedup (an id can be both read and searched-up)
                ids = list(dict.fromkeys(ids))
                parts.append('%s: %s' % (label, _short_refs(ids, titles, tf)))
    elif link.get('authored'):
        parts.append('encoded(me): %s' % _short_refs(link['authored'], titles))
    return ' | '.join(parts)


def _write_pre_traces(brain, dispatch_fn, messages, user_content, counter,
                      session_id, catalog_ids=(), prompt_pointer=None):
    """Write S1 encode traces: O (encoding prompt) and K (node catalog).

    `prompt_pointer` — the db_dir-relative payload pointer for this run's
    recorded prompt (brain.record_payload). It becomes the O trace's ref_id;
    '' when recording was gated off or failed (readers render "no payload").
    Pre-migration rows carry an absolute /tmp path here instead — readers
    branch on pointer shape (time-bounded legacy).

    `catalog_ids` — the ids build_node_catalog actually rendered for this
    run. The K trace's ref_id carries them (8-char, comma-separated); the S2
    consolidation blindness check (_load_catalog_data) reads that list to
    decide whether node B was created blind to A. It used to be parsed from
    messages' `recalled_raw`, which no producer populates — every K trace
    said "0 unique nodes" and the blindness check saw an empty catalog.
    [associated] stub ids are IN this set by design — they render in the
    catalog, so the encoder was not blind to them and the K trace must say so.
    """
    try:
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id=session_id, stop_counter=counter)
        enc_chain = ctx.s1e_chain()
        turn_count = len(messages) if messages else 0
        node_ids = {(cid or '')[:8] for cid in (catalog_ids or ()) if cid}

        dispatch_fn('trace_append', {
            'chain_id': enc_chain, 'scale': 's1', 'event_type': 'O',
            'ref_type': 'encoding_prompt',
            'ref_id': prompt_pointer or '',
            'summary': '%d turns, %d chars context, interaction: encoding-agent-v3' % (
                turn_count, len(user_content)),
            'session_id': session_id})
        dispatch_fn('trace_append', {
            'chain_id': enc_chain, 'scale': 's1', 'event_type': 'K',
            'ref_type': 'node_catalog',
            'ref_id': ','.join(sorted(node_ids)),
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

    CONTROL-ARM ONLY (flag off): the lived arm's prompt emits a `## Arc` fence
    instead of a SESSION_CONTEXT: line, written via brain.write_session_arc —
    the journal mechanism's arc component. This legacy parse retires with the
    flag at v26 activation.

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
    from servers.scales.s1.encode_contract import ENCODING_TOOLS
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["inputSchema"]}
            for t in brain_mcp.TOOLS if t["name"] in ENCODING_TOOLS]
