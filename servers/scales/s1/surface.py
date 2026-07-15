"""S1 Surface Chain — surface relevant memories, graph expansion, correction enrichment, trace writing.

Scale: S1 (Turn integration)
Chain: s1r (surface)
Interaction: 'surface' in interactions table (learnable boundary)

Triggered by: hook_recall (UserPromptSubmit) in daemon_hooks.py
Reads: recall candidates from brain.recall(), interactions table
Writes: S1 traces (O/K/Δ), tmp files for Hebbian + dashboard
"""

import json
import os
import time

from servers.scales.dispatch import load_env
from servers.scales.runner import read_usage, sum_usage
from servers.scales.s1 import surface_capture
from servers.trace_contract import (
    build_selection_metadata, build_run_telemetry, check_surface_telemetry)
from servers.daemon_config import brain_tmp_dir


# SURFACE_SELECTION_SCHEMA lives in surface_contract.py alongside the other
# surface I/O contracts (the render formats for each mode). Imported below.
def _get_recently_surfaced(brain, session_id):
    """Get recently surfaced node IDs from S1 traces (for dedup).

    Scoped to this session — parallel sessions must not see each other's
    surfaced nodes in their exclusion lists.
    """
    from servers.scales.s1.surface_contract import SURFACE
    lookback = SURFACE.get('recent_recalls_messages', 10)
    recent_k = brain.query_traces(
        ref_type='surface_selected', scale='s1', hours=None, limit=lookback,
        session_id=session_id)['events']
    seen_ids = set()
    for evt in recent_k:
        raw = evt.get('ref_id', '[]')
        try:
            for nid in json.loads(raw):
                seen_ids.add(nid)
        except (ValueError, TypeError) as _e:
            # Malformed ref_id silently dropped node IDs from the
            # surfaced-recently dedup set — Haiku could re-surface nodes
            # we already showed. Log so a corrupt producer surfaces.
            try:
                brain._log_error(
                    'surface_seen_ids_parse', _e,
                    'malformed ref_id in surface_selected trace; '
                    'sample=%r' % str(raw)[:120])
            except Exception:
                pass
    dal = brain._nodes
    recently_surfaced = []
    for nid in list(seen_ids)[:20]:
        title = dal.get_title(nid)
        if title:
            recently_surfaced.append({"id": nid, "title": title})
    return recently_surfaced


def _call_surface(brain, candidates_data, user_message,
                  recent_messages, session_id, result, frame=''):
    """Call Haiku to surface relevant nodes from candidates.

    Returns: (surfaced_dict, surface_prompt, max_tokens, interaction_id, telemetry)
        surfaced_dict has 'selected' list. Empty on failure.
        telemetry is the shared run-cost dict (build_run_telemetry kwargs:
        token counts + elapsed_ms + rounds + truncated) for the K trace.

    Surface variant gating (2026-05-10):
      BRAIN_SURFACE_VARIANT=v4 (default) — current path, single Haiku call,
        no tools. Production default. Byte-identical to pre-2026-05-10.
      BRAIN_SURFACE_VARIANT=v5_agentic — agentic loop with 6 fetch tools.
        Eval-only until explicitly opted in. Reads surface prompt v5 from
        the active version pointer (must be activated separately).
    """
    from servers.scales.s1.surface_contract import (
        build_surface_prompt, prepare_presented_candidates,
        presentation_shuffle_seed, SURFACE_MODEL)

    # Recently surfaced (for dedup)
    recently_surfaced = []
    try:
        recently_surfaced = _get_recently_surfaced(brain, session_id)
    except Exception as e:
        brain._log_error('surface_recently_recalled', e, 'fetching recently surfaced titles')

    # Retrieval stats from recall result. (Recall no longer returns an `intent`
    # field — the regex classifier and the field were both removed.)
    retrieval_stats = result.get('_retrieval_stats') if isinstance(result, dict) else None

    # interaction_seed.py guarantees 'surface' is registered on every boot.
    surface_interaction = brain.get_interaction('surface')
    if not surface_interaction or not surface_interaction.get('template'):
        raise RuntimeError(
            "S1 Surface: no 'surface' interaction registered in "
            "brain_logs.db. interaction_seed should have populated "
            "this on Brain construction — check seed/DAL state.")
    surface_instructions = surface_interaction['template']
    interaction_id = surface_interaction.get('id')

    # Layout rides in the interaction CONFIG ({"layout": "xml_v13"}), so a
    # version flip changes template and renderer atomically — a v13 template
    # can never run against the legacy user content or vice versa.
    layout = 'legacy'
    try:
        layout = (brain.get_interaction_config('surface') or {}).get(
            'layout', 'legacy')
    except Exception as _cfg_err:
        brain._log_error('surface_layout_config', _cfg_err,
                         'reading layout from surface interaction config')

    # Presentation shuffle (2026-07-14, RECALL-SR-REDESIGN.md §20.12 A2):
    # the menu Haiku sees is a deterministic per-turn shuffle of the
    # scorer's ranking — position bias dies at the source, and picked/
    # dropped trace rows become uniform-propensity P3 training data.
    # candidates_data itself is NEVER reordered: traces (cand_detail),
    # short_to_full, and the admission floor all keep scorer order.
    # presented_order (the exact round-1 menu, pre-tool-fetches) is stashed
    # per-session for the K trace — same pattern as _surface_tool_traces.
    shuffle_seed = presentation_shuffle_seed(session_id, user_message)
    presented_order = [
        str(c.get('id', ''))[:8]
        for c in prepare_presented_candidates(candidates_data, shuffle_seed)]
    if not hasattr(brain, '_surface_presented'):
        brain._surface_presented = {}
    brain._surface_presented[session_id] = {
        'shuffle_seed': shuffle_seed, 'presented_order': presented_order}

    user_content, max_tokens = build_surface_prompt(
        candidates_data, user_message,
        recent_messages=recent_messages,
        recently_recalled=recently_surfaced,
        retrieval_stats=retrieval_stats,
        frame=frame,
        layout=layout,
        shuffle_seed=shuffle_seed)

    surface_prompt = (surface_instructions + "\n\n---\n\n" + user_content) \
        if surface_instructions else user_content

    # Variant gate — env var picks the path. v4 is the production default.
    variant = os.environ.get('BRAIN_SURFACE_VARIANT', 'v4').strip().lower()

    # Production capture at the Haiku boundary (replay-bench corpus). Must
    # begin BEFORE the agentic loop: it deep-copies candidates_data, which
    # the loop mutates in place with tool-fetched entries — replay needs
    # the round-1 pool. Returns None when disabled; every consumer of
    # `capture` below is None-safe.
    capture = surface_capture.begin(
        brain, candidates_data=candidates_data, user_message=user_message,
        recent_messages=recent_messages, recently_surfaced=recently_surfaced,
        retrieval_stats=retrieval_stats, frame=frame, layout=layout,
        shuffle_seed=shuffle_seed,
        surface_instructions=surface_instructions,
        interaction_version=surface_interaction.get('version'),
        interaction_id=interaction_id, user_content=user_content,
        max_tokens=max_tokens, variant=variant, model=SURFACE_MODEL,
        session_id=session_id)

    # Shared client, built-and-cached once on the brain. No per-call throwaway
    # fallback: Brain._ensure_anthropic_client is the single construction site
    # and self-heals if a boot-warmup failure left the client unset.
    client = brain._ensure_anthropic_client()

    if variant == 'v5_agentic':
        # Agentic path: Haiku has tools, can extend the candidate pool
        # before final selection. Tool-fetched candidates are appended to
        # `candidates_data` in place so the downstream short_to_full
        # mapping resolves them.
        raw, tool_trace, telemetry = _call_surface_agentic(
            client, brain, candidates_data, surface_instructions,
            user_content, max_tokens, session_id, SURFACE_MODEL,
            layout=layout, capture=capture)
        # Attach tool trace to brain for the caller to write into K trace.
        # Stashed on the brain instance per-session-id so parallel sessions
        # don't clobber each other.
        try:
            if not hasattr(brain, '_surface_tool_traces'):
                brain._surface_tool_traces = {}
            brain._surface_tool_traces[session_id] = tool_trace
        except Exception:
            pass
    else:
        _t0 = time.time()
        api_resp = client.messages.create(
            model=SURFACE_MODEL,
            max_tokens=max_tokens,
            system=surface_instructions,
            messages=[{"role": "user", "content": user_content}])
        raw = api_resp.content[0].text.strip()
        # Cost telemetry — single Haiku call, built through the shared builder
        # (the one construction point). read_usage maps .usage onto the token
        # field-set; one round; truncated if Haiku hit max_tokens mid-selection.
        telemetry = build_run_telemetry(
            **read_usage(api_resp),
            elapsed_ms=int((time.time() - _t0) * 1000),
            rounds=1,
            truncated=1 if getattr(api_resp, 'stop_reason', None) == 'max_tokens' else 0)

    # Parse JSON — robust to the three shapes Haiku sometimes returns:
    #   (a) bare JSON: {"selected": [...]}
    #   (b) fenced: ```json\n{...}\n```
    #   (c) JSON + trailing prose: {...}\n\nHere's why I picked...
    # `raw_decode` consumes the first valid JSON object and reports the
    # tail, which we discard — no "Extra data" crash on (c).
    # Stash the capture (raw attached) for run_surface to finish() with the
    # resolved selection — same per-session-id stash pattern as
    # _surface_tool_traces, so parallel sessions can't clobber each other.
    if capture is not None:
        capture['output'] = {'raw': raw}
        if not hasattr(brain, '_surface_captures'):
            brain._surface_captures = {}
        brain._surface_captures[session_id] = capture

    surfaced = _parse_surfacer_json(raw)
    if surfaced is None and raw:
        # We had a non-empty Haiku response but couldn't parse anything
        # dict-shaped from it. Surface this — it's the silent-failure mode
        # that produced empty additionalContext at N=15 (fishing query).
        brain._log_error(
            'surface_haiku_unparseable',
            ValueError('Haiku response did not yield a parseable JSON dict'),
            'first 300 chars: %r' % raw[:300])
        surfaced = {"selected": []}
    elif surfaced is None:
        surfaced = {"selected": []}

    return surfaced, surface_prompt, max_tokens, interaction_id, telemetry


def _call_surface_agentic(client, brain, candidates_data, surface_instructions,
                           user_content, max_tokens, session_id, model,
                           max_rounds=2, layout='legacy', capture=None):
    """Agentic surface call: Haiku may use fetch tools to extend the candidate
    pool before final JSON selection.

    The final round is sent with tool_choice='none', so Haiku must finalize
    with the selection JSON — max_rounds is the hard cap on API calls.

    Returns: (raw_final_text, tool_trace, telemetry) where tool_trace is a list
    of per-round dicts {round, stop_reason, total_ms, <USAGE_FIELDS>,
    tool_calls: [...]} for trace observability (total_ms + usage per API call,
    mirroring run_llm_loop's per_round_stats), and telemetry is the shared
    run-cost dict (build_run_telemetry kwargs) summed across the loop's
    Haiku rounds.

    Mutates `candidates_data` IN PLACE — tool-fetched candidates are appended
    so the downstream short_to_full ID mapping (in run_surface) can resolve them.
    """
    from servers.scales.s1.fetch_tools import (
        TOOL_DEFINITIONS, execute_tool, format_tool_result_for_haiku,
    )
    from servers.scales.s1.surface_contract import (
        SURFACE_SELECTION_SCHEMA, CACHE_MIN_PREFIX_TOKENS)

    # Track existing IDs to dedupe tool-fetched candidates against cosine pool
    existing_ids = {c.get('id') for c in candidates_data if isinstance(c, dict)}

    # Admission-floor reference: median score of the ORIGINAL cosine pool,
    # snapshotted before any tool results join (recall_topical results are
    # score-comparable and must beat this to be admitted — see the floor
    # at the execute_tool site below).
    import statistics as _stats
    _pool_scores = [c.get('score') or 0 for c in candidates_data
                    if isinstance(c, dict) and (c.get('score') or 0) > 0]
    _pool_median = _stats.median(_pool_scores) if _pool_scores else 0.0

    # Cache breakpoints (runner convention: BP1 system 1h, BP2 user 5m):
    #   BP1 (in api_kwargs below) — last system block. Caches tools+system,
    #   which is byte-identical ACROSS recalls (cross-recall reuse) and
    #   survives the final round's tool_choice flip — tool_choice changes
    #   invalidate only the messages-tier cache (Anthropic invalidation
    #   hierarchy), never the tools/system tiers.
    #   BP2 (here) — the round-1 user content (~20K tokens incl. the
    #   candidate pool): read back by round 2 at 0.1× price and much
    #   faster prefill. Requires tools to be byte-identical across rounds.
    # Prompts under the model's 4096-token cacheable minimum are silently
    # not cached (no error).
    messages = [{"role": "user", "content": [
        {"type": "text", "text": user_content,
         "cache_control": {"type": "ephemeral"}}]}]
    tool_trace = []
    raw_final = ''

    # Cost telemetry — summed across rounds (the agentic loop calls Haiku up to
    # max_rounds times) via the shared sum_usage, then built through the shared
    # builder (one construction point, same as v4 + the encoders).
    _t0 = time.time()
    usage_total = read_usage(None)   # all-zero baseline of the right keys
    rounds_used = 0
    truncated = 0

    def _telemetry():
        return build_run_telemetry(
            **usage_total,
            elapsed_ms=int((time.time() - _t0) * 1000),
            rounds=rounds_used, truncated=truncated)

    def _absorb_response(resp):
        """Fold one Haiku response into the running cost totals and capture
        any text content as the candidate final answer. The ONE place a
        response's usage/truncation/text is absorbed, so telemetry can't
        drift. Returns the response's usage dict for per-round checks."""
        nonlocal rounds_used, truncated, raw_final
        rounds_used += 1
        ru = read_usage(resp)
        sum_usage(usage_total, ru)
        if getattr(resp, 'stop_reason', None) == 'max_tokens':
            truncated += 1
        for block in resp.content:
            if getattr(block, 'type', None) == 'text':
                raw_final = block.text.strip()
        return ru

    # Anthropic Structured Outputs runs on EVERY round, alongside tools.
    # When Haiku tool-uses, the schema doesn't apply to tool_use blocks;
    # when Haiku finalizes with a text response, the schema enforces
    # SURFACE_SELECTION_SCHEMA. Previous design only applied output_config
    # on the final round — that left round 1 unprotected, so when Haiku
    # skipped tools entirely and went straight to chat-style narration on
    # round 1 ("I need to understand what topic this message is asking..."),
    # the loop exited at `stop_reason != 'tool_use'` with the unparseable
    # prose. Verified failure: surface_haiku_unparseable at 16:21:23 UTC.
    output_config = {
        'format': {
            'type': 'json_schema',
            'schema': SURFACE_SELECTION_SCHEMA,
        },
    }

    for round_idx in range(max_rounds):
        is_final = (round_idx == max_rounds - 1)

        # Tools ride on EVERY round byte-identical (2026-07-02) so the BP1
        # prefix never shifts. The FINAL round adds tool_choice='none'
        # (2026-07-11): Haiku must answer with the selection JSON, so
        # max_rounds IS the hard API-call cap — no forced-finalize third
        # call (that path cost an extra ~5.7s and breached the 20s hook
        # budget on 2-tool-round recalls).
        api_kwargs = {
            'model': model,
            'max_tokens': max_tokens,
            'system': [{"type": "text", "text": surface_instructions,
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            'messages': messages,
            'output_config': output_config,
            'tools': TOOL_DEFINITIONS,
        }
        if is_final:
            api_kwargs['tool_choice'] = {'type': 'none'}

        _t_call = time.time()
        try:
            api_resp = client.messages.create(**api_kwargs)
        except Exception as e:
            brain._log_error('surface_agentic_api', e,
                              'agentic Haiku call round=%d' % round_idx)
            return raw_final, tool_trace, _telemetry()
        _call_ms = int((time.time() - _t_call) * 1000)

        # Accumulate cost + capture any text (the candidate final answer).
        round_usage = _absorb_response(api_resp)
        # Cache visibility (operator ask): round 2+ shares round 1's prefix
        # byte-for-byte, so zero cache reads on a cacheable-sized prompt
        # means the cache broke (tools drift, block-shape change). Gate on
        # prompt size so sub-minimum prompts (tests, tiny brains) don't warn.
        if round_idx > 0 and (round_usage.get('cache_read_tokens') or 0) == 0 \
                and (round_usage.get('input_tokens') or 0) \
                    + (round_usage.get('cache_creation_tokens') or 0) \
                    > CACHE_MIN_PREFIX_TOKENS:
            brain._log_warning(
                'surface_cache_miss',
                'agentic round %d read 0 cache tokens (input=%d created=%d) '
                '— round-1 prefix should have hit' % (
                    round_idx, round_usage.get('input_tokens') or 0,
                    round_usage.get('cache_creation_tokens') or 0),
                'session=%s' % session_id)

        stop_reason = api_resp.stop_reason
        # Per-API-call cost, same field names as run_llm_loop's
        # per_round_stats (total_ms + USAGE_FIELDS; no ttft_ms — this path
        # is non-streaming). Distinguishes a slow/retried call (high
        # total_ms, normal output) from a verbose one (total_ms tracks
        # output_tokens at ~20ms/token).
        round_record = {'round': round_idx, 'stop_reason': stop_reason,
                         'total_ms': _call_ms, **round_usage,
                         'tool_calls': []}

        if stop_reason != 'tool_use':
            tool_trace.append(round_record)
            break

        if is_final:
            # Unreachable via the API: this round was sent with
            # tool_choice='none', so stop_reason can't be tool_use. Reaching
            # it means the constraint stopped constraining — log loudly,
            # keep whatever text was absorbed, and spend no extra call.
            brain._log_warning(
                'surface_final_round_tool_use',
                "stop_reason=tool_use despite tool_choice='none' on the "
                'final round — constraint not honored',
                'round=%d session=%s' % (round_idx, session_id))
            tool_trace.append(round_record)
            break

        # Append Haiku's full assistant message (with tool_use blocks) to history.
        assistant_blocks = []
        tool_results = []
        for block in api_resp.content:
            btype = getattr(block, 'type', None)
            if btype == 'text':
                assistant_blocks.append({"type": "text", "text": block.text})
            elif btype == 'tool_use':
                # Convert to dict for the message history
                tool_use_id = block.id
                tool_name = block.name
                tool_input = block.input or {}
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": tool_name,
                    "input": tool_input,
                })
                # Execute the tool
                exec_result = execute_tool(brain, tool_name, tool_input,
                                            session_id=session_id)
                # Admission floor (2026-06-12): recall_topical scores come
                # from the same recall pipeline as the cosine pool, so they
                # are directly comparable — a fetched node scoring below the
                # original pool's median doesn't beat what's already here.
                # Score field parity is pinned by surface_contract.recall_score
                # (both the pool and the tool read it). Filter exec_result IN
                # PLACE so the rendered tool output and the candidate pool
                # agree (Haiku must never see an id it can't select). Other
                # tools keep synthetic scores — no floor.
                _raw_results = [c for c in (exec_result.get('results') or [])
                                if isinstance(c, dict)]
                _dropped_below_floor = 0
                _dropped_ids = []
                if tool_name == 'recall_topical' and _pool_median > 0:
                    _kept = []
                    _dropped = []
                    for c in _raw_results:
                        if (c.get('score') or 0) >= _pool_median:
                            _kept.append(c)
                        else:
                            _dropped.append(c)
                    _dropped_below_floor = len(_dropped)
                    _dropped_ids = [str(c.get('id') or '')[:8] for c in _dropped]
                    exec_result['results'] = _kept
                    if _raw_results and not _kept:
                        # Tripwire: the tool fetched candidates and the floor
                        # dropped every one. Occasional all-drops are fine;
                        # EVERY call all-dropping means the score contract
                        # forked again (the 3-week silent death of 2026-07).
                        _top_fetched = max((c.get('score') or 0)
                                           for c in _raw_results)
                        brain._log_warning(
                            'surface_floor_dropped_all',
                            'admission floor dropped ALL %d recall_topical '
                            'results (pool_median=%.3f top_fetched=%.3f) — '
                            'score-contract drift if this repeats'
                            % (len(_raw_results), _pool_median, _top_fetched),
                            'args=%r' % str(tool_input)[:200])
                # Append fetched results to candidates_data (dedupe)
                for cand in exec_result.get('results') or []:
                    cid = cand.get('id') if isinstance(cand, dict) else None
                    if cid and cid not in existing_ids:
                        candidates_data.append(cand)
                        existing_ids.add(cid)
                # Record for trace. result_ids/dropped_ids make per-tool
                # selection attribution computable downstream (LAF training
                # reads these traces) — counts alone can't say which tool
                # sourced a picked node.
                round_record['tool_calls'].append({
                    'tool': tool_name,
                    'args': tool_input,
                    'result_count': len(exec_result.get('results') or []),
                    'result_ids': [str(c.get('id') or '')[:8]
                                   for c in (exec_result.get('results') or [])
                                   if isinstance(c, dict)],
                    'dropped_below_floor': _dropped_below_floor,
                    'dropped_ids': _dropped_ids,
                    'latency_ms': exec_result.get('latency_ms', 0),
                    'error': exec_result.get('error'),
                })
                # Compose tool_result message block. Pass `brain` so tool
                # results are rendered with the SAME formatter that produced
                # the initial 25 cosine candidates (content + situation +
                # edges + _corrections) — no data-richness asymmetry.
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": format_tool_result_for_haiku(exec_result, layout=layout),
                })
        if not tool_results:
            # stop_reason said tool_use but no tool_use block arrived (the
            # May-2026 empty-tool_use Haiku mode). Appending an assistant
            # message with no tool_use + an empty tool_results user message
            # would 400 the next round. Leave `messages` untouched and go
            # around again — the identical request re-reads the cached
            # prefix, and the final round's tool_choice='none' guarantees
            # a JSON finalize.
            brain._log_warning(
                'surface_empty_tool_use',
                'stop_reason=tool_use with no tool_use blocks — retrying '
                'without history append',
                'round=%d session=%s' % (round_idx, session_id))
            tool_trace.append(round_record)
            continue
        messages.append({"role": "assistant", "content": assistant_blocks})
        messages.append({"role": "user", "content": tool_results})
        tool_trace.append(round_record)

    # Replay-bench capture: the literal loop history (round-1 user content,
    # assistant tool_use blocks, full rendered tool results) — the round-2
    # story that can't be eyeballed from a prompt diff. Normal exits only;
    # an in-loop API-error return leaves rounds empty (visible as such).
    surface_capture.record_rounds(capture, messages=messages,
                                  raw_final=raw_final, tool_trace=tool_trace)
    return raw_final, tool_trace, _telemetry()


def _parse_surfacer_json(raw):
    """Extract the {"selected": [...]} object from Haiku's response.

    Returns the parsed dict, or None if no valid JSON object is found.
    Strips ```-fenced blocks first, then uses raw_decode so trailing prose
    after a valid JSON object doesn't trigger "Extra data" errors.
    """
    if not raw:
        return None

    text = raw.strip()

    # Strip ```…``` fences if present (any language tag)
    if text.startswith("```"):
        # After the first newline, up to the last ``` fence
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()

    decoder = json.JSONDecoder()

    # First: try decoding from the first '{' we see. raw_decode parses
    # the first valid object and returns how far it got — any trailing
    # text is ignored. This is the common case when Haiku adds prose.
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _end = decoder.raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # Fallback: bracket-slice (handles malformed JSON inside but a clean
    # outer pair of braces). Preserves the prior behavior for edge cases
    # the raw_decode path doesn't cover.
    end = text.rfind("}") + 1
    if end > start:
        try:
            obj = json.loads(text[start:end])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _sanitize_selected_id(raw):
    """Strip whitespace Haiku occasionally injects inside emitted ids.

    Observed (eval run v12_1_full, item d7c942c3-r1): picks emitted as
    '9 9a 2e ' / 'd 6d3 f8' — real candidate ids with spaces inserted.
    Must run BEFORE any [:8] truncation, or the cut keeps the spaces and
    drops real hex chars.
    """
    return ''.join(str(raw or '').split())


def _unique_prefix_match(short_id, candidate_short_ids):
    """Recover a corrupted pick by unique-prefix match against the menu.

    Returns the single candidate short id that starts with `short_id`, or
    None when zero or several match. Requires >= 4 chars so a near-empty
    fragment can't land on a candidate by coincidence. Candidates only —
    never the whole brain — so recovery can't resurrect an archived node
    the menu already excluded.
    """
    if len(short_id) < 4:
        return None
    hits = [cid for cid in candidate_short_ids if cid.startswith(short_id)]
    return hits[0] if len(hits) == 1 else None


# Module-level flag so variant verification logs once per process, not per
# call. Reset on import (when the harness reloads modules between runs).
_VARIANT_FIRST_CALL_LOGGED = False


def _graph_expand(brain, selected_ids, query_vec=None, prior_vecs=None):
    """Expand the graph from selected seeds via spreading activation.

    Variant selection via BRAIN_RECALL_VARIANT env var:
      - 'baseline' (default): spread_activation — depth-emergent breadth-first
        traversal with median-threshold gate.
      - 'cluster': spread_activation_cluster — cluster-completion variant
        with distribution-derived per-hop gate, family-aware lineage
        ride-along, and convergence tagging.

    Args:
        brain: Brain instance (for interaction_config + conn)
        selected_ids: list of seed node IDs (typically Haiku's ≤5 picks)
        query_vec: numpy array (768d) — query embedding. Required for
            meaningful activation; without it, returns empty.
        prior_vecs: prior-turn query embeddings for multi-turn blend.

    Returns dict:
        'node_activation':  {full_node_id: float in [0,1]}
        'field_activation': {full_node_id: {field_name: float}}
        'rich_nodes':       {full_node_id: rich_node_dict} — from get_node batch
        'convergence':      {full_node_id: int} — count of distinct sources
                            that reached this target (cluster variants only).
                            Render layer can use to prioritize cluster boundaries.
        'trace':            per-hop diagnostics
    """
    import os
    from servers.scales.s1.surface_contract import (
        spread_activation, spread_activation_cluster)

    if query_vec is None or not selected_ids:
        return {'node_activation': {}, 'field_activation': {},
                'rich_nodes': {}, 'convergence': {}, 'trace': []}

    raw_variant = os.environ.get('BRAIN_RECALL_VARIANT', 'baseline').lower()

    # Variant names are composable tags. Knobs:
    #   spread:    'baseline' (default) | 'cluster' (cluster-completion v1)
    #   limit:     'lim15' / 'lim10' / 'lim20' set per-source neighbor cap
    #   thickness: edge weight as transmission multiplier (read in spread)
    #   lineage:   lineage families bypass median gate (read in spread)
    use_cluster = 'cluster' in raw_variant
    if 'lim10' in raw_variant:
        os.environ['BRAIN_SPREAD_NEIGHBOR_LIMIT'] = '10'
    elif 'lim15' in raw_variant:
        os.environ['BRAIN_SPREAD_NEIGHBOR_LIMIT'] = '15'
    elif 'lim20' in raw_variant:
        os.environ['BRAIN_SPREAD_NEIGHBOR_LIMIT'] = '20'
    elif 'lim25' in raw_variant:
        os.environ['BRAIN_SPREAD_NEIGHBOR_LIMIT'] = '25'
    # else: unset, _build_edge_coeffs falls back to '50'

    # Verification log — emit once per process so eval runs have proof the
    # variant flag actually took effect at the surface layer (not just at
    # harness invocation). Without this, the only evidence is timing
    # differences which can be confounded.
    global _VARIANT_FIRST_CALL_LOGGED
    if not _VARIANT_FIRST_CALL_LOGGED:
        import sys as _sys
        limit_value = os.environ.get('BRAIN_SPREAD_NEIGHBOR_LIMIT', '50')
        _sys.stderr.write(
            "[surface-variant first-call pid=%d] raw=%r cluster=%s "
            "limit=%s thickness=%s lineage=%s\n" % (
                os.getpid(), raw_variant, use_cluster,
                limit_value, 'thickness' in raw_variant,
                'lineage' in raw_variant))
        _VARIANT_FIRST_CALL_LOGGED = True

    # Resolve to full IDs (the kernel reads vectors keyed on full id)
    ndal = brain._nodes
    resolved = []
    for sid in selected_ids:
        full = ndal.resolve_id(sid) if len(str(sid)) < 16 else sid
        if full:
            resolved.append(full)

    if not resolved:
        return {'node_activation': {}, 'field_activation': {},
                'rich_nodes': {}, 'convergence': {}, 'trace': []}

    if use_cluster:
        result = spread_activation_cluster(resolved, query_vec, brain,
                                           prior_vecs=prior_vecs)
        convergence = result.get('convergence', {})
    else:
        result = spread_activation(resolved, query_vec, brain,
                                   prior_vecs=prior_vecs)
        convergence = {}

    node_activation = result['node_activation']
    field_activation = result['field_activation']

    # Batch-load rich node data for everything activated
    all_ids = list(node_activation.keys())
    rich_nodes = brain.get_node(all_ids) if all_ids else {}

    return {
        'node_activation':  node_activation,
        'field_activation': field_activation,
        'rich_nodes':       rich_nodes,
        'convergence':      convergence,
        'trace':            result['trace'],
    }


def _write_traces(brain, ctx, candidates_data, selected_ids, selected,
                  graph_neighbors, additional_context, enriched, results,
                  recall_ref, interaction_id, session_id, expansion=None,
                  frame='', telemetry=None, pt=None, selection_reason=''):
    """Write S1 surface traces: O (candidates), K (surfaced), Δ (additionalContext).

    `expansion` carries activation data from spread_activation when present —
    we attach per-node activation values and the kernel's per-hop trace to
    the K-event metadata so dashboards / S3 can see which nodes lit up and
    by how much, not just which were surfaced.

    `telemetry` is the run-cost dict from _call_surface (build_run_telemetry
    kwargs). It's emitted FLAT into the K-event metadata via build_run_telemetry
    — the same shared cost block the encoder delta carries — so Surface's
    input/output tokens + cache + elapsed_ms are queryable from traces, closing
    the long-standing surface cost-telemetry gap. Guarded by
    check_surface_telemetry so it can't silently regress to zeros.

    `pt` (optional PhaseTimer): when supplied, its per-phase breakdown is
    snapshotted into the K trace as `phase_timing` — the structured, queryable
    form of the hook_phase_timing debug string. The K trace also records an
    `outcome` flag (served / empty) so a turn that surfaced nothing is
    distinguishable from one that did, without cross-referencing other logs.

    `selection_reason` (Haiku's recall-level rationale, in practice the
    why-nothing-was-picked note) is the S1Surface journal on the K event —
    the trace is its only durable home; it is never rendered to Anchor.
    """
    recall_chain = ctx.s1r_chain()

    # O: candidates detail
    cand_detail = ['%s|%s|%.2f|%s' % (
        c.get('id', '')[:8], c.get('title', '')[:80],
        c.get('score', 0), c.get('type', ''))
        for c in candidates_data[:25]]

    # K: surfaced detail
    sel_detail = ['%s|%s' % (c.get('id', '')[:8], c.get('title', ''))
                  for c in candidates_data if c.get('id', '')[:8] in selected_ids]

    # Expanded detail
    exp_detail = ['%s|%s|%s' % (
        nb.get('id', '')[:8], nb.get('title', '')[:60], nb.get('relation', ''))
        for nb in graph_neighbors[:10]]

    # Selection delta metadata — unified shape for decode-style units.
    candidate_ids = [c.get('id', '')[:8] for c in candidates_data]
    selected_short = sorted(selected_ids)
    dropped_short = [cid for cid in candidate_ids if cid not in selected_ids]
    outcomes_per_candidate = {
        cid: ('selected' if cid in selected_ids else 'dropped')
        for cid in candidate_ids
    }

    # Frame metadata — how much partnership context was injected this turn.
    # Tracking the size + section count gives the dashboard observability into
    # what Anchor's prior actually looked like, without bloating traces with
    # the full Frame text. Empty dict when frame was unavailable (degraded).
    frame_meta = {}
    if frame:
        frame_meta = {
            'frame_chars': len(frame),
            'frame_tokens_est': len(frame) // 4,
            'frame_sections': frame.count('\n## ') + (1 if frame.startswith('## ') else 0),
        }
    else:
        frame_meta = {'frame_chars': 0, 'frame_unavailable': True}

    # Activation metadata — per-node activation values + kernel trace.
    # Empty when no expansion ran (e.g. no selected seeds, query_vec absent).
    activation_meta = {}
    if expansion:
        node_act = expansion.get('node_activation') or {}
        field_act = expansion.get('field_activation') or {}
        kernel_trace = expansion.get('trace') or []

        # Compact: short-id → activation value. Sorted descending for readability.
        activation_meta['activations'] = [
            {'id': nid[:8], 'act': round(act, 3)}
            for nid, act in sorted(node_act.items(), key=lambda x: x[1], reverse=True)
        ][:30]  # cap to 30 for trace size
        activation_meta['activation_count'] = len(node_act)

        # Top-3 per-node field activations (which fields lit up, for render debug)
        top_fields = {}
        for nid, fa in field_act.items():
            if not fa:
                continue
            top3 = sorted(fa.items(), key=lambda x: x[1], reverse=True)[:3]
            top_fields[nid[:8]] = [(f, round(a, 3)) for f, a in top3]
        # Cap at 10 nodes to bound trace size
        activation_meta['top_fields'] = dict(list(top_fields.items())[:10])

        # Kernel trace — hops, new nodes, threshold applied, edges transmitted
        activation_meta['kernel_trace'] = kernel_trace

    # K-event metadata — built as a local so the telemetry guard can inspect it
    # before the write (loud-at-the-write-boundary). The shared run-cost block
    # (build_run_telemetry) sits flat alongside the rich tool_trace/kernel_trace,
    # so Surface now carries BOTH cost and loop detail — the gap this closes.
    k_metadata = {
        'selected': sel_detail, 'expanded': exp_detail,
        # S1Surface journal (2026-07-11): Haiku's recall-level `reason` —
        # in practice the why-nothing-was-picked note (the prompt asks for
        # it only on empty selections). Not rendered to Anchor; the K trace
        # is its only durable home. Bounded so a runaway can't bloat it.
        'selection_reason': (selection_reason or '')[:500],
        **frame_meta, **activation_meta,
        # Agentic surface tool trace (v5 only; empty for v4). Stashed by
        # _call_surface_agentic on the brain instance so we don't change the
        # run_surface signature.
        'tool_trace': (getattr(brain, '_surface_tool_traces', {}) or {}).get(session_id) or [],
        # Presentation shuffle record (§20.12 A2): shuffle_seed + the exact
        # round-1 menu order Haiku saw (presented_order, 8-char ids).
        # cand_detail in the O trace stays scorer-ordered — propensity
        # analysis joins picked/dropped against presented_order.
        **((getattr(brain, '_surface_presented', {}) or {}).get(session_id) or {}),
        'surface_variant': os.environ.get('BRAIN_SURFACE_VARIANT', 'v4'),
        # telemetry is already a complete build_run_telemetry dict from both
        # surface paths; spread it flat (fallback to the all-zero block on None).
        **(telemetry or build_run_telemetry()),
        # Per-phase latency (structured, queryable — the hook_phase_timing debug
        # string's data) + the run outcome. 'served' when context reached Anchor,
        # 'empty' when nothing surfaced. ('timeout' is deferred — the daemon
        # can't observe a client abandoning the recall.)
        'phase_timing': pt.snapshot() if pt is not None else [],
        'outcome': 'served' if additional_context else 'empty',
    }
    # Loud guard — Haiku ran but recorded 0 output tokens means the cost
    # telemetry wasn't threaded. Log, don't block (write the full payload
    # regardless), same contract as the encoder check_delta_telemetry.
    _tel_warn = check_surface_telemetry(k_metadata)
    if _tel_warn:
        try:
            brain._log_error('surface_telemetry_missing',
                             RuntimeError(_tel_warn), 'K trace write boundary')
        except Exception:
            pass

    # Batch all three trace writes in one transaction
    brain._trace_dal.append_batch([
        dict(chain_id=recall_chain, scale='s1', event_type='O',
             ref_type='recall', ref_id=str(recall_ref or ''),
             summary='%d candidates for: %s' % (len(results), enriched[:100]),
             metadata={'source': 'hook', 'query': enriched[:500], 'candidates': cand_detail},
             session_id=session_id),
        dict(chain_id=recall_chain, scale='s1', event_type='K',
             ref_type='surface_selected',
             ref_id=json.dumps(list(selected_ids)),
             summary='%d surfaced, %d expanded, %d activated' % (
                 len(selected_ids), len(graph_neighbors),
                 activation_meta.get('activation_count', 0)),
             metadata=k_metadata,
             session_id=session_id),
        dict(chain_id=recall_chain, scale='s1', event_type='delta',
             ref_type='additionalContext',
             summary='%d nodes surfaced' % len(selected_ids) if selected_ids else '(no selection)',
             metadata=build_selection_metadata(
                 candidates_considered=len(results),
                 selected=selected_short,
                 dropped=dropped_short,
                 outcomes_per_candidate=outcomes_per_candidate,
                 content=additional_context or '',
                 expanded=exp_detail,
                 query=enriched[:500],
             ),
             interaction_id=interaction_id,
             session_id=session_id),
    ])


def _write_surface_result_file(recall_ref, surface_prompt, output, brain):
    """Write surface result to tmp file for dashboard pickup."""
    try:
        path = os.path.join(brain_tmp_dir(), "brain-judge-result-%s.json" % recall_ref)  # keep filename for dashboard compat
        with open(path, 'w') as f:
            json.dump({
                "recall_ref": recall_ref,
                "surface_prompt": surface_prompt,
                "surface_output": output,
            }, f)
    except Exception as e:
        brain._log_error('surface_result_write', e, 'writing surface result file')


def _drop_archived_selected(brain, selected_mode, selected_short_ids):
    """Drop archived nodes from Haiku's resolved selection, in place.

    Mutates selected_mode (full-id keyed) and selected_short_ids (8-char
    set), and logs an ERROR per event — operator mandate: an archived node
    being picked anywhere must be loud, never stat-only. The surfaced-ids
    file is written by the caller AFTER this gate — single write site, only
    the filtered set ever lands on disk.

    Returns the list of dropped full ids (for tests / callers).
    """
    if not selected_mode:
        return []
    try:
        archived = brain._nodes.archived_subset(list(selected_mode))
    except Exception as e:
        brain._log_error(
            'surface_liveness_gate', e,
            'archived check failed — selection passes unfiltered '
            '(hebbian drain gate backstops)')
        return []
    dead = sorted(nid for nid in selected_mode if nid in archived)
    if not dead:
        return []
    for nid in dead:
        selected_mode.pop(nid, None)
    selected_short_ids -= {nid[:8] for nid in dead}
    brain._log_error(
        'surface_selected_archived',
        RuntimeError('Haiku selected archived node(s) %s — dropped before '
                     'seeding' % ','.join(nid[:8] for nid in dead)),
        'liveness gate in run_surface; the id came from session history '
        '(conversation / recently-surfaced block), not the candidate menu')
    return dead


def _write_surface_selected_file(brain, session_id, stop_counter, short_ids):
    """Single write site for the per-turn surfaced-ids file.

    Hebbian + Stop hook read it. Path is scoped to session_id +
    stop_counter so consecutive turns don't overwrite each other's
    surface output before the Stop hook reads it (the counter
    increments AFTER post_response_common). Called once per
    run_surface, after the liveness gate, so only the filtered
    selection ever lands on disk.
    """
    from servers.scales.s1.surface_contract import surface_selected_path
    try:
        with open(surface_selected_path(session_id, stop_counter), 'w') as f:
            json.dump({"selected_ids": list(short_ids)}, f)
    except Exception as e:
        brain._log_error('surface_selected_write', e,
                         'writing surface-selected file')


def run_surface(brain, ctx, candidates_data, user_message,
                recent_messages, result, enriched, results, recall_ref,
                session_id, graph_changes, query_vec=None, prior_vecs=None,
                frame='', pt=None):
    """S1 Surface: Haiku-select → spread_activation → activation-render → trace.

    The complete S1 Surface chain. Called from hook_recall in daemon_hooks.py.
    query_vec + prior_vecs are required for the spreading-activation kernel —
    when absent, the surface falls back to Haiku-selected rendering only
    (no graph expansion), which is degraded but safe.

    `pt` (optional PhaseTimer): when supplied by hook_recall, surface marks
    its internal phases on the same timer so the daemon log line splits
    `surface_haiku`, `surface_spread`, `surface_render`, `surface_trace`.
    No-op if None — surface still runs, just without the breakdown.

    Returns: additional_context string or None.
    """
    load_env()

    def _mark(label):
        if pt is not None:
            pt.mark(label)

    # Call Haiku selector (unchanged — picks ≤5 from 25 candidates)
    surfaced, surface_prompt, max_tokens, interaction_id, telemetry = _call_surface(
        brain, candidates_data, user_message, recent_messages,
        session_id, result, frame=frame)
    _mark('surface_haiku')

    selected = surfaced.get("selected", [])
    # Haiku's per-recall rationale. Rendered nowhere (Anchor never sees
    # it) — its one consumer is the S1Surface journal in the K trace.
    selection_reason = surfaced.get("reason") or ''

    # Replay-bench capture stashed by _call_surface — popped (not read) so
    # a failed finish can't leak a stale capture into the next recall.
    capture = (getattr(brain, '_surface_captures', {}) or {}).pop(
        session_id, None)

    if not selected:
        _write_surface_selected_file(brain, session_id, ctx.stop_counter,
                                     set())
        try:
            _write_traces(brain, ctx, candidates_data, set(), [], [],
                          None, enriched, results,
                          recall_ref, interaction_id, session_id,
                          frame=frame, telemetry=telemetry, pt=pt,
                          selection_reason=selection_reason)
        except Exception as e:
            brain._log_error('trace_s1_surface_empty', e, 'S1 surface trace (no selection)')
        _write_surface_result_file(recall_ref, surface_prompt, "(no selection)", brain)
        # Empty selections are corpus-worthy — a prompt candidate that
        # changes WHEN Haiku picks nothing needs these to be judged.
        surface_capture.finish(
            brain, capture, recall_ref=recall_ref, surfaced=surfaced,
            resolved_mode={}, selection_reason=selection_reason,
            telemetry=telemetry)
        return None

    # Map short-id → full-id over the WHOLE candidate pool (≤25 entries) —
    # sanitized / prefix-recovered ids below must be able to land on any
    # candidate, not just ones whose raw emitted form matched.
    short_to_full = {}
    for c in candidates_data:
        cid = c.get('id', '')
        if cid:
            short_to_full[cid[:8]] = cid
    # selected_mode is the resolved-pick registry: {full_id: render_mode}.
    # Its keys ARE the selection (what spread seeds on, what the render and
    # liveness gate key off); the value is the per-node render mode, default
    # 'arc'. Valid modes come from the contract (SURFACE_MODES) — the schema
    # enum derives from the same constant, so both stay in sync by construction.
    from servers.scales.s1.surface_contract import (
        SURFACE_MODES, SURFACE_MODE_DEFAULT)
    selected_mode = {}
    for s in selected:
        raw_id = s.get('id', '')
        short_id = _sanitize_selected_id(raw_id)[:8]
        mode = (s.get('mode') or SURFACE_MODE_DEFAULT).strip().lower()
        if mode not in SURFACE_MODES:
            mode = SURFACE_MODE_DEFAULT
        full_id = short_to_full.get(short_id)
        if not full_id and short_id:
            # Whitespace corruption often leaves fewer than 8 real chars
            # ('d 6d3 f8' → 'd6d3f8') — a unique prefix of exactly one
            # candidate is still an unambiguous pick. Recover it rather
            # than dropping the selection.
            recovered = _unique_prefix_match(short_id, short_to_full)
            if recovered:
                full_id = short_to_full[recovered]
                brain._log_warning(
                    'surface_id_fuzzy_recovered',
                    'emitted id %r recovered to candidate %s by unique prefix'
                    % (raw_id, recovered),
                    'session=%s' % session_id)
        if full_id:
            selected_mode[full_id] = mode
        else:
            # Haiku returned an ID not in its candidate menu — either a
            # hallucination or a typo. Try to resolve it directly against
            # the brain (the ID might happen to match a real node from
            # session context). If found, use it; if not, log loudly so
            # the failure isn't silent.
            try:
                ndal = brain._nodes
                resolved = ndal.resolve_id(short_id)
                # 2026-05-02: Haiku occasionally drops a leading '0' from
                # 8-char IDs, producing a 7-char output (e.g. '95c2b96'
                # instead of '095c2b96'). Verified via brain error logs:
                # 2 of 4 'unresolvable' cases were leading-0 drops to real
                # nodes. When the short_id is 7 chars and doesn't resolve,
                # retry with '0' prepended. If THAT resolves, recover the
                # selection and log it as a leading-zero recovery (distinct
                # from real hallucinations).
                if not resolved and len(short_id) == 7:
                    resolved = ndal.resolve_id('0' + short_id)
                    if resolved:
                        try:
                            brain._log_error(
                                'haiku_id_leading_zero_recovered',
                                RuntimeError('Haiku dropped leading 0 — recovered'),
                                'short_id=%s recovered_as=%s' % (short_id, resolved[:8]))
                        except Exception:
                            pass
            except Exception as _re:
                # A bare except here used to mask real DB errors as
                # "ID is hallucinated" — a SQL/index issue would become
                # indistinguishable from a Haiku confabulation, breaking
                # the diagnostic value of the haiku_id_outside_candidates
                # vs surface_unknown_selected_id distinction below.
                resolved = None
                try:
                    brain._log_error(
                        'haiku_id_resolve_failed', _re,
                        'resolve_id raised for short_id=%s — treating as unresolvable but real cause logged'
                        % short_id)
                except Exception:
                    pass
            if resolved:
                selected_mode[resolved] = mode
                brain._log_error(
                    'haiku_id_outside_candidates',
                    RuntimeError('Haiku selected an ID not in its candidate menu but it resolves to a real node'),
                    'short_id=%s resolved=%s' % (short_id, resolved[:12]))
            else:
                # Single loud channel for an id that exists nowhere — the
                # scoreboard's drift section counts this stream, and the
                # dashboard error feed shows warnings alongside errors. A
                # silent drop here is exactly how the v12_1_full
                # empty-context miss went unnoticed.
                brain._log_warning(
                    'surface_unknown_selected_id',
                    'emitted id %r matches no candidate and resolves to '
                    'no node — pick dropped' % raw_id,
                    'sanitized=%s session=%s' % (short_id, session_id))

    # Trace + Hebbian-file input derives from what actually RESOLVED, so a
    # recovered pick lands as its real short id (not the corrupted emission)
    # and unresolvable ids never leak downstream.
    selected_short_ids = {fid[:8] for fid in selected_mode}

    # Liveness gate — Haiku's prompt carries node ids in historical text
    # (conversation, recently-surfaced block) that read-time archived
    # filters can't reach, so a node archived mid-session (S2 absorb) can
    # come back as a selection: the outside-candidates path above resolves
    # it to a real-but-archived node and admits it. Its vectors are gone
    # (deleted at archive), so seeding it yields zero activation, and every
    # acceptance re-writes the id into the surface_selected trace that the
    # recently-surfaced block is built from — a self-perpetuating loop
    # (2026-06-12: node 90664c51, 4 selections over 2.5h). Enforce
    # liveness structurally — code beats prompt compliance.
    _drop_archived_selected(brain, selected_mode, selected_short_ids)

    # Surfaced-ids file (Hebbian + Stop hook input) — written once,
    # after the gate, so only the filtered selection lands on disk.
    _write_surface_selected_file(brain, session_id, ctx.stop_counter,
                                 selected_short_ids)

    _mark('surface_id_resolve')

    # Graph expansion via spreading activation. The kernel replaces what
    # select_edges + per-seed top-3 neighbors + mutual-traversal used to do.
    expansion = _graph_expand(
        brain, list(selected_mode.keys()),
        query_vec=query_vec, prior_vecs=prior_vecs)
    _mark('surface_spread')

    # Activation-driven render — fields appear by their own per-field activation
    # score, budget is allocated softmax-weighted by node activation.
    from servers.scales.s1.surface_contract import format_surface_output_activation
    additional_context = format_surface_output_activation(
        node_activation=expansion['node_activation'],
        field_activation=expansion['field_activation'],
        rich_nodes=expansion['rich_nodes'],
        selected_mode=selected_mode,
        query_vec=query_vec,
        brain=brain,
    )
    _mark('surface_render')

    # Trace writing — compat neighbor list for legacy readers, plus full
    # activation data in metadata for S3 / dashboard observability.
    try:
        graph_neighbors_compat = _activation_to_trace_list(
            expansion, selected_mode)
        _write_traces(brain, ctx, candidates_data, selected_short_ids, selected,
                      graph_neighbors_compat, additional_context,
                      enriched, results,
                      recall_ref, interaction_id, session_id,
                      expansion=expansion, frame=frame, telemetry=telemetry, pt=pt,
                      selection_reason=selection_reason)
    except Exception as e:
        brain._log_error('trace_s1_surface', e, 'S1 surface trace capture')

    # Dashboard file
    _write_surface_result_file(recall_ref, surface_prompt, additional_context, brain)

    # Replay-bench capture — written last, with the post-gate resolution
    # (production's actual picks are the concordance baseline for replay).
    surface_capture.finish(
        brain, capture, recall_ref=recall_ref, surfaced=surfaced,
        resolved_mode=selected_mode, selection_reason=selection_reason,
        telemetry=telemetry)
    _mark('surface_trace')

    return additional_context


def _activation_to_trace_list(expansion, selected_mode):
    """Convert activation expansion output to the legacy neighbor-list shape
    the trace writer expects. Kept minimal — Part I will upgrade the trace
    contract itself to carry activation data natively.

    `selected_mode` keys are the seed ids — excluded from the neighbor list.
    """
    out = []
    for nid, act in expansion['node_activation'].items():
        if nid in selected_mode:
            continue  # seeds aren't "neighbors"
        rich = expansion['rich_nodes'].get(nid, {})
        out.append({
            "id": nid,
            "type": rich.get('type', ''),
            "title": rich.get('title', ''),
            "content": (rich.get('content') or '')[:300],
            "relation": "activation_spread",
            "edge_description": "activation=%.2f" % act,
            "confidence": rich.get('confidence', 0),
            "locked": rich.get('locked', 0) == 1,
            "direction": "outgoing",
            "created_at": rich.get('created_at'),
            "revised_at": rich.get('revised_at'),
            "seed_id": "(activation)",
        })
    return out


# Backward compat — old name
run_judge = run_surface
