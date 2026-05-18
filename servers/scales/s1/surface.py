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

from servers.scales.dispatch import load_env
from servers.trace_contract import build_selection_metadata


# SURFACE_SELECTION_SCHEMA lives in surface_contract.py alongside the other
# surface I/O contracts (the render formats for each mode). Imported below.
def _get_recently_surfaced(brain, session_id):
    """Get recently surfaced node IDs from S1 traces (for dedup)."""
    from servers.scales.s1.surface_contract import SURFACE
    lookback = SURFACE.get('recent_recalls_messages', 10)
    recent_k = brain._trace_dal.get_by_ref_type(
        'surface_selected', scale='s1', hours=24, limit=lookback)
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
    from servers.dal import NodeDAL
    dal = NodeDAL(brain.conn)
    recently_surfaced = []
    for nid in list(seen_ids)[:20]:
        title = dal.get_title(nid)
        if title:
            recently_surfaced.append({"id": nid, "title": title})
    return recently_surfaced


def _call_surface(brain, candidates_data, user_message,
                  recent_messages, session_id, result, frame=''):
    """Call Haiku to surface relevant nodes from candidates.

    Returns: (surfaced_dict, surface_prompt, max_tokens, interaction_id)
        surfaced_dict has 'selected' list. Empty on failure.

    Surface variant gating (2026-05-10):
      BRAIN_SURFACE_VARIANT=v4 (default) — current path, single Haiku call,
        no tools. Production default. Byte-identical to pre-2026-05-10.
      BRAIN_SURFACE_VARIANT=v5_agentic — agentic loop with 6 fetch tools.
        Eval-only until explicitly opted in. Reads surface prompt v5 from
        the active version pointer (must be activated separately).
    """
    from servers.scales.s1.surface_contract import (
        build_surface_prompt, SURFACE_MODEL)

    # Recently surfaced (for dedup)
    recently_surfaced = []
    try:
        recently_surfaced = _get_recently_surfaced(brain, session_id)
    except Exception as e:
        brain._log_error('surface_recently_recalled', e, 'fetching recently surfaced titles')

    # Retrieval stats and intent from recall result
    retrieval_stats = result.get('_retrieval_stats') if isinstance(result, dict) else None
    intent = result.get('intent') if isinstance(result, dict) else None

    # interaction_seed.py guarantees 'surface' is registered on every boot.
    surface_interaction = brain.get_interaction('surface')
    if not surface_interaction or not surface_interaction.get('template'):
        raise RuntimeError(
            "S1 Surface: no 'surface' interaction registered in "
            "brain_logs.db. interaction_seed should have populated "
            "this on Brain construction — check seed/DAL state.")
    surface_instructions = surface_interaction['template']
    interaction_id = surface_interaction.get('id')

    user_content, max_tokens = build_surface_prompt(
        candidates_data, user_message,
        recent_messages=recent_messages,
        recently_recalled=recently_surfaced,
        retrieval_stats=retrieval_stats,
        intent=intent,
        frame=frame)

    surface_prompt = (surface_instructions + "\n\n---\n\n" + user_content) \
        if surface_instructions else user_content

    # Variant gate — env var picks the path. v4 is the production default.
    variant = os.environ.get('BRAIN_SURFACE_VARIANT', 'v4').strip().lower()

    client = getattr(brain, 'anthropic_client', None)
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    if variant == 'v5_agentic':
        # Agentic path: Haiku has tools, can extend the candidate pool
        # before final selection. Tool-fetched candidates are appended to
        # `candidates_data` in place so the downstream short_to_full
        # mapping resolves them.
        raw, tool_trace = _call_surface_agentic(
            client, brain, candidates_data, surface_instructions,
            user_content, max_tokens, session_id, SURFACE_MODEL)
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
        api_resp = client.messages.create(
            model=SURFACE_MODEL,
            max_tokens=max_tokens,
            system=surface_instructions,
            messages=[{"role": "user", "content": user_content}])
        raw = api_resp.content[0].text.strip()

    # Parse JSON — robust to the three shapes Haiku sometimes returns:
    #   (a) bare JSON: {"selected": [...]}
    #   (b) fenced: ```json\n{...}\n```
    #   (c) JSON + trailing prose: {...}\n\nHere's why I picked...
    # `raw_decode` consumes the first valid JSON object and reports the
    # tail, which we discard — no "Extra data" crash on (c).
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

    return surfaced, surface_prompt, max_tokens, interaction_id


def _call_surface_agentic(client, brain, candidates_data, surface_instructions,
                           user_content, max_tokens, session_id, model,
                           max_rounds=2):
    """Agentic surface call: Haiku may use fetch tools to extend the candidate
    pool before final JSON selection.

    Returns: (raw_final_text, tool_trace) where tool_trace is a list of
    per-round dicts {round, tool_calls: [...]} for trace observability.

    Mutates `candidates_data` IN PLACE — tool-fetched candidates are appended
    so the downstream short_to_full ID mapping (in run_surface) can resolve them.
    """
    from servers.scales.s1.fetch_tools import (
        TOOL_DEFINITIONS, execute_tool, format_tool_result_for_haiku,
    )
    from servers.scales.s1.surface_contract import SURFACE_SELECTION_SCHEMA

    # Track existing IDs to dedupe tool-fetched candidates against cosine pool
    existing_ids = {c.get('id') for c in candidates_data if isinstance(c, dict)}

    messages = [{"role": "user", "content": user_content}]
    tool_trace = []
    raw_final = ''

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

        api_kwargs = {
            'model': model,
            'max_tokens': max_tokens,
            'system': surface_instructions,
            'messages': messages,
            'output_config': output_config,
        }
        if not is_final:
            # Final round omits tools — forces a finalization. Earlier rounds
            # offer tools alongside the schema; Haiku may either tool_use (no
            # schema constraint on tool_use blocks) or finalize JSON.
            api_kwargs['tools'] = TOOL_DEFINITIONS

        try:
            api_resp = client.messages.create(**api_kwargs)
        except Exception as e:
            brain._log_error('surface_agentic_api', e,
                              'agentic Haiku call round=%d' % round_idx)
            return raw_final, tool_trace

        stop_reason = api_resp.stop_reason
        round_record = {'round': round_idx, 'stop_reason': stop_reason,
                         'tool_calls': []}

        # Collect any text content from this round (last round usually has it).
        for block in api_resp.content:
            if getattr(block, 'type', None) == 'text':
                raw_final = block.text.strip()

        if stop_reason != 'tool_use':
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
                # Append fetched results to candidates_data (dedupe)
                for cand in exec_result.get('results') or []:
                    cid = cand.get('id') if isinstance(cand, dict) else None
                    if cid and cid not in existing_ids:
                        candidates_data.append(cand)
                        existing_ids.add(cid)
                # Record for trace
                round_record['tool_calls'].append({
                    'tool': tool_name,
                    'args': tool_input,
                    'result_count': len(exec_result.get('results') or []),
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
                    "content": format_tool_result_for_haiku(exec_result),
                })
        messages.append({"role": "assistant", "content": assistant_blocks})
        messages.append({"role": "user", "content": tool_results})
        tool_trace.append(round_record)

    return raw_final, tool_trace


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
    from servers.dal import NodeDAL
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
    ndal = NodeDAL(brain.conn)
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
                  frame=''):
    """Write S1 surface traces: O (candidates), K (surfaced), Δ (additionalContext).

    `expansion` carries activation data from spread_activation when present —
    we attach per-node activation values and the kernel's per-hop trace to
    the K-event metadata so dashboards / S3 can see which nodes lit up and
    by how much, not just which were surfaced.
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
                 len(selected), len(graph_neighbors),
                 activation_meta.get('activation_count', 0)),
             metadata={'selected': sel_detail, 'expanded': exp_detail,
                       **frame_meta, **activation_meta,
                       # Agentic surface tool trace (v5 only; empty for v4).
                       # Stashed by _call_surface_agentic on the brain instance
                       # so we don't change the run_surface signature.
                       'tool_trace': (getattr(brain, '_surface_tool_traces', {}) or {}).get(session_id) or [],
                       'surface_variant': os.environ.get('BRAIN_SURFACE_VARIANT', 'v4')},
             session_id=session_id),
        dict(chain_id=recall_chain, scale='s1', event_type='delta',
             ref_type='additionalContext',
             summary='%d nodes surfaced' % len(selected) if selected else '(no selection)',
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
        path = "/tmp/brain-judge-result-%s.json" % recall_ref  # keep filename for dashboard compat
        with open(path, 'w') as f:
            json.dump({
                "recall_ref": recall_ref,
                "surface_prompt": surface_prompt,
                "surface_output": output,
            }, f)
    except Exception as e:
        brain._log_error('surface_result_write', e, 'writing surface result file')


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
    surfaced, surface_prompt, max_tokens, interaction_id = _call_surface(
        brain, candidates_data, user_message, recent_messages,
        session_id, result, frame=frame)
    _mark('surface_haiku')

    selected = surfaced.get("selected", [])
    selected_short_ids = {s.get("id", "")[:8] for s in selected}

    # Write surfaced IDs for Hebbian + Stop hook. Path is scoped to
    # session_id + stop_counter so consecutive turns don't overwrite each
    # other's surface output before the Stop hook reads it. Hebbian on the
    # same turn reads the same path (counter hasn't incremented yet at
    # Stop time — increment happens AFTER post_response_common).
    try:
        surface_path = "/tmp/brain-%s-%d-surface-selected.json" % (session_id, ctx.stop_counter)
        with open(surface_path, 'w') as f:
            json.dump({"selected_ids": list(selected_short_ids)}, f)
    except Exception as e:
        brain._log_error('surface_selected_write', e, 'writing surface-selected file')

    if not selected:
        try:
            _write_traces(brain, ctx, candidates_data, set(), [], [],
                          None, enriched, results,
                          recall_ref, interaction_id, session_id,
                          frame=frame)
        except Exception as e:
            brain._log_error('trace_s1_surface_empty', e, 'S1 surface trace (no selection)')
        _write_surface_result_file(recall_ref, surface_prompt, "(no selection)", brain)
        return None

    # Map short-id → full-id + why using candidates_data (they were enriched
    # with full IDs on the way in). Also collect Haiku's "why" per seed.
    short_to_full = {}
    selected_why = {}
    for c in candidates_data:
        cid = c.get('id', '')
        if cid[:8] in selected_short_ids:
            short_to_full[cid[:8]] = cid
    hallucinated_ids = []
    # selected_mode: per-node render mode (fact|arc|background), default 'arc'.
    # Surface v5 may emit `mode` per selected item; v4 omits it → all 'arc'.
    selected_mode = {}
    for s in selected:
        short_id = s.get('id', '')[:8]
        full_id = short_to_full.get(short_id)
        if full_id:
            selected_why[full_id] = s.get('why', '')
            mode = (s.get('mode') or 'arc').strip().lower()
            if mode not in ('fact', 'arc', 'background'):
                mode = 'arc'
            selected_mode[full_id] = mode
        else:
            # Haiku returned an ID not in its candidate menu — either a
            # hallucination or a typo. Try to resolve it directly against
            # the brain (the ID might happen to match a real node from
            # session context). If found, use it; if not, log loudly so
            # the failure isn't silent.
            try:
                from servers.dal import NodeDAL
                ndal = NodeDAL(brain.conn)
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
                # vs haiku_id_unresolvable distinction below.
                resolved = None
                try:
                    brain._log_error(
                        'haiku_id_resolve_failed', _re,
                        'resolve_id raised for short_id=%s — treating as unresolvable but real cause logged'
                        % short_id)
                except Exception:
                    pass
            if resolved:
                selected_why[resolved] = s.get('why', '')
                mode = (s.get('mode') or 'arc').strip().lower()
                if mode not in ('fact', 'arc', 'background'):
                    mode = 'arc'
                selected_mode[resolved] = mode
                brain._log_error(
                    'haiku_id_outside_candidates',
                    RuntimeError('Haiku selected an ID not in its candidate menu but it resolves to a real node'),
                    'short_id=%s resolved=%s why=%r' % (short_id, resolved[:12], s.get('why', '')[:80]))
            else:
                hallucinated_ids.append(short_id)
    if hallucinated_ids:
        brain._log_error(
            'haiku_id_unresolvable',
            RuntimeError('Haiku selected IDs that exist nowhere in the brain'),
            'ids=%s why_samples=%r' % (
                ','.join(hallucinated_ids[:5]),
                [s.get('why', '')[:80] for s in selected if s.get('id', '')[:8] in hallucinated_ids][:3]))

    _mark('surface_id_resolve')

    # Graph expansion via spreading activation. The kernel replaces what
    # select_edges + per-seed top-3 neighbors + mutual-traversal used to do.
    expansion = _graph_expand(
        brain, list(selected_why.keys()),
        query_vec=query_vec, prior_vecs=prior_vecs)
    _mark('surface_spread')

    # Activation-driven render — fields appear by their own per-field activation
    # score, budget is allocated softmax-weighted by node activation.
    from servers.scales.s1.surface_contract import format_surface_output_activation
    additional_context = format_surface_output_activation(
        node_activation=expansion['node_activation'],
        field_activation=expansion['field_activation'],
        rich_nodes=expansion['rich_nodes'],
        selected_why=selected_why,
        selected_mode=selected_mode,
        query_vec=query_vec,
        brain=brain,
    )
    _mark('surface_render')

    # Trace writing — compat neighbor list for legacy readers, plus full
    # activation data in metadata for S3 / dashboard observability.
    try:
        graph_neighbors_compat = _activation_to_trace_list(
            expansion, selected_why)
        _write_traces(brain, ctx, candidates_data, selected_short_ids, selected,
                      graph_neighbors_compat, additional_context,
                      enriched, results,
                      recall_ref, interaction_id, session_id,
                      expansion=expansion, frame=frame)
    except Exception as e:
        brain._log_error('trace_s1_surface', e, 'S1 surface trace capture')

    # Dashboard file
    _write_surface_result_file(recall_ref, surface_prompt, additional_context, brain)
    _mark('surface_trace')

    return additional_context


def _activation_to_trace_list(expansion, selected_why):
    """Convert activation expansion output to the legacy neighbor-list shape
    the trace writer expects. Kept minimal — Part I will upgrade the trace
    contract itself to carry activation data natively.
    """
    out = []
    for nid, act in expansion['node_activation'].items():
        if nid in selected_why:
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
