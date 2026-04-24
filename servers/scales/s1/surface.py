"""S1 Surface Chain — surface relevant memories, graph expansion, correction enrichment, trace writing.

Scale: S1 (Turn integration)
Chain: s1r (surface)
Interaction: 'surface' in interactions table (learnable boundary)

Triggered by: hook_recall (UserPromptSubmit) in daemon_hooks.py
Reads: recall candidates from brain.recall(), interactions table
Writes: S1 traces (O/K/Δ), tmp files for Hebbian + dashboard
"""

import json

from servers.scales.dispatch import load_env
from servers.trace_contract import build_selection_metadata


def _get_recently_surfaced(brain, session_id):
    """Get recently surfaced node IDs from S1 traces (for dedup)."""
    from servers.scales.s1.surface_contract import SURFACE
    lookback = SURFACE.get('recent_recalls_messages', 10)
    recent_k = brain._trace_dal.get_by_ref_type(
        'surface_selected', scale='s1', hours=24, limit=lookback)
    # Backward compat: also check old 'judge_selected' ref_type
    if not recent_k:
        recent_k = brain._trace_dal.get_by_ref_type(
            'judge_selected', scale='s1', hours=24, limit=lookback)
    seen_ids = set()
    for evt in recent_k:
        try:
            for nid in json.loads(evt.get('ref_id', '[]')):
                seen_ids.add(nid)
        except (ValueError, TypeError):
            pass
    from servers.dal import NodeDAL
    dal = NodeDAL(brain.conn)
    recently_surfaced = []
    for nid in list(seen_ids)[:20]:
        title = dal.get_title(nid)
        if title:
            recently_surfaced.append({"id": nid, "title": title})
    return recently_surfaced


def _call_surface(brain, candidates_data, user_message, session_context,
                  recent_messages, session_id, result):
    """Call Haiku to surface relevant nodes from candidates.

    Returns: (surfaced_dict, surface_prompt, max_tokens, interaction_id)
        surfaced_dict has 'selected' list. Empty on failure.
    """
    import anthropic
    from servers.scales.s1.surface_contract import build_surface_prompt

    # Recently surfaced (for dedup)
    recently_surfaced = []
    try:
        recently_surfaced = _get_recently_surfaced(brain, session_id)
    except Exception as e:
        brain._log_error('surface_recently_recalled', e, 'fetching recently surfaced titles')

    # Retrieval stats and intent from recall result
    retrieval_stats = result.get('_retrieval_stats') if isinstance(result, dict) else None
    intent = result.get('intent') if isinstance(result, dict) else None

    # Build prompt — instructions from interactions table (learnable boundary)
    surface_interaction = brain.get_interaction('surface')
    # Backward compat: fall back to 'judge' interaction if 'surface' not found
    if not surface_interaction:
        surface_interaction = brain.get_interaction('judge')
    surface_instructions = surface_interaction.get('template', '') if surface_interaction else ''
    interaction_id = surface_interaction.get('id') if surface_interaction else None
    surface_prompt, max_tokens = build_surface_prompt(
        candidates_data, user_message,
        session_context=session_context,
        recent_messages=recent_messages,
        recently_recalled=recently_surfaced,
        retrieval_stats=retrieval_stats,
        intent=intent,
        prompt_instructions=surface_instructions or None)

    client = anthropic.Anthropic()
    api_resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": surface_prompt}])
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


def _graph_expand(brain, selected_ids, query_vec=None, prior_vecs=None):
    """Expand the graph from selected seeds via spreading activation.

    Replaces the old weight-sorted degree-1 expansion. The new mechanism
    is one kernel (`spread_activation`) that does what select_edges,
    per-seed top-3, and mutual-traversal-via-communities did separately.

    Depth is emergent — high-coefficient edges propagate further; low-
    coefficient edges stop on the first hop. Two seeds whose paths meet
    at a shared neighbor (e.g. community node) boost that neighbor above
    singleton reach automatically.

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
        'trace':            per-hop diagnostics
    """
    from servers.dal import NodeDAL
    from servers.scales.s1.surface_contract import spread_activation

    if query_vec is None or not selected_ids:
        return {'node_activation': {}, 'field_activation': {},
                'rich_nodes': {}, 'trace': []}

    # Resolve to full IDs (the kernel reads vectors keyed on full id)
    ndal = NodeDAL(brain.conn)
    resolved = []
    for sid in selected_ids:
        full = ndal.resolve_id(sid) if len(str(sid)) < 16 else sid
        if full:
            resolved.append(full)

    if not resolved:
        return {'node_activation': {}, 'field_activation': {},
                'rich_nodes': {}, 'trace': []}

    result = spread_activation(resolved, query_vec, brain, prior_vecs=prior_vecs)

    # Batch-load rich node data for everything activated
    all_ids = list(result['node_activation'].keys())
    rich_nodes = brain.get_node(all_ids) if all_ids else {}

    return {
        'node_activation':  result['node_activation'],
        'field_activation': result['field_activation'],
        'rich_nodes':       rich_nodes,
        'trace':            result['trace'],
    }


def _write_traces(brain, ctx, candidates_data, selected_ids, selected,
                  graph_neighbors, additional_context, enriched, results,
                  recall_ref, interaction_id, session_id, expansion=None):
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
                       **activation_meta},
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


def run_surface(brain, ctx, candidates_data, user_message, session_context,
                recent_messages, result, enriched, results, recall_ref,
                session_id, graph_changes, query_vec=None, prior_vecs=None):
    """S1 Surface: Haiku-select → spread_activation → activation-render → trace.

    The complete S1 Surface chain. Called from hook_recall in daemon_hooks.py.
    query_vec + prior_vecs are required for the spreading-activation kernel —
    when absent, the surface falls back to Haiku-selected rendering only
    (no graph expansion), which is degraded but safe.

    Returns: additional_context string or None.
    """
    load_env()

    # Call Haiku selector (unchanged — picks ≤5 from 25 candidates)
    surfaced, surface_prompt, max_tokens, interaction_id = _call_surface(
        brain, candidates_data, user_message, session_context, recent_messages,
        session_id, result)

    selected = surfaced.get("selected", [])
    selected_short_ids = {s.get("id", "")[:8] for s in selected}

    # Write surfaced IDs for Hebbian + Stop hook.
    try:
        surface_path = "/tmp/brain-%s-surface-selected.json" % session_id
        with open(surface_path, 'w') as f:
            json.dump({"selected_ids": list(selected_short_ids)}, f)
    except Exception as e:
        brain._log_error('surface_selected_write', e, 'writing surface-selected file')

    if not selected:
        try:
            _write_traces(brain, ctx, candidates_data, set(), [], [],
                          None, enriched, results,
                          recall_ref, interaction_id, session_id)
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
    for s in selected:
        short_id = s.get('id', '')[:8]
        full_id = short_to_full.get(short_id)
        if full_id:
            selected_why[full_id] = s.get('why', '')
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
            except Exception:
                resolved = None
            if resolved:
                selected_why[resolved] = s.get('why', '')
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

    # Graph expansion via spreading activation. The kernel replaces what
    # select_edges + per-seed top-3 neighbors + mutual-traversal used to do.
    expansion = _graph_expand(
        brain, list(selected_why.keys()),
        query_vec=query_vec, prior_vecs=prior_vecs)

    # Activation-driven render — fields appear by their own per-field activation
    # score, budget is allocated softmax-weighted by node activation.
    from servers.scales.s1.surface_contract import format_surface_output_activation
    additional_context = format_surface_output_activation(
        node_activation=expansion['node_activation'],
        field_activation=expansion['field_activation'],
        rich_nodes=expansion['rich_nodes'],
        selected_why=selected_why,
        query_vec=query_vec,
        brain=brain,
        session=ctx,
    )

    # Trace writing — compat neighbor list for legacy readers, plus full
    # activation data in metadata for S3 / dashboard observability.
    try:
        graph_neighbors_compat = _activation_to_trace_list(
            expansion, selected_why)
        _write_traces(brain, ctx, candidates_data, selected_short_ids, selected,
                      graph_neighbors_compat, additional_context,
                      enriched, results,
                      recall_ref, interaction_id, session_id,
                      expansion=expansion)
    except Exception as e:
        brain._log_error('trace_s1_surface', e, 'S1 surface trace capture')

    # Dashboard file
    _write_surface_result_file(recall_ref, surface_prompt, additional_context, brain)

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
