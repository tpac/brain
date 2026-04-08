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

    # Call Haiku
    client = anthropic.Anthropic()
    api_resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": surface_prompt}])
    raw = api_resp.content[0].text.strip()

    # Parse JSON
    json_str = raw
    if json_str.startswith("```"):
        json_str = json_str.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = json_str.find("{")
    end = json_str.rfind("}") + 1
    if start >= 0 and end > start:
        surfaced = json.loads(json_str[start:end])
    else:
        surfaced = {"selected": []}

    return surfaced, surface_prompt, max_tokens, interaction_id


def _graph_expand(brain, selected_ids):
    """Fetch graph neighbors for selected nodes. Seeds already enriched via get_rich_node.

    Only discovers new neighbor nodes for the "Related knowledge" section.
    No correction re-fetch (already in candidates_data._corrections).
    No metadata re-fetch (already in candidates_data._metadata).

    Returns: list of neighbor dicts [{id, type, title, content, edge_type, ...}]
    """
    from servers.pipeline_contract import TRAVERSE_EXCLUDED_EDGES
    from servers.dal import NodeDAL

    conn = brain.conn
    ndal = NodeDAL(conn)
    excluded = TRAVERSE_EXCLUDED_EDGES
    excl_placeholders = ','.join('?' for _ in excluded)

    seen = set()
    # Resolve and track seed IDs
    resolved = set()
    for sid in selected_ids:
        full = ndal.resolve_id(sid) if len(str(sid)) < 16 else sid
        if full:
            resolved.add(full)
            seen.add(full)
            seen.add(full[:8])

    neighbors = []
    for full_id in resolved:
        rows = conn.execute("""
            SELECT n.id, n.type, n.title, substr(n.content, 1, 300),
                   e.edge_type, e.weight, e.description,
                   n.confidence, n.locked
            FROM edges e
            JOIN nodes n ON n.id = CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
            WHERE (e.source_id = ? OR e.target_id = ?) AND n.archived = 0
            AND n.id != ?
            AND e.edge_type NOT IN ({excl})
            ORDER BY e.weight DESC LIMIT 3
        """.format(excl=excl_placeholders),
            [full_id, full_id, full_id, full_id] + list(excluded)).fetchall()

        for r in rows:
            if r[0] not in seen:
                seen.add(r[0])
                neighbors.append({
                    "id": r[0], "type": r[1], "title": r[2],
                    "content": r[3], "edge_type": r[4],
                    "edge_weight": r[5], "edge_description": r[6] or "",
                    "confidence": r[7], "locked": r[8] == 1,
                    "seed_id": full_id,
                })

    return neighbors


def _write_traces(brain, ctx, candidates_data, selected_ids, selected,
                  graph_neighbors, additional_context, enriched, results,
                  recall_ref, interaction_id, session_id):
    """Write S1 surface traces: O (candidates), K (surfaced), Δ (additionalContext)."""
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
             summary='%d surfaced, %d expanded' % (len(selected), len(graph_neighbors)),
             metadata={'selected': sel_detail, 'expanded': exp_detail},
             session_id=session_id),
        dict(chain_id=recall_chain, scale='s1', event_type='delta',
             ref_type='additionalContext',
             summary='%d nodes surfaced' % len(selected) if selected else '(no selection)',
             metadata={'content': (additional_context or '')[:4000]},
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
                session_id, graph_changes):
    """S1 Surface: surface → expand → enrich → format → trace.

    The complete S1 Surface chain. Called from hook_recall in daemon_hooks.py.
    Returns: additional_context string or None.
    """
    load_env()

    # Call the surfacer
    surfaced, surface_prompt, max_tokens, interaction_id = _call_surface(
        brain, candidates_data, user_message, session_context, recent_messages,
        session_id, result)

    selected = surfaced.get("selected", [])
    selected_ids = {s.get("id", "")[:8] for s in selected}

    # Write surfaced IDs for Hebbian + Stop hook
    try:
        surface_path = "/tmp/brain-%s-judge-selected.json" % session_id  # keep filename for compat
        with open(surface_path, 'w') as f:
            json.dump({"selected_ids": list(selected_ids)}, f)
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

    # Graph expansion — neighbors only (seeds already enriched via get_rich_node)
    graph_neighbors = _graph_expand(brain, selected_ids)

    # Format output — candidates already have _metadata, _corrections, connections
    from servers.scales.s1.surface_contract import format_surface_output
    additional_context = format_surface_output(selected, candidates_data, graph_neighbors)

    # Write traces
    try:
        _write_traces(brain, ctx, candidates_data, selected_ids, selected,
                      graph_neighbors, additional_context, enriched, results,
                      recall_ref, interaction_id, session_id)
    except Exception as e:
        brain._log_error('trace_s1_surface', e, 'S1 surface trace capture')

    # Dashboard file
    _write_surface_result_file(recall_ref, surface_prompt, additional_context, brain)

    return additional_context


# Backward compat — old name
run_judge = run_surface
