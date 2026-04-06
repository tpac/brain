"""S1 Recall Chain — judge selection, graph expansion, correction enrichment, trace writing.

Scale: S1 (Turn integration)
Chain: s1r (recall)
Interaction: 'judge' in interactions table (learnable boundary)

Triggered by: hook_recall (UserPromptSubmit) in daemon_hooks.py
Reads: recall candidates from brain.recall(), interactions table
Writes: S1 traces (O/K/Δ), tmp files for Hebbian + dashboard
"""

import json

from servers.scales.dispatch import load_env


def _get_recently_recalled(brain, session_id):
    """Get recently judge-selected node IDs from S1 traces (for dedup)."""
    from servers.scales.s1.recall_contract import JUDGE
    lookback = JUDGE.get('recent_recalls_messages', 10)
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
    recently_recalled = []
    for nid in list(seen_ids)[:20]:
        title = dal.get_title(nid)
        if title:
            recently_recalled.append({"id": nid, "title": title})
    return recently_recalled


def _call_judge(brain, candidates_data, user_message, session_context,
                recent_messages, session_id, result):
    """Call Haiku judge to select relevant nodes from candidates.

    Returns: (judgment_dict, judge_prompt, max_tokens, interaction_id)
        judgment_dict has 'selected' list. Empty on failure.
    """
    import anthropic
    from servers.scales.s1.recall_contract import build_judge_prompt

    # Recently recalled (for dedup)
    recently_recalled = []
    try:
        recently_recalled = _get_recently_recalled(brain, session_id)
    except Exception as e:
        brain._log_error('judge_recently_recalled', e, 'fetching recently recalled titles')

    # Retrieval stats and intent from recall result
    retrieval_stats = result.get('_retrieval_stats') if isinstance(result, dict) else None
    intent = result.get('intent') if isinstance(result, dict) else None

    # Build prompt — instructions from interactions table (learnable boundary)
    judge_interaction = brain.get_interaction('judge')
    judge_instructions = judge_interaction.get('template', '') if judge_interaction else ''
    interaction_id = judge_interaction.get('id') if judge_interaction else None
    judge_prompt, max_tokens = build_judge_prompt(
        candidates_data, user_message,
        session_context=session_context,
        recent_messages=recent_messages,
        recently_recalled=recently_recalled,
        retrieval_stats=retrieval_stats,
        intent=intent,
        prompt_instructions=judge_instructions or None)

    # Call Haiku
    client = anthropic.Anthropic()
    api_resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": judge_prompt}])
    raw = api_resp.content[0].text.strip()

    # Parse JSON
    json_str = raw
    if json_str.startswith("```"):
        json_str = json_str.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = json_str.find("{")
    end = json_str.rfind("}") + 1
    if start >= 0 and end > start:
        judgment = json.loads(json_str[start:end])
    else:
        judgment = {"selected": []}

    return judgment, judge_prompt, max_tokens, interaction_id


def _expand_and_enrich(brain, selected_ids, graph_changes):
    """Graph expansion + correction enrichment from judge-selected seeds.

    Returns: (graph_neighbors, corrections)
    """
    # Graph expansion
    graph_neighbors = []
    try:
        from servers.daemon_dispatch import COMMAND_TABLE
        expand_entry = COMMAND_TABLE.get("graph_expand")
        if expand_entry:
            expand_result = expand_entry.handler(brain, {
                "node_ids": list(selected_ids),
                "depth": 1, "limit_per_seed": 3,
            }, graph_changes)
            if expand_result.get("ok"):
                graph_neighbors = expand_result.get("result", {}).get("neighbors", [])
    except Exception as e:
        brain._log_error('judge_graph_expand', e, 'graph expand failed')

    # Correction enrichment
    corrections = {}
    try:
        from servers.scales.s1.recall_contract import correction_enrich
        all_ids = set(selected_ids)
        for nb in graph_neighbors:
            if nb.get("id"):
                all_ids.add(nb["id"])
        corrections = correction_enrich(all_ids, brain.conn)
    except Exception as e:
        brain._log_error('correction_enrich', e, 'correction enrichment')

    return graph_neighbors, corrections


def _write_traces(brain, ctx, candidates_data, selected_ids, selected,
                  graph_neighbors, additional_context, enriched, results,
                  recall_ref, interaction_id, session_id):
    """Write S1 recall traces: O (candidates), K (judge-selected), Δ (additionalContext)."""
    recall_chain = ctx.s1r_chain()

    # O: candidates detail
    cand_detail = ['%s|%s|%.2f|%s' % (
        c.get('id', '')[:8], c.get('title', '')[:80],
        c.get('score', 0), c.get('type', ''))
        for c in candidates_data[:25]]

    # K: selected detail
    sel_detail = ['%s|%s' % (c.get('id', '')[:8], c.get('title', ''))
                  for c in candidates_data if c.get('id', '')[:8] in selected_ids]

    # Expanded detail
    exp_detail = ['%s|%s|%s' % (
        nb.get('id', '')[:8], nb.get('title', '')[:60], nb.get('relation', ''))
        for nb in graph_neighbors[:10]]

    # Batch all three trace writes in one transaction to avoid
    # "database is locked" from encoding agent thread writing between commits
    brain._trace_dal.append_batch([
        dict(chain_id=recall_chain, scale='s1', event_type='O',
             ref_type='recall', ref_id=str(recall_ref or ''),
             summary='%d candidates for: %s' % (len(results), enriched[:100]),
             metadata={'source': 'hook', 'query': enriched[:500], 'candidates': cand_detail},
             session_id=session_id),
        dict(chain_id=recall_chain, scale='s1', event_type='K',
             ref_type='judge_selected',
             ref_id=json.dumps(list(selected_ids)),
             summary='%d selected, %d expanded' % (len(selected), len(graph_neighbors)),
             metadata={'selected': sel_detail, 'expanded': exp_detail},
             session_id=session_id),
        dict(chain_id=recall_chain, scale='s1', event_type='delta',
             ref_type='additionalContext',
             summary='%d nodes surfaced' % len(selected) if selected else '(no selection)',
             metadata={'content': (additional_context or '')[:4000]},
             interaction_id=interaction_id,
             session_id=session_id),
    ])


def _write_judge_result_file(recall_ref, judge_prompt, output, brain):
    """Write judge result to tmp file for dashboard pickup."""
    try:
        path = "/tmp/brain-judge-result-%s.json" % recall_ref
        with open(path, 'w') as f:
            json.dump({
                "recall_ref": recall_ref,
                "judge_prompt": judge_prompt,
                "judge_output": output,
            }, f)
    except Exception as e:
        brain._log_error('judge_result_write', e, 'writing judge result file')


def run_judge(brain, ctx, candidates_data, user_message, session_context,
              recent_messages, result, enriched, results, recall_ref,
              session_id, graph_changes):
    """S1 recall: judge → expand → enrich → format → trace.

    The complete S1R chain. Called from hook_recall in daemon_hooks.py.
    Returns: additional_context string or None.
    """
    load_env()

    # Call the judge
    judgment, judge_prompt, max_tokens, interaction_id = _call_judge(
        brain, candidates_data, user_message, session_context, recent_messages,
        session_id, result)

    selected = judgment.get("selected", [])
    selected_ids = {s.get("id", "")[:8] for s in selected}

    # Write judge-selected IDs for Hebbian + Stop hook
    try:
        judge_path = "/tmp/brain-%s-judge-selected.json" % session_id
        with open(judge_path, 'w') as f:
            json.dump({"selected_ids": list(selected_ids)}, f)
    except Exception as e:
        brain._log_error('judge_selected_write', e, 'writing judge-selected file')

    if not selected:
        # Still write traces so dashboard shows the recall attempt
        try:
            _write_traces(brain, ctx, candidates_data, set(), [], [],
                          None, enriched, results,
                          recall_ref, interaction_id, session_id)
        except Exception as e:
            brain._log_error('trace_s1_recall_empty', e, 'S1 recall trace (no selection)')
        _write_judge_result_file(recall_ref, judge_prompt, "(no selection)", brain)
        return None

    # Expand and enrich
    graph_neighbors, corrections = _expand_and_enrich(brain, selected_ids, graph_changes)

    # Format output
    from servers.scales.s1.recall_contract import format_judge_output
    additional_context = format_judge_output(selected, candidates_data, graph_neighbors,
                                             corrections=corrections)

    # Write traces
    try:
        _write_traces(brain, ctx, candidates_data, selected_ids, selected,
                      graph_neighbors, additional_context, enriched, results,
                      recall_ref, interaction_id, session_id)
    except Exception as e:
        brain._log_error('trace_s1_recall', e, 'S1 recall trace capture')

    # Dashboard file
    _write_judge_result_file(recall_ref, judge_prompt, additional_context, brain)

    return additional_context
