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

    # Instructions live in the system block; per-turn delta (Frame,
    # conversation, candidates, message) goes in the user message. The
    # earlier design relied on cache_control: ephemeral to amortize the
    # prefix — removed 2026-05-09 after measuring the prefix at ~2390
    # tokens, below Haiku 4.5's 4096-token cacheable minimum. See the
    # client-construction comment below for the full reasoning.
    #
    # interaction_seed.py guarantees 'surface' is registered on every boot.
    # If it's missing here, that's a real bug in the seed path, not
    # something to silently degrade past — crash loudly and let the seed
    # be fixed. (Loud-by-Default: silent fallbacks are how the prior
    # cache_control marker stayed broken for six days.)
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

    # surface_prompt kept for trace/debug (concatenation of system + user)
    surface_prompt = (surface_instructions + "\n\n---\n\n" + user_content) \
        if surface_instructions else user_content

    # Reuse the daemon-lifetime client constructed in Brain.warm_up(). The
    # client holds the warm httpx connection pool + already-handshaken TLS
    # session to api.anthropic.com — constructing per call here would re-pay
    # those costs. Falls back to a fresh client only if warmup didn't run
    # (e.g. early test harness, warmup raised, brain freshly loaded outside
    # the daemon path) — degrades gracefully to the old behavior.
    #
    # The cache_control: ephemeral marker that used to live on the system
    # block was removed 2026-05-09 — measured by count_tokens, the prefix is
    # ~2390 tokens, below Haiku 4.5's 4096-token minimum cacheable prefix.
    # The marker silently no-ops at that size (cache_creation_input_tokens=0
    # on every call). It was scaffolding that lied about runtime behavior.
    # If the system block ever grows past 4096, add it back; until then,
    # don't claim a feature we don't have.
    client = getattr(brain, 'anthropic_client', None)
    if client is None:
        # warmup didn't run (test harness, ad-hoc Brain instance, or
        # warmup raised) — construct on-demand and pay cold-start. Keeps
        # the call site working in environments where Brain.warm_up()
        # isn't on the boot path. The daemon always has it.
        import anthropic
        client = anthropic.Anthropic()

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
      - 'cluster_l4': cluster + L4 identity lane (locked nodes always-include,
        corrects-of-touched edges, recent-session moments).

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
        'l4_lane':          {full_node_id: str} — origin tag for L4 nodes
                            ('locked' / 'correction_of:<id>' / 'session_moment')
                            (cluster_l4 only).
        'trace':            per-hop diagnostics
    """
    import os
    from servers.dal import NodeDAL
    from servers.scales.s1.surface_contract import (
        spread_activation, spread_activation_cluster)

    if query_vec is None or not selected_ids:
        return {'node_activation': {}, 'field_activation': {},
                'rich_nodes': {}, 'convergence': {}, 'l4_lane': {}, 'trace': []}

    raw_variant = os.environ.get('BRAIN_RECALL_VARIANT', 'baseline').lower()

    # Variant names are composable tags: e.g. 'baseline_l4_lim15' means
    # original spread + L4 lane + per-source neighbor limit 15. Three knobs:
    #   spread:  'baseline' (default) | 'cluster' (cluster-completion v1)
    #   lane:    'l4' present in name = L4 identity lane on
    #   limit:   'lim15' / 'lim10' / 'lim20' set per-source neighbor cap
    #   thickness: edge weight as transmission multiplier (read in spread)
    #   lineage:   lineage families bypass median gate (read in spread)
    use_cluster = 'cluster' in raw_variant
    use_l4 = 'l4' in raw_variant
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
            "[surface-variant first-call pid=%d] raw=%r cluster=%s l4=%s "
            "limit=%s thickness=%s lineage=%s\n" % (
                os.getpid(), raw_variant, use_cluster, use_l4,
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
                'rich_nodes': {}, 'convergence': {}, 'l4_lane': {}, 'trace': []}

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

    # L4 — identity lane. Augment node_activation with always-include nodes
    # whose presence is identity-determined, not relevance-determined.
    l4_lane = {}
    if use_l4:
        l4_nodes = _l4_identity_lane(brain, resolved + list(node_activation.keys()))
        for nid, origin in l4_nodes.items():
            l4_lane[nid] = origin
            # Set activation high enough to render but not above the kernel's
            # peak (so render budget still favors strongly-activated nodes
            # over identity-lane ones in tight budgets). 0.5 is a placeholder
            # until we measure how the renderer actually uses these.
            if nid not in node_activation:
                node_activation[nid] = max(node_activation.get(nid, 0.0), 0.5)
                # No field_activation for L4-only nodes — render layer
                # will fall back to title+content.

    # Batch-load rich node data for everything activated
    all_ids = list(node_activation.keys())
    rich_nodes = brain.get_node(all_ids) if all_ids else {}

    return {
        'node_activation':  node_activation,
        'field_activation': field_activation,
        'rich_nodes':       rich_nodes,
        'convergence':      convergence,
        'l4_lane':          l4_lane,
        'trace':            result['trace'],
    }


def _l4_identity_lane(brain, touched_ids):
    """L4 — surface identity-determined nodes regardless of relevance score.

    Three sources, each with a different origin tag for traceability:
      - 'locked': all locked nodes (axioms — Anchor identity rules)
      - 'correction_of:<id>': any node that `corrects` a touched node, so
        Anchor doesn't speak from outdated knowledge.
      - 'session_moment': recent S0 message-stream entries from this session
        (handled separately at the surface layer; not yet wired here).

    Returns: {full_node_id: origin_tag}

    Cheap by design — three SQL queries, no embeddings, no traversal.
    Touched_ids is the set of nodes already surfaced by L0-L3, used to
    target the corrections lookup.
    """
    out = {}

    try:
        # 1. Locked nodes — identity axioms, always present.
        locked_rows = brain.conn.execute(
            "SELECT id FROM nodes WHERE locked = 1 AND archived = 0"
        ).fetchall()
        for row in locked_rows:
            out[row[0]] = 'locked'
    except Exception as e:
        brain._log_error('l4_locked_query', e, 'L4 identity lane: locked nodes')

    # 2. Corrections OF touched nodes. The brain's design rule
    # ([ec1ef964](https://example/node/ec1ef964)): corrections are a
    # traversal rule, not a judge rule. If something Anchor's about to
    # speak from has been corrected, the correction comes too.
    if touched_ids:
        try:
            placeholders = ','.join('?' * len(touched_ids))
            # source corrects target — surface the source (the correction)
            # when target is a touched node.
            correction_rows = brain.conn.execute(
                "SELECT DISTINCT e.source_id, e.target_id "
                "FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id "
                "WHERE er.relation IN ('corrects', 'corrected_by', 'updates', 'supersedes') "
                "AND er.archived = 0 "
                "AND e.target_id IN (%s)" % placeholders,
                touched_ids).fetchall()
            for source_id, target_id in correction_rows:
                if source_id not in out:
                    out[source_id] = 'correction_of:' + target_id[:8]
        except Exception as e:
            brain._log_error('l4_correction_query', e,
                             'L4 identity lane: corrections of touched')

    return out


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
                       **frame_meta, **activation_meta},
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
        query_vec=query_vec,
        brain=brain,
        session=ctx,
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
