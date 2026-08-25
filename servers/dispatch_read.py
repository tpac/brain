"""Daemon dispatch — read handlers (recall + node/graph reads).

Lock-free reads: recall, batch recall, get_node(s), title lookup, structured
filter, graph expansion, boot context.
"""

from .dispatch_common import _resolve_id, caller_session, _agent_limit


def _handle_recall(brain, args, graph_changes):
    # By-ID recall: returns single enriched node (canonicalized inside
    # recall_node, so this door and the by-query one below agree)
    node_id = args.get("node_id")
    if node_id:
        node_id = _resolve_id(brain, node_id)
        result = brain.recall_node(
            node_id, session_id=caller_session(args))
        return {"ok": True, "result": result}

    sid = caller_session(args)  # identity: drives this session's fatigue/activity
    # By-query recall. brain.recall() already blends keyword + semantic and
    # degrades internally, so there is no separate keyword-only mode to fall
    # back to at this layer. If it raises, surface the failure loudly instead
    # of masking it as an empty/degraded success.
    #
    # mark_accessed defaults True (every real caller). A read-only observer —
    # the dashboard's recall probe — passes False so its looking never lands
    # in access_count / last_accessed / fatigue, which are the brain's own
    # record of what IT recalled.
    try:
        result = brain.recall(
            query=args.get("query", ""), filter=args.get("filter"),
            limit=args.get("limit", 8), session_id=sid,
            source=args.get("source") or 'mcp',
            mark_accessed=args.get("mark_accessed", True))
    except Exception as e:
        brain._log_error("recall_failed", e, "MCP by-query recall raised")
        return {"ok": False, "error": "recall failed: %s" % e}

    brain.canonicalize_results(result.get("results", []), session_id=sid)

    return {"ok": True, "result": result}


def _handle_recall_batch(brain, args, graph_changes):
    """Batch recall — N queries, each through the SAME door as single recall.

    Delegates to _handle_recall per query instead of re-running the
    recall + canonicalize sequence, so the batch cannot drift from single
    recall: one handler body, one place a failure gets logged
    (brain._log_error inside _handle_recall), one place mark_accessed /
    source are honored. The per-query shape stays compact — {query, results}
    — dropping the stats/gap chrome single recall renders, which is the
    intended batch compaction, not drift. A canonicalize failure propagates
    loud, exactly as it does for single recall — a batch is no reason to
    swallow an unreadable correction chain.
    """
    queries = args.get("queries", [])
    # Control fields shared by every query — everything except the batch's own
    # `queries` (and any by-id / single-query key a caller mis-bundled, so each
    # delegated call is an unambiguous by-query recall). Carries the
    # proxy-stamped caller session so identity resolves the same per call.
    # `limit` is injected with the batch default (5, vs single recall's 8) so
    # delegation honors the batch contract.
    base = {k: v for k, v in args.items()
            if k not in ("queries", "query", "node_id")}
    base["limit"] = args.get("limit", 5)
    results = []
    for q in queries[:10]:  # cap at 10 queries
        r = _handle_recall(brain, {**base, "query": q}, graph_changes)
        if r.get("ok"):
            results.append(
                {"query": q, "results": (r.get("result") or {}).get("results", [])})
        else:
            results.append(
                {"query": q, "results": [], "error": r.get("error", "recall failed")})
    return {"ok": True, "result": results}


def _handle_get_node(brain, args, graph_changes):
    node_id = _resolve_id(brain, args.get("node_id", ""))
    if not node_id:
        return {"ok": False, "error": "node_id is required"}

    node = brain.get_node(node_id)
    if not node:
        return {"ok": False, "error": "Node not found: {}".format(node_id)}

    return {"ok": True, "result": node}


def _handle_get_nodes(brain, args, graph_changes):
    """Batch get_node — multiple node IDs in one call, same rich shape as get_node."""
    node_ids = args.get("node_ids", [])
    resolved_ids = []
    errors = []
    for nid in node_ids[:20]:  # cap at 20
        resolved = _resolve_id(brain, nid)
        if resolved:
            resolved_ids.append(resolved)
        else:
            errors.append({"id": nid, "error": "not found"})

    rich_map = brain.get_node(resolved_ids) if resolved_ids else {}
    # Batch returns {node_id: rich_dict}. Preserve request order.
    results = [rich_map[nid] for nid in resolved_ids if nid in rich_map]
    results.extend(errors)
    return {"ok": True, "result": results}


def _handle_find_node_by_title(brain, args, graph_changes):
    result = brain.find_node_by_title(
        title_query=args.get("title_query", ""),
        threshold=args.get("threshold", 0.75),
        top_k=args.get("top_k", 1),
        session_id=caller_session(args))
    return {"ok": True, "result": result}


def _handle_filter_nodes(brain, args, graph_changes):
    """Structured query: filter nodes by any structural field."""
    from servers.contract import NODE_QUERY_DEFAULT_LIMIT, NODE_QUERY_MAX_LIMIT
    result = brain.filter_nodes(
        field=args.get("field", ""),
        include=args.get("include"),
        exclude=args.get("exclude"),
        lt=args.get("lt"),
        gt=args.get("gt"),
        contains=args.get("contains"),
        prefix=args.get("prefix"),
        limit=_agent_limit(args.get("limit"),
                           NODE_QUERY_DEFAULT_LIMIT, NODE_QUERY_MAX_LIMIT),
        sort_by=args.get("sort_by", "created_at"),
        sort_order=args.get("sort_order", "desc"),
        rich=args.get("rich", True),
        session_id=caller_session(args))
    if "error" in result:
        return {"ok": False, "error": result["error"]}
    return {"ok": True, "result": result}


def _handle_graph_expand(brain, args, graph_changes):
    """Layer 3: expand from surface-selected seed nodes via structural edges.

    Args:
        node_ids: list of seed node IDs (from surface selection)
        depth: how many hops (default 1)
        limit_per_seed: max neighbors per seed (default 3)

    Returns: list of neighbor nodes with edge info, deduplicated.
    """
    node_ids = args.get("node_ids", [])
    depth = min(args.get("depth", 1), 2)  # Cap at 2 hops
    limit_per = args.get("limit_per_seed", 3)

    if not node_ids:
        return {"ok": True, "result": {"neighbors": []}}

    node_dal = brain._nodes
    graph_dal = brain._graph
    # Scope veil: this door returns 300-char content previews across edges
    # — exactly how content crosses a wall. Seeding `seen` with the veil
    # excludes walled nodes from every hop (no production caller today;
    # gated so the next caller inherits the wall).
    _veil = brain.scope_veil(caller_session(args))
    seen = set(node_ids) | set(_veil)
    neighbors = []
    # Same noise-derived exclusion every traversal uses (was a drifted
    # 1-member literal that leaked co_accessed edges into graph_expand).
    excluded = set(brain.aspects.traversal_exclusions)

    for seed_id in node_ids:
        full_id = seed_id
        if len(seed_id) < 16:
            resolved = node_dal.resolve_id(seed_id)
            if resolved:
                full_id = resolved

        rows = graph_dal.get_neighbors(
            full_id,
            limit=limit_per,
            exclude_relations=excluded,
            exclude_node_ids=seen,
            content_preview_chars=300,
        )
        for r in rows:
            if r['id'] not in seen:
                seen.add(r['id'])
                neighbors.append({
                    "id": r['id'], "type": r['type'], "title": r['title'],
                    "content": r.get('content_preview', ''),
                    "edge_type": r['relation'],
                    "edge_weight": r['weight'],
                    "edge_description": r.get('edge_description') or '',
                    "confidence": r['confidence'],
                    "direction": r['direction'],
                    "seed_id": full_id,
                })

    return {"ok": True, "result": {"neighbors": neighbors, "seeds": len(node_ids)}}


def _handle_context_boot(brain, args, graph_changes):
    text = brain.format_boot_context(
        user=args.get("user", "User"),
        project=args.get("project", "default"),
        db_dir=args.get("db_dir", ""),
        session_id=args.get("session_id", ""))
    # Faithful boot capture: record exactly what we served this session so the
    # dashboard can show "what actually got to boot". Only on real
    # SessionStarts (session_id present); record_boot_render is best-effort and
    # never raises. format_boot_context returns the wrapped string the hook
    # prints verbatim; tolerate a dict shape defensively.
    sid = args.get("session_id", "")
    if sid:
        served = text if isinstance(text, str) else (
            (text or {}).get("for_claude") or (text or {}).get("text") or "")
        brain.record_boot_render(sid, served, args.get("user", ""), args.get("project", ""))
    return {"ok": True, "result": text}
