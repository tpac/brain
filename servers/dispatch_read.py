"""Daemon dispatch — read handlers (recall + node/graph reads).

Lock-free reads: recall, batch recall, get_node(s), title lookup, structured
filter, graph expansion, boot context.
"""

from .dispatch_common import _resolve_id


def _enrich_recall_results(brain, result, graph_changes):
    """Enrich recall results via brain.get_node() — the shared data atom.

    Anchor's MCP recall gets full enrichment per node:
    metadata, corrections, connections, situation.
    """
    results = result.get("results", [])
    if not results:
        return

    # Enrich each result with brain.get_node() data
    for r in results[:8]:
        rich = brain.get_node(r.get("id", ""))
        if rich:
            r["_metadata"] = rich.get("_metadata", {})
            r["_corrections"] = rich.get("_corrections", [])
            r["connections"] = rich.get("connections", [])
            r["situation"] = rich.get("situation", "")


def _handle_recall(brain, args, graph_changes):
    # By-ID recall: returns single enriched node
    node_id = args.get("node_id")
    if node_id:
        node_id = _resolve_id(brain, node_id)
        result = brain.recall_node(
            node_id, neighbor_limit=args.get("neighbor_limit", 3))
        return {"ok": True, "result": result}

    sid = args.get("session_id", "")
    # By-query recall. brain.recall() already blends keyword + semantic and
    # degrades internally, so there is no separate keyword-only mode to fall
    # back to at this layer. If it raises, surface the failure loudly instead
    # of masking it as an empty/degraded success.
    try:
        result = brain.recall(
            query=args.get("query", ""), filter=args.get("filter"),
            limit=args.get("limit", 8), session_id=sid, source='mcp')
    except Exception as e:
        brain._log_error("recall_failed", e, "MCP by-query recall raised")
        return {"ok": False, "error": "recall failed: %s" % e}

    # Enrich with corrections, graph expansion, metadata — same context as hook path
    _enrich_recall_results(brain, result, graph_changes)

    return {"ok": True, "result": result}


def _handle_recall_batch(brain, args, graph_changes):
    """Batch recall — multiple queries in one call."""
    queries = args.get("queries", [])
    limit = args.get("limit", 5)
    batch_filter = args.get("filter")
    sid = args.get("session_id", "")
    results = []
    for q in queries[:10]:  # cap at 10 queries
        try:
            result = brain.recall(query=q, filter=batch_filter, limit=limit,
                                  session_id=sid, source='mcp')
            results.append({"query": q, "results": result.get("results", [])})
        except Exception as e:
            results.append({"query": q, "results": [], "error": str(e)})
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
        top_k=args.get("top_k", 1))
    return {"ok": True, "result": result}


def _handle_filter_nodes(brain, args, graph_changes):
    """Structured query: filter nodes by any structural field."""
    result = brain.filter_nodes(
        field=args.get("field", ""),
        include=args.get("include"),
        exclude=args.get("exclude"),
        lt=args.get("lt"),
        gt=args.get("gt"),
        limit=args.get("limit", 50),
        sort_by=args.get("sort_by", "created_at"),
        sort_order=args.get("sort_order", "desc"),
        rich=args.get("rich", True))
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

    from .brain_constants import EXCLUDED_EDGE_TYPES
    node_dal = brain._nodes
    graph_dal = brain._graph
    seen = set(node_ids)
    neighbors = []
    excluded = set(EXCLUDED_EDGE_TYPES)

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
