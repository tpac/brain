"""Fetch toolbox for the agentic surface.

Six tools, each a thin wrapper over an existing brain primitive. The tool
NAMES carry intent vocabulary — Haiku picking a tool IS the intent
classification. No separate classifier.

DESIGN CONTRACT
---------------
- Brain primitives are single source of truth (servers/brain.py).
  Tools wrap, never duplicate.
- Each tool returns the same candidate shape so spread_activation + render
  don't need new branches.
- Inputs accept natural language ('last 10 hours', 'yesterday') — tools
  parse to timestamps. Haiku doesn't compute dates.
- Tools never raise; failures return [] with a logged warning.

LOOP CONTROL
------------
- Max 3 rounds (Haiku-thinks → tools-run → Haiku-thinks → ...).
- Parallel tool calls per round encouraged (Anthropic native).
- Behavioral discipline (in surface prompt) prevents iterating same query.

See docs/AGENTIC-SURFACE-CONTRACT.md for the full spec.
"""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


# ─── Tool definitions exposed to Haiku via Anthropic `tools` param ───────

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "recall_topical",
        "description": (
            "Topical semantic recall — embeddings + lexical bridge. Use when the user "
            "asks about a topic or concept. NOT for time-bounded ('yesterday'), "
            "verbatim quotes ('what did X say'), or session-continuation queries. "
            "This is the default fall-back when no other tool's intent matches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The semantic query — what to find similar to."},
                "k": {"type": "integer", "description": "Max results (default 25)", "default": 25},
            },
            "required": ["query"],
        },
    },
    {
        "name": "recall_recent",
        "description": (
            "Chronological session-aware recall. Use when the user signals continuation "
            "or time-recency: 'what did we do', 'last session', 'this morning', 'pick "
            "up from yesterday', 'recent work'. Returns nodes touched/created recently "
            "in chronological order, not topical similarity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "window": {
                    "type": "string",
                    "description": (
                        "Natural-language window. Examples: 'last 10 hours', 'last 3 "
                        "turns', 'today', 'yesterday', 'since last session', 'last 24h'. "
                        "Tool parses to timestamps — you do NOT need to compute dates."
                    ),
                },
                "k": {"type": "integer", "description": "Max results (default 25)", "default": 25},
            },
            "required": ["window"],
        },
    },
    {
        "name": "recall_by_time",
        "description": (
            "Time-bounded recall, optionally combined with a semantic query. "
            "Use when the user asks about a specific time window — 'what did we "
            "do in March 2023', 'Q1 2024 launch work', 'before May 2024'. "
            "Distinct from recall_recent (which is rolling from now). "
            "Distinct from recall_by_date (REMOVED) — this version filters on "
            "extracted EVENT time (when something happened) by default, not "
            "when the node was encoded.\n\n"
            "Ranking tiers when both `query` and time are given:\n"
            "  1. Entities matching BOTH query and time range (top)\n"
            "  2. Entities matching just the query (no time match)\n"
            "  3. Entities matching just the time range\n\n"
            "If an edge is matched by time, the response also includes its "
            "source and target nodes so the relation has context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_when": {
                    "type": "string",
                    "description": (
                        "Range start. Natural language ok: 'January 2023', "
                        "'last month', 'Q1 2024', '2024-05'. Omit / empty "
                        "string = open-ended past."
                    ),
                },
                "end_when": {
                    "type": "string",
                    "description": (
                        "Range end. Natural language ok: 'before May', "
                        "'March 2024', '2024-03-31'. Omit / empty string = "
                        "open-ended future."
                    ),
                },
                "time_anchor": {
                    "type": "string",
                    "enum": ["event", "created", "updated"],
                    "default": "event",
                    "description": (
                        "Which date to filter on:\n"
                        "  event   — extracted event time (default; matches "
                        "what the user usually means)\n"
                        "  created — when the node/edge was first encoded\n"
                        "  updated — when last revised / strengthened"
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Optional semantic filter. When provided, results "
                        "are tiered: query∩time first, query-only second, "
                        "time-only third."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max results (default 20)",
                },
            },
        },
    },
    {
        "name": "recall_verbatim",
        "description": (
            "Verbatim phrase lookup. Use when the user is asking about EXACT wording "
            "('what did X say about Y', 'find the quote where', 'literal phrase'). "
            "Bypasses semantic similarity — pure full-text lexical match."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "phrase": {"type": "string", "description": "The exact phrase or distinctive words to match."},
                "k": {"type": "integer", "description": "Max results (default 10)", "default": 10},
            },
            "required": ["phrase"],
        },
    },
    {
        "name": "recall_by_aspect",
        "description": (
            "Recall by semantic family (aspect). Use when the user asks for a CATEGORY "
            "of knowledge: 'what corrections', 'open threads', 'identity nodes', "
            "'recent decisions'. Aspect is one of: identity_bearing (principles, "
            "rules, visions), episodic_anchor (moments, quotes, events), "
            "active_thread (open work, gaps, hypotheses), lesson_insight (decisions, "
            "findings, lessons, mechanisms, patterns), correction_improvement "
            "(corrections, fixes), or one of the edge-family aspects "
            "(temporal_sequence, dependency_flow, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "aspect": {
                    "type": "string",
                    "description": (
                        "Aspect name from the 14-aspect taxonomy. Common: "
                        "identity_bearing, episodic_anchor, active_thread, "
                        "lesson_insight, correction_improvement."
                    ),
                },
                "recent_first": {"type": "boolean", "description": "Sort newest first (default true)", "default": True},
                "k": {"type": "integer", "description": "Max results (default 25)", "default": 25},
            },
            "required": ["aspect"],
        },
    },
    {
        "name": "expand_node",
        "description": (
            "Constellation expansion from a known node. Use when you already have ONE "
            "good candidate and want its graph neighborhood — the things connected to "
            "it. Returns nodes within `hops` distance. Good for 'tell me more about X'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_ref": {
                    "type": "string",
                    "description": (
                        "Node identifier — either 8+ character node_id OR a fuzzy "
                        "title match (tool will resolve to the best match)."
                    ),
                },
                "hops": {"type": "integer", "description": "Traversal depth (default 1)", "default": 1},
            },
            "required": ["node_ref"],
        },
    },
]


# ─── Natural-language window/date parser ─────────────────────────────────

def _parse_window(window: str, now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """Parse a natural-language 'recent window' string into (start, end) UTC.

    Always returns (start <= end). End is `now` for rolling windows.
    Conservative defaults — unknown strings fall back to last 24h.
    """
    if now is None:
        from servers.clock import brain_now
        now = brain_now()
    s = (window or '').strip().lower()

    # 'last N hours', 'last N h', 'last Nh'
    m = re.match(r"last\s+(\d+)\s*h(?:ours?)?$", s) or re.match(r"last\s+(\d+)h$", s)
    if m:
        return (now - timedelta(hours=int(m.group(1))), now)

    # 'last N minutes'
    m = re.match(r"last\s+(\d+)\s*m(?:in(?:utes?)?)?$", s)
    if m:
        return (now - timedelta(minutes=int(m.group(1))), now)

    # 'last N days', 'last N d', 'last Nd'
    m = re.match(r"last\s+(\d+)\s*d(?:ays?)?$", s) or re.match(r"last\s+(\d+)d$", s)
    if m:
        return (now - timedelta(days=int(m.group(1))), now)

    # 'last N weeks'
    m = re.match(r"last\s+(\d+)\s*w(?:eeks?)?$", s)
    if m:
        return (now - timedelta(weeks=int(m.group(1))), now)

    # 'last 24h' / '24h' / '10h' shorthand
    m = re.match(r"(\d+)\s*h$", s)
    if m:
        return (now - timedelta(hours=int(m.group(1))), now)

    if s in ('today',):
        return (now.replace(hour=0, minute=0, second=0, microsecond=0), now)
    if s in ('yesterday',):
        start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return (start_today - timedelta(days=1), start_today)
    if s in ('this morning',):
        return (now.replace(hour=0, minute=0, second=0, microsecond=0),
                now.replace(hour=12, minute=0, second=0, microsecond=0))
    if s in ('this afternoon',):
        return (now.replace(hour=12, minute=0, second=0, microsecond=0), now)
    if s in ('this week',):
        return (now - timedelta(days=now.weekday()), now)
    if s in ('last week',):
        end = now - timedelta(days=now.weekday())
        return (end - timedelta(days=7), end)
    if s in ('this month',):
        return (now.replace(day=1, hour=0, minute=0, second=0, microsecond=0), now)

    # 'since last session' / 'last session' — heuristic: last 24h
    if 'last session' in s or 'previous session' in s:
        return (now - timedelta(hours=24), now)

    # Fallback: try dateutil parser
    try:
        from dateutil import parser as _dp
        dt = _dp.parse(s, fuzzy=True)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (dt, now)
    except Exception:
        pass

    # Final fallback: last 24h
    return (now - timedelta(hours=24), now)


def _parse_date_expr(expr: Any, now: Optional[datetime] = None) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Parse a date expression into (since, until) UTC. Either may be None
    (open-ended). Accepts strings OR dicts {since: ..., until: ...}."""
    if now is None:
        from servers.clock import brain_now
        now = brain_now()

    if isinstance(expr, dict):
        since = expr.get('since')
        until = expr.get('until')

        def _to_dt(v):
            if not v:
                return None
            try:
                from dateutil import parser as _dp
                dt = _dp.parse(str(v), fuzzy=True)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                return None
        return (_to_dt(since), _to_dt(until))

    s = (str(expr) if expr is not None else '').strip().lower()

    if s.startswith('since '):
        rest = s[len('since '):].strip()
        try:
            from dateutil import parser as _dp
            dt = _dp.parse(rest, fuzzy=True)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (dt, now)
        except Exception:
            return (None, now)

    if s.startswith('before '):
        rest = s[len('before '):].strip()
        try:
            from dateutil import parser as _dp
            dt = _dp.parse(rest, fuzzy=True)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (None, dt)
        except Exception:
            return (None, None)

    if s.startswith('on '):
        rest = s[len('on '):].strip()
        try:
            from dateutil import parser as _dp
            dt = _dp.parse(rest, fuzzy=True)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            return (day_start, day_end)
        except Exception:
            return (None, None)

    # Reuse window parser for common shortcuts
    start, end = _parse_window(s, now=now)
    return (start, end)


# ─── Candidate shape normalization ──────────────────────────────────────

def _to_candidate(node: Dict[str, Any], score: float, source_tool: str) -> Dict[str, Any]:
    """Attach recall-specific fields (score, source_tool, discovery) to a node dict.

    Preserves the full rich shape — every field the fetcher provided rides
    through to the formatter. execute_tool() runs a batched brain.get_node
    pass after this so the final candidate is fully enriched (_corrections,
    _metadata, connections), keeping tool-fetched results symmetric with
    the initial cosine 25.
    """
    if not isinstance(node, dict):
        return None
    nid = node.get('id') or node.get('node_id')
    if not nid:
        return None
    cand = dict(node)
    cand['id'] = nid
    cand['score'] = float(score)
    cand['source_tool'] = source_tool
    # `discovery` is what format_candidate_for_surface reads for the
    # `via:<tool>` header. Set here so the initial-25 vs tool-fetched
    # rendering is symmetric without format_tool_result_for_haiku
    # having to setdefault later.
    cand.setdefault('discovery', source_tool)
    return cand


# ─── The six tools ───────────────────────────────────────────────────────

def recall_topical(brain, query: str, k: int = 25, **_) -> List[Dict[str, Any]]:
    """Topical semantic recall — wraps brain.recall(). The current cosine + FTS5
    path. Default fallback when no other tool's intent fires."""
    try:
        results = brain.recall(query=query, limit=int(k))
        if isinstance(results, dict):
            results = results.get('results') or results.get('items') or []
        out = []
        for r in (results or []):
            cand = _to_candidate(r, r.get('score', 0.0) if isinstance(r, dict) else 0.0,
                                  'recall_topical')
            if cand:
                out.append(cand)
        return out[:int(k)]
    except Exception as e:
        print('[fetch_tools] recall_topical failed: %s' % e, file=sys.stderr)
        return []


def recall_recent(brain, session_id: str = '', window: str = 'last 10 hours',
                  k: int = 25, **_) -> List[Dict[str, Any]]:
    """Chronological session-aware recall. Returns nodes touched recently —
    by trace events in this session and/or created in the time window."""
    try:
        since, until = _parse_window(window)
        since_iso = since.isoformat()
        until_iso = until.isoformat()
        # Strategy: filter_nodes by updated_at in the window, sort recent first.
        # Filter on 'updated_at' to catch revised nodes too, not just created.
        # filter_nodes exposes gt/lt (exclusive); for window queries the
        # exclusive vs inclusive distinction at second-level boundaries is
        # noise. Use gt/lt with the parsed window endpoints.
        rows = brain.filter_nodes(field='updated_at', gt=since_iso, lt=until_iso,
                                   sort_by='updated_at', sort_order='desc',
                                   limit=int(k), rich=True)
        nodes = rows.get('nodes') if isinstance(rows, dict) else rows
        if not nodes:
            return []
        out = []
        for i, n in enumerate(nodes):
            # Recency-score: 1.0 newest, decays linearly to 0.5 across window
            score = 1.0 - (0.5 * i / max(1, len(nodes)))
            cand = _to_candidate(n, score, 'recall_recent')
            if cand:
                out.append(cand)
        return out[:int(k)]
    except Exception as e:
        print('[fetch_tools] recall_recent failed: %s' % e, file=sys.stderr)
        return []


def _parse_when_to_ts(when: str, default_to_end: bool = False) -> Optional[int]:
    """Convert a natural-language time expression to Unix seconds.

    `default_to_end`: when the parsed expression has month/year precision
    and represents an endpoint, push to end-of-period (e.g. 'March 2023'
    as end_when -> 2023-03-31T23:59:59 vs 2023-03-01 as start_when).

    Returns None if `when` is empty or unparseable.
    """
    if not when or not when.strip():
        return None
    try:
        from servers.temporal_extraction import (
            extract_intervals_from_text,
        )
        intervals = extract_intervals_from_text(when)
        if intervals:
            start_ts, end_ts, _ = intervals[0]
            return end_ts if default_to_end else start_ts
    except Exception:
        pass
    # Fallback: existing date-expr parser (handles 'yesterday', 'last
    # week', etc. via dateutil).
    try:
        since, until = _parse_date_expr(when)
        if default_to_end:
            target = until or since
        else:
            target = since or until
        if target is None:
            return None
        return int(target.timestamp())
    except Exception:
        return None


def _fetch_edges_with_endpoints(brain, edge_ids: List[str]) -> List[Dict[str, Any]]:
    """For each edge_id, return [edge_synthetic_candidate, source_node,
    target_node]. The edge candidate is synthetic — title carries the
    relation, content carries the description, type='edge_relation'.

    Endpoint nodes are fetched in ONE batch via brain.get_node(list),
    not per-endpoint, to avoid N+1 lookups when an operator's question
    surfaces many edges in the same time window.
    """
    if not edge_ids:
        return []
    out = []
    conn = brain.conn
    placeholders = ','.join('?' * len(edge_ids))
    rows = conn.execute(
        f'''SELECT e.edge_id, e.source_id, e.target_id,
                   er.relation, er.description
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            WHERE e.edge_id IN ({placeholders})
              AND (er.archived IS NULL OR er.archived = 0)''',
        edge_ids,
    ).fetchall()
    if not rows:
        return []

    # One batch fetch for every endpoint we'll need.
    endpoint_ids: set = set()
    for _, source_id, target_id, _, _ in rows:
        endpoint_ids.add(source_id)
        endpoint_ids.add(target_id)
    try:
        nodes_map = brain.get_node(list(endpoint_ids))
        if not isinstance(nodes_map, dict):
            nodes_map = {}
    except Exception:
        nodes_map = {}

    seen_endpoint_ids: set = set()
    for edge_id, source_id, target_id, relation, description in rows:
        # Synthetic edge candidate.
        out.append({
            'id': edge_id,
            'title': '%s → %s : %s' % (
                source_id[:8], target_id[:8], (relation or '')[:60]),
            'type': 'edge_relation',
            'content': description or '',
            'score': 0.5,
            'source_tool': 'recall_by_time',
            'kv': {
                'edge_relation': relation,
                'source_id': source_id,
                'target_id': target_id,
            },
            'tier': 'edge_time',
        })
        # Source / target as context candidates (one per endpoint, dedup'd).
        for endpoint_id, role in [(source_id, 'edge_source'),
                                    (target_id, 'edge_target')]:
            if endpoint_id in seen_endpoint_ids:
                continue
            seen_endpoint_ids.add(endpoint_id)
            node = nodes_map.get(endpoint_id)
            if not node:
                continue
            cand = _to_candidate(node, 0.4, 'recall_by_time')
            if cand:
                cand['tier'] = role
                out.append(cand)
    return out


def recall_by_time(brain, start_when: str = '', end_when: str = '',
                    time_anchor: str = 'event', query: Optional[str] = None,
                    limit: int = 20, **_) -> List[Dict[str, Any]]:
    """Time-bounded recall with optional semantic ranking.

    Ranking tiers (when query provided):
      1. query AND time match — top
      2. query only — middle
      3. time only — bottom (edges include their source/target nodes)

    When query is empty/None, returns only tier 3 (time matches).
    When both start_when and end_when are empty, returns []
    (no scope to filter on).
    """
    try:
        # 1. Parse time range. Use very wide bounds for open-ended sides.
        start_ts = _parse_when_to_ts(start_when, default_to_end=False)
        end_ts = _parse_when_to_ts(end_when, default_to_end=True)
        if start_ts is None and end_ts is None:
            return []
        if start_ts is None:
            start_ts = 0
        if end_ts is None:
            end_ts = 9_999_999_999  # year 2286

        # 2. Get time-matching entity ids based on anchor.
        time_node_ids: set = set()
        time_edge_ids: set = set()
        conn = brain.conn
        if time_anchor == 'event':
            from servers.temporal_extraction import _SENTINEL_SOURCE
            # Nodes: filter archived via JOIN. Archived nodes' entity_dates
            # rows are stale and must not surface. Sentinel rows (marking
            # "processed, no dates") are filtered via extraction_source.
            for r in conn.execute(
                '''SELECT DISTINCT ed.entity_id
                   FROM entity_dates ed
                   JOIN nodes n ON n.id = ed.entity_id
                   WHERE ed.entity_kind = 'node'
                     AND ed.extraction_source != ?
                     AND ed.start_ts <= ?
                     AND ed.end_ts >= ?
                     AND n.archived = 0''',
                (_SENTINEL_SOURCE, end_ts, start_ts),
            ):
                time_node_ids.add(r[0])
            # Edges: archived-relation filter applied downstream by
            # _fetch_edges_with_endpoints (it joins active edge_relations
            # only). Pulling all edge entity_ids here is cheap.
            for r in conn.execute(
                '''SELECT DISTINCT entity_id FROM entity_dates
                   WHERE entity_kind = 'edge'
                     AND extraction_source != ?
                     AND start_ts <= ?
                     AND end_ts >= ?''',
                (_SENTINEL_SOURCE, end_ts, start_ts),
            ):
                time_edge_ids.add(r[0])
        elif time_anchor == 'created':
            # Point-in-time anchors use exclusive bounds (gt/lt), matching
            # the prior `recall_by_date` semantics via `filter_nodes`.
            # Second-level boundary precision is irrelevant for date windows
            # and keeping the legacy bounds avoids surprises for callers
            # that depended on exclusive-boundary behavior.
            from datetime import datetime as _dt, timezone as _tz
            start_iso = _dt.fromtimestamp(start_ts, tz=_tz.utc).isoformat()
            end_iso = _dt.fromtimestamp(end_ts, tz=_tz.utc).isoformat()
            for r in conn.execute(
                "SELECT id FROM nodes WHERE archived = 0 "
                "AND created_at > ? AND created_at < ?",
                (start_iso, end_iso),
            ):
                time_node_ids.add(r[0])
            for r in conn.execute(
                "SELECT edge_id FROM edges "
                "WHERE created_at > ? AND created_at < ?",
                (start_iso, end_iso),
            ):
                time_edge_ids.add(r[0])
        elif time_anchor == 'updated':
            from datetime import datetime as _dt, timezone as _tz
            start_iso = _dt.fromtimestamp(start_ts, tz=_tz.utc).isoformat()
            end_iso = _dt.fromtimestamp(end_ts, tz=_tz.utc).isoformat()
            for r in conn.execute(
                "SELECT id FROM nodes WHERE archived = 0 "
                "AND COALESCE(revised_at, created_at) > ? "
                "AND COALESCE(revised_at, created_at) < ?",
                (start_iso, end_iso),
            ):
                time_node_ids.add(r[0])
            for r in conn.execute(
                "SELECT edge_id FROM edges "
                "WHERE COALESCE(last_strengthened, created_at) > ? "
                "AND COALESCE(last_strengthened, created_at) < ?",
                (start_iso, end_iso),
            ):
                time_edge_ids.add(r[0])
        else:
            return []  # unknown anchor

        # 3. Semantic matches (if query). Over-fetch so tiers can fill.
        semantic_results: List[Dict[str, Any]] = []
        if query and query.strip():
            try:
                raw = brain.recall(query=query, limit=int(limit) * 3)
                if isinstance(raw, dict):
                    raw = raw.get('results') or raw.get('items') or []
                semantic_results = list(raw or [])
            except Exception:
                semantic_results = []
        semantic_node_ids = {
            r.get('id') for r in semantic_results if r.get('id')
        }

        # 4. Compute tiers.
        out: List[Dict[str, Any]] = []

        def _emit(node_or_dict, tier: str, score: float) -> None:
            if len(out) >= limit:
                return
            cand = _to_candidate(node_or_dict, score, 'recall_by_time')
            if cand:
                cand['tier'] = tier
                out.append(cand)

        # Tier 1: query ∩ time (semantic_results filtered to time-matched ids)
        for r in semantic_results:
            if len(out) >= limit:
                break
            if r.get('id') in time_node_ids:
                _emit(r, 'query+time', r.get('score', 0.7))

        # Tier 2: query only (semantic_results not in time)
        for r in semantic_results:
            if len(out) >= limit:
                break
            if r.get('id') not in time_node_ids:
                _emit(r, 'query', r.get('score', 0.6) * 0.9)

        # Tier 3: time only — nodes not already covered + edges with endpoints
        if len(out) < limit:
            tier3_node_ids = list(time_node_ids - semantic_node_ids)
            if tier3_node_ids:
                try:
                    nodes_map = brain.get_node(tier3_node_ids[:limit])
                    if isinstance(nodes_map, dict):
                        for nid, node in nodes_map.items():
                            if len(out) >= limit:
                                break
                            _emit(node, 'time', 0.5)
                except Exception:
                    pass

        if len(out) < limit and time_edge_ids:
            edge_cands = _fetch_edges_with_endpoints(
                brain, list(time_edge_ids)[:limit])
            for c in edge_cands:
                if len(out) >= limit:
                    break
                out.append(c)

        return out[:int(limit)]
    except Exception as e:
        print('[fetch_tools] recall_by_time failed: %s' % e, file=sys.stderr)
        return []


def recall_verbatim(brain, phrase: str = '', k: int = 10, **_) -> List[Dict[str, Any]]:
    """Verbatim phrase lookup via FTS5 — bypasses embedding similarity entirely."""
    try:
        from servers.dal import Fts5DAL
        fts = Fts5DAL(brain.conn)
        hit_ids = fts.search(phrase, limit=int(k) * 2) or []
        if not hit_ids:
            return []
        # Hydrate via brain.get_node (canonical batch path — takes a list,
        # returns {id: rich_node_dict}).
        nodes_map = brain.get_node(hit_ids[:int(k)])
        if not isinstance(nodes_map, dict):
            return []
        # Preserve FTS5 hit order so rank-based scoring is meaningful.
        ordered_nodes = [nodes_map[nid] for nid in hit_ids[:int(k)]
                          if nid in nodes_map]
        out = []
        for i, n in enumerate(ordered_nodes):
            # FTS5 doesn't rank by relevance per match in this wrapper —
            # use position as a weak prior (rank-1 highest).
            score = 1.0 - (0.4 * i / max(1, len(ordered_nodes)))
            cand = _to_candidate(n, score, 'recall_verbatim')
            if cand:
                out.append(cand)
        return out[:int(k)]
    except Exception as e:
        print('[fetch_tools] recall_verbatim failed: %s' % e, file=sys.stderr)
        return []


def recall_by_aspect(brain, aspect: str = '', recent_first: bool = True,
                     k: int = 25, **_) -> List[Dict[str, Any]]:
    """Recall by aspect — resolves aspect name → node_types via brain.aspects,
    then filters nodes by those types."""
    try:
        if not aspect:
            return []
        # Resolve aspect to node_types
        aspect_obj = brain.aspects.by_name(aspect) if hasattr(brain, 'aspects') else None
        if aspect_obj is None:
            print('[fetch_tools] recall_by_aspect: unknown aspect %r' % aspect, file=sys.stderr)
            return []
        node_types = list(aspect_obj.node_types) if hasattr(aspect_obj, 'node_types') else []
        if not node_types:
            return []  # aspect is edge-only
        sort_order = 'desc' if recent_first else 'asc'
        rows = brain.filter_nodes(field='type', include=node_types,
                                   sort_by='created_at', sort_order=sort_order,
                                   limit=int(k), rich=True)
        nodes = rows.get('nodes') if isinstance(rows, dict) else rows
        if not nodes:
            return []
        out = []
        for i, n in enumerate(nodes):
            score = 1.0 - (0.5 * i / max(1, len(nodes)))
            cand = _to_candidate(n, score, 'recall_by_aspect')
            if cand:
                out.append(cand)
        return out[:int(k)]
    except Exception as e:
        print('[fetch_tools] recall_by_aspect failed: %s' % e, file=sys.stderr)
        return []


def expand_node(brain, node_ref: str = '', hops: int = 1, **_) -> List[Dict[str, Any]]:
    """Constellation expansion from a node — wraps traverse() from
    pipeline_contract.py. Accepts node_id (≥8 char) or fuzzy title."""
    try:
        from servers.pipeline_contract import traverse
        # Resolve node_ref to a real id
        node_id = None
        if node_ref and len(node_ref) >= 8 and re.match(r'^[0-9a-f]+$', node_ref[:8]):
            # Looks like an id (full or 8-char prefix)
            n = brain.get_node(node_ref) if hasattr(brain, 'get_node') else None
            if n:
                node_id = n.get('id') if isinstance(n, dict) else None
        if not node_id and node_ref:
            # Fuzzy title lookup
            found = brain.find_node_by_title(node_ref) if hasattr(brain, 'find_node_by_title') else None
            if isinstance(found, dict):
                node_id = found.get('id')
            elif isinstance(found, list) and found:
                node_id = found[0].get('id') if isinstance(found[0], dict) else None
        if not node_id:
            return []
        traversed = traverse(brain, [node_id], depth=int(hops), limit_per_seed=20)
        out = []
        seen = set()
        for n in (traversed or []):
            nid = n.get('id') if isinstance(n, dict) else None
            if not nid or nid == node_id or nid in seen:
                continue
            seen.add(nid)
            cand = _to_candidate(n, 0.65, 'expand_node')
            if cand:
                out.append(cand)
        return out
    except Exception as e:
        print('[fetch_tools] expand_node failed: %s' % e, file=sys.stderr)
        return []


# ─── Tool dispatch ───────────────────────────────────────────────────────

_TOOL_FN_MAP = {
    'recall_topical':    recall_topical,
    'recall_recent':     recall_recent,
    'recall_by_time':    recall_by_time,
    'recall_verbatim':   recall_verbatim,
    'recall_by_aspect':  recall_by_aspect,
    'expand_node':       expand_node,
}


def execute_tool(brain, tool_name: str, tool_input: Dict[str, Any],
                 session_id: str = '') -> Dict[str, Any]:
    """Execute a single tool call. Returns {results, latency_ms, error?}.

    Every tool's output passes through one batched brain.get_node() pass —
    so _corrections, full metadata, and connections come along regardless
    of which tool fired. Tool authors only need to return ID-bearing dicts;
    they can't forget to enrich because the boundary does it.
    """
    fn = _TOOL_FN_MAP.get(tool_name)
    if fn is None:
        return {'results': [], 'latency_ms': 0,
                'error': 'unknown_tool: %s' % tool_name}
    t0 = time.time()
    kwargs = dict(tool_input or {})
    # Inject session_id for recall_recent (only tool that needs it)
    if tool_name == 'recall_recent':
        kwargs.setdefault('session_id', session_id)
    try:
        results = fn(brain, **kwargs)
    except Exception as e:
        return {'results': [], 'latency_ms': int((time.time() - t0) * 1000),
                'error': str(e)[:200]}

    # Unified enrichment — single source of "tool results are fully rich".
    # Rich fields from get_node fill missing keys; recall-specific fields
    # set by _to_candidate (score, source_tool, discovery) are preserved.
    if results and brain is not None:
        ids = [r.get('id') for r in results
               if isinstance(r, dict) and r.get('id')]
        if ids:
            try:
                rich_map = brain.get_node(ids) or {}
            except Exception as e:
                # Best-effort: if enrichment fails, fall back to whatever the
                # tool returned. Log so the failure surfaces.
                try:
                    brain._log_error(
                        'execute_tool_enrich', e,
                        'batched get_node enrichment for tool=%s' % tool_name)
                except Exception:
                    pass
                rich_map = {}
            for r in results:
                rid = r.get('id') if isinstance(r, dict) else None
                if rid and rid in rich_map:
                    for k, v in rich_map[rid].items():
                        if k not in r:
                            r[k] = v

    return {'results': results or [], 'latency_ms': int((time.time() - t0) * 1000)}


def format_tool_result_for_haiku(result: Dict[str, Any], brain=None) -> str:
    """Format a tool's output using the SAME renderer as the initial 25
    cosine candidates — `format_candidate_for_surface`. Tool results have
    already been fully enriched by `execute_tool()` (batched brain.get_node
    pass attaches _corrections, _metadata, connections), so this function
    just hands the rich shape to the formatter.

    `brain` parameter is retained for signature stability; the rich path
    no longer needs it. Thin fallback runs when results lack rich fields
    (legacy callers not going through execute_tool).
    """
    results = result.get('results') or []
    if result.get('error'):
        return 'ERROR: %s' % result['error']
    if not results:
        return 'No results.'

    from servers.scales.s1.surface_contract import format_candidate_for_surface

    lines = ['%d results (evaluate on the same merit as the initial 25):'
             % len(results)]

    for i, r in enumerate(results[:25], start=1):
        if not isinstance(r, dict):
            continue
        lines.append('')
        lines.append(format_candidate_for_surface(r, i))

    if len(results) > 25:
        lines.append('')
        lines.append('... (%d more truncated)' % (len(results) - 25))
    return '\n'.join(lines)


__all__ = [
    'TOOL_DEFINITIONS',
    'execute_tool',
    'format_tool_result_for_haiku',
    'recall_topical', 'recall_recent', 'recall_by_time',
    'recall_verbatim', 'recall_by_aspect', 'expand_node',
]
