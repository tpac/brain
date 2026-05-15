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
        "name": "recall_by_date",
        "description": (
            "Date-bounded recall. Use for specific date queries: 'on 2026-05-09', "
            "'since the start of May', 'before yesterday', 'last week's work'. Different "
            "from recall_recent (which is rolling from now) — this is anchored."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "when": {
                    "type": "string",
                    "description": (
                        "Date expression. Examples: 'yesterday', 'today', 'last week', "
                        "'this morning', 'on 2026-05-09', 'since 2026-05-01', "
                        "'before 2026-04-30'. Tool parses — you do NOT need timestamps."
                    ),
                },
                "k": {"type": "integer", "description": "Max results (default 25)", "default": 25},
            },
            "required": ["when"],
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
    """Normalize any node row/dict into the candidate shape Haiku + render expect."""
    if not isinstance(node, dict):
        return None
    nid = node.get('id') or node.get('node_id')
    if not nid:
        return None
    return {
        'id': nid,
        'title': (node.get('title') or '')[:120],
        'type': node.get('type') or '',
        'score': float(score),
        'content': node.get('content') or '',
        'kv': node.get('kv') or {},
        'source_tool': source_tool,
    }


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


def recall_by_date(brain, when: Any = '', k: int = 25, **_) -> List[Dict[str, Any]]:
    """Date-bounded recall via filter_nodes on created_at."""
    try:
        since, until = _parse_date_expr(when)
        kwargs = {
            'field': 'created_at',
            'sort_by': 'created_at',
            'sort_order': 'desc',
            'limit': int(k),
            'rich': True,
        }
        # filter_nodes API uses gt/lt (exclusive). Boundary precision at
        # second-level resolution is irrelevant for date-window recall.
        if since:
            kwargs['gt'] = since.isoformat()
        if until:
            kwargs['lt'] = until.isoformat()
        rows = brain.filter_nodes(**kwargs)
        nodes = rows.get('nodes') if isinstance(rows, dict) else rows
        if not nodes:
            return []
        out = []
        for i, n in enumerate(nodes):
            score = 1.0 - (0.5 * i / max(1, len(nodes)))
            cand = _to_candidate(n, score, 'recall_by_date')
            if cand:
                out.append(cand)
        return out[:int(k)]
    except Exception as e:
        print('[fetch_tools] recall_by_date failed: %s' % e, file=sys.stderr)
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
    'recall_by_date':    recall_by_date,
    'recall_verbatim':   recall_verbatim,
    'recall_by_aspect':  recall_by_aspect,
    'expand_node':       expand_node,
}


def execute_tool(brain, tool_name: str, tool_input: Dict[str, Any],
                 session_id: str = '') -> Dict[str, Any]:
    """Execute a single tool call. Returns {results, latency_ms, error?}."""
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
        return {'results': results or [], 'latency_ms': int((time.time() - t0) * 1000)}
    except Exception as e:
        return {'results': [], 'latency_ms': int((time.time() - t0) * 1000),
                'error': str(e)[:200]}


def format_tool_result_for_haiku(result: Dict[str, Any]) -> str:
    """Format a tool's output as a compact text block Haiku reads as tool_result.

    Lists each candidate with id/title/type/score so Haiku can select from
    them in the next round.
    """
    results = result.get('results') or []
    if result.get('error'):
        return 'ERROR: %s' % result['error']
    if not results:
        return 'No results.'
    lines = ['%d results:' % len(results)]
    for r in results[:25]:
        nid = (r.get('id') or '')[:8]
        title = (r.get('title') or '')[:100]
        typ = r.get('type') or ''
        score = r.get('score', 0.0)
        lines.append('  [%s] %.2f [%s] %s' % (nid, score, typ, title))
    if len(results) > 25:
        lines.append('  ... (%d more truncated)' % (len(results) - 25))
    return '\n'.join(lines)


__all__ = [
    'TOOL_DEFINITIONS',
    'execute_tool',
    'format_tool_result_for_haiku',
    'recall_topical', 'recall_recent', 'recall_by_date',
    'recall_verbatim', 'recall_by_aspect', 'expand_node',
]
