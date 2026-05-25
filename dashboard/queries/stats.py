"""Top-bar stats + insights — high-level summaries."""

import json
import os

from ..clock import utc_cutoff
from ..daemon_client import DAEMON_PORT, daemon_alive
from ..db import brain_db_path, direct_query, logs_db_path
from ..log import warn


def query_stats():
    """Top-bar counters: nodes, edges, locked, last 24h, orphans, type histogram,
    daemon status, encoding position within the 5-stop window."""
    db = brain_db_path()
    nodes = direct_query("SELECT COUNT(*) FROM nodes WHERE archived = 0", db_path=db)
    edges = direct_query("SELECT COUNT(*) FROM edges", db_path=db)
    locked = direct_query("SELECT COUNT(*) FROM nodes WHERE locked = 1", db_path=db)
    types = direct_query(
        "SELECT type, COUNT(*) FROM nodes WHERE archived = 0 GROUP BY type ORDER BY COUNT(*) DESC",
        db_path=db,
    )
    recent = direct_query(
        "SELECT COUNT(*) FROM nodes WHERE created_at > ?",
        args=(utc_cutoff(hours=24),), db_path=db,
    )
    orphans = direct_query(
        "SELECT COUNT(*) FROM nodes n WHERE archived = 0 "
        "AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.source_id = n.id OR e.target_id = n.id)",
        db_path=db,
    )

    enc_counter = 0
    enc_position = 0
    try:
        enc_row = direct_query(
            "SELECT session_id, value FROM session_state "
            "WHERE key = '_session_context' ORDER BY updated_at DESC LIMIT 1",
            db_path=logs_db_path(),
        )
        if enc_row and enc_row[0][1]:
            state = json.loads(enc_row[0][1])
            enc_counter = state.get('stop_counter', 0)
            enc_position = enc_counter % 5
    except Exception as e:
        warn('queries.stats', 'reading session_context for encode position failed', exc=e)

    return {
        "nodes": nodes[0][0] if nodes else 0,
        "edges": edges[0][0] if edges else 0,
        "locked": locked[0][0] if locked else 0,
        "recent_24h": recent[0][0] if recent else 0,
        "orphans": orphans[0][0] if orphans else 0,
        "types": {t: cnt for t, cnt in types},
        "daemon": "alive" if daemon_alive() else "unavailable",
        "encoding": {
            "counter": enc_counter,
            "position": enc_position,
            "next_in": 5 - enc_position if enc_position else 0,
        },
    }


def query_insights():
    """Anchor-facing diagnostic insights — orphan-locked, thin nodes, zero quotes,
    trace coverage, daemon liveness."""
    db = brain_db_path()
    insights = []

    orphan_locked = direct_query(
        "SELECT id, title, type, created_at FROM nodes "
        "WHERE locked = 1 AND archived = 0 "
        "AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.source_id = nodes.id OR e.target_id = nodes.id)",
        db_path=db,
    )
    if orphan_locked:
        insights.append({
            "severity": "high", "icon": "\U0001f512",
            "title": "%d locked nodes are orphaned" % len(orphan_locked),
            "detail": "Important memories disconnected from everything. Recall can't find them through graph traversal.",
            "nodes": [{"id": r[0], "title": r[1], "type": r[2]} for r in orphan_locked],
        })

    thin = direct_query(
        "SELECT COUNT(*), AVG(LENGTH(content)) FROM nodes "
        "WHERE archived = 0 AND LENGTH(content) < 100 "
        "AND created_at > ?",
        args=(utc_cutoff(days=7),), db_path=db,
    )
    if thin and thin[0][0] > 5:
        insights.append({
            "severity": "medium", "icon": "\U0001f4cf",
            "title": "%d thin nodes this week (avg %d chars)" % (thin[0][0], thin[0][1] or 0),
            "detail": "Nodes under 100 chars lack context for future recall.",
        })

    try:
        s1_traces = direct_query(
            "SELECT COUNT(*) FROM trace_events WHERE scale = 's1' "
            "AND created_at > ?",
            args=(utc_cutoff(hours=24),), db_path=logs_db_path(),
        )
        s1_count = s1_traces[0][0] if s1_traces else 0
        if s1_count == 0:
            insights.append({
                "severity": "high", "icon": "\U0001f4ca",
                "title": "No S1 traces in 24h",
                "detail": "No recall or encoding traces. Check daemon and hook pipeline.",
            })
    except Exception as e:
        warn('queries.stats', 'S1 trace coverage insight failed', exc=e)

    _7d = utc_cutoff(days=7)
    quotes = direct_query(
        "SELECT COUNT(*) FROM nodes WHERE archived = 0 "
        "AND created_at > ? "
        "AND (content LIKE '%Tom said%' OR content LIKE '%Tom:%' "
        "     OR content LIKE '%Claude:%' OR title LIKE '%quote%')",
        args=(_7d,), db_path=db,
    )
    types = dict(direct_query(
        "SELECT type, COUNT(*) FROM nodes WHERE archived = 0 AND created_at > ? GROUP BY type",
        args=(_7d,), db_path=db,
    ))
    total_recent = sum(types.values())
    if quotes and quotes[0][0] == 0 and total_recent > 5:
        insights.append({
            "severity": "high", "icon": "\U0001f4ad",
            "title": "Zero quotes preserved this week",
            "detail": "Tom's exact words and Claude's own insights weren't captured.",
        })

    if not daemon_alive():
        insights.insert(0, {
            "severity": "high", "icon": "⚠️",
            "title": "Daemon is not running",
            "detail": "Brain daemon on port %d is not responding. Dashboard is showing read-only data from SQLite directly. Live features (recall, encoding) are unavailable." % DAEMON_PORT,
        })

    return insights
