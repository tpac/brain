"""Recent sessions — populates the session-filter dropdowns."""

from ..clock import utc_cutoff
from ..db import logs_db_path
from ..query import safe_query


@safe_query('queries.sessions', logs_db_path)
def query_recent_sessions(conn, limit: int = 20, days: int = 7):
    """Return recent sessions from trace_events, newest first."""
    rows = conn.execute(
        "SELECT DISTINCT session_id, MIN(created_at) as first_seen, "
        "MAX(created_at) as last_seen, COUNT(*) as event_count "
        "FROM trace_events WHERE session_id != '' "
        "AND created_at > ? "
        "GROUP BY session_id ORDER BY last_seen DESC LIMIT ?",
        (utc_cutoff(days=days), limit),
    ).fetchall()
    return [
        {"id": r[0], "short": r[0][:8], "first": r[1], "last": r[2], "events": r[3]}
        for r in rows
    ]
