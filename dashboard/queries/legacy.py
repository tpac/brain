"""Legacy queries kept for backward compatibility.

`hook_log` and `assembler_comparison` live in `brain_dashboard.db` — the
dashboard's own sidecar DB that pre-dates the unified trace_events table.
New code should not depend on these; they remain only so the legacy routes
keep returning shaped data instead of 500s.
"""

from ..db import dashboard_db_path
from ..query import safe_query


@safe_query('queries.legacy', dashboard_db_path)
def query_hook_log(conn, since_id: int = 0, limit: int = 50):
    """DEPRECATED: read hook_log entries from brain_dashboard.db."""
    rows = conn.execute(
        "SELECT id, hook_name, timestamp, output_text, operator_text, metadata, session_id, user_prompt "
        "FROM hook_log WHERE id > ? ORDER BY id DESC LIMIT ?",
        (since_id, limit),
    ).fetchall()
    return [
        {
            "id": r[0], "hook_name": r[1], "timestamp": r[2],
            "output_text": r[3] or "", "operator_text": r[4] or "",
            "metadata": r[5] or "", "session_id": r[6] or "",
            "user_prompt": r[7] if len(r) > 7 else "",
        }
        for r in rows
    ]


@safe_query('queries.legacy', dashboard_db_path)
def query_assembler_comparison(conn, limit: int = 20):
    """DEPRECATED: assembler comparison log from brain_dashboard.db."""
    rows = conn.execute(
        "SELECT id, timestamp, user_prompt, old_chars, new_chars, new_output, stats "
        "FROM assembler_comparison ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {
            'id': r[0], 'timestamp': r[1], 'user_prompt': r[2],
            'old_chars': r[3], 'new_chars': r[4],
            'new_output': r[5], 'stats': r[6],
        }
        for r in rows
    ]
