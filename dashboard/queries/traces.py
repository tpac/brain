"""Generic trace_events query — used by the Traces tab and the S2-on-Live feed.

Surfaces identity (`human_identity` / `agent_identity`) and the S0 session
stamp (`model` / `host`) from trace metadata as top-level fields so the UI
doesn't have to re-parse JSON: every trace records who was speaking when it was
written, and S0 rows record which model produced the turn on which host.
"""

from ..clock import utc_cutoff
from ..db import logs_db_path
from ..query import safe_query
from ._meta import extract_meta_fields, S0_SESSION_STAMP_FIELDS

_PROMOTED = ('human_identity', 'agent_identity') + S0_SESSION_STAMP_FIELDS


@safe_query('queries.traces', logs_db_path)
def query_traces(conn, hours: int = 24, scale: str = '', limit: int = 500, session_id: str = ''):
    """Read trace_events from brain_logs.db, filtered by time window + scale + session.

    Order/limit: DESC keeps the NEWEST `limit` events — both UI callers
    (the Traces tab and the live S2-decode feed) re-sort within chains
    after fetching, so wire order doesn't affect display. ASC + LIMIT 200
    used to silently drop newer events in busy sessions (462+ events in
    24h), which broke navigation from node-detail source-refs: the
    target trace existed in the DB but never reached the client. Bumped
    default limit 200 → 500 to cover the long tail.
    """
    conditions = ["created_at > ?"]
    params = [utc_cutoff(hours=hours)]
    if scale:
        conditions.append('scale = ?')
        params.append(scale)
    if session_id:
        conditions.append('session_id = ?')
        params.append(session_id)
    where = ' AND '.join(conditions)
    rows = conn.execute(
        "SELECT id, chain_id, scale, event_type, ref_type, ref_id, "
        "summary, metadata, session_id, created_at, interaction_id "
        "FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?" % where,
        params + [limit],
    ).fetchall()
    out = []
    for r in rows:
        row = {
            'id': r[0], 'chain_id': r[1], 'scale': r[2],
            'event_type': r[3], 'ref_type': r[4] or '', 'ref_id': r[5] or '',
            'summary': r[6] or '', 'metadata': r[7], 'session_id': r[8] or '',
            'created_at': r[9], 'interaction_id': r[10],
        }
        # One parse per row for every promoted field.
        row.update(zip(_PROMOTED, extract_meta_fields(r[7], *_PROMOTED)))
        out.append(row)
    return out
