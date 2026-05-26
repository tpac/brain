"""Generic trace_events query — used by the Traces tab and the S2-on-Live feed.

Surfaces identity (`human_identity` / `agent_identity`) from trace metadata as
top-level fields so the UI doesn't have to re-parse JSON. Identity stamping
landed with the trace identity migration (75075eb / 65bf483 / 5cff407): every
trace now records who was speaking when it was written, and the dashboard had
no view of it. Now it does.
"""

from ..clock import utc_cutoff
from ..db import logs_db_path
from ..query import safe_query
from ._meta import extract_identity


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
        "summary, metadata, session_id, created_at "
        "FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?" % where,
        params + [limit],
    ).fetchall()
    out = []
    for r in rows:
        hi, ai = extract_identity(r[7])
        out.append({
            'id': r[0], 'chain_id': r[1], 'scale': r[2],
            'event_type': r[3], 'ref_type': r[4] or '', 'ref_id': r[5] or '',
            'summary': r[6] or '', 'metadata': r[7], 'session_id': r[8] or '',
            'created_at': r[9],
            'human_identity': hi, 'agent_identity': ai,
        })
    return out
