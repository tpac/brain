"""Unified errors view — pulls from every component that records errors.

Sources: brain `debug_log` and `hook_errors` (incl. DAEMON_DOWN — written by
both the hook-side detector and the MCP health monitor). Each source
contributes a row with a uniform shape so the UI can render them in one feed.
(`conflict_log` and `brain_telemetry` were removed in schema v21 — querying
them only ever warned loudly into the dashboard's own error ring, so those
dead sources are gone.)

Each row carries full `error` / `context` / `traceback` text (capped, not
list-truncated) so the Logs tab's click-to-expand can show the whole story —
the hook traceback in particular was recorded but never surfaced.

Sources are declared as a config list and one helper runs each with
loud-by-default warnings on any failure.
"""

import json

from ..clock import utc_cutoff
from ..db import logs_db_path, ro_connect
from ..log import warn


def _decode_meta(raw):
    """Parse a JSON metadata blob, return {} on failure. Inner-row silence
    is intentional — old debug_log rows often have non-JSON `metadata`."""
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {}


# Shape: a row mapper takes the SQL row tuple and returns the uniform error
# dict the UI consumes. Each source has its own mapper because column order
# differs per table, but the OUTPUT shape is identical.

# Caps: generous, not list-truncation. The list view clamps display in CSS;
# the click-to-expand wants the whole message / context / traceback.
_ERR_CAP, _CTX_CAP, _TB_CAP = 2000, 1000, 6000


def _shape_brain(r):
    meta = _decode_meta(r[3])
    return {
        'source': 'brain', 'component': r[2], 'timestamp': r[1],
        'error': (meta.get('error', r[3] or ''))[:_ERR_CAP],
        'context': (meta.get('context', '') or '')[:_CTX_CAP],
        'traceback': (meta.get('traceback', '') or '')[:_TB_CAP],
        'level': meta.get('level', 'error') or 'error',
    }


def _shape_hook(r):
    return {
        'source': 'hook', 'component': r[2], 'timestamp': r[1],
        'error': (r[4] or '')[:_ERR_CAP], 'context': (r[5] or '')[:_CTX_CAP],
        'traceback': (r[6] or '')[:_TB_CAP],
        'level': r[3] or 'error',
    }


# (db_path_fn, sql, shape_fn, label) — label is for the warn() message.
# Sources are intentionally explicit (not auto-discovered): adding a new
# error source is a deliberate one-line addition here.
_SOURCES = [
    (logs_db_path,
     "SELECT id, created_at, source, metadata FROM debug_log "
     "WHERE event_type='error' AND created_at > ? "
     "ORDER BY created_at DESC LIMIT ?",
     _shape_brain, 'brain debug_log'),
    (logs_db_path,
     "SELECT id, created_at, hook_name, level, error, context, traceback FROM hook_errors "
     "WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
     _shape_hook, 'hook_errors'),
]


def _pull_source(db_path_fn, sql, shape_fn, label, since_ts, limit):
    """Run one source's query. On any failure: warn loudly, return []."""
    try:
        with ro_connect(db_path_fn()) as conn:
            if conn is None:
                return []
            rows = conn.execute(sql, (since_ts, limit)).fetchall()
            return [shape_fn(r) for r in rows]
    except Exception as e:
        # The brain_telemetry table may not exist on fresh brains — that's
        # expected and the warn line tells you which source.
        warn('queries.errors', '%s pull failed' % label, exc=e)
        return []


def query_all_errors(limit: int = 50, hours: int = 24):
    """Aggregate errors from every component into one timestamp-sorted list."""
    since_ts = utc_cutoff(hours=hours)
    errors = []
    for db_fn, sql, shape, label in _SOURCES:
        errors.extend(_pull_source(db_fn, sql, shape, label, since_ts, limit))
    errors.sort(key=lambda e: e.get('timestamp', ''), reverse=True)
    return errors[:limit]
