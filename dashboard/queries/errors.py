"""Unified errors view — pulls from every component that records errors.

Sources: brain `debug_log`, `hook_errors`, `conflict_log`, dashboard
`hook_log` (legacy DAEMON_DOWN events), and `brain_telemetry`. Each source
contributes a row with a uniform shape so the UI can render them in one feed.

Previously this file had five copies of the same
``with ro_connect: if conn: try: ... except: pass`` block — silent failures
all the way down. Now sources are declared as a config list and one helper
runs each through @safe_query-equivalent boilerplate, with loud-by-default
warnings on any failure.
"""

import json

from ..clock import utc_cutoff
from ..db import dashboard_db_path, logs_db_path, ro_connect
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

def _shape_brain(r):
    meta = _decode_meta(r[3])
    return {
        'source': 'brain', 'component': r[2], 'timestamp': r[1],
        'error': (meta.get('error', r[3] or ''))[:200],
        'context': (meta.get('context', '') or '')[:100],
        'level': 'error',
    }


def _shape_hook(r):
    return {
        'source': 'hook', 'component': r[2], 'timestamp': r[1],
        'error': (r[4] or '')[:200], 'context': (r[5] or '')[:100],
        'level': r[3] or 'error',
    }


def _shape_conflict(r):
    return {
        'source': 'conflict', 'component': r[2], 'timestamp': r[1],
        'error': 'Rule: %s — Decision: %s' % (r[3] or '?', r[4] or '?'),
        'context': 'Resolution: %s' % (r[5] or 'pending'),
        'level': 'warning',
    }


def _shape_daemon_down(r):
    return {
        'source': 'daemon', 'component': 'daemon_down', 'timestamp': r[1],
        'error': (r[2] or '')[:200], 'context': '',
        'level': 'critical',
    }


def _shape_telemetry(r):
    return {
        'source': 'telemetry', 'component': r[2], 'timestamp': r[1],
        'error': (r[3] or '')[:200], 'context': '',
        'level': 'warning',
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
     "SELECT id, created_at, hook_name, level, error, context FROM hook_errors "
     "WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
     _shape_hook, 'hook_errors'),
    (logs_db_path,
     "SELECT id, created_at, hook_name, rule_title, brain_decision, resolution "
     "FROM conflict_log WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
     _shape_conflict, 'conflict_log'),
    (dashboard_db_path,
     "SELECT id, timestamp, output_text FROM hook_log "
     "WHERE hook_name='DAEMON_DOWN' AND timestamp > ? "
     "ORDER BY id DESC LIMIT ?",
     _shape_daemon_down, 'dashboard hook_log (legacy)'),
    (logs_db_path,
     "SELECT id, timestamp, operation, error_message FROM brain_telemetry "
     "WHERE success=0 AND timestamp > ? "
     "ORDER BY timestamp DESC LIMIT ?",
     _shape_telemetry, 'brain_telemetry'),
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
        # Legacy tables (dashboard/hook_log, brain_telemetry) may not exist on
        # fresh brains — that's expected and the warn line tells you which one.
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
