"""
brain — Data Access Layer (DAL)

Thin abstraction over SQLite tables. Each table has read/write methods.
Only this module knows which connection (brain.db vs brain_logs.db) owns which table.

Usage in brain.py:
    from servers.dal import LogsDAL, MetaDAL

    self._logs = LogsDAL(self.logs_conn)
    self._meta = MetaDAL(self.conn)

    self._logs.write_error("source", "error msg", "context")
    errors = self._logs.get_recent_errors(hours=24)

Incrementally adoptable: brain.py can migrate one table at a time.
Direct self.conn.execute() calls continue to work alongside the DAL.
"""

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional


class LogsDAL:
    """Access layer for brain_logs.db tables: debug_log, access_log, recall_log,
    miss_log, dream_log, staged_learnings."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ── debug_log ──
    # Schema: id, session_id, event_type, source, file_target,
    #         suggestions_served, procedures_served, node_ids_served,
    #         latency_ms, brain_reachable, metadata, created_at
    # Errors use event_type='error' with details in metadata JSON.

    def write_error(self, source: str, error: str, context: str = "",
                    traceback_str: str = "", session_id: str = "") -> None:
        """Write an error to the debug_log table."""
        now = datetime.now(timezone.utc).isoformat()
        metadata = json.dumps({
            'error': error[:500],
            'type': 'Exception',
            'context': context[:500],
            'traceback': traceback_str[:500] if traceback_str else '',
        })
        self.conn.execute(
            'INSERT INTO debug_log (session_id, event_type, source, metadata, created_at) '
            'VALUES (?, ?, ?, ?, ?)',
            (session_id, 'error', source, metadata, now)
        )
        self.conn.commit()

    def write_debug(self, source: str, message: str, session_id: str = "",
                    metadata: Optional[Dict] = None) -> None:
        """Write a debug entry to the debug_log table."""
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata) if metadata else json.dumps({'message': message[:500]})
        self.conn.execute(
            'INSERT INTO debug_log (session_id, event_type, source, metadata, created_at) '
            'VALUES (?, ?, ?, ?, ?)',
            (session_id, 'debug', source, meta_json, now)
        )
        self.conn.commit()

    def get_recent_errors(self, hours: int = 24, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent errors from debug_log."""
        rows = self.conn.execute(
            "SELECT source, metadata, created_at FROM debug_log "
            "WHERE event_type = 'error' AND created_at > datetime('now', '-%d hours') "
            "ORDER BY created_at DESC LIMIT ?" % hours,
            (limit,)
        ).fetchall()
        results = []
        for source, metadata, created_at in rows:
            try:
                meta = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                meta = {'error': str(metadata)}
            results.append({
                'source': source,
                'error': meta.get('error', ''),
                'type': meta.get('type', ''),
                'context': meta.get('context', ''),
                'created_at': created_at,
            })
        return results

    def get_error_count(self, hours: int = 24) -> int:
        """Count errors in the last N hours."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type = 'error' "
            "AND created_at > datetime('now', '-%d hours')" % hours
        ).fetchone()
        return row[0] if row else 0

    # ── access_log ──

    def log_access(self, node_id: str, session_id: str, query: str = "",
                   context: str = "") -> None:
        """Record a node access in the access log."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            'INSERT INTO access_log (node_id, session_id, query, context, created_at) '
            'VALUES (?, ?, ?, ?, ?)',
            (node_id, session_id, query[:500], context[:500], now)
        )
        self.conn.commit()

    def get_access_count(self, node_id: str) -> int:
        """Get total access count for a node."""
        row = self.conn.execute(
            'SELECT COUNT(*) FROM access_log WHERE node_id = ?', (node_id,)
        ).fetchone()
        return row[0] if row else 0

    def get_recent_accesses(self, hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent access log entries."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self.conn.execute(
            'SELECT node_id, session_id, query, context, created_at FROM access_log '
            'WHERE created_at > ? ORDER BY created_at DESC LIMIT ?',
            (cutoff, limit)
        ).fetchall()
        return [
            {'node_id': r[0], 'session_id': r[1], 'query': r[2],
             'context': r[3], 'created_at': r[4]}
            for r in rows
        ]

    # ── recall_log ──

    def log_recall(self, session_id: str, query: str, result_count: int,
                   result_ids: Optional[List[str]] = None, intent: str = "") -> int:
        """Record a recall event. Returns the recall_log id."""
        now = datetime.now(timezone.utc).isoformat()
        ids_json = json.dumps(result_ids) if result_ids else '[]'
        cursor = self.conn.execute(
            'INSERT INTO recall_log (session_id, query, result_count, result_ids, intent, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (session_id, query[:500], result_count, ids_json, intent, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    # ── miss_log ──

    def log_miss(self, session_id: str, signal: str, query: str = "",
                 expected_node_id: str = "", context: str = "") -> None:
        """Record a recall miss."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            'INSERT INTO miss_log (session_id, signal, query, expected_node_id, context, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (session_id, signal, query[:500], expected_node_id, context[:500], now)
        )
        self.conn.commit()

    def get_miss_trends(self, days: int = 7, limit: int = 5) -> List[Dict[str, Any]]:
        """Get queries that frequently miss."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            'SELECT query, COUNT(*) as cnt FROM miss_log '
            'WHERE created_at > ? GROUP BY query HAVING cnt >= 2 '
            'ORDER BY cnt DESC LIMIT ?',
            (cutoff, limit)
        ).fetchall()
        return [{'query': r[0], 'count': r[1]} for r in rows]

    # ── dream_log ──

    def log_dream(self, seed_id: str, connections_found: int,
                  dreams_created: int, session_id: str = "") -> None:
        """Record a dream event."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            'INSERT INTO dream_log (seed_id, connections_found, dreams_created, session_id, created_at) '
            'VALUES (?, ?, ?, ?, ?)',
            (seed_id, connections_found, dreams_created, session_id, now)
        )
        self.conn.commit()

    # ── staged_learnings ──

    def get_staged(self, status: str = "pending", limit: int = 10) -> List[Dict[str, Any]]:
        """Get staged learnings by status."""
        rows = self.conn.execute(
            'SELECT id, node_id, title, content, confidence, times_revisited, status, created_at '
            'FROM staged_learnings WHERE status = ? ORDER BY created_at DESC LIMIT ?',
            (status, limit)
        ).fetchall()
        return [
            {'id': r[0], 'node_id': r[1], 'title': r[2], 'content': r[3],
             'confidence': r[4], 'times_revisited': r[5], 'status': r[6],
             'created_at': r[7]}
            for r in rows
        ]


class MetaDAL:
    """Access layer for brain_meta table — key-value config store."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get(self, key: str, default: str = "") -> str:
        """Get a config value."""
        row = self.conn.execute(
            'SELECT value FROM brain_meta WHERE key = ?', (key,)
        ).fetchone()
        return row[0] if row else default

    def set(self, key: str, value: str) -> None:
        """Set a config value."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            'INSERT OR REPLACE INTO brain_meta (key, value, updated_at) VALUES (?, ?, ?)',
            (key, str(value), now)
        )
        self.conn.commit()

    def get_json(self, key: str, default: Any = None) -> Any:
        """Get a JSON-decoded config value."""
        raw = self.get(key, "")
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return default

    def set_json(self, key: str, value: Any) -> None:
        """Set a JSON-encoded config value."""
        self.set(key, json.dumps(value))

    def get_session_activity(self) -> Dict[str, Any]:
        """Read all session activity keys."""
        keys = ('remember_count', 'edit_check_count', 'session_id',
                'message_count', 'last_encode_at_message', 'boot_time')
        placeholders = ','.join('?' * len(keys))
        cursor = self.conn.execute(
            'SELECT key, value FROM brain_meta WHERE key IN (%s)' % placeholders,
            keys
        )
        result = {}
        for key, value in cursor.fetchall():
            if key.endswith('_count') or key == 'last_encode_at_message':
                result[key] = int(value) if value else 0
            else:
                result[key] = value
        return result

    def increment(self, key: str) -> int:
        """Increment a counter and return new value."""
        current = self.get(key, "0")
        try:
            new_val = int(current) + 1
        except (ValueError, TypeError):
            new_val = 1
        self.set(key, str(new_val))
        return new_val
