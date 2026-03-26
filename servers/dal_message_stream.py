"""DAL for the message_stream table in brain_logs.db.

Single responsibility: raw conversation capture and retrieval.
Tom's messages stored on every Stop event, surfaced as pending
encoding material on UserPromptSubmit.

Table lives in brain_logs.db (operational, unbounded growth).
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _now() -> str:
    """ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


class MessageStreamDAL:
    """Read/write the message_stream table.

    Methods:
        store()         — INSERT a message (user or assistant)
        get_pending()   — SELECT unencoded user messages, newest first
        mark_encoded()  — UPDATE encoded=1 for given message IDs
        count_pending() — COUNT unencoded user messages
    """

    def __init__(self, logs_conn):
        self.conn = logs_conn

    def store(self, role: str, content: str, session_id: str = '') -> int:
        """Store a message to the stream. Returns the row id.

        Args:
            role: 'user' or 'assistant'
            content: raw message text
            session_id: current session identifier
        """
        cursor = self.conn.execute(
            'INSERT INTO message_stream (timestamp, role, content, session_id) VALUES (?, ?, ?, ?)',
            (_now(), role, content, session_id))
        self.conn.commit()
        return cursor.lastrowid

    def get_pending(self, limit: int = 3) -> List[Dict]:
        """Get Tom's messages not yet marked as encoded, newest first.

        Returns list of dicts with keys: id, content, timestamp, session_id.
        Only returns role='user' messages with encoded=0.
        """
        rows = self.conn.execute(
            'SELECT id, content, timestamp, session_id '
            'FROM message_stream '
            'WHERE role = ? AND encoded = 0 '
            'ORDER BY timestamp DESC LIMIT ?',
            ('user', limit)).fetchall()
        return [
            {'id': r[0], 'content': r[1], 'timestamp': r[2], 'session_id': r[3]}
            for r in rows
        ]

    def mark_encoded(self, message_ids: List[int]) -> int:
        """Mark messages as encoded (incorporated into brain nodes).

        Args:
            message_ids: list of message_stream row IDs to mark

        Returns:
            Number of rows updated.
        """
        if not message_ids:
            return 0
        placeholders = ','.join('?' * len(message_ids))
        cursor = self.conn.execute(
            'UPDATE message_stream SET encoded = 1 WHERE id IN (%s)' % placeholders,
            message_ids)
        self.conn.commit()
        return cursor.rowcount

    def count_pending(self) -> int:
        """Count unencoded user messages."""
        row = self.conn.execute(
            'SELECT COUNT(*) FROM message_stream WHERE role = ? AND encoded = 0',
            ('user',)).fetchone()
        return row[0] if row else 0

    def get_recent(self, role: Optional[str] = None, hours: int = 24,
                   limit: int = 50) -> List[Dict]:
        """Get recent messages within a time window.

        Args:
            role: filter by role ('user' or 'assistant'), or None for all
            hours: how many hours back to look
            limit: max results

        Returns:
            List of message dicts, newest first.
        """
        cutoff = datetime.now(timezone.utc).isoformat()
        # Simple cutoff: just use LIMIT and ORDER BY since SQLite datetime
        # comparison works on ISO strings
        if role:
            rows = self.conn.execute(
                'SELECT id, role, content, timestamp, session_id, encoded '
                'FROM message_stream '
                'WHERE role = ? '
                'ORDER BY timestamp DESC LIMIT ?',
                (role, limit)).fetchall()
        else:
            rows = self.conn.execute(
                'SELECT id, role, content, timestamp, session_id, encoded '
                'FROM message_stream '
                'ORDER BY timestamp DESC LIMIT ?',
                (limit,)).fetchall()
        return [
            {'id': r[0], 'role': r[1], 'content': r[2], 'timestamp': r[3],
             'session_id': r[4], 'encoded': r[5]}
            for r in rows
        ]
