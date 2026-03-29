"""DAL for the message_stream table in brain_logs.db.

## Purpose
Raw conversation capture and lifecycle management.
Tom's messages are stored on every Stop event, surfaced as pending
encoding material on UserPromptSubmit, and resolved when encoded.

## Architecture
This DAL owns the full lifecycle of pending messages:
- STORE: save message + optional signal annotation
- SURFACE: return actionable messages with escalation level
- RESOLVE: mark messages when encoded/dismissed
- EXPIRE: age out messages beyond TTL (48h)

Hooks are flat dispatchers — they call these methods but contain no logic.
brain_voice.py renders what this DAL returns but contains no logic.

## Escalation Model (WHY)
Agent experiments (2026-03-26) proved 3 features drive encoding:
1. Pending messages showing the journey → encoding quality
2. Action menu per node → encoding happens (active, not passive)
3. Red alert escalation → encoding happens NOW (urgency)

Escalation levels:
- 'pending'   — surfaced ≤ 2 times. Gentle: "Tom's recent messages"
- 'attention'  — surfaced 3-4 times. Pressure: "pending encoding N+ turns"
- 'urgent'     — surfaced ≥ 5 + decision/correction signal. Red alert.
                  OR surfaced ≥ 7 regardless of signal.

## Signal Types (WHY)
Signal types stored at write time:
  'decision'    — "let's do", "ship it", "go with"
  'correction'  — "no,", "wrong", "actually,"
  'insight'     — "the reason", "because", "the key"
  'exploration' — question + long response
  None          — no signal detected

Storing signal_type at write time avoids re-detection. Signal drives
escalation urgency: decisions and corrections escalate faster (5 turns)
than explorations (7 turns).

Table lives in brain_logs.db (operational, unbounded growth).
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional


def _now() -> str:
    """ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


# Escalation thresholds
_ESCALATION_URGENT_WITH_SIGNAL = 5   # turns before urgent for decision/correction
_ESCALATION_URGENT_ANY = 7           # turns before urgent regardless of signal
_ESCALATION_ATTENTION = 3            # turns before attention level
_URGENT_SIGNAL_TYPES = frozenset({'decision', 'correction'})


class MessageStreamDAL:
    """Read/write/lifecycle for the message_stream table.

    Methods:
        store()           — INSERT a message with optional signal annotation
        get_actionable()  — SELECT pending messages with escalation level (chronological)
        mark_resolved()   — UPDATE resolved=1 for given message IDs
        expire_old()      — Resolve messages older than TTL
        count_actionable() — COUNT pending actionable messages

    Deprecated:
        get_pending()     — Use get_actionable() instead
        mark_encoded()    — Use mark_resolved() instead
        count_pending()   — Use count_actionable() instead
    """

    def __init__(self, logs_conn):
        self.conn = logs_conn

    # ── STORE ──

    def store(self, role: str, content: str, session_id: str = '',
              signal_type: Optional[str] = None) -> int:
        """Store a message to the stream. Returns the row id.

        Args:
            role: 'user' or 'assistant'
            content: raw message text
            session_id: current session identifier
            signal_type: 'decision', 'correction', 'insight', 'exploration', or None
        """
        cursor = self.conn.execute(
            'INSERT INTO message_stream '
            '(timestamp, role, content, session_id, signal_type) '
            'VALUES (?, ?, ?, ?, ?)',
            (_now(), role, content, session_id, signal_type))
        self.conn.commit()
        return cursor.lastrowid

    # ── SURFACE ──

    def get_actionable(self, limit: int = 3, max_age_hours: int = 48) -> List[Dict]:
        """Get pending messages that need attention, oldest first (chronological).

        Returns messages where:
        - role = 'user'
        - resolved = 0
        - age < max_age_hours (safety: don't surface stale context)

        Orders ASC (chronological) so the journey reads top-to-bottom:
        explore → refine → decide.

        Increments surfaced_count for each returned message so escalation
        is data-driven (DAL tracks it, not the hook).

        Returns list of dicts with keys:
            id, content, timestamp, signal_type, surfaced_count, escalation_level
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()

        rows = self.conn.execute(
            'SELECT id, content, timestamp, signal_type, surfaced_count '
            'FROM message_stream '
            'WHERE role = ? AND resolved = 0 AND timestamp > ? '
            'ORDER BY timestamp ASC LIMIT ?',
            ('user', cutoff, limit)).fetchall()

        if not rows:
            return []

        results = []
        ids_to_update = []
        for r in rows:
            msg_id, content, timestamp, signal_type, surfaced_count = r
            new_count = surfaced_count + 1
            ids_to_update.append((new_count, msg_id))

            # Compute escalation level (logic lives here, not in hooks/voice)
            if (new_count >= _ESCALATION_URGENT_WITH_SIGNAL
                    and signal_type in _URGENT_SIGNAL_TYPES):
                escalation = 'urgent'
            elif new_count >= _ESCALATION_URGENT_ANY:
                escalation = 'urgent'
            elif new_count >= _ESCALATION_ATTENTION:
                escalation = 'attention'
            else:
                escalation = 'pending'

            results.append({
                'id': msg_id,
                'content': content,
                'timestamp': timestamp,
                'signal_type': signal_type,
                'surfaced_count': new_count,
                'escalation_level': escalation,
            })

        # Increment surfaced_count in bulk
        for new_count, msg_id in ids_to_update:
            self.conn.execute(
                'UPDATE message_stream SET surfaced_count = ? WHERE id = ?',
                (new_count, msg_id))
        self.conn.commit()

        return results

    def count_actionable(self, max_age_hours: int = 48) -> int:
        """Count pending actionable messages (not resolved, within age limit)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
        row = self.conn.execute(
            'SELECT COUNT(*) FROM message_stream '
            'WHERE role = ? AND resolved = 0 AND timestamp > ?',
            ('user', cutoff)).fetchone()
        return row[0] if row else 0

    # ── RESOLVE ──

    def mark_resolved(self, message_ids: List[int],
                      reason: str = 'encoded') -> int:
        """Mark messages as resolved.

        Args:
            message_ids: list of message_stream row IDs
            reason: 'encoded' (Claude stored to brain), 'dismissed' (explicitly skipped),
                    'expired' (aged out past TTL)

        Returns:
            Number of rows updated.
        """
        if not message_ids:
            return 0
        now = _now()
        placeholders = ','.join('?' * len(message_ids))
        cursor = self.conn.execute(
            'UPDATE message_stream SET resolved = 1, resolved_at = ? '
            'WHERE id IN (%s)' % placeholders,
            [now] + message_ids)
        self.conn.commit()
        return cursor.rowcount

    # ── EXPIRE ──

    def expire_old(self, max_age_hours: int = 48) -> int:
        """Expire messages older than max_age_hours.

        Called by idle_maintenance. Messages that nobody encoded in 48h
        are stale context — expire them silently.

        Returns count of expired messages.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
        now = _now()
        cursor = self.conn.execute(
            'UPDATE message_stream SET resolved = 1, resolved_at = ? '
            'WHERE resolved = 0 AND timestamp < ?',
            (now, cutoff))
        self.conn.commit()
        return cursor.rowcount

    # ── DEPRECATED (kept for backward compat, redirect to new methods) ──

    def get_pending(self, limit: int = 3) -> List[Dict]:
        """DEPRECATED: Use get_actionable() instead.
        Returns newest-first for backward compat with existing callers."""
        return self.get_actionable(limit=limit)

    def mark_encoded(self, message_ids: List[int]) -> int:
        """DEPRECATED: Use mark_resolved() instead."""
        return self.mark_resolved(message_ids, reason='encoded')

    def count_pending(self) -> int:
        """DEPRECATED: Use count_actionable() instead."""
        return self.count_actionable()

    # ── QUERY ──

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
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        if role:
            rows = self.conn.execute(
                'SELECT id, role, content, timestamp, session_id, encoded, '
                'signal_type, surfaced_count, resolved '
                'FROM message_stream '
                'WHERE role = ? AND timestamp > ? '
                'ORDER BY timestamp DESC LIMIT ?',
                (role, cutoff, limit)).fetchall()
        else:
            rows = self.conn.execute(
                'SELECT id, role, content, timestamp, session_id, encoded, '
                'signal_type, surfaced_count, resolved '
                'FROM message_stream '
                'WHERE timestamp > ? '
                'ORDER BY timestamp DESC LIMIT ?',
                (cutoff, limit)).fetchall()
        return [
            {'id': r[0], 'role': r[1], 'content': r[2], 'timestamp': r[3],
             'session_id': r[4], 'encoded': r[5], 'signal_type': r[6],
             'surfaced_count': r[7], 'resolved': r[8]}
            for r in rows
        ]
