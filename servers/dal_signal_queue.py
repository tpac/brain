"""DAL for the signal_queue table in brain_logs.db.

## Purpose
Priority queue for proactive brain signals. Producers enqueue signals
(reminders, tensions, encoding alerts, etc.) and the surface assembler
pulls by priority within a char budget.

## Architecture
Producers are stateless — they enqueue freely using deterministic IDs.
The DB deduplicates via INSERT OR REPLACE. The assembler reads, surfaces,
and increments counters. The queue owns all state.

## Lifecycle
  ENQUEUE → producers write signals with priority + content
  PULL    → assembler reads top-N by priority within char budget
  SURFACE → times_surfaced incremented, last_surfaced_at updated
  DISMISS → Claude or Tom explicitly silences a signal
  EXPIRE  → TTL or max_surfaces reached → auto-dismissed

Table lives in brain_logs.db (operational, unbounded growth).
"""
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


def _now() -> str:
    """ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


class SignalQueueDAL:
    """Read/write/lifecycle for the signal_queue table.

    Methods:
        enqueue()            — INSERT OR REPLACE a signal (upsert by deterministic ID)
        pull()               — Get top signals by priority within a char budget
        pull_preempt()       — Get preempt-level signals (skip recall)
        dismiss()            — Silence a specific signal
        dismiss_by_producer() — Silence all signals from a producer
        expire_stale()       — Auto-dismiss expired or over-surfaced signals
        get_queue_state()    — Full queue snapshot for dashboard
        update_priority()    — Adjust priority dynamically
    """

    def __init__(self, logs_conn: sqlite3.Connection):
        self.conn = logs_conn

    # ── ENQUEUE ──

    def enqueue(self, id: str, producer: str, signal_type: str,
                priority: float, content: str,
                ttl_seconds: Optional[int] = None,
                max_surfaces: Optional[int] = None,
                cooldown_seconds: Optional[int] = None,
                preempt: bool = False,
                metadata: Optional[str] = None) -> None:
        """Upsert a signal into the queue.

        Uses INSERT OR REPLACE — stateless producers can call this every
        turn without creating duplicates. The deterministic ID is the key.

        If the signal already exists, priority and content are updated
        but times_surfaced is preserved (carried forward).
        """
        now = _now()
        content_chars = len(content)

        # Preserve times_surfaced if signal already exists
        existing = self.conn.execute(
            'SELECT times_surfaced, last_surfaced_at FROM signal_queue WHERE id = ?',
            (id,)).fetchone()
        times_surfaced = existing[0] if existing else 0
        last_surfaced = existing[1] if existing else None

        self.conn.execute(
            'INSERT OR REPLACE INTO signal_queue '
            '(id, producer, signal_type, priority, content, content_chars, '
            'metadata, created_at, updated_at, ttl_seconds, times_surfaced, '
            'max_surfaces, last_surfaced_at, cooldown_seconds, dismissed, preempt) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)',
            (id, producer, signal_type, priority, content, content_chars,
             metadata, now if not existing else now, now,
             ttl_seconds, times_surfaced, max_surfaces,
             last_surfaced, cooldown_seconds, int(preempt)))
        self.conn.commit()

    # ── PULL ──

    def pull(self, budget_chars: int, limit: int = 10) -> List[Dict]:
        """Pull top signals by priority within a char budget.

        Filters out:
        - dismissed signals
        - expired signals (TTL exceeded)
        - over-surfaced signals (times_surfaced >= max_surfaces)
        - signals still in cooldown

        Returns signals fitting within budget_chars, ordered by priority DESC.
        Increments times_surfaced and updates last_surfaced_at for each.
        """
        now = _now()
        now_dt = datetime.now(timezone.utc)

        # Get all eligible signals ordered by priority
        rows = self.conn.execute(
            'SELECT id, producer, signal_type, priority, content, content_chars, '
            'metadata, created_at, ttl_seconds, times_surfaced, max_surfaces, '
            'last_surfaced_at, cooldown_seconds, preempt '
            'FROM signal_queue '
            'WHERE dismissed = 0 AND preempt = 0 '
            'ORDER BY priority DESC LIMIT ?',
            (limit * 3,)  # over-fetch to account for budget/cooldown filtering
        ).fetchall()

        results = []
        chars_used = 0
        ids_to_update = []

        for r in rows:
            (sid, producer, signal_type, priority, content, content_chars,
             metadata, created_at, ttl_seconds, times_surfaced, max_surfaces,
             last_surfaced_at, cooldown_seconds, preempt) = r

            # TTL check
            if ttl_seconds is not None:
                try:
                    created_dt = datetime.fromisoformat(created_at)
                    if (now_dt - created_dt).total_seconds() > ttl_seconds:
                        continue
                except (ValueError, TypeError):
                    log.warning("signal_queue: bad created_at for id=%s: %r", sid, created_at)
                    continue  # skip — can't evaluate TTL with bad timestamp

            # Max surfaces check
            if max_surfaces is not None and times_surfaced >= max_surfaces:
                continue

            # Cooldown check
            if cooldown_seconds and last_surfaced_at:
                try:
                    last_dt = datetime.fromisoformat(last_surfaced_at)
                    if (now_dt - last_dt).total_seconds() < cooldown_seconds:
                        continue
                except (ValueError, TypeError):
                    log.warning("signal_queue: bad last_surfaced_at for id=%s: %r", sid, last_surfaced_at)

            # Budget check
            if chars_used + content_chars > budget_chars:
                continue  # skip this one, try smaller ones

            chars_used += content_chars
            ids_to_update.append(sid)
            results.append({
                'id': sid,
                'producer': producer,
                'signal_type': signal_type,
                'priority': priority,
                'content': content,
                'content_chars': content_chars,
                'metadata': metadata,
                'created_at': created_at,
                'times_surfaced': times_surfaced + 1,
            })

            if len(results) >= limit:
                break

        # Update surfaced state
        for sid in ids_to_update:
            self.conn.execute(
                'UPDATE signal_queue SET times_surfaced = times_surfaced + 1, '
                'last_surfaced_at = ? WHERE id = ?',
                (now, sid))
        if ids_to_update:
            self.conn.commit()

        return results

    def pull_preempt(self) -> List[Dict]:
        """Get preempt-level signals. These skip recall entirely.

        No budget constraint — if something is preempt-level, it surfaces.
        """
        now = _now()
        rows = self.conn.execute(
            'SELECT id, producer, signal_type, priority, content, content_chars, '
            'metadata, created_at, times_surfaced '
            'FROM signal_queue '
            'WHERE dismissed = 0 AND preempt = 1 '
            'ORDER BY priority DESC',
        ).fetchall()

        results = []
        for r in rows:
            self.conn.execute(
                'UPDATE signal_queue SET times_surfaced = times_surfaced + 1, '
                'last_surfaced_at = ? WHERE id = ?',
                (now, r[0]))
            results.append({
                'id': r[0], 'producer': r[1], 'signal_type': r[2],
                'priority': r[3], 'content': r[4], 'content_chars': r[5],
                'metadata': r[6], 'created_at': r[7],
                'times_surfaced': r[8] + 1,
            })

        if results:
            self.conn.commit()
        return results

    # ── DISMISS ──

    def dismiss(self, signal_id: str) -> bool:
        """Dismiss a specific signal. Returns True if found."""
        cursor = self.conn.execute(
            'UPDATE signal_queue SET dismissed = 1, updated_at = ? WHERE id = ?',
            (_now(), signal_id))
        self.conn.commit()
        return cursor.rowcount > 0

    def dismiss_by_producer(self, producer: str) -> int:
        """Dismiss all signals from a producer. Returns count."""
        cursor = self.conn.execute(
            'UPDATE signal_queue SET dismissed = 1, updated_at = ? WHERE producer = ?',
            (_now(), producer))
        self.conn.commit()
        return cursor.rowcount

    # ── EXPIRE ──

    def expire_stale(self) -> int:
        """Auto-dismiss expired signals. Returns count dismissed.

        Dismisses signals where:
        - TTL exceeded (created_at + ttl_seconds < now)
        - Over-surfaced (times_surfaced >= max_surfaces)
        """
        now = _now()
        now_dt = datetime.now(timezone.utc)
        count = 0

        # Over-surfaced
        cursor = self.conn.execute(
            'UPDATE signal_queue SET dismissed = 1, updated_at = ? '
            'WHERE dismissed = 0 AND max_surfaces IS NOT NULL '
            'AND times_surfaced >= max_surfaces',
            (now,))
        count += cursor.rowcount

        # TTL expired — need to check each row
        rows = self.conn.execute(
            'SELECT id, created_at, ttl_seconds FROM signal_queue '
            'WHERE dismissed = 0 AND ttl_seconds IS NOT NULL'
        ).fetchall()
        expired_ids = []
        for sid, created_at, ttl in rows:
            try:
                created_dt = datetime.fromisoformat(created_at)
                if (now_dt - created_dt).total_seconds() > ttl:
                    expired_ids.append(sid)
            except (ValueError, TypeError):
                log.warning("signal_queue: bad created_at during expire for id=%s: %r", sid, created_at)
        if expired_ids:
            placeholders = ','.join('?' * len(expired_ids))
            cursor = self.conn.execute(
                'UPDATE signal_queue SET dismissed = 1, updated_at = ? '
                'WHERE id IN (%s)' % placeholders,
                [now] + expired_ids)
            count += cursor.rowcount

        if count:
            self.conn.commit()
        return count

    # ── QUERY ──

    def get_queue_state(self) -> List[Dict]:
        """Full queue snapshot for dashboard. Returns all non-dismissed signals."""
        rows = self.conn.execute(
            'SELECT id, producer, signal_type, priority, content, content_chars, '
            'metadata, created_at, updated_at, ttl_seconds, times_surfaced, '
            'max_surfaces, last_surfaced_at, cooldown_seconds, preempt '
            'FROM signal_queue '
            'WHERE dismissed = 0 '
            'ORDER BY priority DESC'
        ).fetchall()
        return [{
            'id': r[0], 'producer': r[1], 'signal_type': r[2],
            'priority': r[3], 'content': r[4], 'content_chars': r[5],
            'metadata': r[6], 'created_at': r[7], 'updated_at': r[8],
            'ttl_seconds': r[9], 'times_surfaced': r[10],
            'max_surfaces': r[11], 'last_surfaced_at': r[12],
            'cooldown_seconds': r[13], 'preempt': bool(r[14]),
        } for r in rows]

    def update_priority(self, signal_id: str, new_priority: float) -> bool:
        """Adjust priority dynamically. Returns True if found."""
        cursor = self.conn.execute(
            'UPDATE signal_queue SET priority = ?, updated_at = ? WHERE id = ? AND dismissed = 0',
            (new_priority, _now(), signal_id))
        self.conn.commit()
        return cursor.rowcount > 0
