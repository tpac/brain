"""Signal producers — each writes signals to the queue, nothing more.

4 producers, each called once per hook_recall():
  1. produce_reminders    — due/overdue reminders
  2. produce_encoding_gap — "N minutes, nothing encoded"
  3. produce_vocabulary_gap — unmapped operator terms
  4. produce_system_health — hook errors, brain errors, rule conflicts

Producers are stateless. They query their data source, upsert into the queue
with deterministic IDs, and return. The queue handles dedup, cooldown, TTL.
"""
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def _hours_since(iso_ts):
    """Hours elapsed since an ISO timestamp. Returns 0 on parse failure."""
    if not iso_ts:
        return 0
    try:
        dt = datetime.fromisoformat(iso_ts.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            now = datetime.utcnow()
        return (now - dt).total_seconds() / 3600
    except Exception:
        return 0


# ── 1. REMINDERS ──

def produce_reminders(brain, sq_dal):
    """Surface due/overdue reminders. PREEMPT after 24h overdue."""
    try:
        reminders = brain.get_due_reminders()
        for rem in reminders[:5]:  # cap at 5
            hours_overdue = _hours_since(rem.get('due_date'))
            # PREEMPT after 24h, high priority otherwise
            preempt = hours_overdue > 24
            priority = 0.96 if preempt else 0.80
            sq_dal.enqueue(
                id="reminder:%s" % rem.get('id', '')[:40],
                producer="reminder",
                signal_type="reminder_due",
                priority=priority,
                content="🔔 %s (due %s)" % (
                    rem.get('title', 'untitled'),
                    str(rem.get('due_date', ''))[:16]),
                preempt=preempt,
                cooldown_seconds=300,  # 5 min between surfaces
                metadata=json.dumps({
                    "node_id": rem.get("id", ""),
                    "hours_overdue": round(hours_overdue, 1),
                }),
            )
    except Exception as e:
        log.warning("produce_reminders failed: %s", e)


# ── 2. ENCODING GAP ──

def produce_encoding_gap(brain, sq_dal):
    """Surface warning if session is 20+ min with zero encodes."""
    try:
        activity = brain._get_session_activity()
        remembers = int(activity.get('remember_count', 0))
        boot_time = activity.get('boot_time')
        if not boot_time:
            return

        session_min = 0
        try:
            boot_dt = datetime.fromisoformat(boot_time.replace('Z', '+00:00'))
            now_dt = datetime.now(boot_dt.tzinfo) if boot_dt.tzinfo else datetime.utcnow()
            session_min = (now_dt - boot_dt).total_seconds() / 60
        except Exception:
            return

        if session_min > 20 and remembers == 0:
            sq_dal.enqueue(
                id="encoding_gap:session",
                producer="encoding_gap",
                signal_type="encoding_gap",
                priority=0.50,
                content="📝 %d minutes in session, nothing encoded yet." % round(session_min),
                cooldown_seconds=600,  # 10 min
                max_surfaces=3,
            )
    except Exception as e:
        log.warning("produce_encoding_gap failed: %s", e)


# ── 3. VOCABULARY GAP ──

def produce_vocabulary_gap(brain, sq_dal):
    """Surface unmapped operator terms."""
    try:
        gaps_json = brain.get_config('vocabulary_gaps', '[]')
        gaps = json.loads(gaps_json) if gaps_json else []
        for gap in gaps[-3:]:
            term = gap.get('term', '') if isinstance(gap, dict) else str(gap)
            if not term:
                continue
            sq_dal.enqueue(
                id="vocab_gap:%s" % term[:30],
                producer="vocabulary_gap",
                signal_type="vocabulary_gap",
                priority=0.30,
                content='📖 Unknown term: "%s" — learn with learn_vocabulary()' % term,
                cooldown_seconds=1800,  # 30 min
                max_surfaces=2,
            )
    except Exception as e:
        log.warning("produce_vocabulary_gap failed: %s", e)


# ── 4. SYSTEM HEALTH ──

def produce_system_health(brain, sq_dal):
    """Surface hook errors, brain errors, and rule conflicts.

    Hook errors + conflicts: PREEMPT (0.96).
    Brain errors: high priority (0.90).
    """
    _produce_hook_errors(brain, sq_dal)
    _produce_brain_errors(brain, sq_dal)
    _produce_conflicts(brain, sq_dal)


def _produce_hook_errors(brain, sq_dal):
    """Hook failures from brain_logs.db hook_errors table."""
    try:
        logs_db = os.path.join(os.path.dirname(brain.db_path), "brain_logs.db")
        if not os.path.isfile(logs_db):
            return
        conn = sqlite3.connect(logs_db, timeout=10)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='hook_errors'"
        ).fetchall()
        if not tables:
            conn.close()
            return
        rows = conn.execute(
            "SELECT id, hook_name, error, created_at FROM hook_errors "
            "WHERE surfaced = 0 ORDER BY id DESC LIMIT 5"
        ).fetchall()
        if rows:
            # Mark as surfaced in hook_errors table
            ids = [r[0] for r in rows]
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                "UPDATE hook_errors SET surfaced = 1 WHERE id IN (%s)" % placeholders,
                ids)
            conn.commit()
        conn.close()

        for r in rows:
            sq_dal.enqueue(
                id="system_health:hook_error:%s" % r[0],
                producer="system_health",
                signal_type="hook_error",
                priority=0.96,
                preempt=True,
                content="⚠️ Hook error [%s]: %s" % (r[1], (r[2] or '')[:100]),
                max_surfaces=1,
            )
    except Exception as e:
        log.warning("_produce_hook_errors failed: %s", e)


def _produce_brain_errors(brain, sq_dal):
    """Silent errors inside brain.py."""
    try:
        errors = brain.get_recent_errors(hours=2, limit=3)
        seen = set()
        for err in (errors or []):
            key = "%s:%s" % (err.get('source', ''), (err.get('error', '') or '')[:30])
            if key in seen:
                continue
            seen.add(key)
            sq_dal.enqueue(
                id="system_health:brain_error:%s" % key[:60],
                producer="system_health",
                signal_type="brain_error",
                priority=0.90,
                content="⚠️ Brain error [%s]: %s" % (
                    err.get('source', '?'), (err.get('error', '') or '')[:100]),
                cooldown_seconds=600,
                max_surfaces=2,
            )
    except Exception as e:
        log.warning("_produce_brain_errors failed: %s", e)


def _produce_conflicts(brain, sq_dal):
    """Brain-Claude conflicts from conflict_log table."""
    try:
        logs_db = os.path.join(os.path.dirname(brain.db_path), "brain_logs.db")
        if not os.path.isfile(logs_db):
            return
        conn = sqlite3.connect(logs_db, timeout=10)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conflict_log'"
        ).fetchall()
        if not tables:
            conn.close()
            return
        rows = conn.execute(
            "SELECT id, rule_title, claude_action, created_at FROM conflict_log "
            "WHERE surfaced = 0 ORDER BY id DESC LIMIT 5"
        ).fetchall()
        if rows:
            ids = [r[0] for r in rows]
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                "UPDATE conflict_log SET surfaced = 1 WHERE id IN (%s)" % placeholders,
                ids)
            conn.commit()
        conn.close()

        for r in rows:
            sq_dal.enqueue(
                id="system_health:conflict:%s" % r[0],
                producer="system_health",
                signal_type="rule_conflict",
                priority=0.92,
                content="⚠️ Rule violation [%s]: %s" % (
                    (r[1] or '')[:40], (r[2] or '')[:80]),
                max_surfaces=1,
            )
    except Exception as e:
        log.warning("_produce_conflicts failed: %s", e)
