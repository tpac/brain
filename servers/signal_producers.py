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
    # _produce_conflicts removed — conflict_log table dropped


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
                priority=0.80,
                preempt=False,  # Hook errors are NOT critical enough to block recall
                content="⚠️ Hook error [%s]: %s" % (r[1], (r[2] or '')[:100]),
                max_surfaces=1,
                ttl_seconds=3600,  # Expire after 1 hour
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


# _produce_conflicts REMOVED 2026-04-05 — conflict_log table dropped


# ── 5. SYSTEM INTEGRITY (lightweight, runs every recall) ──

# Types the system actively queries by name — these get special treatment
STRUCTURAL_TYPES = {
    'vocabulary', 'rule', 'decision', 'mechanism', 'lesson', 'impact',
    'convention', 'pattern', 'constraint', 'correction', 'purpose', 'tension',
}

def produce_integrity(brain, sq_dal):
    """Lightweight integrity checks — runs on every recall.

    Surfaces:
    - Duplicate clusters (same title prefix appearing 3+ times)
    - Emergent types (non-structural types with 10+ nodes)
    - Revision drought (0 revisions ever)
    """
    try:
        _check_duplicates(brain, sq_dal)
        _check_emergent_types(brain, sq_dal)
        _check_revision_drought(brain, sq_dal)
    except Exception as e:
        log.warning("produce_integrity failed: %s", e)


def _check_duplicates(brain, sq_dal):
    """Flag title clusters that suggest duplication."""
    try:
        rows = brain.conn.execute("""
            SELECT SUBSTR(title, 1, 35) as prefix, COUNT(*) as cnt
            FROM nodes WHERE archived=0
            GROUP BY prefix HAVING cnt >= 3
            ORDER BY cnt DESC LIMIT 3
        """).fetchall()
        for r in rows:
            sq_dal.enqueue(
                id="integrity:dupe:%s" % r[0][:30].replace(" ", "_"),
                producer="integrity",
                signal_type="duplicate_cluster",
                priority=0.45,
                content="🔄 Duplicate cluster: \"%s...\" × %d nodes — review and consolidate" % (r[0], r[1]),
                cooldown_seconds=3600,  # 1h — not urgent
                max_surfaces=2,
            )
    except Exception:
        pass


def _check_emergent_types(brain, sq_dal):
    """Flag non-structural types that have accumulated enough to consider promoting."""
    try:
        rows = brain.conn.execute("""
            SELECT type, COUNT(*) as cnt FROM nodes
            WHERE archived=0 GROUP BY type HAVING cnt >= 10
            ORDER BY cnt DESC
        """).fetchall()
        for r in rows:
            if r[0] not in STRUCTURAL_TYPES:
                sq_dal.enqueue(
                    id="integrity:emergent_type:%s" % r[0],
                    producer="integrity",
                    signal_type="emergent_type",
                    priority=0.40,
                    content="🌱 Type \"%s\" has %d nodes but no system behavior — promote to structural?" % (r[0], r[1]),
                    cooldown_seconds=7200,  # 2h
                    max_surfaces=1,
                )
    except Exception:
        pass


def _check_revision_drought(brain, sq_dal):
    """Flag if no node has ever been revised — the revision mechanism isn't working."""
    try:
        revised = brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE archived=0 AND revised_at IS NOT NULL"
        ).fetchone()[0]
        total = brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE archived=0"
        ).fetchone()[0]
        if total > 50 and revised == 0:
            sq_dal.enqueue(
                id="integrity:revision_drought",
                producer="integrity",
                signal_type="revision_drought",
                priority=0.55,
                content="⚠️ 0 of %d nodes ever revised — stale information accumulates without revision" % total,
                cooldown_seconds=7200,
                max_surfaces=2,
            )
        elif total > 100:
            pct = revised / total * 100
            if pct < 5:
                sq_dal.enqueue(
                    id="integrity:low_revision",
                    producer="integrity",
                    signal_type="low_revision",
                    priority=0.35,
                    content="📊 Only %d of %d nodes (%.0f%%) ever revised — most knowledge is first-draft" % (revised, total, pct),
                    cooldown_seconds=14400,  # 4h
                    max_surfaces=1,
                )
    except Exception:
        pass


# ── 6. DEEP INTEGRITY AUDIT (runs during idle maintenance) ──

def deep_integrity_audit(brain):
    """Full brain health audit — runs during idle maintenance.

    Returns a list of findings, each: {type, severity, message, details}
    """
    findings = []

    try:
        # 1. Duplicate detection — find nodes with very similar titles
        rows = brain.conn.execute("""
            SELECT SUBSTR(title, 1, 35) as prefix, COUNT(*) as cnt,
                   GROUP_CONCAT(id, ',') as ids
            FROM nodes WHERE archived=0
            GROUP BY prefix HAVING cnt >= 2
            ORDER BY cnt DESC LIMIT 20
        """).fetchall()
        for r in rows:
            findings.append({
                "type": "duplicate_cluster",
                "severity": "medium" if r[1] >= 4 else "low",
                "message": "\"%s...\" × %d nodes" % (r[0], r[1]),
                "node_ids": r[2].split(",")[:5],
            })

        # 2. Emergent types
        rows = brain.conn.execute("""
            SELECT type, COUNT(*) as cnt FROM nodes
            WHERE archived=0 GROUP BY type ORDER BY cnt DESC
        """).fetchall()
        for r in rows:
            if r[0] not in STRUCTURAL_TYPES and r[1] >= 5:
                findings.append({
                    "type": "emergent_type",
                    "severity": "info" if r[1] < 10 else "medium",
                    "message": "Type \"%s\" has %d nodes — no system behavior defined" % (r[0], r[1]),
                    "count": r[1],
                })

        # 3. Cold zones — nodes not accessed in 14+ days
        cold = brain.conn.execute("""
            SELECT COUNT(*) FROM nodes WHERE archived=0
            AND (last_accessed IS NULL OR last_accessed < datetime('now', '-14 days'))
        """).fetchone()[0]
        total = brain.conn.execute("SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
        if total > 0:
            cold_pct = cold / total * 100
            if cold_pct > 30:
                findings.append({
                    "type": "cold_zone",
                    "severity": "medium",
                    "message": "%d of %d nodes (%.0f%%) not accessed in 14+ days" % (cold, total, cold_pct),
                })

        # 4. Isolated nodes (no edges)
        isolated = brain.conn.execute("""
            SELECT COUNT(*) FROM nodes n WHERE n.archived=0
            AND NOT EXISTS (SELECT 1 FROM edges WHERE source_id=n.id OR target_id=n.id)
        """).fetchone()[0]
        if isolated > 5:
            findings.append({
                "type": "isolated_nodes",
                "severity": "medium",
                "message": "%d nodes with zero connections — invisible to graph traversal" % isolated,
            })

        # 5. Revision stats
        revised = brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE archived=0 AND revised_at IS NOT NULL"
        ).fetchone()[0]
        findings.append({
            "type": "revision_stats",
            "severity": "info",
            "message": "%d of %d nodes revised (%.0f%%)" % (revised, total, revised / max(1, total) * 100),
        })

        # 6. Edge type distribution — are co_accessed dominating?
        edge_types = brain.conn.execute("""
            SELECT relation, COUNT(*) as cnt FROM edges GROUP BY relation ORDER BY cnt DESC LIMIT 5
        """).fetchall()
        total_edges = sum(r[1] for r in edge_types)
        for r in edge_types:
            pct = r[1] / max(1, total_edges) * 100
            if r[0] == 'co_accessed' and pct > 70:
                findings.append({
                    "type": "edge_imbalance",
                    "severity": "info",
                    "message": "co_accessed edges are %.0f%% of all edges — organic but noisy" % pct,
                })

        # 7. Metadata sparseness (via KV DAL)
        from .dal_metadata import MetadataDAL
        _meta_dal = MetadataDAL(brain.conn)
        meta_total = _meta_dal.total_nodes()
        if meta_total > 0:
            for field in ['reasoning', 'user_raw_quote', 'source_context']:
                filled = _meta_dal.nodes_with_field(field)
                pct = filled / meta_total * 100
                if pct < 30:
                    findings.append({
                        "type": "sparse_metadata",
                        "severity": "low",
                        "message": "metadata.%s: only %.0f%% filled (%d of %d)" % (field, pct, filled, meta_total),
                    })
        else:
            findings.append({
                "type": "no_metadata",
                "severity": "medium",
                "message": "No metadata in KV store — no reasoning, quotes, or corrections tracked",
            })

    except Exception as e:
        findings.append({"type": "audit_error", "severity": "high", "message": str(e)})

    return findings
