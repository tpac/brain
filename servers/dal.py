"""
brain — Data Access Layer (DAL)

Thin abstraction over SQLite tables. Each table has read/write methods.
Only this module knows which connection (brain.db vs brain_logs.db) owns which table.

Usage in brain.py:
    from servers.dal import LogsDAL, BrainMetaDAL

    self._logs = LogsDAL(self.logs_conn)
    self._meta = BrainMetaDAL(self.conn)

    self._logs.write_error("source", "error msg", "context")
    errors = self._logs.get_recent_errors(hours=24)

Incrementally adoptable: brain.py can migrate one table at a time.
Direct self.conn.execute() calls continue to work alongside the DAL.
"""

import json
import secrets
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from .clock import iso_cutoff, iso_now
from .db_backends.sqlite import commit_unless_batched


def _new_trace_id(conn) -> str:
    """Generate a fresh 8-char hex trace id, retrying on the vanishingly rare
    collision with an existing id. Matches node id shape (also 8-char hex via
    secrets.token_hex(4)). Collision space is 4 billion vs ~60K existing rows
    — first-try success is ~99.9985%."""
    for _ in range(5):
        candidate = secrets.token_hex(4)
        row = conn.execute(
            'SELECT 1 FROM trace_events WHERE id = ? LIMIT 1',
            (candidate,)
        ).fetchone()
        if row is None:
            return candidate
    raise RuntimeError("_new_trace_id: 5 consecutive collisions — investigate")


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

    def write_event(self, event_type: str, source: str,
                    metadata: Dict[str, Any], session_id: str = "") -> None:
        """Single INSERT path for debug_log rows (error / warning / debug).

        Owns the SQL + the batch-aware commit gate; the caller owns the
        policy and builds `metadata`. Brain._log_event (and the typed
        delegators below) route here so there is exactly ONE debug_log
        writer — a per-type footgun can't silently skip the INSERT, and
        every log row commits immediately (durable + visible to the
        dashboard's separate connection) unless inside a batch.
        """
        self.conn.execute(
            'INSERT INTO debug_log (session_id, event_type, source, metadata, created_at) '
            'VALUES (?, ?, ?, ?, ?)',
            (session_id, event_type, source, json.dumps(metadata), iso_now())
        )
        commit_unless_batched(self.conn)

    def write_error(self, source: str, error: str, context: str = "",
                    traceback_str: str = "", session_id: str = "") -> None:
        """Write an error to the debug_log table."""
        self.write_event('error', source, {
            'error': error[:500],
            'type': 'Exception',
            'context': context[:500],
            'traceback': traceback_str[:500] if traceback_str else '',
        }, session_id=session_id)

    def write_debug(self, source: str, message: str, session_id: str = "",
                    metadata: Optional[Dict] = None) -> None:
        """Write a debug entry to the debug_log table."""
        self.write_event('debug', source,
                         metadata if metadata else {'message': message[:500]},
                         session_id=session_id)

    # ── hook_errors ──
    # The daemon-independent error table the dashboard + boot read. In-process
    # callers (e.g. the MCP health monitor) route here so the hook_errors write
    # lives in the DAL, not raw in the MCP layer. The table is canonically
    # defined in schema.py (LOG_TABLES['hook_errors']) — referenced here
    # defensively, never re-declared, so the schema can't drift.
    def log_hook_error(self, hook_name: str, error: str, context: str = "",
                       level: str = "error", traceback_str: str = "") -> None:
        """Append a hook_errors row (creating the table from the canonical schema
        if absent) and prune to the most recent 200."""
        from servers.schema import LOG_TABLES
        self.conn.execute(LOG_TABLES['hook_errors']['create'])
        self.conn.execute(
            "INSERT INTO hook_errors (created_at, hook_name, level, error, context, traceback) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (iso_now(), hook_name, level, str(error), context[:500],
             (traceback_str or "")[:2000]))
        self.conn.execute(
            "DELETE FROM hook_errors WHERE id NOT IN "
            "(SELECT id FROM hook_errors ORDER BY id DESC LIMIT 200)")
        commit_unless_batched(self.conn)

    def get_recent_errors(self, hours: int = 24, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent errors from debug_log."""
        rows = self.conn.execute(
            "SELECT source, metadata, created_at FROM debug_log "
            "WHERE event_type = 'error' AND created_at > ? "
            "ORDER BY created_at DESC LIMIT ?",
            (iso_cutoff(hours=hours), limit)
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
            "AND created_at > ?",
            (iso_cutoff(hours=hours),)
        ).fetchone()
        return row[0] if row else 0

    # ── access_log — REMOVED 2026-04-05 ──
    # Table dropped. 415K rows, 151K/day writes, never used for anything meaningful.
    # Node access_count on nodes table is the durable stat.

    # ── recall_log — REMOVED 2026-04-05 ──
    # All recall_log write methods deleted. Traces (trace_events) are source of truth.
    # Table still exists with historical data. Dashboard reads from traces.

    # ── miss_log — REMOVED 2026-04-05 (table dropped) ──

    # ── dream_log — REMOVED 2026-04-22 (dream system deleted, table kept for
    # historical queries but no live writes). The INSERT helper was the last
    # writer and had no live callers. DROP TABLE is a schema migration task
    # not done here. ──

    # ── DB maintenance ──

    def run_maintenance(self, graph_conn: sqlite3.Connection) -> Dict[str, Any]:
        """Run DB maintenance: prune old logs, clean orphans, reindex.

        Retention policy (Option C — significance-based):
          - access_log: 30 days (just counters, node.access_count is the durable stat)
          - debug_log errors: keep forever (errors are always signal)
          - debug_log telemetry: 30 days
          - recall_log evaluated: keep forever (precision data is gold)
          - recall_log unevaluated: 30 days
          - suggest_log: 30 days
          - health_log: 90 days
          - dream_log: keep forever (small, useful for trends)

        Also cleans orphaned graph data (vectors, edges, embeddings for deleted nodes).
        """
        stats = {}

        # --- Logs DB retention ---
        # access_log: REMOVED 2026-04-05 (table dropped)
        # recall_log: REMOVED 2026-04-05 (table dropped)

        # debug_log: keep errors forever, prune telemetry/other after 30 days
        cur = self.conn.execute(
            "DELETE FROM debug_log WHERE event_type != 'error' "
            "AND created_at < ?",
            (iso_cutoff(days=30),))
        stats['debug_log_pruned'] = cur.rowcount

        # suggest_log: REMOVED 2026-04-05 (table dropped)

        # health_log: REMOVED 2026-04-05 (table dropped)

        # hook_errors: 30 days (surfaced ones only — unsurfaced kept until shown)
        try:
            cur = self.conn.execute(
                "DELETE FROM hook_errors WHERE surfaced = 1 "
                "AND created_at < ?",
                (iso_cutoff(days=30),))
            stats['hook_errors_pruned'] = cur.rowcount
        except Exception:
            stats['hook_errors_pruned'] = 0

        commit_unless_batched(self.conn)

        # --- Graph DB orphan cleanup ---
        if graph_conn:
            cur = graph_conn.execute(
                "DELETE FROM node_vectors WHERE node_id NOT IN (SELECT id FROM nodes)")
            stats['orphaned_vectors'] = cur.rowcount

            # edge_relations FIRST, scoped to exactly the edges this pass is
            # about to delete. Deleting edges alone strands their relation
            # rows as permanent orphans — invisible to every JOIN-based read
            # but polluting counts and blocking recovery (2026-06-12 audit:
            # 17,982 stranded rows, 84% of active relations, accumulated by
            # this one-sided delete). Scoped IN (...) — NOT a blanket
            # "NOT IN (SELECT edge_id FROM edges)" — so pre-existing orphans
            # stay untouched for the trace-based recovery effort.
            cur = graph_conn.execute(
                "DELETE FROM edge_relations WHERE edge_id IN ("
                "SELECT edge_id FROM edges "
                "WHERE source_id NOT IN (SELECT id FROM nodes) "
                "OR target_id NOT IN (SELECT id FROM nodes))")
            stats['orphaned_edge_relations'] = cur.rowcount

            cur = graph_conn.execute(
                "DELETE FROM edges WHERE source_id NOT IN (SELECT id FROM nodes) "
                "OR target_id NOT IN (SELECT id FROM nodes)")
            stats['orphaned_edges'] = cur.rowcount

            cur = graph_conn.execute(
                "DELETE FROM node_enrichments WHERE node_id NOT IN (SELECT id FROM nodes)")
            stats['orphaned_enrichments'] = cur.rowcount

            # node_metadata orphan cleanup removed 2026-04-13 — table dropped.

            cur = graph_conn.execute(
                "DELETE FROM doc_freq WHERE term NOT IN (SELECT DISTINCT term FROM node_vectors)")
            stats['orphaned_terms'] = cur.rowcount

            graph_conn.commit()

        # Summarize
        total_pruned = sum(v for k, v in stats.items() if 'pruned' in k)
        total_orphans = sum(v for k, v in stats.items() if 'orphaned' in k)
        stats['total_pruned'] = total_pruned
        stats['total_orphans'] = total_orphans

        return stats

    # ── recall_log precision lifecycle — REMOVED 2026-04-05 ──
    # All methods deleted: insert_recall_log, update_recall_judge,
    # update_recall_response, update_recall_evaluation, update_recall_feedback,
    # get_recall_row, get_pending_response, get_pending_followups.
    # brain_precision.py (the only caller) was deleted. Traces are source of truth.

    # ── staged_learnings — REMOVED 2026-04-05 (table dropped) ──
    # ── recall_gaps — REMOVED 2026-04-05 (table dropped) ──
    # ── pending_consolidation — REMOVED 2026-04-05 (table dropped) ──


    def query_logs(self, source: str = 'all', hours: int = 24,
                   level: str = 'all', hook_name: str = '',
                   limit: int = 50) -> Dict[str, Any]:
        """Unified log query across hook_errors and debug_log.

        Args:
            source: 'errors' (hook_errors), 'debug' (debug_log), or 'all'
                    (merged, sorted by time).
            hours: look back window (default 24).
            level: filter by level — 'error', 'critical', 'warning', or 'all'.
            hook_name: filter hook_errors by hook name (e.g. 'hook_recall').
            limit: max results per source (capped at 200).

        Returns: dict with 'entries' list and 'counts' summary.
        """
        limit = min(max(limit, 1), 200)
        cutoff = iso_cutoff(hours=hours)
        entries = []
        counts = {}

        # ── hook_errors ──
        if source in ('errors', 'all'):
            try:
                conditions = ['created_at > ?']
                params = [cutoff]
                if level != 'all':
                    conditions.append('level = ?')
                    params.append(level)
                if hook_name:
                    conditions.append('hook_name = ?')
                    params.append(hook_name)
                where = ' AND '.join(conditions)
                rows = self.conn.execute(
                    'SELECT id, hook_name, level, error, context, created_at '
                    'FROM hook_errors WHERE %s ORDER BY created_at DESC LIMIT ?' % where,
                    params + [limit]
                ).fetchall()
                count_row = self.conn.execute(
                    'SELECT COUNT(*) FROM hook_errors WHERE %s' % where, params
                ).fetchone()
                counts['hook_errors'] = count_row[0] if count_row else 0
                for r in rows:
                    entries.append({
                        'source': 'hook_errors', 'id': r[0], 'hook_name': r[1],
                        'level': r[2], 'message': r[3], 'context': r[4] or '',
                        'created_at': r[5]
                    })
            except Exception:
                counts['hook_errors'] = 0

        # ── debug_log ──
        if source in ('debug', 'all'):
            try:
                conditions = ['created_at > ?']
                params = [cutoff]
                if level == 'error':
                    conditions.append("event_type = 'error'")
                elif level != 'all':
                    conditions.append('event_type = ?')
                    params.append(level)
                where = ' AND '.join(conditions)
                rows = self.conn.execute(
                    'SELECT id, source, event_type, metadata, created_at '
                    'FROM debug_log WHERE %s ORDER BY created_at DESC LIMIT ?' % where,
                    params + [limit]
                ).fetchall()
                count_row = self.conn.execute(
                    'SELECT COUNT(*) FROM debug_log WHERE %s' % where, params
                ).fetchone()
                counts['debug_log'] = count_row[0] if count_row else 0
                for r in rows:
                    meta = {}
                    try:
                        meta = json.loads(r[3]) if r[3] else {}
                    except (json.JSONDecodeError, TypeError):
                        meta = {'raw': str(r[3])[:200]}
                    entries.append({
                        'source': 'debug_log', 'id': r[0], 'origin': r[1] or '',
                        'level': r[2], 'message': meta.get('error', meta.get('message', str(meta)[:200])),
                        'context': meta.get('context', ''),
                        'created_at': r[4]
                    })
            except Exception:
                counts['debug_log'] = 0

        # Sort merged entries by time descending
        entries.sort(key=lambda e: e.get('created_at', ''), reverse=True)
        if source == 'all':
            entries = entries[:limit]

        return {'entries': entries, 'counts': counts}


class InteractionDAL:
    """Access layer for interactions — versioned templates for system boundaries.

    Every learnable boundary (surface prompt, encoding prompt, voice format,
    signal assembly) is an interaction. Versioned, traceable, optimizable
    by higher scales.

    Active-version model (2026-05-10):
      - `register()` inserts a new version row. Does NOT change which version
        the runtime reads. Decoupled by design.
      - `set_active()` flips the per-name active pointer to a chosen version.
      - `get_active()` reads the active version. Falls back to MAX(version)
        when no pointer exists (covers fresh brains pre-seed).
      - `get_version()` reads a specific version (used by eval overrides).
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def register(self, name: str, template: str, parameters: str = '',
                 created_by: str = 'anchor') -> Dict[str, Any]:
        """Register a new version of an interaction. Auto-increments version.

        Activation semantics:
          - If this is version 1 (first registration for this name), AUTO-ACTIVATE.
            Otherwise nothing is reading anything for this name — making it
            active is the right default.
          - If this is version 2 or later, do NOT activate. Caller must call
            `set_active()` explicitly to flip the runtime pointer.
        """
        now = iso_now()
        # Get current max version
        row = self.conn.execute(
            'SELECT MAX(version) FROM interactions WHERE name = ?', (name,)
        ).fetchone()
        version = (row[0] or 0) + 1 if row else 1
        parent = version - 1 if version > 1 else None
        self.conn.execute(
            'INSERT INTO interactions (name, version, template, parameters, created_at, created_by, parent_version) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (name, version, template, parameters, now, created_by, parent))
        new_id = self.conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        # Auto-activate v1 ONLY. Subsequent versions require explicit set_active.
        was_activated = False
        if version == 1:
            self.conn.execute(
                'INSERT OR REPLACE INTO interaction_active (name, version, set_at, set_by) '
                'VALUES (?, ?, ?, ?)',
                (name, version, now, 'register:auto_v1'))
            was_activated = True
        commit_unless_batched(self.conn)
        return {'name': name, 'version': version, 'id': new_id,
                'auto_activated': was_activated}

    def set_active(self, name: str, version: int,
                   set_by: str = 'anchor') -> Dict[str, Any]:
        """Flip the active pointer for `name` to `version`.

        UPSERT into interaction_active. Verifies the (name, version) pair
        actually exists in `interactions` before flipping — refuses to activate
        a non-existent version.
        """
        # Verify the target version exists
        row = self.conn.execute(
            'SELECT 1 FROM interactions WHERE name = ? AND version = ?',
            (name, version)).fetchone()
        if not row:
            raise ValueError(
                "Cannot activate %s v%d: no such version registered" % (name, version))
        now = iso_now()
        self.conn.execute(
            'INSERT INTO interaction_active (name, version, set_at, set_by) '
            'VALUES (?, ?, ?, ?) '
            'ON CONFLICT(name) DO UPDATE SET '
            'version = excluded.version, set_at = excluded.set_at, set_by = excluded.set_by',
            (name, version, now, set_by))
        commit_unless_batched(self.conn)
        return {'name': name, 'version': version, 'set_at': now, 'set_by': set_by}

    def get_active(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the currently-active version of an interaction.

        Reads via interaction_active pointer. Falls back to MAX(version) when
        no pointer exists (fresh brain, mid-seed, or unmigrated state).
        Returns None when no version exists at all.
        """
        # Try the pointer first
        row = self.conn.execute(
            'SELECT i.id, i.name, i.version, i.template, i.parameters, '
            'i.created_at, i.created_by '
            'FROM interaction_active a '
            'JOIN interactions i ON i.name = a.name AND i.version = a.version '
            'WHERE a.name = ?', (name,)).fetchone()
        if row:
            return {'id': row[0], 'name': row[1], 'version': row[2],
                    'template': row[3], 'parameters': row[4],
                    'created_at': row[5], 'created_by': row[6]}
        # Fallback: MAX(version) — covers pre-seed bootstrap windows
        row = self.conn.execute(
            'SELECT id, name, version, template, parameters, created_at, created_by '
            'FROM interactions WHERE name = ? ORDER BY version DESC LIMIT 1',
            (name,)).fetchone()
        if not row:
            return None
        return {'id': row[0], 'name': row[1], 'version': row[2],
                'template': row[3], 'parameters': row[4],
                'created_at': row[5], 'created_by': row[6]}

    def get_version(self, name: str, version: int) -> Optional[Dict[str, Any]]:
        """Get a specific version of an interaction."""
        row = self.conn.execute(
            'SELECT id, name, version, template, parameters, created_at, created_by, parent_version '
            'FROM interactions WHERE name = ? AND version = ?',
            (name, version)).fetchone()
        if not row:
            return None
        return {'id': row[0], 'name': row[1], 'version': row[2],
                'template': row[3], 'parameters': row[4],
                'created_at': row[5], 'created_by': row[6],
                'parent_version': row[7]}

    def list_all(self) -> List[Dict[str, Any]]:
        """List all interactions: name, max_version, total_versions, active_version."""
        rows = self.conn.execute(
            'SELECT i.name, MAX(i.version) as v, COUNT(*) as versions, a.version '
            'FROM interactions i '
            'LEFT JOIN interaction_active a ON a.name = i.name '
            'GROUP BY i.name ORDER BY i.name').fetchall()
        return [{'name': r[0], 'max_version': r[1], 'total_versions': r[2],
                 'active_version': r[3]} for r in rows]


class TraceDAL:
    """Access layer for trace_events — the fractal learning loop.

    Append-only event chains. Each chain tracks one integrate() cycle:
    what was observed, what knowledge was selected, what was produced,
    and what happened next (corrections, recalls, outcomes).

    Over time, all small log tables migrate into trace_events.
    Each becomes a different event_type + ref_type in one table.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        # Identity stamped onto every trace event's metadata at write time.
        # Empty strings = unset → no stamping (graceful degradation; no
        # placeholder sentinel tokens). Set once at Brain init via
        # set_identity() from daemon_config env vars.
        self._human_identity: str = ''
        self._agent_identity: str = ''
        # One-shot guard for the "identity unset on write" warning.
        # The gap doesn't change per-write, only per-restart — logging
        # every write would flood; logging once per TraceDAL lifetime
        # surfaces the gap without noise.
        self._identity_missing_logged: bool = False

    def set_identity(self, human_identity: str, agent_identity: str) -> None:
        """Configure the identity tokens stamped onto every trace event.

        Source values come from BRAIN_OPERATOR_NAME / BRAIN_AGENT_NAME env
        vars resolved at Brain construction. Identity is a property of the
        substrate — each trace_event records the speakers present when it
        was written. Per-event override is possible by passing
        human_identity / agent_identity directly in `metadata`.
        """
        self._human_identity = (human_identity or '').strip()
        self._agent_identity = (agent_identity or '').strip()

    def _maybe_warn_identity_unset(self, scale: str, ref_type: str) -> None:
        """Loud-by-default at the scale-write boundary: when a trace
        gets written but identity stamping is empty, surface it once.

        The check lives at the write site (not at Brain.__init__)
        because boot is a moment but writes are continuous — every
        scale write through this DAL passes here. The first one that
        fires with unset identity tells the operator the gap exists.
        Subsequent writes stay silent (one-shot via
        `_identity_missing_logged`) — the config didn't suddenly
        re-break, no value in spamming.

        Output goes to stderr → daemon.log (launchd-managed
        StandardErrorPath). TraceDAL has no Brain reference, so we
        don't route through brain._log_error here.
        """
        if self._human_identity and self._agent_identity:
            return
        if self._identity_missing_logged:
            return
        import sys as _sys
        missing = []
        if not self._human_identity:
            missing.append('BRAIN_OPERATOR_NAME')
        if not self._agent_identity:
            missing.append('BRAIN_AGENT_NAME')
        print('[trace_dal] identity unset on first scale-%s write '
              '(ref_type=%s) — %s missing in env. Trace events will '
              'be written without identity metadata until '
              '~/.config/brain/env is configured and the daemon '
              'restarts.' %
              (scale, ref_type or '?', ', '.join(missing)),
              file=_sys.stderr, flush=True)
        self._identity_missing_logged = True

    def _stamp_identity(self, metadata: Optional[Dict]) -> Optional[Dict]:
        """Inject configured identity tokens into a metadata dict.

        setdefault semantics — explicit per-event values win. Returns
        metadata unchanged if neither identity is configured, or if
        metadata is a non-dict value (defensive: callers that pass
        unexpected shapes don't crash the trace write — the daemon
        dispatch layer is responsible for normalizing to dict).
        """
        if not self._human_identity and not self._agent_identity:
            return metadata
        if metadata is not None and not isinstance(metadata, dict):
            return metadata
        meta = dict(metadata) if metadata else {}
        if self._human_identity:
            meta.setdefault('human_identity', self._human_identity)
        if self._agent_identity:
            meta.setdefault('agent_identity', self._agent_identity)
        return meta

    def _warn_metadata_invalid(self, ref_type, error):
        """Loud-by-default at the trace-write chokepoint: a ref_type with a
        declared payload schema carried a malformed metadata dict. NEVER blocks
        the write (a malformed trace beats a lost one) — logs to stderr →
        daemon.log, the same channel as _maybe_warn_identity_unset (TraceDAL has
        no Brain reference). Fires per occurrence — each malformed trace is a
        distinct signal."""
        import sys as _sys
        print('[trace_dal] trace metadata invalid (ref_type=%s): %s — wrote anyway'
              % (ref_type, error), file=_sys.stderr, flush=True)

    def append(self, chain_id: str, scale: str, event_type: str,
               ref_type: str = '', ref_id: str = '', summary: str = '',
               metadata: Optional[Dict] = None, session_id: str = '',
               interaction_id: int = None) -> int:
        """Append an event to a trace chain. Returns event id.

        Validates against trace_contract before writing. Configured
        identity tokens (set_identity) are stamped into metadata via
        setdefault — explicit per-event values win.
        """
        from .trace_contract import validate_trace_event, validate_trace_metadata
        ok, error = validate_trace_event(scale, event_type, ref_type)
        if not ok:
            raise ValueError("Trace contract violation: %s" % error)
        # Payload shape — loud, never block. This is the SINGLE chokepoint every
        # writer passes (inline S1/S2 + the dispatched command), so the guard
        # actually fires in production: the command-boundary check missed every
        # in-process delta write (S2 units + S1 Scribe run with dispatch=None).
        meta_ok, meta_err = validate_trace_metadata(event_type, ref_type, metadata)
        if not meta_ok:
            self._warn_metadata_invalid(ref_type, meta_err)

        self._maybe_warn_identity_unset(scale, ref_type)
        metadata = self._stamp_identity(metadata)
        now = iso_now()
        meta_json = json.dumps(metadata) if metadata else None
        trace_id = _new_trace_id(self.conn)
        self.conn.execute(
            'INSERT INTO trace_events '
            '(id, chain_id, scale, event_type, ref_type, ref_id, summary, metadata, session_id, interaction_id, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (trace_id, chain_id, scale, event_type, ref_type, ref_id,
             summary if summary else '', meta_json, session_id, interaction_id, now))
        commit_unless_batched(self.conn)
        return trace_id

    def append_batch(self, events: list) -> List[str]:
        """Append multiple trace events in a single transaction.

        Reduces WAL lock contention — one commit instead of N.
        Each event dict uses the same keys as append(). Identity stamping
        applies per-event via the same setdefault semantics as append().
        """
        from .trace_contract import validate_trace_event, validate_trace_metadata
        now = iso_now()
        ids = []
        for ev in events:
            ok, error = validate_trace_event(ev['scale'], ev['event_type'], ev.get('ref_type', ''))
            if not ok:
                raise ValueError("Trace contract violation: %s" % error)
            meta_ok, meta_err = validate_trace_metadata(
                ev['event_type'], ev.get('ref_type', ''), ev.get('metadata'))
            if not meta_ok:
                self._warn_metadata_invalid(ev.get('ref_type', ''), meta_err)
            self._maybe_warn_identity_unset(ev['scale'], ev.get('ref_type', ''))
            metadata = self._stamp_identity(ev.get('metadata'))
            meta_json = json.dumps(metadata) if metadata else None
            trace_id = _new_trace_id(self.conn)
            self.conn.execute(
                'INSERT INTO trace_events '
                '(id, chain_id, scale, event_type, ref_type, ref_id, summary, metadata, session_id, interaction_id, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (trace_id, ev['chain_id'], ev['scale'], ev['event_type'], ev.get('ref_type', ''),
                 ev.get('ref_id', ''), ev.get('summary', ''), meta_json,
                 ev.get('session_id', ''), ev.get('interaction_id'), now))
            ids.append(trace_id)
        commit_unless_batched(self.conn)
        return ids

    def _decode_metadata(self, raw: Optional[str]) -> Dict[str, Any]:
        """Decode a trace_events.metadata JSON cell into a dict.

        Defensive against pre-v27 tool_result rows where the metadata
        was double-encoded (the dispatch handler json.dumps'd a string
        client payload, then trace_dal.append json.dumps'd it again).
        Single-encoded rows (post-2026-05-23 dispatch fix) decode in
        one pass; double-encoded legacy rows decode in two. Returns
        an empty dict on any failure so callers can rely on dict shape.
        """
        if not raw:
            return {}
        try:
            meta = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                return {}
        return meta if isinstance(meta, dict) else {}

    def get_by_ids(self, trace_ids: List[str]) -> List[Dict[str, Any]]:
        """Point/batch lookup by trace_event.id (v29: 8-char hex strings).
        Returns rows in ascending id order (deterministic); missing ids are
        silently skipped (mirrors NodeDAL.get_bulk behavior — caller checks
        len(result) vs len(input) if presence matters).

        Rejects int input loudly per v29 contract.
        """
        if not trace_ids:
            return []
        for tid in trace_ids:
            if not isinstance(tid, str):
                raise ValueError(
                    "get_by_ids: trace_ids must be strings, got %s (%r). "
                    "v29 trace ids are 8-char hex." % (
                        type(tid).__name__, tid))
        placeholders = ','.join('?' * len(trace_ids))
        rows = self.conn.execute(
            'SELECT id, chain_id, scale, event_type, ref_type, ref_id, '
            '       summary, metadata, session_id, created_at '
            'FROM trace_events WHERE id IN (%s) '
            'ORDER BY id ASC' % placeholders,
            list(trace_ids)).fetchall()
        return [{
            'id': r[0], 'chain_id': r[1], 'scale': r[2],
            'event_type': r[3], 'ref_type': r[4] or '', 'ref_id': r[5] or '',
            'summary': r[6] or '', 'metadata': self._decode_metadata(r[7]),
            'session_id': r[8] or '', 'created_at': r[9],
        } for r in rows]

    def get_chain(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get all events in a trace chain, ordered by time."""
        rows = self.conn.execute(
            'SELECT id, scale, event_type, ref_type, ref_id, summary, metadata, created_at, session_id '
            'FROM trace_events WHERE chain_id = ? ORDER BY created_at ASC',
            (chain_id,)).fetchall()
        results = []
        for r in rows:
            results.append({
                'id': r[0], 'scale': r[1], 'event_type': r[2],
                'ref_type': r[3], 'ref_id': r[4], 'summary': r[5],
                'metadata': self._decode_metadata(r[6]),
                'created_at': r[7], 'session_id': r[8] or ''})
        return results

    def get_recent(self, scale: str = '', hours: Optional[int] = 24,
                   event_type: str = '', session_id: str = '',
                   session_ids: Optional[List[str]] = None,
                   limit: int = 100, chain_suffix: str = '',
                   exclude_ref_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get recent trace events, optionally filtered by scale/type/session.

        Session filtering — three modes:

        - **session_id** (singular, str): authoritative for one session. The
          `hours` window is ignored — historical sessions older than the
          default cutoff must not silently truncate to empty. Zero-row
          results log loud to stderr.

        - **session_ids** (plural, List[str]): authoritative for N sessions.
          Returns events from any session in the list, ordered by time.
          The `hours` window is ignored (same reasoning). Zero-row results
          log loud. Useful for cross-session audits, eval cohorts,
          partnership-arc lookups across distinct conversations.

        - **neither**: default hours-based recent window.

        Passing both session_id and session_ids raises ValueError — the
        caller's intent is ambiguous and we don't guess.
        """
        if session_id and session_ids:
            raise ValueError(
                "Pass either session_id (single str) or session_ids (List[str]), "
                "not both. Got session_id=%r session_ids=%r"
                % (session_id, session_ids))

        conditions: list = []
        params: list = []
        if session_ids:
            # Plural authoritative — IN clause across requested sessions,
            # skip the hours cutoff entirely.
            placeholders = ','.join(['?'] * len(session_ids))
            conditions.append('session_id IN (%s)' % placeholders)
            params.extend(session_ids)
        elif session_id:
            # Singular authoritative — equality, skip hours cutoff.
            conditions.append('session_id = ?')
            params.append(session_id)
        else:
            if hours is not None:    # None = no time window (caller bounds via limit)
                conditions.append('created_at > ?')
                params.append(iso_cutoff(hours=hours))
        if scale:
            conditions.append('scale = ?')
            params.append(scale)
        if event_type:
            conditions.append('event_type = ?')
            params.append(event_type)
        if chain_suffix:
            conditions.append("chain_id LIKE ? ESCAPE '\\'")
            params.append(self._like_suffix_param(chain_suffix))
        if exclude_ref_types:
            # Exclude residue (journal_note) so "recent integration deltas"
            # don't count encoder notes. NULL ref_type is kept (treated as
            # non-residue), not dropped by NOT IN.
            ph = ','.join(['?'] * len(exclude_ref_types))
            conditions.append('(ref_type IS NULL OR ref_type NOT IN (%s))' % ph)
            params.extend(exclude_ref_types)
        where = ' AND '.join(conditions) if conditions else '1=1'
        rows = self.conn.execute(
            'SELECT id, chain_id, scale, event_type, ref_type, ref_id, summary, created_at, session_id '
            'FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?' % where,
            params + [limit]).fetchall()
        out = [{'id': r[0], 'chain_id': r[1], 'scale': r[2], 'event_type': r[3],
                 'ref_type': r[4], 'ref_id': r[5], 'summary': r[6], 'created_at': r[7],
                 'session_id': r[8] or ''}
                for r in rows]
        # Loud-by-default: explicit session filter with zero rows is a real
        # signal (typo, archived session, wrong DB), not a quiet noop.
        if (session_id or session_ids) and not out:
            import sys as _sys
            who = session_id[:12] if session_id else ('[' + ','.join(s[:8] for s in session_ids[:3]) +
                                                       (',…' if len(session_ids) > 3 else '') + ']')
            print('[query_traces] WARN session filter %s returned 0 rows '
                  '(scale=%r event_type=%r limit=%d)'
                  % (who, scale, event_type, limit),
                  file=_sys.stderr)
        return out

    def _event_where(self, contains: str, scale: str, event_type: str,
                     ref_types: Optional[List[str]], session_id: str,
                     session_ids: Optional[List[str]],
                     younger_than: str, older_than: str):
        """Build the shared WHERE clause + params for recall_episodes' two
        access paths (filter_events / filter_event_vectors). Columns are
        `te.`-qualified so the same clause works under the trace_embeddings
        JOIN, where `created_at` would otherwise be ambiguous.

        Needles: contains → (summary OR metadata) LIKE %s% (same idiom as
        find_by_metadata_substring; metadata is JSON text, so it greps the full
        body, not the 200-char summary). Structural: scale/event_type equality.
        ref_types: an INCLUDE whitelist (te.ref_type IN (...)); None/empty = no
        ref_type filter (all types). The caller (BrainEpisodesMixin) sources the
        default whitelist from the trace_contract dial, so there's no hardcoded
        list here to drift. Scope: session_id XOR session_ids (both raises, like
        get_recent). Time (ISO, caller pre-resolves shorthand): younger_than →
        created_at >, older_than → <.
        """
        if session_id and session_ids:
            raise ValueError(
                "Pass either session_id (single str) or session_ids (List[str]), "
                "not both. Got session_id=%r session_ids=%r"
                % (session_id, session_ids))
        conditions: list = []
        params: list = []
        if session_ids:
            placeholders = ','.join(['?'] * len(session_ids))
            conditions.append('te.session_id IN (%s)' % placeholders)
            params.extend(session_ids)
        elif session_id:
            conditions.append('te.session_id = ?')
            params.append(session_id)
        if scale:
            conditions.append('te.scale = ?')
            params.append(scale)
        if event_type:
            conditions.append('te.event_type = ?')
            params.append(event_type)
        if ref_types:
            placeholders = ','.join(['?'] * len(ref_types))
            conditions.append('te.ref_type IN (%s)' % placeholders)
            params.extend(ref_types)
        if contains:
            like = '%' + contains + '%'
            conditions.append('(te.summary LIKE ? OR te.metadata LIKE ?)')
            params.extend([like, like])
        if younger_than:
            conditions.append('te.created_at > ?')
            params.append(younger_than)
        if older_than:
            conditions.append('te.created_at < ?')
            params.append(older_than)
        return (' AND '.join(conditions) if conditions else '1=1'), params

    def filter_events(self, contains: str = '', scale: str = '',
                      event_type: str = '', ref_types: Optional[List[str]] = None,
                      session_id: str = '', session_ids: Optional[List[str]] = None,
                      younger_than: str = '', older_than: str = '',
                      sort_order: str = 'desc',
                      limit: int = 10) -> List[Dict[str, Any]]:
        """Structured + lexical query over trace_events — the filter_nodes
        analog for the traces layer. Returns full DECODED records (same shape
        as get_by_ids), the substance recall_episodes returns to the caller.
        The time/no-query path: indexed WHERE + ORDER BY created_at + LIMIT
        early-exits, so only `limit` rows are decoded. See _event_where for the
        filter semantics (ref_types is an INCLUDE whitelist). Semantic ranking
        lives in BrainEpisodesMixin.recall_episodes.
        """
        from .brain_constants import EPISODE_MAX_LIMIT
        limit = min(max(int(limit), 1), EPISODE_MAX_LIMIT)
        where, params = self._event_where(
            contains, scale, event_type, ref_types, session_id, session_ids,
            younger_than, older_than)
        order = 'ASC' if sort_order == 'asc' else 'DESC'
        rows = self.conn.execute(
            'SELECT te.id, te.chain_id, te.scale, te.event_type, te.ref_type, '
            'te.ref_id, te.summary, te.metadata, te.session_id, te.created_at '
            'FROM trace_events te WHERE %s ORDER BY te.created_at %s LIMIT ?'
            % (where, order),
            params + [limit]).fetchall()
        return [{'id': r[0], 'chain_id': r[1], 'scale': r[2], 'event_type': r[3],
                 'ref_type': r[4] or '', 'ref_id': r[5] or '', 'summary': r[6] or '',
                 'metadata': self._decode_metadata(r[7]), 'session_id': r[8] or '',
                 'created_at': r[9]}
                for r in rows]

    def filter_event_vectors(self, contains: str = '', scale: str = '',
                             event_type: str = '', ref_types: Optional[List[str]] = None,
                             session_id: str = '',
                             session_ids: Optional[List[str]] = None,
                             younger_than: str = '', older_than: str = '',
                             limit: int = 500) -> List[tuple]:
        """Lean candidate scan for semantic recall_episodes: returns
        [(trace_id, vector)] for embedded traces matching the same filter as
        filter_events, INNER JOINed to trace_embeddings (only embedded traces
        are rankable). No metadata decode and no second query — the embedding
        rides the JOIN, and only the top-k full records get hydrated (via
        get_by_ids) after ranking. Newest-first so the cap keeps recent ones.
        """
        from .brain_constants import EPISODE_MAX_LIMIT
        limit = min(max(int(limit), 1), EPISODE_MAX_LIMIT)
        where, params = self._event_where(
            contains, scale, event_type, ref_types, session_id, session_ids,
            younger_than, older_than)
        rows = self.conn.execute(
            'SELECT te.id, tem.vector FROM trace_events te '
            'JOIN trace_embeddings tem ON tem.trace_id = te.id '
            'WHERE %s ORDER BY te.created_at DESC LIMIT ?' % where,
            params + [limit]).fetchall()
        return [(r[0], r[1]) for r in rows if r[1]]

    def get_chains_for_session(self, session_id: str) -> List[str]:
        """Get all chain IDs from a session."""
        rows = self.conn.execute(
            'SELECT DISTINCT chain_id FROM trace_events WHERE session_id = ? ORDER BY created_at ASC',
            (session_id,)).fetchall()
        return [r[0] for r in rows]

    def append_outcome(self, chain_id: str, scale: str, ref_type: str, ref_id: str,
                       summary: str, session_id: str = '') -> int:
        """Append an outcome event to an existing chain. Called later when
        we learn what happened (correction, recall, revision)."""
        return self.append(
            chain_id=chain_id, scale=scale, event_type='outcome',
            ref_type=ref_type, ref_id=ref_id, summary=summary,
            session_id=session_id)

    def get_chains(self, session_id: str = '', scale: str = '',
                   hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        """Get complete chains grouped, with all events and metadata.

        Returns: [{chain_id, scale, events: [{event_type, ref_type, summary, metadata, created_at}]}]
        Ordered by most recent chain first.
        """
        conditions = ['created_at > ?']
        params = [iso_cutoff(hours=hours)]
        if session_id:
            conditions.append('session_id = ?')
            params.append(session_id)
        if scale:
            conditions.append('scale = ?')
            params.append(scale)
        where = ' AND '.join(conditions)

        rows = self.conn.execute(
            'SELECT chain_id, scale, event_type, ref_type, ref_id, summary, metadata, created_at, session_id '
            'FROM trace_events WHERE %s ORDER BY created_at DESC' % where,
            params).fetchall()

        # Group by chain_id, preserve order of first appearance.
        # A chain belongs to one session; we record the session_id from
        # the first event seen in each chain.
        chains = {}
        chain_order = []
        for r in rows:
            cid = r[0]
            if cid not in chains:
                chains[cid] = {'chain_id': cid, 'scale': r[1],
                               'session_id': r[8] or '', 'events': []}
                chain_order.append(cid)
            chains[cid]['events'].append({
                'event_type': r[2], 'ref_type': r[3] or '', 'ref_id': r[4] or '',
                'summary': r[5] or '', 'metadata': self._decode_metadata(r[6]),
                'created_at': r[7]})

        # Reverse events within each chain to chronological order
        for cid in chain_order:
            chains[cid]['events'].reverse()

        result = [chains[cid] for cid in chain_order[:limit]]
        return result

    @staticmethod
    def _like_suffix_param(suffix: str) -> str:
        """LIKE param matching chains ENDING in '-{suffix}', with LIKE metachars
        in the suffix escaped so a '_' in a unit name (community_detection)
        matches literally, not as a single-char wildcard. Pair with the clause
        `chain_id LIKE ? ESCAPE '\\'`."""
        esc = suffix.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        return '%-' + esc

    def get_by_ref_type(self, ref_type: str, scale: str = '',
                        hours: Optional[int] = 24, limit: int = 100,
                        session_id: str = '', ref_id: str = '',
                        chain_suffix: str = '') -> List[Dict[str, Any]]:
        """Get events filtered by ref_type.

        Use: "all corrections", "all recall_hits", "all encoding_runs".
        Pass hours=None to disable the time-window filter (caller controls
        recency purely via `limit` + `ORDER BY created_at DESC`).
        Pass session_id to scope results to a single session — required for
        per-session reads (e.g. surface's recently-surfaced dedup list).
        Pass ref_id to scope to a single subject (e.g. journal notes about one
        node — `ref_type='journal_note', ref_id=<node>`).
        Pass chain_suffix to scope to chains ENDING in '-{suffix}' — the S2 unit
        identity lives in the chain suffix (`s2-{ts}-{unit}`). LIMIT then bounds
        the per-unit result, not the global stream.
        """
        conditions = ['ref_type = ?']
        params: List[Any] = [ref_type]
        if hours is not None:
            conditions.append('created_at > ?')
            params.append(iso_cutoff(hours=hours))
        if scale:
            conditions.append('scale = ?')
            params.append(scale)
        if session_id:
            conditions.append('session_id = ?')
            params.append(session_id)
        if ref_id:
            conditions.append('ref_id = ?')
            params.append(ref_id)
        if chain_suffix:
            conditions.append("chain_id LIKE ? ESCAPE '\\'")
            params.append(self._like_suffix_param(chain_suffix))
        where = ' AND '.join(conditions)

        rows = self.conn.execute(
            'SELECT id, chain_id, scale, event_type, ref_type, ref_id, summary, metadata, created_at, session_id '
            'FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?' % where,
            params + [limit]).fetchall()

        results = []
        for r in rows:
            results.append({
                'id': r[0], 'chain_id': r[1], 'scale': r[2], 'event_type': r[3],
                'ref_type': r[4] or '', 'ref_id': r[5] or '', 'summary': r[6] or '',
                'metadata': self._decode_metadata(r[7]),
                'created_at': r[8], 'session_id': r[9] or ''})
        return results

    def get_outcomes(self, chain_id: str = '', scale: str = '',
                     hours: int = 168) -> List[Dict[str, Any]]:
        """Get outcome events, optionally for a specific chain or scale.

        Use: S3 checks which S1 chains got corrected vs validated.
        Default 168h = 7 days.
        """
        conditions = ["event_type = 'outcome'", 'created_at > ?']
        params = [iso_cutoff(hours=hours)]
        if chain_id:
            conditions.append('chain_id = ?')
            params.append(chain_id)
        if scale:
            conditions.append('scale = ?')
            params.append(scale)
        where = ' AND '.join(conditions)

        rows = self.conn.execute(
            'SELECT id, chain_id, scale, ref_type, ref_id, summary, metadata, created_at '
            'FROM trace_events WHERE %s ORDER BY created_at DESC' % where,
            params).fetchall()

        results = []
        for r in rows:
            results.append({
                'id': r[0], 'chain_id': r[1], 'scale': r[2], 'event_type': 'outcome',
                'ref_type': r[3] or '', 'ref_id': r[4] or '', 'summary': r[5] or '',
                'metadata': self._decode_metadata(r[6]), 'created_at': r[7]})
        return results

    def count_by(self, field: str, scale: str = '', hours: int = 24) -> Dict[str, int]:
        """Count events grouped by a field.

        field: 'event_type', 'ref_type', or 'chain_id'
        Returns: {value: count} dict
        """
        allowed = {'event_type', 'ref_type', 'chain_id', 'scale'}
        if field not in allowed:
            return {}

        conditions = ['created_at > ?']
        params = [iso_cutoff(hours=hours)]
        if scale:
            conditions.append('scale = ?')
            params.append(scale)
        where = ' AND '.join(conditions)

        rows = self.conn.execute(
            'SELECT %s, COUNT(*) FROM trace_events WHERE %s GROUP BY %s' % (field, where, field),
            params).fetchall()

        return {r[0] or '': r[1] for r in rows}

    def latest_in_window(self, scale: str, ref_type: str,
                         upper_iso: str, lower_iso: str) -> Optional[Dict[str, str]]:
        """Most recent trace matching (scale, ref_type) with
        lower_iso <= created_at <= upper_iso. Returns
        {'session_id', 'created_at'} or None. Forensic/historic lookup.
        """
        row = self.conn.execute(
            "SELECT session_id, created_at FROM trace_events "
            "WHERE scale = ? AND ref_type = ? AND created_at <= ? AND created_at >= ? "
            "ORDER BY created_at DESC LIMIT 1",
            (scale, ref_type, upper_iso, lower_iso)).fetchone()
        return {'session_id': row[0], 'created_at': row[1]} if row else None

    def active_sessions_by_turn(self, cutoff_iso: str, exclude_session: str = '',
                                limit: int = 5,
                                sort_by: str = 'recency') -> List[Dict[str, Any]]:
        """Sessions reachable RIGHT NOW since `cutoff_iso`, newest first — the
        wall-clock presence signal, sourced from S0 traces.

        Counts user_message / assistant_message AND heartbeat turns. Heartbeats
        are emitted by /watch listeners on every quiet tick — a stream living
        purely in watch mode is the MOST reachable (it can be triggered to act),
        so it must count as present, not vanish after 30 min (2026-06-04).
        Autosave writes no traces at all, so this still reflects ACTUAL turns —
        not cached/autosaved session_state (whose updated_at is bumped for every
        cached session each tick, falsely marking idle/stale sids "live"). A
        stream that stops taking turns of any kind ages out once its last turn
        passes the cutoff; a sid relaunched under a new id drops its stale sid.

        Returns [{'session_id', 'last_turn', 'focus'}] where `focus` is the
        latest CONVERSATIONAL turn — user_message OR assistant_message, per
        trace_contract.CONVERSATIONAL_REF_TYPES (not user-only): a watcher's
        last real work is often its own last reply. Turns whose summary starts
        with the wake-envelope marker (a `<task-notification>` ignition) are
        skipped so the focus shows work, not the wake envelope. Both the
        conversational set and the marker come from the contract — no filters
        reproduced here. (Raw; the render layer first-lines/truncates.) Caller
        computes the cutoff (wall-clock vs conversation-time is the caller's
        policy, not the DAL's)."""
        from .trace_contract import CONVERSATIONAL_REF_TYPES, WAKE_ENVELOPE_MARKER
        conv_ph = ','.join('?' * len(CONVERSATIONAL_REF_TYPES))
        # Presence (liveness) also counts heartbeats — a watch listener living
        # purely on heartbeats is the most reachable stream (B2, 2026-06-04).
        live_types = CONVERSATIONAL_REF_TYPES + ('heartbeat',)
        live_ph = ','.join('?' * len(live_types))
        # scale='s0' is a behavior-preserving predicate (every conversational +
        # heartbeat ref_type is s0-only per trace_contract.REF_TYPES) that lets
        # the query use idx_trace_scope_created (scale, ref_type, created_at) —
        # the composite leads with scale, so without the equality it can't engage.
        # Sort key whitelist (never interpolate raw sort_by into SQL):
        #  recency (default) — most recent CONVERSATIONAL turn first, so a fresh
        #    boot (heartbeat only, conv_recency NULL → sorts last in DESC) can't
        #    crowd out a substantial stream. Membership still counts heartbeats
        #    (live_types below) so watch listeners / fresh boots stay VISIBLE,
        #    just ranked below real work.
        #  length — by number of conversational turns (user_message count).
        order = ("turn_count DESC, conv_recency DESC"
                 if sort_by == 'length'
                 else "conv_recency DESC, last_turn DESC")
        rows = self.conn.execute(
            "SELECT t.session_id, MAX(t.created_at) AS last_turn, "
            "  (SELECT u.summary FROM trace_events u "
            "   WHERE u.scale = 's0' AND u.session_id = t.session_id AND u.ref_type IN (%s) "
            "     AND u.summary NOT LIKE ? "
            "   ORDER BY u.created_at DESC LIMIT 1) AS focus, "
            "  (SELECT MAX(c.created_at) FROM trace_events c "
            "   WHERE c.scale = 's0' AND c.session_id = t.session_id AND c.ref_type IN (%s) "
            "     AND c.summary NOT LIKE ?) AS conv_recency, "
            "  (SELECT COUNT(*) FROM trace_events c2 "
            "   WHERE c2.scale = 's0' AND c2.session_id = t.session_id "
            "     AND c2.ref_type = 'user_message' AND c2.summary NOT LIKE ?) AS turn_count "
            "FROM trace_events t "
            "WHERE t.scale = 's0' AND t.ref_type IN (%s) "
            "  AND t.created_at > ? AND t.session_id != ? "
            "GROUP BY t.session_id "
            "ORDER BY %s LIMIT ?" % (conv_ph, conv_ph, live_ph, order),
            (*CONVERSATIONAL_REF_TYPES, WAKE_ENVELOPE_MARKER + '%',
             *CONVERSATIONAL_REF_TYPES, WAKE_ENVELOPE_MARKER + '%',
             WAKE_ENVELOPE_MARKER + '%',
             *live_types, cutoff_iso, exclude_session or '', limit)).fetchall()
        return [{'session_id': r[0], 'last_turn': r[1], 'focus': r[2] or '',
                 'conv_recency': r[3] or '', 'turn_count': r[4] or 0}
                for r in rows]

    def session_activity(self, session_id: str, msg_limit: int = 2) -> Dict[str, Any]:
        """Activity snapshot for ONE session — self_peek's enrichment data.

        Returns {'started_at', 'last_active_at', 'recent_msgs'} from this
        session's S0 traces:
          started_at     — first conversational turn (MIN), tenure
          last_active_at — most recent turn of ANY live kind incl. heartbeats
                           (MAX), for liveness; a watch listener's quiet ticks
                           ARE activity (same rule as active_sessions_by_turn)
          recent_msgs    — last `msg_limit` conversational turns, newest first,
                           [{'ts','ref_type','text'}], wake-envelope turns
                           skipped so a peek shows work not ignition (raw; the
                           render layer truncates). Read-only.
        """
        from .trace_contract import CONVERSATIONAL_REF_TYPES, WAKE_ENVELOPE_MARKER
        conv_ph = ','.join('?' * len(CONVERSATIONAL_REF_TYPES))
        live_types = CONVERSATIONAL_REF_TYPES + ('heartbeat',)
        live_ph = ','.join('?' * len(live_types))
        agg = self.conn.execute(
            "SELECT "
            "  (SELECT MIN(created_at) FROM trace_events "
            "     WHERE scale='s0' AND session_id=? AND ref_type IN (%s)), "
            "  (SELECT MAX(created_at) FROM trace_events "
            "     WHERE scale='s0' AND session_id=? AND ref_type IN (%s)), "
            "  (SELECT COUNT(*) FROM trace_events "
            "     WHERE scale='s0' AND session_id=? AND ref_type='user_message')"
            % (conv_ph, live_ph),
            (session_id, *CONVERSATIONAL_REF_TYPES,
             session_id, *live_types, session_id)).fetchone()
        started_at = (agg[0] or '') if agg else ''
        last_active_at = (agg[1] or '') if agg else ''
        turn_count = (agg[2] or 0) if agg else 0
        rows = self.conn.execute(
            "SELECT created_at, ref_type, summary FROM trace_events "
            "WHERE scale='s0' AND session_id=? AND ref_type IN (%s) "
            "  AND summary NOT LIKE ? "
            "ORDER BY created_at DESC LIMIT ?" % conv_ph,
            (session_id, *CONVERSATIONAL_REF_TYPES,
             WAKE_ENVELOPE_MARKER + '%', msg_limit)).fetchall()
        recent = [{'ts': r[0], 'ref_type': r[1], 'text': r[2] or ''} for r in rows]
        return {'started_at': started_at, 'last_active_at': last_active_at,
                'turn_count': turn_count, 'recent_msgs': recent}

    def conversational_turns_since(self, session_id: str, since_iso: str = '') -> int:
        """Count this session's conversational turns — optionally only those after
        `since_iso`. A turn == one s0 `user_message` trace, wake-envelope
        (`<task-notification>`) ignitions excluded — the same definition
        active_sessions_by_turn's turn_count uses. This is the S1 Scribe's cadence
        primitive: turns-since-last-encode, read live from traces instead of a
        maintained counter (which desynced across resume). Uses idx_trace_session
        + the scale/ref_type predicates. since_iso='' counts all turns."""
        from .trace_contract import WAKE_ENVELOPE_MARKER
        sql = ("SELECT COUNT(*) FROM trace_events WHERE scale='s0' "
               "AND session_id=? AND ref_type='user_message' AND summary NOT LIKE ?")
        params: List[Any] = [session_id, WAKE_ENVELOPE_MARKER + '%']
        if since_iso:
            sql += " AND created_at > ?"
            params.append(since_iso)
        return self.conn.execute(sql, params).fetchone()[0]

    def find_by_metadata_substring(self, scale: str, ref_type: str,
                                   substring: str) -> Optional[Dict[str, str]]:
        """First trace matching (scale, ref_type) whose metadata contains
        `substring` (LIKE %substring%). Returns {'session_id', 'created_at'}
        or None. Used to locate the trace that recorded a given node id.
        """
        row = self.conn.execute(
            "SELECT session_id, created_at FROM trace_events "
            "WHERE scale = ? AND ref_type = ? AND metadata LIKE ? LIMIT 1",
            (scale, ref_type, '%' + substring + '%')).fetchone()
        return {'session_id': row[0], 'created_at': row[1]} if row else None

    def get_session_turns(self, session_id: str, limit: int = 20,
                          around_timestamp: str = None,
                          before: int = None, after: int = None) -> List[Dict[str, Any]]:
        """Get chronological turns for a session from S0 + S1 traces.

        Returns same shape as encoding_agent._gather_messages():
        [{role, content, signal_type, timestamp, surface_output, recalled_raw}]

        Groups S0 K (user_message) + S0 delta (assistant_message) per chain,
        cross-references S1 delta (additionalContext) via recall_chain in metadata.

        Args:
            session_id: Full session UUID
            limit: Max turns to return (most recent if no around_timestamp)
            around_timestamp: ISO timestamp to center the window on.
                If provided, returns `before` turns before + `after` turns after
                this timestamp instead of the most recent `limit`.
            before: Turns before around_timestamp (default 10)
            after: Turns after around_timestamp (default 5)
        """
        # Get S0 events for this session, chronologically.
        # v29: select `id` (8-char hex trace_event.id) so callers can render
        # [trace:<hex>] markers — the encoder copies these into source_refs.
        # Conversation window = the conversational ref_types defined by the S0
        # turn-classification contract (single source of truth for what the
        # encoder reads — see trace_contract.CONVERSATIONAL_REF_TYPES).
        from .trace_contract import CONVERSATIONAL_REF_TYPES
        _refs = ",".join("?" * len(CONVERSATIONAL_REF_TYPES))
        rows = self.conn.execute(
            "SELECT id, chain_id, event_type, ref_type, summary, metadata, created_at "
            "FROM trace_events WHERE scale = 's0' AND session_id = ? "
            "AND event_type IN ('K', 'delta') AND ref_type IN (%s) "
            "ORDER BY created_at ASC" % _refs,
            (session_id, *CONVERSATIONAL_REF_TYPES)).fetchall()

        # Group by chain (each chain = one stop = user+assistant pair)
        chains = {}
        for r in rows:
            trace_id = r[0]
            chain_id = r[1]
            if chain_id not in chains:
                chains[chain_id] = {}
            meta = self._decode_metadata(r[5])
            # Content lives in metadata (full), summary is truncated for display
            content = meta.get('content', '') or r[4] or ''
            if r[3] == 'user_message':
                chains[chain_id]['user'] = {
                    'trace_id': trace_id,
                    'content': content,
                    'timestamp': r[6],
                    'recall_chain': meta.get('recall_chain', ''),
                }
            elif r[3] == 'assistant_message':
                chains[chain_id]['assistant'] = {
                    'trace_id': trace_id,
                    'content': content,
                    'timestamp': r[6],
                }

        # Cross-reference S1 delta (additionalContext) for judge_output
        recall_chains = set()
        for data in chains.values():
            rc = data.get('user', {}).get('recall_chain', '')
            if rc:
                recall_chains.add(rc)

        judge_outputs = {}
        if recall_chains:
            placeholders = ','.join('?' for _ in recall_chains)
            s1_rows = self.conn.execute(
                "SELECT chain_id, metadata FROM trace_events "
                "WHERE scale = 's1' AND event_type = 'delta' AND ref_type = 'additionalContext' "
                "AND chain_id IN (%s)" % placeholders,
                list(recall_chains)).fetchall()
            for r in s1_rows:
                judge_outputs[r[0]] = self._decode_metadata(r[1]).get('content', '')

        # Build result in encoding_agent._gather_messages() shape
        turns = []
        for chain_id in sorted(chains.keys(), key=lambda c: chains[c].get('user', {}).get('timestamp', '')):
            data = chains[chain_id]
            if 'user' in data:
                recall_chain = data['user'].get('recall_chain', '')
                turns.append({
                    'role': 'user',
                    'trace_id': data['user'].get('trace_id'),
                    'content': data['user']['content'],
                    'timestamp': data['user']['timestamp'],
                    'signal': None,
                    'judge_output': judge_outputs.get(recall_chain, ''),
                    'recalled_raw': None,  # Not stored in S0 traces (available in S1 O)
                })
            if 'assistant' in data:
                turns.append({
                    'role': 'assistant',
                    'trace_id': data['assistant'].get('trace_id'),
                    'content': data['assistant']['content'],
                    'timestamp': data['assistant']['timestamp'],
                    'signal': None,
                    'judge_output': None,
                    'recalled_raw': None,
                })

        # Apply windowing
        if around_timestamp:
            # Find the turn closest to around_timestamp, then take window
            _before = before if before is not None else 10
            _after = after if after is not None else 5
            center_idx = 0
            for i, t in enumerate(turns):
                if t.get('timestamp', '') <= around_timestamp:
                    center_idx = i
            start = max(0, center_idx - _before * 2)  # ×2 because user+assistant = 2 turns per exchange
            end = min(len(turns), center_idx + _after * 2 + 1)
            turns = turns[start:end]
        elif len(turns) > limit:
            # Most recent turns (default behavior)
            turns = turns[-limit:]

        return turns

    # --- Embeddings (v27: episodic references) ---

    def store_embeddings(self, rows: List[tuple], model: str) -> int:
        """Upsert per-trace embeddings (one row per unique trace_id).

        Args:
            rows: iterable of (trace_id, vector_blob, text). Rows with
                  vector=None are skipped.
            model: embedder model tag stored on each row.

        Returns: count of rows actually written.

        INSERT OR REPLACE handles both new inserts and updates to the
        same trace_id (e.g., re-embed after rendering change).
        """
        now = iso_now()
        prepared = []
        for trace_id, vector, text in rows:
            if vector is None or trace_id is None:
                continue
            prepared.append((trace_id, vector, (text or '')[:500], model, now))
        if not prepared:
            return 0
        self.conn.executemany(
            'INSERT OR REPLACE INTO trace_embeddings '
            '(trace_id, vector, text, model, created_at) '
            'VALUES (?, ?, ?, ?, ?)',
            prepared)
        commit_unless_batched(self.conn)
        return len(prepared)

    def get_embeddings(self, trace_ids: List[str]) -> Dict[str, bytes]:
        """Batch fetch trace embeddings by id (v29: 8-char hex). Missing ids
        absent from result. Rejects int input loudly per v29 contract."""
        if not trace_ids:
            return {}
        for tid in trace_ids:
            if not isinstance(tid, str):
                raise ValueError(
                    "get_embeddings: trace_ids must be strings, got "
                    "%s (%r). v29 trace ids are 8-char hex." % (
                        type(tid).__name__, tid))
        placeholders = ','.join('?' * len(trace_ids))
        rows = self.conn.execute(
            'SELECT trace_id, vector FROM trace_embeddings '
            'WHERE trace_id IN (%s)' % placeholders,
            list(trace_ids)).fetchall()
        return {r[0]: r[1] for r in rows}

    def find_unembedded(self, limit: int, scales: List[str],
                        ref_types: List[str],
                        since: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find recent trace events with no embedding row yet.

        Pull-reconciliation primitive: the worker calls this every tick,
        gets up to `limit` traces in newest-first order, embeds them.
        No queue state; restart-safe by construction.

        Args:
            limit: max rows to return.
            scales: required scale filter (e.g., ['s0']). Empty raises ValueError.
            ref_types: required ref_type filter (e.g., ['user_message',
                       'assistant_message', 'tool_result']). Empty raises
                       ValueError.
            since: optional ISO timestamp lower bound on created_at.
                   Worker uses this to skip historical traces that
                   pre-date identity stamping — they'd render with
                   OPERATOR/ANCHOR sentinels and pollute the embedding
                   neighborhood that decision 19 keeps concrete.

        Returns rows with id/scale/event_type/ref_type/summary/metadata/
        session_id/created_at fields. Caller renders to text and embeds.
        """
        if not scales:
            raise ValueError("find_unembedded: scales required")
        if not ref_types:
            raise ValueError("find_unembedded: ref_types required")
        scale_ph = ','.join('?' * len(scales))
        ref_ph = ','.join('?' * len(ref_types))
        params: List[Any] = list(scales) + list(ref_types)
        since_clause = ''
        if since:
            since_clause = ' AND te.created_at > ? '
            params.append(since)
        params.append(limit)
        rows = self.conn.execute(
            'SELECT te.id, te.scale, te.event_type, te.ref_type, '
            '       te.summary, te.metadata, te.session_id, te.created_at '
            'FROM trace_events te '
            'LEFT JOIN trace_embeddings tem ON tem.trace_id = te.id '
            'WHERE tem.trace_id IS NULL '
            '  AND te.scale IN (%s) '
            '  AND te.ref_type IN (%s) '
            '  %s'
            'ORDER BY te.created_at DESC '
            'LIMIT ?' % (scale_ph, ref_ph, since_clause),
            params).fetchall()
        results = []
        for r in rows:
            results.append({
                'id': r[0], 'scale': r[1], 'event_type': r[2],
                'ref_type': r[3], 'summary': r[4] or '',
                'metadata': self._decode_metadata(r[5]),
                'session_id': r[6] or '',
                'created_at': r[7]})
        return results


class SessionStateDAL:
    """Access layer for session_state table in brain_logs.db.

    First-class session-scoped data: fatigue, journal, context, counters.
    Keyed by (session_id, key, node_id). Replaces scattered in-memory
    dicts and brain_meta config keys.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get(self, session_id: str, key: str, node_id: str = '') -> Optional[str]:
        """Get a single session state value."""
        row = self.conn.execute(
            "SELECT value FROM session_state WHERE session_id = ? AND key = ? AND node_id = ?",
            (session_id, key, node_id)).fetchone()
        return row[0] if row else None

    def set(self, session_id: str, key: str, value: str, node_id: str = ''):
        """Set a session state value (upsert)."""
        from datetime import datetime, timezone
        ts = iso_now()
        self.conn.execute(
            """INSERT INTO session_state (session_id, key, node_id, value, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(session_id, key, node_id)
               DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
            (session_id, key, node_id, value, ts))
        commit_unless_batched(self.conn)

    # --- session-context lifecycle (the rows keyed '_session_context') ---

    def ensure_default(self, session_id: str, key: str, value: str,
                       node_id: str = '') -> None:
        """Insert a default row only if absent (INSERT OR IGNORE).

        Distinct from set(): set() overwrites via upsert; this preserves an
        existing row, so a racing thread's already-mutated state is never
        clobbered on first touch.
        """
        self.conn.execute(
            'INSERT OR IGNORE INTO session_state '
            '(session_id, key, node_id, value, updated_at) VALUES (?, ?, ?, ?, ?)',
            (session_id, key, node_id, value, iso_now()))
        commit_unless_batched(self.conn)

    def recently_updated(self, key: str, cutoff_iso: str,
                         exclude_session: str = '', limit: int = 5) -> list:
        """Sessions whose `key` row updated since `cutoff_iso`, newest first.

        Returns [{'session_id', 'updated_at'}]. The caller computes the cutoff
        (wall-clock vs conversation-time is the caller's policy, not the DAL's).
        """
        rows = self.conn.execute(
            "SELECT session_id, updated_at FROM session_state "
            "WHERE key = ? AND updated_at > ? AND session_id != ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (key, cutoff_iso, exclude_session or '', limit)).fetchall()
        return [{'session_id': r[0], 'updated_at': r[1]} for r in rows]

    def sessions_by_message_count(self, key: str, min_messages: int,
                                  limit: int = 5) -> list:
        """Session ids whose `key` row JSON has message_count >= min_messages,
        newest first. Returns [session_id].
        """
        rows = self.conn.execute(
            "SELECT session_id FROM session_state WHERE key = ? "
            "AND CAST(COALESCE(json_extract(value, '$.message_count'), 0) AS INTEGER) >= ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (key, min_messages, limit)).fetchall()
        return [r[0] for r in rows]

    def increment(self, session_id: str, key: str, node_id: str) -> int:
        """Increment a counter value. Returns new count."""
        from datetime import datetime, timezone
        ts = iso_now()
        self.conn.execute(
            """INSERT INTO session_state (session_id, key, node_id, value, updated_at)
               VALUES (?, ?, ?, '1', ?)
               ON CONFLICT(session_id, key, node_id)
               DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT),
                            updated_at = excluded.updated_at""",
            (session_id, key, node_id, ts))
        commit_unless_batched(self.conn)
        row = self.conn.execute(
            "SELECT value FROM session_state WHERE session_id = ? AND key = ? AND node_id = ?",
            (session_id, key, node_id)).fetchone()
        return int(row[0]) if row else 1

    def load_all(self, session_id: str, key: str) -> Dict[str, str]:
        """Load all values for a key in a session. Returns {node_id: value}."""
        rows = self.conn.execute(
            "SELECT node_id, value FROM session_state WHERE session_id = ? AND key = ?",
            (session_id, key)).fetchall()
        return {r[0]: r[1] for r in rows}

    def load_fatigue(self, session_id: str) -> Dict[str, int]:
        """Load fatigue counts for a session. Returns {node_id: count}.

        Supports both formats:
        - New: single JSON blob row (node_id='', value=JSON)
        - Legacy: per-node rows (node_id=X, value=count)
        """
        # Try new JSON blob format first
        row = self.conn.execute(
            "SELECT value FROM session_state "
            "WHERE session_id = ? AND key = 'fatigue' AND node_id = ''",
            (session_id,)).fetchone()
        if row and row[0]:
            try:
                return {k: int(v) for k, v in json.loads(row[0]).items()}
            except (json.JSONDecodeError, ValueError, AttributeError):
                pass

        # Fall back to legacy per-node rows
        rows = self.conn.execute(
            "SELECT node_id, CAST(value AS INTEGER) FROM session_state "
            "WHERE session_id = ? AND key = 'fatigue' AND node_id != ''",
            (session_id,)).fetchall()
        return {r[0]: r[1] for r in rows}

    def save_fatigue(self, session_id: str, fatigue: Dict[str, int]):
        """Save fatigue dict as a single JSON blob. Replaces per-node rows."""
        from datetime import datetime, timezone
        ts = iso_now()
        self.conn.execute(
            """INSERT INTO session_state (session_id, key, node_id, value, updated_at)
               VALUES (?, 'fatigue', '', ?, ?)
               ON CONFLICT(session_id, key, node_id)
               DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
            (session_id, json.dumps(fatigue), ts))
        # Clean up legacy per-node rows for this session
        self.conn.execute(
            "DELETE FROM session_state WHERE session_id = ? AND key = 'fatigue' AND node_id != ''",
            (session_id,))
        commit_unless_batched(self.conn)

    def cleanup_old_sessions(self, keep_last_n: int = 5):
        """Remove session_state for old sessions, keeping the N most recent."""
        self.conn.execute(
            """DELETE FROM session_state WHERE session_id NOT IN (
                SELECT DISTINCT session_id FROM session_state
                ORDER BY updated_at DESC LIMIT ?
            )""", (keep_last_n,))
        commit_unless_batched(self.conn)


class BrainMetaDAL:
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
        now = iso_now()
        self.conn.execute(
            'INSERT OR REPLACE INTO brain_meta (key, value, updated_at) VALUES (?, ?, ?)',
            (key, str(value), now)
        )
        commit_unless_batched(self.conn)

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

    def increment(self, key: str) -> int:
        """Increment a counter and return new value."""
        current = self.get(key, "0")
        try:
            new_val = int(current) + 1
        except (ValueError, TypeError):
            new_val = 1
        self.set(key, str(new_val))
        return new_val


class NodeDAL:
    """Access layer for brain.db nodes table.

    ALL node SQL lives here. When we move to in-memory graph,
    swap this implementation — nothing else changes.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # --- Reads ---

    def get_naked_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node row by ID. Returns all columns as a dict.

        This is the bare DB row — no metadata, no corrections, no connections.
        For the full assembled node, use brain.get_node().

        Uses PRAGMA table_info to get column names dynamically,
        then SELECT * to get all values. New columns automatically included.
        Boolean fields (locked, archived, critical) coerced to Python bool.
        """
        row = self.conn.execute(
            'SELECT * FROM nodes WHERE id = ?', (node_id,)
        ).fetchone()
        if not row:
            return None

        # Get column names from cursor description
        cols = [desc[0] for desc in self.conn.execute(
            'SELECT * FROM nodes LIMIT 0').description]
        d = dict(zip(cols, row))

        # Coerce booleans (SQLite stores as 0/1 or NULL)
        for bool_field in ('locked', 'archived', 'critical'):
            d[bool_field] = d.get(bool_field) == 1
        # Defaults for nullable fields
        d['emotion'] = d.get('emotion') or 0
        d['emotion_label'] = d.get('emotion_label') or 'neutral'
        return d

    def get_bulk(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Bulk-fetch naked node rows. Returns {node_id: row_dict}.

        Same shape as get_naked_node() per row. Missing/invalid ids are
        silently omitted from the result. Used by callers that need many
        nodes in one query (recall enrichment, correction enrichment,
        rich-node assembly) — replaces the N+1 get_naked_node loop.
        """
        if not node_ids:
            return {}
        ph = ','.join('?' * len(node_ids))
        cols = [desc[0] for desc in self.conn.execute(
            'SELECT * FROM nodes LIMIT 0').description]
        rows = self.conn.execute(
            'SELECT * FROM nodes WHERE id IN (%s)' % ph,
            list(node_ids)).fetchall()
        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            d = dict(zip(cols, row))
            for bool_field in ('locked', 'archived', 'critical'):
                d[bool_field] = d.get(bool_field) == 1
            d['emotion'] = d.get('emotion') or 0
            d['emotion_label'] = d.get('emotion_label') or 'neutral'
            out[d['id']] = d
        return out

    def resolve_id(self, prefix: str) -> Optional[str]:
        """Resolve a short ID prefix (e.g. 8-char) to a full node ID."""
        if not prefix:
            return None
        row = self.conn.execute(
            'SELECT id FROM nodes WHERE id LIKE ?', (prefix + '%',)
        ).fetchone()
        return row[0] if row else None

    def get_title(self, node_id: str) -> Optional[str]:
        """Get just the title of a node. Accepts full ID or prefix."""
        row = self.conn.execute(
            'SELECT title FROM nodes WHERE id LIKE ?', (node_id + '%',)
        ).fetchone()
        return row[0] if row else None

    def exists(self, node_id: str) -> bool:
        """Check if a node exists."""
        row = self.conn.execute(
            'SELECT id FROM nodes WHERE id = ?', (node_id,)
        ).fetchone()
        return row is not None

    def archived_subset(self, node_ids) -> set:
        """Return the subset of `node_ids` that are archived.

        Single source for liveness checks (surface selection gate,
        Hebbian drain gate). Exact-id match — no prefix resolution;
        unknown ids are simply absent from the result. Empty input
        returns an empty set (no `IN ()` SQL).
        """
        ids = [nid for nid in node_ids if nid]
        if not ids:
            return set()
        ph = ','.join('?' * len(ids))
        rows = self.conn.execute(
            'SELECT id FROM nodes WHERE id IN (%s) AND archived = 1' % ph,
            ids).fetchall()
        return {r[0] for r in rows}

    # --- Survivor-pointer resolution (read-only) ---

    # Metadata key recording where an archived node's content survived to.
    # Today this is the ONLY survivor source: absorb/consolidation stamps it
    # on the absorbed node before archiving. A future `absorbed_into` graph
    # edge would be a SECOND source — fold it into `_survivor_pointers_bulk`
    # below (read the edge, prefer/merge with the kv value) without touching
    # `resolve_live`'s walk.
    _SURVIVOR_META_KEY = '_sys_archived_survivor_id'

    def _live_status(self, node_id: str) -> Optional[str]:
        """Liveness of a single node: 'live', 'archived', or None if missing.

        Single-id probe (used by tests and one-off callers). `resolve_live`
        uses the batched `_live_status_bulk` on its hot path.
        """
        row = self.conn.execute(
            'SELECT archived FROM nodes WHERE id = ?', (node_id,)).fetchone()
        if row is None:
            return None
        return 'archived' if row[0] == 1 else 'live'

    def _live_status_bulk(self, node_ids) -> Dict[str, str]:
        """Batched `_live_status`: {id: 'live'|'archived'} for the ids that
        exist in `nodes`. Missing ids are absent from the result, which the
        resolve_live walk treats as orphan. One query for the whole frontier —
        no per-id probing on the hot path.
        """
        ids = [n for n in node_ids if n]
        if not ids:
            return {}
        ph = ','.join('?' * len(ids))
        rows = self.conn.execute(
            'SELECT id, archived FROM nodes WHERE id IN (%s)' % ph, ids
        ).fetchall()
        return {r[0]: ('archived' if r[1] == 1 else 'live') for r in rows}

    def _survivor_pointers_bulk(self, node_ids) -> Dict[str, str]:
        """Batched survivor-pointer lookup: {archived_id: survivor_id} for the
        ids that carry a pointer (others simply absent).

        SEAM: reads `_sys_archived_survivor_id` from node_metadata_kv today via
        the canonical batch getter (one IN-query, PK-covered on (node_id, key)).
        When the `absorbed_into` edge becomes the survivor source, swap the body
        HERE — `resolve_live`'s walk never changes.
        """
        ids = [n for n in node_ids if n]
        if not ids:
            return {}
        from .dal_metadata import MetadataDAL
        got = MetadataDAL(self.conn).get_fields_bulk(
            ids, [self._SURVIVOR_META_KEY])
        return {nid: kv[self._SURVIVOR_META_KEY]
                for nid, kv in got.items()
                if kv.get(self._SURVIVOR_META_KEY)}

    def resolve_live(self, ids, *, on_orphan: str = 'drop',
                     max_hops: int = 8) -> Dict[str, Any]:
        """Resolve a set of node ids to their live survivors. READ-ONLY.

        For each input id: a LIVE node passes through unchanged; an ARCHIVED
        node is followed forward along its survivor pointer (see
        `_survivor_pointers_bulk`) until a live terminal, an orphan (missing
        node or no pointer), a cycle, or `max_hops` redirects. Many inputs
        collapsing to one survivor are deduped, first-seen order preserved.

        Batched walk: all in-flight inputs advance in lockstep, so the DB is
        hit twice per chain LEVEL (liveness + survivor lookup), not twice per
        id per hop. Cost is O(max chain depth) queries, independent of the
        input count — the hot-path shape the 6 history→node sites in
        docs/TRACE-NODE-RESOLUTION.md need.

        Returns ids, not hydrated nodes — callers hydrate via get_node():
            {
              'live':       [live ids, deduped, order-preserved],
              'redirected': {input_id: survivor_id},   # only redirected inputs
              'orphans':    [input ids with no live terminal],
            }

        `on_orphan='drop'` (default) returns `orphans: []`; `'mark'` returns
        the orphan input ids in `orphans`. Either way orphans never appear in
        `live`.
        """
        inputs = [i for i in (ids or []) if i]
        if not inputs:
            return {'live': [], 'redirected': {}, 'orphans': []}

        pos = {i: i for i in inputs}        # input_id -> current node in its walk
        hops = {i: 0 for i in inputs}       # redirects taken (== round reached)
        visited = {i: {i} for i in inputs}  # per-input cycle guard
        terminal: Dict[str, str] = {}       # input_id -> live terminal id
        orphaned: set = set()
        pending = list(inputs)

        while pending:
            # Round liveness: one query for every distinct current position.
            status = self._live_status_bulk({pos[i] for i in pending})
            advancers = []                  # inputs sitting on an archived node
            for i in pending:
                st = status.get(pos[i])     # None => id not in nodes (missing)
                if st == 'live':
                    terminal[i] = pos[i]
                elif st is None:
                    orphaned.add(i)         # missing node
                elif hops[i] >= max_hops:
                    orphaned.add(i)         # hop budget exhausted
                else:
                    advancers.append(i)     # archived, may still redirect
            if not advancers:
                break
            # Round survivor lookup: one query for the archived frontier.
            survivors = self._survivor_pointers_bulk(
                {pos[i] for i in advancers})
            next_pending = []
            for i in advancers:
                surv = survivors.get(pos[i])
                if not surv:
                    orphaned.add(i)         # archived, no pointer
                elif surv in visited[i]:
                    orphaned.add(i)         # cycle
                else:
                    visited[i].add(surv)
                    pos[i] = surv
                    hops[i] += 1
                    next_pending.append(i)
            pending = next_pending

        # Assemble in first-seen input order; dedup live; mark redirects.
        live_out: List[str] = []
        seen_live: set = set()
        redirected: Dict[str, str] = {}
        for i in inputs:
            t = terminal.get(i)
            if t is None:
                continue
            if hops[i] > 0:
                redirected[i] = t
            if t not in seen_live:
                seen_live.add(t)
                live_out.append(t)

        return {
            'live': live_out,
            'redirected': redirected,
            'orphans': [i for i in inputs if i in orphaned]
                       if on_orphan == 'mark' else [],
        }

    def count(self, archived: bool = False) -> int:
        """Count nodes, optionally excluding archived."""
        if archived:
            row = self.conn.execute('SELECT COUNT(*) FROM nodes').fetchone()
        else:
            row = self.conn.execute(
                'SELECT COUNT(*) FROM nodes WHERE archived = 0'
            ).fetchone()
        return row[0] if row else 0

    def count_locked(self, include_archived: bool = False) -> int:
        """Count locked nodes. Excludes archived by default (the
        identity-meaningful count); pass include_archived=True for the raw
        lock count regardless of archive state.

        Default (non-archived) matches all current call sites — brain.py,
        brain_assembly.py, and daemon_server's status count (migrated 2026-05-30,
        which intentionally drops archived-locked nodes from the status total).
        Pass include_archived=True for the raw all-state lock count."""
        sql = 'SELECT COUNT(*) FROM nodes WHERE locked = 1'
        if not include_archived:
            sql += ' AND archived = 0'
        row = self.conn.execute(sql).fetchone()
        return row[0] if row else 0

    def count_by_type(self, node_type: str) -> int:
        """Count nodes of a specific type."""
        row = self.conn.execute(
            'SELECT COUNT(*) FROM nodes WHERE type = ? AND archived = 0',
            (node_type,)
        ).fetchone()
        return row[0] if row else 0

    def filter_nodes(self, field: str, include=None, exclude=None,
                     lt=None, gt=None, limit: int = 50,
                     sort_by: str = 'created_at', sort_order: str = 'desc'):
        """Filter nodes by any structural field.

        Args:
            field: column name — must be in STRUCTURAL_FIELDS whitelist.
            include: list of values to match (exact, IN).
            exclude: list of values to exclude (exact, NOT IN).
            lt/gt: numeric comparisons for float/int fields.
            limit: max results (capped at 200).
            sort_by: column to sort by.
            sort_order: 'asc' or 'desc'.

        Returns: dict with 'nodes' list and 'total_count'.
        """
        from .contract import STRUCTURAL_FIELDS

        # Whitelist field
        if field not in STRUCTURAL_FIELDS:
            return {"error": "Unknown field '%s'. Valid: %s" % (
                field, ', '.join(sorted(STRUCTURAL_FIELDS.keys())))}

        # Whitelist sort_by
        allowed_sort = {'created_at', 'confidence', 'access_count', 'title', 'type',
                        'updated_at', 'last_accessed', 'revised_at'}
        if sort_by not in allowed_sort:
            sort_by = 'created_at'
        if sort_order not in ('asc', 'desc'):
            sort_order = 'desc'

        limit = min(max(limit, 1), 200)

        # Build WHERE clauses
        conditions = ['archived = 0']
        params = []

        if include and exclude:
            return {"error": "Cannot use both include and exclude"}

        if include:
            placeholders = ','.join('?' for _ in include)
            conditions.append('%s IN (%s)' % (field, placeholders))
            params.extend(include)
        elif exclude:
            placeholders = ','.join('?' for _ in exclude)
            conditions.append('%s NOT IN (%s)' % (field, placeholders))
            params.extend(exclude)

        if lt is not None:
            conditions.append('%s < ?' % field)
            params.append(lt)
        if gt is not None:
            conditions.append('%s > ?' % field)
            params.append(gt)

        where = ' AND '.join(conditions)

        # Count total matches
        count_row = self.conn.execute(
            'SELECT COUNT(*) FROM nodes WHERE %s' % where, params
        ).fetchone()
        total_count = count_row[0] if count_row else 0

        # Fetch results
        sql = 'SELECT id, title, type, confidence, created_at, %s FROM nodes WHERE %s ORDER BY %s %s LIMIT ?' % (
            field, where, sort_by, sort_order)
        rows = self.conn.execute(sql, params + [limit]).fetchall()

        nodes = []
        for r in rows:
            node = {'id': r[0], 'title': r[1], 'type': r[2],
                    'confidence': r[3], 'created_at': r[4]}
            if field not in ('id', 'title', 'type', 'confidence', 'created_at'):
                node[field] = r[5]
            nodes.append(node)

        return {"nodes": nodes, "total_count": total_count}

    # --- Writes ---

    def delete(self, node_id: str) -> None:
        """Hard delete a node (use archive() for soft delete)."""
        self.conn.execute('DELETE FROM nodes WHERE id = ?', (node_id,))
        commit_unless_batched(self.conn)

    # get_metadata removed 2026-04-13 — old node_metadata table dropped, use MetadataDAL (KV).
    # NodeDAL write-helpers (update_field/update_confidence/set_critical/unlock/
    # update_type/append_content/set_evolution_status/mark_accessed) removed
    # 2026-06-26 — dead since the revise()-is-the-only-content-path invariant
    # (b2f97fb1); content/title/confidence/critical/locked go through
    # brain_remember's revise, access goes through recall_write_queue's drain.
    # delete_for_node removed 2026-05-30 (DAL cleanup Phase 0) — was a dup of
    # VectorDAL.delete_for_node (node_enrichments is the vector table, owned by
    # VectorDAL); had zero callers.


class TfIdfDAL:
    """Access layer for node_vectors and doc_freq tables (TF-IDF index)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_doc_freq(self, term: str) -> int:
        """Get document frequency for a term."""
        row = self.conn.execute(
            'SELECT count FROM doc_freq WHERE term = ?', (term,)
        ).fetchone()
        return row[0] if row else 0

    def get_node_terms(self, node_id: str) -> Dict[str, float]:
        """Get TF vector for a node. Returns {term: tf_value}."""
        rows = self.conn.execute(
            'SELECT term, tf FROM node_vectors WHERE node_id = ?', (node_id,)
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_tf_vectors_for(self, terms: List[str],
                           node_ids: List[str]) -> List[tuple]:
        """TF values for `terms` restricted to `node_ids`. Returns raw
        (node_id, term, tf) rows. Used by recall's batch TF-IDF scoring
        (term IN ... AND node_id IN ...) — the one term+node-filtered read.
        """
        if not terms or not node_ids:
            return []
        term_ph = ','.join('?' * len(terms))
        node_ph = ','.join('?' * len(node_ids))
        return self.conn.execute(
            'SELECT node_id, term, tf FROM node_vectors '
            'WHERE term IN (%s) AND node_id IN (%s)' % (term_ph, node_ph),
            list(terms) + list(node_ids)).fetchall()

    def get_nodes_matching_terms(self, terms: List[str]) -> List[str]:
        """Find node IDs that have any of the given terms."""
        if not terms:
            return []
        placeholders = ','.join('?' * len(terms))
        rows = self.conn.execute(
            'SELECT DISTINCT nv.node_id FROM node_vectors nv '
            'JOIN nodes n ON n.id = nv.node_id '
            'WHERE nv.term IN (%s) AND n.archived = 0' % placeholders,
            terms
        ).fetchall()
        return [r[0] for r in rows]

    def store_tf_vector(self, node_id: str, tf_map: Dict[str, float]) -> None:
        """Store TF vector for a node, replacing any existing."""
        self.conn.execute(
            'DELETE FROM node_vectors WHERE node_id = ?', (node_id,)
        )
        for term, tf_val in tf_map.items():
            self.conn.execute(
                'INSERT OR REPLACE INTO node_vectors (node_id, term, tf) '
                'VALUES (?, ?, ?)', (node_id, term, tf_val)
            )
            # Update doc frequency
            self.conn.execute(
                'INSERT INTO doc_freq (term, count) VALUES (?, 1) '
                'ON CONFLICT(term) DO UPDATE SET count = count + 1',
                (term,)
            )
        commit_unless_batched(self.conn)

    def delete_for_node(self, node_id: str) -> None:
        """Delete TF-IDF data for a node."""
        self.conn.execute(
            'DELETE FROM node_vectors WHERE node_id = ?', (node_id,)
        )
        commit_unless_batched(self.conn)

    def clear_all(self) -> None:
        """Clear entire TF-IDF index (for reindex)."""
        self.conn.execute('DELETE FROM node_vectors')
        self.conn.execute('DELETE FROM doc_freq')
        commit_unless_batched(self.conn)

    def get_total_docs(self) -> int:
        """Count total documents with TF-IDF vectors."""
        row = self.conn.execute(
            'SELECT COUNT(DISTINCT node_id) FROM node_vectors'
        ).fetchone()
        return row[0] if row else 0


class Fts5DAL:
    """Access layer for nodes_fts (FTS5 full-text search).

    FTS5 provides word-level search alongside embedding similarity.
    Different signal: embeddings match meaning, FTS5 matches words.
    Both feed into the surfacer which decides relevance.
    """

    def __init__(self, conn):
        self.conn = conn

    def search(self, query: str, limit: int = 30,
               include_archived: bool = False) -> List[str]:
        """Full-text search. Returns node_ids ranked by BM25 relevance.

        Title matches weighted 10x over content.
        bm25() column weights: (node_id=0, title=10, content=1)

        Excludes archived nodes by default. FTS5 (nodes_fts) is a separate
        virtual table with no `archived` column — historically the ONE recall
        candidate lane that didn't filter liveness, so a lingering FTS entry for
        an archived node surfaced it in recall (the dead-node leak; see
        docs/TRACE-NODE-RESOLUTION.md). JOINing `nodes` and filtering
        `archived = 0` makes the flag the single source of truth at READ time —
        so the FTS-delete on archive becomes hygiene, not a correctness
        requirement, and `LIMIT` now returns live hits instead of spending
        slots on dead ones. The survivor-redirect reader passes
        include_archived=True to SEE an archived hit and resolve_live it to its
        living survivor rather than drop it.

        Note: prior to schema v28 there was a 4th column `keywords` carrying
        an auto-extracted tokenizer dump. The column was dropped because the
        extractor produced near-duplicate noise. Porter stemming on
        title+content provides the same lexical signal without the noise.
        """
        safe_query = self._sanitize_query(query)
        if not safe_query:
            return []
        try:
            if include_archived:
                sql = """SELECT node_id FROM nodes_fts
                         WHERE nodes_fts MATCH ?
                         ORDER BY bm25(nodes_fts, 0, 10.0, 1.0)
                         LIMIT ?"""
            else:
                sql = """SELECT nodes_fts.node_id FROM nodes_fts
                         JOIN nodes ON nodes.id = nodes_fts.node_id
                         WHERE nodes_fts MATCH ? AND nodes.archived = 0
                         ORDER BY bm25(nodes_fts, 0, 10.0, 1.0)
                         LIMIT ?"""
            rows = self.conn.execute(sql, (safe_query, limit)).fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            # Loud-by-default: a malformed query or corrupt FTS5 index must not
            # look identical to "no matches" — FTS5 is one of two recall signals.
            # Log before degrading (matches add_relation's de-silenced pattern).
            import sys as _sys
            print('[Fts5DAL.search] FTS5 query failed (%r): %s'
                  % (safe_query, e), file=_sys.stderr)
            return []

    def upsert(self, node_id: str, title: str, content: str, _legacy_keywords: str = ''):
        """Insert or update a node in the FTS5 index.

        The 4th positional arg is kept for back-compat with callers that
        still pass an (ignored) keywords string. After all callers update,
        the parameter can be dropped.
        """
        self.delete(node_id)
        self.conn.execute(
            "INSERT INTO nodes_fts (node_id, title, content) VALUES (?, ?, ?)",
            (node_id, title, content or ''))

    def delete(self, node_id: str):
        """Remove a node from FTS5 index."""
        try:
            self.conn.execute(
                "DELETE FROM nodes_fts WHERE node_id = ?", (node_id,))
        except Exception as e:
            # Loud-by-default: a failed delete leaves a stale index entry.
            # Lower stakes than search (self-heals on next upsert) but log
            # rather than swallow silently.
            import sys as _sys
            print('[Fts5DAL.delete] FTS5 delete failed for %s: %s'
                  % (node_id, e), file=_sys.stderr)

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Sanitize query for FTS5 MATCH syntax.

        Wraps each meaningful term in quotes, joins with OR.
        Caps at 8 terms to prevent explosion.
        """
        from .brain_constants import TFIDF_STOP_WORDS
        words = query.strip().split()
        terms = [w for w in words if w.lower() not in TFIDF_STOP_WORDS and len(w) > 1]
        if not terms:
            terms = [w for w in words if len(w) > 1]
        if not terms:
            return ''
        # Quote each term, join with OR (any match, not all)
        return ' OR '.join('"%s"' % t.replace('"', '') for t in terms[:8])


# ═══════════════════════════════════════════════════════════════
# GRAPH QUERY CONTRACT
# ═══════════════════════════════════════════════════════════════
# Every edge-reading method in GraphDAL conforms to this shape and
# these defaults. Centralizes what the rest of the code can assume.
# When we change what "an edge" means, we change it HERE once, and
# every consumer inherits the update.

# Canonical edge-row shape returned by GraphDAL reads. Node-centric:
# each row is a (owner → neighbor) relationship from the queried
# node's perspective. Matches get_neighbors output.
EDGE_ROW_SHAPE = {
    # Neighbor node fields (the node on the OTHER side of the edge)
    'id':                 'str  — neighbor node_id',
    'type':               'str  — neighbor node type',
    'title':              'str  — neighbor title',
    'content_summary':    'str  — neighbor content summary (may be None)',
    'confidence':         'float',
    'locked':             'int (0|1)',
    'created_at':         'str ISO',
    'revised_at':         'str ISO or None',
    # Edge metadata
    'edge_id':            'str — stable pair hash',
    'relation':           'str — typed relation name',
    'edge_description':   'str — relation description',
    'weight':             'float — edge-aggregate weight',
    'direction':          "str — 'outgoing' | 'incoming' from queried node",
    # Optional (present on richer methods)
    'last_accessed':      'str ISO',
    'access_count':       'int',
    'emotion':            'float',
    'emotion_label':      'str',
    'last_strengthened':  'str ISO',
    'co_access_count':    'int',
    'content_preview':    'str — substr of content when caller requests',
}

# Relations considered noise for semantic edge queries. Default-excluded
# by callers that want knowledge edges only. Override per-call when you
# need co_accessed (fatigue) or emergent_bridge (auto-links).
# community_member is NOT in this default — it's real thematic context,
# just not migrated by consolidation (see ABSORB_EXCLUDED_RELATIONS).
DEFAULT_EXCLUDED_RELATIONS = frozenset(['co_accessed', 'emergent_bridge'])
# Minimum edge-description length to feed the edge_context embedding. Single
# source of truth shared by GraphDAL.get_edge_descriptions_for (the text
# producer) and VectorDAL.find_missing (the backfill candidate filter) — the
# two MUST agree, or find_missing queues edgeless/short-desc nodes that yield
# no text, and they starve the edged nodes out of the backfill batch forever.
EDGE_CONTEXT_MIN_DESC_LENGTH = 10

# Relations absorb() must NOT migrate to the survivor. Community placement
# is the community unit's judged decision (affinity gate ≥0.25 + encoder
# accept/reject + drift detection re-evaluation) — a merge inheriting the
# absorbed node's membership would bypass all three. The absorbed node is
# archived, so its membership edge dies with it (dangling-edge restorer);
# the survivor gets (re-)placed through the normal community cycle, scored
# on the semantic edges the absorb just enriched. Audit 2026-06-12: the
# consolidation prompt + the comment above stated this exclusion as fact
# while the code migrated everything — this constant makes it true.
ABSORB_EXCLUDED_RELATIONS = frozenset(['community_member'])

# When `include_archived=False` is the default, every edge-reading method
# filters `archived = 0` in its WHERE clause. v25 added the column;
# this contract is the reason the filter lives in one place.


def _relation_not_in_clause(values):
    """Build an `AND relation NOT IN (?,?...)` SQL fragment + its param list for
    exempting relations from an edge-archival UPDATE. Returns ('', []) when
    empty. Single source for the survivor_lineage exemption shared by
    delete_node_edges + archive_dangling_edges, so the clause shape can't drift
    between them."""
    vals = list(values or ())
    if not vals:
        return '', []
    return 'AND relation NOT IN (%s)' % ','.join('?' * len(vals)), vals


class GraphDAL:
    """Access layer for brain.db graph tables: edges + edge_relations.

    ALL edge SQL lives here. When we move to in-memory graph, swap this
    implementation — nothing else changes. Every edge-reading method
    honors the GRAPH QUERY CONTRACT above:
      - Returns EDGE_ROW_SHAPE dicts (node-centric, neighbor fields flat)
      - Defaults include_archived=False (v25 soft-archive filter)
      - Accepts exclude_relations set to drop noise (see
        DEFAULT_EXCLUDED_RELATIONS for the standard noise set)

    Raises on invalid args — no silent empty returns masking bad calls.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # --- Reads ---

    def count_total(self) -> int:
        """Count total edges."""
        row = self.conn.execute('SELECT COUNT(*) FROM edges').fetchone()
        return row[0] if row else 0

    def get_edge(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get edge between two nodes (checks both directions — single-direction storage).

        Returns edge dict with edge_id, direction, and relations list.
        """
        row = self.conn.execute(
            'SELECT edge_id, source_id, target_id, weight, co_access_count, last_strengthened '
            'FROM edges WHERE (source_id = ? AND target_id = ?) '
            'OR (source_id = ? AND target_id = ?)',
            (source_id, target_id, target_id, source_id)
        ).fetchone()
        if not row:
            return None

        edge_id = row[0]
        direction = 'outgoing' if row[1] == source_id else 'incoming'
        relations = self.get_relations(edge_id)

        return {
            'edge_id': edge_id, 'weight': row[3],
            'co_access_count': row[4], 'last_strengthened': row[5],
            'direction': direction, 'relations': relations,
        }

    def edge_exists(self, source_id: str, target_id: str) -> bool:
        """Check if edge exists in either direction."""
        row = self.conn.execute(
            'SELECT 1 FROM edges WHERE (source_id = ? AND target_id = ?) '
            'OR (source_id = ? AND target_id = ?)',
            (source_id, target_id, target_id, source_id)
        ).fetchone()
        return row is not None

    def get_edge_id(self, source_id: str, target_id: str) -> Optional[str]:
        """Get edge_id for a pair (checks both directions)."""
        row = self.conn.execute(
            'SELECT edge_id FROM edges WHERE (source_id = ? AND target_id = ?) '
            'OR (source_id = ? AND target_id = ?)',
            (source_id, target_id, target_id, source_id)
        ).fetchone()
        return row[0] if row else None

    def get_neighbors(self, node_id: str, limit: int = 8,
                      exclude_relations: set = None,
                      exclude_node_ids: set = None,
                      include_archived: bool = False,
                      content_preview_chars: int = 0) -> List[Dict[str, Any]]:
        """Get neighbors with node + edge + relation data (EDGE_ROW_SHAPE).

        Single-direction storage: queries both directions, flags each as
        outgoing/incoming. Relations from edge_relations via edge_id JOIN.

        Args:
            node_id: Node to find neighbors of. Empty → raises ValueError.
            limit: Max neighbors (ordered by edge weight desc).
            exclude_relations: Relation types to skip. None → no exclusion.
                Pass DEFAULT_EXCLUDED_RELATIONS for the standard noise set.
            exclude_node_ids: Node IDs to skip (already visited in traversal).
            include_archived: if False (default), filters er.archived=0.
                Enables forensic/recovery queries when True.
            content_preview_chars: if > 0, adds `content_preview` field to
                each row — substr(content, 1, N). Default 0 skips content
                to keep result size small.
        """
        if not node_id:
            raise ValueError("get_neighbors: node_id required")

        where_parts = ["n.archived = 0"]
        params = [node_id, node_id, node_id, node_id]

        if not include_archived:
            where_parts.append("er.archived = 0")

        if exclude_node_ids:
            placeholders = ",".join("?" * len(exclude_node_ids))
            where_parts.append("n.id NOT IN (%s)" % placeholders)
            params.extend(exclude_node_ids)

        if exclude_relations:
            placeholders = ",".join("?" * len(exclude_relations))
            where_parts.append("er.relation NOT IN (%s)" % placeholders)
            params.extend(exclude_relations)

        params.append(limit)
        where_clause = " AND ".join(where_parts)

        preview_col = ''
        if content_preview_chars and content_preview_chars > 0:
            preview_col = ', substr(n.content, 1, %d) as content_preview' % int(content_preview_chars)

        rows = self.conn.execute("""
            SELECT
                n.id, n.type, n.title, n.content_summary, n.confidence,
                n.revised_at, n.created_at, n.last_accessed, n.access_count,
                n.locked, n.emotion, n.emotion_label,
                er.relation, er.weight, er.description,
                e.last_strengthened, e.co_access_count, e.edge_id,
                CASE WHEN e.source_id = ? THEN 'outgoing' ELSE 'incoming' END as direction
                {preview_col}
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n ON n.id = CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
            WHERE (e.source_id = ? OR e.target_id = ?)
            AND {where_clause}
            ORDER BY e.weight DESC
            LIMIT ?
        """.format(preview_col=preview_col, where_clause=where_clause),
            params).fetchall()

        out = []
        for r in rows:
            row = {
                'id': r[0], 'type': r[1], 'title': r[2], 'content_summary': r[3],
                'confidence': r[4], 'revised_at': r[5], 'created_at': r[6],
                'last_accessed': r[7], 'access_count': r[8], 'locked': r[9],
                'emotion': r[10], 'emotion_label': r[11],
                'relation': r[12] or '', 'weight': r[13],
                'edge_description': r[14], 'last_strengthened': r[15],
                'co_access_count': r[16], 'edge_id': r[17],
                'direction': r[18],
            }
            if preview_col:
                row['content_preview'] = r[19] or ''
            out.append(row)
        return out

    # ──────────────────────────────────────────────────────────────
    # v25 consolidated edge-read API — see GRAPH QUERY CONTRACT above.
    # Every method defaults include_archived=False.
    # ──────────────────────────────────────────────────────────────

    def nodes_touched_by_relations(self, relations, include_archived: bool = False):
        """Set of node IDs that participate (as source or target) in any edge
        with one of the given relations.

        Used by the 'Unreviewed Node' suppression pattern — any unit can ask
        "which nodes have already been seen by this kind of edge?" without
        writing raw SQL. Defaults to active (non-archived) edge_relations.
        """
        rels = list(relations)
        if not rels:
            return set()
        ph = ','.join('?' * len(rels))
        archived_clause = '' if include_archived else ' AND er.archived = 0'
        rows = self.conn.execute("""
            SELECT DISTINCT node_id FROM (
                SELECT e.source_id AS node_id FROM edges e
                JOIN edge_relations er ON er.edge_id = e.edge_id
                WHERE er.relation IN (%s)%s
                UNION
                SELECT e.target_id AS node_id FROM edges e
                JOIN edge_relations er ON er.edge_id = e.edge_id
                WHERE er.relation IN (%s)%s
            )
        """ % (ph, archived_clause, ph, archived_clause),
            rels + rels).fetchall()
        return {r[0] for r in rows}

    def get_neighbors_bulk(self, node_ids,
                           exclude_relations=None,
                           include_archived: bool = False):
        """Bulk flat-row version of get_neighbors — one query for many owners.

        Returns dict {owner_id: [row_dict, ...]} where each row_dict has
        EDGE_ROW_SHAPE (one entry per edge_relation, not grouped). Use this
        when the caller iterates relations individually; use
        get_connections_bulk when the caller wants relations grouped per
        (owner, neighbor).

        Defaults to DEFAULT_EXCLUDED_RELATIONS when exclude_relations is None
        (drops co_accessed, emergent_bridge). Pass an empty set to include all.
        """
        ids = list(node_ids)
        if not ids:
            return {}

        if exclude_relations is None:
            exclude_relations = DEFAULT_EXCLUDED_RELATIONS

        owner_ph = ",".join("?" * len(ids))
        where_parts = ["n.archived = 0"]
        params = list(ids) + list(ids) + list(ids)

        if not include_archived:
            where_parts.append("er.archived = 0")

        if exclude_relations:
            rel_ph = ",".join("?" * len(exclude_relations))
            where_parts.append("er.relation NOT IN (%s)" % rel_ph)
            params.extend(exclude_relations)

        where_clause = " AND ".join(where_parts)

        rows = self.conn.execute("""
            SELECT
                CASE WHEN e.source_id IN ({owner_ph}) THEN e.source_id ELSE e.target_id END AS owner_id,
                n.id, n.type, n.title, n.content_summary, n.confidence,
                n.revised_at, n.created_at, n.last_accessed, n.access_count,
                n.locked, n.emotion, n.emotion_label,
                er.relation, er.weight, er.description,
                e.last_strengthened, e.co_access_count, e.edge_id,
                CASE WHEN e.source_id IN ({owner_ph}) THEN 'outgoing' ELSE 'incoming' END as direction
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n ON n.id = CASE WHEN e.source_id IN ({owner_ph}) THEN e.target_id ELSE e.source_id END
            WHERE (e.source_id IN ({owner_ph}) OR e.target_id IN ({owner_ph}))
            AND {where_clause}
            ORDER BY e.weight DESC
        """.format(owner_ph=owner_ph, where_clause=where_clause),
            params + list(ids) + list(ids)).fetchall()

        result = {owner: [] for owner in ids}
        for r in rows:
            owner = r[0]
            if owner not in result:
                # Edge row where owner matched join but not in our list — skip
                continue
            result[owner].append({
                'id': r[1], 'type': r[2], 'title': r[3], 'content_summary': r[4],
                'confidence': r[5], 'revised_at': r[6], 'created_at': r[7],
                'last_accessed': r[8], 'access_count': r[9], 'locked': r[10],
                'emotion': r[11], 'emotion_label': r[12],
                'relation': r[13] or '', 'weight': r[14],
                'edge_description': r[15], 'last_strengthened': r[16],
                'co_access_count': r[17], 'edge_id': r[18],
                'direction': r[19],
            })
        return result

    def get_connections_bulk(self, node_ids,
                             exclude_relations=None,
                             include_relations=None,
                             include_archived: bool = False,
                             include_neighbor_archived: bool = False):
        """Grouped neighbor fetch — multiple relations per (owner, neighbor)
        collapsed into a single entry with a `relations` list.

        The rich-node builder in brain_recall needs this shape: one entry
        per unique (owner, neighbor) pair, carrying aggregate edge weight
        and all relations on that pair.

        Args:
            node_ids: owner node ids to walk from
            exclude_relations: relations to skip (defaults to noise-relation list).
                Ignored if include_relations is set.
            include_relations: when set, ONLY relations in this iterable are
                returned. Use for aspect-scoped walks (e.g. correction-aspect
                relations only). Mutually exclusive with exclude_relations —
                if include_relations is provided, exclude_relations is ignored.
            include_archived: include archived edge_relations rows
            include_neighbor_archived: include edges whose neighbor node is archived

        Returns dict {owner_id: [connection_dict, ...]} where each
        connection_dict has:
            id, type, title, created_at, revised_at, confidence, locked,
            weight, direction, relations: [{relation, description, weight}, ...]

        Raises ValueError on empty node_ids.
        """
        ids = list(node_ids)
        if not ids:
            raise ValueError("get_connections_bulk: node_ids is empty")

        id_ph = ','.join('?' * len(ids))

        archived_clause = '' if include_archived else 'AND er.archived = 0'
        neighbor_archived_clause = (
            'AND n1.archived = 0 AND n2.archived = 0'
            if not include_neighbor_archived else ''
        )
        rel_clause = ''
        rel_params = []
        if include_relations is not None:
            inc_list = list(include_relations)
            if not inc_list:
                # Empty whitelist → no edges match. Return empty grouping.
                return {nid: [] for nid in ids}
            rel_ph = ','.join('?' * len(inc_list))
            rel_clause = 'AND er.relation IN (%s)' % rel_ph
            rel_params = inc_list
        else:
            if exclude_relations is None:
                exclude_relations = DEFAULT_EXCLUDED_RELATIONS
            if exclude_relations:
                rel_ph = ','.join('?' * len(exclude_relations))
                rel_clause = 'AND er.relation NOT IN (%s)' % rel_ph
                rel_params = list(exclude_relations)

        sql = """
            SELECT e.source_id, e.target_id, e.weight,
                   er.relation, er.description, er.weight as rel_weight,
                   n1.id, n1.type, n1.title, n1.created_at, n1.revised_at,
                   n1.confidence, n1.locked,
                   n2.id, n2.type, n2.title, n2.created_at, n2.revised_at,
                   n2.confidence, n2.locked
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n1 ON n1.id = e.target_id
            JOIN nodes n2 ON n2.id = e.source_id
            WHERE (e.source_id IN ({id_ph}) OR e.target_id IN ({id_ph}))
              {archived_clause}
              {neighbor_archived_clause}
              {rel_clause}
        """.format(
            id_ph=id_ph,
            archived_clause=archived_clause,
            neighbor_archived_clause=neighbor_archived_clause,
            rel_clause=rel_clause,
        )

        rows = self.conn.execute(sql, ids + ids + rel_params).fetchall()

        owner_set = set(ids)
        grouped = {nid: {} for nid in ids}

        for row in rows:
            src, tgt = row[0], row[1]
            agg_weight = row[2]
            rel = row[3] or 'related'
            desc = row[4] or ''
            rel_weight = row[5] if row[5] is not None else agg_weight
            n1 = {'id': row[6], 'type': row[7], 'title': row[8],
                  'created_at': row[9], 'revised_at': row[10],
                  'confidence': row[11], 'locked': row[12] == 1}
            n2 = {'id': row[13], 'type': row[14], 'title': row[15],
                  'created_at': row[16], 'revised_at': row[17],
                  'confidence': row[18], 'locked': row[19] == 1}
            relation_entry = {'relation': rel, 'description': desc,
                              'weight': rel_weight}

            if src in owner_set and tgt != src:
                entry = grouped[src].setdefault(n1['id'], {
                    **n1, 'weight': agg_weight, 'direction': 'outgoing',
                    'relations': [],
                })
                entry['relations'].append(relation_entry)

            if tgt in owner_set and src != tgt:
                entry = grouped[tgt].setdefault(n2['id'], {
                    **n2, 'weight': agg_weight, 'direction': 'incoming',
                    'relations': [],
                })
                entry['relations'].append(relation_entry)

        return {owner: list(nbrs.values()) for owner, nbrs in grouped.items()}

    def archive_dangling_edges(self, archived_by: str,
                               exempt_relations=()) -> int:
        """Archive active edge_relations rows whose source or target node is archived.

        Invariant restorer: the brain's rule is `Archive edges alongside nodes —
        no dangling edges after committing`. Historical leak paths
        (pre-April-2026 archive_node deletion bug, mid-migration races) can
        leave active edges pointing at archived nodes; this method scrubs them.

        Args:
            archived_by: encoding_source-style tag for the archive action
                (e.g. 's2:healer', 'migration:cleanup_2026_05_16').
            exempt_relations: relations that are SUPPOSED to span an archived
                endpoint and must NOT be scrubbed — the survivor-redirect link
                `absorbed_into` (resolve_live walks it; in a chain A→B→C its
                target is itself archived). The caller sources these from
                `brain.aspects.relations_in(['survivor_lineage'])` so the
                taxonomy, not this method, owns the list. DAL stays
                aspect-agnostic — it just takes the strings.

        Returns count of edge_relations rows newly archived.
        """
        # archived_at is ISO-T via clock.iso_now() — the same format
        # brain_remember.archive_node and every other edge_relations
        # writer uses. (Unified 2026-06-12: this method was the lone
        # unix-ms writer into the TEXT column, which broke lexicographic
        # time reads and rendered as 1970 epoch dates.)
        ts = _now()
        exempt_clause, exempt = _relation_not_in_clause(exempt_relations)
        cur = self.conn.execute("""
            UPDATE edge_relations
               SET archived = 1,
                   archived_at = ?,
                   archived_by = ?
             WHERE archived = 0
               %s
               AND edge_id IN (
                 SELECT er.edge_id FROM edge_relations er
                 JOIN edges e ON e.edge_id = er.edge_id
                 JOIN nodes n_src ON n_src.id = e.source_id
                 JOIN nodes n_tgt ON n_tgt.id = e.target_id
                 WHERE er.archived = 0
                   AND (n_src.archived = 1 OR n_tgt.archived = 1)
               )
        """ % exempt_clause, [ts, archived_by] + exempt)
        return cur.rowcount

    def reconcile_community_membership(self,
                                       encoding_source='s2:community_repair'):
        """Restorer: back-fill `community_member` edges for ORPHANED communities.

        A community declares its members in two places that can silently
        diverge: the `community_members` metadata string AND the actual
        `community_member` edges. The community encoder (Haiku) sometimes
        creates the node + metadata but omits the edge field entirely — or
        used the retired `connections=` param (dropped by remember()'s guard).
        The node then claims N members with ZERO edges: a structural
        inconsistency nothing else catches, because the declared list is the
        only diffable intent (community is the only encoder that records its
        expected structure as data). See also `archive_dangling_edges` — same
        per-cycle integrity-restorer pattern.

        Scope is deliberately the ZERO-edge case only. A community with SOME
        member edges is left alone: a partial gap is far more likely intentional
        drift (a member was disconnected) than omission, and re-adding from the
        possibly-stale metadata would resurrect a removed member. Omission is
        all-or-nothing at the encoder (one `connect_to` field for all members),
        so it always presents as zero edges — exactly what this targets.

        Idempotent: once edges exist the community is skipped. Archived/missing
        declared members are skipped (legitimate drift, not omission).

        Caller must hold brain.write_lock (writes via add_relation).
        Returns {communities_healed, edges_backfilled, details: [(cid, n), ...]}.
        """
        import re
        # 1. Declared members per live community (id -> {member_id: label}).
        #    Anchor the id match to a segment start (^ or comma) so an 8-hex
        #    token inside a member's TITLE can't be mistaken for a member id.
        declared = {}
        for cid, val in self.conn.execute(
                "SELECT kv.node_id, kv.value FROM node_metadata_kv kv "
                "JOIN nodes n ON n.id = kv.node_id "
                "WHERE kv.key = 'community_members' "
                "AND n.type = 'community' AND n.archived = 0").fetchall():
            members = {}
            for mt in re.finditer(r'(?:^|,)\s*([0-9a-f]{8})\s*:\s*([^,]*)',
                                  val or ''):
                members[mt.group(1)] = mt.group(2).strip()
            if members:
                declared[cid] = members
        if not declared:
            return {'communities_healed': 0, 'edges_backfilled': 0,
                    'details': []}

        # 2. Communities that ALREADY have >=1 active community_member edge,
        #    in EITHER direction — skip them (partial gap == drift, not
        #    omission). get_community_members reads membership both ways
        #    (historical mix of community->member and legacy member->community
        #    edges), so this orphan check must too — else a legacy-direction
        #    community is falsely flagged as orphan every cycle.
        edged = set()
        for src, tgt in self.conn.execute(
                "SELECT e.source_id, e.target_id FROM edges e "
                "JOIN edge_relations er ON er.edge_id = e.edge_id "
                "WHERE er.relation = 'community_member' "
                "AND er.archived = 0").fetchall():
            edged.add(src)
            edged.add(tgt)

        orphans = {cid: ms for cid, ms in declared.items() if cid not in edged}
        if not orphans:
            return {'communities_healed': 0, 'edges_backfilled': 0,
                    'details': []}

        # 3. Which declared members are LIVE (skip archived/missing targets —
        #    a declared member that was archived is drift, not an omission).
        all_members = sorted({m for ms in orphans.values() for m in ms})
        live = set()
        for i in range(0, len(all_members), 400):
            chunk = all_members[i:i + 400]
            rows = self.conn.execute(
                "SELECT id FROM nodes WHERE archived = 0 AND id IN (%s)"
                % ','.join('?' * len(chunk)), chunk).fetchall()
            live.update(r[0] for r in rows)

        # 4. Back-fill the gap on orphaned communities only.
        healed = edges = 0
        details = []
        for cid, members in orphans.items():
            # Skip self: community_members occasionally echoes the community's
            # own id, and add_relation has no self-edge guard (the LLM path
            # does, via _apply_connect_to exclude_self).
            live_missing = [(m, lbl) for m, lbl in members.items()
                            if m in live and m != cid]
            if not live_missing:
                continue
            for mid, label in live_missing:
                desc = ((label or 'community member')
                        + ' — member edge restored by membership '
                          'reconciliation')[:200]
                self.add_relation(cid, mid, 'community_member',
                                  description=desc, weight=0.6,
                                  encoding_source=encoding_source)
                edges += 1
            healed += 1
            details.append((cid, len(live_missing)))
        return {'communities_healed': healed, 'edges_backfilled': edges,
                'details': details}

    def has_edge_between(self, source_ids, target_ids,
                         relations=None,
                         include_archived: bool = False) -> bool:
        """Existence check — is there any edge with these relations between
        any node in source_ids and any node in target_ids?

        Used by correction/tension detection, bridge-count guards.
        Raises ValueError if either set is empty.
        """
        src_list = list(source_ids)
        tgt_list = list(target_ids)
        if not src_list or not tgt_list:
            raise ValueError(
                "has_edge_between: source_ids and target_ids must both be non-empty")

        src_ph = ','.join('?' * len(src_list))
        tgt_ph = ','.join('?' * len(tgt_list))
        params = src_list + tgt_list

        rel_clause = ''
        if relations:
            rel_list = list(relations)
            rel_ph = ','.join('?' * len(rel_list))
            rel_clause = 'AND er.relation IN (%s)' % rel_ph
            params += rel_list

        archived_clause = '' if include_archived else 'AND er.archived = 0'

        sql = """
            SELECT 1 FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            WHERE e.source_id IN (%s) AND e.target_id IN (%s)
              %s
              %s
            LIMIT 1
        """ % (src_ph, tgt_ph, rel_clause, archived_clause)

        return self.conn.execute(sql, params).fetchone() is not None

    def get_community_members(self, community_id: str,
                              include_archived: bool = False,
                              require_active_member: bool = True):
        """Members of a community via community_member edges.

        Walks both directions of the edge to handle the historical mix
        where some edges point node→community and others community→node.
        Returns neighbor node dicts (subset of EDGE_ROW_SHAPE:
        id, type, title, created_at, confidence, locked).

        Raises ValueError if community_id is empty.
        """
        if not community_id:
            raise ValueError("get_community_members: community_id required")

        archived_clause = '' if include_archived else 'AND er.archived = 0'
        member_archived_clause = 'AND member.archived = 0' if require_active_member else ''

        sql = """
            SELECT DISTINCT member.id, member.type, member.title,
                            member.created_at, member.confidence, member.locked
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes member ON member.id = CASE
                WHEN e.source_id = ? THEN e.target_id
                ELSE e.source_id END
            WHERE er.relation = 'community_member'
              AND (e.source_id = ? OR e.target_id = ?)
              AND member.type != 'community'
              %s
              %s
        """ % (archived_clause, member_archived_clause)

        rows = self.conn.execute(sql, (community_id, community_id, community_id)).fetchall()
        return [{
            'id': r[0], 'type': r[1], 'title': r[2],
            'created_at': r[3], 'confidence': r[4], 'locked': r[5],
        } for r in rows]

    def get_members_bulk(self, community_ids, include_archived: bool = False):
        """Members of MANY communities via community_member edges, batched.

        Bulk sibling of get_community_members: same bidirectional walk (the
        historical community->member and legacy member->community mix) and the
        same EDGE_ROW_SHAPE subset, but for a list of communities in one pass.
        Returns {community_id: [member dict, ...]}; communities with no member
        edges are simply absent (caller treats as empty). DISTINCT collapses a
        member reachable via edges in both directions.
        """
        ids = [c for c in (community_ids or []) if c]
        if not ids:
            return {}
        archived_clause = '' if include_archived else 'AND er.archived = 0'
        out = {}
        # Chunk to stay within SQLite's bind-variable limit.
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            placeholders = ','.join('?' * len(chunk))
            sql = """
                SELECT DISTINCT c.id AS community_id,
                                member.id, member.type, member.title,
                                member.created_at, member.confidence, member.locked
                FROM nodes c
                JOIN edges e ON (e.source_id = c.id OR e.target_id = c.id)
                JOIN edge_relations er ON er.edge_id = e.edge_id
                    AND er.relation = 'community_member' %s
                JOIN nodes member ON member.id = CASE
                    WHEN e.source_id = c.id THEN e.target_id
                    ELSE e.source_id END
                    AND member.archived = 0 AND member.type != 'community'
                WHERE c.id IN (%s) AND c.type = 'community' AND c.archived = 0
            """ % (archived_clause, placeholders)
            for r in self.conn.execute(sql, chunk).fetchall():
                out.setdefault(r[0], []).append({
                    'id': r[1], 'type': r[2], 'title': r[3],
                    'created_at': r[4], 'confidence': r[5], 'locked': r[6],
                })
        return out

    def get_communities_for(self, node_ids,
                            include_archived: bool = False,
                            require_active_community: bool = True):
        """Reverse of get_community_members: for each given node, list the
        communities it belongs to via community_member edges.

        Returns dict {node_id: [{id, title}, ...]} — symmetric to
        get_community_members's shape. Used by consolidation_decoder to
        enrich clusters with their community placement.

        Raises ValueError on empty node_ids.
        """
        ids = list(node_ids)
        if not ids:
            raise ValueError("get_communities_for: node_ids is empty")

        id_ph = ','.join('?' * len(ids))
        archived_clause = '' if include_archived else 'AND er.archived = 0'
        community_clause = 'AND n.archived = 0' if require_active_community else ''

        sql = """
            SELECT
                CASE WHEN e.source_id IN ({id_ph}) THEN e.source_id
                     ELSE e.target_id END as member,
                CASE WHEN e.source_id IN ({id_ph}) THEN e.target_id
                     ELSE e.source_id END as community,
                n.title
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n ON n.id = CASE
                WHEN e.source_id IN ({id_ph}) THEN e.target_id
                ELSE e.source_id END
            WHERE (e.source_id IN ({id_ph}) OR e.target_id IN ({id_ph}))
              AND er.relation = 'community_member'
              AND n.type = 'community'
              {archived_clause}
              {community_clause}
        """.format(
            id_ph=id_ph,
            archived_clause=archived_clause,
            community_clause=community_clause,
        )

        rows = self.conn.execute(sql, ids * 5).fetchall()

        from collections import defaultdict
        membership = defaultdict(list)
        for member_id, comm_id, comm_title in rows:
            membership[member_id].append({'id': comm_id, 'title': comm_title})
        return dict(membership)

    def count_by_relation(self, include_archived: bool = False):
        """Edge count grouped by relation type.

        Returns dict {relation: count}, ordered by count desc. Used by
        integrity_audit, edge_families, health_check.
        """
        where = '' if include_archived else 'WHERE archived = 0'
        rows = self.conn.execute(
            "SELECT relation, COUNT(*) as cnt FROM edge_relations %s "
            "GROUP BY relation ORDER BY cnt DESC" % where
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_edge_descriptions_for(self, node_id: str,
                                  min_length: int = EDGE_CONTEXT_MIN_DESC_LENGTH,
                                  exclude_relations=None,
                                  include_archived: bool = False,
                                  limit: int = 5):
        """Return meaningful edge descriptions for a node's edges.

        Feeds edge_context embedding in _compute_group_vectors. Filters
        out short/empty descriptions (below min_length) and noise relations.
        Default exclusion: DEFAULT_EXCLUDED_RELATIONS + 'community_member'
        (structural, not semantic text).

        Returns list[str] of descriptions. Raises ValueError if node_id empty.
        """
        if not node_id:
            raise ValueError("get_edge_descriptions_for: node_id required")
        if exclude_relations is None:
            exclude_relations = DEFAULT_EXCLUDED_RELATIONS | {'community_member'}

        archived_clause = '' if include_archived else 'AND er.archived = 0'
        rel_clause = ''
        if exclude_relations:
            rel_ph = ','.join('?' * len(exclude_relations))
            rel_clause = 'AND er.relation NOT IN (%s)' % rel_ph

        sql = """
            SELECT er.description FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            WHERE (e.source_id = ? OR e.target_id = ?)
              %s
              %s
              AND er.description IS NOT NULL
              AND length(er.description) > ?
            ORDER BY e.weight DESC
            LIMIT ?
        """ % (archived_clause, rel_clause)

        params = [node_id, node_id]
        if exclude_relations:
            params += list(exclude_relations)
        params += [min_length, limit]

        rows = self.conn.execute(sql, params).fetchall()
        return [r[0] for r in rows if r[0]]

    def count_node_edges(self, node_id: str, min_weight: float = 0.1,
                         relations=None,
                         include_archived: bool = False) -> int:
        """Count edges touching a node (both directions).

        Args:
            node_id: node whose edges to count. Empty → raises ValueError.
            min_weight: edges-table weight floor. Default 0.1.
            relations: iterable of relation names — if provided, only count
                edges that carry at least one of these relations (via JOIN).
                None → count the aggregate edges table, relation-agnostic.
            include_archived: if False (default) and `relations` is set,
                filters er.archived=0. Ignored when relations is None
                (aggregate edges has no archived column).

        Why two paths: when you don't care about relation, counting from
        `edges` directly is cheap. When you do, the JOIN is needed.
        """
        if not node_id:
            raise ValueError("count_node_edges: node_id required")

        if relations:
            rel_list = list(relations)
            rel_ph = ','.join('?' * len(rel_list))
            archived_clause = '' if include_archived else 'AND er.archived = 0'
            row = self.conn.execute(
                'SELECT COUNT(DISTINCT er.edge_id) FROM edges e '
                'JOIN edge_relations er ON er.edge_id = e.edge_id '
                'WHERE (e.source_id = ? OR e.target_id = ?) '
                'AND e.weight >= ? '
                'AND er.relation IN (%s) '
                '%s' % (rel_ph, archived_clause),
                [node_id, node_id, min_weight] + rel_list
            ).fetchone()
            return row[0] if row else 0

        row = self.conn.execute(
            'SELECT COUNT(*) FROM edges WHERE (source_id = ? OR target_id = ?) AND weight >= ?',
            (node_id, node_id, min_weight)
        ).fetchone()
        return row[0] if row else 0

    # get_edge_count removed 2026-05-30 (DAL cleanup Phase 0) — exact dup of
    # count_total; both were dead (brain._get_edge_count uses raw SQL, to be
    # routed through count_total in Phase 3).
    # get_well_connected + get_random_walk_neighbors removed 2026-05-30 — the
    # consolidation/promotion + random-walk paths that used them are retired
    # (brain_connections._random_walk, their only kin, was also dead).

    # --- Writes ---

    # create_edge removed 2026-05-30 (DAL cleanup Phase 0) — DEPRECATED since the
    # Hebbian path moved to recall_write_queue via add_relation; 0 callers (its
    # docstring's claimed brain_recall caller was already gone). Use
    # brain.connect_typed() (write-path embed hook) for all edge creation.

    # strengthen_edge REMOVED 2026-05-18 (Phase 8 of bg_writer migration).
    # Was a deprecated read-modify-write helper used only by the old
    # brain_recall._hebbian_strengthen mixin, which Phase 5 deleted.
    # Hebbian strengthening now uses atomic UPSERT inside
    # recall_write_queue._apply_hebbian_pairs via add_relation.

    def delete_node_edges(self, node_id: str,
                          archived_by: str = 'delete_node_edges',
                          exempt_relations=()) -> int:
        """Soft-archive all edge_relations touching a node (v25).

        Single source for "archive a node's edges" — archive_node routes here
        (passing its real `archived_by`) instead of duplicating the SQL.

        Commit is gated on self.conn.in_batch (commit_unless_batched) — a no-op
        inside a brain_batch envelope, a real commit standalone.

        Was a hard DELETE prior to v25 — the asymmetry with node archive
        destroyed edge provenance forever. Now sets archived=1 on the
        relations and leaves the edges aggregate row intact. Returns count
        of relations archived.

        Args:
            archived_by: encoding_source-style tag (e.g. 's2:consolidation').
            exempt_relations: relations that must outlive the node — the
                survivor-redirect link `absorbed_into`, or the resolve_live
                chain breaks. Caller sources these from
                `brain.aspects.relations_in(['survivor_lineage'])`; the DAL
                stays aspect-agnostic. hard_delete_node_edges removes
                everything regardless (a deleted endpoint leaves no chain).
        """
        edge_ids = [r[0] for r in self.conn.execute(
            'SELECT edge_id FROM edges WHERE source_id = ? OR target_id = ?',
            (node_id, node_id)
        ).fetchall()]

        archived_count = 0
        if edge_ids:
            ts = _now()
            exempt_clause, exempt = _relation_not_in_clause(exempt_relations)
            # NULL the stored embedding on archive — same pattern node
            # archive uses (DELETE FROM node_enrichments). Archived edges
            # are never read by spread/select_edges (every read filters
            # archived=0), so the blob is dead weight in the table. If
            # the relation is later revived via add_relation Branch 3,
            # created=True fires enqueue_edge and the embed_queue worker
            # re-embeds async. Symmetric with nodes; storage isn't burned
            # on history that no one queries.
            for i in range(0, len(edge_ids), 500):
                chunk = edge_ids[i:i + 500]
                ph = ','.join('?' * len(chunk))
                cur = self.conn.execute(
                    'UPDATE edge_relations '
                    'SET archived = 1, archived_at = ?, archived_by = ?, '
                    '    embedding = NULL, embedding_model = NULL '
                    'WHERE edge_id IN (%s) AND archived = 0 %s' % (
                        ph, exempt_clause),
                    [ts, archived_by] + chunk + exempt)
                archived_count += cur.rowcount

        commit_unless_batched(self.conn)
        return archived_count

    def hard_delete_node_edges(self, node_id: str) -> int:
        """HARD-delete every edge touching a node — both the `edge_relations`
        rows and the `edges` aggregate rows. For the node delete-cascade: a hard
        node delete must leave no edge or relation rows. Contrast
        delete_node_edges, which SOFT-archives (archived=1) to preserve history
        for a still-live node. Returns the count of `edges` rows removed."""
        if not node_id:
            return 0
        edge_ids = [r[0] for r in self.conn.execute(
            'SELECT edge_id FROM edges WHERE source_id = ? OR target_id = ?',
            (node_id, node_id)).fetchall()]
        if edge_ids:
            ph = ','.join('?' * len(edge_ids))
            self.conn.execute(
                'DELETE FROM edge_relations WHERE edge_id IN (%s)' % ph, edge_ids)
        n = self.conn.execute(
            'DELETE FROM edges WHERE source_id = ? OR target_id = ?',
            (node_id, node_id)).rowcount
        commit_unless_batched(self.conn)
        return n

    def decay_edges(self) -> Dict[str, Any]:
        """Apply exponential decay to auto-generated edge relations.

        Commit is gated on self.conn.in_batch (commit_unless_batched).

        Operates on edge_relations via edge_id.
        When a relation's weight drops below threshold, that relation is removed.
        If an edge has no relations left, the physical edge is also removed.

        Formula: new_weight = weight * 0.5^(hours_since_created / half_life)

        Returns: {decayed: int, pruned: int, by_type: {relation: {decayed, pruned}}}
        """
        from .brain_constants import EDGE_TYPES, EDGE_PRUNE_THRESHOLD

        total_decayed = 0
        total_pruned = 0
        by_type = {}

        for relation, config in EDGE_TYPES.items():
            if not config.get('decays'):
                continue

            half_life = config['halfLife']

            # Apply decay on active edge_relations only (v25)
            self.conn.execute("""
                UPDATE edge_relations
                SET weight = weight * power(0.5,
                    (julianday('now') - julianday(created_at)) * 24.0 / ?)
                WHERE relation = ?
                  AND archived = 0
                  AND created_at IS NOT NULL
                  AND (julianday('now') - julianday(created_at)) * 24.0 > 0
            """, (half_life, relation))
            decayed = self.conn.execute('SELECT changes()').fetchone()[0]

            # Collect edge_ids that will be pruned (active only)
            pruned_edge_ids = [r[0] for r in self.conn.execute(
                "SELECT edge_id FROM edge_relations "
                "WHERE relation = ? AND weight < ? AND archived = 0",
                (relation, EDGE_PRUNE_THRESHOLD)
            ).fetchall()]

            # Soft-archive relations below threshold (v25 — was DELETE).
            # NULL the embedding too — see delete_node_edges for rationale
            # (archived edges are unread; revive re-embeds via add_relation).
            prune_ts = _now()
            self.conn.execute(
                "UPDATE edge_relations "
                "SET archived = 1, archived_at = ?, archived_by = ?, "
                "    embedding = NULL, embedding_model = NULL "
                "WHERE relation = ? AND weight < ? AND archived = 0",
                (prune_ts, 'decay_pruned', relation, EDGE_PRUNE_THRESHOLD))
            pruned = self.conn.execute('SELECT changes()').fetchone()[0]

            if decayed or pruned:
                by_type[relation] = {'decayed': decayed, 'pruned': pruned}
                total_decayed += decayed
                total_pruned += pruned

            # Recompute aggregate weight from remaining active relations.
            # Edges aggregate row stays regardless — reads join on archived=0.
            for eid in set(pruned_edge_ids):
                self._update_aggregate_weight(eid)

        commit_unless_batched(self.conn)
        return {'decayed': total_decayed, 'pruned': total_pruned, 'by_type': by_type}

    # --- edge_relations (multi-relation semantic layer via edge_id) ---

    @staticmethod
    def _generate_edge_id(source_id, target_id):
        """Deterministic edge ID from source+target pair."""
        import hashlib
        h = hashlib.md5((source_id + ':' + target_id).encode()).hexdigest()[:8]
        return 'edg_' + h

    # Sentinel for distinguishing "field not specified" (preserve existing)
    # from "explicit value passed" (replace). Plain default values can't
    # express this distinction.
    _UNSET = object()

    def add_relation(self, source_id, target_id, relation,
                     description=_UNSET, weight=_UNSET, encoding_source=_UNSET):
        """Upsert a relation on an edge pair. Creates the physical edge if needed.

        Stage 1B contract — field-preserving upsert + lifecycle audit via traces.

        Commit is gated on self.conn.in_batch (commit_unless_batched): a no-op
        when a wider transaction owns the envelope (brain_batch, or the
        bg_writer queue drain that opens BEGIN IMMEDIATE around a batch of
        pairs and commits once), a real commit standalone. The owner flips
        conn.in_batch — letting add_relation self-commit inside would break
        atomicity (earlier writes persist while a later failure rolls back
        only the most-recent statements).

        Three branches by row state for (edge_id, relation):
          - No row              → INSERT with passed values + sensible defaults
          - Active row exists   → field-preserving UPDATE (only specified fields update)
          - Archived row exists → REVIVE: archived=0, fresh created_at, all fields
                                  reset to passed values + defaults (semantic
                                  fresh row; PK forces row reuse, trace events
                                  capture the lifecycle)

        Auto-strengthen behavior dropped (Stage 1B Option α). Use the
        sibling `strengthen_relation()` method for Hebbian weight bumps —
        encoder-explicit connect calls are now idempotent.

        Field-preservation rule (active row branch): caller passes _UNSET
        (the default) for a field → existing value preserved. Caller passes
        an explicit value → that value replaces existing.

        Returns:
            {'edge_id': str,
             'created': bool,              # new INSERT or revived archive
             'revived_from_archive': bool, # subset of created
             'updated': bool,              # active row had specified fields update
             'deltas': [{'field', 'old', 'new'}, ...],  # for trace emission
             'warnings': []}               # reserved for future warnings

        Raises ValueError if source or target node doesn't exist.
        """
        ts = _now()

        # Resolve defaults for fields that get a value-or-default (used in
        # INSERT and revive branches; active-update preserves unspecified).
        desc_specified = (description is not GraphDAL._UNSET)
        weight_specified = (weight is not GraphDAL._UNSET)
        es_specified = (encoding_source is not GraphDAL._UNSET)
        desc_value = description if desc_specified else ''
        weight_value = weight if weight_specified else 0.5
        es_value = encoding_source if es_specified else ''

        # Validate both nodes exist
        for nid, label in [(source_id, 'source'), (target_id, 'target')]:
            exists = self.conn.execute(
                'SELECT 1 FROM nodes WHERE id = ?', (nid,)).fetchone()
            if not exists:
                raise ValueError("Cannot create edge: %s node '%s' does not exist" % (
                    label, nid[:12]))

        # Find or create the physical edge (check both directions)
        edge_id = self.get_edge_id(source_id, target_id)

        if not edge_id:
            # Create new physical edge row
            edge_id = self._generate_edge_id(source_id, target_id)
            self.conn.execute(
                'INSERT OR IGNORE INTO edges '
                '(edge_id, source_id, target_id, weight, co_access_count, last_strengthened, created_at) '
                'VALUES (?, ?, ?, ?, 0, ?, ?)',
                (edge_id, source_id, target_id, weight_value, ts, ts))

        # Look up this (edge_id, relation) pair. PK is (edge_id, relation),
        # so at most one row exists — may be active or archived.
        existing = self.conn.execute(
            'SELECT description, weight, encoding_source, archived '
            'FROM edge_relations WHERE edge_id = ? AND relation = ?',
            (edge_id, relation)
        ).fetchone()

        result = {
            'edge_id': edge_id,
            'created': False,
            'revived_from_archive': False,
            'updated': False,
            'deltas': [],
            'warnings': [],
        }

        if existing is None:
            # Branch 1: No row → INSERT
            self.conn.execute(
                'INSERT INTO edge_relations '
                '(edge_id, relation, description, weight, encoding_source, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (edge_id, relation, desc_value, weight_value, es_value, ts))
            result['created'] = True
            # Treat all initial fields as create-deltas (old=None)
            result['deltas'] = [
                {'field': 'description', 'old': None, 'new': desc_value},
                {'field': 'weight', 'old': None, 'new': weight_value},
                {'field': 'encoding_source', 'old': None, 'new': es_value},
            ]

        elif existing[3] == 0:
            # Branch 2: Active row → field-preserving UPDATE
            old_desc, old_weight, old_es, _archived = existing
            updates = {}
            if desc_specified and description != old_desc:
                updates['description'] = description
                result['deltas'].append({
                    'field': 'description', 'old': old_desc, 'new': description})
            if weight_specified and weight != old_weight:
                updates['weight'] = weight
                result['deltas'].append({
                    'field': 'weight', 'old': old_weight, 'new': weight})
            if es_specified and encoding_source != old_es:
                updates['encoding_source'] = encoding_source
                result['deltas'].append({
                    'field': 'encoding_source', 'old': old_es, 'new': encoding_source})

            if updates:
                set_clause = ', '.join('%s = ?' % k for k in updates)
                self.conn.execute(
                    'UPDATE edge_relations SET %s '
                    'WHERE edge_id = ? AND relation = ?' % set_clause,
                    [*updates.values(), edge_id, relation])
                result['updated'] = True
            # If no specified fields differ → true no-op (no SQL write).

        else:
            # Branch 3: Archived row → revive with fresh state
            # Semantic 'fresh row': all fields reset to passed values + defaults.
            # Schema PK forces row reuse; trace events tell the lifecycle.
            old_desc, old_weight, old_es, _archived = existing
            self.conn.execute(
                'UPDATE edge_relations '
                'SET archived = 0, archived_at = NULL, archived_by = NULL, '
                '    description = ?, weight = ?, encoding_source = ?, '
                '    created_at = ? '
                'WHERE edge_id = ? AND relation = ?',
                (desc_value, weight_value, es_value, ts, edge_id, relation))
            result['created'] = True
            result['revived_from_archive'] = True
            result['deltas'] = [
                {'field': 'description', 'old': None, 'new': desc_value},
                {'field': 'weight', 'old': None, 'new': weight_value},
                {'field': 'encoding_source', 'old': None, 'new': es_value},
            ]

        # Update aggregate weight + last_strengthened on physical edge
        self._update_aggregate_weight(edge_id)
        self.conn.execute(
            'UPDATE edges SET last_strengthened = ? WHERE edge_id = ?', (ts, edge_id))
        commit_unless_batched(self.conn)

        # Enqueue for temporal extraction AND async edge re-embedding when the
        # description (part of compose_edge_text) changed. enqueue_edge() is a
        # cheap set.add; the embed_queue worker runs backfill_entity_dates +
        # backfill_edge_embeddings. Lazy import avoids a module-load cycle.
        _desc_changed = any(d.get('field') == 'description' for d in result['deltas'])
        if result['created'] or _desc_changed:
            # Invalidate the stored embedding so the worker re-embeds. New rows
            # are already NULL; only an existing row whose description changed
            # needs explicit NULLing.
            if _desc_changed and not result['created']:
                self.conn.execute(
                    'UPDATE edge_relations SET embedding = NULL, embedding_model = NULL '
                    'WHERE edge_id = ? AND relation = ?', (edge_id, relation))
                commit_unless_batched(self.conn)
            try:
                from . import embed_queue
                embed_queue.enqueue_edge(edge_id)
            except Exception as _eq_err:
                # No-silent-errors: was bare `except: pass` pre-migration.
                # The enqueue is a set.add — failure here is exotic (lock
                # contention, import collapse). Log so a real producer
                # outage is visible. Best-effort; we do not have a brain
                # reference here, so route via _log_error on the only
                # plausibly-reachable receiver (the connection's brain),
                # falling back to stderr.
                try:
                    import sys as _sys
                    print('[GraphDAL.add_relation] enqueue_edge failed: %s'
                          % _eq_err, file=_sys.stderr)
                except Exception:
                    pass

        return result

    def strengthen_relation(self, source_id, target_id, relation):
        """Hebbian strengthening — bump weight on existing active relation.

        Used by callers that want to record co-access / repeated reinforcement
        without changing description or encoding_source. Replaces the implicit
        auto-strengthen behavior that used to live inside add_relation().

        Behavior:
          - Active row exists → bump weight by LEARNING_RATE * 0.5 (capped at MAX_WEIGHT)
          - Archived row exists → no-op (won't resurrect via Hebbian)
          - No row → no-op

        Returns:
            {'strengthened': bool, 'old_weight': float|None, 'new_weight': float|None}
        """
        from .brain_constants import LEARNING_RATE, MAX_WEIGHT

        edge_id = self.get_edge_id(source_id, target_id)
        if not edge_id:
            return {'strengthened': False, 'old_weight': None, 'new_weight': None}

        row = self.conn.execute(
            'SELECT weight FROM edge_relations '
            'WHERE edge_id = ? AND relation = ? AND archived = 0',
            (edge_id, relation)
        ).fetchone()
        if not row:
            return {'strengthened': False, 'old_weight': None, 'new_weight': None}

        old_weight = row[0]
        new_weight = min(MAX_WEIGHT, old_weight + LEARNING_RATE * 0.5)
        if new_weight == old_weight:
            return {'strengthened': False, 'old_weight': old_weight,
                    'new_weight': old_weight}

        ts = _now()
        self.conn.execute(
            'UPDATE edge_relations SET weight = ?, created_at = ? '
            'WHERE edge_id = ? AND relation = ?',
            (new_weight, ts, edge_id, relation))
        self._update_aggregate_weight(edge_id)
        self.conn.execute(
            'UPDATE edges SET last_strengthened = ? WHERE edge_id = ?',
            (ts, edge_id))
        commit_unless_batched(self.conn)
        return {'strengthened': True, 'old_weight': old_weight,
                'new_weight': new_weight}

    def get_relations(self, edge_id, include_archived: bool = False):
        """Get active relations for an edge by edge_id.

        Returns list of dicts: [{relation, description, weight, encoding_source, created_at}, ...]
        include_archived=True surfaces archived rows too (forensics / recovery).
        """
        where = 'WHERE edge_id = ?'
        if not include_archived:
            where += ' AND archived = 0'
        rows = self.conn.execute(
            'SELECT relation, description, weight, encoding_source, created_at '
            'FROM edge_relations %s '
            'ORDER BY weight DESC' % where,
            (edge_id,)
        ).fetchall()
        return [{'relation': r[0], 'description': r[1] or '',
                 'weight': r[2], 'encoding_source': r[3] or '',
                 'created_at': r[4]}
                for r in rows]

    def remove_relation(self, source_id, target_id, relation, archived_by: str = 'unknown'):
        """Soft-archive a specific relation on a pair (v25).

        Commit is gated on self.conn.in_batch (commit_unless_batched) — a no-op
        inside the brain_batch `disconnect` envelope, a real commit standalone.

        Flips archived=1 on the matching row. Previously hard-DELETEd; the
        change preserves edge history for recovery. The edges aggregate row
        stays regardless — reads filter via edge_relations joins.
        """
        edge_id = self.get_edge_id(source_id, target_id)
        if not edge_id:
            return

        ts = _now()
        # NULL the embedding too — same pattern as delete_node_edges and
        # decay_edges. Symmetric with node archive (which DELETEs from
        # node_enrichments). Revive via add_relation Branch 3 re-embeds.
        self.conn.execute(
            'UPDATE edge_relations '
            'SET archived = 1, archived_at = ?, archived_by = ?, '
            '    embedding = NULL, embedding_model = NULL '
            'WHERE edge_id = ? AND relation = ? AND archived = 0',
            (ts, archived_by, edge_id, relation))

        # Recompute aggregate weight from remaining active relations
        self._update_aggregate_weight(edge_id)
        commit_unless_batched(self.conn)

    def rename_relation(self, edge_id: str, old_relation: str,
                        new_relation: str, encoding_source: str) -> None:
        """Rename a relation on an edge in place — updates the matching row's
        relation + encoding_source. No weight recompute: a rename changes neither
        weights nor the active-relation count. Commit gated on self.conn.in_batch.

        The relation string is part of compose_edge_text, so the stored embedding
        is now stale — NULL it here (storage-only invalidation, DAL-appropriate)
        and enqueue the edge for async re-embed by the embed_queue worker. Callers
        (reclassify, revise_edge) stay embedding-ignorant; the worker owns the
        actual re-embed via Brain.backfill_edge_embeddings.
        """
        self.conn.execute(
            "UPDATE edge_relations SET relation = ?, encoding_source = ?, "
            "embedding = NULL, embedding_model = NULL "
            "WHERE edge_id = ? AND relation = ?",
            (new_relation, encoding_source, edge_id, old_relation))
        commit_unless_batched(self.conn)
        try:
            from . import embed_queue
            embed_queue.enqueue_edge(edge_id)
        except Exception as _eq_err:
            import sys as _sys
            print('[GraphDAL.rename_relation] enqueue_edge failed: %s' % _eq_err,
                  file=_sys.stderr)

    def _update_aggregate_weight(self, edge_id):
        """Set edges.weight to max weight across ACTIVE relation rows.

        Archived relations do not contribute — they're history, not signal.
        When all relations on an edge are archived, edges.weight is
        explicitly set to 0 so weight-based reads (min_weight filters,
        get_well_connected) skip the orphan edges row. The row itself
        persists for edge_id stability; reads that JOIN edge_relations
        with archived=0 already get zero rows regardless.

        No silent no-op — the weight is always written (0 or max), so the
        edges row state reflects the truth of its active relations.
        """
        row = self.conn.execute(
            'SELECT MAX(weight) FROM edge_relations '
            'WHERE edge_id = ? AND archived = 0',
            (edge_id,)
        ).fetchone()
        new_weight = row[0] if row and row[0] is not None else 0.0
        self.conn.execute(
            'UPDATE edges SET weight = ? WHERE edge_id = ?',
            (new_weight, edge_id))


class SourceRefDAL:
    """Access layer for `node_source_refs` — node→trace_event pointers (v29:
    8-char hex) anchoring a node to the S0 moments it came from (episodic
    references). Extracted from GraphDAL: source_refs are NOT edges — they're
    the engram-cohort substrate (get_nodes_referencing). One row per
    (node_id, trace_id, position). Writers gate on conn.in_batch like every
    other DAL writer."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add_source_refs(self, node_id: str, trace_ids: List[str]) -> int:
        """Append trace_event pointers to a node (v29: 8-char hex strings).
        Used at NEW-node creation (remember()). Position derived from list
        order (1-indexed); first ref is the primary anchor.

        INSERT OR IGNORE — first-write-wins. Re-calling with the same refs
        is a no-op. For revise() use replace_source_refs() instead — that's
        where field-level replace semantics belong (decision 995ffeb1).

        Reject int input loudly per the v29 contract — coercion was removed
        because random hex generation made it unsafe.

        Returns count of refs newly inserted (existing ignored).
        """
        if not node_id or not trace_ids:
            return 0
        # Reject int input loudly — v29 contract is hex strings end-to-end.
        for tid in trace_ids:
            if not isinstance(tid, str):
                raise ValueError(
                    "add_source_refs: trace_ids must be strings, got "
                    "%s (%r). v29 trace ids are 8-char hex." % (
                        type(tid).__name__, tid))
        now = _now()
        rows = [(node_id, tid, idx + 1, now)
                for idx, tid in enumerate(trace_ids)]
        cur = self.conn.executemany(
            'INSERT OR IGNORE INTO node_source_refs '
            '(node_id, trace_id, position, created_at) '
            'VALUES (?, ?, ?, ?)',
            rows)
        commit_unless_batched(self.conn)
        return cur.rowcount

    def replace_source_refs(self, node_id: str, trace_ids: List[str]) -> int:
        """Replace the node's source_refs with the given list. v29: 8-char hex.

        Per the unified 2-class revise contract (decision 995ffeb1):
        - field present → REPLACE entire value
        - field absent → preserve (caller decides whether to call this)
        Called only by revise() when source_refs is in the update payload.
        Atomic: DELETE old rows, then INSERT new ones in a single transaction.

        Pass empty list to clear all refs. Returns count inserted.
        """
        if not node_id:
            return 0
        # Reject int input loudly — v29 contract is hex strings end-to-end.
        # Coercion was removed (reviewer F2) — silent int→hex was unsafe
        # against random hex generation; loud rejection is the doctrine.
        for tid in (trace_ids or []):
            if not isinstance(tid, str):
                raise ValueError(
                    "replace_source_refs: trace_ids must be strings, got "
                    "%s (%r). v29 trace ids are 8-char hex." % (
                        type(tid).__name__, tid))
        now = _now()
        self.conn.execute(
            'DELETE FROM node_source_refs WHERE node_id = ?', (node_id,))
        rows = [(node_id, tid, idx + 1, now)
                for idx, tid in enumerate(trace_ids or [])]
        if rows:
            self.conn.executemany(
                'INSERT INTO node_source_refs '
                '(node_id, trace_id, position, created_at) '
                'VALUES (?, ?, ?, ?)',
                rows)
        commit_unless_batched(self.conn)
        return len(rows)

    def get_source_refs(self, node_id: str) -> List[str]:
        """Trace ids anchoring this node, ordered by encoder-written
        position (primary first). v29: returns 8-char hex strings."""
        if not node_id:
            return []
        rows = self.conn.execute(
            'SELECT trace_id FROM node_source_refs '
            'WHERE node_id = ? ORDER BY position ASC',
            (node_id,)).fetchall()
        return [r[0] for r in rows]

    def get_nodes_referencing(self, trace_id: str) -> List[str]:
        """All node_ids anchored to a given trace (v29: 8-char hex). Engram
        cohort detection primitive — nodes that share a trace are part of
        the same memory at the substrate level. Rejects int input loudly."""
        if trace_id is None:
            return []
        if not isinstance(trace_id, str):
            raise ValueError(
                "get_nodes_referencing: trace_id must be a string, got "
                "%s (%r). v29 trace ids are 8-char hex." % (
                    type(trace_id).__name__, trace_id))
        rows = self.conn.execute(
            'SELECT node_id FROM node_source_refs WHERE trace_id = ?',
            (trace_id,)).fetchall()
        return [r[0] for r in rows]

    def delete_source_refs(self, node_id: str) -> None:
        """Delete all source_refs for a node (node_source_refs table). Used by
        the node delete-cascade so a hard delete leaves no orphan ref rows."""
        if not node_id:
            return
        self.conn.execute(
            'DELETE FROM node_source_refs WHERE node_id = ?', (node_id,))
        commit_unless_batched(self.conn)


def _now() -> str:
    """UTC ISO timestamp for edge operations."""
    return iso_now()


class VectorDAL:
    """Unified access layer for all node vectors (node_enrichments table, v23+).

    After v23 migration, ALL vectors live in node_enrichments with vector_type:
      _primary    — title+content blend (was in node_embeddings)
      _situation  — situation embedding (was in node_embeddings.situation_embedding)
                    NOTE: text column is DEPRECATED for _situation rows —
                    kv is canonical (see contract.py PROMOTED_FIELDS.situation).
                    Callers should pass empty string for text when storing _situation.
      title       — title-only diagnostic pointer
      high_meta   — situation + quotes
      other_meta  — reasoning + correction_pattern
      edge_context — edge descriptions
      question    — legacy V5 question vector
      anchor, bridge, keywords — legacy V5 enrichment vectors
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def store(self, node_id: str, vector_type: str, text: str,
              embedding: Optional[bytes], model: str = 'nomic-ai/nomic-embed-text-v1.5-Q') -> None:
        """Store or replace a single vector for a node.

        Uses deterministic ID '{node_id}__{vector_type}' for INSERT OR REPLACE.
        For bulk writes, prefer store_batch() — one round-trip instead of N.
        """
        vid = '%s__%s' % (node_id, vector_type)
        now = iso_now()
        try:
            self.conn.execute(
                '''INSERT OR REPLACE INTO node_enrichments
                   (id, node_id, vector_type, text, embedding, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (vid, node_id, vector_type, text[:500] if text else '',
                 embedding, model, now))
        except Exception as e:
            import sys
            print('[VectorDAL] store error for %s/%s: %s' % (
                node_id[:12], vector_type, e), file=sys.stderr)

    def store_batch(self, rows, model: str = 'nomic-ai/nomic-embed-text-v1.5-Q') -> int:
        """Batch insert-or-update many vectors in one executemany round-trip.

        Args:
            rows: iterable of (node_id, vector_type, text, embedding_blob).
                  Rows with embedding=None are skipped.
            model: model tag stored on each row.

        Returns: count of rows actually written.

        INSERT OR REPLACE handles both new inserts and updates to existing
        (node_id, vector_type) rows via the deterministic id key.
        """
        now = iso_now()
        prepared = []
        for node_id, vector_type, text, blob in rows:
            if blob is None or not node_id or not vector_type:
                continue
            vid = '%s__%s' % (node_id, vector_type)
            prepared.append((vid, node_id, vector_type,
                             text[:500] if text else '',
                             blob, model, now))
        if not prepared:
            return 0
        try:
            self.conn.executemany(
                '''INSERT OR REPLACE INTO node_enrichments
                   (id, node_id, vector_type, text, embedding, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                prepared)
            return len(prepared)
        except Exception as e:
            import sys
            print('[VectorDAL] store_batch error (%d rows): %s' % (len(prepared), e),
                  file=sys.stderr)
            return 0

    def get_primary(self, node_id: str) -> Optional[bytes]:
        """Get primary embedding blob for a node."""
        row = self.conn.execute(
            "SELECT embedding FROM node_enrichments WHERE node_id = ? AND vector_type = '_primary'",
            (node_id,)).fetchone()
        return row[0] if row else None

    def get_all_with_context(self, exclude_archived: bool = True,
                             types: List[str] = None,
                             project: str = None,
                             model: str = None) -> List[Dict[str, Any]]:
        """Get all primary embeddings with node context for recall STEP 3 scan.

        When `model` is given, only vectors produced by that model are returned.
        Stale-model rows are invisible — prevents cosine noise after a swap.
        """
        where = ["ne.vector_type = '_primary'"]
        params: List[Any] = []
        if exclude_archived:
            where.append('n.archived = 0')
        if types:
            where.append('n.type IN (%s)' % ','.join('?' * len(types)))
            params.extend(types)
        if project:
            where.append('(n.project = ? OR n.project IS NULL)')
            params.append(project)
        if model:
            where.append('ne.model = ?')
            params.append(model)
        where_sql = ' WHERE ' + ' AND '.join(where)
        rows = self.conn.execute(
            'SELECT ne.node_id, ne.embedding, n.personal, n.personal_context, '
            'n.confidence, n.critical, n.title, n.type, '
            'n.created_at, n.emotion, n.access_count '
            'FROM node_enrichments ne '
            'JOIN nodes n ON n.id = ne.node_id' + where_sql,
            params).fetchall()
        return [{'node_id': r[0], 'embedding': r[1], 'personal': r[2],
                 'personal_context': r[3], 'confidence': r[4],
                 'critical': r[5] or 0, 'title': r[6] or '', 'type': r[7] or '',
                 'created_at': r[8], 'emotion': r[9] or 0,
                 'access_count': r[10] or 0}
                for r in rows]

    def get_all_vectors(self, exclude_archived: bool = True,
                        vector_types: Optional[List[str]] = None,
                        model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get vectors for unified recall scan, optionally filtered.

        Args:
            exclude_archived: skip archived nodes (default True)
            vector_types: restrict to these types, e.g. ['_primary']. None = all.
            model: restrict to rows produced by this model. None = all.

        Returns: [{node_id, vector_type, embedding}] for rows with non-null embeddings.
        """
        sql = ('SELECT ne.node_id, ne.vector_type, ne.embedding '
               'FROM node_enrichments ne '
               'JOIN nodes n ON n.id = ne.node_id '
               'WHERE ne.embedding IS NOT NULL')
        params: List[Any] = []
        if exclude_archived:
            sql += ' AND n.archived = 0'
        if vector_types:
            ph = ','.join('?' * len(vector_types))
            sql += f' AND ne.vector_type IN ({ph})'
            params.extend(vector_types)
        if model:
            sql += ' AND ne.model = ?'
            params.append(model)
        rows = self.conn.execute(sql, params).fetchall()
        return [{'node_id': r[0], 'vector_type': r[1], 'embedding': r[2]}
                for r in rows]

    def get_all_situations(self, model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all situation embeddings for cosine scan (recall STEP 3.5b).

        When `model` is given, only rows produced by that model are returned —
        stale-model vectors are excluded so cosine scans stay in matched geometry.
        """
        sql = ("SELECT ne.node_id, ne.embedding "
               "FROM node_enrichments ne "
               "JOIN nodes n ON n.id = ne.node_id "
               "WHERE ne.vector_type = '_situation' AND ne.embedding IS NOT NULL "
               "AND n.archived = 0")
        params: tuple = ()
        if model:
            sql += " AND ne.model = ?"
            params = (model,)
        rows = self.conn.execute(sql, params).fetchall()
        return [{'node_id': r[0], 'situation_embedding': r[1]} for r in rows]

    def find_missing(self, vector_type: str, limit: int = 50,
                     model: Optional[str] = None,
                     node_ids: Optional[set] = None,
                     require_kv_keys_any: Optional[List[str]] = None,
                     source_kv_keys: Optional[List[str]] = None,
                     require_described_edge: bool = False) -> List[Dict[str, Any]]:
        """Find active nodes whose vector for `vector_type` is missing or stale.

        A row is "present" only if it has a non-null embedding AND (if `model`
        is given) was produced by the same model. On model swaps, rows
        embedded by prior models become eligible for re-embedding.

        When `node_ids` is given, scope the scan to just those IDs (queue
        drain path — don't re-scan the whole graph on every tick).

        When `require_kv_keys_any` is given, restrict to nodes that have at
        least one of those keys present in node_metadata_kv with a non-empty
        value. **OR semantics** — node passes if ANY listed key matches; the
        list is *required-any-of*, not *all-of-these*. This prevents the
        field-cohort backfill from filling its top-N batch with nodes that
        lack the source field — older nodes-with-the-field would otherwise
        be stuck below the LIMIT cutoff and never embedded.

        `source_kv_keys` is an accepted alias for `require_kv_keys_any`
        (the prior name was misleading — sounded like "the keys that ARE
        the source" rather than "keys that must exist"). Either kwarg works;
        if both are provided, `require_kv_keys_any` wins.

        Returns [{id, title, content}] ordered by recency of access.
        """
        # Resolve alias: prefer the new explicit name; fall back to the old.
        if require_kv_keys_any is None:
            require_kv_keys_any = source_kv_keys
        where = ['n.archived = 0']
        params: list = []

        if model:
            where.append('''n.id NOT IN (
                SELECT ne.node_id FROM node_enrichments ne
                WHERE ne.vector_type = ?
                  AND ne.embedding IS NOT NULL
                  AND ne.model = ?
            )''')
            params.extend([vector_type, model])
        else:
            where.append('''n.id NOT IN (
                SELECT ne.node_id FROM node_enrichments ne
                WHERE ne.vector_type = ? AND ne.embedding IS NOT NULL
            )''')
            params.append(vector_type)

        if node_ids:
            ids = list(node_ids)
            ph = ','.join('?' * len(ids))
            where.append('n.id IN (%s)' % ph)
            params.extend(ids)

        if require_kv_keys_any:
            ph = ','.join('?' * len(require_kv_keys_any))
            where.append(
                'EXISTS (SELECT 1 FROM node_metadata_kv kv '
                'WHERE kv.node_id = n.id AND kv.key IN (%s) '
                # trim() mirrors the text-builder's `val.strip()` — a
                # whitespace-only value is NOT eligible (it yields no embed
                # text), so it neither clogs the batch nor false-trips the
                # dead-handler alarm. Keeps "eligible <=> yields text" exact.
                "AND kv.value IS NOT NULL AND trim(kv.value) != '')" % ph)
            params.extend(require_kv_keys_any)

        if require_described_edge:
            # edge_context group: its only source is _edge_descriptions, which
            # lives on edges, not node_metadata_kv — so require_kv_keys_any can't
            # gate it. Without this clause the edgeless nodes (no described edge,
            # never get a vector) sit at the front of the last_accessed queue
            # forever and starve the edged nodes. Mirror
            # GraphDAL.get_edge_descriptions_for's eligibility filter EXACTLY
            # (same exclusions, same min length) so "eligible" ⇔ "yields text".
            excl = sorted(DEFAULT_EXCLUDED_RELATIONS | {'community_member'})
            excl_ph = ','.join('?' * len(excl))
            where.append(
                'EXISTS (SELECT 1 FROM edges e '
                'JOIN edge_relations er ON er.edge_id = e.edge_id '
                'WHERE (e.source_id = n.id OR e.target_id = n.id) '
                'AND er.archived = 0 '
                'AND er.relation NOT IN (%s) '
                'AND er.description IS NOT NULL '
                'AND length(er.description) > ?)' % excl_ph)
            params.extend(excl)
            params.append(EDGE_CONTEXT_MIN_DESC_LENGTH)

        sql = ('SELECT n.id, n.title, n.content FROM nodes n '
               'WHERE ' + ' AND '.join(where) +
               ' ORDER BY n.last_accessed DESC LIMIT ?')
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [{'id': r[0], 'title': r[1] or '', 'content': r[2] or ''} for r in rows]

    def delete_for_node(self, node_id: str) -> int:
        """Delete all vectors for a node."""
        self.conn.execute('DELETE FROM node_enrichments WHERE node_id = ?', (node_id,))
        return self.conn.execute('SELECT changes()').fetchone()[0]

    def get_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all vectors for a single node."""
        rows = self.conn.execute(
            'SELECT vector_type, text, embedding FROM node_enrichments WHERE node_id = ?',
            (node_id,)).fetchall()
        return [{'vector_type': r[0], 'text': r[1], 'embedding': r[2]} for r in rows]

    def get_coverage_stats(self) -> Dict[str, Any]:
        """Vector coverage statistics."""
        total_nodes = self.conn.execute(
            'SELECT COUNT(*) FROM nodes WHERE archived = 0').fetchone()[0]
        by_type = self.conn.execute(
            'SELECT vector_type, COUNT(DISTINCT node_id) FROM node_enrichments '
            'WHERE embedding IS NOT NULL GROUP BY vector_type'
        ).fetchall()
        return {
            'total_nodes': total_nodes,
            'by_type': {r[0]: r[1] for r in by_type},
        }

    def count(self) -> int:
        """Count total vectors."""
        return self.conn.execute(
            'SELECT COUNT(*) FROM node_enrichments WHERE embedding IS NOT NULL'
        ).fetchone()[0]


class EntityDatesDAL:
    """Access layer for the `entity_dates` table — temporal intervals per
    node/edge that power recall_by_time. One row per (entity_kind, entity_id,
    interval); an empty interval set is recorded as a single sentinel row so the
    backfill indexer treats the entity as processed.

    The sentinel source string lives in temporal_extraction (its owner); it's
    imported lazily here to avoid a module-load cycle.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def write(self, entity_kind: str, entity_id: str,
              intervals: List[tuple]) -> int:
        """Replace all rows for (entity_kind, entity_id). Empty `intervals` →
        one sentinel row (processed-no-dates). Returns the count of REAL
        interval rows written (sentinel doesn't count). Idempotent.

        Caller-managed transaction: like the function it replaces, this does
        NOT commit — the backfill batch path (embed_queue drain) owns the
        BEGIN/COMMIT on the connection passed at construction, and routes via
        conn_bg_writer off the foreground slot. Constructing EntityDatesDAL(conn)
        with the handed connection preserves that routing.
        """
        from .temporal_extraction import _SENTINEL_SOURCE, MAX_INTERVALS_PER_ENTITY
        self.conn.execute(
            'DELETE FROM entity_dates WHERE entity_kind = ? AND entity_id = ?',
            (entity_kind, entity_id))
        rows = [(entity_kind, entity_id, s, e, src, raw)
                for (s, e, src, raw) in intervals]
        if not rows:
            self.conn.execute(
                'INSERT INTO entity_dates (entity_kind, entity_id, start_ts, '
                'end_ts, extraction_source, raw_text) VALUES (?, ?, 0, 0, ?, NULL)',
                (entity_kind, entity_id, _SENTINEL_SOURCE))
            return 0
        if len(rows) > MAX_INTERVALS_PER_ENTITY:
            rows = rows[:MAX_INTERVALS_PER_ENTITY]
        self.conn.executemany(
            'INSERT OR REPLACE INTO entity_dates (entity_kind, entity_id, '
            'start_ts, end_ts, extraction_source, raw_text) VALUES (?, ?, ?, ?, ?, ?)',
            rows)
        return len(rows)

    def node_entities_in_window(self, start_ts: int, end_ts: int) -> List[str]:
        """Non-archived node ids whose date interval overlaps [start_ts, end_ts]
        (sentinel rows excluded). The recall_by_time 'event' anchor, node side."""
        from .temporal_extraction import _SENTINEL_SOURCE
        rows = self.conn.execute(
            "SELECT DISTINCT ed.entity_id FROM entity_dates ed "
            "JOIN nodes n ON n.id = ed.entity_id "
            "WHERE ed.entity_kind = 'node' AND ed.extraction_source != ? "
            "AND ed.start_ts <= ? AND ed.end_ts >= ? AND n.archived = 0",
            (_SENTINEL_SOURCE, end_ts, start_ts)).fetchall()
        return [r[0] for r in rows]

    def edge_entities_in_window(self, start_ts: int, end_ts: int) -> List[str]:
        """Edge ids whose date interval overlaps [start_ts, end_ts] (sentinel
        excluded). Archived-relation filtering happens downstream."""
        from .temporal_extraction import _SENTINEL_SOURCE
        rows = self.conn.execute(
            "SELECT DISTINCT entity_id FROM entity_dates "
            "WHERE entity_kind = 'edge' AND extraction_source != ? "
            "AND start_ts <= ? AND end_ts >= ?",
            (_SENTINEL_SOURCE, end_ts, start_ts)).fetchall()
        return [r[0] for r in rows]

    def node_ids_without_dates(self) -> List[str]:
        """Non-archived node ids with no entity_dates rows yet — the cold-start
        backfill work-list (a node is 'done' once it has rows incl. sentinel)."""
        rows = self.conn.execute(
            "SELECT n.id FROM nodes n "
            "LEFT JOIN entity_dates e ON e.entity_id = n.id AND e.entity_kind = 'node' "
            "WHERE n.archived = 0 AND e.entity_id IS NULL").fetchall()
        return [r[0] for r in rows]

    def edge_ids_without_dates(self) -> List[str]:
        """Active edge ids with no entity_dates rows yet — cold-start work-list."""
        rows = self.conn.execute(
            "SELECT DISTINCT er.edge_id FROM edge_relations er "
            "LEFT JOIN entity_dates e ON e.entity_id = er.edge_id AND e.entity_kind = 'edge' "
            "WHERE (er.archived IS NULL OR er.archived = 0) AND e.entity_id IS NULL").fetchall()
        return [r[0] for r in rows]


# TelemetryDAL — REMOVED 2026-04-05 (brain_telemetry table dropped, never used)
