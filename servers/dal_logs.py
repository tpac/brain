"""
brain — Data Access Layer (DAL): logs & traces (brain_logs.db)

Access layer for the four brain_logs.db-backed concerns: raw debug/hook
logs (LogsDAL), versioned prompt/config templates (InteractionDAL), the
fractal trace-event chain (TraceDAL), and per-session ephemeral state
(SessionStateDAL). Split out of dal.py (which now holds the brain.db
classes) along the one structural boundary that matters: which SQLite
file owns the table.

Usage in brain.py:
    from servers.dal_logs import LogsDAL, TraceDAL

    self._logs = LogsDAL(self.logs_conn)
    self._trace_dal = TraceDAL(self.logs_conn)

See servers/dal.py for the brain.db-backed classes (nodes, edges, vectors, ...).
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

    def _row_to_event(self, r) -> Dict[str, Any]:
        """Map ONE canonical-order trace_events row tuple → the event dict.

        The single column→dict mapping for the whole DAL: every reader's SELECT
        is realigned to feed this exact column order, so the index→field binding
        lives in ONE place instead of ~8 hand-maintained copies that drifted into
        subtly different orders. The SELECT feeding this MUST be exactly:

            id, chain_id, scale, event_type, ref_type, ref_id,
            summary, metadata, session_id, created_at

        ref_type/ref_id/summary are ''-guarded (they're '' at write, never NULL
        in practice — the guard keeps the dict total). metadata is decoded.
        """
        if len(r) != 10:
            # Loud-by-default: a mis-arity SELECT would otherwise mis-index
            # silently (over-select ignores extra cols; under-select IndexErrors
            # at a confusing offset). Name the contract at the boundary instead.
            raise ValueError(
                "_row_to_event expects 10 columns in canonical order "
                "(id, chain_id, scale, event_type, ref_type, ref_id, summary, "
                "metadata, session_id, created_at), got %d" % len(r))
        return {
            'id': r[0], 'chain_id': r[1], 'scale': r[2], 'event_type': r[3],
            'ref_type': r[4] or '', 'ref_id': r[5] or '', 'summary': r[6] or '',
            'metadata': self._decode_metadata(r[7]),
            'session_id': r[8] or '', 'created_at': r[9],
        }

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
        return [self._row_to_event(r) for r in rows]

    def get_chain(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get all events in a trace chain, ordered by time. Each event is the
        full canonical row (incl. its own chain_id — = the queried id)."""
        rows = self.conn.execute(
            'SELECT id, chain_id, scale, event_type, ref_type, ref_id, summary, metadata, session_id, created_at '
            'FROM trace_events WHERE chain_id = ? ORDER BY created_at ASC',
            (chain_id,)).fetchall()
        return [self._row_to_event(r) for r in rows]

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
            'SELECT id, chain_id, scale, event_type, ref_type, ref_id, summary, metadata, session_id, created_at '
            'FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?' % where,
            params + [limit]).fetchall()
        out = [self._row_to_event(r) for r in rows]
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
        return [self._row_to_event(r) for r in rows]

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

    def event_vector_rows(self, scale: str = 's0',
                          ref_types: Optional[List[str]] = None,
                          since: str = None) -> List[tuple]:
        """UNCAPPED embedded-trace pull for field consumers (LAF episodic matrix).

        Returns [(chain_id, session_id, created_at, vector)] for every embedded
        trace matching scale/ref_types, created_at ASC. `since` (exclusive ISO
        bound) makes refreshes incremental — callers keep a resident matrix and
        append only new rows. Deliberately separate from filter_event_vectors:
        that is the recall_episodes BROWSING scan (newest-first, EPISODE_MAX_LIMIT
        capped); this is the substrate pull for a scorer that must see the whole
        history (the newest-500 cap was a coverage ceiling, not a feature —
        2026-07-02, eval/laf/composition_probe.md).
        """
        conditions = ['tem.vector IS NOT NULL']
        params: List[Any] = []
        if scale:
            conditions.append('te.scale = ?')
            params.append(scale)
        if ref_types:
            conditions.append('te.ref_type IN (%s)' % ','.join('?' * len(ref_types)))
            params.extend(ref_types)
        if since:
            conditions.append('te.created_at > ?')
            params.append(since)
        rows = self.conn.execute(
            'SELECT te.chain_id, te.session_id, te.created_at, tem.vector '
            'FROM trace_events te '
            'JOIN trace_embeddings tem ON tem.trace_id = te.id '
            'WHERE %s ORDER BY te.created_at ASC' % ' AND '.join(conditions),
            params).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def get_chains(self, session_id: str = '', scale: str = '',
                   hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        """Get complete chains grouped, with all events and metadata.

        Returns: [{chain_id, scale, session_id, events: [{id, event_type,
        ref_type, ref_id, summary, metadata, created_at}]}] — chain_id/scale/
        session_id are chain-level; each event is the chain-relative subset
        (no chain_id/scale/session_id). Ordered by most recent chain first.
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
            'SELECT id, chain_id, scale, event_type, ref_type, ref_id, summary, metadata, session_id, created_at '
            'FROM trace_events WHERE %s ORDER BY created_at DESC' % where,
            params).fetchall()

        # Group by chain_id, preserve order of first appearance.
        # A chain belongs to one session; we record the session_id from
        # the first event seen in each chain. Per-event `id` is included so a
        # grouped view stays drillable (get_trace by id); chain_id/scale/
        # session_id are chain-level — deliberately NOT repeated per event (the
        # render layer propagates them onto events). The full canonical row is
        # built once via _row_to_event, then projected to the chain-relative
        # event subset, so the decode/guard logic still lives in one place.
        EVENT_KEYS = ('id', 'event_type', 'ref_type', 'ref_id',
                      'summary', 'metadata', 'created_at')
        chains = {}
        chain_order = []
        for r in rows:
            ev = self._row_to_event(r)
            cid = ev['chain_id']
            if cid not in chains:
                chains[cid] = {'chain_id': cid, 'scale': ev['scale'],
                               'session_id': ev['session_id'], 'events': []}
                chain_order.append(cid)
            chains[cid]['events'].append({k: ev[k] for k in EVENT_KEYS})

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
            'SELECT id, chain_id, scale, event_type, ref_type, ref_id, summary, metadata, session_id, created_at '
            'FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?' % where,
            params + [limit]).fetchall()

        return [self._row_to_event(r) for r in rows]

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
                          before: int = None, after: int = None,
                          with_judge_output: bool = True,
                          exclude_trace_id: str = None) -> List[Dict[str, Any]]:
        """Get chronological turns for a session from S0 + S1 traces.

        Returns: [{role, trace_id, content, timestamp, judge_output}]
        (s0/conversation.py get_conversation and the encoder's
        _gather_messages derive their shapes from this.)

        One turn per trace row — never grouped by chain. A chain can hold N
        user messages (an interrupted turn never fires Stop, so stop_counter
        never advances and the next prompt lands in the same chain); keying
        on chain would overwrite all but the last user message.

        Cross-references S1 delta (additionalContext) via recall_chain in
        metadata to fill judge_output on user turns.

        Args:
            session_id: Full session UUID
            limit: Max turns to return (most recent if no around_timestamp)
            around_timestamp: ISO timestamp to center the window on.
                If provided, returns `before` turns before + `after` turns after
                this timestamp instead of the most recent `limit`.
            before: Turns before around_timestamp (default 10)
            after: Turns after around_timestamp (default 5)
            with_judge_output: fill judge_output on user turns (one extra
                query over the window's recall chains). Callers that only
                read role/content pass False.
            exclude_trace_id: trace_event id of one row to drop. The
                user_message trace is written at prompt-arrival (not Stop),
                so mid-turn readers that want PREVIOUS turns only — the
                surface conversation window, prior-query embeddings — pass
                the current prompt's trace id (returned by the append).
                Keyed on the trace row, NOT the chain: after an interrupt
                the current chain also holds the previous real prompt,
                which must stay in the window. Readers that want the
                conversation as-is (Scribe, historic lookups) omit it.
        """
        # v29: select `id` (8-char hex trace_event.id) so callers can render
        # [trace:<hex>] markers — the encoder copies these into source_refs.
        # Conversation window = the conversational ref_types defined by the S0
        # turn-classification contract (single source of truth for what the
        # encoder reads — see trace_contract.CONVERSATIONAL_REF_TYPES).
        from .trace_contract import CONVERSATIONAL_REF_TYPES
        _refs = ",".join("?" * len(CONVERSATIONAL_REF_TYPES))
        base_sql = (
            "SELECT id, ref_type, summary, metadata, created_at "
            "FROM trace_events WHERE scale = 's0' AND session_id = ? "
            "AND event_type IN ('K', 'delta') AND ref_type IN (%s) " % _refs)
        params = (session_id, *CONVERSATIONAL_REF_TYPES)
        if exclude_trace_id:
            # In SQL (not post-filter) so the LIMIT below still returns
            # `limit` full rows of previous turns.
            base_sql += "AND id != ? "
            params = (*params, exclude_trace_id)
        if around_timestamp:
            # Historic window — needs the whole session to locate the center.
            rows = self.conn.execute(
                base_sql + "ORDER BY created_at ASC, rowid ASC",
                params).fetchall()
        else:
            # Hot path (hook_recall, once per user prompt): newest-first with
            # a SQL LIMIT so cost is O(limit), not O(session length) —
            # idx_trace_session_created walks created_at DESC and stops.
            # rowid breaks same-timestamp ties in insert order.
            rows = self.conn.execute(
                base_sql + "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (*params, limit)).fetchall()
            rows.reverse()

        # Build result in encoding_agent._gather_messages() shape —
        # one turn per row, chronological.
        turns = []
        user_refs = []  # (turn index, recall_chain) for the judge_output fill
        for r in rows:
            meta = self._decode_metadata(r[3])
            # Content lives in metadata (full), summary is truncated for display
            content = meta.get('content', '') or r[2] or ''
            if r[1] == 'assistant_message':
                turns.append({
                    'role': 'assistant',
                    'trace_id': r[0],
                    'content': content,
                    'timestamp': r[4],
                    'judge_output': None,
                })
            else:
                # Incoming side (user_message today; self_message if flipped on)
                rc = meta.get('recall_chain', '')
                if rc and with_judge_output:
                    user_refs.append((len(turns), rc))
                turns.append({
                    'role': 'user',
                    'trace_id': r[0],
                    'content': content,
                    'timestamp': r[4],
                    'judge_output': '',
                })

        # Cross-reference S1 delta (additionalContext) for judge_output
        if user_refs:
            recall_chains = {rc for _, rc in user_refs}
            placeholders = ','.join('?' for _ in recall_chains)
            s1_rows = self.conn.execute(
                "SELECT chain_id, metadata FROM trace_events "
                "WHERE scale = 's1' AND event_type = 'delta' AND ref_type = 'additionalContext' "
                "AND chain_id IN (%s)" % placeholders,
                list(recall_chains)).fetchall()
            judge_outputs = {r[0]: self._decode_metadata(r[1]).get('content', '')
                             for r in s1_rows}
            for i, rc in user_refs:
                turns[i]['judge_output'] = judge_outputs.get(rc, '')

        # Apply windowing (default mode is already limited in SQL)
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

        Returns full canonical trace rows (id/chain_id/scale/event_type/ref_type/
        ref_id/summary/metadata/session_id/created_at). Caller renders to text
        and embeds (it reads metadata/ref_type/summary; chain_id/ref_id ride
        along for free via the shared row mapping).
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
            'SELECT te.id, te.chain_id, te.scale, te.event_type, te.ref_type, te.ref_id, '
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
        return [self._row_to_event(r) for r in rows]


class SessionStateDAL:
    """Access layer for session_state table in brain_logs.db.

    Backs SessionContext's save/load/seed: each session collapses to a single
    `_session_context` JSON blob row, keyed by (session_id, key, node_id).
    get/set are the read/upsert pair; ensure_default seeds without clobbering
    a racing thread's already-mutated row.
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


