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

    # ── access_log — REMOVED 2026-04-05 ──
    # Table dropped. 415K rows, 151K/day writes, never used for anything meaningful.
    # Node access_count on nodes table is the durable stat.

    # ── recall_log — REMOVED 2026-04-05 ──
    # All recall_log write methods deleted. Traces (trace_events) are source of truth.
    # Table still exists with historical data. Dashboard reads from traces.

    # ── miss_log — REMOVED 2026-04-05 (table dropped) ──

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
            "AND created_at < datetime('now', '-30 days')")
        stats['debug_log_pruned'] = cur.rowcount

        # suggest_log: REMOVED 2026-04-05 (table dropped)

        # health_log: REMOVED 2026-04-05 (table dropped)

        # hook_errors: 30 days (surfaced ones only — unsurfaced kept until shown)
        try:
            cur = self.conn.execute(
                "DELETE FROM hook_errors WHERE surfaced = 1 "
                "AND created_at < datetime('now', '-30 days')")
            stats['hook_errors_pruned'] = cur.rowcount
        except Exception:
            stats['hook_errors_pruned'] = 0

        self.conn.commit()

        # --- Graph DB orphan cleanup ---
        if graph_conn:
            cur = graph_conn.execute(
                "DELETE FROM node_vectors WHERE node_id NOT IN (SELECT id FROM nodes)")
            stats['orphaned_vectors'] = cur.rowcount

            cur = graph_conn.execute(
                "DELETE FROM edges WHERE source_id NOT IN (SELECT id FROM nodes) "
                "OR target_id NOT IN (SELECT id FROM nodes)")
            stats['orphaned_edges'] = cur.rowcount

            cur = graph_conn.execute(
                "DELETE FROM node_embeddings WHERE node_id NOT IN (SELECT id FROM nodes)")
            stats['orphaned_embeddings'] = cur.rowcount

            cur = graph_conn.execute(
                "DELETE FROM node_metadata WHERE node_id NOT IN (SELECT id FROM nodes)")
            stats['orphaned_metadata'] = cur.rowcount

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
        """Unified log query across hook_errors, debug_log, and signal_queue.

        Args:
            source: 'errors' (hook_errors), 'debug' (debug_log), 'signals'
                    (signal_queue), or 'all' (merged, sorted by time).
            hours: look back window (default 24).
            level: filter by level — 'error', 'critical', 'warning', or 'all'.
            hook_name: filter hook_errors by hook name (e.g. 'hook_recall').
            limit: max results per source (capped at 200).

        Returns: dict with 'entries' list and 'counts' summary.
        """
        limit = min(max(limit, 1), 200)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
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

        # ── signal_queue ──
        if source in ('signals', 'all'):
            try:
                conditions = ['created_at > ?']
                params = [cutoff]
                where = ' AND '.join(conditions)
                rows = self.conn.execute(
                    'SELECT id, producer, signal_type, priority, content, dismissed, created_at '
                    'FROM signal_queue WHERE %s ORDER BY priority DESC, created_at DESC LIMIT ?' % where,
                    params + [limit]
                ).fetchall()
                count_row = self.conn.execute(
                    'SELECT COUNT(*) FROM signal_queue WHERE %s' % where, params
                ).fetchone()
                counts['signals'] = count_row[0] if count_row else 0
                for r in rows:
                    entries.append({
                        'source': 'signal_queue', 'id': r[0], 'producer': r[1],
                        'level': r[2], 'priority': r[3],
                        'message': (r[4] or '')[:300], 'dismissed': bool(r[5]),
                        'created_at': r[6]
                    })
            except Exception:
                counts['signals'] = 0

        # Sort merged entries by time descending
        entries.sort(key=lambda e: e.get('created_at', ''), reverse=True)
        if source == 'all':
            entries = entries[:limit]

        return {'entries': entries, 'counts': counts}


class InteractionDAL:
    """Access layer for interactions — versioned templates for system boundaries.

    Every learnable boundary (judge prompt, encoding prompt, voice format,
    signal assembly) is an interaction. Versioned, traceable, optimizable
    by higher scales.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def register(self, name: str, template: str, parameters: str = '',
                 created_by: str = 'anchor') -> Dict[str, Any]:
        """Register a new version of an interaction. Auto-increments version."""
        now = datetime.now(timezone.utc).isoformat()
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
        self.conn.commit()
        return {'name': name, 'version': version, 'id': self.conn.execute(
            'SELECT last_insert_rowid()').fetchone()[0]}

    def get_latest(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the latest version of an interaction."""
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
        """List all interactions with their latest version."""
        rows = self.conn.execute(
            'SELECT name, MAX(version) as v, COUNT(*) as versions '
            'FROM interactions GROUP BY name ORDER BY name').fetchall()
        return [{'name': r[0], 'latest_version': r[1], 'total_versions': r[2]}
                for r in rows]


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

    def append(self, chain_id: str, scale: str, event_type: str,
               ref_type: str = '', ref_id: str = '', summary: str = '',
               metadata: Optional[Dict] = None, session_id: str = '',
               interaction_id: int = None) -> int:
        """Append an event to a trace chain. Returns event id.

        Validates against trace_contract before writing.
        """
        from .trace_contract import validate_trace_event
        ok, error = validate_trace_event(scale, event_type, ref_type)
        if not ok:
            raise ValueError("Trace contract violation: %s" % error)

        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata) if metadata else None
        cursor = self.conn.execute(
            'INSERT INTO trace_events '
            '(chain_id, scale, event_type, ref_type, ref_id, summary, metadata, session_id, interaction_id, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (chain_id, scale, event_type, ref_type, ref_id,
             summary if summary else '', meta_json, session_id, interaction_id, now))
        self.conn.commit()
        return cursor.lastrowid

    def get_chain(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get all events in a trace chain, ordered by time."""
        rows = self.conn.execute(
            'SELECT id, scale, event_type, ref_type, ref_id, summary, metadata, created_at '
            'FROM trace_events WHERE chain_id = ? ORDER BY created_at ASC',
            (chain_id,)).fetchall()
        results = []
        for r in rows:
            meta = {}
            try:
                meta = json.loads(r[6]) if r[6] else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            results.append({
                'id': r[0], 'scale': r[1], 'event_type': r[2],
                'ref_type': r[3], 'ref_id': r[4], 'summary': r[5],
                'metadata': meta, 'created_at': r[7]})
        return results

    def get_recent(self, scale: str = '', hours: int = 24,
                   event_type: str = '', limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trace events, optionally filtered by scale and type."""
        conditions = ['created_at > ?']
        params = [(datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()]
        if scale:
            conditions.append('scale = ?')
            params.append(scale)
        if event_type:
            conditions.append('event_type = ?')
            params.append(event_type)
        where = ' AND '.join(conditions)
        rows = self.conn.execute(
            'SELECT id, chain_id, scale, event_type, ref_type, ref_id, summary, created_at '
            'FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?' % where,
            params + [limit]).fetchall()
        return [{'id': r[0], 'chain_id': r[1], 'scale': r[2], 'event_type': r[3],
                 'ref_type': r[4], 'ref_id': r[5], 'summary': r[6], 'created_at': r[7]}
                for r in rows]

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
        params = [(datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()]
        if session_id:
            conditions.append('session_id = ?')
            params.append(session_id)
        if scale:
            conditions.append('scale = ?')
            params.append(scale)
        where = ' AND '.join(conditions)

        rows = self.conn.execute(
            'SELECT chain_id, scale, event_type, ref_type, ref_id, summary, metadata, created_at '
            'FROM trace_events WHERE %s ORDER BY created_at DESC' % where,
            params).fetchall()

        # Group by chain_id, preserve order of first appearance
        chains = {}
        chain_order = []
        for r in rows:
            cid = r[0]
            if cid not in chains:
                chains[cid] = {'chain_id': cid, 'scale': r[1], 'events': []}
                chain_order.append(cid)
            meta = {}
            try:
                meta = json.loads(r[6]) if r[6] else {}
            except (json.JSONDecodeError, TypeError):
                pass
            chains[cid]['events'].append({
                'event_type': r[2], 'ref_type': r[3] or '', 'ref_id': r[4] or '',
                'summary': r[5] or '', 'metadata': meta, 'created_at': r[7]})

        # Reverse events within each chain to chronological order
        for cid in chain_order:
            chains[cid]['events'].reverse()

        result = [chains[cid] for cid in chain_order[:limit]]
        return result

    def get_by_ref_type(self, ref_type: str, scale: str = '',
                        hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """Get events filtered by ref_type.

        Use: "all corrections", "all recall_hits", "all encoding_runs".
        """
        conditions = ['ref_type = ?', 'created_at > ?']
        params = [ref_type, (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()]
        if scale:
            conditions.append('scale = ?')
            params.append(scale)
        where = ' AND '.join(conditions)

        rows = self.conn.execute(
            'SELECT id, chain_id, scale, event_type, ref_type, ref_id, summary, metadata, created_at '
            'FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?' % where,
            params + [limit]).fetchall()

        results = []
        for r in rows:
            meta = {}
            try:
                meta = json.loads(r[7]) if r[7] else {}
            except (json.JSONDecodeError, TypeError):
                pass
            results.append({
                'id': r[0], 'chain_id': r[1], 'scale': r[2], 'event_type': r[3],
                'ref_type': r[4] or '', 'ref_id': r[5] or '', 'summary': r[6] or '',
                'metadata': meta, 'created_at': r[8]})
        return results

    def get_outcomes(self, chain_id: str = '', scale: str = '',
                     hours: int = 168) -> List[Dict[str, Any]]:
        """Get outcome events, optionally for a specific chain or scale.

        Use: S3 checks which S1 chains got corrected vs validated.
        Default 168h = 7 days.
        """
        conditions = ["event_type = 'outcome'", 'created_at > ?']
        params = [(datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()]
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
            meta = {}
            try:
                meta = json.loads(r[6]) if r[6] else {}
            except (json.JSONDecodeError, TypeError):
                pass
            results.append({
                'id': r[0], 'chain_id': r[1], 'scale': r[2], 'event_type': 'outcome',
                'ref_type': r[3] or '', 'ref_id': r[4] or '', 'summary': r[5] or '',
                'metadata': meta, 'created_at': r[7]})
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
        params = [(datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()]
        if scale:
            conditions.append('scale = ?')
            params.append(scale)
        where = ' AND '.join(conditions)

        rows = self.conn.execute(
            'SELECT %s, COUNT(*) FROM trace_events WHERE %s GROUP BY %s' % (field, where, field),
            params).fetchall()

        return {r[0] or '': r[1] for r in rows}

    def get_session_turns(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get chronological turns for a session from S0 + S1 traces.

        Returns same shape as encoding_agent._gather_messages():
        [{role, content, signal_type, timestamp, judge_output, recalled_raw}]

        Groups S0 K (user_message) + S0 delta (assistant_message) per chain,
        cross-references S1 delta (additionalContext) via recall_chain in metadata.
        """
        # Get S0 events for this session, chronologically
        rows = self.conn.execute(
            "SELECT chain_id, event_type, ref_type, summary, metadata, created_at "
            "FROM trace_events WHERE scale = 's0' AND session_id = ? "
            "AND event_type IN ('K', 'delta') AND ref_type IN ('user_message', 'assistant_message') "
            "ORDER BY created_at ASC",
            (session_id,)).fetchall()

        # Group by chain (each chain = one stop = user+assistant pair)
        chains = {}
        for r in rows:
            chain_id = r[0]
            if chain_id not in chains:
                chains[chain_id] = {}
            meta = {}
            try:
                meta = json.loads(r[4]) if r[4] else {}
            except (json.JSONDecodeError, TypeError):
                pass
            # Content lives in metadata (full), summary is truncated for display
            content = meta.get('content', '') or r[3] or ''
            if r[2] == 'user_message':
                chains[chain_id]['user'] = {
                    'content': content,
                    'timestamp': r[5],
                    'recall_chain': meta.get('recall_chain', ''),
                }
            elif r[2] == 'assistant_message':
                chains[chain_id]['assistant'] = {
                    'content': content,
                    'timestamp': r[5],
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
                try:
                    meta = json.loads(r[1]) if r[1] else {}
                    judge_outputs[r[0]] = meta.get('content', '')
                except (json.JSONDecodeError, TypeError):
                    judge_outputs[r[0]] = ''

        # Build result in encoding_agent._gather_messages() shape
        turns = []
        for chain_id in sorted(chains.keys(), key=lambda c: chains[c].get('user', {}).get('timestamp', '')):
            data = chains[chain_id]
            if 'user' in data:
                recall_chain = data['user'].get('recall_chain', '')
                turns.append({
                    'role': 'user',
                    'content': data['user']['content'],
                    'timestamp': data['user']['timestamp'],
                    'signal': None,
                    'judge_output': judge_outputs.get(recall_chain, ''),
                    'recalled_raw': None,  # Not stored in S0 traces (available in S1 O)
                })
            if 'assistant' in data:
                turns.append({
                    'role': 'assistant',
                    'content': data['assistant']['content'],
                    'timestamp': data['assistant']['timestamp'],
                    'signal': None,
                    'judge_output': None,
                    'recalled_raw': None,
                })

        # Apply limit (most recent turns)
        if len(turns) > limit:
            turns = turns[-limit:]

        return turns


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
        ts = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO session_state (session_id, key, node_id, value, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(session_id, key, node_id)
               DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
            (session_id, key, node_id, value, ts))
        self.conn.commit()

    def increment(self, session_id: str, key: str, node_id: str) -> int:
        """Increment a counter value. Returns new count."""
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO session_state (session_id, key, node_id, value, updated_at)
               VALUES (?, ?, ?, '1', ?)
               ON CONFLICT(session_id, key, node_id)
               DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT),
                            updated_at = excluded.updated_at""",
            (session_id, key, node_id, ts))
        self.conn.commit()
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
        """Load fatigue counts for a session. Returns {node_id: count}."""
        rows = self.conn.execute(
            "SELECT node_id, CAST(value AS INTEGER) FROM session_state "
            "WHERE session_id = ? AND key = 'fatigue'",
            (session_id,)).fetchall()
        return {r[0]: r[1] for r in rows}

    def cleanup_old_sessions(self, keep_last_n: int = 5):
        """Remove session_state for old sessions, keeping the N most recent."""
        self.conn.execute(
            """DELETE FROM session_state WHERE session_id NOT IN (
                SELECT DISTINCT session_id FROM session_state
                ORDER BY updated_at DESC LIMIT ?
            )""", (keep_last_n,))
        self.conn.commit()


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


class NodeDAL:
    """Access layer for brain.db nodes table.

    ALL node SQL lives here. When we move to in-memory graph,
    swap this implementation — nothing else changes.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # --- Reads ---

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node by ID. Returns all columns as a dict.

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

    def get_title(self, node_id: str) -> Optional[str]:
        """Get just the title of a node."""
        row = self.conn.execute(
            'SELECT title FROM nodes WHERE id = ?', (node_id,)
        ).fetchone()
        return row[0] if row else None

    def exists(self, node_id: str) -> bool:
        """Check if a node exists."""
        row = self.conn.execute(
            'SELECT id FROM nodes WHERE id = ?', (node_id,)
        ).fetchone()
        return row is not None

    def count(self, archived: bool = False) -> int:
        """Count nodes, optionally excluding archived."""
        if archived:
            row = self.conn.execute('SELECT COUNT(*) FROM nodes').fetchone()
        else:
            row = self.conn.execute(
                'SELECT COUNT(*) FROM nodes WHERE archived = 0'
            ).fetchone()
        return row[0] if row else 0

    def count_locked(self) -> int:
        """Count locked nodes."""
        row = self.conn.execute(
            'SELECT COUNT(*) FROM nodes WHERE locked = 1'
        ).fetchone()
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
        allowed_sort = {'created_at', 'confidence', 'access_count', 'title', 'type', 'updated_at'}
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

    def get_all_for_reindex(self) -> List[Dict[str, Any]]:
        """Get all non-archived nodes for TF-IDF reindex."""
        rows = self.conn.execute(
            'SELECT id, title, content, keywords FROM nodes WHERE archived = 0'
        ).fetchall()
        return [
            {'id': r[0], 'title': r[1], 'content': r[2], 'keywords': r[3]}
            for r in rows
        ]

    def get_all_with_titles(self) -> List[Dict[str, Any]]:
        """Get all nodes with titles (for absorb/dedup)."""
        rows = self.conn.execute(
            'SELECT id, title, type, locked FROM nodes WHERE title IS NOT NULL'
        ).fetchall()
        return [
            {'id': r[0], 'title': r[1], 'type': r[2], 'locked': r[3] == 1}
            for r in rows
        ]

    # --- Writes ---

    # Allowed columns for update_field — whitelist prevents SQL injection
    _UPDATABLE_FIELDS = frozenset({
        'content', 'content_summary', 'confidence', 'locked', 'archived',
        'critical', 'keywords', 'revised_at', 'updated_at', 'encoding_source',
        'encoding_version', 'evolution_status', 'resolved_at', 'resolved_by',
    })

    def update_field(self, node_id: str, field: str, value) -> None:
        """Update a single field on a node. Field must be in whitelist.
        Automatically sets updated_at."""
        if field not in self._UPDATABLE_FIELDS:
            raise ValueError("Cannot update field '%s' — not in whitelist" % field)
        self.conn.execute(
            'UPDATE nodes SET %s = ?, updated_at = ? WHERE id = ?' % field,
            (value, _now(), node_id))
        self.conn.commit()

    def update_confidence(self, node_id: str, confidence: float) -> None:
        """Update a node's confidence score."""
        self.conn.execute(
            'UPDATE nodes SET confidence = ?, updated_at = ? WHERE id = ?',
            (confidence, _now(), node_id)
        )
        self.conn.commit()

    def set_critical(self, node_id: str, critical: bool = True) -> None:
        """Mark a node as critical."""
        self.conn.execute(
            'UPDATE nodes SET critical = ?, updated_at = ? WHERE id = ?',
            (1 if critical else 0, _now(), node_id)
        )
        self.conn.commit()

    def archive(self, node_id: str) -> None:
        """Archive a node (soft delete)."""
        self.conn.execute(
            'UPDATE nodes SET archived = 1, updated_at = ? WHERE id = ?',
            (_now(), node_id)
        )
        self.conn.commit()

    def purge(self, node_id: str) -> None:
        """Hard delete a node and ALL associated data.
        Removes: node, embeddings, enrichments, edges (both directions), metadata KV.
        Use archive() for soft delete. This is irreversible."""
        self.conn.execute('DELETE FROM node_enrichments WHERE node_id = ?', (node_id,))
        self.conn.execute('DELETE FROM node_embeddings WHERE node_id = ?', (node_id,))
        self.conn.execute('DELETE FROM node_metadata_kv WHERE node_id = ?', (node_id,))
        self.conn.execute('DELETE FROM edges WHERE source_id = ? OR target_id = ?', (node_id, node_id))
        self.conn.execute('DELETE FROM nodes WHERE id = ?', (node_id,))
        self.conn.commit()

    def unlock(self, node_id: str) -> None:
        """Unlock a node."""
        self.conn.execute(
            'UPDATE nodes SET locked = 0, updated_at = ? WHERE id = ?',
            (_now(), node_id)
        )
        self.conn.commit()

    def update_type(self, node_id: str, new_type: str, title_prefix_old: str = '',
                    title_prefix_new: str = '') -> None:
        """Change a node's type, optionally updating title prefix."""
        if title_prefix_old and title_prefix_new:
            self.conn.execute(
                "UPDATE nodes SET type = ?, title = REPLACE(title, ?, ?), updated_at = ? WHERE id = ?",
                (new_type, title_prefix_old, title_prefix_new, _now(), node_id)
            )
        else:
            self.conn.execute(
                'UPDATE nodes SET type = ?, updated_at = ? WHERE id = ?',
                (new_type, _now(), node_id)
            )
        self.conn.commit()

    def append_content(self, node_id: str, text: str) -> None:
        """Append text to a node's content."""
        self.conn.execute(
            'UPDATE nodes SET content = content || ?, updated_at = ? WHERE id = ?',
            (text, _now(), node_id)
        )
        self.conn.commit()

    def set_evolution_status(self, node_id: str, status: str) -> None:
        """Set evolution_status on a node."""
        self.conn.execute(
            "UPDATE nodes SET evolution_status = ? WHERE id = ?",
            (status, node_id)
        )
        self.conn.commit()

    def delete(self, node_id: str) -> None:
        """Hard delete a node (use archive() for soft delete)."""
        self.conn.execute('DELETE FROM nodes WHERE id = ?', (node_id,))
        self.conn.commit()

    def mark_accessed(self, node_id: str, activation_boost: float = 0.1) -> None:
        """Update access tracking fields on a node."""
        ts = _now()
        self.conn.execute(
            'UPDATE nodes SET access_count = access_count + 1, '
            'activation = MIN(1.0, activation + ?), '
            'recency_score = 1.0, last_accessed = ?, updated_at = ? WHERE id = ?',
            (activation_boost, ts, ts, node_id)
        )

    def get_metadata(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node_metadata for a node. Returns None if no metadata exists."""
        row = self.conn.execute(
            'SELECT reasoning, user_raw_quote, correction_of, last_validated '
            'FROM node_metadata WHERE node_id = ?',
            (node_id,)
        ).fetchone()
        if not row or not any(row):
            return None
        return {
            'reasoning': row[0], 'user_raw_quote': row[1],
            'correction_of': row[2], 'last_validated': row[3],
        }


class EmbeddingDAL:
    """Access layer for node_embeddings and node_enrichments tables."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # --- node_embeddings ---

    def get_embedding(self, node_id: str) -> Optional[bytes]:
        """Get embedding blob for a node."""
        row = self.conn.execute(
            'SELECT embedding FROM node_embeddings WHERE node_id = ?',
            (node_id,)
        ).fetchone()
        return row[0] if row else None

    def get_all_embeddings(self, exclude_archived: bool = True) -> List[Dict[str, Any]]:
        """Get all embeddings for cosine scan. Returns node_id + embedding blob."""
        sql = ('SELECT ne.node_id, ne.embedding FROM node_embeddings ne '
               'JOIN nodes n ON n.id = ne.node_id')
        if exclude_archived:
            sql += ' WHERE n.archived = 0'
        rows = self.conn.execute(sql).fetchall()
        return [{'node_id': r[0], 'embedding': r[1]} for r in rows]

    def get_all_with_context(self, exclude_archived: bool = True,
                             types: List[str] = None,
                             project: str = None) -> List[Dict[str, Any]]:
        """Get all embeddings with node context for recall STEP 3 scan.

        Returns: [{node_id, embedding, personal, personal_context,
                   confidence, critical, title, type,
                   created_at, emotion, access_count}]

        The last 3 fields feed unified_score() in recall_scoring.py:
          created_at → freshness_from_created (recency from birth, not access)
          emotion → emotion amplification (GANE model)
          access_count → frequency penalty (hub dampening)

        Filters: archived, type, project — matching recall pipeline needs.
        """
        where = []
        params = []
        if exclude_archived:
            where.append('n.archived = 0')
        if types:
            where.append('n.type IN (%s)' % ','.join('?' * len(types)))
            params.extend(types)
        if project:
            where.append('(n.project = ? OR n.project IS NULL)')
            params.append(project)
        where_sql = (' WHERE ' + ' AND '.join(where)) if where else ''
        rows = self.conn.execute(
            'SELECT ne.node_id, ne.embedding, n.personal, n.personal_context, '
            'n.confidence, n.critical, n.title, n.type, '
            'n.created_at, n.emotion, n.access_count '
            'FROM node_embeddings ne '
            'JOIN nodes n ON n.id = ne.node_id' + where_sql,
            params
        ).fetchall()
        return [{'node_id': r[0], 'embedding': r[1], 'personal': r[2],
                 'personal_context': r[3], 'confidence': r[4],
                 'critical': r[5] or 0, 'title': r[6] or '', 'type': r[7] or '',
                 'created_at': r[8], 'emotion': r[9] or 0,
                 'access_count': r[10] or 0}
                for r in rows]

    def store_embedding(self, node_id: str, embedding: bytes, model: str) -> None:
        """Store or replace an embedding for a node."""
        self.conn.execute(
            'INSERT OR REPLACE INTO node_embeddings '
            '(node_id, embedding, model, created_at) VALUES (?, ?, ?, ?)',
            (node_id, embedding, model, _now())
        )
        self.conn.commit()

    def count(self) -> int:
        """Count total embeddings."""
        row = self.conn.execute('SELECT COUNT(*) FROM node_embeddings').fetchone()
        return row[0] if row else 0

    # --- situation embeddings ---

    def store_situation(self, node_id: str, situation_text: str, situation_blob: bytes) -> None:
        """Store situation embedding + text for a node. Node must already have a content embedding."""
        self.conn.execute(
            'UPDATE node_embeddings SET situation_embedding=?, situation_text=? WHERE node_id=?',
            (situation_blob, situation_text, node_id))
        self.conn.commit()

    def get_all_situations(self) -> List[Dict[str, Any]]:
        """Get all situation embeddings for cosine scan. Skips nodes without situation."""
        rows = self.conn.execute(
            'SELECT ne.node_id, ne.situation_embedding '
            'FROM node_embeddings ne JOIN nodes n ON n.id = ne.node_id '
            'WHERE n.archived = 0 AND ne.situation_embedding IS NOT NULL'
        ).fetchall()
        return [{'node_id': r[0], 'situation_embedding': r[1]} for r in rows]

    def get_situation_text(self, node_id: str) -> Optional[str]:
        """Get the raw situation text for a node."""
        row = self.conn.execute(
            'SELECT situation_text FROM node_embeddings WHERE node_id = ?', (node_id,)
        ).fetchone()
        return row[0] if row else None

    # --- node_enrichments ---

    def get_all_enrichments(self) -> List[Dict[str, Any]]:
        """Get all enrichment vectors for cosine scan."""
        rows = self.conn.execute(
            'SELECT node_id, vector_type, embedding FROM node_enrichments '
            'WHERE embedding IS NOT NULL'
        ).fetchall()
        return [
            {'node_id': r[0], 'vector_type': r[1], 'embedding': r[2]}
            for r in rows
        ]

    def store_enrichment(self, node_id: str, vector_type: str, text: str,
                         embedding: Optional[bytes], model: str) -> None:
        """Store an enrichment vector."""
        import uuid
        self.conn.execute(
            'INSERT INTO node_enrichments '
            '(id, node_id, vector_type, text, embedding, model, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (uuid.uuid4().hex, node_id, vector_type, text, embedding, model, _now())
        )
        self.conn.commit()

    def delete_for_node(self, node_id: str) -> int:
        """Delete all enrichments for a node. Returns count deleted."""
        cur = self.conn.execute(
            'DELETE FROM node_enrichments WHERE node_id = ?', (node_id,)
        )
        self.conn.commit()
        return cur.rowcount


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
        self.conn.commit()

    def delete_for_node(self, node_id: str) -> None:
        """Delete TF-IDF data for a node."""
        self.conn.execute(
            'DELETE FROM node_vectors WHERE node_id = ?', (node_id,)
        )
        self.conn.commit()

    def clear_all(self) -> None:
        """Clear entire TF-IDF index (for reindex)."""
        self.conn.execute('DELETE FROM node_vectors')
        self.conn.execute('DELETE FROM doc_freq')
        self.conn.commit()

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
    Both feed into the judge which decides relevance.
    """

    def __init__(self, conn):
        self.conn = conn

    def search(self, query: str, limit: int = 30) -> List[str]:
        """Full-text search. Returns node_ids ranked by BM25 relevance.

        Title matches weighted 10x over content, keywords 2x.
        bm25() column weights: (node_id=0, title=10, content=1, keywords=2)
        """
        safe_query = self._sanitize_query(query)
        if not safe_query:
            return []
        try:
            rows = self.conn.execute(
                """SELECT node_id FROM nodes_fts
                   WHERE nodes_fts MATCH ?
                   ORDER BY bm25(nodes_fts, 0, 10.0, 1.0, 2.0)
                   LIMIT ?""",
                (safe_query, limit)
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    def upsert(self, node_id: str, title: str, content: str, keywords: str):
        """Insert or update a node in the FTS5 index."""
        self.delete(node_id)
        self.conn.execute(
            "INSERT INTO nodes_fts (node_id, title, content, keywords) VALUES (?, ?, ?, ?)",
            (node_id, title, content or '', keywords or ''))

    def delete(self, node_id: str):
        """Remove a node from FTS5 index."""
        try:
            self.conn.execute(
                "DELETE FROM nodes_fts WHERE node_id = ?", (node_id,))
        except Exception:
            pass

    def rebuild(self):
        """Full rebuild of FTS5 index from nodes table."""
        self.conn.execute("DELETE FROM nodes_fts")
        self.conn.execute("""
            INSERT INTO nodes_fts (node_id, title, content, keywords)
            SELECT id, title, COALESCE(content, ''), COALESCE(keywords, '')
            FROM nodes WHERE archived = 0
        """)
        self.conn.commit()

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


class GraphDAL:
    """Access layer for brain.db graph tables: edges.

    ALL edge SQL lives here. When we move to in-memory graph,
    swap this implementation — nothing else changes.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # --- Reads ---

    def count_total(self) -> int:
        """Count total edges."""
        row = self.conn.execute('SELECT COUNT(*) FROM edges').fetchone()
        return row[0] if row else 0

    def get_edge(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get a single edge between two nodes (directional)."""
        row = self.conn.execute(
            'SELECT weight, co_access_count, stability, relation, last_strengthened '
            'FROM edges WHERE source_id = ? AND target_id = ?',
            (source_id, target_id)
        ).fetchone()
        if not row:
            return None
        return {
            'weight': row[0], 'co_access_count': row[1], 'stability': row[2],
            'relation': row[3], 'last_strengthened': row[4],
        }

    def edge_exists(self, source_id: str, target_id: str) -> bool:
        """Check if edge exists in either direction."""
        row = self.conn.execute(
            'SELECT 1 FROM edges WHERE (source_id = ? AND target_id = ?) '
            'OR (source_id = ? AND target_id = ?)',
            (source_id, target_id, target_id, source_id)
        ).fetchone()
        return row is not None

    def get_neighbors(self, node_id: str, min_weight: float = 0.05,
                      limit: int = 50) -> List[Dict[str, Any]]:
        """Get outgoing neighbors for spreading activation.

        Returns list of dicts with keys: target_id, weight.
        """
        rows = self.conn.execute(
            'SELECT target_id, weight FROM edges '
            'WHERE source_id = ? AND weight > ? '
            'ORDER BY weight DESC LIMIT ?',
            (node_id, min_weight, limit)
        ).fetchall()
        return [{'target_id': r[0], 'weight': r[1]} for r in rows]

    def get_typed_neighbors(self, node_id: str, edge_types: set,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Get 1-hop neighbors via intentional (typed) edges only.

        Searches both directions (source->target and target->source).
        Returns list of dicts with keys: neighbor_id, relation, weight.
        """
        placeholders = ','.join('?' * len(edge_types))
        params = [node_id] + list(edge_types) + [node_id] + list(edge_types) + [limit]
        rows = self.conn.execute(f"""
            SELECT neighbor_id, relation, weight FROM (
                SELECT target_id AS neighbor_id, relation, weight
                FROM edges
                WHERE source_id = ? AND relation IN ({placeholders})
                UNION
                SELECT source_id AS neighbor_id, relation, weight
                FROM edges
                WHERE target_id = ? AND relation IN ({placeholders})
            )
            ORDER BY weight DESC
            LIMIT ?
        """, params).fetchall()
        return [
            {'neighbor_id': r[0], 'relation': r[1], 'weight': r[2]}
            for r in rows
        ]

    def get_neighbors_with_context(self, node_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get neighbors with their full context (title, type, keywords, confidence).

        Used by V5 enrichment to build the structured prompt with neighbor info.
        Returns neighbors sorted by edge weight, enriched with node data.
        """
        rows = self.conn.execute("""
            SELECT n.id, n.type, n.title, n.keywords, n.confidence, e.relation, e.weight
            FROM (
                SELECT target_id AS nid, relation, weight FROM edges WHERE source_id = ?
                UNION
                SELECT source_id AS nid, relation, weight FROM edges WHERE target_id = ?
            ) e
            JOIN nodes n ON n.id = e.nid
            WHERE n.archived = 0
            ORDER BY e.weight DESC
            LIMIT ?
        """, (node_id, node_id, limit)).fetchall()
        return [
            {'id': r[0], 'type': r[1], 'title': r[2], 'keywords': r[3],
             'confidence': r[4], 'relation': r[5], 'weight': r[6]}
            for r in rows
        ]

    def get_neighbors_rich(self, node_id: str, limit: int = 8,
                           exclude_relations: set = None,
                           exclude_node_ids: set = None) -> List[Dict[str, Any]]:
        """Get neighbors with full node + edge + metadata in one query.

        Filters happen in SQL — no wasted rows from back-edges or visited nodes.

        Args:
            node_id: Source node to find neighbors of
            limit: Max neighbors to return
            exclude_relations: Edge types to skip (e.g. {'co_accessed'})
            exclude_node_ids: Node IDs to skip (already visited in traversal)
        """
        # Build dynamic WHERE clauses
        where_parts = ["n.archived = 0"]
        params = [node_id, node_id]

        if exclude_node_ids:
            placeholders = ",".join("?" * len(exclude_node_ids))
            where_parts.append("n.id NOT IN (%s)" % placeholders)
            params.extend(exclude_node_ids)

        if exclude_relations:
            placeholders = ",".join("?" * len(exclude_relations))
            where_parts.append("e.relation NOT IN (%s)" % placeholders)
            params.extend(exclude_relations)

        params.append(limit)
        where_clause = " AND ".join(where_parts)

        rows = self.conn.execute("""
            SELECT
                n.id, n.type, n.title, n.content_summary, n.confidence,
                n.revised_at, n.created_at, n.last_accessed, n.access_count,
                n.locked, n.emotion, n.emotion_label,
                e.relation, e.weight, e.description,
                e.last_strengthened, e.co_access_count,
                m.reasoning, m.user_raw_quote, m.correction_of, m.correction_pattern,
                m.source_context, m.validation_count
            FROM (
                SELECT target_id AS nid, relation, weight, description,
                       last_strengthened, co_access_count FROM edges WHERE source_id = ?
                UNION
                SELECT source_id AS nid, relation, weight, description,
                       last_strengthened, co_access_count FROM edges WHERE target_id = ?
            ) e
            JOIN nodes n ON n.id = e.nid
            LEFT JOIN node_metadata m ON m.node_id = n.id
            WHERE %s
            ORDER BY e.weight DESC
            LIMIT ?
        """ % where_clause, params).fetchall()

        return [{
            'id': r[0], 'type': r[1], 'title': r[2], 'content_summary': r[3],
            'confidence': r[4], 'revised_at': r[5], 'created_at': r[6],
            'last_accessed': r[7], 'access_count': r[8], 'locked': r[9],
            'emotion': r[10], 'emotion_label': r[11],
            'relation': r[12] or '', 'weight': r[13],
            'edge_description': r[14], 'last_strengthened': r[15],
            'co_access_count': r[16],
            'reasoning': r[17], 'user_raw_quote': r[18],
            'correction_of': r[19], 'correction_pattern': r[20],
            'source_context': r[21], 'validation_count': r[22],
        } for r in rows]

    def count_node_edges(self, node_id: str, min_weight: float = 0.1) -> int:
        """Count edges from a node (used by dreams, surface)."""
        row = self.conn.execute(
            'SELECT COUNT(*) FROM edges WHERE source_id = ? AND weight >= ?',
            (node_id, min_weight)
        ).fetchone()
        return row[0] if row else 0

    def get_edge_count(self) -> int:
        """Total edge count in the graph."""
        row = self.conn.execute('SELECT COUNT(*) FROM edges').fetchone()
        return row[0] if row else 0

    def get_well_connected(self, min_weight: float = 0.3,
                           min_edges: int = 5) -> List[Dict[str, Any]]:
        """Find well-connected nodes for consolidation/promotion."""
        rows = self.conn.execute(
            'SELECT source_id, SUM(weight) as total_weight, COUNT(*) as edge_count '
            'FROM edges WHERE weight > ? '
            'GROUP BY source_id HAVING edge_count >= ?',
            (min_weight, min_edges)
        ).fetchall()
        return [
            {'node_id': r[0], 'total_weight': r[1], 'edge_count': r[2]}
            for r in rows
        ]

    def get_random_walk_neighbors(self, node_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get neighbors for random walk (dreams), ordered randomly."""
        rows = self.conn.execute(
            'SELECT target_id, weight FROM edges '
            'WHERE source_id = ? ORDER BY RANDOM() LIMIT ?',
            (node_id, limit)
        ).fetchall()
        return [{'target_id': r[0], 'weight': r[1]} for r in rows]

    # --- Writes ---

    def create_edge(self, source_id: str, target_id: str, relation: str = 'related',
                    weight: float = 0.5, description: str = '') -> bool:
        """Create a bidirectional edge. Returns False if already exists."""
        ts = _now()
        # Check if already exists
        if self.get_edge(source_id, target_id) is not None:
            return False
        # Forward
        self.conn.execute(
            'INSERT OR IGNORE INTO edges '
            '(source_id, target_id, weight, relation, edge_type, description, '
            'co_access_count, stability, last_strengthened, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, 1, 1.0, ?, ?)',
            (source_id, target_id, weight, relation, relation, description, ts, ts)
        )
        # Reverse
        self.conn.execute(
            'INSERT OR IGNORE INTO edges '
            '(source_id, target_id, weight, relation, edge_type, description, '
            'co_access_count, stability, last_strengthened, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, 1, 1.0, ?, ?)',
            (target_id, source_id, weight, relation, relation, description, ts, ts)
        )
        self.conn.commit()
        return True

    def strengthen_edge(self, source_id: str, target_id: str,
                        amount: float = 0.1, relation: Optional[str] = None) -> bool:
        """Strengthen an existing edge (Hebbian learning).

        Returns True if edge was found and strengthened, False if not found.
        """
        from .brain_constants import MAX_WEIGHT, STABILITY_BOOST
        ts = _now()
        row = self.get_edge(source_id, target_id)
        if not row:
            return False
        new_weight = min(MAX_WEIGHT, row['weight'] + amount)
        new_stability = min(row['stability'] * STABILITY_BOOST, 10.0)
        params = [new_weight, row['co_access_count'] + 1, new_stability, ts,
                  source_id, target_id]
        sql = ('UPDATE edges SET weight = ?, co_access_count = ?, '
               'stability = ?, last_strengthened = ?')
        if relation:
            sql += ', relation = ?, edge_type = ?'
            params = [new_weight, row['co_access_count'] + 1, new_stability, ts,
                      relation, relation, source_id, target_id]
        sql += ' WHERE source_id = ? AND target_id = ?'
        self.conn.execute(sql, params)
        self.conn.commit()
        return True

    def create_or_strengthen(self, source_id: str, target_id: str,
                             relation: str = 'related', weight: float = 0.5,
                             strengthen_amount: float = 0.1,
                             description: str = '') -> str:
        """Create edge if new, strengthen if exists. Returns 'created' or 'strengthened'."""
        from .brain_constants import LEARNING_RATE
        existing = self.get_edge(source_id, target_id)
        if existing:
            self.strengthen_edge(source_id, target_id, LEARNING_RATE * 0.5, relation)
            return 'strengthened'
        else:
            self.create_edge(source_id, target_id, relation, weight, description)
            return 'created'

    def delete_node_edges(self, node_id: str) -> int:
        """Delete all edges touching a node. Returns count deleted."""
        cur = self.conn.execute(
            'DELETE FROM edges WHERE source_id = ? OR target_id = ?',
            (node_id, node_id)
        )
        self.conn.commit()
        return cur.rowcount

    def decay_edges(self) -> Dict[str, Any]:
        """Apply exponential decay to auto-generated edges based on EDGE_TYPES half-lives.

        Formula: new_weight = weight * 0.5^(hours_since_reinforced / half_life)
        Edges below EDGE_PRUNE_THRESHOLD after decay are deleted.

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

            # Apply decay: weight * 0.5^(hours_since / half_life)
            # hours_since = (julianday('now') - julianday(last_strengthened)) * 24
            self.conn.execute("""
                UPDATE edges
                SET weight = weight * power(0.5,
                    (julianday('now') - julianday(last_strengthened)) * 24.0 / ?)
                WHERE relation = ?
                  AND last_strengthened IS NOT NULL
                  AND (julianday('now') - julianday(last_strengthened)) * 24.0 > 0
            """, (half_life, relation))
            decayed = self.conn.execute('SELECT changes()').fetchone()[0]

            # Prune edges that decayed below threshold
            self.conn.execute("""
                DELETE FROM edges WHERE relation = ? AND weight < ?
            """, (relation, EDGE_PRUNE_THRESHOLD))
            pruned = self.conn.execute('SELECT changes()').fetchone()[0]

            if decayed or pruned:
                by_type[relation] = {'decayed': decayed, 'pruned': pruned}
                total_decayed += decayed
                total_pruned += pruned

        self.conn.commit()
        return {'decayed': total_decayed, 'pruned': total_pruned, 'by_type': by_type}


def _now() -> str:
    """UTC ISO timestamp for edge operations."""
    return datetime.now(timezone.utc).isoformat()


class EnrichmentDAL:
    """Access layer for node_enrichments table — multi-vector encoding.

    Each node can have up to 4 enrichment vectors:
    - question: natural-language question the node answers
    - anchor: short phrase using neighbor vocabulary
    - bridge: sentence connecting to most important neighbor
    - keywords: shared keywords from neighbors
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def store(self, node_id: str, vector_type: str, text: str,
              embedding: Optional[bytes] = None, model: str = 'snowflake-arctic-embed-m') -> str:
        """Store an enrichment vector for a node. Returns enrichment ID."""
        import uuid
        eid = str(uuid.uuid4().hex[:16])
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            '''INSERT OR REPLACE INTO node_enrichments
               (id, node_id, vector_type, text, embedding, model, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (eid, node_id, vector_type, text, embedding, model, now)
        )
        self.conn.commit()
        return eid

    def get_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all enrichments for a node."""
        rows = self.conn.execute(
            'SELECT id, vector_type, text, embedding FROM node_enrichments WHERE node_id = ?',
            (node_id,)
        ).fetchall()
        return [
            {'id': r[0], 'vector_type': r[1], 'text': r[2], 'embedding': r[3]}
            for r in rows
        ]

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all enrichment embeddings for recall scanning.

        Returns list of dicts with node_id, vector_type, embedding.
        Only returns enrichments that have embeddings (not NULL).
        """
        rows = self.conn.execute(
            '''SELECT node_id, vector_type, embedding
               FROM node_enrichments
               WHERE embedding IS NOT NULL'''
        ).fetchall()
        return [
            {'node_id': r[0], 'vector_type': r[1], 'embedding': r[2]}
            for r in rows
        ]

    def count_for_node(self, node_id: str) -> int:
        """Count enrichments for a node."""
        row = self.conn.execute(
            'SELECT COUNT(*) FROM node_enrichments WHERE node_id = ?', (node_id,)
        ).fetchone()
        return row[0] if row else 0

    def delete_for_node(self, node_id: str) -> int:
        """Delete all enrichments for a node. Returns count deleted."""
        self.conn.execute('DELETE FROM node_enrichments WHERE node_id = ?', (node_id,))
        self.conn.commit()
        return self.conn.execute('SELECT changes()').fetchone()[0]

    def get_coverage_stats(self) -> Dict[str, Any]:
        """Get enrichment coverage statistics."""
        total_nodes = self.conn.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]
        enriched_nodes = self.conn.execute(
            'SELECT COUNT(DISTINCT node_id) FROM node_enrichments WHERE embedding IS NOT NULL'
        ).fetchone()[0]
        by_type = self.conn.execute(
            'SELECT vector_type, COUNT(*) FROM node_enrichments GROUP BY vector_type'
        ).fetchall()
        return {
            'total_nodes': total_nodes,
            'enriched_nodes': enriched_nodes,
            'coverage_pct': round(enriched_nodes / total_nodes * 100, 1) if total_nodes else 0,
            'by_type': {r[0]: r[1] for r in by_type},
        }


# TelemetryDAL — REMOVED 2026-04-05 (brain_telemetry table dropped, never used)
