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
        """Record a recall event. Returns the recall_log id.

        DEPRECATED: Use RecallPrecision.log_recall() from servers/brain_precision.py instead.
        This method uses wrong column names (result_ids/intent vs the actual schema's
        returned_ids/returned_count) and lacks the precision tracking columns.
        Kept for backward compatibility only — do not add new callers.
        """
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
        # access_log: 30 days (uses 'timestamp' column, not 'created_at')
        cur = self.conn.execute(
            "DELETE FROM access_log WHERE timestamp < datetime('now', '-30 days')")
        stats['access_log_pruned'] = cur.rowcount

        # debug_log: keep errors forever, prune telemetry/other after 30 days
        cur = self.conn.execute(
            "DELETE FROM debug_log WHERE event_type != 'error' "
            "AND created_at < datetime('now', '-30 days')")
        stats['debug_log_pruned'] = cur.rowcount

        # recall_log: keep evaluated forever, prune unevaluated after 30 days
        cur = self.conn.execute(
            "DELETE FROM recall_log WHERE evaluated_at IS NULL "
            "AND created_at < datetime('now', '-30 days')")
        stats['recall_log_pruned'] = cur.rowcount

        # suggest_log: 30 days
        cur = self.conn.execute(
            "DELETE FROM suggest_log WHERE created_at < datetime('now', '-30 days')")
        stats['suggest_log_pruned'] = cur.rowcount

        # health_log: 90 days
        cur = self.conn.execute(
            "DELETE FROM health_log WHERE created_at < datetime('now', '-90 days')")
        stats['health_log_pruned'] = cur.rowcount

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

    # ── recall_log (precision lifecycle) ──
    # Schema: id, session_id, query, returned_ids, returned_count, used_ids, used_count,
    #         precision_score, embeddings_used, recalled_titles, recalled_snippets,
    #         assistant_response_snippet, match_method, evaluation_metadata,
    #         followup_signal, explicit_feedback, evaluated_at, created_at
    #
    # Row lifecycle: LOGGED → RESPONSE_STORED → EVALUATED → FEEDBACK_RECEIVED
    # Hooks query the table for pending work — no config keys for handoff.

    def insert_recall_log(self, session_id: str, query: str, returned_ids: str,
                          returned_count: int, embeddings_used: int,
                          recalled_titles: str, recalled_snippets: str,
                          created_at: str) -> int:
        """Insert a new recall_log row (Stage 1: LOGGED). Returns row ID."""
        cursor = self.conn.execute(
            """INSERT INTO recall_log
               (session_id, query, returned_ids, returned_count,
                embeddings_used, recalled_titles, recalled_snippets, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, query, returned_ids, returned_count,
             embeddings_used, recalled_titles, recalled_snippets, created_at))
        self.conn.commit()
        return cursor.lastrowid

    def update_recall_response(self, recall_log_id: int, response_snippet: str,
                                match_method: str, evaluation_metadata: Optional[str],
                                evaluated_at: str) -> None:
        """Store Claude's response on a recall row (Stage 2: RESPONSE_STORED)."""
        self.conn.execute(
            """UPDATE recall_log
               SET assistant_response_snippet = ?,
                   match_method = ?,
                   evaluation_metadata = COALESCE(?, evaluation_metadata),
                   evaluated_at = ?
               WHERE id = ?""",
            (response_snippet, match_method, evaluation_metadata, evaluated_at, recall_log_id))
        self.conn.commit()

    def update_recall_evaluation(self, recall_log_id: int, followup_signal: str,
                                  match_method: str, precision_score: Optional[float],
                                  evaluation_metadata: str, evaluated_at: str) -> None:
        """Store followup evaluation (Stage 3: EVALUATED). Won't override explicit feedback."""
        self.conn.execute(
            """UPDATE recall_log
               SET followup_signal = ?,
                   match_method = ?,
                   precision_score = ?,
                   evaluation_metadata = ?,
                   evaluated_at = ?
               WHERE id = ? AND explicit_feedback IS NULL""",
            (followup_signal, match_method, precision_score,
             evaluation_metadata, evaluated_at, recall_log_id))
        self.conn.commit()

    def update_recall_feedback(self, recall_log_id: int, explicit_feedback: str,
                                precision_score: float, evaluated_at: str) -> None:
        """Store explicit operator feedback (Stage 4: FEEDBACK_RECEIVED). Overrides auto-score."""
        self.conn.execute(
            """UPDATE recall_log
               SET explicit_feedback = ?,
                   precision_score = ?,
                   evaluated_at = ?
               WHERE id = ?""",
            (explicit_feedback, precision_score, evaluated_at, recall_log_id))
        self.conn.commit()

    def get_recall_row(self, recall_log_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single recall_log row by ID."""
        row = self.conn.execute(
            """SELECT id, session_id, query, returned_ids, returned_count,
                      recalled_titles, recalled_snippets, assistant_response_snippet,
                      match_method, evaluation_metadata, followup_signal,
                      explicit_feedback, precision_score, evaluated_at, created_at
               FROM recall_log WHERE id = ?""",
            (recall_log_id,)).fetchone()
        if not row:
            return None
        return {
            'id': row[0], 'session_id': row[1], 'query': row[2],
            'returned_ids': row[3], 'returned_count': row[4],
            'recalled_titles': row[5], 'recalled_snippets': row[6],
            'assistant_response_snippet': row[7], 'match_method': row[8],
            'evaluation_metadata': row[9], 'followup_signal': row[10],
            'explicit_feedback': row[11], 'precision_score': row[12],
            'evaluated_at': row[13], 'created_at': row[14],
        }

    def get_pending_response(self, session_id: str) -> Optional[int]:
        """Find the most recent recall awaiting response storage (Stage 1 → 2).

        Returns recall_log ID or None.
        """
        row = self.conn.execute(
            """SELECT id FROM recall_log
               WHERE session_id = ? AND assistant_response_snippet IS NULL
                 AND returned_count > 0
               ORDER BY created_at DESC LIMIT 1""",
            (session_id,)).fetchone()
        return row[0] if row else None

    def get_pending_followups(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find recalls awaiting followup evaluation (any stage before EVALUATED).

        Returns recalls that have no followup signal and no explicit feedback,
        regardless of whether a response was stored. This catches:
        - Stage 2 rows (response stored, awaiting followup) — normal flow
        - Stage 1 rows (no response — hook timeout or short response) — recovery

        The followup evaluation can run without the response — it just won't
        have embedding signals from evaluate_response. Better to evaluate with
        partial data than not evaluate at all.
        """
        rows = self.conn.execute(
            """SELECT id, created_at FROM recall_log
               WHERE session_id = ? AND followup_signal IS NULL
                 AND explicit_feedback IS NULL AND returned_count > 0
               ORDER BY created_at DESC LIMIT ?""",
            (session_id, limit)).fetchall()
        return [{'id': r[0], 'created_at': r[1]} for r in rows]

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


class GraphDAL:
    """Access layer for brain.db graph tables: edges."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_typed_neighbors(self, node_id: str, edge_types: set,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Get 1-hop neighbors via intentional (typed) edges only.

        Searches both directions (source→target and target→source).
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


class TelemetryDAL:
    """Access layer for brain_telemetry table — operation logging.

    Every critical operation logs timing, success/failure, and metadata.
    No silent failures — this is the audit trail.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def log(self, operation: str, success: bool, duration_ms: float = None,
            error_message: str = None, **metadata) -> None:
        """Log a telemetry event.

        NOTE: This uses brain_logs.db (not brain.db). Caller must pass logs_conn.
        Errors are printed to stderr but NOT silenced — they propagate so callers
        know telemetry is broken and can fix it.
        """
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata) if metadata else None
        self.conn.execute(
            '''INSERT INTO brain_telemetry
               (timestamp, operation, duration_ms, success, error_message, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (now, operation, duration_ms, 1 if success else 0,
             error_message, meta_json, now)
        )
        self.conn.commit()

    def get_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get telemetry stats for the last N hours."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self.conn.execute('''
            SELECT operation,
                   COUNT(*) as total,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures,
                   AVG(duration_ms) as avg_ms,
                   MAX(duration_ms) as max_ms
            FROM brain_telemetry
            WHERE timestamp > ?
            GROUP BY operation
        ''', (cutoff,)).fetchall()
        return {
            r[0]: {'total': r[1], 'failures': r[2], 'avg_ms': round(r[3], 1) if r[3] else None,
                   'max_ms': round(r[4], 1) if r[4] else None}
            for r in rows
        }

    def get_recent_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent failures across all operations."""
        rows = self.conn.execute('''
            SELECT timestamp, operation, duration_ms, error_message, metadata
            FROM brain_telemetry
            WHERE success = 0
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,)).fetchall()
        return [
            {'timestamp': r[0], 'operation': r[1], 'duration_ms': r[2],
             'error': r[3], 'metadata': json.loads(r[4]) if r[4] else None}
            for r in rows
        ]

    def get_enrichment_hit_rate(self, hours: int = 24) -> Dict[str, Any]:
        """Calculate how often enrichment vectors are used in recall."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self.conn.execute('''
            SELECT metadata FROM brain_telemetry
            WHERE operation = 'recall' AND success = 1 AND timestamp > ?
        ''', (cutoff,)).fetchall()

        total_recalls = len(rows)
        enrichment_hits = 0
        by_type = {'question': 0, 'anchor': 0, 'bridge': 0, 'keywords': 0}

        for (meta_json,) in rows:
            if meta_json:
                try:
                    meta = json.loads(meta_json)
                    if meta.get('enrichment_hits', 0) > 0:
                        enrichment_hits += 1
                    for vtype in by_type:
                        by_type[vtype] += meta.get(f'enrichment_hit_{vtype}', 0)
                except (json.JSONDecodeError, TypeError):
                    pass

        return {
            'total_recalls': total_recalls,
            'enrichment_hits': enrichment_hits,
            'hit_rate_pct': round(enrichment_hits / total_recalls * 100, 1) if total_recalls else 0,
            'by_type': by_type,
        }
