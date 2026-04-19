"""CachedVectorDAL — drop-in decorator over VectorDAL.

Same public surface as VectorDAL (see servers/dal.py). Brain operations
don't know this exists — brain.py wires self._vec_dal to this class
instead of VectorDAL, nothing else changes.

Architecture:
  - Writes pass through to the inner VectorDAL first. On SQL success,
    the cache is updated. On SQL failure, cache is not touched — SQLite
    is always the truth, cache can never hold data the DB doesn't.
  - Reads serve from VectorCache. For get_all_with_context, the cache
    supplies the vectors and a small SQL join to the `nodes` table
    supplies the per-node context (personal, confidence, title, etc).
    That join is cheap — it operates on ~2500 small rows with indexes,
    not the 25 MB vector scan that thrashed the page cache.
  - find_missing delegates to inner VectorDAL — it's the backfill path,
    not the hot recall path, and it needs node titles/content the cache
    doesn't track.
  - Archive: drop_node() removes all vectors for a node from the cache.
    Called by brain.archive_node(). The underlying SQL row stays
    (archived flag on the nodes table), the cache just masks it.

Thread safety: a single RLock guards cache mutations. Reads snapshot
dicts under the lock then release — negligible contention.
"""
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Set

from .dal import VectorDAL
from .vector_cache import VectorCache


class CachedVectorDAL:
    """VectorDAL decorator with an in-memory vector cache.

    Thread safety model:
      - VectorCache has its own RLock — queries against the in-memory store
        are safe concurrently.
      - This class also guards the shared SQLite connection with a lock.
        Python's sqlite3 is built in SERIALIZED mode (with per-process
        mutex) but macOS ARM builds have historically crashed under
        concurrent execute() on one connection. The daemon's dispatch
        serializes commands, so production never hits concurrency here,
        but this lock makes the class standalone-safe (tests, benchmarks,
        future callers).
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._sql_lock = threading.Lock()
        self._inner = VectorDAL(conn)
        self._cache = VectorCache()
        self._load_cache_from_db()

    # ── Boot ────────────────────────────────────────────────────────

    def _load_cache_from_db(self) -> int:
        """One-shot SELECT at construction time. ~200 ms for 8k rows."""
        with self._sql_lock:
            rows = self.conn.execute(
                'SELECT node_id, vector_type, embedding, text, model '
                'FROM node_enrichments WHERE embedding IS NOT NULL'
            ).fetchall()
        return self._cache.load(rows)

    def reload(self) -> int:
        """Public: drop cache and rebuild from SQL. Callers (migration
        scripts, manual SQL writes) should invoke this after bypassing
        the write path."""
        return self._load_cache_from_db()

    # ── Writes (pass through + update cache) ────────────────────────

    def store(self, node_id: str, vector_type: str, text: str,
              embedding: Optional[bytes],
              model: str = 'nomic-ai/nomic-embed-text-v1.5-Q') -> None:
        """Upsert one vector. DB first, cache after."""
        with self._sql_lock:
            self._inner.store(node_id, vector_type, text, embedding, model)
        if embedding is not None:
            self._cache.add(node_id, vector_type, embedding,
                            (text or '')[:500], model)

    def store_batch(self, rows,
                    model: str = 'nomic-ai/nomic-embed-text-v1.5-Q') -> int:
        """Upsert many. DB first, cache after. Returns count written to DB."""
        rows_list = list(rows)
        with self._sql_lock:
            written = self._inner.store_batch(rows_list, model=model)
        if written:
            self._cache.add_batch(
                (nid, vt, blob, (text or '')[:500], model)
                for (nid, vt, text, blob) in rows_list
                if blob is not None)
        return written

    def delete_for_node(self, node_id: str) -> int:
        """Full delete (rare — tests, migrations). Cache drops too."""
        with self._sql_lock:
            n = self._inner.delete_for_node(node_id)
        self._cache.drop_node(node_id)
        return n

    def drop_node(self, node_id: str) -> int:
        """Remove a node's vectors from the cache without deleting from DB.

        Called by brain.archive_node(): the DB row stays (archived flag
        on the nodes table), the cache masks it so recall can't surface it.
        """
        return self._cache.drop_node(node_id)

    # ── Reads (serve from cache) ────────────────────────────────────

    def get_primary(self, node_id: str) -> Optional[bytes]:
        return self._cache.get_embedding(node_id, '_primary')

    def get_situation_text(self, node_id: str) -> Optional[str]:
        # As of v24, situation text lives in node_metadata_kv (not in the
        # VectorCache's text field, which was a vector-regeneration artifact).
        # Delegate directly to the inner DAL — one small indexed SQL read.
        # No need to cache: situation text isn't on any hot recall path;
        # get_node hydration batches all kv fields together via MetadataDAL.
        with self._sql_lock:
            return self._inner.get_situation_text(node_id)

    def get_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        return self._cache.get_for_node(node_id)

    def get_all_vectors(self, exclude_archived: bool = True,
                        vector_types: Optional[List[str]] = None,
                        model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Cache-served. Shape matches VectorDAL exactly."""
        exclude_ids = self._archived_ids() if exclude_archived else None
        return self._cache.query(vector_types=vector_types, model=model,
                                 exclude_node_ids=exclude_ids)

    def get_all_situations(self, model: Optional[str] = None
                           ) -> List[Dict[str, Any]]:
        """Cache-served. Always excludes archived — matches VectorDAL."""
        return self._cache.query_situations(
            model=model, exclude_node_ids=self._archived_ids())

    def get_all_with_context(self, exclude_archived: bool = True,
                             types: List[str] = None,
                             project: str = None,
                             model: str = None) -> List[Dict[str, Any]]:
        """Primary vectors + per-node context fields.

        Cache supplies the vectors, SQL supplies the node context (a small
        join against the nodes table — ~2500 rows, indexed, fits in page
        cache comfortably). The expensive scan — 25 MB of vector data per
        call — is gone.
        """
        # 1) Pull all primary vectors from cache.
        primary = self._cache.query_primary_with_text(model=model)
        if not primary:
            return []
        ids = [nid for nid, _ in primary]

        # 2) Fetch per-node context for those ids, filtered.
        ctx_by_id = self._fetch_node_context(
            ids, exclude_archived=exclude_archived,
            types=types, project=project)

        # 3) Zip results — same dict shape as VectorDAL.get_all_with_context.
        out = []
        for nid, blob in primary:
            ctx = ctx_by_id.get(nid)
            if ctx is None:
                # Filtered out by archived/type/project → skip.
                continue
            out.append({
                'node_id': nid,
                'embedding': blob,
                'personal': ctx['personal'],
                'personal_context': ctx['personal_context'],
                'confidence': ctx['confidence'],
                'critical': ctx['critical'] or 0,
                'title': ctx['title'] or '',
                'type': ctx['type'] or '',
                'created_at': ctx['created_at'],
                'emotion': ctx['emotion'] or 0,
                'access_count': ctx['access_count'] or 0,
            })
        return out

    def find_missing(self, vector_type: str, limit: int = 50,
                     model: Optional[str] = None,
                     node_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        """Delegate — backfill path, cold, needs node.title/content."""
        with self._sql_lock:
            return self._inner.find_missing(vector_type, limit,
                                            model=model, node_ids=node_ids)

    def get_coverage_stats(self) -> Dict[str, Any]:
        """Delegate — cold path, used by dashboard."""
        with self._sql_lock:
            return self._inner.get_coverage_stats()

    def count(self) -> int:
        """Total vector row count from cache."""
        return self._cache.stats()['total_rows']

    # ── Diagnostics ─────────────────────────────────────────────────

    def cache_stats(self) -> Dict[str, Any]:
        """Expose VectorCache stats — useful for `brain diagnose` later."""
        return self._cache.stats()

    # ── Internal helpers ────────────────────────────────────────────

    def _archived_ids(self) -> Set[str]:
        """Fetch the set of archived node ids — a small indexed query."""
        with self._sql_lock:
            rows = self.conn.execute(
                'SELECT id FROM nodes WHERE archived = 1').fetchall()
        return {r[0] for r in rows}

    def _fetch_node_context(self, ids: List[str], *,
                            exclude_archived: bool,
                            types: Optional[List[str]],
                            project: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """One bounded SQL: SELECT ... FROM nodes WHERE id IN (...)
        with the filters that get_all_with_context previously applied
        server-side. Returns {id → context dict}.

        Chunked at 900 ids to stay under SQLite's param limit.
        """
        if not ids:
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        for chunk_start in range(0, len(ids), 900):
            chunk = ids[chunk_start:chunk_start + 900]
            where = ['n.id IN (%s)' % ','.join('?' * len(chunk))]
            params: List[Any] = list(chunk)
            if exclude_archived:
                where.append('n.archived = 0')
            if types:
                where.append('n.type IN (%s)' % ','.join('?' * len(types)))
                params.extend(types)
            if project:
                where.append('(n.project = ? OR n.project IS NULL)')
                params.append(project)
            sql = ('SELECT n.id, n.personal, n.personal_context, '
                   'n.confidence, n.critical, n.title, n.type, '
                   'n.created_at, n.emotion, n.access_count '
                   'FROM nodes n WHERE ' + ' AND '.join(where))
            with self._sql_lock:
                fetched = self.conn.execute(sql, params).fetchall()
            for r in fetched:
                result[r[0]] = {
                    'personal': r[1], 'personal_context': r[2],
                    'confidence': r[3], 'critical': r[4],
                    'title': r[5], 'type': r[6],
                    'created_at': r[7], 'emotion': r[8],
                    'access_count': r[9],
                }
        return result
