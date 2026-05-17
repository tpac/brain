"""
brain — BrainRemember Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from .brain_constants import TYPE_CONFIDENCE
from .dal import GraphDAL, VectorDAL
from .brain_constants import (
    ENRICHMENT_NEIGHBOR_COUNT,
    ENRICHMENT_PROMPT_TEMPLATE,
)
from typing import Any, Dict, List, Optional, Set
import json
import math
import re
import sys
import time

from .brain_constants import (
    TFIDF_STOP_WORDS,
)



class BrainRememberMixin:
    """Remember methods for Brain."""

    # ═══════════════════════════════════════════════════════════════
    # Unified metadata storage — single path for remember() and revise()
    # ═══════════════════════════════════════════════════════════════

    # Fields that are control parameters, not node metadata.
    # These are consumed by remember()/revise() logic and should never be stored.
    _CONTROL_FIELDS = frozenset({
        'connections', 'auto_connect', 'skip_embedding',
        'reason', 'updates', 'connect_to',
    })

    def _store_node_metadata(self, node_id: str, fields: Dict[str, Any],
                             caller: str = 'unknown') -> int:
        """Store metadata fields for a node. Single path for all write operations.

        Routes each field to the correct storage:
          - STRUCTURAL_FIELDS → already on nodes table (skip)
          - situation → node_metadata_kv (canonical; _situation embedding
            derived later by backfill)
          - PROMOTED metadata_kv fields → node_metadata_kv
          - Emergent/unknown fields → node_metadata_kv
          - Control fields → skip silently (connections, auto_connect, etc.)

        Warns on fields that don't match any storage path.

        Returns count of fields stored.
        """
        from .contract import STRUCTURAL_FIELDS, PROMOTED_FIELDS
        from .dal_metadata import MetadataDAL

        kv_fields = {}
        stored = 0

        for field, value in fields.items():
            # Control params — consumed by callers, never stored
            if field in self._CONTROL_FIELDS:
                continue

            # Structural — already on nodes table, handled by INSERT/UPDATE
            if field in STRUCTURAL_FIELDS:
                continue

            # Empty values — skip
            if value is None or (isinstance(value, str) and not value.strip()):
                continue

            # Situation lives in node_metadata_kv as of v24 — alongside
            # question, reasoning, etc. The embedding (BLOB) is generated
            # later by backfill_vectors() reading from kv.
            #
            # Pass raw values through to set_many — its _encode_value handles
            # str / list / dict / primitives consistently. Doing str(value)
            # here would str()-ify lists into Python repr (`"['a','b']"`)
            # which isn't JSON-parseable. Aspects (Step 5b) need clean lists.
            if field == 'situation':
                kv_fields['situation'] = value
                continue

            # Promoted metadata_kv field — store in KV
            if field in PROMOTED_FIELDS:
                pf = PROMOTED_FIELDS[field]
                if pf.get('store') == 'metadata_kv':
                    kv_fields[field] = value
                    continue
                # Promoted field with different store — log error so it's visible
                self._log_error('store_metadata_unhandled',
                                ValueError('field "%s" (store=%s) not handled' % (
                                    field, pf.get('store'))),
                                '%s: node %s' % (caller, node_id[:8]))
                continue

            # Emergent field — any unknown field goes to metadata_kv
            kv_fields[field] = value

        if kv_fields:
            try:
                count = MetadataDAL(self.conn).set_many(node_id, kv_fields)
                stored += count
            except Exception as _e:
                self._log_error('store_metadata_kv', _e,
                                '%s: KV for %s (%d fields)' % (
                                    caller, node_id[:8], len(kv_fields)))

        return stored

    # ═══════════════════════════════════════════════════════════════
    # Unified archive — single path for all archive operations
    # ═══════════════════════════════════════════════════════════════

    def archive_node(self, node_id: str, archived_by: str,
                     reason: str = '', extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Archive a node. Single path for all callers.

        What it does:
          1. Guards: rejects locked/critical nodes
          2. Sets archived=1, updated_at=now
          3. Stores audit metadata: archived_by, archived_reason, archived_at
          4. Soft-archives edge_relations (v25 — archived=1 preserves history
             for future recovery; edges aggregate row stays for edge_id stability)
          5. Deletes vectors from node_enrichments (embeddings are expensive to keep)
          6. Removes from FTS5 index

        Args:
            node_id: Node to archive.
            archived_by: Who is archiving. Convention: "s2:consolidation",
                         "s2:community_detection", "hook:integrity", "anchor", etc.
            reason: Human-readable reason for the archive.
            extra: Optional dict of additional metadata to store (e.g. consolidated_into).

        Returns:
            Dict with ok=True/False and details.
        """
        from datetime import datetime, timezone
        from .dal_metadata import MetadataDAL

        ts = datetime.now(timezone.utc).isoformat()

        # Fetch node
        row = self.conn.execute(
            'SELECT id, locked, critical, title, type FROM nodes WHERE id = ?',
            (node_id,)).fetchone()
        if not row:
            return {'ok': False, 'error': 'Node not found', 'node_id': node_id}

        full_id, locked, critical, title, node_type = row

        # Guard: never archive locked or critical nodes
        if locked or critical:
            flag = 'locked' if locked else 'critical'
            self._log_error('archive_guarded',
                            ValueError('Cannot archive %s node' % flag),
                            '%s tried to archive %s "%s"' % (
                                archived_by, node_id[:8], (title or '')[:40]))
            return {'ok': False, 'error': 'Cannot archive %s node' % flag,
                    'node_id': node_id}

        # 1. Set archived=1
        self.conn.execute(
            'UPDATE nodes SET archived = 1, updated_at = ? WHERE id = ?',
            (ts, full_id))

        # 2. Store audit metadata (_sys_ prefix = system fields, filtered from LLM rendering)
        audit = {
            '_sys_archived_by': archived_by,
            '_sys_archived_reason': reason or 'no reason provided',
            '_sys_archived_at': ts,
        }
        if extra:
            for k, v in extra.items():
                if v is not None:
                    audit['_sys_archived_%s' % k] = str(v)
        try:
            MetadataDAL(self.conn).set_many(full_id, audit)
        except Exception as _e:
            self._log_error('archive_metadata', _e,
                            'storing audit for %s' % node_id[:8])

        # 3. Soft-archive edge_relations touching this node (v25). Preserves
        # edge history so future rescue/reconstruction is possible — old
        # hard-delete destroyed provenance irreversibly. The edges aggregate
        # row is left intact; all reads filter via edge_relations joins,
        # so archived edges stop joining in.
        edge_ids = [r[0] for r in self.conn.execute(
            'SELECT edge_id FROM edges WHERE source_id = ? OR target_id = ?',
            (full_id, full_id)).fetchall()]
        edges_deleted = 0
        if edge_ids:
            for i in range(0, len(edge_ids), 500):
                chunk = edge_ids[i:i + 500]
                ph = ','.join('?' * len(chunk))
                cur = self.conn.execute(
                    'UPDATE edge_relations SET archived = 1, archived_at = ?, archived_by = ? '
                    'WHERE edge_id IN (%s) AND archived = 0' % ph,
                    [ts, archived_by] + chunk)
                edges_deleted += cur.rowcount

        # 4. Delete vectors from node_enrichments
        vectors_deleted = self.conn.execute(
            'DELETE FROM node_enrichments WHERE node_id = ?',
            (full_id,)).rowcount

        # 5. Remove from FTS5 index. Some test DBs don't enable FTS5 —
        # skip cleanly when the virtual table is absent, but log any
        # real failure (production always has FTS5).
        has_fts5 = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='nodes_fts'"
        ).fetchone() is not None
        if has_fts5:
            try:
                from .dal import Fts5DAL
                Fts5DAL(self.conn).delete(full_id)
            except Exception as _e:
                self._log_error('archive_fts5', _e,
                                'FTS5 delete for %s' % full_id[:8])

        self.conn.commit()

        # 6. AFTER commit — invalidate the in-memory vector cache so recall's
        # cached matrix doesn't retain dead rows. Order matters: if we
        # dropped before commit and commit failed, the cache would be ahead
        # of the DB (node gone from cache but archived=0 in the DB) — that
        # causes transient "node disappears from recall" until the next
        # daemon restart repairs the cache. No-op when
        # BRAIN_DISABLE_VECTOR_CACHE=1 (plain VectorDAL has no drop_node).
        if hasattr(self._vec_dal, 'drop_node'):
            try:
                self._vec_dal.drop_node(full_id)
            except Exception as _e:
                self._log_error('archive_cache_drop', _e,
                                'cache drop for %s' % full_id[:8])

        # 7. Trace event — S3 + dashboards see who archived what.
        # Tracing must never block the archive itself, but a failure
        # here is real audit data loss — log it so we know.
        try:
            self._trace_dal.append(
                chain_id='archive-%s' % full_id[:8],
                scale='s0', event_type='delta', ref_type='tool_result',
                summary='archived %s by %s' % (full_id[:8], archived_by),
                metadata={
                    'node_id': full_id,
                    'title': (title or '')[:80],
                    'type': node_type,
                    'archived_by': archived_by,
                    'reason': reason,
                    'edges_deleted': edges_deleted,
                    'vectors_deleted': vectors_deleted,
                })
        except Exception as _e:
            self._log_error('archive_trace', _e,
                            'trace write for archived %s' % full_id[:8])

        return {
            'ok': True,
            'node_id': full_id,
            'title': (title or '')[:60],
            'type': node_type,
            'archived_by': archived_by,
            'reason': reason,
            'edges_deleted': edges_deleted,
            'vectors_deleted': vectors_deleted,
        }

    def _tfidf_tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for TF-IDF: expand CamelCase, lowercase, remove stopwords.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (length > 2, non-stopword)
        """
        if not text:
            return []

        # Split CamelCase before lowercasing: "UserDashboard" → "User Dashboard"
        expanded = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        expanded = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', expanded)

        # Lowercase, remove non-alphanumeric (keep hyphens, dots), split
        tokens = expanded.lower()
        tokens = re.sub(r'[^a-z0-9\s\-\.]', ' ', tokens)
        tokens = re.split(r'[\s\-\.]+', tokens)

        # Filter: length > 2, not stopword, remove trailing non-alphanumeric
        result = []
        for w in tokens:
            w = re.sub(r'[^a-z0-9]', '', w)
            if len(w) > 2 and w not in TFIDF_STOP_WORDS:
                result.append(w)

        return result

    def _compute_tf(self, text: str) -> Dict[str, float]:
        """
        Compute term frequency vector (augmented TF formula).

        Args:
            text: Text to analyze

        Returns:
            Dict of term→TF value (0-1)
        """
        tokens = self._tfidf_tokenize(text)
        if not tokens:
            return {}

        # Count term frequencies
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        # Augmented TF: 0.5 + 0.5 * (count / max_freq)
        max_freq = max(freq.values()) if freq else 1
        tf = {}
        for term, count in freq.items():
            tf[term] = 0.5 + 0.5 * (count / max_freq)

        return tf

    def _store_tfidf_vector(self, node_id: str, title: str, content: Optional[str], keywords: Optional[str]):
        """
        Store TF-IDF vector for a node (title + content + keywords).

        Args:
            node_id: Node ID
            title: Node title
            content: Node content (optional)
            keywords: Node keywords (optional)
        """
        full_text = ' '.join(filter(None, [title, content, keywords]))
        tf = self._compute_tf(full_text)

        # Delete old vectors for this node
        self.conn.execute('DELETE FROM node_vectors WHERE node_id = ?', (node_id,))

        # Update document frequency counts
        for term in tf.keys():
            self.conn.execute(
                'INSERT INTO doc_freq (term, count) VALUES (?, 1) ON CONFLICT(term) DO UPDATE SET count = count + 1',
                (term,)
            )

        # Store TF values (TF-IDF computed at query time)
        for term, tf_val in tf.items():
            self.conn.execute(
                'INSERT OR REPLACE INTO node_vectors (node_id, term, tf) VALUES (?, ?, ?)',
                (node_id, term, tf_val)
            )

        self.conn.commit()

    def _tfidf_score(self, query_terms: List[str], node_id: str) -> float:
        """
        Compute TF-IDF cosine similarity between query and single node.

        Args:
            query_terms: Tokenized query
            node_id: Node to score

        Returns:
            Cosine similarity (0-1)
        """
        if not query_terms:
            return 0

        total_docs = self._get_node_count()
        if total_docs == 0:
            return 0

        # Build query vector
        query_vec = {}
        for term in query_terms:
            query_vec[term] = query_vec.get(term, 0) + 1

        # Normalize query vector
        q_max = max(query_vec.values()) if query_vec else 1
        for t in query_vec:
            query_vec[t] /= q_max

        # Get node's TF values for matching terms
        placeholders = ','.join('?' * len(query_terms))
        cursor = self.conn.execute(
            f'SELECT term, tf FROM node_vectors WHERE node_id = ? AND term IN ({placeholders})',
            [node_id] + query_terms
        )
        node_terms = {row[0]: row[1] for row in cursor.fetchall()}

        if not node_terms:
            return 0

        # Compute cosine similarity with IDF weighting
        dot_product = 0
        query_norm = 0
        doc_norm = 0

        for term in set(list(query_vec.keys()) + list(node_terms.keys())):
            # IDF = log(N / df)
            cursor = self.conn.execute('SELECT count FROM doc_freq WHERE term = ?', (term,))
            row = cursor.fetchone()
            df = row[0] if row else 1
            idf = math.log((total_docs + 1) / (df + 1)) + 1  # smoothed IDF

            q_val = (query_vec.get(term, 0) or 0) * idf
            d_val = (node_terms.get(term, 0) or 0) * idf

            dot_product += q_val * d_val
            query_norm += q_val * q_val
            doc_norm += d_val * d_val

        denom = math.sqrt(query_norm) * math.sqrt(doc_norm)
        return dot_product / denom if denom > 0 else 0

    def _batch_tfidf_scores(self, query_terms: List[str], node_ids: List[str]) -> Dict[str, float]:
        """
        Batch compute TF-IDF scores for multiple nodes (efficient).

        Args:
            query_terms: Tokenized query
            node_ids: List of node IDs to score

        Returns:
            Dict of node_id→score
        """
        if not query_terms or not node_ids:
            return {}

        total_docs = self._get_node_count()
        if total_docs == 0:
            return {}

        # Precompute IDF for all query terms
        idf_map = {}
        for term in set(query_terms):
            cursor = self.conn.execute('SELECT count FROM doc_freq WHERE term = ?', (term,))
            row = cursor.fetchone()
            df = row[0] if row else 1
            idf_map[term] = math.log((total_docs + 1) / (df + 1)) + 1

        # Build query vector
        query_vec = {}
        for term in query_terms:
            query_vec[term] = query_vec.get(term, 0) + 1

        q_max = max(query_vec.values()) if query_vec else 1
        for t in query_vec:
            query_vec[t] /= q_max

        # Query norm (constant for all docs)
        query_norm_sq = 0
        for term, q_val in query_vec.items():
            idf = idf_map.get(term, 1)
            query_norm_sq += (q_val * idf) ** 2

        query_norm = math.sqrt(query_norm_sq)
        if query_norm == 0:
            return {}

        # Get all matching vectors in one query
        unique_terms = list(set(query_terms))
        term_placeholders = ','.join('?' * len(unique_terms))
        node_placeholders = ','.join('?' * len(node_ids))
        cursor = self.conn.execute(
            f'SELECT node_id, term, tf FROM node_vectors WHERE term IN ({term_placeholders}) AND node_id IN ({node_placeholders})',
            unique_terms + node_ids
        )

        # Group by node_id
        node_term_maps = {}
        for node_id, term, tf in cursor.fetchall():
            if node_id not in node_term_maps:
                node_term_maps[node_id] = {}
            node_term_maps[node_id][term] = tf

        # Compute similarity for each node
        scores = {}
        for node_id in node_ids:
            node_term_map = node_term_maps.get(node_id)
            if not node_term_map:
                scores[node_id] = 0
                continue

            dot_product = 0
            doc_norm_sq = 0

            for term, tf_val in node_term_map.items():
                idf = idf_map.get(term, 1)
                d_val = tf_val * idf
                q_val = (query_vec.get(term, 0) or 0) * idf
                dot_product += q_val * d_val
                doc_norm_sq += d_val * d_val

            doc_norm = math.sqrt(doc_norm_sq)
            scores[node_id] = dot_product / (query_norm * doc_norm) if doc_norm > 0 else 0

        return scores

    def _rebuild_tfidf_index(self):
        """Rebuild TF-IDF index for all existing (non-archived) nodes."""
        # Clear existing index
        self.conn.execute('DELETE FROM node_vectors')
        self.conn.execute('DELETE FROM doc_freq')

        # Fetch all non-archived nodes
        cursor = self.conn.execute('SELECT id, title, content, keywords FROM nodes WHERE archived = 0')
        all_nodes = cursor.fetchall()

        for node_id, title, content, keywords in all_nodes:
            full_text = ' '.join(filter(None, [title, content, keywords]))
            tf = self._compute_tf(full_text)

            # Update doc_freq
            for term in tf.keys():
                self.conn.execute(
                    'INSERT INTO doc_freq (term, count) VALUES (?, 1) ON CONFLICT(term) DO UPDATE SET count = count + 1',
                    (term,)
                )

            # Store TF values
            for term, tf_val in tf.items():
                self.conn.execute(
                    'INSERT OR REPLACE INTO node_vectors (node_id, term, tf) VALUES (?, ?, ?)',
                    (node_id, term, tf_val)
                )

        self.conn.commit()

    def remember(self, type: str, title: str, content: Optional[str] = None,
                 keywords: Optional[str] = None, locked: bool = False,
                 connections: Optional[List[Dict[str, Any]]] = None,
                 emotion: float = 0, emotion_label: str = 'neutral',
                 emotion_source: str = 'auto', project: Optional[str] = None,
                 confidence: float = 1.0,
                 personal: Optional[str] = None,
                 personal_context: Optional[str] = None,
                 critical: bool = False,
                 encoding_source: Optional[str] = None,
                 situation: Optional[str] = None,
                 source_turn_id: Optional[str] = None,
                 evolution_status: Optional[str] = None,
                 # Promoted metadata fields (stored in node_metadata_kv)
                 reasoning: Optional[str] = None,
                 user_raw_quote: Optional[str] = None,
                 anchor_raw_quote: Optional[str] = None,
                 # `correction_of` parameter removed 2026-05-17 — corrections
                 # tracked via correction_improvement-aspect edges
                 # (corrects/supersedes/reframes/...). See render_corrections()
                 # + correction_enrich() for the read path.
                 correction_pattern: Optional[str] = None,
                 source_context: Optional[str] = None,
                 confidence_rationale: Optional[str] = None,
                 alternatives: Optional[List[Dict[str, str]]] = None,
                 change_impacts: Optional[List[Dict[str, str]]] = None,
                 source_attribution: Optional[str] = None,
                 scope: Optional[str] = None,
                 auto_connect: bool = True,
                 connect_to: Optional[List[Any]] = None,
                 **extra_fields) -> Dict[str, Any]:
        """
        Store a new memory node with semantic indexing and connections.

        Accepts ALL contract fields. Core fields go to the nodes table,
        promoted fields go to node_metadata_kv/node_enrichments, and any
        unknown fields are stored as emergent metadata in node_metadata_kv.

        Returns:
            Dict with id, type, title, and related_nodes (top 5 similar existing nodes).
        """
        # Validate personal flag
        if personal and personal not in ('fixed', 'fluid', 'contextual'):
            personal = None

        # Constitution: only intentional (anchor) encoding can create locked nodes.
        # All automated sources (encoder, idle, hook) must earn permanence.
        # encoding_source convention: "category:process" e.g. "encoder:sonnet", "idle:redistribution"
        if encoding_source and not encoding_source.startswith('anchor') and locked:
            locked = False

        node_id = self._generate_id(type)
        ts = self.now()

        # ══════════════════════════════════════════════════════════════
        # v6: AUTO-ENRICHMENT — make every node rich by default
        # The brain's data was shallow because rich encoding required
        # extra effort. Now remember() fills in what it can automatically.
        # ══════════════════════════════════════════════════════════════

        # Auto-set confidence by type if caller left it at default
        # TYPE_CONFIDENCE from brain_constants defines how reliable each type tends to be
        if confidence == 1.0:  # default = unset by caller
            confidence = TYPE_CONFIDENCE.get(type, 0.70)

        # Extract keywords if not provided
        if not keywords:
            keywords = self._extract_keywords(f'{title} {content or ""}')

        # v4: Fixed personal nodes are always locked — their whole point is permanence
        if personal == 'fixed':
            locked = True

        # v5: Auto-generate content summary for tiered recall
        content_summary = self._generate_summary(title, content)

        # INSERT into nodes table
        from .brain_constants import CURRENT_ENCODING_VERSION
        self.conn.execute(
            '''INSERT INTO nodes
               (id, type, title, content, content_summary, keywords,
                activation, stability, locked, confidence,
                recency_score, emotion, emotion_label, emotion_source, project,
                personal, personal_context, encoding_version, encoding_source,
                evolution_status, source_turn_id,
                last_accessed, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 1.0, 1.0, ?, ?, 1.0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (node_id, type, title, content, content_summary, keywords,
             1 if locked else 0, confidence,
             emotion, emotion_label, emotion_source, project,
             personal, personal_context, CURRENT_ENCODING_VERSION,
             encoding_source or 'anchor',
             evolution_status, source_turn_id,
             ts, ts, ts)
        )
        self.conn.commit()

        # Store all metadata via unified path — promoted, emergent, and extra fields.
        _meta_fields = {}
        # Promoted fields passed as explicit args
        for _name, _val in [
            ('reasoning', reasoning), ('user_raw_quote', user_raw_quote),
            ('anchor_raw_quote', anchor_raw_quote),
            ('correction_pattern', correction_pattern), ('source_context', source_context),
            ('confidence_rationale', confidence_rationale), ('scope', scope),
            ('source_attribution', source_attribution), ('situation', situation),
        ]:
            if _val is not None:
                _meta_fields[_name] = _val
        # Extra fields from callers (community metadata, any emergent fields)
        _meta_fields.update(extra_fields)
        if _meta_fields:
            self._store_node_metadata(node_id, _meta_fields, caller='remember')

        # v5.2: Critical flag requires operator approval — don't set directly
        if critical:
            self._add_pending_critical(node_id, title)

        # v5: Build TF-IDF vector for this node
        try:
            self._store_tfidf_vector(node_id, title, content, keywords)
        except Exception as e:
            self._log_error('tfidf_vector_store', e, 'storing TF-IDF vector for node %s' % node_id[:12])

        # v9: Sync FTS5 full-text search index
        try:
            from .dal import Fts5DAL
            Fts5DAL(self.conn).upsert(node_id, title, content or '', keywords or '')
            self.conn.commit()
        except Exception as e:
            self._log_error('fts5_sync_remember', e, 'syncing FTS5 for node %s' % node_id[:12])

        # Vector computation handled by the embed_queue worker — this node
        # is marked dirty and will be embedded within ~5s. S2 Heal catches
        # anything that slips through on crash.
        try:
            from . import embed_queue
            embed_queue.enqueue(node_id)
        except Exception as e:
            self._log_error('embed_enqueue_remember', e, 'enqueue %s' % node_id[:12])
        embedding_stored = False

        # Create connections
        if connections:
            for conn in connections:
                target_id = conn.get('target_id')
                relation = conn.get('relation', 'related')
                weight = conn.get('weight', 0.5)
                if target_id:
                    try:
                        self.connect(node_id, target_id, relation, weight)
                    except (ValueError, Exception) as e:
                        self._log_error('remember_connection', e,
                                        'connecting %s → %s' % (node_id[:8], target_id[:8]))

        # connect_to: title-resolved typed edges. When called standalone (not
        # from a batch), there are no siblings — only catalog fallback applies.
        # Inside remember_batch / brain_batch, connect_to is popped from the
        # spec BEFORE this call and processed AFTER all siblings are created.
        if connect_to:
            # remember() (single-node path) doesn't track connect_to failures
            # in its return shape; per-call logging via _log_error covers that.
            self._apply_connect_to(node_id, connect_to, sibling_map=None)

        # v6→v7: Auto-connect to conversation context (Machine 1)
        # Connect new node to top 3 most semantically similar recently-accessed nodes.
        # Disabled via auto_connect=False when caller manages connections explicitly
        # (e.g. S2 community nodes, batch imports).
        if auto_connect:
            try:
                new_node_emb = None
                if embedding_stored:
                    _vdal = self._vec_dal
                    _emb = _vdal.get_primary(node_id)
                    if _emb:
                        new_node_emb = _emb

                recent = self.conn.execute('''
                    SELECT n.id, ne.embedding FROM nodes n
                    LEFT JOIN node_enrichments ne ON ne.node_id = n.id AND ne.vector_type = '_primary'
                    WHERE n.id != ? AND n.archived = 0
                      AND n.last_accessed > datetime('now', '-1 hour')
                      AND n.type NOT IN ('thought', 'intuition')
                    ORDER BY n.last_accessed DESC LIMIT 10
                ''', (node_id,)).fetchall()

                if new_node_emb and recent:
                    scored = []
                    for (recent_id, recent_emb) in recent:
                        if recent_emb:
                            sim = embedder.cosine_similarity(new_node_emb, recent_emb)
                            scored.append((recent_id, sim))
                        else:
                            scored.append((recent_id, 0.0))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    for recent_id, sim in scored[:3]:
                        if sim > 0.3:
                            from .dal import GraphDAL
                            graph_dal = GraphDAL(self.conn)
                            if not graph_dal.edge_exists(node_id, recent_id):
                                self.connect(node_id, recent_id, 'co_accessed', max(0.2, sim * 0.5))
                elif recent:
                    for (recent_id, _) in recent[:3]:
                        from .dal import GraphDAL
                        graph_dal = GraphDAL(self.conn)
                        if not graph_dal.edge_exists(node_id, recent_id):
                            self.connect(node_id, recent_id, 'co_accessed', 0.2)
            except Exception as e:
                self._log_error('auto_connect', e, 'auto-connecting node %s to recent context' % node_id[:12])

        # v11: Emergent bridging at store-time
        bridges = []
        try:
            bridges = self._bridge_at_store_time(node_id)
        except Exception as e:
            self._log_error('bridge_at_store', e, 'emergent bridging for node %s' % node_id[:12])

        # v5: Track encoding for heartbeat
        try:
            self.record_remember()
        except Exception as e:
            self._log_error('record_remember', e, 'tracking encoding for heartbeat')

        # v5.1: Track node in current conversation segment
        try:
            self.add_to_segment(node_id)
        except Exception as e:
            self._log_error('add_to_segment', e, 'tracking node %s in conversation segment' % node_id[:12])

        # v6: Generate enrichment prompt for Claude to fill in.
        # The brain recalls neighbors and builds a structured prompt.
        # If enrichments are provided inline (from a previous enrich() call), store them.
        # Otherwise, return the prompt so Claude can fill it in.
        enrichment_prompt = None
        enrichment_stored = 0
        try:
            enrichment_prompt = self._build_enrichment_prompt(node_id, title, content)
        except Exception as e:
            print(f'[brain] V5 enrichment prompt failed for {node_id}: {e}', file=sys.stderr)

        # message_stream escalation REMOVED 2026-04-05 — encoding reads from traces

        # _store_node_metadata removed 2026-04-13 — old table, KV handles this via _store_metadata_kv above.

        # v9: Recall-on-create — return related nodes so caller can connect immediately
        related_nodes = []
        try:
            from .pipeline_contract import ENCODING_AGENT
            if embedding_stored:
                recall_result = self.recall(query='%s %s' % (title, (content or '')[:ENCODING_AGENT['recall_on_create_query_limit']]), limit=ENCODING_AGENT['recall_on_create_limit'] + 1, source='internal')
                for r in recall_result.get('results', []):
                    if r.get('id') != node_id:
                        related_nodes.append({
                            'id': r.get('id', ''),
                            'type': r.get('type', ''),
                            'title': r.get('title', ''),
                            'content': (r.get('content', '') or '')[:ENCODING_AGENT['recall_on_create_content_limit']],
                            'confidence': r.get('confidence', 0),
                            'score': round(r.get('effective_activation', 0), 3),
                        })
                    if len(related_nodes) >= ENCODING_AGENT['recall_on_create_limit']:
                        break
        except Exception as e:
            self._log_error('remember_recall_on_create', e, 'recall-on-create for %s' % node_id[:8])

        return {
            'id': node_id,
            'type': type,
            'title': title,
            'embedding_stored': embedding_stored,
            'enrichment_prompt': enrichment_prompt,
            'related_nodes': related_nodes,
        }

    # ═══════════════════════════════════════════════════════════════
    # v8: revise() — Encoding IS updating existing knowledge
    # ═══════════════════════════════════════════════════════════════

    def revise(self, node_id: str, content: str = None, reason: str = '',
               updates: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Update fields on an existing node. Per-field replace semantics.

        Three ways to call (all equivalent):
          revise(node_id, content="new text", reason="why")
          revise(node_id, updates={"confidence": 0.9, "keywords": "new kw"}, reason="why")
          revise(node_id, situation="When debugging", reason="adding situation")

        Behavior contract:
        - Immutable fields ({id, created_at, locked}) are skipped with a
          warning. Other fields in the same call still process; the skipped
          field surfaces in the result dict's `warnings` list.
        - Specified fields are REPLACED with the passed value.
        - Unspecified fields are PRESERVED (only the keys you pass are touched).
        - Returns deltas in the result dict — caller (typically daemon_dispatch)
          emits a trace event with these deltas as the canonical revision
          history. There is no per-node history blob; query traces instead.

        After any revision: re-embeds, re-indexes TF-IDF, updates timestamps.
        """
        # Merge updates from all sources
        all_updates = dict(updates or {})
        all_updates.update(kwargs)
        if content:
            all_updates['content'] = content

        if not all_updates:
            return {'error': 'No updates provided', 'node_id': node_id}

        # Capture the FULL field set NOW for vector invalidation. `all_updates`
        # gets mutated below (content is popped, etc.), so we need the
        # original set or the invalidation step misses fields.
        fields_changed_for_invalidation = set(all_updates.keys())

        # Fetch existing node
        row = self.conn.execute(
            'SELECT id, type, title, content, archived FROM nodes WHERE id = ?',
            (node_id,)).fetchone()
        if not row:
            return {'error': 'Node not found', 'node_id': node_id}
        if row[4] == 1:
            return {'error': 'Cannot revise archived node', 'node_id': node_id}

        existing_id, node_type, title, old_content, _ = row
        old_content = old_content or ''
        ts = self.now()

        # ── Field classification ──
        # Top-level fields live on the nodes table (updatable via SQL).
        # Immutable fields are silently skipped with a warning.
        NODES_TABLE_FIELDS = {
            'title', 'type', 'keywords', 'confidence', 'emotion',
            'emotion_label', 'project', 'personal', 'personal_context',
            'critical', 'evolution_status', 'encoding_source',
            'archived',  # allows revise(archived=True) for consolidation
        }
        IMMUTABLE = {'id', 'created_at', 'locked'}

        # ── Filter skipped fields (immutable, locked-archive) ──
        # Skipped fields don't write to nodes/KV/SQL and don't appear in
        # deltas. They surface in the return dict's `warnings` list so
        # callers can detect partial-success without parsing logs.
        skipped_fields = []  # list of (field, reason)
        writable = {}
        for field, value in all_updates.items():
            if field in IMMUTABLE:
                self._log_error('revise_immutable',
                                ValueError('Cannot revise immutable field: %s' % field),
                                'node %s attempted to revise %s' % (node_id[:8], field))
                skipped_fields.append((field, 'immutable'))
                continue
            if field == 'archived' and value:
                lock_row = self.conn.execute(
                    'SELECT locked, critical FROM nodes WHERE id = ?',
                    (node_id,)).fetchone()
                if lock_row and (lock_row[0] or lock_row[1]):
                    self._log_error('revise_archive_locked',
                                    ValueError('Cannot archive locked/critical node'),
                                    'node %s' % node_id[:8])
                    skipped_fields.append((field, 'locked_or_critical'))
                    continue
            writable[field] = value

        # ── Capture old values for delta computation (before any write) ──
        # Used by callers (typically dispatch) to emit trace events with
        # field-level history. Replaces the old _sys_revision_history blob.
        old_values = {}
        if 'content' in writable:
            old_values['content'] = old_content

        top_level_to_capture = [k for k in writable if k in NODES_TABLE_FIELDS]
        if top_level_to_capture:
            cols = ', '.join(top_level_to_capture)
            old_row = self.conn.execute(
                'SELECT %s FROM nodes WHERE id = ?' % cols, (node_id,)
            ).fetchone()
            if old_row:
                for i, k in enumerate(top_level_to_capture):
                    old_values[k] = old_row[i]

        kv_to_capture = [
            k for k in writable
            if k not in NODES_TABLE_FIELDS and k != 'content'
        ]
        if kv_to_capture:
            from .dal_metadata import MetadataDAL
            kv_old = MetadataDAL(self.conn).get_fields(node_id, kv_to_capture)
            for k in kv_to_capture:
                old_values[k] = kv_old.get(k)  # None if not previously set

        # Content: replace with new value (history lives in trace deltas now,
        # not the legacy _sys_revision_history KV blob).
        new_content = old_content
        if 'content' in writable:
            new_content = writable.pop('content')

        # Build SQL UPDATE for all fields.
        # Always update: content, content_summary, updated_at, revised_at.
        set_parts = ['content = ?', 'content_summary = ?', 'updated_at = ?', 'revised_at = ?']
        params = [new_content, self._generate_summary(title, new_content), ts, ts]

        for field, value in writable.items():
            if field in NODES_TABLE_FIELDS:
                set_parts.append('%s = ?' % field)
                params.append(value)
                if field == 'title':
                    title = value  # track for re-embed

        params.append(node_id)
        self.conn.execute(
            'UPDATE nodes SET %s WHERE id = ?' % ', '.join(set_parts), params)
        self.conn.commit()

        # Store metadata via unified path — handles promoted, emergent, situation.
        # Only writable (non-skipped) fields get persisted.
        if writable:
            self._store_node_metadata(node_id, writable, caller='revise')

        # Vector invalidation: when a source field changes, the corresponding
        # embedding vector becomes stale. Delete the affected rows so the
        # embed_queue's backfill scan re-embeds from the updated text.
        # WITHOUT this, VectorDAL.find_missing() skips the row (it exists)
        # and the vector keeps encoding outdated text indefinitely. Title
        # changes invalidate the title slot too — collected via SQL UPDATE
        # above and added to the field set here.
        #
        # Failure here is a CORRECTNESS issue (recall serves stale embeddings
        # until next backfill cycle). We log loudly AND surface the failure
        # in the return dict so callers can detect partial-success — silent
        # swallow would hide drift indefinitely.
        vector_invalidation_failed = False
        try:
            from .pipeline_contract import vectors_affected_by
            invalidated_vectors = set()
            for field in fields_changed_for_invalidation:
                invalidated_vectors |= vectors_affected_by(field)
            if invalidated_vectors:
                ph = ','.join('?' * len(invalidated_vectors))
                self.conn.execute(
                    'DELETE FROM node_enrichments WHERE node_id = ? '
                    'AND vector_type IN (%s)' % ph,
                    [node_id, *invalidated_vectors])
                self.conn.commit()
                # Invalidate the in-memory vector cache so recall doesn't
                # serve stale embeddings between now and embed_queue's drain.
                # Replaced hasattr() guard with explicit AttributeError catch:
                # property-access exceptions used to fall through hasattr() as
                # False, silently skipping cache invalidation when a cache
                # IS present but momentarily broken.
                try:
                    self._vec_dal.drop_node(node_id)
                except AttributeError:
                    pass  # plain VectorDAL — no in-memory cache to drop
                except Exception as _ce:
                    self._log_error('revise_vector_cache_drop', _ce,
                                    'cache drop for %s' % node_id[:8])
        except Exception as e:
            vector_invalidation_failed = True
            self._log_error('revise_vector_invalidate', e,
                            'invalidating vectors for %s — STALE EMBEDDINGS '
                            'will be served by recall until next backfill '
                            'cycle catches up' % node_id[:8])

        # Vector (re)computation handled by the embed_queue worker — revisions
        # mark the node dirty so stale text→vector pairs get refreshed within ~5s.
        try:
            from . import embed_queue
            embed_queue.enqueue(node_id)
        except Exception as e:
            self._log_error('embed_enqueue_revise', e, 'enqueue %s' % node_id[:12])
        from .dal import VectorDAL
        _vdal = self._vec_dal
        embedding_updated = False

        # Re-index TF-IDF
        try:
            kw_row = self.conn.execute(
                'SELECT keywords FROM nodes WHERE id = ?', (node_id,)).fetchone()
            current_keywords = kw_row[0] if kw_row else None
            self._store_tfidf_vector(node_id, title, new_content, current_keywords)
        except Exception as e:
            self._log_error("revise_tfidf", e, "Failed to re-index TF-IDF for %s" % node_id[:8])

        # v9: Re-sync FTS5 full-text search index
        try:
            from .dal import Fts5DAL
            kw_for_fts = current_keywords if 'current_keywords' in dir() else ''
            Fts5DAL(self.conn).upsert(node_id, title, new_content, kw_for_fts or '')
            self.conn.commit()
        except Exception as e:
            self._log_error("fts5_sync_revise", e, "syncing FTS5 for %s" % node_id[:8])

        # pending_consolidation table dropped 2026-04-05

        # message_stream escalation REMOVED 2026-04-05

        # ── VERIFICATION: read-back to confirm writes landed ──
        verification_failures = []

        # Verify nodes table fields
        from .dal import NodeDAL
        readback = NodeDAL(self.conn).get_naked_node(node_id)
        if readback:
            for field in list(writable.keys()):
                if field in readback:
                    expected = writable[field]
                    actual = readback.get(field)
                    if actual != expected and str(actual) != str(expected):
                        verification_failures.append(field)
            # Content was popped from `writable` earlier; verify separately
            # (REPLACE semantic — readback must equal new_content exactly).
            # Use old_values to detect that content was actually a write target —
            # `content` named-arg path AND `updates={'content': ...}` path both
            # populate old_values['content'], so this catches either.
            if 'content' in old_values:
                actual_content = readback.get('content') or ''
                if actual_content != new_content:
                    verification_failures.append('content')

        # Situation embedding deferred to backfill — no inline verification needed

        # Vector invalidation failure surfaces here too — same severity as a
        # missed field write, since recall correctness depends on it.
        if vector_invalidation_failed:
            verification_failures.append('vector_invalidation')

        verified = len(verification_failures) == 0

        # ── Build deltas for trace event emission ──
        # Caller (typically daemon_dispatch) uses these to write a single
        # node_revised trace event capturing what changed in this call.
        deltas = []
        for field, new_val in writable.items():
            old_val = old_values.get(field)
            if old_val != new_val:
                deltas.append({
                    'field': field,
                    'old': old_val,
                    'new': new_val,
                })
        # Content was popped from `writable` earlier; check separately.
        if 'content' in old_values and old_values['content'] != new_content:
            deltas.append({
                'field': 'content',
                'old': old_values['content'],
                'new': new_content,
            })

        # Warnings surface skipped fields without requiring log parsing.
        warnings = []
        for field, _reason in skipped_fields:
            if _reason == 'immutable':
                warnings.append('immutable field skipped: %s' % field)
            elif _reason == 'locked_or_critical':
                warnings.append('archive blocked (locked/critical): %s' % field)

        # fields_updated: what was actually written. Excludes skipped fields.
        # Includes 'content' if it was passed (popped from writable earlier).
        fields_updated = list(writable.keys())
        if 'content' in old_values:
            fields_updated.append('content')

        return {
            'id': node_id,
            'type': writable.get('type', node_type),
            'title': title,
            'revised_at': ts,
            'content_length': len(new_content),
            'embedding_updated': embedding_updated,
            'fields_updated': fields_updated,
            'deltas': deltas,
            'warnings': warnings,
            'verified': verified,
            'verification_failures': verification_failures if not verified else [],
            'pending_resolved': 0,
        }

    # ═══════════════════════════════════════════════════════════════
    # v5.2: Critical node approval flow
    # Critical nodes get force-surfaced at boot and boosted in recall.
    # Setting critical=1 requires explicit operator approval.
    # ═══════════════════════════════════════════════════════════════

    def _add_pending_critical(self, node_id: str, title: str):
        """Add a node to the pending critical approvals list."""
        try:
            import json as _json
            pending_json = self.get_config('pending_critical_approvals', '[]')
            pending = _json.loads(pending_json) if pending_json else []
            pending.append({
                'node_id': node_id,
                'title': title,
                'requested_at': self.now()
            })
            self.set_config('pending_critical_approvals', _json.dumps(pending))
        except Exception as e:
            self._log_error('_add_pending_critical', e, 'adding pending critical approval')

    def mark_critical(self, node_id: str, reason: str = '') -> Dict[str, Any]:
        """Propose a node as critical. Does NOT set the flag — requires operator approval via revise().

        Args:
            node_id: The node to mark as critical
            reason: Why this node is critical (for the operator to review)

        Returns:
            Dict with node_id, status='pending', reason
        """
        # Verify node exists
        row = self.conn.execute('SELECT title FROM nodes WHERE id = ?', (node_id,)).fetchone()
        if not row:
            return {'error': f'Node {node_id} not found'}

        try:
            import json as _json
            pending_json = self.get_config('pending_critical_approvals', '[]')
            pending = _json.loads(pending_json) if pending_json else []
            # Don't duplicate
            existing_ids = {p['node_id'] for p in pending}
            if node_id not in existing_ids:
                pending.append({
                    'node_id': node_id,
                    'title': row[0],
                    'reason': reason,
                    'requested_at': self.now()
                })
                self.set_config('pending_critical_approvals', _json.dumps(pending))
        except Exception as e:
            self._log_error('mark_critical', e, 'adding pending critical approval')
            return {'error': str(e)}

        return {'node_id': node_id, 'status': 'pending', 'reason': reason}

    # approve_critical removed 2026-04-13 — never wired to MCP, direct DB write.

    def get_pending_critical(self) -> List[Dict[str, Any]]:
        """Get all pending critical approval requests."""
        try:
            import json as _json
            pending_json = self.get_config('pending_critical_approvals', '[]')
            return _json.loads(pending_json) if pending_json else []
        except Exception as e:
            self._log_error('get_pending_critical', e, 'parsing pending critical approvals JSON')
            return []

    def backfill_summaries(self, batch_size: int = 50) -> Dict[str, Any]:
        """Generate content_summary for existing nodes that lack one. Run during idle."""
        cur = self.conn.execute(
            "SELECT id, title, content FROM nodes WHERE content IS NOT NULL AND content != '' AND content_summary IS NULL LIMIT ?",
            (batch_size,)
        )
        rows = cur.fetchall()
        count = 0
        for node_id, title, content in rows:
            summary = self._generate_summary(title, content)
            if summary:
                self.conn.execute(
                    "UPDATE nodes SET content_summary = ? WHERE id = ?",
                    (summary, node_id)
                )
                count += 1
        if count:
            self.conn.commit()
        return {'backfilled': count, 'remaining': len(rows) - count}

    def _compute_group_vectors(self, node_id: str, title: str, content: str,
                               situation: str = None, **metadata_fields):
        """Compute and store multi-vector group embeddings for a node.

        v23: This function is NO LONGER called from remember()/revise().
        Group vectors are computed by backfill_vectors() in idle maintenance.
        Kept for backward compat and manual use.

        Architecture: 4 groups defined in pipeline_contract.EMBEDDING_GROUPS.
        - title: always computed (diagnostic pointer)
        - blend (_primary): stored separately on write path (skip here)
        - high_meta: situation + quotes — only if fields exist
        - other_meta: reasoning + correction_pattern + emergent — only if fields exist

        Vectors stored in node_enrichments with vector_type matching the group name.
        """
        from . import embedder
        from .pipeline_contract import (EMBEDDING_GROUPS, EMBEDDING_SKIP_FIELDS,
                                        EMBEDDING_FIELD_CHAR_LIMIT)

        if not embedder.is_ready():
            self._log_error('group_vectors_skip', None,
                            'embedder not ready — skipping group vectors for %s' % node_id[:8])
            return

        # Build field value lookup: all available fields for this node
        field_values = {'title': title, 'content': content}
        if situation:
            field_values['situation'] = situation
        for k, v in metadata_fields.items():
            if v and isinstance(v, str) and v.strip() and k not in EMBEDDING_SKIP_FIELDS:
                field_values[k] = v

        # Also read any KV metadata not passed as args (emergent fields)
        try:
            from .dal_metadata import MetadataDAL
            dal = MetadataDAL(self.conn)
            kv = dal.get(node_id)
            for k, v in kv.items():
                if k not in field_values and k not in EMBEDDING_SKIP_FIELDS and v and v.strip():
                    field_values[k] = v
        except Exception as _e:
            self._log_error('enrichment_field_values', _e, 'collecting field values for enrichment')

        for group_name, group_config in EMBEDDING_GROUPS.items():
            # Skip blend — it's the primary embedding (_primary in node_enrichments)
            if group_config.get('vector_type') == '_primary':
                continue

            vector_type = group_config['vector_type']
            group_fields = group_config.get('fields', [])

            # Collect text parts for this group
            parts = []
            for field_name in group_fields:
                if field_name == '_emergent':
                    # Emergent: any KV field not explicitly in ANY group
                    all_explicit = set()
                    for g in EMBEDDING_GROUPS.values():
                        all_explicit.update(f for f in g.get('fields', []) if f != '_emergent')
                    for k, v in field_values.items():
                        if k not in all_explicit and k not in ('title', 'content'):
                            parts.append(v[:EMBEDDING_FIELD_CHAR_LIMIT])
                elif field_name == '_edge_descriptions':
                    # Edge context: descriptions via GraphDAL helper.
                    # Centralizes: archived=0 (v25), noise exclusion
                    # (co_accessed, emergent_bridge, community_member),
                    # min_length, and the weight-ordered LIMIT.
                    try:
                        from .dal import GraphDAL
                        descriptions = GraphDAL(self.conn).get_edge_descriptions_for(
                            node_id, min_length=10, limit=5)
                        for desc in descriptions:
                            parts.append(desc[:EMBEDDING_FIELD_CHAR_LIMIT])
                    except Exception as _e:
                        self._log_error('edge_context_descriptions', _e,
                                        'building edge_context for %s' % node_id[:8])
                else:
                    val = field_values.get(field_name, '')
                    if val:
                        parts.append(val[:EMBEDDING_FIELD_CHAR_LIMIT])

            # Skip group if no data (unless always_compute)
            if not parts and not group_config.get('always_compute'):
                continue

            if not parts:
                continue  # Even always_compute needs at least one field

            # Embed and store
            embed_text = ". ".join(parts)
            blob = embedder.embed_document(embed_text)
            if blob:
                self.conn.execute(
                    '''INSERT OR REPLACE INTO node_enrichments
                       (id, node_id, vector_type, text, embedding, model, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (f'{node_id}_{vector_type}', node_id, vector_type,
                     embed_text[:500], blob, embedder.stats.get('model_name', ''),
                     self.now()))

        self.conn.commit()

    # _store_node_metadata removed 2026-04-13 — old node_metadata table dropped.

    def remember_rich(self, type: str, title: str, content: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """Backward-compatible wrapper — remember() now handles all fields directly."""
        return self.remember(type=type, title=title, content=content, **kwargs)

    # ═══════════════════════════════════════════════════════════════
    # connect_to resolution — sibling-aware, sequencing-agnostic
    # ═══════════════════════════════════════════════════════════════

    def _resolve_connect_to_entry(self, entry, sibling_map=None, exclude_self=None):
        """Resolve a connect_to entry to (target_id, relation_pairs).

        sibling_map: {lowercased_title: node_id} for nodes created in the
                     same batch. Sibling matching is CASE-INSENSITIVE —
                     keys are lowered when the map is built (see
                     remember_batch sibling_map construction) AND lowered
                     again at lookup. Sibling exact-match (case-insensitive)
                     wins over catalog fuzzy-match — NEW wins on title
                     collision. If you mean an existing catalog node, use
                     revise on its id, not duplicate-title remember.
                     (Catalog fuzzy-match itself preserves case — only the
                     batch-sibling lookup is normalized.)
        exclude_self: source node_id; resolution to this id is treated as a
                      self-reference and rejected.

        Returns (target_id, [(relation, description), ...]) or (None, []) on
        any failure. All failures log loudly via _log_error so the dashboard
        sees them — no silent skips.
        """
        # Parse entry shape (string or dict)
        if isinstance(entry, str):
            title_query = entry
            relation_pairs = [('related', '')]
        elif isinstance(entry, dict):
            title_query = entry.get('title', '')
            if not title_query:
                self._log_error(
                    'connect_to_invalid',
                    ValueError("connect_to entry missing 'title' field"),
                    'entry=%s' % str(entry)[:200])
                return None, []
            if isinstance(entry.get('relations'), list):
                relation_pairs = []
                for r in entry['relations']:
                    if not isinstance(r, dict):
                        continue
                    rel = r.get('relation', 'related')
                    desc = r.get('why', r.get('description', ''))
                    relation_pairs.append((rel, desc))
                if not relation_pairs:
                    self._log_error(
                        'connect_to_invalid',
                        ValueError("connect_to relations array is empty or malformed"),
                        'entry=%s' % str(entry)[:200])
                    return None, []
            else:
                rel = entry.get('relation', 'related')
                desc = entry.get('why', entry.get('description', ''))
                relation_pairs = [(rel, desc)]
        else:
            self._log_error(
                'connect_to_invalid',
                TypeError("connect_to entry must be str or dict, got %s"
                          % type(entry).__name__),
                'entry=%s' % str(entry)[:200])
            return None, []

        target_id = None

        # Pass 0: ID-shape pre-check. The encoder sometimes passes an 8+ char
        # hex ID in the `title` field when it really means "connect to this
        # specific known node by id" — e.g. when an ID was visible in the
        # conversation (recalled context, prior tool result, surfaced trace).
        # Without this check, sibling-map and fuzzy-title both miss because
        # neither matches an opaque hash. Resolve via id-prefix lookup; if
        # found, prefer it over the title-based passes. Log a soft warning
        # so we can see how often the encoder does this (signal for prompt
        # tuning, not a hard error).
        import re as _re
        if _re.fullmatch(r'[0-9a-fA-F]{8,}', title_query.strip()):
            try:
                row = self.conn.execute(
                    'SELECT id FROM nodes WHERE id LIKE ? LIMIT 2',
                    (title_query.strip().lower() + '%',)).fetchall()
                if len(row) == 1:
                    target_id = row[0][0]
                elif len(row) > 1:
                    # Ambiguous prefix — log and fall through to title path.
                    self._log_error(
                        'connect_to_id_prefix_ambiguous',
                        ValueError(
                            "connect_to title looked like an id but matched "
                            "multiple nodes; falling back to title search"),
                        'prefix=%s matches=%d' % (title_query[:16], len(row)))
            except Exception as e:
                self._log_error(
                    'connect_to_id_lookup_failed', e,
                    'id-prefix lookup for %r' % title_query[:80])
                # fall through to the title path

        # Pass 1: sibling map (NEW wins on title collision)
        if not target_id and sibling_map:
            target_id = sibling_map.get(title_query.lower())

        # Pass 2: catalog fallback via fuzzy title match
        if not target_id:
            try:
                match = self.find_node_by_title(title_query, threshold=0.75)
                if match:
                    target_id = match.get('id')
            except Exception as e:
                self._log_error(
                    'connect_to_failed', e,
                    'find_node_by_title for %r' % title_query[:80])
                return None, []

        # Self-reference guard
        if target_id and exclude_self and target_id == exclude_self:
            self._log_error(
                'connect_to_self',
                ValueError("connect_to would create self-edge"),
                'node=%s title=%s' % (exclude_self[:8], title_query[:80]))
            return None, []

        # Unresolved — neither sibling nor catalog matched
        if not target_id:
            self._log_error(
                'connect_to_unresolved',
                ValueError("connect_to title resolved to nothing"),
                'title=%s' % title_query[:80])
            return None, []

        return target_id, relation_pairs

    def _apply_connect_to(self, src_id, connect_to_spec, sibling_map=None):
        """Resolve and create edges for each connect_to entry from src_id.

        Each entry is independent — failures on one don't affect others.
        All failures log loudly; the function never raises and never blocks
        the surrounding write path.

        Returns (edges_created, failures) — failures is the count of
        connect_to entries that failed (resolve returned None OR the
        connect_typed call raised). The encoder uses this so a cycle
        with N requested connect_to and 0 connect_to_edges has a visible
        reason ("connect_to_failures=N") in the batch result.
        """
        if not connect_to_spec:
            return 0, 0
        if not isinstance(connect_to_spec, list):
            self._log_error(
                'connect_to_invalid',
                TypeError("connect_to must be a list, got %s"
                          % type(connect_to_spec).__name__),
                'src=%s' % src_id[:8])
            return 0, 0

        created = 0
        failures = 0
        for entry in connect_to_spec:
            target_id, relation_pairs = self._resolve_connect_to_entry(
                entry, sibling_map=sibling_map, exclude_self=src_id)
            if target_id is None:
                # Resolution failed — _resolve_connect_to_entry already
                # logged via _log_error. Count it so the batch result can
                # surface "tried N, failed M" instead of just "0 created".
                failures += 1
                continue
            for rel, desc in relation_pairs:
                try:
                    self.connect_typed(src_id, target_id, relation=rel,
                                       weight=0.6, description=desc)
                    created += 1
                except Exception as e:
                    failures += 1
                    self._log_error(
                        'connect_to_failed', e,
                        'src=%s target=%s rel=%s' % (
                            src_id[:8], target_id[:8], rel))
        return created, failures

    def remember_batch(self, nodes: List[Dict],
                        connect_to: Optional[List[str]] = None,
                        auto_connect: bool = True) -> Dict[str, Any]:
        """Create multiple nodes in one call. Each node uses the same contract as remember().

        Args:
            nodes: List of dicts, each with the same fields remember() accepts
                   (type, title, content, keywords, situation, reasoning, etc.)
            connect_to: List of existing node titles to fuzzy-match and connect all new nodes to
            auto_connect: If True, auto-connect new nodes to each other

        Returns:
            {nodes_created, results: [{id, title, related_nodes}], connections_created}
        """
        results = []
        created_ids = []
        connections_created = 0

        # Pass 1: create all nodes. Pop per-node connect_to BEFORE calling
        # remember() so it doesn't fire there with an empty sibling_map —
        # we'll process them all together once siblings exist (Pass 2).
        # Build sibling_map: lowercased title → node_id for sequencing-
        # agnostic resolution (B can connect_to A even if declared first).
        sibling_map = {}
        deferred_connects = []  # [(node_id, ct_spec)]
        for spec in nodes:
            if isinstance(spec, dict):
                ct_spec = spec.pop('connect_to', None)
                # Disable inner remember()'s conversation-context auto_connect.
                # The batch owns sibling connection logic (deferred connect_to +
                # auto_connect-each-other below). Inner auto_connect creates
                # co_accessed edges in the WRONG direction relative to typed
                # connect_to (sibling-to-sibling), which causes get_edge_id to
                # return the existing edge and the typed relation gets attached
                # with the physical direction reversed. Caller can still opt in
                # explicitly per node via spec['auto_connect'] = True.
                spec.setdefault('auto_connect', False)
            else:
                ct_spec = None
            result = self.remember(**spec)
            results.append(result)
            if result.get('id'):
                created_ids.append(result['id'])
                title = (spec.get('title') or '').lower()
                if title:
                    sibling_map[title] = result['id']
                if ct_spec:
                    deferred_connects.append((result['id'], ct_spec))

        # Pass 2: resolve per-node connect_to with full sibling_map populated.
        connect_to_failures = 0
        for src_id, ct_spec in deferred_connects:
            edges, fails = self._apply_connect_to(
                src_id, ct_spec, sibling_map=sibling_map)
            connections_created += edges
            connect_to_failures += fails

        # Auto-connect new nodes to each other
        if auto_connect and len(created_ids) > 1:
            for i, src_id in enumerate(created_ids):
                for dst_id in created_ids[i + 1:]:
                    try:
                        self.connect(src_id, dst_id, relation='related_to', weight=0.5)
                        connections_created += 1
                    except Exception as _e:
                        self._log_error('batch_auto_connect', _e, 'connecting %s → %s' % (src_id[:8], dst_id[:8]))

        # Fuzzy-match connect_to titles
        if connect_to:
            created_set = set(created_ids)
            for entry in connect_to:
                # Accept three formats:
                # 1. String: "node title" → relation='related', no description
                # 2. Dict old: {title, why, relation} → single relation
                # 3. Dict new: {title, relations: [{relation, why}, ...]} → multiple relations
                if isinstance(entry, dict):
                    title_query = entry.get('title', '')
                else:
                    title_query = str(entry)

                match = self.find_node_by_title(title_query, threshold=0.75)
                if not match or match.get('id') in created_set:
                    continue

                # Build list of (relation, description) pairs to create
                relation_pairs = []
                if isinstance(entry, dict) and 'relations' in entry:
                    # New format: multiple relations
                    for rel_spec in entry['relations']:
                        rel = rel_spec.get('relation', 'related')
                        desc = rel_spec.get('why', rel_spec.get('description', ''))
                        relation_pairs.append((rel, desc))
                elif isinstance(entry, dict):
                    # Old format: single relation
                    rel = entry.get('relation', 'related')
                    desc = entry.get('why', '')
                    relation_pairs.append((rel, desc))
                else:
                    # String format
                    relation_pairs.append(('related', ''))

                for node_id in created_ids:
                    for rel, desc in relation_pairs:
                        try:
                            self.connect_typed(node_id, match['id'], relation=rel,
                                              weight=0.6, description=desc)
                            connections_created += 1
                        except Exception as _e:
                            self._log_error('batch_connect_to', _e, 'connecting %s → %s' % (node_id[:8], match['id'][:8]))

        return {
            'nodes_created': len(created_ids),
            'results': results,
            'connections_created': connections_created,
            'connect_to_failures': connect_to_failures,
        }

    def revise_batch(self, revisions: List[Dict]) -> Dict[str, Any]:
        """Revise multiple nodes in one call. Each revision uses the same
        contract as revise() — per-field replace, immutable fields skipped
        with warning, deltas captured for trace history.

        Args:
            revisions: List of dicts, each with:
                - node_id (required): ID of node to revise
                - reason (required): why this revision
                - content, situation, reasoning, etc.: any revisable field

        Example:
            revise_batch(revisions=[
                {"node_id": "abc123", "content": "Judge now runs in daemon", "reason": "architecture changed"},
                {"node_id": "def456", "situation": "When debugging daemon connectivity", "reason": "adding situation"},
                {"node_id": "ghi789", "reasoning": "updated — encoder uses node catalog", "reason": "encoder v3.2"},
            ])

        Returns:
            {revised: count,
             results: [{node_id, status, error?, deltas?, warnings?}]}

            Per-result `deltas` and `warnings` mirror what revise() returns —
            callers (typically dispatch) use them to emit one trace event per
            revised node.
        """
        results = []
        revised_count = 0

        for spec in revisions:
            node_id = spec.get('node_id')
            if not node_id:
                results.append({'error': 'missing node_id', 'status': 'skipped'})
                continue

            reason = spec.get('reason', '')
            content = spec.get('content')

            # Extract all fields except node_id/reason/content for updates
            updates = {k: v for k, v in spec.items()
                       if k not in ('node_id', 'reason', 'content') and v is not None}

            try:
                result = self.revise(node_id=node_id, content=content,
                                     reason=reason, updates=updates)
                if result.get('error'):
                    results.append({'node_id': node_id, 'status': 'error', 'error': result['error']})
                else:
                    results.append({
                        'node_id': node_id,
                        'status': 'revised',
                        'deltas': result.get('deltas', []),
                        'warnings': result.get('warnings', []),
                    })
                    revised_count += 1
            except Exception as e:
                self._log_error('revise_batch', e, 'revising %s' % node_id[:8])
                results.append({'node_id': node_id, 'status': 'error', 'error': str(e)})

        return {
            'revised': revised_count,
            'results': results,
        }

    # validate_node removed 2026-04-13 — old node_metadata table dropped.

    def _generate_summary(self, title: str, content: Optional[str] = None) -> Optional[str]:
        """Generate a content_summary (max 200 chars) for tiered recall.

        Returns first sentence of content, or first 200 chars if no sentence boundary.
        Returns None if content is empty or very short (title suffices).
        """
        if not content or len(content) < 30:
            return None
        # First sentence
        period_idx = content.find('. ')
        if 0 < period_idx < 200:
            return content[:period_idx + 1]
        # First 200 chars with ellipsis
        if len(content) > 200:
            return content[:197] + '...'
        return content

    def _extract_keywords(self, text: str) -> str:
        """
        Extract keywords from text (numbers, proper nouns, technical terms, common words).

        Args:
            text: Text to extract from

        Returns:
            Space-separated keywords string
        """
        if not text:
            return ''

        # PHASE 1: Extract numbers and values before lowercasing
        number_patterns = re.findall(r'\$?\d+(?:\.\d+)?%?(?:px|ms|s|d|kb|mb|gb)?', text, re.IGNORECASE)
        number_keywords = [n.lower().replace(re.sub(r'[^a-z0-9%$.]', '', n), '') for n in number_patterns]
        number_keywords = [n for n in number_keywords if len(n) >= 1]

        # PHASE 2: Extract proper nouns and technical terms
        proper_nouns = re.findall(r'[A-Z][a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*', text)
        technical_terms = re.findall(r'[a-z]+[A-Z][a-zA-Z0-9]*', text)
        snake_terms = re.findall(r'[a-z][a-z0-9]*_[a-z0-9_]+', text)
        dotted_terms = re.findall(r'[a-z]+(?:\.[a-z]+)+', text)

        preserved_terms = set()
        for term in proper_nouns + technical_terms + snake_terms + dotted_terms:
            lower = term.lower()
            if len(lower) > 2 and lower not in TFIDF_STOP_WORDS:
                preserved_terms.add(lower)
                stripped = re.sub(r'[^a-z0-9]', '', lower)
                if len(stripped) > 2 and stripped != lower:
                    preserved_terms.add(stripped)

        # PHASE 3: Standard word extraction
        words = re.sub(r'[^a-z0-9\s\-\./]', ' ', text.lower()).split()
        words = [w for w in words if len(w) > 2 and w not in TFIDF_STOP_WORDS]

        # Also add variants
        variants = set()
        for w in words:
            variants.add(w)
            stripped = re.sub(r'[^a-z0-9]', '', w)
            if stripped != w and len(stripped) > 2:
                variants.add(stripped)

        all_keywords = list(preserved_terms | variants | set(number_keywords))
        return ' '.join(all_keywords[:50])  # Cap at 50 keywords

    def _bridge_at_store_time(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Detect bridge opportunities at store-time.
        Returns array of bridges created.
        """
        max_bridges = self.get_config('bridge_max_per_remember', 2)
        candidates = self._find_bridge_candidates(node_id, limit=max_bridges)
        created = []

        for c in candidates:
            bridge = self._create_bridge(node_id, c['targetId'], c.get('sharedTitles', ''))
            if bridge:
                created.append(bridge)

        return created

    def set_personal(self, node_id: str, personal: str,
                     personal_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Mark a node as personal information.

        Args:
            node_id: Node to mark
            personal: 'fixed' (permanent fact, auto-locks), 'fluid' (evolving truth,
                      10x slower decay), 'contextual' (depends on conditions), or
                      None to remove personal flag
            personal_context: For contextual nodes — when/where this applies
                              (e.g. "during technical sprints", "at work")

        Returns:
            Dict with node_id, personal, locked status
        """
        if personal and personal not in ('fixed', 'fluid', 'contextual'):
            return {'error': f'Invalid personal flag: {personal}. Use fixed/fluid/contextual/None.'}

        ts = self.now()

        # Fixed personal nodes are always locked
        if personal == 'fixed':
            self.conn.execute(
                'UPDATE nodes SET personal = ?, personal_context = ?, locked = 1, updated_at = ? WHERE id = ?',
                (personal, personal_context, ts, node_id)
            )
        else:
            self.conn.execute(
                'UPDATE nodes SET personal = ?, personal_context = ?, updated_at = ? WHERE id = ?',
                (personal, personal_context, ts, node_id)
            )
        self.conn.commit()

        # Fetch updated node
        cursor = self.conn.execute(
            'SELECT title, locked, personal, personal_context FROM nodes WHERE id = ?',
            (node_id,)
        )
        row = cursor.fetchone()
        if not row:
            return {'error': f'Node {node_id} not found'}

        return {
            'node_id': node_id,
            'title': row[0],
            'locked': row[1] == 1,
            'personal': row[2],
            'personal_context': row[3],
        }

    def get_personal_nodes(self, personal_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all personal nodes, optionally filtered by type.

        Args:
            personal_type: 'fixed', 'fluid', 'contextual', or None for all personal nodes

        Returns:
            List of personal node dicts
        """
        if personal_type:
            cursor = self.conn.execute(
                'SELECT id, type, title, content, personal, personal_context, locked FROM nodes WHERE personal = ? AND archived = 0 ORDER BY updated_at DESC',
                (personal_type,)
            )
        else:
            cursor = self.conn.execute(
                'SELECT id, type, title, content, personal, personal_context, locked FROM nodes WHERE personal IS NOT NULL AND archived = 0 ORDER BY updated_at DESC'
            )

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0], 'type': row[1], 'title': row[2],
                'content': row[3], 'personal': row[4],
                'personal_context': row[5], 'locked': row[6] == 1,
            })
        return results

    # ═══════════════════════════════════════════════════════════════
    # v6: Multi-vector enrichment (Embedding Migration to LLM)
    # The brain builds a structured prompt with neighbors.
    # Claude (or a local LLM) fills in Q/A/B/K.
    # Each is embedded and stored in node_enrichments.
    # ═══════════════════════════════════════════════════════════════

    def _build_enrichment_prompt(self, node_id: str, title: str,
                                  content: Optional[str] = None) -> Optional[str]:
        """Build the V5 structured enrichment prompt for a node.

        Finds neighbors via edges, formats them, and returns the prompt
        for Claude (or local LLM) to fill in.

        Returns None if node has no neighbors (nothing to anchor to).
        """
        try:
            graph_dal = GraphDAL(self.conn)
            neighbors = graph_dal.get_neighbors(
                node_id, limit=ENRICHMENT_NEIGHBOR_COUNT
            )
            if not neighbors:
                return None

            neighbor_lines = []
            for nb in neighbors:
                kw = nb.get('keywords', '') or ''
                kw_short = ', '.join(kw.split()[:5]) if kw else 'none'
                neighbor_lines.append(
                    f"- {nb['title'][:80]} ({nb['type']}, keywords: {kw_short})"
                )

            content_preview = (content or '')[:200]
            prompt = ENRICHMENT_PROMPT_TEMPLATE.format(
                neighbors='\n'.join(neighbor_lines),
                title=title,
                content=content_preview,
            )
            return prompt
        except Exception as e:
            print(f'[brain] _build_enrichment_prompt failed: {e}', file=sys.stderr)
            return None

    def store_enrichments(self, node_id: str, question: Optional[str] = None,
                          anchor: Optional[str] = None, bridge: Optional[str] = None,
                          keywords: Optional[str] = None) -> Dict[str, Any]:
        """Store enrichment vectors for a node (called after Claude fills in the prompt).

        Each non-None enrichment text is embedded and stored in node_enrichments.
        Returns count of enrichments stored and any errors.
        """
        vdal = self._vec_dal
        stored = 0
        errors = []

        enrichments = {
            'question': question,
            'anchor': anchor,
            'bridge': bridge,
            'keywords': keywords,
        }

        for vtype, text in enrichments.items():
            if not text or not text.strip():
                continue
            text = text.strip()
            try:
                blob = None
                if embedder.is_ready():
                    blob = embedder.embed_document(text)
                vdal.store(node_id, vtype, text, blob,
                           model=embedder.stats.get('model_name', 'unknown') if embedder.is_ready() else 'none')
                stored += 1
            except Exception as e:
                errors.append(f'{vtype}: {str(e)[:100]}')
                self._log_error('store_enrichment', e, '%s/%s' % (node_id[:8], vtype))

        return {
            'node_id': node_id,
            'enrichments_stored': stored,
            'errors': errors if errors else None,
        }

    def get_enrichment_coverage(self) -> Dict[str, Any]:
        """Get vector coverage stats."""
        try:
            return self._vec_dal.get_coverage_stats()
        except Exception as e:
            return {'error': str(e)}

    def find_node_by_title(self, title_query: str, threshold: float = 0.75,
                           top_k: int = 1) -> Optional[Dict[str, Any]]:
        """Find a node by fuzzy title matching using embedding similarity.

        Embeds the query, scans all node embeddings, returns the best match(es)
        above threshold with context (content snippet, keywords) so the caller
        can verify correctness.

        Args:
            title_query: Title to search for (fuzzy)
            threshold: Minimum similarity (0.0-1.0). Default 0.75 is conservative
                       to prevent false matches. Lower to 0.6 for broader search.
            top_k: Return top K matches (default 1 = best match only)

        Returns: {id, title, type, similarity, content_snippet, keywords} or None.
                 If top_k > 1, returns list of matches.
        """
        scored = {}  # id → result dict, dedup by node

        # Path 1: Text matching — fast SQL LIKE on title
        query_lower = title_query.lower()
        text_rows = self.conn.execute(
            "SELECT id, title, type, SUBSTR(content, 1, 200), keywords "
            "FROM nodes WHERE archived = 0 AND LOWER(title) LIKE ?",
            ("%" + query_lower.replace(" ", "%") + "%",)
        ).fetchall()
        for nid, title, ntype, snippet, keywords in text_rows:
            scored[nid] = {
                "id": nid, "title": title, "type": ntype,
                "similarity": 0.95,  # text match = high confidence
                "content_snippet": snippet or "",
                "keywords": keywords or "",
            }

        # Path 2: Embedding similarity — semantic fallback
        if embedder.is_ready() and len(scored) < top_k:
            query_vec = embedder.embed_query(title_query)
            if query_vec:
                rows = self.conn.execute(
                    "SELECT ne.node_id, ne.embedding, n.title, n.type, "
                    "SUBSTR(n.content, 1, 200) as snippet, n.keywords "
                    "FROM node_enrichments ne JOIN nodes n ON ne.node_id = n.id "
                    "WHERE ne.vector_type = '_primary' AND n.archived = 0"
                ).fetchall()
                for node_id, emb_blob, title, ntype, snippet, keywords in rows:
                    if not emb_blob or node_id in scored:
                        continue
                    sim = embedder.cosine_similarity(query_vec, emb_blob)
                    if sim >= threshold:
                        scored[node_id] = {
                            "id": node_id, "title": title, "type": ntype,
                            "similarity": round(sim, 3),
                            "content_snippet": snippet or "",
                            "keywords": keywords or "",
                        }

        results = sorted(scored.values(), key=lambda x: x["similarity"], reverse=True)

        if top_k == 1:
            return results[0] if results else None
        return results[:top_k]

    def encode_cluster(self, nodes: List[Dict], connect_to: Optional[List[str]] = None,
                       auto_connect: bool = True) -> Dict[str, Any]:
        """Compound encoding operation — store multiple nodes in one call.

        Each node dict: {type, title, content, keywords?, enrichment?: {question?, anchor?, bridge?, keywords?}}
        connect_to: list of existing node titles to fuzzy-match and connect to.
        auto_connect: if True, find related existing nodes automatically.

        Returns: {nodes_created, connections_created, suggested_connections, duplicates, missing}
        """
        created_ids = []
        connections_created = 0
        suggested = []
        duplicates = []
        missing = []

        # 1. Store all nodes
        for spec in nodes:
            ntype = spec.get("type", "concept")
            title = spec.get("title", "")
            content = spec.get("content", "")

            # Check for near-duplicate by title
            existing = self.find_node_by_title(title, threshold=0.92)
            if existing:
                duplicates.append({
                    "new_title": title,
                    "existing_title": existing["title"],
                    "existing_id": existing["id"],
                    "similarity": existing["similarity"]
                })

            result = self.remember(
                type=ntype, title=title, content=content,
                keywords=spec.get("keywords"),
                locked=spec.get("locked", False),
                confidence=spec.get("confidence", None),
                project=spec.get("project"),
            )
            node_id = result.get("id")
            if not node_id:
                continue
            created_ids.append(node_id)

            # Store inline enrichments if provided
            enrichment = spec.get("enrichment")
            if enrichment and isinstance(enrichment, dict):
                self.store_enrichments(
                    node_id=node_id,
                    question=enrichment.get("question"),
                    anchor=enrichment.get("anchor"),
                    bridge=enrichment.get("bridge"),
                    keywords=enrichment.get("keywords"),
                )
            else:
                missing.append("%s: no enrichment provided" % title[:50])

        # 2. Connect nodes within the cluster to each other
        for i, src_id in enumerate(created_ids):
            for dst_id in created_ids[i+1:]:
                self.connect(src_id, dst_id, relation="related_to", weight=0.5)
                connections_created += 1

        # 3. Fuzzy-match connect_to titles and create edges (exclude just-created nodes)
        connected_to = []
        if connect_to:
            created_set = set(created_ids)
            for title_query in connect_to:
                match = self.find_node_by_title(title_query, threshold=0.75)
                if match and match["id"] not in created_set:
                    for node_id in created_ids:
                        self.connect(node_id, match["id"], relation="related_to", weight=0.6)
                        connections_created += 1
                    connected_to.append({
                        "query": title_query, "matched": match["title"],
                        "id": match["id"], "similarity": match["similarity"]
                    })
                else:
                    missing.append("connect_to '%s': no match found (threshold 0.75)" % title_query[:50])

        # 4. Auto-connect: find similar existing nodes for each new node
        if auto_connect:
            created_set = set(created_ids)
            for node_id in created_ids:
                row = self.conn.execute(
                    "SELECT title FROM nodes WHERE id = ?", (node_id,)
                ).fetchone()
                if not row:
                    continue
                match = self.find_node_by_title(row[0], threshold=0.75)
                if match and match["id"] != node_id and match["id"] not in created_set:
                    suggested.append({
                        "from_id": node_id, "from_title": row[0],
                        "to_id": match["id"], "to_title": match["title"],
                        "similarity": match["similarity"]
                    })
                    self.connect(node_id, match["id"], relation="related_to", weight=0.4)
                    connections_created += 1

        self.conn.commit()

        # Mark all created nodes dirty — embed_queue worker picks them up
        # within ~5s. Single batch drain for the whole group.
        try:
            from . import embed_queue
            for nid in created_ids:
                embed_queue.enqueue(nid)
        except Exception as e:
            self._log_error('embed_enqueue_remember_batch', e,
                            'enqueue %d nodes' % len(created_ids))

        return {
            "nodes_created": len(created_ids),
            "node_ids": created_ids,
            "connections_created": connections_created,
            "connected_to": connected_to,
            "suggested_connections": suggested,
            "duplicates": duplicates,
            "missing": missing,
        }

    def enrich_keywords(self, node_id: str) -> Optional[str]:
        """
        Enrich keywords on a node from its content.
        Used by health check for frequently-missed nodes.
        """
        try:
            row = self.conn.execute(
                'SELECT content, keywords FROM nodes WHERE id = ?',
                (node_id,)
            ).fetchone()
            if not row or not row[0]:
                return None

            content, existing_kw = row
            new_kw = self._extract_keywords(content)
            combined = f'{existing_kw} {new_kw}' if existing_kw else new_kw

            self.conn.execute(
                'UPDATE nodes SET keywords = ?, updated_at = ? WHERE id = ?',
                (combined, self.now(), node_id)
            )
            return combined
        except Exception as e:
            self._log_error('enrich_keywords', e, 'enriching keywords for node %s' % node_id[:12])
            return None
