"""
brain — BrainRemember Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from .brain_constants import TYPE_CONFIDENCE
from .dal import EnrichmentDAL, GraphDAL
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
                 correction_of: Optional[str] = None,
                 correction_pattern: Optional[str] = None,
                 source_context: Optional[str] = None,
                 confidence_rationale: Optional[str] = None,
                 alternatives: Optional[List[Dict[str, str]]] = None,
                 change_impacts: Optional[List[Dict[str, str]]] = None,
                 source_attribution: Optional[str] = None,
                 scope: Optional[str] = None,
                 **extra_fields) -> Dict[str, Any]:
        """
        Store a new memory node with semantic indexing and connections.

        Accepts ALL contract fields. Core fields go to the nodes table,
        promoted fields go to node_metadata/node_embeddings, and any
        unknown fields are silently ignored (future-proof).

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

        # v5.2: Critical flag requires operator approval — don't set directly
        if critical:
            self._add_pending_critical(node_id, title)

        # v5: Build TF-IDF vector for this node
        try:
            self._store_tfidf_vector(node_id, title, content, keywords)
        except Exception as e:
            self._log_error('tfidf_vector_store', e, 'storing TF-IDF vector for node %s' % node_id[:12])

        # Phase 0.5C: Store dense embedding SYNCHRONOUSLY at encode time.
        # Every node must have a semantic vector from birth so it's immediately
        # findable via embedding search. ~50ms per node — acceptable for remember().
        embed_text = f'{title}{" " + content if content else ""}'
        embedding_stored = False

        if embedder.is_ready():
            try:
                blob = embedder.embed(embed_text)
                if blob:
                    self.conn.execute(
                        'INSERT OR REPLACE INTO node_embeddings (node_id, embedding, model, created_at) VALUES (?, ?, ?, ?)',
                        (node_id, blob, embedder.stats['model_name'], self.now())
                    )
                    self.conn.commit()
                    embedding_stored = True
            except Exception as e:
                print(f'[brain] Phase 0.5C: Embedding failed for {node_id}: {e}', file=sys.stderr)
                # Node still stored — just without embedding. Keyword fallback works.
        else:
            print(f'[brain] Phase 0.5C: Embedder not ready — node {node_id} stored WITHOUT embedding', file=sys.stderr)

        # Situation embedding — when this knowledge matters
        if situation and embedding_stored:
            try:
                from .dal import EmbeddingDAL
                sit_blob = embedder.embed(situation)
                if sit_blob:
                    emb_dal = EmbeddingDAL(self.conn)
                    emb_dal.store_situation(node_id, situation, sit_blob)
            except Exception as e:
                print(f'[brain] Situation embedding failed for {node_id}: {e}', file=sys.stderr)

        # ── Multi-vector group embeddings (z-indexed architecture) ──
        # Each node gets 2-4 vectors stored in node_enrichments.
        # Group 1 (title) always computed. Groups 3-4 only if metadata exists.
        # Group 2 (blend) is the primary embedding already stored above.
        # These vectors enable z-weighted top2-avg scoring in recall:
        # score = avg(top 2 of [weight * cosine(query, vec) for each group])
        # See pipeline_contract.EMBEDDING_GROUPS for weights and field mappings.
        if embedding_stored:
            try:
                self._compute_group_vectors(node_id, title, content, situation,
                                            reasoning=reasoning,
                                            user_raw_quote=user_raw_quote,
                                            anchor_raw_quote=anchor_raw_quote,
                                            correction_pattern=correction_pattern,
                                            source_context=source_context,
                                            **{k: v for k, v in (extra_fields or {}).items()
                                               if isinstance(v, str) and v.strip()})
            except Exception as e:
                print(f'[brain] Group vector embedding failed for {node_id}: {e}', file=sys.stderr)

        # Create connections
        if connections:
            for conn in connections:
                target_id = conn.get('target_id')
                relation = conn.get('relation', 'related')
                weight = conn.get('weight', 0.5)
                if target_id:
                    self.connect(node_id, target_id, relation, weight)

        # v6→v7: Auto-connect to conversation context (Machine 1)
        # Connect new node to top 3 most semantically similar recently-accessed nodes.
        # v6 connected to ALL recent nodes — created massive co_accessed noise.
        # v7 uses embedding similarity to pick only relevant connections.
        try:
            new_node_emb = None
            if embedding_stored:
                row = self.conn.execute(
                    'SELECT embedding FROM node_embeddings WHERE node_id = ?',
                    (node_id,)
                ).fetchone()
                if row:
                    new_node_emb = row[0]

            recent = self.conn.execute('''
                SELECT n.id, ne.embedding FROM nodes n
                LEFT JOIN node_embeddings ne ON ne.node_id = n.id
                WHERE n.id != ? AND n.archived = 0
                  AND n.last_accessed > datetime('now', '-1 hour')
                  AND n.type NOT IN ('thought', 'intuition')
                ORDER BY n.last_accessed DESC LIMIT 10
            ''', (node_id,)).fetchall()

            if new_node_emb and recent:
                # Rank by similarity, pick top 3
                scored = []
                for (recent_id, recent_emb) in recent:
                    if recent_emb:
                        sim = embedder.cosine_similarity(new_node_emb, recent_emb)
                        scored.append((recent_id, sim))
                    else:
                        scored.append((recent_id, 0.0))
                scored.sort(key=lambda x: x[1], reverse=True)
                for recent_id, sim in scored[:3]:
                    if sim > 0.3:  # Only connect if meaningfully similar
                        from .dal import GraphDAL
                        graph_dal = GraphDAL(self.conn)
                        if not graph_dal.edge_exists(node_id, recent_id):
                            self.connect(node_id, recent_id, 'co_accessed', max(0.2, sim * 0.5))
            elif recent:
                # Fallback if no embedding: connect to top 3 by recency (old behavior)
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

        # v8: Mark recent pending messages as resolved (encoding closes the loop)
        # Uses resolve_recent_pending() — the single entry point for ALL encoding paths
        resolved_count = 0
        try:
            resolved_count = self.resolve_recent_pending(reason='encoded')
        except Exception as e:
            self._log_error('remember_resolve_pending', e,
                            'Failed to resolve pending messages after encoding %s' % node_id[:8])

        # v9: Store promoted metadata fields in sidecar table
        try:
            self._store_node_metadata(
                node_id, reasoning=reasoning, user_raw_quote=user_raw_quote,
                anchor_raw_quote=anchor_raw_quote,
                correction_of=correction_of, correction_pattern=correction_pattern,
                source_context=source_context, confidence_rationale=confidence_rationale,
                alternatives=alternatives, change_impacts=change_impacts,
                source_attribution=source_attribution, scope=scope,
                **{k: v for k, v in extra_fields.items()
                   if k not in ('type', 'title', 'content', 'keywords', 'locked',
                                'connections', 'emotion', 'emotion_label', 'emotion_source',
                                'project', 'confidence', 'personal', 'personal_context',
                                'critical', 'encoding_source', 'situation', 'source_turn_id',
                                'evolution_status')})
        except Exception as e:
            self._log_error('remember_metadata', e, 'storing metadata for %s' % node_id[:8])

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
        """Update any field(s) on an existing node. Generic revision.

        Three ways to call:
          revise(node_id, content="new text", reason="why")  — append content (legacy)
          revise(node_id, updates={"confidence": 0.9, "keywords": "new kw"}, reason="why")
          revise(node_id, situation="When debugging", reason="adding situation")  — kwargs

        Content is special: it APPENDS with a revision divider (preserves history).
        All other fields REPLACE the existing value.

        After any revision: re-embeds, re-indexes TF-IDF, updates timestamps.
        """
        # Merge updates from all sources
        all_updates = dict(updates or {})
        all_updates.update(kwargs)
        if content:
            all_updates['content'] = content

        if not all_updates:
            return {'error': 'No updates provided', 'node_id': node_id}

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

        # Content is special — append with revision divider
        new_content = old_content
        if 'content' in all_updates:
            divider = "\n\n--- Revised %s: %s ---\n" % (ts[:10], reason)
            new_content = old_content + divider + all_updates.pop('content')

        # Build SQL UPDATE for all fields
        # Always update: content, content_summary, updated_at, revised_at
        set_parts = ['content = ?', 'content_summary = ?', 'updated_at = ?', 'revised_at = ?']
        params = [new_content, self._generate_summary(title, new_content), ts, ts]

        # Safe fields that can be updated via revise
        SAFE_FIELDS = {
            'title', 'type', 'keywords', 'locked', 'confidence', 'emotion',
            'emotion_label', 'project', 'personal', 'personal_context',
            'critical', 'evolution_status',
        }

        for field, value in all_updates.items():
            if field in SAFE_FIELDS:
                set_parts.append('%s = ?' % field)
                params.append(value)
                if field == 'title':
                    title = value  # track for re-embed

        params.append(node_id)
        self.conn.execute(
            'UPDATE nodes SET %s WHERE id = ?' % ', '.join(set_parts), params)
        self.conn.commit()

        # Handle situation embedding separately (lives in node_embeddings, not nodes)
        if 'situation' in all_updates:
            try:
                from . import embedder
                from .dal import EmbeddingDAL
                sit_text = all_updates['situation']
                sit_blob = embedder.embed(sit_text)
                if sit_blob:
                    EmbeddingDAL(self.conn).store_situation(node_id, sit_text, sit_blob)
            except Exception as e:
                self._log_error("revise_situation", e,
                                "Failed to update situation for %s" % node_id[:8])

        # Re-embed combined content for better retrieval
        # NOTE: UPDATE not INSERT OR REPLACE — preserve situation_embedding columns
        embedding_updated = False
        try:
            from . import embedder
            if embedder.is_ready():
                embed_text = '%s %s' % (title, new_content)
                blob = embedder.embed(embed_text)
                if blob:
                    self.conn.execute(
                        'UPDATE node_embeddings SET embedding=?, model=?, created_at=? WHERE node_id=?',
                        (blob, embedder.stats.get('model_name', ''), ts, node_id))
                    self.conn.commit()
                    embedding_updated = True
        except Exception as e:
            self._log_error("revise_embed", e, "Failed to re-embed node %s" % node_id[:8])

        # Re-compute group vectors (z-indexed multi-vector architecture)
        # Reads current metadata from KV store + situation from node_embeddings
        # so the vectors reflect the latest state after revision.
        if embedding_updated:
            try:
                sit_row = self.conn.execute(
                    'SELECT situation_text FROM node_embeddings WHERE node_id = ?',
                    (node_id,)).fetchone()
                current_situation = sit_row[0] if sit_row else all_updates.get('situation')
                self._compute_group_vectors(
                    node_id, title, new_content, situation=current_situation)
            except Exception as e:
                self._log_error("revise_group_vectors", e,
                                "Failed to re-compute group vectors for %s" % node_id[:8])

        # Re-index TF-IDF
        try:
            kw_row = self.conn.execute(
                'SELECT keywords FROM nodes WHERE id = ?', (node_id,)).fetchone()
            current_keywords = kw_row[0] if kw_row else None
            self._store_tfidf_vector(node_id, title, new_content, current_keywords)
        except Exception as e:
            self._log_error("revise_tfidf", e, "Failed to re-index TF-IDF for %s" % node_id[:8])

        # Auto-resolve consolidation pairs
        try:
            from .dal import LogsDAL
            LogsDAL(self.logs_conn).resolve_consolidation_for_node(node_id)
        except Exception as e:
            self._log_error("revise_consolidation_resolve", e,
                            "Failed to auto-resolve consolidation pair for %s" % node_id[:8])

        # Mark pending messages as resolved
        pending_resolved = 0
        try:
            pending_resolved = self.resolve_recent_pending(reason='encoded')
        except Exception as e:
            self._log_error('revise_resolve_pending', e,
                            'Failed to resolve pending messages after revising %s' % node_id[:8])

        # ── VERIFICATION: read-back to confirm writes landed ──
        verification_failures = []

        # Verify nodes table fields
        from .dal import NodeDAL
        readback = NodeDAL(self.conn).get_node(node_id)
        if readback:
            for field in list(all_updates.keys()):
                if field in readback:
                    # Content is appended, not replaced — check it contains the new text
                    if field == 'content' and content:
                        if content not in (readback.get('content') or ''):
                            verification_failures.append(field)
                    else:
                        expected = all_updates[field]
                        actual = readback.get(field)
                        if actual != expected and str(actual) != str(expected):
                            verification_failures.append(field)

        # Verify situation embedding (stored in node_embeddings, not nodes)
        if 'situation' in all_updates:
            from .dal import EmbeddingDAL
            sit_text = EmbeddingDAL(self.conn).get_situation_text(node_id)
            if not sit_text:
                verification_failures.append('situation')

        verified = len(verification_failures) == 0

        return {
            'id': node_id,
            'type': all_updates.get('type', node_type),
            'title': title,
            'revised_at': ts,
            'content_length': len(new_content),
            'embedding_updated': embedding_updated,
            'fields_updated': list(all_updates.keys()),
            'verified': verified,
            'verification_failures': verification_failures if not verified else [],
            'pending_resolved': pending_resolved,
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
        """Propose a node as critical. Does NOT set the flag — requires approve_critical().

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

    def approve_critical(self, node_id: str) -> Dict[str, Any]:
        """Approve a node as critical — sets critical=1. Requires explicit operator action.

        Args:
            node_id: The node to approve as critical

        Returns:
            Dict with node_id, critical=1, approved_at
        """
        # Set the flag
        ts = self.now()
        self.conn.execute('UPDATE nodes SET critical = 1, updated_at = ? WHERE id = ?', (ts, node_id))
        self.conn.commit()

        # Remove from pending list
        try:
            import json as _json
            pending_json = self.get_config('pending_critical_approvals', '[]')
            pending = _json.loads(pending_json) if pending_json else []
            pending = [p for p in pending if p.get('node_id') != node_id]
            self.set_config('pending_critical_approvals', _json.dumps(pending))
        except Exception as e:
            self._log_error('approve_critical', e, 'removing from pending list')

        return {'node_id': node_id, 'critical': 1, 'approved_at': ts}

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

        Architecture: 4 groups defined in pipeline_contract.EMBEDDING_GROUPS.
        - title: always computed (diagnostic pointer)
        - blend: already stored in node_embeddings (skip here)
        - high_meta: situation + quotes — only if fields exist
        - other_meta: reasoning + correction_pattern + emergent — only if fields exist

        Vectors stored in node_enrichments with vector_type matching the group name.
        At recall time, recall scoring reads these and applies z-weighted top2-avg.
        See: brain_recall.py Step 3.5 for how these are scored.
        """
        from . import embedder
        from .pipeline_contract import (EMBEDDING_GROUPS, EMBEDDING_SKIP_FIELDS,
                                        EMBEDDING_FIELD_CHAR_LIMIT)

        if not embedder.is_ready():
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
            # Skip blend — it's the primary embedding, already stored in node_embeddings
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
            blob = embedder.embed(embed_text)
            if blob:
                self.conn.execute(
                    '''INSERT OR REPLACE INTO node_enrichments
                       (id, node_id, vector_type, text, embedding, model, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (f'{node_id}_{vector_type}', node_id, vector_type,
                     embed_text[:500], blob, embedder.stats.get('model_name', ''),
                     self.now()))

        self.conn.commit()

    def _store_node_metadata(self, node_id: str,
                             reasoning: Optional[str] = None,
                             user_raw_quote: Optional[str] = None,
                             anchor_raw_quote: Optional[str] = None,
                             correction_of: Optional[str] = None,
                             correction_pattern: Optional[str] = None,
                             source_context: Optional[str] = None,
                             confidence_rationale: Optional[str] = None,
                             alternatives: Optional[List[Dict[str, str]]] = None,
                             change_impacts: Optional[List[Dict[str, str]]] = None,
                             source_attribution: Optional[str] = None,
                             scope: Optional[str] = None,
                             **extra_metadata):
        """Store promoted metadata fields for a node.

        Core fields live in the nodes table. Metadata fields live in node_metadata_kv
        (key-value store). Extra kwargs are stored as emergent metadata keys.

        Called by remember() after node creation.
        """
        # Update source_attribution and scope on nodes table (direct columns)
        updates = []
        params = []
        if source_attribution:
            updates.append('source_attribution = ?')
            params.append(source_attribution)
        if scope:
            updates.append('scope = ?')
            params.append(scope)
        if updates:
            params.append(node_id)
            self.conn.execute(
                "UPDATE nodes SET %s WHERE id = ?" % ', '.join(updates), params)

        # Build metadata dict from all non-None kwargs
        from .dal_metadata import MetadataDAL
        meta = {}
        if reasoning: meta['reasoning'] = reasoning
        if user_raw_quote: meta['user_raw_quote'] = user_raw_quote
        if anchor_raw_quote: meta['anchor_raw_quote'] = anchor_raw_quote
        if correction_of: meta['correction_of'] = correction_of
        if correction_pattern: meta['correction_pattern'] = correction_pattern
        if source_context: meta['source_context'] = source_context
        if confidence_rationale: meta['confidence_rationale'] = confidence_rationale
        if alternatives: meta['alternatives'] = json.dumps(alternatives)
        if change_impacts: meta['change_impacts'] = json.dumps(change_impacts)
        # Emergent metadata — any extra kwargs flow through
        for k, v in extra_metadata.items():
            if v is not None and str(v).strip():
                meta[k] = str(v)

        if meta:
            dal = MetadataDAL(self.conn)
            dal.set_many(node_id, meta)

        # If this corrects another node, create edge and lower its confidence
        if correction_of:
            try:
                self.connect(node_id, correction_of, 'corrected_by', 0.8)
                self.conn.execute(
                    "UPDATE nodes SET confidence = MAX(0.2, COALESCE(confidence, 0.7) * 0.7) WHERE id = ?",
                    (correction_of,))
            except Exception as e:
                self._log_error('metadata_correction_link', e,
                                'linking correction %s → %s' % (node_id[:8], correction_of[:8]))

        self.conn.commit()

    def remember_rich(self, type: str, title: str, content: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """Backward-compatible wrapper — remember() now handles all fields directly."""
        return self.remember(type=type, title=title, content=content, **kwargs)

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

        for spec in nodes:
            result = self.remember(**spec)
            results.append(result)
            if result.get('id'):
                created_ids.append(result['id'])

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
                # Accept both old format (string) and new format (dict with title + why)
                if isinstance(entry, dict):
                    title_query = entry.get('title', '')
                    description = entry.get('why', '')
                else:
                    title_query = str(entry)
                    description = ''
                match = self.find_node_by_title(title_query, threshold=0.75)
                if match and match.get('id') not in created_set:
                    for node_id in created_ids:
                        try:
                            self.connect_typed(node_id, match['id'], relation='related',
                                              weight=0.6, description=description)
                            connections_created += 1
                        except Exception as _e:
                            self._log_error('batch_connect_to', _e, 'connecting %s → %s' % (node_id[:8], match['id'][:8]))

        return {
            'nodes_created': len(created_ids),
            'results': results,
            'connections_created': connections_created,
        }

    def validate_node(self, node_id: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Mark a node as validated — its knowledge has been confirmed as still accurate.

        Updates last_validated timestamp and increments validation_count.
        Resets any age-based confidence decay.
        """
        ts = self.now()
        # Upsert into node_metadata
        existing = self.conn.execute(
            'SELECT node_id FROM node_metadata WHERE node_id = ?', (node_id,)
        ).fetchone()
        if existing:
            self.conn.execute(
                '''UPDATE node_metadata
                   SET last_validated = ?, validation_count = validation_count + 1
                   WHERE node_id = ?''',
                (ts, node_id)
            )
        else:
            self.conn.execute(
                '''INSERT INTO node_metadata (node_id, last_validated, validation_count, created_at)
                   VALUES (?, ?, 1, ?)''',
                (node_id, ts, ts)
            )
        # Boost confidence slightly
        self.conn.execute(
            "UPDATE nodes SET confidence = MIN(1.0, COALESCE(confidence, 0.7) + 0.05) WHERE id = ?",
            (node_id,)
        )
        self.conn.commit()
        return {'node_id': node_id, 'last_validated': ts, 'context': context}

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
            neighbors = graph_dal.get_neighbors_with_context(
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
        enrichment_dal = EnrichmentDAL(self.conn)
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
                    blob = embedder.embed(text)
                enrichment_dal.store(node_id, vtype, text, blob,
                                    model=embedder.stats.get('model_name', 'unknown') if embedder.is_ready() else 'none')
                stored += 1
            except Exception as e:
                errors.append(f'{vtype}: {str(e)[:100]}')
                print(f'[brain] Enrichment embed failed for {node_id}/{vtype}: {e}', file=sys.stderr)

        return {
            'node_id': node_id,
            'enrichments_stored': stored,
            'errors': errors if errors else None,
        }

    def get_enrichment_coverage(self) -> Dict[str, Any]:
        """Get enrichment coverage stats."""
        try:
            enrichment_dal = EnrichmentDAL(self.conn)
            return enrichment_dal.get_coverage_stats()
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
            query_vec = embedder.embed(title_query)
            if query_vec:
                rows = self.conn.execute(
                    "SELECT ne.node_id, ne.embedding, n.title, n.type, "
                    "SUBSTR(n.content, 1, 200) as snippet, n.keywords "
                    "FROM node_embeddings ne JOIN nodes n ON ne.node_id = n.id "
                    "WHERE n.archived = 0"
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
