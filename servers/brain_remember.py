"""
brain — BrainRemember Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from .brain_constants import TYPE_CONFIDENCE
from typing import Any, Dict, List, Optional, Set
import json
import math
import re
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

    async def store_embedding(self, node_id: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Embed text via embedder and store as BLOB in node_embeddings table.
        Fire-and-forget async; non-critical if fails.

        Args:
            node_id: Node ID
            text: Text to embed

        Returns:
            {'node_id': str, 'embed_ms': int} or None on failure
        """
        if not embedder.is_ready():
            return None

        t0 = time.time()
        blob = embedder.embed(text)  # Already returns bytes
        if not blob:
            return None

        try:
            self.conn.execute(
                'INSERT OR REPLACE INTO node_embeddings (node_id, embedding, model, created_at) VALUES (?, ?, ?, ?)',
                (node_id, blob, embedder.stats['model_name'], self.now())
            )
            self.conn.commit()

            return {'node_id': node_id, 'embed_ms': int((time.time() - t0) * 1000)}
        except Exception as e:
            print(f'[brain] Failed to store embedding for {node_id}: {e}')
            return None

    def remember(self, type: str, title: str, content: Optional[str] = None,
                 keywords: Optional[str] = None, locked: bool = False,
                 connections: Optional[List[Dict[str, Any]]] = None,
                 emotion: float = 0, emotion_label: str = 'neutral',
                 emotion_source: str = 'auto', project: Optional[str] = None,
                 confidence: float = 1.0,
                 personal: Optional[str] = None,
                 personal_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Store a new memory node with semantic indexing and connections.

        Args:
            type: Node type (person, project, task, etc.)
            title: Node title
            content: Optional detailed content
            keywords: Optional keywords (auto-extracted if not provided)
            locked: If True, node never decays
            connections: List of {target_id, relation, weight} dicts
            emotion: Emotional intensity (0-1)
            emotion_label: 'positive', 'negative', 'neutral', 'frustration', etc.
            emotion_source: 'auto', 'user', 'system'
            project: Optional project ID to associate
            confidence: Confidence score (for future use)
            personal: v4 personal flag — 'fixed' (permanent fact), 'fluid' (evolving truth),
                      'contextual' (depends on conditions), or None (not personal)
            personal_context: v4 qualifier for contextual personal nodes — describes when/where
                              the personal info applies (e.g. "during technical sprints")

        Returns:
            Dict with id, type, title, emotion, emotion_label, bridges_created, personal
        """
        # Validate personal flag
        if personal and personal not in ('fixed', 'fluid', 'contextual'):
            personal = None
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
        self.conn.execute(
            '''INSERT INTO nodes
               (id, type, title, content, content_summary, keywords,
                activation, stability, locked, confidence,
                recency_score, emotion, emotion_label, emotion_source, project,
                personal, personal_context,
                last_accessed, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 1.0, 1.0, ?, ?, 1.0, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (node_id, type, title, content, content_summary, keywords,
             1 if locked else 0, confidence,
             emotion, emotion_label, emotion_source, project,
             personal, personal_context,
             ts, ts, ts)
        )
        self.conn.commit()

        # v5: Build TF-IDF vector for this node
        try:
            self._store_tfidf_vector(node_id, title, content, keywords)
        except Exception:
            pass  # Non-critical — recall still works via keywords

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

        # Create connections
        if connections:
            for conn in connections:
                target_id = conn.get('target_id')
                relation = conn.get('relation', 'related')
                weight = conn.get('weight', 0.5)
                if target_id:
                    self.connect(node_id, target_id, relation, weight)

        # v6: Auto-connect to conversation context
        # New nodes should be connected to recently accessed nodes — this is the
        # "conversation context" connection. Eliminates orphan nodes.
        try:
            recent = self.conn.execute('''
                SELECT id FROM nodes
                WHERE id != ? AND archived = 0
                  AND last_accessed > datetime('now', '-1 hour')
                  AND type NOT IN ('thought', 'intuition')
                ORDER BY last_accessed DESC LIMIT 3
            ''', (node_id,)).fetchall()
            for (recent_id,) in recent:
                # Only create if no edge already exists
                existing = self.conn.execute(
                    'SELECT 1 FROM edges WHERE source_id = ? AND target_id = ?',
                    (node_id, recent_id)
                ).fetchone()
                if not existing:
                    self.connect(node_id, recent_id, 'co_accessed', 0.2)
        except Exception:
            pass  # Non-critical

        # v11: Emergent bridging at store-time
        bridges = []
        try:
            bridges = self._bridge_at_store_time(node_id)
        except Exception:
            pass  # Non-critical — bridging failure should never block remember

        # v5: Track encoding for heartbeat
        try:
            self.record_remember()
        except Exception:
            pass

        # v5.1: Track node in current conversation segment
        try:
            self.add_to_segment(node_id)
        except Exception:
            pass

        return {
            'id': node_id,
            'type': type,
            'title': title,
            'emotion': emotion,
            'emotion_label': emotion_label,
            'bridges_created': len(bridges),
            'embedding_stored': embedding_stored,  # Phase 0.5C
            'personal': personal,  # v4
        }

    def recall_expand(self, node_id: str) -> Dict[str, Any]:
        """Return full content + metadata for a specific node (on-demand expansion).

        Used when tiered recall returned a summary and the caller needs the full content.
        """
        cur = self.conn.execute(
            'SELECT * FROM nodes WHERE id = ?', (node_id,)
        )
        row = cur.fetchone()
        if not row:
            return {'error': f'Node {node_id} not found'}
        cols = [d[0] for d in cur.description]
        node = dict(zip(cols, row))

        # Attach metadata if it exists
        meta_cur = self.conn.execute(
            'SELECT * FROM node_metadata WHERE node_id = ?', (node_id,)
        )
        meta_row = meta_cur.fetchone()
        if meta_row:
            meta_cols = [d[0] for d in meta_cur.description]
            node['_metadata'] = dict(zip(meta_cols, meta_row))

        return node

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

    def remember_rich(self, type: str, title: str, content: Optional[str] = None,
                      reasoning: Optional[str] = None,
                      alternatives: Optional[List[Dict[str, str]]] = None,
                      user_raw_quote: Optional[str] = None,
                      correction_of: Optional[str] = None,
                      correction_pattern: Optional[str] = None,
                      source_context: Optional[str] = None,
                      confidence_rationale: Optional[str] = None,
                      change_impacts: Optional[List[Dict[str, str]]] = None,
                      source_attribution: Optional[str] = None,
                      scope: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """Store a memory node with rich metadata (v5 cognitive encoding).

        Wraps remember() with additional metadata stored in node_metadata sidecar table.
        Use this instead of remember() when you have reasoning, alternatives, corrections,
        or other rich context to preserve.

        Args:
            type, title, content, **kwargs: Passed through to remember()
            reasoning: The reasoning chain that produced this conclusion
            alternatives: List of {option, rejected_because} dicts
            user_raw_quote: User's exact words (prevents encoding bias)
            correction_of: node_id this knowledge corrects
            correction_pattern: The underlying divergence pattern
            source_context: What prompted this knowledge (user correction, code reading, etc.)
            confidence_rationale: WHY the confidence level was set
            change_impacts: List of {if_modified, must_check, because} for engineering memory
            source_attribution: user_stated | claude_inferred | session_synthesis | correction | code_reading
            scope: system | module | file | function | cross-system | cross-file | cross-function
        """
        # v5.2: Auto-populate user_raw_quote from last user message if not provided.
        # The pre-response hook stores the user's message in brain_meta.
        # This ensures operator voice is captured structurally, not by discipline.
        #
        # IMPORTANT: Quotes must not float. An auto-captured quote gets anchored with
        # source_context that records what was being discussed (the node title acts as
        # Claude's interpretation; source_context bridges the two).
        if user_raw_quote is None:
            try:
                last_msg = self.get_config('last_user_message')
                if last_msg and len(last_msg) >= 10:
                    user_raw_quote = last_msg
                    # Auto-anchor: if no source_context provided, generate one
                    # that ties the quote to the node being created
                    if source_context is None:
                        source_context = (
                            'Auto-captured operator message during encoding of: '
                            '%s. Host understood this as: %s' % (
                                title[:80],
                                (content or title)[:150]
                            )
                        )
            except Exception:
                pass

        # Store the core node via remember()
        result = self.remember(type=type, title=title, content=content, **kwargs)
        node_id = result['id']

        # Set source_attribution and scope on the node
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
                f"UPDATE nodes SET {', '.join(updates)} WHERE id = ?", params
            )

        # Store rich metadata in sidecar table
        has_metadata = any([reasoning, alternatives, user_raw_quote, correction_of,
                           correction_pattern, source_context, confidence_rationale,
                           change_impacts])
        if has_metadata:
            self.conn.execute(
                '''INSERT OR REPLACE INTO node_metadata
                   (node_id, reasoning, alternatives, user_raw_quote, correction_of,
                    correction_pattern, source_context, confidence_rationale,
                    last_validated, validation_count, change_impacts, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)''',
                (node_id, reasoning,
                 json.dumps(alternatives) if alternatives else None,
                 user_raw_quote, correction_of, correction_pattern,
                 source_context, confidence_rationale,
                 self.now() if correction_of else None,  # corrections are self-validating
                 json.dumps(change_impacts) if change_impacts else None,
                 self.now())
            )

        # If this corrects another node, create edge and lower its confidence
        if correction_of:
            try:
                self.connect(node_id, correction_of, 'corrected_by', 0.8)
                self.conn.execute(
                    "UPDATE nodes SET confidence = MAX(0.2, COALESCE(confidence, 0.7) * 0.7) WHERE id = ?",
                    (correction_of,)
                )
            except Exception as _e:
                self._log_error("remember_rich", _e, "self.connect(node_id, correction_of, corrected_by")

        self.conn.commit()

        result['source_attribution'] = source_attribution
        result['scope'] = scope
        result['has_metadata'] = has_metadata
        return result

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

    def get_node_with_metadata(self, node_id: str) -> Dict[str, Any]:
        """Return full node data joined with metadata sidecar."""
        cur = self.conn.execute('SELECT * FROM nodes WHERE id = ?', (node_id,))
        row = cur.fetchone()
        if not row:
            return {'error': f'Node {node_id} not found'}
        cols = [d[0] for d in cur.description]
        node = dict(zip(cols, row))

        meta_cur = self.conn.execute(
            'SELECT * FROM node_metadata WHERE node_id = ?', (node_id,)
        )
        meta_row = meta_cur.fetchone()
        if meta_row:
            meta_cols = [d[0] for d in meta_cur.description]
            meta = dict(zip(meta_cols, meta_row))
            # Parse JSON fields
            for json_field in ('alternatives', 'change_impacts'):
                if meta.get(json_field):
                    try:
                        meta[json_field] = json.loads(meta[json_field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            node['_metadata'] = meta
        return node

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
        except:
            return None
