"""
brain — BrainRecall Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import math
import re
import struct
import sys
import time
from .brain_constants import (
    CRITICAL_BOOST,
    CRITICAL_SIMILARITY_THRESHOLD,
    DECAY_HALF_LIFE,
    EDGE_TYPES,
    EMBEDDING_PRIMARY_WEIGHT,
    GRAPH_AUGMENT_TOP_N,
    GRAPH_NEIGHBOR_LIMIT,
    INTENTIONAL_EDGE_TYPES,
    INTENT_PATTERNS,
    INTENT_TYPE_BOOSTS,
    KEYWORD_FALLBACK_WEIGHT,
    LEARNING_RATE,
    MAX_HOPS,
    MAX_NEIGHBORS,
    MAX_PAGE_SIZE,
    MAX_WEIGHT,
    NEIGHBOR_DAMPEN,
    PRUNE_THRESHOLD,
    RELEVANCE_FLOOR_ENRICHED,
    RELEVANCE_FLOOR_PRIMARY,
    SPREAD_DECAY,
    STABILITY_BOOST,
    STABILITY_FLOOR_ACCESS_THRESHOLD,
    STABILITY_FLOOR_RETENTION,
    TEMPORAL_PATTERNS,
    TFIDF_KEYWORD_WEIGHT,
    TFIDF_SEMANTIC_WEIGHT,
    TFIDF_STOP_WORDS,
    VOCAB_EXPANSION_MAX,
    TRAVERSE_DEPTH,
    TRAVERSE_DAMPEN,
    TRAVERSE_LIMITS,
    TRAVERSE_SEMANTIC_BONUS,
    TRAVERSE_SEMANTIC_THRESHOLD,
    TRAVERSE_CONVERGENCE_BOOST,
    FRESHNESS_MULTIPLIERS,
    EXCLUDED_EDGE_TYPES,
)
from .dal import GraphDAL


class BrainRecallMixin:
    """Recall methods for Brain."""

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node by ID with full content and connections."""
        row = self.conn.execute(
            "SELECT id, type, title, content, keywords, confidence, locked, "
            "created_at, updated_at, revised_at, access_count, emotion, "
            "emotion_label, project, encoding_source, content_summary "
            "FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if not row:
            return None
        node = {
            "id": row[0], "type": row[1], "title": row[2], "content": row[3],
            "keywords": row[4], "confidence": row[5], "locked": bool(row[6]),
            "created_at": row[7], "updated_at": row[8], "revised_at": row[9],
            "access_count": row[10], "emotion": row[11],
            "emotion_label": row[12] or "neutral", "project": row[13],
            "encoding_source": row[14], "content_summary": row[15],
        }
        # Attach connections
        edges = self.conn.execute(
            "SELECT e.target_id, e.relation, e.weight, n.title, n.type "
            "FROM edges e LEFT JOIN nodes n ON n.id = e.target_id "
            "WHERE e.source_id = ? ORDER BY e.weight DESC LIMIT 10",
            (node_id,)
        ).fetchall()
        node["connections"] = [
            {"target_id": e[0], "relation": e[1], "weight": e[2],
             "title": e[3] or "", "type": e[4] or ""}
            for e in edges
        ]
        return node

    def semantic_recall(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Pure embedding-based search (brute-force cosine scan).
        Embed query, compute cosine similarity against all stored embeddings.

        Args:
            query: Query text
            limit: Max results

        Returns:
            List of {'id': str, 'similarity': float} dicts, sorted by similarity
        """
        if not embedder.is_ready():
            return []

        t0 = time.time()
        query_vec = embedder.embed(query)
        if not query_vec:
            return []

        # Load all embeddings (excluding archived nodes)
        cursor = self.conn.execute(
            'SELECT ne.node_id, ne.embedding FROM node_embeddings ne JOIN nodes n ON n.id = ne.node_id WHERE n.archived = 0'
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        # Score every node
        scored = []
        for node_id, blob in rows:
            if not blob:
                continue
            similarity = embedder.cosine_similarity(query_vec, blob)
            scored.append({'id': node_id, 'similarity': similarity})

        # Sort and take top-k
        scored.sort(key=lambda x: x['similarity'], reverse=True)
        return scored[:limit]

    def backfill_embeddings(self, batch_size: int = 20) -> int:
        """
        Backfill embeddings for nodes missing them.
        Runs during consolidation; picks recently-accessed nodes first.

        Args:
            batch_size: Max nodes to embed in this batch

        Returns:
            Number of embeddings stored
        """
        if not embedder.is_ready():
            return 0

        # Find up to batch_size nodes without embeddings (order by last_accessed DESC)
        cursor = self.conn.execute(
            '''SELECT n.id, n.title, n.content FROM nodes n
               LEFT JOIN node_embeddings ne ON ne.node_id = n.id
               WHERE ne.node_id IS NULL AND n.archived = 0
               ORDER BY n.last_accessed DESC
               LIMIT ?''',
            (batch_size,)
        )
        nodes = cursor.fetchall()

        if not nodes:
            return 0

        # Build embed texts: title + content (same as store_embedding)
        texts = [f'{title}{(" " + content) if content else ""}' for _, title, content in nodes]

        # Batch embed
        embeddings = embedder.embed_batch(texts)
        stored = 0

        for i, (node_id, _, _) in enumerate(nodes):
            if i >= len(embeddings) or not embeddings[i]:
                continue  # Skip failed individual embeds

            blob = embeddings[i]  # Already bytes from embed_batch
            try:
                self.conn.execute(
                    'INSERT OR REPLACE INTO node_embeddings (node_id, embedding, model, created_at) VALUES (?, ?, ?, ?)',
                    (node_id, blob, embedder.stats['model_name'], self.now())
                )
                stored += 1
            except Exception:
                pass  # Skip failed; will retry next cycle

        self.conn.commit()
        return stored

    def _expand_query_with_vocabulary(self, query: str) -> str:
        """Expand query terms using vocabulary mappings (Step 0.5 in recall pipeline).

        Resolves operator vocabulary to find additional search terms.
        Example: "working copy" → "working copy worktree git worktree"
        """
        try:
            # Check individual terms and bigrams
            words = query.lower().split()
            expansion_terms = []

            # Check bigrams first (more specific)
            checked_bigrams = set()
            for i in range(len(words) - 1):
                bigram = f'{words[i]} {words[i+1]}'
                checked_bigrams.add(bigram)
                result = self.resolve_vocabulary(bigram)
                if result and not result.get('ambiguous'):
                    # Parse maps_to from content: '"term" → X, Y'
                    content = result.get('content', '')
                    if '\u2192' in content:
                        mapped = content.split('\u2192', 1)[1].strip()
                        for t in mapped.split(','):
                            t = t.strip()
                            if t and t.lower() not in query.lower():
                                expansion_terms.append(t)

            # Check individual words (skip if part of a matched bigram)
            for word in words:
                if len(word) <= 2:
                    continue
                # Skip if this word was part of a matched bigram
                if any(word in bg for bg in checked_bigrams if self.resolve_vocabulary(bg)):
                    continue
                result = self.resolve_vocabulary(word)
                if result and not result.get('ambiguous'):
                    content = result.get('content', '')
                    if '\u2192' in content:
                        mapped = content.split('\u2192', 1)[1].strip()
                        for t in mapped.split(','):
                            t = t.strip()
                            if t and t.lower() not in query.lower():
                                expansion_terms.append(t)

            # Cap expansion
            expansion_terms = expansion_terms[:VOCAB_EXPANSION_MAX]

            if expansion_terms:
                # Log expansion via recall_log (use returned_ids for expansion info)
                try:
                    session_id = getattr(self, 'session_id', 'unknown')
                    self.logs_conn.execute(
                        "INSERT INTO recall_log (session_id, query, returned_ids, returned_count, created_at) VALUES (?, ?, ?, ?, ?)",
                        (session_id, query, 'vocab_expansion: ' + ', '.join(expansion_terms), len(expansion_terms), self.now())
                    )
                    self.logs_conn.commit()
                except Exception:
                    pass  # Non-critical logging

                return query + ' ' + ' '.join(expansion_terms)

            return query
        except Exception:
            return query  # Never break recall due to vocab expansion failure

    def _keyword_recall(self, query: str, types: Optional[List[str]] = None, limit: int = 20,
                        offset: int = 0, include_archived: bool = False, min_recency: float = 0,
                        project: Optional[str] = None, session_id: Optional[str] = None,
                        _skip_log: bool = False) -> Dict[str, Any]:
        """INTERNAL: TF-IDF keyword recall. Used by recall() for keyword blending.
        Do NOT call directly — use recall() (embeddings + graph traversal) instead.
        Retrieve relevant nodes with TF-IDF scoring, spreading activation, and decay.

        Args:
            query: Search query
            types: Filter by node types
            limit: Max results to return
            offset: Pagination offset
            include_archived: Include archived nodes
            min_recency: Minimum recency score threshold
            project: Filter to specific project
            session_id: Optional session ID for logging

        Returns:
            Dict with results (list of nodes), _recall_log_id, intent
        """
        limit = min(limit, MAX_PAGE_SIZE)

        # v5.2 Step 0.5: Vocabulary expansion
        expanded_query = self._expand_query_with_vocabulary(query)

        # v5 Step 0: Intent detection
        intent_data = self._classify_intent(query)
        intent = intent_data['intent']
        type_boosts = intent_data['typeBoosts']
        temporal_filter = intent_data['temporalFilter']

        # Step 1: Keyword search for seeds
        seeds = self._search_keywords(expanded_query, 30)

        all_seeds = {}
        for seed in seeds:
            if seed['id'] not in all_seeds:
                all_seeds[seed['id']] = seed

        # v5: Also find seeds via TF-IDF
        tfidf_query_terms = self._tfidf_tokenize(query)
        if tfidf_query_terms:
            try:
                unique_terms = list(set(tfidf_query_terms))
                placeholders = ','.join('?' * len(unique_terms))
                cursor = self.conn.execute(
                    f'''SELECT DISTINCT nv.node_id FROM node_vectors nv
                        JOIN nodes n ON n.id = nv.node_id
                        WHERE nv.term IN ({placeholders}) AND n.archived = 0
                        LIMIT 50''',
                    unique_terms
                )
                for row in cursor.fetchall():
                    nid = row[0]
                    if nid not in all_seeds:
                        # Fetch node data
                        cursor2 = self.conn.execute(
                            '''SELECT id, type, title, content, keywords, activation, stability,
                                      access_count, locked, archived, last_accessed, created_at, critical
                               FROM nodes WHERE id = ?''',
                            (nid,)
                        )
                        row2 = cursor2.fetchone()
                        if row2:
                            all_seeds[nid] = {
                                'id': row2[0], 'type': row2[1], 'title': row2[2],
                                'content': row2[3], 'keywords': row2[4],
                                'activation': row2[5], 'stability': row2[6],
                                'access_count': row2[7], 'locked': row2[8] == 1,
                                'archived': row2[9] == 1, 'last_accessed': row2[10],
                                'created_at': row2[11],
                                'critical': row2[12] == 1 if len(row2) > 12 else False
                            }
            except Exception as _e:
                self._log_error("recall", _e, "fetching seed node details from database")

        if not all_seeds:
            # Return recent nodes if no seeds found
            return {
                'results': self._get_recent(limit, types),
                '_recall_log_id': None,
                'intent': intent
            }

        # Step 1b: Compute direct keyword match strength per seed
        query_terms = [w.replace('[^a-z0-9]', '', ) for w in query.lower().split()
                       if len(w) > 2]
        query_terms = [w for w in query_terms if w]

        direct_match_scores = {}
        for seed_id, seed in all_seeds.items():
            kw = (seed.get('keywords') or '').lower()
            title = (seed.get('title') or '').lower()
            content = (seed.get('content') or '').lower()
            match_count = 0
            for term in query_terms:
                if term in kw or term in title or term in content:
                    match_count += 1
            direct_match_scores[seed_id] = (match_count / len(query_terms)) if query_terms else 0

        # Step 2: Spreading activation
        activated = self.spread_activation(list(all_seeds.keys()), types)

        # v5: Compute batch TF-IDF scores
        activated_ids = [n['id'] for n in activated]
        tfidf_scores = self._batch_tfidf_scores(tfidf_query_terms, activated_ids)

        # Step 3: Compute combined score with TF-IDF + keyword + intent boosts
        max_spread = max([n.get('spread_activation', 0.001) for n in activated] or [0.001])

        now_ms = time.time() * 1000  # milliseconds

        scored = []
        for node in activated:
            # Keyword-based relevance
            keyword_relevance = node.get('spread_activation', 0) / max_spread

            direct_match = direct_match_scores.get(node['id'], 0)
            if direct_match == 0 and query_terms:
                nkw = (node.get('keywords') or '').lower()
                ntitle = (node.get('title') or '').lower()
                ncontent = (node.get('content') or '').lower()
                mc = 0
                for term in query_terms:
                    if term in nkw or term in ntitle or term in ncontent:
                        mc += 1
                direct_match = mc / len(query_terms)

            if direct_match > 0:
                keyword_relevance = min(1.0, keyword_relevance + direct_match * 0.5)

            # v5: TF-IDF semantic relevance
            semantic_score = tfidf_scores.get(node['id'], 0)

            # v5: Blend keyword and semantic scores (tunable blend)
            blend = self._get_tunable('embedding_blend', {
                'embedding': EMBEDDING_PRIMARY_WEIGHT, 'keyword': KEYWORD_FALLBACK_WEIGHT
            })
            emb_w = blend.get('embedding', EMBEDDING_PRIMARY_WEIGHT) if isinstance(blend, dict) else EMBEDDING_PRIMARY_WEIGHT
            kw_w = blend.get('keyword', KEYWORD_FALLBACK_WEIGHT) if isinstance(blend, dict) else KEYWORD_FALLBACK_WEIGHT
            relevance = (kw_w * keyword_relevance + emb_w * semantic_score)

            # v4: Hub dampening (nodes with 20+ connections get reduced relevance)
            cursor = self.conn.execute(
                'SELECT COUNT(*) FROM edges WHERE source_id = ?',
                (node['id'],)
            )
            edge_count = cursor.fetchone()[0]
            hub = self._get_tunable('hub_dampening', {'threshold': 40, 'penalty': 0.5})
            hub_threshold = hub.get('threshold', 40) if isinstance(hub, dict) else 40
            if edge_count > hub_threshold:
                relevance *= hub_threshold / edge_count

            # v4: Type dampening
            if node.get('type') in ('project', 'person'):
                relevance *= 0.5

            # v5: Intent-based type boosting
            type_boost = type_boosts.get(node.get('type'), 1.0)
            relevance *= type_boost

            # v5.2: Critical node boost
            if node.get('critical'):
                relevance *= CRITICAL_BOOST

            recency = self._recency_score(node.get('last_accessed'))
            frequency = self._frequency_score(node.get('access_count', 0))
            emotion_intensity = abs(node.get('emotion', 0))

            # v6: Recency boost — recent nodes get relevance multiplier
            # This ensures recent work is findable even if embedding match is imperfect
            created = node.get('created_at') or node.get('last_accessed')
            if created:
                try:
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    hours_since = (datetime.utcnow() - created_dt.replace(tzinfo=None)).total_seconds() / 3600
                    if hours_since <= 48:
                        relevance *= 1.5  # Strong boost for last 2 days
                    elif hours_since <= 168:
                        relevance *= 1.2  # Moderate boost for last week
                except Exception:
                    pass

            # ─── Ebbinghaus retention with time-dilation ───
            retention = 1.0
            # v4: Personal flag overrides decay
            # - fixed: never decays (permanent facts — birthday, family, role)
            # - fluid: 10x slower decay (evolving truths — current interests, active projects)
            # - contextual: normal decay but carries qualifier for context matching
            node_personal = node.get('personal')
            if node_personal == 'fixed':
                retention = 1.0  # Skip all decay — same as locked
            elif not node.get('locked'):
                half_lives = self._get_tunable('decay_half_lives', DECAY_HALF_LIFE)
                half_life = half_lives.get(node.get('type'), 168)
                # v5.2: Comprehensive guard against inf bug variants
                # auto_heal can corrupt half_lives via JSON round-trip:
                #   float('inf') → str("inf") → breaks decay computation
                if isinstance(half_life, str):
                    if half_life.lower() in ('inf', 'infinity', '__inf__'):
                        half_life = float('inf')
                    else:
                        try:
                            half_life = float(half_life)
                        except (ValueError, TypeError):
                            half_life = 168  # fallback to default
                if isinstance(half_life, (int, float)) and not isinstance(half_life, bool):
                    if half_life >= 999999:  # sentinel for infinity
                        half_life = float('inf')
                    elif half_life != half_life:  # NaN check
                        half_life = 168
                # v4: Fluid personal nodes decay 10x slower
                if node_personal == 'fluid':
                    half_life = half_life * 10 if half_life != float('inf') else half_life
                # v4: Evolution-informed decay protection.
                # Nodes connected to active tensions/hypotheses/aspirations get 3x slower decay.
                # They're part of an active investigation and shouldn't fade while it's ongoing.
                if half_life != float('inf'):
                    try:
                        evo_conn = self.conn.execute(
                            """SELECT COUNT(*) FROM edges e
                               JOIN nodes n ON (e.source_id = n.id OR e.target_id = n.id)
                               WHERE (e.source_id = ? OR e.target_id = ?)
                                 AND n.type IN ('tension','hypothesis','aspiration')
                                 AND n.evolution_status = 'active' AND n.archived = 0""",
                            (node['id'], node['id'])
                        ).fetchone()[0]
                        if evo_conn > 0:
                            half_life = half_life * 3
                    except Exception as _e:
                        self._log_error("recall", _e, "checking evolution connections for half-life adjustment")
                if half_life != float('inf'):
                    last_accessed_str = node.get('last_accessed')
                    if last_accessed_str:
                        try:
                            last_accessed_dt = datetime.fromisoformat(last_accessed_str.replace('Z', '+00:00'))
                            last_accessed_ms = last_accessed_dt.timestamp() * 1000
                        except Exception:
                            last_accessed_ms = now_ms

                        wall_clock_hours = (now_ms - last_accessed_ms) / (1000 * 60 * 60)

                        # Read time-dilation rates from brain config
                        active_rate = self.get_config('decay_active_rate', 1.0)
                        idle_rate = self.get_config('decay_idle_rate', 0.1)

                        # Calculate session vs idle hours since access
                        total_session_minutes = float(self.get_config('total_session_minutes', 0) or 0)
                        last_session_minutes = float(self.get_config('_last_session_minutes_at_access', 0) or 0)
                        session_hours_since_access = max(0, (total_session_minutes - last_session_minutes) / 60)
                        idle_hours = max(0, wall_clock_hours - session_hours_since_access)

                        # Dilated time
                        dilated_hours = (session_hours_since_access * active_rate +
                                        idle_hours * idle_rate)

                        effective_s = node.get('stability', 1.0) * half_life
                        retention = math.exp(-dilated_hours / effective_s) if effective_s > 0 else 1.0

                        # Stability floor
                        if (node.get('access_count', 0) >= STABILITY_FLOOR_ACCESS_THRESHOLD and
                            retention < STABILITY_FLOOR_RETENTION):
                            retention = STABILITY_FLOOR_RETENTION

            if emotion_intensity > 0.5:
                retention = min(1.0, retention * (1 + emotion_intensity * 0.5))

            combined = self._combined_score(relevance, recency, frequency,
                                          emotion_intensity, node.get('locked', False))
            effective = combined * retention

            scored.append({
                **node,
                'recency_score': recency,
                'frequency_score': frequency,
                'relevance_score': relevance,
                'semantic_score': semantic_score,
                'keyword_relevance': keyword_relevance,
                'emotion_intensity': emotion_intensity,
                'retention': retention,
                'effective_activation': effective
            })

        # Step 4: Filter
        filtered = scored
        if not include_archived:
            filtered = [n for n in filtered if not n.get('archived')]
        if types:
            filtered = [n for n in filtered if n.get('type') in types]
        if min_recency > 0:
            filtered = [n for n in filtered if n.get('recency_score', 0) >= min_recency]

        # v5: Project filter
        if project:
            filtered.sort(key=lambda n: (1 if n.get('project') == project else 0, -n.get('effective_activation', 0)))

        # v5: Temporal filter
        if temporal_filter:
            after = temporal_filter.get('after')
            before = temporal_filter.get('before')
            filtered = [n for n in filtered if self._matches_temporal_filter(n.get('created_at'), after, before)]

        # Step 5: Sort by effective activation (if no project filter)
        if not project:
            filtered.sort(key=lambda n: -n.get('effective_activation', 0))

        # Step 6: Pagination
        page = filtered[offset:offset + limit]

        # Step 7: Mark accessed + Hebbian
        if not session_id:
            session_id = f'ses_{int(time.time() * 1000)}'

        for node in page:
            self._mark_accessed(node['id'], session_id)

        self._hebbian_strengthen([n['id'] for n in page])

        # v4: Auto-instrument (skipped when called from recall
        # or hooks — they log via the precision module instead)
        returned_ids = [n['id'] for n in page]
        recall_log_id = None
        if not _skip_log:
            try:
                recall_log_id = self._log_recall(session_id, query, returned_ids)
            except Exception as _e:
                self._log_error("recall", _e, "logging recall event to recall_log")

        # v6: Attach reasoning chains when intent is reasoning_chain
        reasoning_chains = []
        if intent == 'reasoning_chain':
            # 1. Pull chains for decision nodes in results
            decision_nodes = [n for n in page if n.get('type') == 'decision']
            for dn in decision_nodes:
                # Note: reasoning methods not yet implemented, skipping for now
                pass

        # TODO: _recall_log_id is returned but nothing calls mark_recall_used()
        # to close the loop.  Until that method exists, recall_log.used_count
        # stays 0 and the "recall precision" consciousness signal is always 0%.
        # See: tests/relearning.py header for the full TODO list.
        result = {
            'results': page,
            '_recall_log_id': recall_log_id,
            'intent': intent,
        }

        if reasoning_chains:
            result['reasoning_chains'] = reasoning_chains

        return result

    def recall(self, query: str, types: Optional[List[str]] = None,
               limit: int = 20, offset: int = 0,
               include_archived: bool = False,
               min_recency: float = 0, project: Optional[str] = None,
               session_id: Optional[str] = None) -> Dict[str, Any]:
        """Recall: embeddings + 3-degree graph traversal + keyword blending.

        OLD approach: Run keyword recall first, sprinkle embedding scores on top.
        NEW approach: Embed the query, scan ALL nodes by embedding similarity,
        use keywords only as a tiebreaker for exact matches (proper nouns, versions).

        Graceful degradation: if embedder isn't ready, falls back to keyword-only
        recall via self._keyword_recall() — but logs a LOUD warning because keyword-only
        recall is fundamentally broken for semantic understanding.

        Args:
            query: Search query
            types: Filter by node types
            limit: Max results
            offset: Pagination offset
            include_archived: Include archived
            min_recency: Min recency threshold
            project: Optional project filter
            session_id: Optional session ID

        Returns:
            Dict with results, _recall_log_id, _embedding_stats, intent, _recall_mode
        """
        t0 = time.time()
        limit = min(limit, MAX_PAGE_SIZE)

        # ── FALLBACK: If embedder not ready, degrade to keyword-only ──
        if not embedder.is_ready():
            result = self._keyword_recall(query, types, limit, offset, include_archived,
                               min_recency, project, session_id, _skip_log=True)
            result['_recall_mode'] = 'keyword_only_DEGRADED'
            result['_embedding_stats'] = {
                'embedder_ready': False,
                'embedder_status': embedder.get_model_status(),
                'warning': 'Recall is keyword-only. Semantic understanding disabled.',
            }
            print(f'[brain] WARNING: keyword-only recall (embedder not ready)', file=sys.stderr)
            return result

        # ── PRIMARY PATH: Embeddings-first ──

        # v5.2 Step 0.5: Vocabulary expansion (before embedding AND keyword search)
        expanded_query = self._expand_query_with_vocabulary(query)

        # STEP 1: Embed the query
        try:
            query_vec = embedder.embed(expanded_query)
            if not query_vec:
                # Embedding failed for this query — fall back
                result = self._keyword_recall(query, types, limit, offset, include_archived,
                                   min_recency, project, session_id)
                result['_recall_mode'] = 'keyword_only_DEGRADED'
                return result
        except Exception as e:
            result = self._keyword_recall(query, types, limit, offset, include_archived,
                               min_recency, project, session_id)
            result['_recall_mode'] = 'keyword_only_DEGRADED'
            return result

        # STEP 2: Get intent classification (from keyword recall path — still useful)
        intent_data = self._classify_intent(query)
        intent = intent_data['intent']
        type_boosts = intent_data['typeBoosts']

        # STEP 3: Brute-force cosine similarity against ALL stored embeddings
        # This is the core change: embeddings drive retrieval, not keywords.
        # For 600 nodes this is fast (<50ms). At 10k+ nodes, switch to sqlite-vec.
        # v6: Also scans enrichment embeddings (question, anchor, bridge, keywords vectors).
        embedding_scores = {}  # node_id → cosine_similarity (best across primary + enrichments)
        enrichment_hits = {}   # node_id → vector_type that matched best (for telemetry)
        node_personal_data = {}  # node_id → (personal, personal_context) for pre-sort penalty
        node_confidence = {}    # node_id → confidence (0-1, None=default)
        nodes_with_embeddings = 0
        nodes_without_embeddings = 0

        # Pre-compute query terms for contextual qualifier matching (applied in STEP 6)
        _query_terms_set = set(query.lower().split()) if query else set()

        try:
            archive_filter = '' if include_archived else 'AND n.archived = 0'
            type_filter = ''
            type_params = []
            if types:
                type_placeholders = ','.join('?' * len(types))
                type_filter = f'AND n.type IN ({type_placeholders})'
                type_params = list(types)
            project_filter = ''
            project_params = []
            if project:
                project_filter = 'AND (n.project = ? OR n.project IS NULL)'
                project_params = [project]

            node_critical = {}  # node_id → critical flag
            cursor = self.conn.execute(
                f'''SELECT ne.node_id, ne.embedding, n.personal, n.personal_context, n.confidence, n.critical
                    FROM node_embeddings ne
                    JOIN nodes n ON n.id = ne.node_id
                    WHERE 1=1 {archive_filter} {type_filter} {project_filter}''',
                type_params + project_params
            )
            for row in cursor.fetchall():
                node_id = row[0]
                blob = row[1]
                node_personal_data[node_id] = (row[2], row[3])  # (personal, personal_context)
                node_confidence[node_id] = row[4]  # may be None
                node_critical[node_id] = row[5] or 0  # v5.2: critical flag
                if blob:
                    sim = embedder.cosine_similarity(query_vec, blob)
                    embedding_scores[node_id] = sim
                    enrichment_hits[node_id] = 'primary'
                    nodes_with_embeddings += 1

            # v6 STEP 3.5: Scan enrichment embeddings (V5 multi-vector encoding)
            # Each node may have up to 4 enrichment vectors. Take the BEST score
            # across primary + all enrichments for each node.
            enrichment_count = 0
            enrichment_used = 0
            try:
                enrich_cursor = self.conn.execute(
                    'SELECT node_id, vector_type, embedding FROM node_enrichments WHERE embedding IS NOT NULL'
                )
                for erow in enrich_cursor.fetchall():
                    e_node_id, e_type, e_blob = erow
                    if not e_blob:
                        continue
                    enrichment_count += 1
                    # Skip if node is filtered out by type/project/archive
                    if e_node_id not in node_confidence and e_node_id not in embedding_scores:
                        # Check if node passes filters
                        check = self.conn.execute(
                            f'''SELECT 1 FROM nodes n WHERE n.id = ? {archive_filter} {type_filter} {project_filter}''',
                            [e_node_id] + type_params + project_params
                        ).fetchone()
                        if not check:
                            continue
                    e_sim = embedder.cosine_similarity(query_vec, e_blob)
                    current_best = embedding_scores.get(e_node_id, 0)
                    if e_sim > current_best:
                        embedding_scores[e_node_id] = e_sim
                        enrichment_hits[e_node_id] = e_type
                        enrichment_used += 1
            except Exception as e:
                # Table might not exist yet during migration
                if 'no such table' not in str(e):
                    self._log_error("recall_enrichment_scan", e, "STEP 3.5 enrichment scan")

        except Exception as e:
            print(f'[brain] Embedding scan error: {e}', file=sys.stderr)

        # STEP 4: Also run keyword recall to catch nodes WITHOUT embeddings
        # and to get keyword precision scores for exact-match tiebreaking
        # _skip_log=True: precision module handles logging via hooks, not here.
        keyword_result = self._keyword_recall(query, types, limit * 3, offset, include_archived,
                                    min_recency, project, session_id, _skip_log=True)
        keyword_scores = {}  # node_id → keyword_effective_activation
        keyword_nodes = {}   # node_id → full node dict
        for node in keyword_result.get('results', []):
            nid = node['id']
            keyword_scores[nid] = node.get('effective_activation', 0)
            keyword_nodes[nid] = node
            if nid not in embedding_scores:
                nodes_without_embeddings += 1

        # STEP 5: Build unified candidate set (all nodes seen by either path)
        all_candidate_ids = set(embedding_scores.keys()) | set(keyword_scores.keys())

        # STEP 6: Score each candidate — embeddings primary, keywords fallback
        scored_results = []
        for nid in all_candidate_ids:
            emb_score = embedding_scores.get(nid, 0)
            kw_score = keyword_scores.get(nid, 0)

            # Determine source and compute blended score
            if emb_score > 0 and kw_score > 0:
                # Both signals available — blend with embeddings primary
                blended = (EMBEDDING_PRIMARY_WEIGHT * emb_score +
                          KEYWORD_FALLBACK_WEIGHT * kw_score)
                source = 'embedding+keyword'
            elif emb_score > 0:
                # Embedding only — use embedding score directly
                blended = emb_score
                source = 'embedding_only'
            else:
                # Keyword only (node has no embedding) — use keyword but PENALIZE.
                # Keyword-only results lack the primary signal. They should never
                # outrank a strong embedding match. Scale by KEYWORD_FALLBACK_WEIGHT
                # so a perfect keyword match (1.0) scores at most 0.10.
                blended = KEYWORD_FALLBACK_WEIGHT * kw_score
                source = 'keyword_only_fallback'

            # Apply intent-based type boosting
            node = keyword_nodes.get(nid)
            if node:
                type_boost = type_boosts.get(node.get('type'), 1.0)
                blended *= type_boost

            # v5.2: Critical node boost — safety-important nodes always surface
            is_critical = node_critical.get(nid, 0) if 'node_critical' in dir() else 0
            if not is_critical and node:
                is_critical = node.get('critical', 0)
            if is_critical:
                blended *= CRITICAL_BOOST

            # v5: Apply confidence as a scoring factor.
            # Confidence < 1.0 penalizes (corrected nodes rank lower).
            # Confidence > default (validated nodes) gets a mild boost.
            # NULL confidence = 1.0 (no effect). Range: [0.1, 1.0] → multiplier [0.7, 1.05]
            conf = node_confidence.get(nid)
            if conf is not None:
                # Map confidence [0.1, 1.0] to multiplier [0.7, 1.05]
                # At 0.1 (heavily corrected): 0.7x penalty
                # At 0.7 (default): 1.0x (neutral)
                # At 1.0 (validated): 1.05x mild boost
                conf_multiplier = 0.7 + (conf - 0.1) * (1.05 - 0.7) / (1.0 - 0.1)
                conf_multiplier = max(0.7, min(1.05, conf_multiplier))
                blended *= conf_multiplier

            # v4 FIX: Apply contextual qualifier penalty BEFORE sorting.
            # Previously this was applied post-hoc to effective_activation after the sort,
            # meaning it had zero effect on ordering. Now it penalizes the sort key itself.
            _context_mismatch = False
            node_personal_pair = node_personal_data.get(nid)
            if not node_personal_pair and node:
                # Fallback: try to get from keyword node (may be missing)
                _np = node.get('personal')
                _npc = node.get('personal_context', '')
                node_personal_pair = (_np, _npc)
            if node_personal_pair:
                _np, _npc = node_personal_pair
                if _np == 'contextual' and _npc:
                    qualifier_terms = set(_npc.lower().split())
                    overlap = qualifier_terms & _query_terms_set
                    if not overlap:
                        blended *= 0.7
                        _context_mismatch = True

            # Minimum threshold — don't return noise
            # v5.2: Critical nodes get a much lower threshold
            min_threshold = CRITICAL_SIMILARITY_THRESHOLD if is_critical else 0.05
            if blended < min_threshold:
                continue

            scored_results.append({
                'node_id': nid,
                'blended_score': blended,
                'embedding_similarity': round(emb_score * 1000) / 1000 if emb_score else None,
                'keyword_score': round(kw_score * 1000) / 1000 if kw_score else None,
                '_source': source,
                '_context_mismatch': _context_mismatch,
            })

        # Sort by blended score descending
        scored_results.sort(key=lambda x: -x['blended_score'])
        scored_results = scored_results[:limit]

        # STEP 6.5: 3-degree graph traversal from top seed nodes.
        # Walks intentional edges at degree 1, all edges (except co_accessed)
        # at degree 2-3. Semantic bonus when graph + embedding converge.
        graph_neighborhoods = {}
        try:
            seeds = scored_results[:GRAPH_AUGMENT_TOP_N]
            graph_candidates, graph_neighborhoods = self._traverse_graph(
                seeds, query_vec=query_vec)

            # Add graph-discovered nodes to scored results
            for nid, info in graph_candidates.items():
                scored_results.append({
                    'node_id': nid,
                    'blended_score': info['score'],
                    'embedding_similarity': None,
                    'keyword_score': None,
                    '_source': info['discovery'],
                    '_context_mismatch': False,
                })

            if graph_candidates:
                scored_results.sort(key=lambda x: -x['blended_score'])
                scored_results = scored_results[:limit]
        except Exception as e:
            self._log_error("recall", e, "STEP 6.5 graph traversal")

        # STEP 6.9: Version-aware relevance floor.
        # Enriched nodes (V5+) get a higher bar — their scores are more meaningful.
        # Bare nodes (primary embedding only) get a lower bar — one vector, one chance.
        # If best result is below applicable floor, return empty.
        if scored_results:
            top_id = scored_results[0]['node_id']
            max_score = scored_results[0]['blended_score']
            top_source = enrichment_hits.get(top_id, 'primary')
            floor = RELEVANCE_FLOOR_ENRICHED if top_source != 'primary' else RELEVANCE_FLOOR_PRIMARY
            if max_score < floor:
                scored_results = []

        # STEP 7: Hydrate full node data for top results
        final_results = []
        for sr in scored_results:
            nid = sr['node_id']
            node = keyword_nodes.get(nid)
            if not node:
                # Node came from embedding-only path — fetch from DB
                try:
                    cursor = self.conn.execute(
                        '''SELECT id, type, title, content, keywords, activation, stability,
                                  access_count, locked, archived, last_accessed, created_at,
                                  emotion, emotion_label, project, personal, personal_context,
                                  content_summary, confidence, updated_at, revised_at
                           FROM nodes WHERE id = ?''',
                        (nid,)
                    )
                    row = cursor.fetchone()
                    if row:
                        node = {
                            'id': row[0], 'type': row[1], 'title': row[2],
                            'content': row[3], 'keywords': row[4],
                            'activation': row[5], 'stability': row[6],
                            'access_count': row[7], 'locked': row[8] == 1,
                            'archived': row[9] == 1, 'last_accessed': row[10],
                            'created_at': row[11], 'emotion': row[12],
                            'emotion_label': row[13], 'project': row[14],
                            'personal': row[15], 'personal_context': row[16],
                            'content_summary': row[17],
                            'confidence': row[18],
                            'updated_at': row[19],
                            'revised_at': row[20],
                        }
                except Exception as e:
                    self._log_error("recall_hydrate", e, "Failed to hydrate node %s" % nid[:8])
                    continue

            if node:
                node['effective_activation'] = sr['blended_score']
                node['embedding_similarity'] = sr['embedding_similarity']
                node['_keyword_score'] = sr['keyword_score']
                node['_source'] = sr['_source']
                if sr.get('_context_mismatch'):
                    node['_context_mismatch'] = True

                # v4: Brain→Host communication dimensions.
                # The brain expresses WHAT it needs to communicate. The host adapter
                # translates HOW. Four dimensions (all 0-1):
                #   priority: how important (locked=high, evolution=high, regular=medium)
                #   confidence: how certain (locked=1.0, hypothesis=its confidence, regular=0.7)
                #   action_expected: should host act on it? (rule/constraint=yes, context=no)
                #   feedback_needed: does brain need a response? (evolution=yes, fact=no)
                ntype = node.get('type', '')
                is_locked = node.get('locked', False)
                is_evolution = ntype in ('tension', 'hypothesis', 'pattern', 'catalyst', 'aspiration')
                is_rule = ntype in ('rule', 'arch_constraint', 'bug_lesson', 'failure_mode',
                                    'constraint', 'lesson', 'convention')  # v5 engineering memory
                is_cognitive = ntype in ('mental_model', 'reasoning_trace', 'uncertainty',
                                         'correction', 'validation')  # v5 cognitive layer
                is_engineering = ntype in ('purpose', 'mechanism', 'impact', 'vocabulary')  # v5 engineering

                # Temporal freshness — how old is this info?
                _freshness = 'unknown'
                _created = node.get('created_at', '')
                if _created:
                    try:
                        from datetime import timezone
                        _cdt = datetime.fromisoformat(_created.replace('Z', '+00:00'))
                        _age_hours = (datetime.now(timezone.utc) - _cdt).total_seconds() / 3600
                        if _age_hours < 1:
                            _freshness = 'just_now'
                        elif _age_hours < 24:
                            _freshness = 'today'
                        elif _age_hours < 168:
                            _freshness = 'this_week'
                        elif _age_hours < 720:
                            _freshness = 'this_month'
                        else:
                            _freshness = 'older'
                    except Exception:
                        pass

                node['_brain_to_host'] = {
                    'priority': 0.9 if is_locked or is_evolution else (
                        0.8 if is_engineering or is_cognitive else (0.7 if is_rule else 0.5)),
                    'confidence': node.get('confidence') or (1.0 if is_locked else 0.7),
                    'action_expected': is_rule or is_locked or ntype == 'impact',
                    'feedback_needed': is_evolution or ntype in ('failure_mode', 'uncertainty', 'correction'),
                    'freshness': _freshness,
                }

                # Attach graph neighborhood from traversal
                node['_graph'] = graph_neighborhoods.get(nid, {})
                node['_discovery'] = sr.get('_source', 'embedding')

                # v4: Contextual qualifier matching.
                # Penalty is applied to blended_score BEFORE sorting in STEP 6.
                # Here we only apply confidence reduction and set the qualifier label.
                node_personal = node.get('personal')
                if node_personal == 'contextual':
                    pctx = node.get('personal_context', '')
                    if pctx and query:
                        qualifier_terms = set(pctx.lower().split())
                        query_terms_set = set(query.lower().split())
                        overlap = qualifier_terms & query_terms_set
                        if not overlap:
                            node['_context_mismatch'] = True
                            node['_context_qualifier'] = pctx
                            # Score penalty already applied in STEP 6 — only reduce confidence here
                            node['_brain_to_host']['confidence'] *= 0.6

                final_results.append(node)

        # STEP 7.5: Enrich top 3 results with metadata + neighbors
        # All results keep full content (no truncation). Top 3 also get
        # metadata and neighbor context for richer understanding.
        self._enrich_results(final_results[:3])

        # STEP 8: Mark accessed (for Hebbian learning)
        sid = session_id or ('ses_%d' % int(time.time() * 1000))
        for node in final_results:
            try:
                self._mark_accessed(node['id'], sid)
            except Exception as _e:
                self._log_error("recall", _e, "marking node as accessed for Hebbian learning")

        # STEP 9: Build result
        recall_ms = (time.time() - t0) * 1000
        result = {
            'results': final_results,
            # Logging is now handled by the precision module in hooks, not here.
            '_recall_log_id': None,
            'intent': intent,
            '_recall_mode': 'embeddings_first',
            '_embedding_stats': {
                'embedder_ready': True,
                'nodes_with_embeddings': nodes_with_embeddings,
                'nodes_without_embeddings': nodes_without_embeddings,
                'embedding_primary_weight': EMBEDDING_PRIMARY_WEIGHT,
                'keyword_fallback_weight': KEYWORD_FALLBACK_WEIGHT,
                'recall_ms': round(recall_ms, 1),
                'results_by_source': {
                    'embedding+keyword': sum(1 for r in final_results if r.get('_discovery') == 'embedding+keyword'),
                    'embedding_only': sum(1 for r in final_results if r.get('_discovery') == 'embedding_only'),
                    'keyword_only_fallback': sum(1 for r in final_results if r.get('_discovery') == 'keyword_only_fallback'),
                    'graph_d1': sum(1 for r in final_results if r.get('_discovery') == 'graph_d1'),
                    'graph_d2': sum(1 for r in final_results if r.get('_discovery') == 'graph_d2'),
                    'graph_d3': sum(1 for r in final_results if r.get('_discovery') == 'graph_d3'),
                    'convergence': sum(1 for r in final_results if r.get('_discovery') == 'convergence'),
                },
                # v6: Enrichment scan stats
                'enrichment_vectors_scanned': enrichment_count if 'enrichment_count' in dir() else 0,
                'enrichment_vectors_used': enrichment_used if 'enrichment_used' in dir() else 0,
                'results_via_enrichment': sum(1 for r in final_results if enrichment_hits.get(r.get('id', ''), 'primary') != 'primary'),
            },
        }

        # Gap detection: when no results pass the relevance floor,
        # flag it so the voice layer can prompt encoding
        if not final_results and query:
            result['_gap'] = {
                'query': query,
                'top_score': max_score if 'max_score' in locals() else 0,
            }

        # Carry over reasoning chains from keyword result
        if keyword_result.get('reasoning_chains'):
            result['reasoning_chains'] = keyword_result['reasoning_chains']

        # v5.1: Return query embedding for segment boundary detection
        # Zero cost — already computed in STEP 1
        result['_query_embedding'] = query_vec

        return result

    def _enrich_results(self, results: List[Dict[str, Any]], neighbor_limit: int = 3) -> None:
        """Enrich recall results with metadata and neighbor context.

        This is the ONE place that makes a node result 'complete'.
        Called by both recall (text search) and recall_node (ID lookup).
        The caller decides WHICH results to enrich — this method enriches all it receives.
        Mutates results in place.
        """
        graph_dal = GraphDAL(self.conn)
        for node in results:
            nid = node.get('id')
            if not nid:
                continue

            # Attach metadata from node_metadata table
            try:
                meta_cur = self.conn.execute(
                    'SELECT reasoning, user_raw_quote, correction_of, last_validated '
                    'FROM node_metadata WHERE node_id = ?',
                    (nid,)
                )
                meta_row = meta_cur.fetchone()
                if meta_row and any(meta_row):
                    node['_metadata'] = {
                        'reasoning': meta_row[0],
                        'user_raw_quote': meta_row[1],
                        'correction_of': meta_row[2],
                        'last_validated': meta_row[3],
                    }
            except Exception:
                pass

            # Attach neighbor context — reuse existing get_neighbors_with_context,
            # filter to intentional edges in Python
            if neighbor_limit > 0:
                try:
                    all_neighbors = graph_dal.get_neighbors_with_context(
                        nid, limit=neighbor_limit * 3  # fetch extra, filter down
                    )
                    node['_neighbors'] = [
                        {'id': nb['id'], 'type': nb['type'], 'title': nb['title'],
                         'relation': nb['relation'], 'weight': nb['weight']}
                        for nb in all_neighbors
                        if nb.get('relation') in INTENTIONAL_EDGE_TYPES
                    ][:neighbor_limit]
                except Exception:
                    node['_neighbors'] = []

    def _traverse_graph(self, seeds: List[Dict], query_vec=None) -> tuple:
        """3-degree graph traversal from seed nodes.

        Walks the graph outward from embedding hits:
        - Degree 1: INTENTIONAL edges only (strong signal)
        - Degree 2-3: All edges except co_accessed (wider net)

        Returns:
            (graph_candidates, graph_neighborhoods)
            - graph_candidates: dict of node_id → {'score': float, 'discovery': str}
            - graph_neighborhoods: dict of seed_id → {'degree_1': [...], 'degree_2': [...], 'degree_3': [...]}
        """
        graph_dal = GraphDAL(self.conn)
        existing_ids = {s['node_id'] for s in seeds}
        seen_ids = set(existing_ids)  # Track all visited to avoid cycles

        # Track: node_id → [(score, parent_id, degree)]
        candidate_hits = {}
        # Track: seed_id → {degree_1: [...], degree_2: [...], degree_3: [...]}
        neighborhoods = {}

        def _freshness(node_data):
            """Compute freshness multiplier from revised_at or created_at."""
            ts = node_data.get('revised_at') or node_data.get('created_at') or ''
            if not ts:
                return FRESHNESS_MULTIPLIERS.get('older', 0.6)
            try:
                dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
                hours = (datetime.now(dt.tzinfo) - dt).total_seconds() / 3600
            except Exception:
                return FRESHNESS_MULTIPLIERS.get('older', 0.6)
            if hours < 24:
                return FRESHNESS_MULTIPLIERS.get('today', 1.2)
            elif hours < 168:
                return FRESHNESS_MULTIPLIERS.get('this_week', 1.0)
            elif hours < 720:
                return FRESHNESS_MULTIPLIERS.get('this_month', 0.8)
            return FRESHNESS_MULTIPLIERS.get('older', 0.6)

        def _semantic_bonus(node_id):
            """Compute additive semantic bonus if query_vec available."""
            if query_vec is None:
                return 0.0
            try:
                row = self.conn.execute(
                    'SELECT embedding FROM node_embeddings WHERE node_id = ?', (node_id,)
                ).fetchone()
                if row and row[0]:
                    node_vec = struct.unpack('%df' % (len(row[0]) // 4), row[0])
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(query_vec, node_vec))
                    mag_a = sum(a * a for a in query_vec) ** 0.5
                    mag_b = sum(b * b for b in node_vec) ** 0.5
                    if mag_a > 0 and mag_b > 0:
                        sim = dot / (mag_a * mag_b)
                        if sim >= TRAVERSE_SEMANTIC_THRESHOLD:
                            return TRAVERSE_SEMANTIC_BONUS * sim
            except Exception:
                pass
            return 0.0

        for seed in seeds:
            seed_id = seed['node_id']
            parent_score = seed['blended_score']
            neighborhoods[seed_id] = {'degree_1': [], 'degree_2': [], 'degree_3': []}

            # Compute non-intentional relations for degree-1 exclusion
            # Degree 1: intentional only → exclude everything NOT intentional + co_accessed
            d1_exclude_relations = EXCLUDED_EDGE_TYPES  # will be filtered post-query for intentional

            # ── Degree 1: intentional edges only, skip visited nodes in SQL ──
            d1_neighbors = graph_dal.get_neighbors_rich(
                seed_id, limit=TRAVERSE_LIMITS[0] * 2,  # fetch extra, filter intentional
                exclude_relations=EXCLUDED_EDGE_TYPES,
                exclude_node_ids=seen_ids)
            # Filter to intentional at degree 1
            d1_neighbors = [n for n in d1_neighbors
                            if n.get('relation') in INTENTIONAL_EDGE_TYPES][:TRAVERSE_LIMITS[0]]

            for nb in d1_neighbors:
                nid = nb['id']
                freshness = _freshness(nb)
                d1_score = parent_score * TRAVERSE_DAMPEN[0] * (nb.get('weight') or 0.5) * freshness
                d1_score += _semantic_bonus(nid)

                neighborhoods[seed_id]['degree_1'].append(nb)
                seen_ids.add(nid)

                if nid not in existing_ids:
                    if nid not in candidate_hits:
                        candidate_hits[nid] = []
                    candidate_hits[nid].append((d1_score, seed_id, 1))

                # ── Degree 2: all edges except co_accessed, skip visited in SQL ──
                if len(neighborhoods[seed_id]['degree_2']) < TRAVERSE_LIMITS[1] * 3:
                    d2_neighbors = graph_dal.get_neighbors_rich(
                        nid, limit=TRAVERSE_LIMITS[1],
                        exclude_relations=EXCLUDED_EDGE_TYPES,
                        exclude_node_ids=seen_ids)

                    for nb2 in d2_neighbors:
                        nid2 = nb2['id']
                        freshness2 = _freshness(nb2)
                        d2_score = d1_score * TRAVERSE_DAMPEN[1] / TRAVERSE_DAMPEN[0] * (nb2.get('weight') or 0.5) * freshness2
                        d2_score += _semantic_bonus(nid2)

                        neighborhoods[seed_id]['degree_2'].append(nb2)
                        seen_ids.add(nid2)

                        if nid2 not in existing_ids:
                            if nid2 not in candidate_hits:
                                candidate_hits[nid2] = []
                            candidate_hits[nid2].append((d2_score, seed_id, 2))

                        # ── Degree 3: all edges except co_accessed, skip visited in SQL ──
                        if len(neighborhoods[seed_id]['degree_3']) < TRAVERSE_LIMITS[2] * 3:
                            d3_neighbors = graph_dal.get_neighbors_rich(
                                nid2, limit=TRAVERSE_LIMITS[2],
                                exclude_relations=EXCLUDED_EDGE_TYPES,
                                exclude_node_ids=seen_ids)

                            for nb3 in d3_neighbors:
                                nid3 = nb3['id']
                                d3_score = d2_score * TRAVERSE_DAMPEN[2] / TRAVERSE_DAMPEN[1] * (nb3.get('weight') or 0.5)
                                neighborhoods[seed_id]['degree_3'].append(
                                    {'id': nid3, 'title': nb3.get('title', '')})
                                seen_ids.add(nid3)

                                if nid3 not in existing_ids:
                                    if nid3 not in candidate_hits:
                                        candidate_hits[nid3] = []
                                    candidate_hits[nid3].append((d3_score, seed_id, 3))

        # ── Convergence boost ──
        graph_candidates = {}
        for nid, hits in candidate_hits.items():
            base_score = max(h[0] for h in hits)
            num_parents = len(set(h[1] for h in hits))
            min_degree = min(h[2] for h in hits)
            convergence = 1.0 + TRAVERSE_CONVERGENCE_BOOST * (num_parents - 1)
            final_score = base_score * convergence

            discovery = 'graph_d%d' % min_degree
            if num_parents > 1:
                discovery = 'convergence'

            graph_candidates[nid] = {'score': final_score, 'discovery': discovery}

        return graph_candidates, neighborhoods

    def recall_node(self, node_id: str, neighbor_limit: int = 3) -> Dict[str, Any]:
        """Recall a specific node by ID with full enrichment.

        Returns same shape as recall() so callers get a
        consistent interface regardless of how the node was found.
        """
        from .dal import NodeDAL
        node_dal = NodeDAL(self.conn)
        node = node_dal.get_node(node_id)
        if not node:
            return {'results': [], 'intent': 'direct_lookup'}

        # Set display fields for format compatibility
        node['effective_activation'] = node.get('activation', 0)
        node['embedding_similarity'] = None
        node['_keyword_score'] = None
        node['_source'] = 'direct_lookup'

        results = [node]
        self._enrich_results(results, neighbor_limit=neighbor_limit)

        return {
            'results': results,
            'intent': 'direct_lookup',
            '_recall_mode': 'by_id',
        }

    def spread_activation(self, seed_ids: List[str], types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Spread activation from seed nodes through graph edges.

        Multi-hop with exponential decay (0.5^hop).
        Each hop: get neighbors, multiply activation by edge_weight * decay.
        MAX_HOPS=3, MAX_NEIGHBORS=50 per node.

        Args:
            seed_ids: Starting node IDs
            types: Optional filter by node types

        Returns:
            List of activated nodes with spread_activation scores
        """
        activation = {}
        node_cache = {}

        for sid in seed_ids:
            activation[sid] = 1.0

        for hop in range(MAX_HOPS):
            decay_factor = SPREAD_DECAY ** (hop + 1)
            current_nodes = [(nid, act) for nid, act in activation.items() if act > 0.01]

            for node_id, node_activation in current_nodes:
                cursor = self.conn.execute(
                    '''SELECT target_id, weight FROM edges
                       WHERE source_id = ? AND weight > ?
                       ORDER BY weight DESC LIMIT ?''',
                    (node_id, PRUNE_THRESHOLD, MAX_NEIGHBORS)
                )

                for row in cursor.fetchall():
                    target_id = row[0]
                    edge_weight = row[1]
                    spread = node_activation * edge_weight * decay_factor
                    current_act = activation.get(target_id, 0)
                    activation[target_id] = current_act + spread

        # Fetch full node data
        results = []
        for node_id, act in activation.items():
            node = node_cache.get(node_id)
            if not node:
                cursor = self.conn.execute(
                    '''SELECT id, type, title, content, keywords, activation, stability,
                              access_count, locked, archived, last_accessed, created_at,
                              emotion, emotion_label, project, critical
                       FROM nodes WHERE id = ?''',
                    (node_id,)
                )
                row = cursor.fetchone()
                if row:
                    node = {
                        'id': row[0], 'type': row[1], 'title': row[2],
                        'content': row[3], 'keywords': row[4],
                        'activation': row[5], 'stability': row[6],
                        'access_count': row[7], 'locked': row[8] == 1,
                        'archived': row[9] == 1, 'last_accessed': row[10],
                        'created_at': row[11],
                        'emotion': row[12] or 0, 'emotion_label': row[13] or 'neutral',
                        'project': row[14],
                        'critical': row[15] == 1 if len(row) > 15 else False
                    }
                    node_cache[node_id] = node

            if node:
                # Type filter
                if types and node.get('type') not in types:
                    continue

                results.append({**node, 'spread_activation': act})

        return results

    def _search_keywords(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search nodes by keyword/title/content.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching nodes
        """
        words = self._tfidf_tokenize(query)
        if not words:
            return []

        # Build OR conditions: (LIKE word1 OR LIKE word2 OR ...)
        conditions = []
        params = []
        for w in words:
            conditions.append('(LOWER(keywords) LIKE ? OR LOWER(title) LIKE ? OR LOWER(content) LIKE ?)')
            params.extend([f'%{w}%', f'%{w}%', f'%{w}%'])

        where_clause = ' OR '.join(conditions)
        params.append(limit)

        try:
            cursor = self.conn.execute(
                f'''SELECT id, type, title, content, keywords, activation, stability,
                           access_count, locked, archived, last_accessed, created_at, project, critical
                    FROM nodes WHERE archived = 0 AND ({where_clause}) LIMIT ?''',
                params
            )
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0], 'type': row[1], 'title': row[2],
                    'content': row[3], 'keywords': row[4],
                    'activation': row[5], 'stability': row[6],
                    'access_count': row[7], 'locked': row[8] == 1,
                    'archived': row[9] == 1, 'last_accessed': row[10],
                    'created_at': row[11], 'project': row[12],
                    'critical': row[13] == 1 if len(row) > 13 else False
                })
            return results
        except Exception:
            return []

    def _mark_accessed(self, node_id: str, session_id: str):
        """Mark a node as accessed and log it."""
        ts = self.now()
        self.conn.execute(
            '''UPDATE nodes
               SET access_count = access_count + 1,
                   activation = MIN(1.0, activation + 0.1),
                   recency_score = 1.0,
                   last_accessed = ?,
                   updated_at = ?
               WHERE id = ?''',
            (ts, ts, node_id)
        )
        self.logs_conn.execute(
            'INSERT INTO access_log (session_id, node_id, timestamp) VALUES (?, ?, ?)',
            (session_id, node_id, ts)
        )
        self.logs_conn.commit()
        self.conn.commit()

    def _hebbian_strengthen(self, node_ids: List[str], segment_node_ids: Optional[List[str]] = None):
        """
        Strengthen connections between co-accessed nodes (Hebbian learning).

        If two nodes are co-recalled but have no edge, CREATE a co_accessed edge.
        If they already have an edge, strengthen it.
        This is how the brain auto-discovers relationships from usage patterns.

        v5.1: When segment_node_ids is provided, only create NEW co_accessed edges
        between nodes that are both in the same segment. Existing edges are always
        strengthened regardless (if they co-fire across segments, the edge earned it).
        """
        if len(node_ids) < 2:
            return

        segment_set = set(segment_node_ids) if segment_node_ids else None

        ts = self.now()

        # Cap pairwise work: only top N nodes to avoid O(n^2) explosion
        ids = node_ids[:15]

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                nid_i = ids[i]
                nid_j = ids[j]

                # Check if edge exists (either direction)
                cursor = self.conn.execute(
                    'SELECT weight, co_access_count, stability FROM edges WHERE source_id = ? AND target_id = ?',
                    (nid_i, nid_j)
                )
                row = cursor.fetchone()

                if not row:
                    # Check reverse direction
                    cursor = self.conn.execute(
                        'SELECT weight, co_access_count, stability FROM edges WHERE source_id = ? AND target_id = ?',
                        (nid_j, nid_i)
                    )
                    row = cursor.fetchone()
                    if row:
                        # Reverse exists — update it
                        w, count, stab = row
                        new_weight = min(MAX_WEIGHT, w + LEARNING_RATE * 0.1)
                        new_stability = min(stab * STABILITY_BOOST, 10.0)
                        self.conn.execute(
                            '''UPDATE edges
                               SET weight = ?, co_access_count = ?, stability = ?, last_strengthened = ?
                               WHERE source_id = ? AND target_id = ?''',
                            (new_weight, count + 1, new_stability, ts, nid_j, nid_i)
                        )
                        continue

                if row:
                    # Edge exists — strengthen it
                    w, count, stab = row
                    new_weight = min(MAX_WEIGHT, w + LEARNING_RATE * 0.1)
                    new_stability = min(stab * STABILITY_BOOST, 10.0)

                    self.conn.execute(
                        '''UPDATE edges
                           SET weight = ?, co_access_count = ?, stability = ?, last_strengthened = ?
                           WHERE source_id = ? AND target_id = ?''',
                        (new_weight, count + 1, new_stability, ts, nid_i, nid_j)
                    )
                else:
                    # NO edge exists — CREATE a co_accessed edge
                    # v5.1: Only create NEW co_accessed edges within same segment
                    # (existing edges always get strengthened — they earned it)
                    if segment_set and (nid_i not in segment_set or nid_j not in segment_set):
                        continue  # Different segments — skip new edge creation

                    # Start with low weight; repeated co-access will strengthen it
                    self.conn.execute(
                        '''INSERT OR IGNORE INTO edges
                           (source_id, target_id, weight, relation, edge_type, co_access_count,
                            stability, last_strengthened, created_at)
                           VALUES (?, ?, ?, 'co_accessed', 'co_accessed', 1, 1.0, ?, ?)''',
                        (nid_i, nid_j, EDGE_TYPES['co_accessed']['defaultWeight'], ts, ts)
                    )

        self.conn.commit()

    def _log_recall(self, session_id: str, query: str, returned_ids: List[str]) -> Optional[str]:
        """Log a recall event."""
        ts = self.now()
        cursor = self.logs_conn.execute(
            '''INSERT INTO recall_log (session_id, query, returned_ids, returned_count, created_at)
               VALUES (?, ?, ?, ?, ?)''',
            (session_id or 'unknown', query, json.dumps(returned_ids), len(returned_ids), ts)
        )
        self.logs_conn.commit()

        # Return the last inserted row ID
        cursor = self.logs_conn.execute('SELECT last_insert_rowid()')
        row = cursor.fetchone()
        return str(row[0]) if row else None

    def _get_recent(self, limit: int = 20, types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get recently accessed nodes."""
        sql = 'SELECT id, type, title, content, keywords, activation, stability, access_count, locked, archived, last_accessed, created_at FROM nodes WHERE archived = 0'
        params = []

        if types:
            placeholders = ','.join('?' * len(types))
            sql += f' AND type IN ({placeholders})'
            params.extend(types)

        sql += ' ORDER BY last_accessed DESC LIMIT ?'
        params.append(limit)

        results = []
        cursor = self.conn.execute(sql, params)
        for row in cursor.fetchall():
            results.append({
                'id': row[0], 'type': row[1], 'title': row[2],
                'content': row[3], 'keywords': row[4],
                'activation': row[5], 'stability': row[6],
                'access_count': row[7], 'locked': row[8] == 1,
                'archived': row[9] == 1, 'last_accessed': row[10],
                'created_at': row[11],
                'spread_activation': row[5],
                'effective_activation': row[5]
            })
        return results

    def _matches_temporal_filter(self, created_at: Optional[str], after: Optional[str], before: Optional[str]) -> bool:
        """Check if a node creation date matches temporal filter."""
        if not created_at:
            return False
        if after and created_at < after:
            return False
        if before and created_at > before:
            return False
        return True
