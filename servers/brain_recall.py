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
    INTENTIONAL_EDGE_TYPES,
    KEYWORD_FALLBACK_WEIGHT,
    LEARNING_RATE,
    MAX_HOPS,
    MAX_NEIGHBORS,
    MAX_PAGE_SIZE,
    MAX_WEIGHT,
    PRUNE_THRESHOLD,
    RELEVANCE_FLOOR_ENRICHED,
    RELEVANCE_FLOOR_PRIMARY,
    SPREAD_DECAY,
    STABILITY_BOOST,
    STABILITY_FLOOR_ACCESS_THRESHOLD,
    STABILITY_FLOOR_RETENTION,

    TRAVERSE_DAMPEN,
    TRAVERSE_LIMITS,
    TRAVERSE_SEMANTIC_BONUS,
    TRAVERSE_SEMANTIC_THRESHOLD,
    TRAVERSE_CONVERGENCE_BOOST,
    FRESHNESS_MULTIPLIERS,
    EXCLUDED_EDGE_TYPES,
    SITUATION_WEIGHT,
    SITUATION_THRESHOLD,
    NOISE_FLOOR_THRESHOLD,
    FTS5_CANDIDATE_LIMIT,
    FTS5_SEARCH_LIMIT,
    FTS5_PASSTHROUGH_SCORE,
    RETRIEVAL_LOW_CONFIDENCE,
)
from .dal import GraphDAL, NodeDAL, EmbeddingDAL, TfIdfDAL, LogsDAL, EnrichmentDAL, Fts5DAL


class BrainRecallMixin:
    """Recall methods for Brain."""

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node by ID with full content and connections."""
        node_dal = NodeDAL(self.conn)
        node = node_dal.get_node(node_id)
        if not node:
            return None
        # Attach connections
        graph_dal = GraphDAL(self.conn)
        neighbors = graph_dal.get_neighbors_with_context(node_id, limit=10)
        node["connections"] = [
            {"target_id": n["id"], "relation": n.get("relation", ""),
             "weight": n.get("weight", 0), "title": n.get("title", ""),
             "type": n.get("type", "")}
            for n in neighbors
        ]
        return node

    def filter_nodes(self, field: str, include=None, exclude=None,
                     lt=None, gt=None, limit: int = 50,
                     sort_by: str = 'created_at', sort_order: str = 'desc'):
        """Structured query: filter nodes by any structural field."""
        node_dal = NodeDAL(self.conn)
        return node_dal.filter_nodes(
            field=field, include=include, exclude=exclude,
            lt=lt, gt=gt, limit=limit, sort_by=sort_by, sort_order=sort_order)

    def query_logs(self, source: str = 'all', hours: int = 24,
                   level: str = 'all', hook_name: str = '',
                   limit: int = 50):
        """Query brain logs: errors, debug events, and signals."""
        return self._logs_dal.query_logs(
            source=source, hours=hours, level=level,
            hook_name=hook_name, limit=limit)

    def query_traces(self, scale: str = '', hours: int = 24,
                     event_type: str = '', chain_id: str = '',
                     session_id: str = '', ref_type: str = '',
                     grouped: bool = False, limit: int = 100):
        """Query trace events — the fractal learning loop data.

        Modes:
        - chain_id set: return single chain with all events
        - ref_type set: filter events by ref_type
        - grouped=True + session_id: return chains grouped with nested events
        - default: return flat recent events
        """
        if chain_id:
            return {'chain': self._trace_dal.get_chain(chain_id)}
        if ref_type:
            return {'events': self._trace_dal.get_by_ref_type(
                ref_type=ref_type, scale=scale, hours=hours, limit=limit)}
        if grouped and session_id:
            return {'chains': self._trace_dal.get_chains(
                session_id=session_id, scale=scale, hours=hours, limit=limit)}
        if session_id:
            return {'events': self._trace_dal.get_recent(
                scale=scale, hours=hours, event_type=event_type, limit=limit)}
        return {'events': self._trace_dal.get_recent(
            scale=scale, hours=hours, event_type=event_type, limit=limit)}

    def query_outcomes(self, chain_id: str = '', scale: str = '',
                       hours: int = 168):
        """Query outcome events — the learning signal."""
        return self._trace_dal.get_outcomes(
            chain_id=chain_id, scale=scale, hours=hours)

    def count_traces(self, field: str, scale: str = '', hours: int = 24):
        """Count trace events grouped by a field."""
        return self._trace_dal.count_by(field=field, scale=scale, hours=hours)

    def list_interactions(self):
        """List all registered interactions with latest versions."""
        return self._interaction_dal.list_all()

    def get_interaction(self, name: str, version: int = 0):
        """Get an interaction by name, optionally a specific version."""
        if version:
            return self._interaction_dal.get_version(name, version)
        return self._interaction_dal.get_latest(name)

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
        _emb_dal = EmbeddingDAL(self.conn)
        emb_rows = _emb_dal.get_all_embeddings(exclude_archived=True)

        if not emb_rows:
            return []

        # Score every node
        scored = []
        for row in emb_rows:
            node_id, blob = row['node_id'], row['embedding']
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
                _emb_dal = EmbeddingDAL(self.conn)
                _emb_dal.store_embedding(node_id, blob, embedder.stats['model_name'])
                stored += 1
            except Exception as e:
                self._log_error('batch_embed_store', e, 'storing embedding for node %s' % node_id[:12])

        self.conn.commit()
        return stored

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

        expanded_query = query

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
                tfidf_dal = TfIdfDAL(self.conn)
                tfidf_node_ids = tfidf_dal.get_nodes_matching_terms(unique_terms)
                _node_dal = NodeDAL(self.conn)
                for nid in tfidf_node_ids[:50]:
                    if nid not in all_seeds:
                        node = _node_dal.get_node(nid)
                        if node and not node.get('archived'):
                            all_seeds[nid] = node
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
            _graph_dal = GraphDAL(self.conn)
            edge_count = _graph_dal.count_node_edges(node['id'], min_weight=0)
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

            # v10: Keyword path produces raw relevance score.
            # Unified scoring is applied ONCE in STEP 6 of recall() after
            # embedding and keyword scores merge. Applying it here would
            # double-modulate. Keep keyword path output as raw relevance.
            emotion_intensity = abs(node.get('emotion', 0))
            effective = relevance

            scored.append({
                **node,
                'relevance_score': relevance,
                'semantic_score': semantic_score,
                'keyword_relevance': keyword_relevance,
                'emotion_intensity': emotion_intensity,
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
            session_id = self.session_id

        for node in page:
            self._mark_accessed(node['id'], session_id)

        # v10: Hebbian co_accessed edge creation DISABLED.
        # Previously: every recall created co_accessed edges between all top-25 results.
        # This produced 71K noise edges (90% of graph) that destroyed topology.
        # Biology: neurons that fire together wire together — but our "firing together"
        # was just "scored similarly on cosine," not meaningful co-activation.
        #
        # Re-enable when: judge-selected node IDs are available in hook_post_response_track.
        # Then: only strengthen between nodes the judge selected AND the assistant used.
        # That's real co-activation — two memories genuinely contributing to the same response.

        # v4: Auto-instrument (skipped when called from recall
        # or hooks — they log via the precision module instead)
        returned_ids = [n['id'] for n in page]
        # recall_log writes REMOVED 2026-04-05 — S1 traces capture all recall data.

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
               session_id: Optional[str] = None,
               situation_vec=None, source: str = 'unknown') -> Dict[str, Any]:
        """Recall: embeddings + 3-degree graph traversal + keyword blending + situation matching.

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
            # recall_log writes REMOVED 2026-04-05 — S1 traces capture all recall data.
            # _recall_log_id no longer returned (consumers use stop counter instead).
            return result

        # ── PRIMARY PATH: Embeddings-first ──

        expanded_query = query

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
        # v8.7: Primary scores stored separately. Enrichments boost but don't replace (ENRICHMENT_CAP).
        embedding_scores = {}  # node_id → final cosine score (primary + capped enrichment)
        primary_scores = {}    # node_id → primary embedding similarity (content match)
        enrichment_hits = {}   # node_id → vector_type that matched best (for telemetry)
        node_personal_data = {}  # node_id → (personal, personal_context) for pre-sort penalty
        node_confidence = {}    # node_id → confidence (0-1, None=default)
        nodes_with_embeddings = 0
        nodes_without_embeddings = 0

        # v9: Brain size for retrieval stats (computed once)
        try:
            _brain_size = self.conn.execute("SELECT COUNT(*) FROM nodes WHERE archived = 0").fetchone()[0]
        except Exception:
            _brain_size = 0

        # Pre-compute query terms for contextual qualifier matching (applied in STEP 6)
        _query_terms_set = set(query.lower().split()) if query else set()

        try:
            node_critical = {}  # node_id → critical flag
            node_titles = {}    # node_id → title (for title-match boost)
            node_types = {}     # node_id → type (for vocab detection)
            _emb_dal = EmbeddingDAL(self.conn)
            emb_rows = _emb_dal.get_all_with_context(
                exclude_archived=not include_archived,
                types=types, project=project)
            # Per-node data collected here for unified_score() in STEP 6.
            # created_at/emotion/access_count feed the modulator formula.
            node_created_at = {}    # node_id → ISO timestamp (for freshness)
            node_emotion = {}       # node_id → float (for emotional amplification)
            node_access_count = {}  # node_id → int (for hub penalty)

            for row in emb_rows:
                node_id = row['node_id']
                blob = row['embedding']
                node_personal_data[node_id] = (row['personal'], row['personal_context'])
                node_confidence[node_id] = row['confidence']
                node_critical[node_id] = row['critical']
                node_titles[node_id] = row['title']
                node_types[node_id] = row['type']
                node_created_at[node_id] = row.get('created_at')
                node_emotion[node_id] = row.get('emotion', 0)
                node_access_count[node_id] = row.get('access_count', 0)
                if blob:
                    sim = embedder.cosine_similarity(query_vec, blob)

                    # v10: Synaptic fatigue — nodes recalled repeatedly this session
                    # get dampened at the base cosine level. Resets between sessions.
                    # Biology: neurotransmitter depletion after repeated firing.
                    #
                    # K (fatigue resistance) scales with structural degree:
                    #   K = base / (1 + degree / scale)
                    #   Hub (30 edges): K=2.5, fatigues fast
                    #   Peripheral (3 edges): K=7.7, fatigues slow
                    #   New node (0 edges): K=10, barely fatigues
                    # The graph structure IS the fatigue signal.
                    if not hasattr(self, '_session_fatigue'):
                        # v9.2: Load fatigue from DB (persists across daemon restarts)
                        # Use the same resolved session_id that _mark_accessed will use
                        _fatigue_sid = session_id or self.session_id
                        try:
                            from .dal import SessionStateDAL
                            if _fatigue_sid and self.logs_conn:
                                self._session_fatigue = SessionStateDAL(self.logs_conn).load_fatigue(_fatigue_sid)
                            else:
                                self._session_fatigue = {}
                        except Exception:
                            self._session_fatigue = {}
                        # Store resolved sid for _mark_accessed consistency
                        self._fatigue_session_id = _fatigue_sid
                    if not hasattr(self, '_structural_degree_cache'):
                        # Cache structural degree per node — recomputed once per session
                        self._structural_degree_cache = {}
                        try:
                            for _row in self.conn.execute(
                                "SELECT source_id, COUNT(*) FROM edges "
                                "WHERE edge_type NOT IN ('co_accessed','emergent_bridge') "
                                "GROUP BY source_id"):
                                self._structural_degree_cache[_row[0]] = \
                                    self._structural_degree_cache.get(_row[0], 0) + _row[1]
                        except Exception as _e:
                            self._log_error('fatigue_degree_cache', _e, 'building structural degree cache')

                    _fatigue_count = self._session_fatigue.get(node_id, 0)
                    if _fatigue_count > 0:
                        _degree = self._structural_degree_cache.get(node_id, 0)
                        _K = 10.0 / (1.0 + _degree / 10.0)
                        _fatigue = _fatigue_count / (_fatigue_count + _K)
                        sim *= (1.0 - _fatigue)

                    embedding_scores[node_id] = sim
                    primary_scores[node_id] = sim
                    enrichment_hits[node_id] = 'primary'
                    nodes_with_embeddings += 1

            # v10: STEP 3.5: Z-weighted multi-vector scoring
            # Each node has 2-4 group vectors (title, high_meta, other_meta) in
            # node_enrichments, plus the primary (blend) in node_embeddings.
            # Old enrichment types (question, anchor, bridge, keywords) still exist
            # and participate with the other_meta weight.
            #
            # Scoring: weight × cosine_sim per vector, average top 2.
            # Requires 2 vectors to agree — prevents noisy single-field matches.
            # Tested 2026-04-02: +20pts R@8, +22pts R@25 vs old enrichment cap.
            # See pipeline_contract.EMBEDDING_GROUPS for weight definitions.
            try:
                from .pipeline_contract import get_group_weight, EMBEDDING_GROUPS
            except ImportError:
                from pipeline_contract import get_group_weight, EMBEDDING_GROUPS

            # Group vector types that get their own weight (from contract)
            _known_group_types = {g['vector_type'] for g in EMBEDDING_GROUPS.values()
                                  if g.get('vector_type') != '_primary'}
            _other_meta_weight = EMBEDDING_GROUPS['other_meta']['weight']
            _blend_weight = EMBEDDING_GROUPS['blend']['weight']

            enrichment_count = 0
            enrichment_used = 0
            # Collect ALL weighted scores per node: [(weighted_score, vector_type), ...]
            node_vector_scores = {}  # node_id → list of (weighted_sim, type)

            # Primary (blend) scores — already computed, add with blend weight
            for nid, prim_sim in primary_scores.items():
                if nid not in node_vector_scores:
                    node_vector_scores[nid] = []
                node_vector_scores[nid].append((_blend_weight * prim_sim, '_primary'))

            try:
                _enrich_dal = EnrichmentDAL(self.conn)
                _enrich_rows = _enrich_dal.get_all_embeddings()
                for erow in _enrich_rows:
                    e_node_id = erow['node_id']
                    e_type = erow['vector_type']
                    e_blob = erow['embedding']
                    if not e_blob:
                        continue
                    enrichment_count += 1
                    # Skip if node not in our filtered set
                    if e_node_id not in node_types:
                        continue

                    e_sim = embedder.cosine_similarity(query_vec, e_blob)

                    # Get z-index weight: known group types get their contract weight,
                    # old enrichment types (question, anchor, etc.) get other_meta weight
                    if e_type in _known_group_types:
                        weight = get_group_weight(e_type)
                    else:
                        weight = _other_meta_weight

                    if e_node_id not in node_vector_scores:
                        node_vector_scores[e_node_id] = []
                    node_vector_scores[e_node_id].append((weight * e_sim, e_type))

                # Z-weighted top2-avg: for each node, sort weighted scores, avg top 2
                for nid, scores in node_vector_scores.items():
                    if len(scores) < 1:
                        continue
                    scores.sort(reverse=True)
                    if len(scores) >= 2:
                        final = (scores[0][0] + scores[1][0]) / 2
                    else:
                        final = scores[0][0]

                    # Update embedding_scores if this beats the current
                    if final > embedding_scores.get(nid, 0):
                        embedding_scores[nid] = final
                        enrichment_hits[nid] = scores[0][1]  # best vector type
                        enrichment_used += 1

            except Exception as e:
                if 'no such table' not in str(e):
                    self._log_error("recall_enrichment_scan", e, "STEP 3.5 z-weighted scoring")

            # v9.2: Apply fatigue AFTER z-weighted scoring (not before).
            # Fatigue was previously applied to primary sim only, but STEP 3.5
            # could overwrite with unfatigued z-weighted scores. Now fatigue
            # dampens the final embedding score regardless of which vector won.
            if hasattr(self, '_session_fatigue') and self._session_fatigue:
                for nid in list(embedding_scores.keys()):
                    _fc = self._session_fatigue.get(nid, 0)
                    if _fc > 0:
                        _deg = self._structural_degree_cache.get(nid, 0) if hasattr(self, '_structural_degree_cache') else 0
                        _K = 10.0 / (1.0 + _deg / 10.0)
                        _fat = _fc / (_fc + _K)
                        embedding_scores[nid] *= (1.0 - _fat)

        except Exception as e:
            print(f'[brain] Embedding scan error: {e}', file=sys.stderr)

        # STEP 3.5: Situation scan — boost nodes whose situation matches current context
        situation_scores = {}
        if situation_vec:
            try:
                emb_dal = EmbeddingDAL(self.conn)
                sit_rows = emb_dal.get_all_situations()
                for row in sit_rows:
                    nid = row['node_id']
                    svec = struct.unpack('%df' % (len(row['situation_embedding']) // 4),
                                         row['situation_embedding'])
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(situation_vec, svec))
                    mag_a = sum(a * a for a in situation_vec) ** 0.5
                    mag_b = sum(b * b for b in svec) ** 0.5
                    if mag_a > 0 and mag_b > 0:
                        sim = dot / (mag_a * mag_b)
                        if sim >= SITUATION_THRESHOLD:
                            situation_scores[nid] = sim
            except Exception as e:
                self._log_error("recall_situation_scan", e, "situation scan failed")

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

        # STEP 4.5: FTS5 independent candidate net
        # FTS5 catches nodes where words match but embeddings didn't connect.
        # These go to the judge as fts5_only candidates (no blended score needed).
        fts5_only_ids = set()
        fts5_all_ids = set()
        try:
            fts5_dal = Fts5DAL(self.conn)
            fts5_hits = fts5_dal.search(query, FTS5_SEARCH_LIMIT)
            fts5_all_ids = set(fts5_hits)
            for nid in fts5_hits:
                if nid not in embedding_scores and nid not in keyword_scores:
                    fts5_only_ids.add(nid)
                    if len(fts5_only_ids) >= FTS5_CANDIDATE_LIMIT:
                        break
        except Exception as e:
            self._log_error('recall_fts5', e, 'FTS5 candidate search')

        # STEP 5: Build unified candidate set (all nodes seen by any path)
        all_candidate_ids = set(embedding_scores.keys()) | set(keyword_scores.keys()) | fts5_only_ids

        # STEP 6: Score each candidate — embeddings primary, keywords fallback
        scored_results = []
        for nid in all_candidate_ids:
            emb_score = embedding_scores.get(nid, 0)
            kw_score = keyword_scores.get(nid, 0)

            # Determine discovery source and compute blended score
            if nid in fts5_only_ids:
                # v9: FTS5-only — word match, no embedding match.
                # Passthrough score above noise floor. Judge decides relevance.
                blended = FTS5_PASSTHROUGH_SCORE
                discovery = 'fts5_only'
            elif emb_score > 0 and kw_score > 0:
                # Both signals available — blend with embeddings primary
                blended = (EMBEDDING_PRIMARY_WEIGHT * emb_score +
                          KEYWORD_FALLBACK_WEIGHT * kw_score)
                discovery = 'embedding+keyword'
                # v9: Tag 'both' if also found by FTS5
                if nid in fts5_all_ids:
                    discovery = 'both'
            elif emb_score > 0:
                # Embedding only — use embedding score directly
                blended = emb_score
                discovery = 'embedding_only'
                # v9: Tag 'both' if also found by FTS5
                if nid in fts5_all_ids:
                    discovery = 'both'
            else:
                # Keyword only (node has no embedding) — use keyword but PENALIZE.
                # Keyword-only results lack the primary signal. They should never
                # outrank a strong embedding match. Scale by KEYWORD_FALLBACK_WEIGHT
                # so a perfect keyword match (1.0) scores at most 0.10.
                blended = KEYWORD_FALLBACK_WEIGHT * kw_score
                discovery = 'keyword_only_fallback'

            # Apply intent-based type boosting
            node = keyword_nodes.get(nid)
            if node:
                type_boost = type_boosts.get(node.get('type'), 1.0)
                blended *= type_boost

            # v8.8: Title-match boost — proportional to query/title word overlap.
            # If query terms appear in the node's title, strong relevance signal.
            from .brain_constants import TITLE_MATCH_BOOST
            title = node_titles.get(nid, '').lower()
            if title and _query_terms_set:
                matched = sum(1 for t in _query_terms_set if t in title)
                title_fraction = matched / len(_query_terms_set)
                if title_fraction > 0:
                    blended += title_fraction * TITLE_MATCH_BOOST

            # v5.2: Critical node boost — safety-important nodes always surface
            is_critical = node_critical.get(nid, 0) if 'node_critical' in dir() else 0
            if not is_critical and node:
                is_critical = node.get('critical', 0)
            if is_critical:
                blended *= CRITICAL_BOOST

            # v4 FIX: Contextual qualifier penalty — nodes marked 'contextual' with
            # a personal_context that doesn't overlap query terms get penalized.
            _context_mismatch = False
            node_personal_pair = node_personal_data.get(nid)
            if not node_personal_pair and node:
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

            # Situation boost — additive, never subtractive
            sit_score = situation_scores.get(nid, 0)
            if sit_score > 0:
                blended += SITUATION_WEIGHT * sit_score

            # v10: unified_score integration DEFERRED.
            # Testing showed that applying modulator formula to the embedding path
            # regresses R@8 by -10pts because the modulators dampen scores that were
            # previously passing the relevance floor. The z-weighted top2-avg embedding
            # groups (STEP 3.5) already provide the R@25 improvement.
            #
            # Next step: investigate why the frequency penalty and hub dampening
            # cause regressions — likely need per-query-type adaptive weights
            # rather than one fixed formula.
            #
            # The recall_scoring.py module is ready but not wired in yet.
            # Data collection (node_created_at, node_emotion, node_access_count)
            # is in place for when we're ready to integrate.

            # Minimum threshold — don't return noise
            # v9: Raised from 0.05 to NOISE_FLOOR_THRESHOLD (0.15)
            min_threshold = CRITICAL_SIMILARITY_THRESHOLD if is_critical else NOISE_FLOOR_THRESHOLD
            if blended < min_threshold:
                continue

            scored_results.append({
                'node_id': nid,
                'blended_score': blended,
                'embedding_similarity': round(emb_score * 1000) / 1000 if emb_score else None,
                'keyword_score': round(kw_score * 1000) / 1000 if kw_score else None,
                '_source': discovery,
                '_context_mismatch': _context_mismatch,
            })

        # Sort by blended score descending
        scored_results.sort(key=lambda x: -x['blended_score'])
        scored_results = scored_results[:limit]

        # STEP 6.5: Graph traversal MOVED to Layer 3 (post-judge).
        # Previously: traversed from top-5 cosine results (often hubs).
        # Now: traversal happens in pre_response_recall.py AFTER the judge
        # selects relevant nodes. The judge's selections are the right seeds.
        # The _traverse_graph() function still exists for Layer 3 to call
        # via the 'graph_expand' daemon command.
        graph_neighborhoods = {}
        try:
            pass  # Traversal deferred to Layer 3
        except Exception as e:
            self._log_error("recall", e, "STEP 6.5 graph traversal")

        # STEP 6.9: Per-result relevance floor.
        # v8.7: Changed from all-or-nothing (top result gates everything) to per-result.
        # Each result must meet its own floor based on how it was discovered.
        # v9: FTS5-only candidates bypass the relevance floor — they go straight to judge.
        scored_results = [
            sr for sr in scored_results
            if sr['_source'] == 'fts5_only'  # FTS5-only: always pass to judge
            or sr['blended_score'] >= (
                RELEVANCE_FLOOR_ENRICHED
                if enrichment_hits.get(sr['node_id'], 'primary') != 'primary'
                else RELEVANCE_FLOOR_PRIMARY
            )
        ]

        # STEP 7: Hydrate full node data for top results
        # v8.8: Vocab nodes go to separate list — they're connectors, not primary results
        final_results = []
        vocab_context = []
        for sr in scored_results:
            nid = sr['node_id']
            node = keyword_nodes.get(nid)
            if not node:
                # Node came from embedding-only path — fetch from DB via DAL
                try:
                    _node_dal = NodeDAL(self.conn)
                    node = _node_dal.get_node(nid)
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
                is_engineering = ntype in ('purpose', 'mechanism', 'impact', 'concept')  # v5 engineering

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
                    except Exception as e:
                        self._log_error('freshness_parse', e, 'parsing created_at for freshness classification')

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

                # v9: All nodes are primary results (vocab→concept migration complete)
                final_results.append(node)

        # STEP 7.5: Enrich top 3 results with metadata + neighbors
        # All results keep full content (no truncation). Top 3 also get
        # metadata and neighbor context for richer understanding.
        self._enrich_results(final_results[:3])

        # STEP 8: Mark accessed (for Hebbian learning)
        # v9.2: Use the same resolved session_id as fatigue (consistency)
        # v9.2: Unified session_id — single source, no timestamp fallbacks
        sid = getattr(self, '_fatigue_session_id', None) or session_id or self.session_id
        for node in final_results:
            try:
                self._mark_accessed(node['id'], sid)
            except Exception as _e:
                self._log_error("recall", _e, "marking node as accessed for Hebbian learning")

        # STEP 9: Log recall to recall_log (single source of truth)
        recall_ms = (time.time() - t0) * 1000
        # recall_log writes REMOVED 2026-04-05 — S1 traces capture all recall data.
        # Previously inserted into recall_log here. Traces (O/K/Δ) in daemon_hooks
        # are the single source of truth for recall events.

        # Build result
        result = {
            'results': final_results,
            'vocab_context': vocab_context,  # v8.8: vocab nodes as connectors, not results
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
                    'fts5_only': sum(1 for r in final_results if r.get('_discovery') == 'fts5_only'),
                    'both': sum(1 for r in final_results if r.get('_discovery') == 'both'),
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
            # v9: Retrieval stats for judge — distribution-aware context
            '_retrieval_stats': {
                'brain_size': _brain_size,
                'candidates_after_floor': len(scored_results) if 'scored_results' in dir() else 0,
                'top_score': round(scored_results[0]['blended_score'], 3) if scored_results else 0,
                'median_score': round(scored_results[len(scored_results)//2]['blended_score'], 3) if scored_results else 0,
                'source_breakdown': {
                    'embedding_only': sum(1 for sr in scored_results if sr['_source'] == 'embedding_only'),
                    'embedding+keyword': sum(1 for sr in scored_results if sr['_source'] == 'embedding+keyword'),
                    'fts5_only': sum(1 for sr in scored_results if sr['_source'] == 'fts5_only'),
                    'both': sum(1 for sr in scored_results if sr['_source'] == 'both'),
                    'keyword_only_fallback': sum(1 for sr in scored_results if sr['_source'] == 'keyword_only_fallback'),
                } if 'scored_results' in dir() else {},
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

            # Attach metadata from KV store
            try:
                from .dal_metadata import MetadataDAL
                _meta_dal = MetadataDAL(self.conn)
                meta = _meta_dal.get(nid)
                if meta:
                    node['_metadata'] = meta
            except Exception as e:
                self._log_error('recall_node_meta', e, 'fetching node metadata for enrichment')

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
                except Exception as e:
                    self._log_error('recall_node_neighbors', e, 'fetching neighbor context')
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
            except Exception as e:
                self._log_error('freshness_multiplier', e, 'parsing timestamp for freshness multiplier')
                return FRESHNESS_MULTIPLIERS.get('older', 0.6)
            if hours < 24:
                return FRESHNESS_MULTIPLIERS.get('today', 1.2)
            elif hours < 168:
                return FRESHNESS_MULTIPLIERS.get('this_week', 1.0)
            elif hours < 720:
                return FRESHNESS_MULTIPLIERS.get('this_month', 0.8)
            return FRESHNESS_MULTIPLIERS.get('older', 0.6)

        emb_dal = EmbeddingDAL(self.conn)
        def _semantic_bonus(node_id):
            """Compute additive semantic bonus if query_vec available."""
            if query_vec is None:
                return 0.0
            try:
                blob = emb_dal.get_embedding(node_id)
                if blob:
                    node_vec = struct.unpack('%df' % (len(blob) // 4), blob)
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(query_vec, node_vec))
                    mag_a = sum(a * a for a in query_vec) ** 0.5
                    mag_b = sum(b * b for b in node_vec) ** 0.5
                    if mag_a > 0 and mag_b > 0:
                        sim = dot / (mag_a * mag_b)
                        if sim >= TRAVERSE_SEMANTIC_THRESHOLD:
                            return TRAVERSE_SEMANTIC_BONUS * sim
            except Exception as e:
                self._log_error('traverse_semantic', e, 'computing semantic bonus for graph traversal')
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
            # Filter to intentional at degree 1, skip vocab nodes (they're hubs, not destinations)
            d1_neighbors = [n for n in d1_neighbors
                            if n.get('relation') in INTENTIONAL_EDGE_TYPES
                            and n.get('type') != 'concept'][:TRAVERSE_LIMITS[0]]

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

                    for nb2 in [n for n in d2_neighbors if n.get('type') != 'concept']:
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

                            for nb3 in [n for n in d3_neighbors if n.get('type') != 'concept']:
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
                _gdal = GraphDAL(self.conn)
                _neighbors = _gdal.get_neighbors(node_id, min_weight=PRUNE_THRESHOLD, limit=MAX_NEIGHBORS)

                for nb in _neighbors:
                    target_id = nb['target_id']
                    edge_weight = nb['weight']
                    spread = node_activation * edge_weight * decay_factor
                    current_act = activation.get(target_id, 0)
                    activation[target_id] = current_act + spread

        # Fetch full node data
        results = []
        for node_id, act in activation.items():
            node = node_cache.get(node_id)
            if not node:
                _ndal = NodeDAL(self.conn)
                node = _ndal.get_node(node_id)
                if node:
                    node_cache[node_id] = node

            if node:
                # Type filter
                if types and node.get('type') not in types:
                    continue

                results.append({**node, 'spread_activation': act})

        return results

    def _search_keywords(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search nodes by full-text search (FTS5).

        v9: Replaced LIKE-based search with FTS5. Porter stemming,
        BM25 ranking, title weighted 10x. TF-IDF scoring layer
        (_keyword_recall) stays unchanged — it scores, FTS5 finds.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching nodes
        """
        try:
            fts5_dal = Fts5DAL(self.conn)
            node_ids = fts5_dal.search(query, limit)
            _ndal = NodeDAL(self.conn)
            results = []
            for nid in node_ids:
                node = _ndal.get_node(nid)
                if node and not node.get('archived'):
                    results.append(node)
            return results
        except Exception as e:
            self._log_error('search_keywords_fts5', e, 'FTS5 search failed, falling back to empty')
            return []

    def _mark_accessed(self, node_id: str, session_id: str):
        """Mark a node as accessed, log it, and increment synaptic fatigue."""
        node_dal = NodeDAL(self.conn)
        logs_dal = LogsDAL(self.logs_conn)
        node_dal.mark_accessed(node_id)
        logs_dal.log_access(session_id, node_id)
        self.conn.commit()
        self.logs_conn.commit()

        # Increment session fatigue counter — next recall will dampen this node's cosine
        # v9.2: Also persist to session_state DB (survives daemon restart)
        if not hasattr(self, '_session_fatigue'):
            self._session_fatigue = {}
        self._session_fatigue[node_id] = self._session_fatigue.get(node_id, 0) + 1
        try:
            from .dal import SessionStateDAL
            SessionStateDAL(self.logs_conn).increment(session_id, 'fatigue', node_id)
        except Exception:
            pass  # In-memory still works if DB write fails

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

        graph_dal = GraphDAL(self.conn)

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                nid_i = ids[i]
                nid_j = ids[j]

                # Check if edge exists (either direction) — strengthen via DAL
                if graph_dal.strengthen_edge(nid_i, nid_j, amount=LEARNING_RATE * 0.1):
                    continue
                if graph_dal.strengthen_edge(nid_j, nid_i, amount=LEARNING_RATE * 0.1):
                    continue
                else:
                    # NO edge — create co_accessed (v5.1: only within same segment)
                    if segment_set and (nid_i not in segment_set or nid_j not in segment_set):
                        continue
                    graph_dal.create_edge(
                        nid_i, nid_j,
                        weight=EDGE_TYPES['co_accessed']['defaultWeight'],
                        relation='co_accessed')

        self.conn.commit()

    def _log_recall(self, session_id: str, query: str, returned_ids: List[str]) -> Optional[str]:
        """Log a recall event."""
        logs_dal = LogsDAL(self.logs_conn)
        row_id = logs_dal.insert_recall_log(
            session_id=session_id or 'unknown',
            query=query,
            returned_ids=json.dumps(returned_ids),
            returned_count=len(returned_ids),
            embeddings_used=0,
            recalled_titles='',
            recalled_snippets='',
            created_at=self.now())
        return str(row_id) if row_id else None

    def _get_recent(self, limit: int = 20, types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get recently accessed nodes."""
        sql = 'SELECT id FROM nodes WHERE archived = 0'
        params = []
        if types:
            sql += ' AND type IN (%s)' % ','.join('?' * len(types))
            params.extend(types)
        sql += ' ORDER BY last_accessed DESC LIMIT ?'
        params.append(limit)

        _ndal = NodeDAL(self.conn)
        results = []
        for row in self.conn.execute(sql, params).fetchall():
            node = _ndal.get_node(row[0])
            if node:
                node['spread_activation'] = node.get('activation', 0)
                node['effective_activation'] = node.get('activation', 0)
                results.append(node)
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
