"""
brain — BrainDreams Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import math
import random
import time
from .brain_constants import (
    DREAM_COUNT,
    DREAM_MIN_NOVELTY,
    DREAM_WALK_LENGTH,
    STABILITY_BOOST,
)


class BrainDreamsMixin:
    """Dreams methods for Brain."""

    def dream(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Spontaneous association: random walks through graph to find interesting connections.
        Seed selection weighted by: connection_count × (1 + emotion) × recency_multiplier.
        Returns dict with dreams, surfaced_insights, bridge_proposals.
        """
        ts = self.now()
        if session_id is None:
            session_id = f'dream_{int(time.time() * 1000)}'

        dreams = []

        # Get pool of candidate seeds (40 random unlocked nodes + some locked engineering)
        seed_pool = self.conn.execute('''
            SELECT n.id, n.emotion, n.last_accessed,
                   (SELECT COUNT(*) FROM edges e WHERE e.source_id = n.id AND e.weight >= 0.1) as edge_count
            FROM nodes n
            WHERE n.archived = 0 AND n.locked = 0
            ORDER BY RANDOM()
            LIMIT 35
        ''').fetchall()

        # v5: Include some locked engineering/vocabulary nodes as dream seeds
        # Dreams that traverse engineering memory can surface unexpected connections
        eng_seeds = self.conn.execute('''
            SELECT n.id, n.emotion, n.last_accessed,
                   (SELECT COUNT(*) FROM edges e WHERE e.source_id = n.id AND e.weight >= 0.1) as edge_count
            FROM nodes n
            WHERE n.archived = 0 AND n.locked = 1
              AND n.type IN ('vocabulary', 'purpose', 'mechanism', 'impact',
                             'lesson', 'mental_model')
            ORDER BY RANDOM()
            LIMIT 5
        ''').fetchall()
        seed_pool = list(seed_pool) + list(eng_seeds)

        if len(seed_pool) < 2:
            return {'dreams': [], 'message': 'Not enough nodes to dream'}

        # Weighted seed selection
        now = time.time() * 1000  # milliseconds
        seed_candidates = []
        for r in seed_pool:
            node_id, emotion, last_accessed, edge_count = r
            emotion = emotion or 0
            edge_count = edge_count or 0

            # Hours since last access → recency multiplier
            if last_accessed:
                try:
                    last_ts = datetime.fromisoformat(last_accessed.replace('Z', '+00:00')).timestamp() * 1000
                except:
                    last_ts = 0
                hours_ago = max(0, (now - last_ts) / (1000 * 60 * 60))
            else:
                hours_ago = 720  # never accessed = 1 month

            recency_boost = 1.0 + 1.0 / (1 + hours_ago / 24)

            weight = math.sqrt(edge_count + 1) * (1 + emotion * 0.5) * recency_boost
            seed_candidates.append({
                'id': node_id,
                'emotion': emotion,
                'edgeCount': edge_count,
                'recencyBoost': round(recency_boost * 100) / 100,
                'weight': weight
            })

        total_seed_weight = sum(c['weight'] for c in seed_candidates)
        if total_seed_weight <= 0:
            return {'dreams': [], 'message': 'Seed candidates have no weight'}

        def pick_weighted_seed(exclude=None):
            roll = random.random() * total_seed_weight
            for c in seed_candidates:
                if c['id'] == exclude:
                    continue
                roll -= c['weight']
                if roll <= 0:
                    return c
            # Fallback
            for c in seed_candidates:
                if c['id'] != exclude:
                    return c
            return seed_candidates[0]

        # Generate dreams (tunable count)
        dream_params = self._get_tunable('dream_params', {
            'count': DREAM_COUNT, 'walk_length': DREAM_WALK_LENGTH, 'min_novelty': DREAM_MIN_NOVELTY
        })
        dream_count = dream_params.get('count', DREAM_COUNT) if isinstance(dream_params, dict) else DREAM_COUNT
        dream_walk_len = dream_params.get('walk_length', DREAM_WALK_LENGTH) if isinstance(dream_params, dict) else DREAM_WALK_LENGTH
        for d in range(dream_count):
            seed_a_data = pick_weighted_seed()
            seed_a = seed_a_data['id']
            seed_b_data = pick_weighted_seed(seed_a)
            seed_b = seed_b_data['id']
            if seed_b == seed_a:
                continue

            # Random walks
            walk_a = self._random_walk(seed_a, dream_walk_len)
            walk_b = self._random_walk(seed_b, dream_walk_len)

            # Endpoints
            end_a = walk_a[-1] if walk_a else seed_a
            end_b = walk_b[-1] if walk_b else seed_b

            # Get titles
            node_a = self._get_node_title(seed_a)
            node_end_a = self._get_node_title(end_a)
            node_b = self._get_node_title(seed_b)
            node_end_b = self._get_node_title(end_b)

            insight = f'Association: "{node_a}" → "{node_end_a}" | "{node_b}" → "{node_end_b}"'

            # Check if edge exists between endpoints
            existing_edge = self.conn.execute(
                'SELECT weight FROM edges WHERE source_id = ? AND target_id = ?',
                (end_a, end_b)
            ).fetchone()

            intuition_id = None
            if not existing_edge and end_a != end_b:
                # Create intuition node
                result = self.remember(
                    type='intuition',
                    title=f'Dream: {node_end_a[:40] if node_end_a else "?"} ↔ {node_end_b[:40] if node_end_b else "?"}',
                    content=insight,
                    keywords=f'dream intuition association {self._extract_keywords(node_end_a + " " + node_end_b)}',
                    locked=False,
                    emotion=0.2,
                    emotion_label='curiosity',
                    emotion_source='dream',
                    connections=[
                        {'target_id': end_a, 'relation': 'dreamed_from', 'weight': 0.3},
                        {'target_id': end_b, 'relation': 'dreamed_from', 'weight': 0.3}
                    ]
                )
                intuition_id = result['id']

            # Log dream
            self.logs_conn.execute('''
                INSERT INTO dream_log (session_id, intuition_node_id, seed_nodes, walk_path, insight, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, intuition_id, json.dumps([seed_a, seed_b]),
                  json.dumps([walk_a, walk_b]), insight, ts))

            # Score dream for interestingness
            interest_score = 0

            # Recency check
            try:
                recency_check = self.conn.execute('''
                    SELECT MAX(CASE WHEN julianday('now') - julianday(last_accessed) < 1 THEN 3
                                    WHEN julianday('now') - julianday(last_accessed) < 3 THEN 2
                                    WHEN julianday('now') - julianday(last_accessed) < 7 THEN 1
                                    ELSE 0 END) as recency
                    FROM nodes WHERE id IN (?, ?) AND archived = 0
                ''', (end_a, end_b)).fetchone()
                if recency_check and recency_check[0]:
                    interest_score += recency_check[0]
            except:
                pass

            # Uniqueness check
            try:
                type_check = self.conn.execute(
                    'SELECT type, project FROM nodes WHERE id IN (?, ?)',
                    (end_a, end_b)
                ).fetchall()
                if len(type_check) >= 2:
                    type_a, proj_a = type_check[0][0], type_check[0][1]
                    type_b, proj_b = type_check[1][0], type_check[1][1]
                    if type_a != type_b:
                        interest_score += 1
                    if proj_a and proj_b and proj_a != proj_b:
                        interest_score += 2
            except:
                pass

            # Emotion check
            try:
                emotion_check = self.conn.execute(
                    'SELECT MAX(emotion) FROM nodes WHERE id IN (?, ?)',
                    (end_a, end_b)
                ).fetchone()
                if emotion_check and emotion_check[0]:
                    max_emo = emotion_check[0]
                    if max_emo > 0.5:
                        interest_score += 2
                    elif max_emo > 0.3:
                        interest_score += 1
            except:
                pass

            # Structural distance
            interest_score += min(2, (len(walk_a) + len(walk_b)) // 4)

            should_surface = interest_score >= 4

            # High-scoring dreams spawn a thought
            dream_thought = None
            if interest_score >= 6 and not any(d.get('_hasThought') for d in dreams):
                try:
                    dream_thought = self._spawn_thought(
                        f'Dream connection: "{node_end_a[:50] if node_end_a else "?"}" and "{node_end_b[:50] if node_end_b else "?"}" — found via random walks from different graph regions. Score {interest_score}. Worth investigating.',
                        [end_a, end_b] if end_a and end_b else [],
                        'dream_observation'
                    )
                except:
                    pass

            dreams.append({
                'seeds': [node_a, node_b],
                'endpoints': [node_end_a, node_end_b],
                'insight': insight,
                'intuition_id': intuition_id,
                'walk_lengths': [len(walk_a), len(walk_b)],
                'interest_score': interest_score,
                'surface': should_surface,
                'thought_id': dream_thought['id'] if dream_thought else None,
                '_hasThought': bool(dream_thought)
            })

        # Dream bridge proposals
        bridge_proposals = 0
        max_dream_proposals = 3
        try:
            recent_dreams = self.logs_conn.execute('''
                SELECT walk_path FROM dream_log WHERE session_id = ? ORDER BY created_at DESC LIMIT ?
            ''', (session_id, DREAM_COUNT)).fetchall()
            for (walk_json,) in recent_dreams:
                if bridge_proposals >= max_dream_proposals:
                    break
                try:
                    walks = json.loads(walk_json)
                    if len(walks) >= 2:
                        walk_a, walk_b = walks[0], walks[1]
                        end_a = walk_a[-1] if walk_a else None
                        end_b = walk_b[-1] if walk_b else None
                        if end_a and end_b and end_a != end_b:
                            shared_context = 'Dream walk convergence: random walks from different regions met here'
                            proposal = self._propose_bridge(end_a, end_b, shared_context, session_id)
                            if proposal:
                                bridge_proposals += 1
                except:
                    pass
        except:
            pass

        # Surfaced insights
        surfaced_insights = [
            {
                'insight': d['insight'],
                'score': d['interest_score'],
                'endpoints': d['endpoints'],
                'intuition_id': d['intuition_id']
            }
            for d in dreams if d['surface']
        ]

        return {
            'dreams': dreams,
            'count': len(dreams),
            'bridge_proposals': bridge_proposals,
            'surfaced_insights': surfaced_insights
        }

    def _spawn_thought(self, content: str, trigger_ids: Optional[List[str]] = None,
                      reason: str = 'prompted_by') -> Dict[str, Any]:
        """
        Create a thought node and connect it to triggers.
        Emotion=0.15 (low emotional charge).
        """
        if trigger_ids is None:
            trigger_ids = []

        # Create the thought
        title = content[:117] + '...' if len(content) > 120 else content
        result = self.remember(
            type='thought',
            title=title,
            content=content,
            keywords=f'thought brain-observation {self._extract_keywords(content)}',
            locked=False,
            emotion=0.15,
            emotion_label='curiosity',
            emotion_source='brain'
        )

        # Connect to triggers
        for trigger_id in trigger_ids:
            try:
                trigger_title = self._get_node_title(trigger_id) or trigger_id
                self.connect_typed(
                    result['id'], trigger_id, reason, 0.3, reason,
                    f'Brain noticed: "{title[:60]}" while processing "{trigger_title[:60]}"'
                )
            except:
                pass

        return result

    def consolidate(self) -> Dict[str, Any]:
        """
        Consolidation: boost stability, form bridges, mature bridge proposals, backfill embeddings.
        Returns dict with consolidated count, bridges_created, bridges_matured.
        """
        ts = self.now()
        stats = {'consolidated': 0}

        # Boost stability for nodes accessed 3+ times in last 24h
        candidates = self.logs_conn.execute('''
            SELECT node_id, COUNT(*) as cnt FROM access_log
            WHERE timestamp > datetime(?, '-24 hours')
            GROUP BY node_id HAVING cnt >= 3
        ''', (ts,)).fetchall()

        stab_boost = self._get_tunable('stability_boost', STABILITY_BOOST)
        for node_id, _ in candidates:
            self.conn.execute('''
                UPDATE nodes SET stability = stability * ?, activation = MIN(1.0, activation + 0.1),
                       updated_at = ? WHERE id = ? AND locked = 0
            ''', (stab_boost, ts, node_id))
            stats['consolidated'] += 1

        # Promote well-connected nodes
        well_connected = self.conn.execute('''
            SELECT source_id, SUM(weight) as total_weight, COUNT(*) as edge_count
            FROM edges WHERE weight > 0.3
            GROUP BY source_id HAVING edge_count >= 5
        ''').fetchall()

        for node_id, _, _ in well_connected:
            self.conn.execute('''
                UPDATE nodes SET stability = MAX(stability, 3.0), updated_at = ? WHERE id = ? AND locked = 0
            ''', (ts, node_id))

        # Emergent bridging at consolidation (wider scan)
        try:
            bridges = self._bridge_at_consolidation()
            stats['bridges_created'] = len(bridges)
        except:
            stats['bridges_created'] = 0

        # Mature pending bridge proposals
        try:
            matured = self._mature_bridge_proposals()
            stats['bridges_matured'] = matured
        except:
            stats['bridges_matured'] = 0

        # Fire-and-forget backfill embeddings (don't wait for async)
        try:
            # Non-blocking call to backfill
            import asyncio
            asyncio.create_task(self.backfill_embeddings(20))
            stats['embeddings_backfilled'] = 'queued'
        except:
            stats['embeddings_backfilled'] = 0

        # ── v4: Auto-discover evolutions (tensions, patterns, hypotheses, aspirations) ──
        try:
            discoveries = self.auto_discover_evolutions()
            stats['discoveries'] = {
                'tensions': len(discoveries.get('tensions', [])),
                'patterns': len(discoveries.get('patterns', [])),
                'hypotheses': len(discoveries.get('hypotheses', [])),
                'aspirations': len(discoveries.get('aspirations', [])),
                'total': discoveries.get('_stats', {}).get('total_created', 0),
            }
        except Exception:
            stats['discoveries'] = {'tensions': 0, 'patterns': 0, 'hypotheses': 0, 'aspirations': 0, 'total': 0}

        return stats
