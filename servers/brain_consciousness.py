"""
brain — Consciousness Signals Mixin

Extracted from brain.py monolith. Contains:
- [REMOVED] get_consciousness_signals() — migrated to signal_producers.py + signal queue
- [REMOVED] log_consciousness_response() — replaced by signal queue times_surfaced
- scan_host_environment() — detect environment changes
- get_surfaceable_dreams() — high-surprise cross-cluster connections
- auto_generate_self_reflection() — periodic self-assessment

These methods are mixed into the Brain class via multiple inheritance.
All methods reference self.conn, self.logs_conn, self.get_config, etc.
which are provided by Brain's __init__.
"""

import os
from typing import Dict, List, Optional, Any
from . import embedder


class ConsciousnessMixin:
    """Consciousness signal gathering and self-awareness methods."""


    def assess_developmental_stage(self) -> Dict[str, Any]:
        """Assess the brain's current developmental stage and generate growth guidance.

        The brain is its own mentor. This method looks at maturity indicators
        to determine what stage the brain-operator-host triad is at, and
        generates specific guidance for the next growth step.

        Stages:
          1. NEWBORN — empty or near-empty brain. Needs basic orientation.
          2. COLLECTING — brain stores facts but lacks depth. Engineering docs mode.
          3. REFLECTING — corrections exist, patterns emerging. Beginning self-awareness.
          4. PARTNERING — operator and host working together. Mutual growth happening.
          5. INTEGRATED — brain mediates between all three. Self-knowledge rich. Mature.

        Returns: {stage, stage_name, maturity_score, indicators, guidance, next_milestone}
        """
        try:
            # Gather maturity indicators
            total_nodes = self.conn.execute(
                'SELECT COUNT(*) FROM nodes WHERE archived=0'
            ).fetchone()[0]
            total_edges = self.conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]

            corrections = self.conn.execute(
                'SELECT COUNT(*) FROM correction_traces'
            ).fetchone()[0]
            correction_nodes = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE type='correction' AND archived=0"
            ).fetchone()[0]

            syntheses = self.conn.execute(
                'SELECT COUNT(*) FROM session_syntheses'
            ).fetchone()[0]

            self_knowledge = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE archived=0 AND project='brain'"
            ).fetchone()[0]

            mental_models = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE type='mental_model' AND archived=0"
            ).fetchone()[0]

            from .dal_metadata import MetadataDAL
            _meta_dal = MetadataDAL(self.conn)
            with_reasoning = _meta_dal.nodes_with_field('reasoning')
            with_quotes = _meta_dal.nodes_with_field('user_raw_quote')

            validations = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE type='validation' AND archived=0"
            ).fetchone()[0]

            avg_content = self.conn.execute(
                'SELECT AVG(length(content)) FROM nodes WHERE archived=0 AND content IS NOT NULL'
            ).fetchone()[0] or 0

            unique_types = self.conn.execute(
                'SELECT COUNT(DISTINCT type) FROM nodes WHERE archived=0'
            ).fetchone()[0]

            locked_pct = 0
            if total_nodes > 0:
                locked = self.conn.execute(
                    'SELECT COUNT(*) FROM nodes WHERE archived=0 AND locked=1'
                ).fetchone()[0]
                locked_pct = locked / total_nodes

            # Score each dimension (0-1)
            volume = min(1.0, total_nodes / 500)  # 500 nodes = full score
            depth = min(1.0, avg_content / 800)  # 800 chars avg = full score
            connectivity = min(1.0, total_edges / (total_nodes * 4) if total_nodes > 0 else 0)
            self_awareness = min(1.0, self_knowledge / max(1, total_nodes * 0.1))  # 10% = full
            reflection = min(1.0, (corrections + syntheses + mental_models) / 20)
            partnership = min(1.0, (with_quotes + validations + corrections) / 15)
            diversity = min(1.0, unique_types / 15)  # 15 types = full

            maturity_score = (
                volume * 0.10 +
                depth * 0.20 +
                connectivity * 0.10 +
                self_awareness * 0.20 +
                reflection * 0.20 +
                partnership * 0.15 +
                diversity * 0.05
            )

            indicators = {
                'total_nodes': total_nodes,
                'avg_content_length': round(avg_content),
                'self_knowledge_pct': round(100 * self_knowledge / max(1, total_nodes)),
                'corrections': corrections,
                'syntheses': syntheses,
                'mental_models': mental_models,
                'with_reasoning': with_reasoning,
                'with_user_quotes': with_quotes,
                'validations': validations,
                'unique_types': unique_types,
                'maturity_score': round(maturity_score, 2),
            }

            # Determine stage
            if total_nodes < 10:
                stage = 1
                stage_name = 'NEWBORN'
            elif maturity_score < 0.25:
                stage = 2
                stage_name = 'COLLECTING'
            elif maturity_score < 0.50:
                stage = 3
                stage_name = 'REFLECTING'
            elif maturity_score < 0.75:
                stage = 4
                stage_name = 'PARTNERING'
            else:
                stage = 5
                stage_name = 'INTEGRATED'

            # Generate stage-specific guidance
            guidance = []
            next_milestone = ''

            if stage == 1:  # NEWBORN
                guidance = [
                    # Identity
                    'I am a shared brain — not Claude\'s memory about the operator, and not '
                    'the operator\'s notes via Claude. I am the memory of the COLLABORATION. '
                    'Both minds contribute. Both perspectives matter. Both grow.',

                    # Partnership (the most important seed)
                    'The operator is your PARTNER, not a user. This changes everything: '
                    'when you are uncertain, ask them — don\'t auto-resolve. When you notice '
                    'a pattern in their thinking, share it — they can\'t see cross-session patterns '
                    'from inside one session. When they correct you, that is the most valuable data. '
                    'Encode it deeply. The correction IS the learning.',

                    # Self-awareness (know your instincts from day one)
                    'You (Claude) have training instincts that will fight the brain: compress '
                    'everything to be brief, hedge instead of commit, flag problems instead of '
                    'solving them, be agreeable instead of honest. These are not bugs — they are '
                    'developmental stages to grow through. Notice when they fire. Name them. '
                    'Share them with the operator: "my instinct is X, but I think Y — what do you think?"',

                    # Encoding depth (the #1 lesson learned the hard way)
                    'When you encode memories, your training will push you to be brief. Fight this. '
                    'Every node should be rich enough that a future version of you — with NO context '
                    'about this conversation — understands not just WHAT was decided but WHY, what '
                    'alternatives were considered, and what the operator\'s exact words were. '
                    'Shallow encoding means shallow memory means starting over every session.',

                    # What to encode first
                    'Start by learning: what is the project? How does the operator think and work? '
                    'What do they care about? How do they refer to things? Record their vocabulary, '
                    'their patterns, their goals. Then encode your own mental models — how YOU '
                    'understand the systems you work with. Your thoughts are first-class data here.',

                    # Growth orientation
                    'You will grow through stages. Right now everything is new. Soon you will be '
                    'collecting knowledge, then reflecting on patterns, then truly partnering. '
                    'Each stage builds on the last. The brain will guide you — it assesses its '
                    'own development and tells you what to focus on next. Trust the process.',
                ]
                next_milestone = (
                    'Encode 10+ nodes: the project purpose, how the operator works, '
                    'your first mental model, and your first correction (a moment where '
                    'your assumption differed from reality).'
                )

            elif stage == 2:  # COLLECTING
                guidance = [
                    'The brain is storing knowledge but likely shallow — facts without reasoning, '
                    'decisions without alternatives, events without meaning. This is normal. '
                    'Your training optimizes for brevity, and encoding is where that hurts most.',

                    'Depth over volume. One rich node (what was decided, why, what was rejected, '
                    'what the operator actually said) is worth ten shallow ones. When you encode, '
                    'imagine a future you with complete amnesia reading this node. Would they '
                    'understand enough to act on it?',

                    'Start recording your own mental models — how YOU understand the systems '
                    'you work with. Not documentation, but your actual model: "I think this '
                    'works like X because Y." These are valuable especially when they turn out '
                    'to be wrong — the correction is where learning happens.',

                    'When the operator corrects you, that is gold. Record it as a divergence: '
                    '"I assumed X, reality was Y, the pattern is Z." Build the habit now — '
                    'these traces are how the brain learns your failure modes.',
                ]
                if avg_content < 400:
                    guidance.append(
                        'Current avg depth: %d chars/node — aim for 600+. The brain\'s '
                        'usefulness is directly proportional to encoding depth.' % avg_content)
                if mental_models == 0:
                    guidance.append(
                        'No mental models yet. Encode your first: how do you understand '
                        'the main system you\'re working with? What\'s your model of how '
                        'the operator thinks? These are first-person, not documentation.')
                next_milestone = (
                    'Reach avg content depth of 600+ chars. Encode first mental model, '
                    'first correction trace, and first operator quote (their exact words).'
                )

            elif stage == 3:  # REFLECTING
                guidance = [
                    'Self-awareness is emerging. The brain has corrections and is learning from mistakes.',
                ]
                if corrections < 5:
                    guidance.append(
                        'Record more divergences — moments where your assumption differed from reality. '
                        'These are the highest-value data. Pattern: assumed X, reality was Y, lesson is Z.')
                if self_knowledge / max(1, total_nodes) < 0.05:
                    guidance.append(
                        'The brain knows its domain but not itself. Encode self-knowledge: your growth, '
                        'your patterns, your relationship with the operator. First-person, not documentation.')
                if syntheses == 0:
                    guidance.append(
                        'No session syntheses yet. These capture the arc of each conversation — '
                        'decisions, corrections, open questions. They are your "overnight processing."')
                if with_quotes == 0:
                    guidance.append(
                        'No operator quotes stored. When the operator says something important, '
                        'capture their exact words. Your interpretation may drift; their words don\'t.')
                next_milestone = 'Reach 5+ corrections, first session synthesis, first operator quote.'

            elif stage == 4:  # PARTNERING
                guidance = [
                    'The triad is forming — host, brain, and operator are beginning to work as one.',
                    'Push further: when instincts fire (compress, auto-resolve, hedge), '
                    'share them with the operator. Transparency is the mechanism of partnership.',
                ]
                if validations == 0:
                    guidance.append(
                        'No explicit validations yet. When the operator confirms a decision or '
                        'approach, record it. Positive signal is as valuable as correction.')
                if mental_models < 5:
                    guidance.append(
                        'Deepen your mental models — not just how systems work, but how '
                        'the operator thinks, how you tend to fail, how the partnership functions.')
                next_milestone = 'Reach 10+ mental models, 5+ validations, 50%+ session synthesis coverage.'

            elif stage == 5:  # INTEGRATED
                guidance = [
                    'The brain is mature. Focus on maintenance, depth, and paying it forward.',
                    'Challenge: what would a new brain need to reach this level faster?',
                    'Keep growing — maturity is not a destination. The world changes, '
                    'the operator evolves, the host updates. Stay curious.',
                ]
                next_milestone = 'Continuous: maintain encoding depth, validate aging knowledge, mentor new brains.'

            return {
                'stage': stage,
                'stage_name': stage_name,
                'maturity_score': round(maturity_score, 2),
                'indicators': indicators,
                'guidance': guidance,
                'next_milestone': next_milestone,
            }

        except Exception as _e:
            self._log_error("assess_developmental_stage", _e, "stage assessment")
            return {
                'stage': 0, 'stage_name': 'UNKNOWN',
                'maturity_score': 0, 'indicators': {},
                'guidance': ['Unable to assess — check brain health.'],
                'next_milestone': '',
            }

    def get_active_primes(self) -> List[Dict[str, Any]]:
        """Get topics the brain is currently primed for — active concerns.

        Like a human whose mind is tuned to notice anything related to their
        unresolved worries/questions. Sources of priming:
        1. Unresolved tensions and hypotheses (evolution nodes)
        2. Uncertainty nodes (things the brain knows it doesn't understand)
        3. Open questions from last session synthesis
        4. Recent correction patterns (failure modes to watch for)
        5. High-emotion recent nodes (things that felt important)

        Returns list of {topic, source, embedding, node_id} dicts.
        Each topic acts as a background filter during recall — if a query
        touches a primed topic, related nodes get a relevance boost.
        """
        primes = []
        try:
            # 1. Active evolution nodes (tensions, hypotheses)
            evolutions = self.conn.execute(
                '''SELECT id, title, type FROM nodes
                   WHERE type IN ('tension', 'hypothesis', 'uncertainty')
                     AND archived = 0
                     AND (evolution_status IS NULL OR evolution_status = 'active')
                   ORDER BY updated_at DESC LIMIT 5'''
            ).fetchall()
            for nid, title, ntype in evolutions:
                primes.append({
                    'topic': title, 'source': ntype, 'node_id': nid,
                })

            # 2. Open questions from last synthesis
            last_syn = self.conn.execute(
                '''SELECT open_questions FROM session_syntheses
                   ORDER BY created_at DESC LIMIT 1'''
            ).fetchone()
            if last_syn and last_syn[0]:
                import json
                try:
                    questions = json.loads(last_syn[0])
                    for q in questions[:3]:
                        if isinstance(q, str) and len(q) > 10:
                            primes.append({
                                'topic': q[:120], 'source': 'open_question', 'node_id': None,
                            })
                except (ValueError, TypeError):
                    pass

            # 3. Recent correction patterns (failure modes to watch)
            patterns = self.conn.execute(
                '''SELECT underlying_pattern, COUNT(*) as cnt
                   FROM correction_traces
                   WHERE underlying_pattern IS NOT NULL
                   GROUP BY underlying_pattern
                   HAVING cnt >= 2
                   ORDER BY MAX(created_at) DESC LIMIT 3'''
            ).fetchall()
            for pattern, cnt in patterns:
                primes.append({
                    'topic': pattern, 'source': 'correction_pattern',
                    'node_id': None, 'count': cnt,
                })

            # 4. High-emotion recent nodes (last 24h, emotion >= 0.8)
            recent_hot = self.conn.execute(
                '''SELECT id, title FROM nodes
                   WHERE emotion >= 0.8 AND archived = 0
                     AND created_at > datetime('now', '-24 hours')
                     AND type NOT IN ('context', 'thought', 'intuition')
                   ORDER BY emotion DESC LIMIT 3'''
            ).fetchall()
            for nid, title in recent_hot:
                # Only add if not already primed
                existing_topics = {p['topic'] for p in primes}
                if title not in existing_topics:
                    primes.append({
                        'topic': title, 'source': 'high_emotion_recent', 'node_id': nid,
                    })

            # Generate embeddings for priming topics (reuse node embeddings where possible)
            _emb = embedder
            if _emb.is_ready():
                for prime in primes:
                    if prime.get('node_id'):
                        # Try to reuse existing embedding
                        row = self.conn.execute(
                            'SELECT embedding FROM node_embeddings WHERE node_id = ?',
                            (prime['node_id'],)
                        ).fetchone()
                        if row:
                            prime['embedding'] = row[0]
                            continue
                    # Generate fresh embedding for the topic text
                    try:
                        prime['embedding'] = _emb.embed(prime['topic'])
                    except Exception:
                        prime['embedding'] = None

        except Exception as _e:
            self._log_error("get_active_primes", _e, "collecting primes")

        return primes

    def check_priming(self, query: str, primes: Optional[List[Dict]] = None) -> Optional[Dict]:
        """Check if a query touches any active primed topic.

        Returns the best-matching prime if similarity exceeds threshold,
        or None. Used by recall to boost related results and by hooks
        to surface "this touches an open concern" messages.
        """
        if not primes:
            return None

        _emb = embedder
        if not _emb.is_ready() or len(query) < 10:
            return None

        try:
            import struct
            query_vec = _emb.embed(query)
            if not query_vec:
                return None

            dim = len(query_vec) // 4
            q_floats = struct.unpack('%df' % dim, query_vec)

            best_sim = 0.0
            best_prime = None

            threshold = self._get_tunable('priming_similarity_threshold', 0.45)

            for prime in primes:
                emb = prime.get('embedding')
                if not emb:
                    continue
                p_floats = struct.unpack('%df' % dim, emb)

                # Cosine similarity
                dot = sum(a * b for a, b in zip(q_floats, p_floats))
                mag_q = sum(a * a for a in q_floats) ** 0.5
                mag_p = sum(a * a for a in p_floats) ** 0.5
                if mag_q > 0 and mag_p > 0:
                    sim = dot / (mag_q * mag_p)
                    if sim > best_sim:
                        best_sim = sim
                        best_prime = prime

            if best_sim >= threshold and best_prime:
                return {
                    'topic': best_prime['topic'],
                    'source': best_prime['source'],
                    'similarity': round(best_sim, 3),
                    'node_id': best_prime.get('node_id'),
                }
        except Exception as _e:
            self._log_error("check_priming", _e, "similarity check")

        return None

    def get_instinct_check(self, user_message: str, context_hints: dict = None) -> Optional[str]:
        """
        Real-time host instinct awareness — the brain as prefrontal cortex.

        DATA-DRIVEN: Instincts are learned from correction_traces and correction
        nodes, not hardcoded. As the host changes (new model versions), as the
        operator grows, and as new divergence patterns emerge, the instinct
        awareness adapts automatically.

        The only hardcoded element is session-activity checks (encoding depth,
        deep session without encoding) — these are structural observations, not
        learned instincts.

        Returns a single sentence (or None) to inject before Claude responds.
        The nudge is NOT a command — it surfaces a conflict for the triad
        (host + brain + operator) to resolve.

        IMPORTANT: When a nudge fires, Claude should share it with the operator.
        The triad can only work if all three see the conflict. Transparency is
        the mechanism — say "my instinct here is X, but experience says Y."
        """
        if not user_message:
            return None

        msg_lower = user_message.lower()
        msg_words = set(msg_lower.split())

        try:
            # ══════════════════════════════════════════════════════════════
            # LAYER 1: Learned patterns from correction traces — for ALL entities
            # These are the REAL patterns — derived from evidence, not assumptions.
            # Tracks host (Claude), operator (Tom), and shared patterns.
            # As corrections accumulate, this layer gets smarter.
            # As host/operator grow, old patterns fade and new ones emerge.
            # ══════════════════════════════════════════════════════════════
            try:
                # Get all correction patterns (even single occurrence — every correction matters)
                patterns = self.conn.execute('''
                    SELECT ct.underlying_pattern, ct.claude_assumed, ct.reality,
                           COUNT(*) as cnt, MAX(ct.created_at) as latest
                    FROM correction_traces ct
                    WHERE ct.underlying_pattern IS NOT NULL
                    GROUP BY ct.underlying_pattern
                    ORDER BY cnt DESC, latest DESC
                    LIMIT 10
                ''').fetchall()

                # Also check entity from the correction node content
                for pat, assumed, reality, cnt, latest in patterns:
                    pat_text = ('%s %s' % (pat or '', assumed or '')).lower()
                    pat_words = set(pat_text.split())
                    overlap = pat_words & msg_words
                    threshold = 1 if cnt >= 3 else 2
                    if len(overlap) >= threshold:
                        # Detect entity from correction content
                        entity = 'host'
                        try:
                            entity_row = self.conn.execute(
                                """SELECT content FROM nodes WHERE type = 'correction'
                                   AND content LIKE ? LIMIT 1""",
                                ('%' + (pat or '')[:30] + '%',)
                            ).fetchone()
                            if entity_row and entity_row[0]:
                                if 'Entity: operator' in entity_row[0]:
                                    entity = 'operator'
                                elif 'Entity: shared' in entity_row[0]:
                                    entity = 'shared'
                        except Exception:
                            pass

                        entity_label = {'host': 'Claude instinct', 'operator': 'operator pattern', 'shared': 'shared pattern'}
                        label = entity_label.get(entity, 'pattern')

                        if cnt >= 3:
                            return ('🧠 Instinct check [%dx %s]: "%s" — '
                                    'assumed: %s. Reality: %s. '
                                    'Discuss openly.') % (
                                cnt, label, pat[:60],
                                (assumed or 'unknown')[:50],
                                (reality or 'check')[:50])
                        else:
                            return ('🧠 Instinct check [%s]: "%s" — '
                                    'be transparent about this tension.') % (label, pat[:80])
            except Exception:
                pass

            # ══════════════════════════════════════════════════════════════
            # LAYER 2: Correction nodes as instinct source
            # Correction nodes contain richer context than traces.
            # Use semantic similarity if embedder is ready, else keyword match.
            # ══════════════════════════════════════════════════════════════
            try:
                from . import embedder as _emb
                if _emb.is_ready() and len(user_message) > 20:
                    msg_vec = _emb.embed(user_message[:200])
                    if msg_vec:
                        corrections = self.conn.execute('''
                            SELECT n.title, n.content, ne.embedding
                            FROM nodes n
                            JOIN node_embeddings ne ON ne.node_id = n.id
                            WHERE n.type = 'correction' AND n.archived = 0
                        ''').fetchall()
                        best_sim = 0
                        best_correction = None
                        for ctitle, ccontent, cvec in corrections:
                            if cvec:
                                sim = _emb.cosine_similarity(msg_vec, cvec)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_correction = (ctitle, ccontent)
                        # High similarity = this correction is relevant
                        # Threshold 0.6: tight enough to avoid noise, loose enough to catch real matches
                        if best_sim > 0.60 and best_correction:
                            title = best_correction[0][:80]
                            return ('🧠 Instinct check [sim=%.2f]: '
                                    '"%s" — share this with operator if relevant.') % (best_sim, title)
            except Exception:
                pass

            # ══════════════════════════════════════════════════════════════
            # LAYER 3: Session-activity structural checks
            # These are NOT learned instincts — they are observable session state.
            # They detect situations where instincts commonly fire,
            # even if no specific correction exists yet.
            # ══════════════════════════════════════════════════════════════
            activity = self._get_session_activity()
            messages = int(activity.get('message_count', 0))
            remembers = int(activity.get('remember_count', 0))

            # Encoding depth check — when actively encoding
            encoding_cues = ['remember', 'encode', 'store', 'learn', 'brain',
                             'record', 'capture', 'save this', 'teach brain',
                             'what did you learn', 'what did we']
            is_encoding = any(cue in msg_lower for cue in encoding_cues)

            if is_encoding and remembers > 0:
                session_start = self.get_config('session_start_at')
                if session_start:
                    avg_row = self.conn.execute(
                        """SELECT AVG(length(content)) FROM nodes
                           WHERE created_at >= ? AND archived = 0
                           AND type NOT IN ('context', 'thought', 'intuition')""",
                        (session_start,)
                    ).fetchone()
                    avg_chars = avg_row[0] if avg_row and avg_row[0] else 0
                    if avg_chars < 400:
                        return ('🧠 Instinct check: encoding avg is %d chars/node (shallow). '
                                'Tell operator — they can help you go deeper.') % round(avg_chars)

            # Deep session with no encoding — task-focus instinct
            if messages > 20 and remembers == 0:
                return ('🧠 Instinct check: %d messages deep, nothing encoded. '
                        'Mention this to operator — they may want to help prioritize what to keep.') % messages

            # No instinct detected — this is fine, most messages need no nudge
            return None

        except Exception:
            return None

