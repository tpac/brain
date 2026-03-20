"""
brain — Consciousness Signals Mixin

Extracted from brain.py monolith. Contains:
- get_consciousness_signals() — 20+ signal types surfaced at boot
- log_consciousness_response() — track signal engagement
- scan_host_environment() — detect environment changes
- get_surfaceable_dreams() — high-surprise cross-cluster connections
- auto_generate_self_reflection() — periodic self-assessment

These methods are mixed into the Brain class via multiple inheritance.
All methods reference self.conn, self.logs_conn, self.get_config, etc.
which are provided by Brain's __init__.
"""

from typing import Dict, List, Optional, Any
from . import embedder


class ConsciousnessMixin:
    """Consciousness signal gathering and self-awareness methods."""

    def get_consciousness_signals(self) -> Dict[str, Any]:
        """
        Gather all conscious-layer signals for surfacing.
        Returns categorized signals: reminders, evolutions, decay_warnings,
        fluid_personal, stale_context, encoding_health, fading_knowledge.
        """
        signals = {}

        # Reminders due
        signals['reminders'] = self.get_due_reminders()

        # Active evolutions
        try:
            signals['evolutions'] = self.get_active_evolutions()
        except Exception:
            signals['evolutions'] = []

        # Fluid personal nodes — may need "still true?" check
        try:
            signals['fluid_personal'] = self.get_personal_nodes('fluid')
        except Exception:
            signals['fluid_personal'] = []

        # Fading knowledge — important nodes with low retention
        try:
            cursor = self.conn.execute(
                """SELECT id, type, title, last_accessed, access_count, locked, emotion
                   FROM nodes
                   WHERE locked = 0 AND archived = 0 AND access_count >= 3
                     AND type NOT IN ('context', 'thought', 'intuition')
                     AND last_accessed < datetime('now', '-14 days')
                   ORDER BY access_count DESC
                   LIMIT 5"""
            )
            signals['fading'] = [
                {'id': r[0], 'type': r[1], 'title': r[2], 'last_accessed': r[3],
                 'access_count': r[4], 'emotion': r[6]}
                for r in cursor.fetchall()
            ]
        except Exception:
            signals['fading'] = []

        # Stale context — old context nodes that should be cleaned up
        try:
            cursor = self.conn.execute(
                """SELECT COUNT(*) FROM nodes
                   WHERE type = 'context' AND archived = 0
                     AND created_at < datetime('now', '-7 days')"""
            )
            stale_count = cursor.fetchone()[0]
            signals['stale_context_count'] = stale_count
        except Exception:
            signals['stale_context_count'] = 0

        # Failure modes (always surface active ones)
        try:
            cursor = self.conn.execute(
                """SELECT id, title, content FROM nodes
                   WHERE type = 'failure_mode' AND evolution_status = 'active'
                     AND archived = 0
                   ORDER BY emotion DESC LIMIT 3"""
            )
            signals['failure_modes'] = [
                {'id': r[0], 'title': r[1], 'content': r[2]}
                for r in cursor.fetchall()
            ]
        except Exception:
            signals['failure_modes'] = []

        # Performance nodes (recent, for trending display)
        try:
            cursor = self.conn.execute(
                """SELECT id, title, content, created_at FROM nodes
                   WHERE type = 'performance' AND archived = 0
                   ORDER BY created_at DESC LIMIT 3"""
            )
            signals['performance'] = [
                {'id': r[0], 'title': r[1], 'content': r[2], 'created_at': r[3]}
                for r in cursor.fetchall()
            ]
        except Exception:
            signals['performance'] = []

        # Capability nodes
        try:
            cursor = self.conn.execute(
                """SELECT id, title, content FROM nodes
                   WHERE type = 'capability' AND archived = 0
                   ORDER BY created_at DESC LIMIT 3"""
            )
            signals['capabilities'] = [
                {'id': r[0], 'title': r[1], 'content': r[2]}
                for r in cursor.fetchall()
            ]
        except Exception:
            signals['capabilities'] = []

        # Interaction observations
        try:
            cursor = self.conn.execute(
                """SELECT id, title, content FROM nodes
                   WHERE type = 'interaction' AND archived = 0
                   ORDER BY created_at DESC LIMIT 3"""
            )
            signals['interactions'] = [
                {'id': r[0], 'title': r[1], 'content': r[2]}
                for r in cursor.fetchall()
            ]
        except Exception:
            signals['interactions'] = []

        # Meta-learning methods
        try:
            cursor = self.conn.execute(
                """SELECT id, title, content FROM nodes
                   WHERE type = 'meta_learning' AND archived = 0
                   ORDER BY created_at DESC LIMIT 2"""
            )
            signals['meta_learning'] = [
                {'id': r[0], 'title': r[1], 'content': r[2]}
                for r in cursor.fetchall()
            ]
        except Exception:
            signals['meta_learning'] = []

        # Novelty detection — nodes created this session with new terms
        try:
            cursor = self.conn.execute(
                """SELECT id, title, content FROM nodes
                   WHERE type = 'concept' AND archived = 0
                     AND created_at > datetime('now', '-2 hours')
                   ORDER BY created_at DESC LIMIT 3"""
            )
            signals['novelty'] = [
                {'id': r[0], 'title': r[1], 'content': r[2]}
                for r in cursor.fetchall()
            ]
        except Exception:
            signals['novelty'] = []

        # #6: Recall miss trends — queries that keep failing
        try:
            cursor = self.logs_conn.execute(
                """SELECT query, COUNT(*) as cnt, MAX(created_at) as latest
                   FROM miss_log
                   WHERE created_at > datetime('now', '-7 days')
                   GROUP BY query HAVING cnt >= 2
                   ORDER BY cnt DESC LIMIT 3"""
            )
            signals['miss_trends'] = [
                {'query': r[0], 'count': r[1], 'latest': r[2]}
                for r in cursor.fetchall()
            ]
        except Exception:
            signals['miss_trends'] = []

        # #7: Encoding gap — long session with no remember() calls
        try:
            activity = self._get_session_activity()
            remembers = int(activity.get('remember_count', 0))
            boot_time = activity.get('boot_time')
            if boot_time:
                from datetime import datetime as _dt
                try:
                    boot_dt = _dt.fromisoformat(boot_time.replace('Z', '+00:00'))
                    now_dt = _dt.now(boot_dt.tzinfo) if boot_dt.tzinfo else _dt.utcnow()
                    session_min = (now_dt - boot_dt).total_seconds() / 60
                except Exception:
                    session_min = 0
                if session_min > 20 and remembers == 0:
                    signals['encoding_gap'] = {
                        'session_minutes': round(session_min),
                        'remembers': remembers,
                        'warning': '%d minutes in session, nothing encoded yet.' % round(session_min)
                    }
                else:
                    signals['encoding_gap'] = None
            else:
                signals['encoding_gap'] = None
        except Exception:
            signals['encoding_gap'] = None

        # #8: Connection density shifts — compare cluster sizes
        try:
            cursor = self.conn.execute(
                """SELECT project, COUNT(*) as cnt FROM nodes
                   WHERE archived = 0 AND project IS NOT NULL
                   GROUP BY project ORDER BY cnt DESC LIMIT 5"""
            )
            clusters = [{'project': r[0], 'nodes': r[1]} for r in cursor.fetchall()]
            if len(clusters) >= 2:
                max_nodes = clusters[0]['nodes']
                min_nodes = clusters[-1]['nodes']
                if max_nodes > 10 * min_nodes:
                    signals['density_shift'] = {
                        'largest': clusters[0],
                        'smallest': clusters[-1],
                        'ratio': round(max_nodes / max(min_nodes, 1)),
                        'warning': 'Knowledge heavily concentrated in %s (%d nodes) vs %s (%d nodes)' % (
                            clusters[0]['project'], max_nodes, clusters[-1]['project'], min_nodes)
                    }
                else:
                    signals['density_shift'] = None
            else:
                signals['density_shift'] = None
        except Exception:
            signals['density_shift'] = None

        # #9: Emotional trajectory — average emotion across recent sessions
        try:
            cursor = self.conn.execute(
                """SELECT AVG(emotion), COUNT(*) FROM nodes
                   WHERE emotion > 0 AND created_at > datetime('now', '-3 days')"""
            )
            row = cursor.fetchone()
            recent_emotion = row[0] if row and row[0] else 0
            recent_count = row[1] if row else 0

            cursor2 = self.conn.execute(
                """SELECT AVG(emotion), COUNT(*) FROM nodes
                   WHERE emotion > 0 AND created_at BETWEEN datetime('now', '-10 days') AND datetime('now', '-3 days')"""
            )
            row2 = cursor2.fetchone()
            older_emotion = row2[0] if row2 and row2[0] else 0

            if recent_count >= 5 and recent_emotion > older_emotion + 0.15:
                signals['emotional_trajectory'] = {
                    'recent_avg': round(recent_emotion, 2),
                    'older_avg': round(older_emotion, 2),
                    'trend': 'increasing',
                    'warning': 'Emotion trending up: %.2f -> %.2f (last 3 days vs prior week)' % (older_emotion, recent_emotion)
                }
            elif recent_count >= 5 and recent_emotion < older_emotion - 0.15:
                signals['emotional_trajectory'] = {
                    'recent_avg': round(recent_emotion, 2),
                    'older_avg': round(older_emotion, 2),
                    'trend': 'decreasing',
                }
            else:
                signals['emotional_trajectory'] = None
        except Exception:
            signals['emotional_trajectory'] = None

        # #10: Rule contradiction detection
        try:
            signals['rule_contradictions'] = []
            recent_nodes = self.conn.execute(
                """SELECT id, title, content FROM nodes
                   WHERE created_at > datetime('now', '-24 hours')
                     AND locked = 0 AND archived = 0 AND type NOT IN ('context', 'thought', 'intuition')
                   ORDER BY created_at DESC LIMIT 10"""
            ).fetchall()
            if recent_nodes and embedder.is_ready():
                locked_rules = self.conn.execute(
                    """SELECT ne.node_id, ne.embedding, n.title FROM node_embeddings ne
                       JOIN nodes n ON n.id = ne.node_id
                       WHERE n.locked = 1 AND n.type IN ('rule', 'decision') AND n.archived = 0"""
                ).fetchall()
                for nid, ntitle, ncontent in recent_nodes:
                    node_vec = None
                    try:
                        row = self.conn.execute('SELECT embedding FROM node_embeddings WHERE node_id = ?', (nid,)).fetchone()
                        if row:
                            node_vec = row[0]
                    except Exception as _e:
                        self._log_error("get_consciousness_signals", _e, "fetching node embedding for rule contradiction check")
                    if not node_vec:
                        continue
                    for rule_id, rule_vec, rule_title in locked_rules:
                        if rule_vec:
                            sim = embedder.cosine_similarity(node_vec, rule_vec)
                            if sim > 0.8:
                                signals['rule_contradictions'].append({
                                    'recent_node': ntitle[:60],
                                    'locked_rule': rule_title[:60],
                                    'similarity': round(sim, 3),
                                })
                                break
            if not signals['rule_contradictions']:
                signals['rule_contradictions'] = []
        except Exception:
            signals['rule_contradictions'] = []

        # Recent encodings
        try:
            session_start = self.get_config('session_start_at')
            if session_start:
                cursor = self.conn.execute(
                    """SELECT id, type, title, content, locked, created_at
                       FROM nodes
                       WHERE created_at >= ? AND archived = 0
                         AND type NOT IN ('context', 'thought', 'intuition')
                       ORDER BY created_at DESC
                       LIMIT 10""",
                    (session_start,)
                )
                signals['recent_encodings'] = [
                    {'id': r[0], 'type': r[1], 'title': r[2],
                     'content': r[3][:150] if r[3] else '',
                     'locked': bool(r[4]), 'created_at': r[5]}
                    for r in cursor.fetchall()
                ]
            else:
                signals['recent_encodings'] = []
        except Exception:
            signals['recent_encodings'] = []

        # Consciousness adaptation — deprioritize ignored signals
        try:
            signal_types = ['tension', 'hypothesis', 'aspiration', 'pattern', 'dream',
                            'fading', 'performance', 'capability', 'interaction']
            engagement = {}
            for st in signal_types:
                yes_key = 'consciousness_response_%s_yes' % st
                no_key = 'consciousness_response_%s_no' % st
                yes = int(self.get_config(yes_key, 0) or 0)
                no = int(self.get_config(no_key, 0) or 0)
                total = yes + no
                if total >= 3:
                    engagement[st] = yes / total
                else:
                    engagement[st] = 0.5

            signals['_engagement_scores'] = engagement

            for st, rate in engagement.items():
                if rate < 0.2:
                    key_map = {
                        'tension': 'evolutions', 'hypothesis': 'evolutions',
                        'aspiration': 'evolutions', 'pattern': 'evolutions',
                        'fading': 'fading', 'performance': 'performance',
                        'capability': 'capabilities', 'interaction': 'interactions',
                    }
                    target_key = key_map.get(st)
                    if target_key and isinstance(signals.get(target_key), list) and len(signals[target_key]) > 1:
                        if target_key == 'evolutions':
                            signals[target_key] = [e for e in signals[target_key] if e.get('type') != st] + \
                                                  [e for e in signals[target_key] if e.get('type') == st][:1]
                        else:
                            signals[target_key] = signals[target_key][:1]
        except Exception as _e:
            self._log_error("get_consciousness_signals", _e, "")

        # v5: Stale reasoning signal
        try:
            stale_cur = self.conn.execute('''
                SELECT n.id, n.type, n.title, nm.reasoning, nm.last_validated, n.confidence, n.created_at
                FROM node_metadata nm
                JOIN nodes n ON n.id = nm.node_id
                WHERE nm.reasoning IS NOT NULL
                  AND n.archived = 0
                  AND (nm.last_validated IS NULL
                       OR julianday('now') - julianday(nm.last_validated) > 21)
                  AND COALESCE(n.confidence, 0.7) > 0.5
                ORDER BY n.access_count DESC
                LIMIT 3
            ''')
            stale_rows = stale_cur.fetchall()
            if stale_rows:
                signals['stale_reasoning'] = [{
                    'id': r[0], 'type': r[1], 'title': r[2],
                    'reasoning_preview': (r[3][:100] + '...') if len(r[3]) > 100 else r[3],
                    'last_validated': r[4], 'confidence': r[5], 'created_at': r[6],
                } for r in stale_rows]
        except Exception as _e:
            self._log_error("get_consciousness_signals", _e, "")

        # v5: UNCHARTED_CODE
        try:
            file_nodes = self.conn.execute(
                """SELECT n.title FROM nodes n
                   WHERE n.type = 'file' AND n.archived = 0 AND n.access_count >= 3
                   AND NOT EXISTS (
                       SELECT 1 FROM nodes p WHERE p.type IN ('purpose', 'mechanism')
                       AND p.archived = 0 AND p.title LIKE '%' || n.title || '%'
                   )
                   ORDER BY n.access_count DESC LIMIT 3"""
            ).fetchall()
            signals['uncharted_code'] = [r[0] for r in file_nodes] if file_nodes else []
        except Exception:
            signals['uncharted_code'] = []

        # v5: STALE_FILE_INVENTORY
        try:
            project = self.get_config('default_project', 'default')
            changes = []
            if project:
                changes = self.detect_file_changes(project)
            signals['stale_file_inventory'] = changes[:5] if changes else []
        except Exception:
            signals['stale_file_inventory'] = []

        # v5: VOCABULARY_GAP
        try:
            gaps_json = self.get_config('vocabulary_gaps', '[]')
            import json as _json
            gaps = _json.loads(gaps_json) if gaps_json else []
            signals['vocabulary_gap'] = gaps[-5:] if gaps else []
        except Exception:
            signals['vocabulary_gap'] = []

        # v5: RECURRING DIVERGENCE
        try:
            recurring = self.conn.execute(
                '''SELECT underlying_pattern, COUNT(*) as cnt
                   FROM correction_traces
                   WHERE underlying_pattern IS NOT NULL
                   GROUP BY underlying_pattern HAVING cnt >= 3
                   ORDER BY cnt DESC LIMIT 3'''
            ).fetchall()
            signals['recurring_divergence'] = [
                {'pattern': r[0], 'count': r[1]} for r in recurring
            ] if recurring else []
        except Exception:
            signals['recurring_divergence'] = []

        # v5: VALIDATED APPROACHES
        try:
            validated = self.conn.execute(
                '''SELECT n.id, n.title, nm.last_validated, nm.validation_count
                   FROM node_metadata nm
                   JOIN nodes n ON n.id = nm.node_id
                   WHERE nm.last_validated IS NOT NULL
                     AND nm.last_validated > datetime('now', '-7 days')
                     AND n.archived = 0
                   ORDER BY nm.last_validated DESC LIMIT 3'''
            ).fetchall()
            signals['validated_approaches'] = [
                {'id': r[0], 'title': r[1], 'last_validated': r[2], 'count': r[3]}
                for r in validated
            ] if validated else []
        except Exception:
            signals['validated_approaches'] = []

        # v5: UNCERTAIN AREAS
        try:
            uncertain = self.conn.execute(
                '''SELECT n.id, n.title, n.content, n.created_at
                   FROM nodes n
                   WHERE n.type = 'uncertainty' AND n.archived = 0
                   ORDER BY n.created_at DESC LIMIT 3'''
            ).fetchall()
            signals['uncertain_areas'] = [
                {'id': r[0], 'title': r[1],
                 'preview': (r[2][:120] + '...') if r[2] and len(r[2]) > 120 else (r[2] or ''),
                 'created_at': r[3]}
                for r in uncertain
            ] if uncertain else []
        except Exception:
            signals['uncertain_areas'] = []

        # v5: MENTAL MODEL DRIFT
        try:
            drifted = self.conn.execute(
                '''SELECT n.id, n.title, n.created_at, n.confidence,
                       COALESCE(nm.last_validated, n.created_at) as last_checked
                   FROM nodes n
                   LEFT JOIN node_metadata nm ON nm.node_id = n.id
                   WHERE n.type = 'mental_model' AND n.archived = 0
                     AND (
                       EXISTS (SELECT 1 FROM correction_traces ct WHERE ct.original_node_id = n.id)
                       OR (nm.last_validated IS NULL AND n.created_at < datetime('now', '-14 days'))
                     )
                   ORDER BY n.created_at ASC LIMIT 3'''
            ).fetchall()
            signals['mental_model_drift'] = [
                {'id': r[0], 'title': r[1], 'created_at': r[2],
                 'confidence': r[3], 'last_checked': r[4]}
                for r in drifted
            ] if drifted else []
        except Exception:
            signals['mental_model_drift'] = []

        # v5: SILENT ERRORS
        try:
            recent_errors = self.get_recent_errors(hours=24, limit=5)
            if recent_errors:
                seen = set()
                deduped = []
                for e in recent_errors:
                    key = (e['source'], e['error'][:50])
                    if key not in seen:
                        seen.add(key)
                        deduped.append(e)
                signals['silent_errors'] = deduped
            else:
                signals['silent_errors'] = []
        except Exception:
            signals['silent_errors'] = []

        return signals

    def log_consciousness_response(self, signal_type: str, responded: bool):
        """
        Track whether the human responded to a surfaced conscious signal.
        Over time: surface more of what gets responses, less of what gets ignored.
        """
        ts = self.now()
        suffix = 'yes' if responded else 'no'
        key = 'consciousness_response_%s_%s' % (signal_type, suffix)
        try:
            current = int(self.get_config(key, 0) or 0)
            self.conn.execute(
                """INSERT OR REPLACE INTO brain_meta (key, value, updated_at)
                   VALUES (?, ?, ?)""",
                (key, str(current + 1), ts)
            )
            self.conn.commit()
        except Exception:
            pass
