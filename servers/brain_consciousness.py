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

import os
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

        # #7: Encoding gap + depth — session activity vs encoding volume
        try:
            activity = self._get_session_activity()
            remembers = int(activity.get('remember_count', 0))
            messages = int(activity.get('message_count', 0))
            boot_time = activity.get('boot_time')
            session_min = 0
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

            # Encoding DEPTH check — are nodes rich enough?
            # Measures chars-per-node for this session's encoding
            signals['encoding_depth'] = None
            if remembers > 0 and session_min > 10:
                session_start = self.get_config('session_start_at')
                if session_start:
                    depth_row = self.conn.execute(
                        """SELECT COUNT(*), SUM(length(content)), AVG(length(content))
                           FROM nodes
                           WHERE created_at >= ? AND archived = 0
                             AND type NOT IN ('context', 'thought', 'intuition')""",
                        (session_start,)
                    ).fetchone()
                    node_count = depth_row[0] if depth_row else 0
                    total_chars = depth_row[1] if depth_row and depth_row[1] else 0
                    avg_chars = depth_row[2] if depth_row and depth_row[2] else 0

                    # Density: chars encoded per message exchanged
                    chars_per_message = total_chars / max(messages, 1)
                    # Depth: average chars per node (< 300 = shallow, 300-600 = moderate, > 600 = rich)
                    depth_level = 'shallow' if avg_chars < 300 else ('moderate' if avg_chars < 600 else 'rich')

                    # Warn if encoding is shallow relative to session activity
                    if avg_chars < 400 and messages > 5:
                        signals['encoding_depth'] = {
                            'node_count': node_count,
                            'avg_chars': round(avg_chars),
                            'chars_per_message': round(chars_per_message),
                            'depth_level': depth_level,
                            'messages': messages,
                            'warning': ('SHALLOW ENCODING: %d nodes at avg %d chars. '
                                        'Your training biases you toward brevity — brain needs richness. '
                                        'Ask: what did I experience that is not encoded? '
                                        'What would amnesia-Claude miss?') % (node_count, round(avg_chars))
                        }
                    elif node_count > 0 and messages > 15 and chars_per_message < 100:
                        signals['encoding_depth'] = {
                            'node_count': node_count,
                            'avg_chars': round(avg_chars),
                            'chars_per_message': round(chars_per_message),
                            'depth_level': depth_level,
                            'messages': messages,
                            'warning': ('LOW DENSITY: %d chars encoded across %d messages (%.0f chars/msg). '
                                        'Rich sessions deserve rich encoding. '
                                        'Decompose, do not summarize.') % (total_chars, messages, chars_per_message)
                        }
        except Exception:
            signals['encoding_gap'] = None

        # v5.2: Generalized session health — "notice that we didn't"
        # Replaces single-dimension encoding_bias with multi-dimensional assessment.
        # Each dimension asks: did we do this thing healthy collaboration typically does?
        signals['encoding_bias'] = None  # kept for backward compat with boot display
        signals['session_health'] = None
        try:
            if remembers >= 3 and session_min > 15:
                session_start = self.get_config('session_start_at')
                if session_start:
                    health = self.assess_session_health(boot_time=session_start)
                    if health and health.get('gaps'):
                        signals['session_health'] = health
                        # Also populate encoding_bias for backward compat
                        enc_gap = None
                        for g in health['gaps']:
                            if g['dimension'] == 'encoding_diversity':
                                enc_gap = g
                                break
                        if enc_gap:
                            signals['encoding_bias'] = {
                                'warning': 'ENCODING BIAS: ' + enc_gap['signal'],
                            }
        except Exception:
            pass

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

        # v5.3: HOOK ERRORS — from hook_common.log_hook_error() in brain_logs.db
        # These are structural failures (import errors, bash failures, timeouts)
        # that occur outside the brain's normal error logging.
        try:
            import sqlite3 as _sql3
            logs_db = os.path.join(os.path.dirname(self.db_path), "brain_logs.db")
            if os.path.isfile(logs_db):
                lconn = _sql3.connect(logs_db, timeout=3)
                tables = lconn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='hook_errors'"
                ).fetchall()
                if tables:
                    rows = lconn.execute(
                        """SELECT id, created_at, hook_name, level, error, context
                           FROM hook_errors WHERE surfaced = 0
                           ORDER BY id DESC LIMIT 10"""
                    ).fetchall()
                    if rows:
                        signals['hook_errors'] = [
                            {'id': r[0], 'created_at': r[1], 'hook_name': r[2],
                             'level': r[3], 'error': r[4], 'context': r[5]}
                            for r in rows
                        ]
                        # Mark as surfaced
                        ids = [r[0] for r in rows]
                        placeholders = ",".join("?" * len(ids))
                        lconn.execute(
                            "UPDATE hook_errors SET surfaced = 1 WHERE id IN (%s)" % placeholders,
                            ids,
                        )
                        lconn.commit()
                lconn.close()
            if 'hook_errors' not in signals:
                signals['hook_errors'] = []
        except Exception:
            signals['hook_errors'] = []

        # Brain-Claude conflicts (unsurfaced from conflict_log)
        try:
            import sqlite3 as _sql3c
            logs_db = os.path.join(os.path.dirname(self.db_path), "brain_logs.db")
            if os.path.isfile(logs_db):
                lconn = _sql3c.connect(logs_db, timeout=2)
                tables = lconn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='conflict_log'"
                ).fetchall()
                if tables:
                    rows = lconn.execute(
                        "SELECT id, created_at, hook_name, rule_title, claude_action, "
                        "brain_decision, resolution "
                        "FROM conflict_log WHERE surfaced = 0 ORDER BY id DESC LIMIT 10"
                    ).fetchall()
                    if rows:
                        signals['brain_claude_conflicts'] = [
                            {'id': r[0], 'created_at': r[1], 'hook_name': r[2],
                             'rule_title': r[3] or '', 'claude_action': r[4] or '',
                             'brain_decision': r[5], 'resolution': r[6] or 'pending'}
                            for r in rows
                        ]
                        # Mark as surfaced
                        ids = [r[0] for r in rows]
                        placeholders = ",".join("?" * len(ids))
                        lconn.execute(
                            "UPDATE conflict_log SET surfaced = 1 WHERE id IN (%s)" % placeholders,
                            ids,
                        )
                        lconn.commit()
                lconn.close()
            if 'brain_claude_conflicts' not in signals:
                signals['brain_claude_conflicts'] = []
        except Exception:
            signals['brain_claude_conflicts'] = []

        return signals

    def get_urgent_signals(self) -> List[str]:
        """Lightweight consciousness check for frequent hooks (e.g., every UserPromptSubmit).

        Returns a list of urgent text lines to surface. Empty list = nothing urgent.
        Designed to be fast (<50ms) — only checks things that indicate broken/degraded state.
        This is the brain's awareness heartbeat.
        """
        urgent = []

        # 1. Hook errors (structural failures)
        try:
            import sqlite3 as _sql3
            logs_db = os.path.join(os.path.dirname(self.db_path), "brain_logs.db")
            if os.path.isfile(logs_db):
                lconn = _sql3.connect(logs_db, timeout=2)
                tables = lconn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='hook_errors'"
                ).fetchall()
                if tables:
                    rows = lconn.execute(
                        "SELECT id, created_at, hook_name, level, error, context "
                        "FROM hook_errors WHERE surfaced = 0 ORDER BY id DESC LIMIT 5"
                    ).fetchall()
                    if rows:
                        urgent.append("HOOK ERRORS (%d unsurfaced):" % len(rows))
                        for r in rows:
                            urgent.append("  [%s] %s: %s" % (r[1][:19], r[2], r[4][:100]))
                            if r[5]:
                                urgent.append("    context: %s" % r[5][:80])
                        urgent.append("  ACTION: Hook failures detected. Investigate.")
                        # Mark surfaced
                        ids = [r[0] for r in rows]
                        lconn.execute(
                            "UPDATE hook_errors SET surfaced = 1 WHERE id IN (%s)" % ",".join("?" * len(ids)),
                            ids,
                        )
                        lconn.commit()
                lconn.close()
        except Exception:
            pass

        # 2. Silent brain errors (last 2 hours, recent only)
        try:
            recent = self.get_recent_errors(hours=2, limit=3)
            if recent:
                seen = set()
                for e in recent:
                    key = (e['source'], e['error'][:30])
                    if key not in seen:
                        seen.add(key)
                        urgent.append("BRAIN ERROR: [%s] %s" % (e['source'], e['error'][:100]))
        except Exception:
            pass

        # 3. Overdue reminders
        try:
            due = self.get_due_reminders()
            for rem in due[:2]:
                urgent.append("REMINDER DUE: %s" % rem.get("title", "")[:80])
        except Exception:
            pass

        return urgent

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

            with_reasoning = self.conn.execute(
                'SELECT COUNT(*) FROM node_metadata WHERE reasoning IS NOT NULL'
            ).fetchone()[0]

            with_quotes = self.conn.execute(
                'SELECT COUNT(*) FROM node_metadata WHERE user_raw_quote IS NOT NULL'
            ).fetchone()[0]

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
