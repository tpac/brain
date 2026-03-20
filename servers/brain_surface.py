"""
brain — BrainSurface Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from .schema import BRAIN_VERSION, BRAIN_VERSION_KEY, NODE_TYPES
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import re
import time

from .brain_constants import (
    CONTEXT_BOOT_LOCKED_LIMIT,
    CONTEXT_BOOT_RECALL_LIMIT,
    CONTEXT_BOOT_RECENT_LIMIT,
)



class BrainSurfaceMixin:
    """Surface methods for Brain."""

    def suggest(self, context: Optional[str] = None, file: Optional[str] = None,
               screen: Optional[str] = None, action: Optional[str] = None,
               project: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Multi-query recall with type boosts, locked node boost, file-term relevance.
        Returns dict with suggestions list and query_count.
        """
        if limit is None:
            limit = self.get_config('suggestion_limit', 5)

        queries = []

        if context:
            queries.append(context)
        if file:
            # Clean filename
            clean_file = file.replace('/', ' ').replace('\\', ' ').replace(os.path.splitext(file)[1], '')
            clean_file = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean_file)
            clean_file = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', clean_file)
            queries.append(clean_file)
            # File parts
            file_parts = [p for p in re.split(r'[\s\-_]+', clean_file) if len(p) > 2]
            if len(file_parts) > 1:
                queries.extend(file_parts)

        if screen:
            queries.append(screen)
        if action:
            queries.append(action)

        if not queries:
            return {'suggestions': [], 'reason': 'no context provided'}

        # Run recall for each query
        seen = set()
        all_results = []
        pool_multiplier = self.get_config('recall_pool_multiplier', 2)
        recall_limit = max(limit * pool_multiplier, 15)

        for q in queries:
            result = self.recall(query=q, limit=recall_limit)
            results = result.get('results', result) if isinstance(result, dict) else result
            for r in results:
                if r['id'] not in seen:
                    seen.add(r['id'])
                    all_results.append(r)

        # Second pass: check edge neighbors of top results for locked nodes
        try:
            top_ids = [r['id'] for r in all_results[:10]]
            if top_ids:
                placeholders = ','.join('?' * len(top_ids))
                neighbor_rows = self.conn.execute(f'''
                    SELECT DISTINCT n.id, n.type, n.title, n.content, n.keywords, n.activation,
                           n.stability, n.access_count, n.locked, n.archived, n.last_accessed, n.created_at
                    FROM edges e
                    JOIN nodes n ON (n.id = CASE WHEN e.source_id = n.id THEN e.target_id ELSE e.source_id END)
                    WHERE (e.source_id IN ({placeholders}) OR e.target_id IN ({placeholders}))
                      AND n.locked = 1 AND n.archived = 0
                      AND n.id NOT IN ({placeholders})
                    LIMIT 20
                ''', top_ids + top_ids + top_ids).fetchall()

                for row in neighbor_rows:
                    nid = row[0]
                    if nid not in seen:
                        seen.add(nid)
                        all_results.append({
                            'id': row[0], 'type': row[1], 'title': row[2], 'content': row[3],
                            'keywords': row[4], 'activation': row[5], 'stability': row[6],
                            'access_count': row[7], 'locked': row[8] == 1, 'archived': row[9] == 1,
                            'last_accessed': row[10], 'created_at': row[11],
                            '_edge_neighbor': True
                        })
        except:
            pass

        # Project filter
        if project:
            all_results.sort(key=lambda a: (
                -(1 if a.get('project') == project else 0),
                -(a.get('effective_activation') or 0)
            ))

        # Scoring
        rule_boost = self.get_config('boost_rule', 1.3)
        decision_boost = self.get_config('boost_decision', 1.2)
        locked_boost = self.get_config('boost_locked', 1.5)
        edge_neighbor_penalty = self.get_config('penalty_edge_neighbor', 0.85)
        file_relevance_max = self.get_config('file_relevance_bonus', 0.15)

        # File-specific terms
        file_terms = set()
        if file:
            clean = file.replace('/', ' ').replace('\\', ' ').replace(os.path.splitext(file)[1], '')
            clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
            clean = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', clean)
            for t in clean.lower().split(r'[\s\-_]+'):
                if len(t) > 2:
                    file_terms.add(t)
        if screen:
            file_terms.add(screen.lower())

        ranked = []
        for r in all_results:
            boost = 1.0
            if r.get('type') == 'rule':
                boost = rule_boost
            if r.get('type') == 'decision':
                boost = decision_boost
            if r.get('locked'):
                boost *= locked_boost
            if r.get('_edge_neighbor'):
                boost *= edge_neighbor_penalty

            # File relevance
            file_relevance = 0
            if file_terms:
                node_text = f"{r.get('title', '')} {r.get('keywords', '')}".lower()
                for term in file_terms:
                    if term in node_text:
                        file_relevance += 1
                file_relevance = (file_relevance / len(file_terms)) * file_relevance_max

            r['suggest_score'] = ((r.get('effective_activation') or 0.5) + file_relevance) * boost
            ranked.append(r)

        ranked.sort(key=lambda r: -r['suggest_score'])

        # Locked node promotion
        selected = ranked[:limit]
        selected_ids = {r['id'] for r in selected}

        if file_terms:
            missed_locked = [
                r for r in ranked
                if r['id'] not in selected_ids and r.get('locked') and
                   r.get('type') in ('rule', 'decision') and
                   any(t in f"{r.get('title', '')} {r.get('keywords', '')}".lower() for t in file_terms)
            ]

            for locked_node in missed_locked:
                worst_idx = -1
                worst_score = float('inf')
                for i in range(len(selected) - 1, -1, -1):
                    if not selected[i].get('locked') and selected[i]['suggest_score'] < worst_score:
                        worst_idx = i
                        worst_score = selected[i]['suggest_score']
                if worst_idx >= 0:
                    selected[worst_idx] = locked_node
                    selected_ids.add(locked_node['id'])
                else:
                    break

            selected.sort(key=lambda r: -r['suggest_score'])

        suggestions = [
            {
                'id': r['id'],
                'type': r.get('type'),
                'title': r.get('title'),
                'content': r.get('content', '')[:300] if r.get('content') else None,
                'locked': r.get('locked', False),
                'relevance': r['suggest_score'],
                'reason': self._suggest_reason(r, queries)
            }
            for r in selected
        ]

        # Log suggestion
        try:
            self.logs_conn.execute('''
                INSERT INTO suggest_log (session_id, context, suggested_ids, created_at)
                VALUES (?, ?, ?, ?)
            ''', ('auto', ' | '.join(queries), json.dumps([s['id'] for s in suggestions]), ts))
        except:
            pass

        return {'suggestions': suggestions, 'query_count': len(queries)}

    def _suggest_reason(self, node: Dict[str, Any], queries: List[str]) -> str:
        """Generate reason string for suggestion."""
        lower_title = (node.get('title') or '').lower()
        for q in queries:
            terms = [w for w in q.lower().split() if len(w) > 2]
            for t in terms:
                if t in lower_title:
                    return f'matches "{t}" from context'
        return 'related via graph connections'

    def context_boot(self, user: str = '', project: str = '', task: Optional[str] = None,
                     hints: Optional[str] = None) -> Dict[str, Any]:
        """
        3-tier progressive loading for context boot.
        Full content for top locked nodes, title-only index for rest,
        recent nodes, task-recalled nodes.
        Returns dict with brain_version, locked, recalled, recent, reset_count, last_session_note.
        """
        boot_limits = self._get_tunable('boot_limits', {
            'locked': CONTEXT_BOOT_LOCKED_LIMIT,
            'recall': CONTEXT_BOOT_RECALL_LIMIT,
            'recent': CONTEXT_BOOT_RECENT_LIMIT
        })
        max_locked = boot_limits.get('locked', CONTEXT_BOOT_LOCKED_LIMIT) if isinstance(boot_limits, dict) else CONTEXT_BOOT_LOCKED_LIMIT
        max_recall = boot_limits.get('recall', CONTEXT_BOOT_RECALL_LIMIT) if isinstance(boot_limits, dict) else CONTEXT_BOOT_RECALL_LIMIT
        max_recent = boot_limits.get('recent', CONTEXT_BOOT_RECENT_LIMIT) if isinstance(boot_limits, dict) else CONTEXT_BOOT_RECENT_LIMIT

        query_parts = [user, project, task, hints]
        query = ' '.join(p for p in query_parts if p)

        # 1. Get locked nodes with full content for top N
        # Project-scoped: return nodes for this project + global (NULL project)
        locked = self.conn.execute('''
            SELECT id, type, title, content, keywords FROM nodes
            WHERE locked = 1 AND archived = 0
              AND (project = ? OR project IS NULL OR project = '')
            ORDER BY
              CASE type WHEN 'rule' THEN 0 WHEN 'decision' THEN 1 ELSE 2 END,
              access_count DESC, last_accessed DESC
            LIMIT ?
        ''', (project, max_locked)).fetchall()

        results = {
            'locked': [],
            'locked_index': [],
            'recalled': [],
            'recent': []
        }

        seen = set()

        for r in locked:
            seen.add(r[0])
            results['locked'].append({
                'id': r[0], 'type': r[1], 'title': r[2],
                'content': r[3], 'keywords': r[4]
            })

        # Title-only index for remaining locked nodes (same project scope)
        locked_index = self.conn.execute('''
            SELECT id, type, title FROM nodes
            WHERE locked = 1 AND archived = 0
              AND (project = ? OR project IS NULL OR project = '')
            ORDER BY
              CASE type WHEN 'rule' THEN 0 WHEN 'decision' THEN 1 ELSE 2 END,
              access_count DESC, last_accessed DESC
            LIMIT 500 OFFSET ?
        ''', (project, max_locked)).fetchall()

        for r in locked_index:
            if r[0] not in seen:
                seen.add(r[0])
                results['locked_index'].append({
                    'id': r[0], 'type': r[1], 'title': r[2]
                })

        # 2. Recently accessed nodes
        recent = self.conn.execute('''
            SELECT id, type, title, content, keywords, activation, last_accessed FROM nodes
            WHERE archived = 0 AND locked = 0
            ORDER BY last_accessed DESC LIMIT ?
        ''', (max_recent,)).fetchall()

        for r in recent:
            if r[0] not in seen:
                seen.add(r[0])
                results['recent'].append({
                    'id': r[0], 'type': r[1], 'title': r[2],
                    'content': r[3]
                })

        # 3. Recall by context query (project-scoped)
        if query:
            recall_result = self.recall(query=query, limit=max_recall, project=project)
            recalled = recall_result.get('results', recall_result) if isinstance(recall_result, dict) else recall_result
            for r in recalled:
                if r['id'] not in seen:
                    seen.add(r['id'])
                    results['recalled'].append({
                        'id': r['id'], 'type': r.get('type'),
                        'title': r.get('title'), 'content': r.get('content')
                    })

        # Get total locked count
        total_locked = self.conn.execute(
            'SELECT COUNT(*) FROM nodes WHERE locked = 1 AND archived = 0'
        ).fetchone()[0]

        # Find last session note
        session_logs = self.conn.execute('''
            SELECT id, title, content, created_at FROM nodes
            WHERE title LIKE '%Session%Log%Reset%' AND archived = 0
            ORDER BY created_at DESC LIMIT 1
        ''').fetchone()

        last_session_note = None
        reset_count = 0
        if session_logs:
            last_session_note = {
                'id': session_logs[0],
                'title': session_logs[1],
                'content': session_logs[2],
                'created_at': session_logs[3]
            }
            # Parse reset number
            match = re.search(r'Reset #(\d+)', session_logs[1])
            reset_count = int(match.group(1)) if match else 0

        return {
            'brain_version': BRAIN_VERSION,
            'total_nodes': self._get_node_count(),
            'total_edges': self._get_edge_count(),
            'total_locked': total_locked,
            'locked_shown': len(results['locked']),
            'has_more_locked': total_locked > max_locked,
            'reset_count': reset_count,
            'last_session_note': last_session_note,
            **results
        }

    def validate_config(self) -> List[Dict[str, Any]]:
        """Validate infrastructure configuration at boot. Returns list of warnings."""
        warnings = []

        # 1. Check DB is writable
        try:
            self.conn.execute("INSERT OR REPLACE INTO brain_meta (key, value, updated_at) VALUES ('_ping', '1', ?)", (self.now(),))
            self.conn.execute("DELETE FROM brain_meta WHERE key = '_ping'")
        except Exception as e:
            warnings.append({'level': 'critical', 'message': 'brain.db is READ-ONLY: %s' % e})

        # 2. Check logs DB is writable
        try:
            self.logs_conn.execute("INSERT INTO debug_log (event_type, source, created_at) VALUES ('ping', '_validate', ?)", (self.now(),))
            self.logs_conn.execute("DELETE FROM debug_log WHERE source = '_validate'")
            self.logs_conn.commit()
        except Exception as e:
            warnings.append({'level': 'critical', 'message': 'brain_logs.db is READ-ONLY: %s' % e})

        # 3. Check schema version matches expected
        try:
            ver = int(self._meta.get(BRAIN_VERSION_KEY, '0'))
            if ver < BRAIN_VERSION:
                warnings.append({'level': 'warning', 'message': 'Schema version %d < expected %d — migration may have failed' % (ver, BRAIN_VERSION)})
        except Exception:
            pass

        # 4. Check embedder status
        if not embedder.is_ready():
            warnings.append({'level': 'warning', 'message': 'Embedder not loaded — recall quality degraded (TF-IDF only)'})

        # 5. Check DB file sizes
        try:
            main_size = os.path.getsize(self.db_path)
            if main_size > 100 * 1024 * 1024:  # 100MB
                warnings.append({'level': 'warning', 'message': 'brain.db is %.0fMB — consider archiving old nodes' % (main_size / 1024 / 1024)})
            logs_size = os.path.getsize(self.logs_db_path)
            if logs_size > self._max_logs_db_size:
                warnings.append({'level': 'warning', 'message': 'brain_logs.db is %.0fMB — will auto-trim' % (logs_size / 1024 / 1024)})
        except Exception:
            pass

        return warnings

    def health_check(self, session_id: str = 'boot', auto_fix: bool = True) -> Dict[str, Any]:
        """
        Check brain health: unresolved compaction boundaries, high miss rate,
        orphaned locked nodes, stale contexts, stale staged learnings.
        Auto-fix: enrich missed nodes, promote staged learnings.
        """
        issues = []
        actions = []
        ts = self.now()

        # 1. Check for unresolved compaction boundary warnings
        boundaries = self.conn.execute('''
            SELECT id, title, created_at FROM nodes
            WHERE type = 'context' AND title LIKE '%Compaction boundary%' AND archived = 0
        ''').fetchall()

        for row in boundaries:
            issues.append({
                'type': 'compaction_boundary',
                'severity': 'high',
                'message': f'Unresolved compaction boundary from {row[2]}. Recap encoding may have been skipped.',
                'node_id': row[0]
            })

        # 2. Check for recent miss logs
        miss_count_row = self.logs_conn.execute(
            "SELECT COUNT(*) FROM miss_log WHERE created_at > datetime('now', '-24 hours')"
        ).fetchone()
        miss_count = miss_count_row[0] if miss_count_row else 0

        if miss_count > 3:
            issues.append({
                'type': 'high_miss_rate',
                'severity': 'medium',
                'message': f'{miss_count} recall misses in the last 24 hours. Consider keyword enrichment.'
            })

        # 3. Check for orphaned locked nodes
        orphaned = self.conn.execute('''
            SELECT n.id, n.title FROM nodes n
            WHERE n.locked = 1 AND n.archived = 0
            AND n.id NOT IN (SELECT source_id FROM edges)
            AND n.id NOT IN (SELECT target_id FROM edges)
            LIMIT 5
        ''').fetchall()

        if orphaned:
            issues.append({
                'type': 'orphaned_locked_nodes',
                'severity': 'low',
                'message': f'{len(orphaned)} locked nodes with no connections.',
                'nodes': [{'id': r[0], 'title': r[1]} for r in orphaned]
            })

        # 4. Check for stale context nodes
        stale_count_row = self.conn.execute('''
            SELECT COUNT(*) FROM nodes
            WHERE type = 'context' AND locked = 0 AND archived = 0
            AND created_at < datetime('now', '-7 days')
        ''').fetchone()
        stale_count = stale_count_row[0] if stale_count_row else 0

        if stale_count > 10:
            issues.append({
                'type': 'stale_contexts',
                'severity': 'low',
                'message': f'{stale_count} context nodes older than 7 days.'
            })
            if auto_fix:
                self.conn.execute('''
                    UPDATE nodes SET archived = 1
                    WHERE type = 'context' AND locked = 0 AND archived = 0
                    AND created_at < datetime('now', '-14 days')
                ''')
                actions.append('Auto-archived context nodes older than 14 days')

        # 5. Auto-enrich keywords on missed nodes
        if auto_fix:
            try:
                missed_nodes = self.logs_conn.execute('''
                    SELECT DISTINCT expected_node_id FROM miss_log
                    WHERE expected_node_id IS NOT NULL
                    ORDER BY rowid DESC LIMIT 10
                ''').fetchall()
                enriched = 0
                for (node_id,) in missed_nodes:
                    try:
                        self.enrich_keywords(node_id)
                        enriched += 1
                    except:
                        pass
                if enriched > 0:
                    actions.append(f'Auto-enriched keywords on {enriched} frequently-missed nodes')
            except:
                pass

        # 6. Auto-promote staged learnings
        if auto_fix:
            try:
                promoted = self.auto_promote_staged(revisit_threshold=3)
                if promoted.get('promoted', 0) > 0:
                    actions.append(f'Auto-promoted {promoted["promoted"]} staged learnings (3+ revisits)')
            except:
                pass

        # 7. Check for stale pending staged learnings
        try:
            stale_staged_row = self.logs_conn.execute('''
                SELECT COUNT(*) FROM staged_learnings
                WHERE status = 'pending' AND created_at < datetime('now', '-7 days')
            ''').fetchone()
            stale_staged_count = stale_staged_row[0] if stale_staged_row else 0
            if stale_staged_count > 0:
                issues.append({
                    'type': 'stale_staged_learnings',
                    'severity': 'medium',
                    'message': f'{stale_staged_count} staged learnings unreviewed for 7+ days.'
                })
        except:
            pass

        # Log health check
        try:
            self.logs_conn.execute('''
                INSERT INTO health_log (session_id, check_type, result, actions_taken, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, 'boot_check', json.dumps(issues), json.dumps(actions), ts))
        except:
            pass

        return {
            'healthy': not any(i['severity'] == 'high' for i in issues),
            'issues': issues,
            'actions': actions,
            'checked_at': ts
        }

    def list_staged(self, status: str = 'pending', limit: int = 20) -> Dict[str, Any]:
        """
        List staged learnings with optional status filter.
        Returns dict with staged list.
        """
        # Two-step: get staged from logs DB, then enrich from main DB
        query = 'SELECT node_id, status, times_revisited, created_at FROM staged_learnings WHERE 1=1'
        params = []
        if status != 'all':
            query += ' AND status = ?'
            params.append(status)
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)

        rows = self.logs_conn.execute(query, params).fetchall()
        results = []
        for node_id, sl_status, times_revisited, created_at in rows:
            node_row = self.conn.execute(
                'SELECT title, content, type, confidence FROM nodes WHERE id = ? AND archived = 0',
                (node_id,)
            ).fetchone()
            if node_row:
                results.append({
                    'node_id': node_id, 'status': sl_status,
                    'times_revisited': times_revisited,
                    'title': node_row[0], 'content': node_row[1],
                    'type': node_row[2], 'confidence': node_row[3],
                })
        return {'staged': results}

    def confirm_staged(self, node_id: str, lock: bool = False, new_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Confirm a staged learning — promotes it to full node.
        Bumps confidence to 0.8, removes [staged] prefix, optionally locks.
        """
        exists = self.conn.execute('SELECT id FROM nodes WHERE id = ?', (node_id,)).fetchone()
        if not exists:
            return {'action': 'error', 'error': f'Node {node_id} not found'}

        ts = self.now()

        # Update node
        if new_title:
            self.conn.execute(
                f'UPDATE nodes SET confidence = 0.8, locked = {1 if lock else 0}, title = ?, updated_at = ? WHERE id = ?',
                (new_title, ts, node_id)
            )
        else:
            self.conn.execute(
                f'UPDATE nodes SET confidence = 0.8, locked = {1 if lock else 0}, title = REPLACE(title, "[staged] ", ""), updated_at = ? WHERE id = ?',
                (ts, node_id)
            )

        # Update staged_learnings
        self.logs_conn.execute(
            "UPDATE staged_learnings SET status = 'confirmed', updated_at = ?, reviewed_session = ? WHERE node_id = ?",
            (ts, 'current', node_id)
        )

        return {'action': 'confirmed', 'node_id': node_id, 'confidence': 0.8, 'locked': lock}

    def dismiss_staged(self, node_id: str, reason: str = '') -> Dict[str, Any]:
        """
        Dismiss a staged learning — archives the node.
        """
        exists = self.conn.execute('SELECT id FROM nodes WHERE id = ?', (node_id,)).fetchone()
        if not exists:
            return {'action': 'error', 'error': f'Node {node_id} not found'}

        ts = self.now()
        self.conn.execute('UPDATE nodes SET archived = 1, updated_at = ? WHERE id = ?', (ts, node_id))
        self.logs_conn.execute(
            "UPDATE staged_learnings SET status = 'dismissed', updated_at = ?, reviewed_session = ? WHERE node_id = ?",
            (ts, reason or 'current', node_id)
        )

        return {'action': 'dismissed', 'node_id': node_id}

    def auto_promote_staged(self, revisit_threshold: int = 3) -> Dict[str, Any]:
        """
        Auto-promote staged learnings with enough revisits.
        Threshold: 3+ revisits = auto-promote to confidence 0.7.
        """
        ts = self.now()
        # Two-step: get pending staged from logs DB, then filter by main DB
        staged_rows = self.logs_conn.execute('''
            SELECT node_id, times_revisited
            FROM staged_learnings
            WHERE status = 'pending' AND times_revisited >= ?
        ''', (revisit_threshold,)).fetchall()
        candidates = []
        for node_id, times_revisited in staged_rows:
            row = self.conn.execute(
                'SELECT title FROM nodes WHERE id = ? AND archived = 0', (node_id,)
            ).fetchone()
            if row:
                candidates.append((node_id, times_revisited, row[0]))

        if not candidates:
            return {'promoted': 0}

        count = 0
        for node_id, _, _ in candidates:
            self.conn.execute(
                'UPDATE nodes SET confidence = 0.7, title = REPLACE(title, "[staged] ", ""), updated_at = ? WHERE id = ?',
                (ts, node_id)
            )
            self.logs_conn.execute(
                "UPDATE staged_learnings SET status = 'promoted', updated_at = ? WHERE node_id = ?",
                (ts, node_id)
            )
            count += 1

        return {'promoted': count, 'threshold': revisit_threshold}

    def get_suggest_metrics(self, period_days: int = 7) -> Dict[str, Any]:
        """
        Aggregate suggest_log stats for a period.
        Returns metrics on suggestion quality and diversity.
        """
        since = datetime.utcnow().timestamp() - (period_days * 86400)
        try:
            rows = self.logs_conn.execute('''
                SELECT COUNT(*) as calls, AVG(LENGTH(suggested_ids)) as avg_pool_size
                FROM suggest_log
                WHERE created_at > datetime(?, 'unixepoch')
            ''', (since,)).fetchone()

            return {
                'period_days': period_days,
                'total_suggest_calls': rows[0] if rows else 0,
                'avg_suggestions_per_call': rows[1] if rows and rows[1] else 0
            }
        except:
            return {'period_days': period_days, 'error': 'Could not aggregate metrics'}

    def pre_edit(self, file: str, tool_name: str = 'Edit') -> dict:
        """
        Batch pre-edit call combining all lookups into one.
        Replaces 8 sequential HTTP calls from the old architecture.

        Args:
            file: Filename being edited
            tool_name: 'Edit' or 'Write'

        Returns:
            Dict with suggestions, procedures, context_files, encoding health, timings
        """
        import time as _time
        t0 = _time.time()
        timings = {}

        # 1. Suggest
        t1 = _time.time()
        suggest_result = self.suggest(
            context=f"editing {file}",
            file=file,
            action=tool_name.lower(),
            limit=10
        )
        timings['suggest_ms'] = round((_time.time() - t1) * 1000)

        # 2. Procedures
        t2 = _time.time()
        proc_result = self.procedure_trigger('pre_edit', {'file': file, 'tool': tool_name})
        timings['procedures_ms'] = round((_time.time() - t2) * 1000)

        # 3. Encoding health
        activity = self._get_session_activity()
        boot_time = activity.get('boot_time', self.now())
        remembers = int(activity.get('remember_count', 0))
        edits_checked = int(activity.get('edit_check_count', 0))
        last_remember = activity.get('last_remember_at', None)

        # Compute session minutes
        try:
            from datetime import datetime as _dt
            boot_dt = _dt.fromisoformat(boot_time.replace('Z', '+00:00'))
            now_dt = _dt.now(boot_dt.tzinfo) if boot_dt.tzinfo else _dt.utcnow()
            session_minutes = (now_dt - boot_dt).total_seconds() / 60
        except Exception:
            session_minutes = 0

        # Compute minutes since last remember
        mins_since_remember = 0
        if last_remember:
            try:
                last_dt = _dt.fromisoformat(last_remember.replace('Z', '+00:00'))
                mins_since_remember = (now_dt - last_dt).total_seconds() / 60
            except Exception as _e:
                self._log_error("pre_edit", _e, "last_dt = _dt.fromisoformat(last_remember.replace(")

        # Determine encoding health status
        edits_since = edits_checked  # approximate — reset on each remember
        if remembers == 0 and session_minutes > 3:
            encoding_health = 'NONE'
        elif edits_since > 8 and mins_since_remember > 5:
            encoding_health = 'STALE'
        else:
            encoding_health = 'OK'

        # Increment edit check counter
        self.record_edit_check()

        # 4. Context files (nodes of type 'file' matching the edited filename)
        context_files = []
        try:
            cursor = self.conn.execute(
                "SELECT id, title, content, keywords, updated_at FROM nodes WHERE type = 'file' AND archived = 0 AND (title LIKE ? OR keywords LIKE ?) LIMIT 3",
                (f'%{file}%', f'%{file}%')
            )
            for row in cursor.fetchall():
                context_files.append({
                    'id': row[0], 'title': row[1], 'summary': (row[2] or '')[:200],
                    'topic': row[3] or '', 'last_updated': row[4],
                })
        except Exception as _e:
            self._log_error("pre_edit", _e, "cursor = self.conn.execute(")

        timings['total_ms'] = round((_time.time() - t0) * 1000)

        return {
            'suggestions': suggest_result.get('suggestions', []),
            'procedures': proc_result.get('matched', []),
            'context_files': context_files,
            'encoding': {
                'health': encoding_health,
                'remembers': remembers,
                'edits_since_last_remember': edits_since,
                'minutes_since_last_remember': round(mins_since_remember),
                'session_minutes': round(session_minutes),
            },
            'embedder_ready': embedder.is_ready(),
            'debug_enabled': self.get_debug_status(),
            'timings': timings,
        }

    def procedure_trigger(self, trigger_type: str, context: dict = None) -> dict:
        """
        Find and return procedures matching a trigger type.

        Args:
            trigger_type: 'session_start', 'pre_edit', 'pre_compact', etc.
            context: Optional context dict with trigger-specific data

        Returns:
            Dict with matched procedures list
        """
        context = context or {}
        matched = []

        try:
            cursor = self.conn.execute(
                "SELECT id, title, content, keywords FROM nodes WHERE type = 'procedure' AND archived = 0 AND locked = 1"
            )
            for row in cursor.fetchall():
                node_id, title, content, keywords = row
                content_lower = (content or '').lower()
                keywords_lower = (keywords or '').lower()

                # Check if procedure matches trigger type
                if trigger_type in content_lower or trigger_type in keywords_lower:
                    # Parse procedure content for steps
                    steps = content or ''
                    category = 'general'
                    if 'session_start' in keywords_lower:
                        category = 'session_start'
                    elif 'pre_edit' in keywords_lower:
                        category = 'pre_edit'
                    elif 'pre_compact' in keywords_lower:
                        category = 'pre_compact'

                    # Check file-specific procedures
                    if trigger_type == 'pre_edit' and 'file' in context:
                        file_name = context['file'].lower()
                        if file_name not in content_lower and file_name not in keywords_lower:
                            # Check for wildcard patterns
                            if '*' not in content_lower:
                                continue

                    matched.append({
                        'id': node_id,
                        'title': title,
                        'steps': steps,
                        'category': category,
                    })
        except Exception as _e:
            self._log_error("procedure_trigger", _e, "matching procedure nodes to trigger context")

        return {'matched': matched}
