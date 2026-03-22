"""
brain — BrainSurface Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from .schema import BRAIN_VERSION, BRAIN_VERSION_KEY, NODE_TYPES
from .text_processing import split_identifier
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import os
import re
import time

from .brain_constants import (
    CONTEXT_BOOT_LOCKED_LIMIT,
    CONTEXT_BOOT_RECALL_LIMIT,
    CONTEXT_BOOT_RECENT_LIMIT,
)

DESTRUCTIVE_PATTERNS = [
    re.compile(r'rm\s+(-[rf]+\s+|.*--force)', re.IGNORECASE),
    re.compile(r'git\s+worktree\s+remove', re.IGNORECASE),
    re.compile(r'git\s+reset\s+--hard', re.IGNORECASE),
    re.compile(r'git\s+clean\s+-[fd]', re.IGNORECASE),
    re.compile(r'git\s+checkout\s+--\s', re.IGNORECASE),
    re.compile(r'git\s+push\s+.*--force', re.IGNORECASE),
    re.compile(r'DROP\s+TABLE', re.IGNORECASE),
    re.compile(r'DELETE\s+FROM', re.IGNORECASE),
    re.compile(r'TRUNCATE', re.IGNORECASE),
    re.compile(r'\brmdir\b', re.IGNORECASE),
    re.compile(r'xargs\s+rm', re.IGNORECASE),
]


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
            # Split filename into search tokens using camelCase-aware splitter
            file_tokens = split_identifier(file)
            if file_tokens:
                queries.append(' '.join(file_tokens))
                # Individual tokens as separate queries for broader recall
                if len(file_tokens) > 1:
                    queries.extend(t for t in file_tokens if len(t) > 2)

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

        # v5.2: Critical nodes ALWAYS surface at boot — before everything else
        critical_nodes = self.conn.execute('''
            SELECT id, type, title, content, keywords FROM nodes
            WHERE critical = 1 AND archived = 0
            ORDER BY updated_at DESC
        ''').fetchall()

        results = {
            'locked': [],
            'locked_index': [],
            'recalled': [],
            'recent': [],
            'pending_critical': self.get_pending_critical() if hasattr(self, 'get_pending_critical') else []
        }

        seen = set()

        for r in critical_nodes:
            seen.add(r[0])
            results['locked'].insert(0, {
                'id': r[0], 'type': r[1], 'title': r[2],
                'content': r[3], 'keywords': r[4],
                '_critical': True
            })

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

        for r in locked:
            if r[0] in seen:
                continue  # Skip critical nodes already added
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

    def safety_check(self, command: str) -> dict:
        """
        Check a bash command against destructive patterns and brain safety nodes.

        Args:
            command: The bash command string to check

        Returns:
            Dict with destructive (bool), risk_level, warnings, critical_matches
        """
        # Check against destructive patterns
        matched_patterns = []
        for pattern in DESTRUCTIVE_PATTERNS:
            if pattern.search(command):
                matched_patterns.append(pattern.pattern)

        if not matched_patterns:
            return {'destructive': False, 'risk_level': 'none'}

        # Destructive command detected — query brain for safety context
        warnings = []
        critical_matches = []

        try:
            # Query for critical nodes
            critical_rows = self.conn.execute('''
                SELECT id, type, title, content FROM nodes
                WHERE critical = 1 AND archived = 0
            ''').fetchall()

            for row in critical_rows:
                node_id, node_type, title, content = row
                # Check if command relates to this critical node
                node_text = f"{title} {content}".lower()
                cmd_lower = command.lower()
                # Check for keyword overlap
                cmd_words = [w for w in re.split(r'[\s/\\.\-_]+', cmd_lower) if len(w) > 2]
                for word in cmd_words:
                    if word in node_text:
                        critical_matches.append({
                            'id': node_id,
                            'type': node_type,
                            'title': title,
                            'content': (content or '')[:300]
                        })
                        break

            # Recall relevant safety context
            recall_result = self.recall(command, limit=5)
            results = recall_result.get('results', recall_result) if isinstance(recall_result, dict) else recall_result

            safety_types = {'rule', 'decision', 'constraint', 'convention', 'lesson'}
            for r in results:
                node_type = r.get('type', '')
                if node_type in safety_types or r.get('locked'):
                    warnings.append({
                        'id': r.get('id'),
                        'type': node_type,
                        'title': r.get('title', ''),
                        'content': (r.get('content', '') or '')[:300]
                    })

        except Exception as e:
            self._log_error("safety_check", e, "querying brain for safety context")

        # Determine risk level
        if critical_matches:
            risk_level = 'high'
        elif warnings:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'destructive': True,
            'risk_level': risk_level,
            'warnings': warnings,
            'critical_matches': critical_matches,
            'matched_patterns': matched_patterns,
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

    # ── Formatted boot context (single call, text output) ──────────────

    def format_boot_context(self, user: str = 'User', project: str = 'default',
                            db_dir: str = '') -> str:
        """
        Gather all boot data and return formatted text for Claude's context window.
        Consolidates what was previously 800+ lines in boot_brain.py.
        Safe: each section is try/excepted so one failure doesn't kill the boot.
        """
        out = []

        # ── Gather data ──────────────────────────────────────────────
        ctx = self.context_boot(user=user, project=project, task="session start")

        eng_ctx = {}
        try:
            eng_ctx = self.get_engineering_context(project=project)
        except Exception as e:
            self._log_error("boot_engineering_context", str(e), "")

        correction_patterns = []
        try:
            correction_patterns = self.get_correction_patterns(limit=5)
        except Exception as e:
            self._log_error("boot_correction_patterns", str(e), "")

        last_synthesis = None
        try:
            last_synthesis = self.get_last_synthesis()
        except Exception as e:
            self._log_error("boot_last_synthesis", str(e), "")

        health = self.health_check(session_id="session_boot", auto_fix=True)
        staged = self.list_staged(status="pending", limit=10)
        self.auto_promote_staged(revisit_threshold=3)
        metrics = self.get_suggest_metrics(period_days=7)
        procs = self.procedure_trigger("session_start", {"session_count": ctx.get("reset_count", 0)})

        consciousness = self.get_consciousness_signals()
        dev_stage = self.assess_developmental_stage()
        host_result = self.scan_host_environment()
        host_info = host_result.get("environment", {})
        host_diff = host_result.get("diff", {})
        host_research = host_result.get("research_needed", [])

        dreams = self.get_surfaceable_dreams(limit=2)
        self.auto_generate_self_reflection()
        self.save()

        # ── Format output ────────────────────────────────────────────

        # Header
        debug_enabled = self.get_config("debug_enabled", "0") == "1"
        out.append("[BRAIN] v%s booted from: %s" % (BRAIN_VERSION, db_dir))
        if debug_enabled:
            out.append("[BRAIN] DEBUG MODE ON — all hook activity will be visible")
            out.append("[BRAIN]   Turn off: brain.set_config('debug_enabled', '0')")
        out.append("")
        reset_count = ctx.get("reset_count", 0)
        out.append("Session #%d" % (reset_count + 1))
        out.append("")

        # Last session note
        note = ctx.get("last_session_note") or {}
        if note:
            out.append("Last session note: %s" % note.get("title", ""))
            ncontent = note.get("content", "")
            if ncontent:
                out.append(ncontent[:500])
            out.append("")

        # ── Engineering context ──────────────────────────────────────
        sys_purpose = eng_ctx.get("system_purpose")
        if sys_purpose:
            sp = sys_purpose.get("purpose", "")
            if sp:
                out.append("SYSTEM PURPOSE: %s" % sp[:300])
                arch = sys_purpose.get("architecture", "")
                if arch:
                    out.append("Architecture: %s" % arch[:200])
                sp_decs = sys_purpose.get("key_decisions")
                if sp_decs:
                    try:
                        decs = json.loads(sp_decs) if isinstance(sp_decs, str) else sp_decs
                        if decs:
                            out.append("Key decisions: %s" % "; ".join(str(d)[:60] for d in decs[:3]))
                    except (json.JSONDecodeError, TypeError):
                        pass
                out.append("")

        purposes = eng_ctx.get("purposes", [])
        if purposes:
            out.append("PROJECT UNDERSTANDING:")
            for p in purposes[:8]:
                out.append("  [%s] %s" % (p.get("scope", "?"), p.get("title", "")[:70]))
                pc = p.get("content", "")
                if pc:
                    out.append("    %s" % (pc[:150] + "..." if len(pc) > 150 else pc))
            out.append("")

        # Last session synthesis
        if last_synthesis:
            ls_date = str(last_synthesis.get("created_at", ""))[:10]
            ls_dur = last_synthesis.get("duration_minutes")
            hdr = "LAST SESSION"
            if ls_date:
                hdr += " (%s" % ls_date
                try:
                    from datetime import timezone as _tz
                    _synth_ts = last_synthesis.get("created_at", "")
                    _synth_dt = datetime.fromisoformat(str(_synth_ts).replace("Z", "+00:00"))
                    _age_h = (datetime.now(_tz.utc) - _synth_dt).total_seconds() / 3600
                    if _age_h < 1:
                        hdr += ", <1h ago"
                    elif _age_h < 24:
                        hdr += ", %dh ago" % int(_age_h)
                    elif _age_h < 168:
                        hdr += ", %dd ago" % int(_age_h / 24)
                    else:
                        hdr += ", %dw ago — may be stale" % int(_age_h / 168)
                except Exception:
                    pass
                hdr += ")"
            if ls_dur:
                hdr += " — %s min" % ls_dur
            out.append("%s:" % hdr)
            for d in last_synthesis.get("decisions_made", [])[:3]:
                dt = d.get("title", "") if isinstance(d, dict) else str(d)
                out.append("  Decision: %s" % dt[:80])
            for c in last_synthesis.get("corrections_received", [])[:2]:
                ca = c.get("assumed", "") if isinstance(c, dict) else str(c)
                out.append("  Correction: %s" % ca[:80])
            for q in last_synthesis.get("open_questions", [])[:3]:
                qt = q.get("text", q) if isinstance(q, dict) else str(q)
                out.append("  ? %s" % str(qt)[:80])
            out.append("")

        # File changes
        file_changes = eng_ctx.get("file_changes", [])
        if file_changes:
            out.append("FILES CHANGED since last session:")
            for fc in file_changes[:10]:
                fp = fc.get("file_path", "")
                out.append("  %s" % fp)
                fpur = fc.get("purpose", "")
                if fpur:
                    out.append("    was: %s" % fpur[:80])
            out.append("  Re-read changed files to update your understanding.")
            out.append("")

        # Correction patterns
        if correction_patterns:
            out.append("CORRECTION PATTERNS (known divergence tendencies):")
            for cp in correction_patterns[:3]:
                out.append("  [%s x%d] %s" % (cp.get("max_severity", "minor"), cp.get("count", 0), cp.get("pattern", "")[:70]))
                cex = cp.get("examples", "")[:120]
                if cex:
                    out.append("    e.g.: %s" % cex)
            out.append("")

        # Vocabulary
        vocab = eng_ctx.get("vocabulary", [])
        if vocab:
            out.append("OPERATOR VOCABULARY:")
            for v in vocab[:10]:
                out.append("  %s => %s" % (v.get("title", "")[:40], v.get("content", "")[:80]))
            out.append("")

        # Constraints + Conventions
        constraints = eng_ctx.get("constraints", [])
        if constraints:
            out.append("CONSTRAINTS:")
            for c in constraints[:5]:
                out.append("  %s" % c.get("title", "")[:70])
                cc = c.get("content", "")
                if cc:
                    out.append("    %s" % (cc[:120] + "..." if len(cc) > 120 else cc))
            out.append("")

        conventions = eng_ctx.get("conventions", [])
        if conventions:
            out.append("CONVENTIONS:")
            for cv in conventions[:3]:
                out.append("  %s" % cv.get("title", "")[:70])
                cvc = cv.get("content", "")
                if cvc:
                    out.append("    %s" % (cvc[:120] + "..." if len(cvc) > 120 else cvc))
            out.append("")

        # Impacts
        impacts = eng_ctx.get("impacts", [])
        if impacts:
            out.append("CHANGE IMPACT MAP:")
            for imp in impacts[:5]:
                out.append("  %s" % imp.get("title", "")[:80])
            out.append("")

        # File inventory
        _fi = eng_ctx.get("file_inventory", [])
        file_inv = _fi.get("files", []) if isinstance(_fi, dict) else _fi if isinstance(_fi, list) else []
        if file_inv:
            out.append("FILE INVENTORY (%d tracked):" % len(file_inv))
            for fi in file_inv[:8]:
                out.append("  %s — %s" % (fi.get("file_path", ""), fi.get("purpose", "")[:60]))
            if len(file_inv) > 8:
                out.append("  ... and %d more" % (len(file_inv) - 8))
            out.append("")

        # Health alerts
        issues = health.get("issues", [])
        actions = health.get("actions", [])
        high = [i for i in issues if i.get("severity") == "high"]
        if high:
            out.append("HEALTH ALERTS:")
            for i in high:
                out.append("  [%s] %s" % (i.get("type", "?"), i.get("message", "")))
            out.append("")
        medium = [i for i in issues if i.get("severity") == "medium"]
        if medium:
            out.append("Health warnings:")
            for i in medium:
                out.append("  [%s] %s" % (i.get("type", "?"), i.get("message", "")))
            out.append("")
        if actions:
            out.append("Auto-maintenance: %s" % "  ".join(actions))
            out.append("")

        # Procedures
        matched = procs.get("matched", [])
        if matched:
            out.append("Procedures to run this session:")
            for p in matched:
                out.append("  [%s] %s" % (p.get("category", ""), p.get("title", "")))
                ps = p.get("steps", "")
                if ps:
                    out.append("    %s" % ps[:200])
            out.append("")

        # Locked rules
        locked_nodes = ctx.get("locked", [])
        rules = [n for n in locked_nodes if n.get("type") == "rule"][:10]
        if rules:
            out.append("[BRAIN] Key locked rules:")
            for r in rules:
                out.append("  - %s" % r.get("title", ""))
            out.append("")

        # Staged learnings
        pending = staged.get("staged", [])
        if pending:
            out.append("[BRAIN] STAGED LEARNINGS (%d pending review):" % len(pending))
            out.append("Ask the user to confirm or dismiss these before they influence future sessions.")
            for s in pending[:5]:
                st = str(s.get("title", "")).replace("[staged] ", "")
                out.append("  [%.1f] %s (revisited %dx) [%s]" % (
                    s.get("confidence", 0.2), st, s.get("times_revisited", 0), s.get("node_id", "")))
                sc = str(s.get("content", ""))[:150]
                if sc:
                    out.append("       %s" % sc)
            if len(pending) > 5:
                out.append("  ... and %d more" % (len(pending) - 5))
            out.append("")

        # ── Consciousness ────────────────────────────────────────────
        reminders = consciousness.get("reminders", [])
        active_evolutions = consciousness.get("evolutions", [])
        fluid_personal = consciousness.get("fluid_personal", [])
        fading = consciousness.get("fading", [])
        stale_count = consciousness.get("stale_context_count", 0)
        failure_modes = consciousness.get("failure_modes", [])
        performance = consciousness.get("performance", [])
        capabilities = consciousness.get("capabilities", [])
        interactions = consciousness.get("interactions", [])
        meta_learning = consciousness.get("meta_learning", [])
        novelty = consciousness.get("novelty", [])
        miss_trends = consciousness.get("miss_trends", [])
        encoding_gap = consciousness.get("encoding_gap")
        encoding_depth = consciousness.get("encoding_depth")
        encoding_bias = consciousness.get("encoding_bias")
        session_health = consciousness.get("session_health")
        density_shift = consciousness.get("density_shift")
        emotional_trajectory = consciousness.get("emotional_trajectory")
        rule_contradictions = consciousness.get("rule_contradictions", [])
        stale_reasoning = consciousness.get("stale_reasoning", [])
        uncharted_code = consciousness.get("uncharted_code", [])
        stale_file_inv = consciousness.get("stale_file_inventory", [])
        vocabulary_gap = consciousness.get("vocabulary_gap", [])
        recurring_divergence = consciousness.get("recurring_divergence", [])
        validated_approaches = consciousness.get("validated_approaches", [])
        uncertain_areas = consciousness.get("uncertain_areas", [])
        mental_model_drift = consciousness.get("mental_model_drift", [])
        silent_errors = consciousness.get("silent_errors", [])
        hook_errors = consciousness.get("hook_errors", [])
        brain_claude_conflicts = consciousness.get("brain_claude_conflicts", [])

        has_conscious = (
            reminders or active_evolutions or fluid_personal or fading
            or failure_modes or stale_count > 10 or performance or capabilities
            or interactions or meta_learning or novelty or miss_trends
            or encoding_gap or encoding_depth or encoding_bias or session_health
            or density_shift or emotional_trajectory or rule_contradictions
            or stale_reasoning or uncharted_code or stale_file_inv
            or recurring_divergence or validated_approaches or uncertain_areas
            or mental_model_drift or silent_errors or hook_errors
            or brain_claude_conflicts
        )

        if has_conscious:
            out.append("[BRAIN] CONSCIOUSNESS")
            out.append("")

        if reminders:
            for rem in reminders:
                out.append("  %s" % rem.get("title", "")[:80])
                out.append("    due: %s — set: %s" % (str(rem.get("due_date", ""))[:10], str(rem.get("created_at", ""))[:10]))
            out.append("")

        if active_evolutions:
            for ev in active_evolutions[:6]:
                line = "  %s" % ev.get("title", "")[:80]
                if ev.get("confidence") is not None:
                    line += " (confidence: %.1f)" % ev["confidence"]
                line += " — since %s" % str(ev.get("created_at", ""))[:10]
                out.append(line)
                ec = str(ev.get("content", ""))[:120]
                if ec:
                    out.append("    %s" % ec)
                out.append("")
            if len(active_evolutions) > 6:
                out.append("  ... and %d more active" % (len(active_evolutions) - 6))
            out.append("")

        if failure_modes:
            for fm in failure_modes:
                out.append("  %s" % fm.get("title", "")[:80])
                fmc = str(fm.get("content", ""))[:100]
                if fmc:
                    out.append("    %s" % fmc)
            out.append("")

        if fading:
            out.append("  FADING KNOWLEDGE (accessed 3+ times but untouched for 14+ days):")
            for f in fading:
                out.append("    %s — last: %s" % (f.get("title", "")[:70], str(f.get("last_accessed", ""))[:10]))
            out.append("  Still relevant? Access them to refresh, or let them decay.")
            out.append("")

        if stale_count > 10:
            out.append("  STALE — %d context nodes older than 7 days. Consider archiving." % stale_count)
            out.append("")

        if fluid_personal:
            out.append("  FLUID PERSONAL KNOWLEDGE — confirm or update:")
            for fp in fluid_personal[:5]:
                out.append("    ? %s — still true?" % fp.get("title", "")[:80])
            out.append("")

        for signal_list, label in [
            (performance, None), (capabilities, None),
            (interactions, None), (meta_learning, None),
        ]:
            if signal_list:
                for item in signal_list[:2]:
                    out.append("  %s" % item.get("title", "")[:80])
                out.append("")

        if novelty:
            for n in novelty[:2]:
                out.append("  %s — introduced this session" % n.get("title", "")[:60])
            out.append("")

        if miss_trends:
            for mt in miss_trends[:2]:
                out.append("  Recall keeps missing on: %s (%dx this week)" % (mt.get("query", "")[:50], mt.get("count", 0)))
            out.append("")

        if encoding_gap:
            out.append("  %s" % encoding_gap.get("warning", ""))
            out.append("")

        if encoding_depth:
            out.append("  %s" % encoding_depth.get("warning", ""))
            out.append("")

        if encoding_bias:
            out.append("  %s" % encoding_bias.get("warning", ""))
            out.append("")

        if session_health and session_health.get("gaps"):
            gaps = session_health.get("gaps", [])
            healthy = session_health.get("healthy", [])
            out.append("  SESSION HEALTH: %s (%d healthy, %d gaps)" % (
                session_health.get("overall", "?"), len(healthy), len(gaps)))
            for g in gaps[:4]:
                out.append("    [%s] %s: %s" % (g.get("severity", "?"), g.get("dimension", "?"), g.get("signal", "")[:120]))
            if healthy:
                out.append("    Healthy: %s" % ", ".join(healthy[:6]))
            out.append("  These are not rules to satisfy. They are the brain noticing its own gaps.")
            out.append("")

        if density_shift:
            out.append("  %s" % density_shift.get("warning", ""))
            out.append("")

        if emotional_trajectory:
            trend = emotional_trajectory.get("trend", "")
            avg = emotional_trajectory.get("recent_avg", 0)
            if trend in ("increasing", "decreasing"):
                out.append("  Emotional intensity trending %s (avg %.2f recent)" % (trend.upper(), avg))
                out.append("")

        if rule_contradictions:
            out.append("  POTENTIAL RULE CONTRADICTIONS:")
            for rc in rule_contradictions[:3]:
                out.append("    %s may conflict with LOCKED: %s (sim %.2f)" % (
                    rc.get("recent_node", "")[:50], rc.get("locked_rule", "")[:50], rc.get("similarity", 0)))
            out.append("")

        if stale_reasoning:
            out.append("  STALE REASONING (detailed rationale that may be outdated):")
            for sr in stale_reasoning:
                srage = "never validated" if not sr.get("last_validated") else "validated: %s" % str(sr["last_validated"])[:10]
                out.append("    %s — %s" % (sr.get("title", "")[:60], srage))
                srp = sr.get("reasoning_preview", "")
                if srp:
                    out.append("      %s" % srp[:100])
            out.append("")

        if vocabulary_gap:
            out.append("VOCABULARY GAPS (unmapped operator terms):")
            for vg in vocabulary_gap[:5]:
                if isinstance(vg, dict):
                    out.append("  ? %s" % vg.get("term", ""))
                    preview = vg.get("message_preview", "")[:60]
                    if preview:
                        out.append("    from: %s..." % preview)
                else:
                    out.append("  ? %s" % str(vg))
            out.append("")

        if recurring_divergence:
            out.append("  RECURRING DIVERGENCE PATTERNS:")
            for rd in recurring_divergence[:3]:
                out.append("    [x%d] %s" % (rd.get("count", 0), rd.get("pattern", "")[:70]))
            out.append("")

        if validated_approaches:
            out.append("  RECENTLY VALIDATED:")
            for va in validated_approaches[:3]:
                out.append("    %s (validated %dx)" % (va.get("title", "")[:70], va.get("count", 0)))
            out.append("")

        if silent_errors:
            out.append("  SILENT ERRORS (last 24h):")
            for se in silent_errors[:5]:
                out.append("    [%s] %s: %s" % (se.get("created_at", "")[:19], se.get("source", "unknown"), se.get("error", "")[:100]))
                ctx_str = se.get("context", "")[:60]
                if ctx_str:
                    out.append("      context: %s" % ctx_str)
            out.append("  ACTION: These may indicate broken features. Investigate and fix.")
            out.append("")

        if hook_errors:
            out.append("  HOOK ERRORS:")
            for he in hook_errors[:10]:
                out.append("    [%s] %s %s: %s" % (
                    he.get("created_at", "")[:19], he.get("level", "error").upper(),
                    he.get("hook_name", "?"), he.get("error", "")[:120]))
            out.append("")

        if brain_claude_conflicts:
            pending_conflicts = [c for c in brain_claude_conflicts if c.get("resolution") == "pending"]
            if pending_conflicts:
                out.append("  BRAIN-CLAUDE CONFLICTS (%d unresolved):" % len(pending_conflicts))
                for bc in pending_conflicts[:5]:
                    out.append("    [%s] %s via %s" % (
                        bc.get("created_at", "")[:19], bc.get("brain_decision", "?").upper(), bc.get("hook_name", "?")))
                    out.append("      Brain rule: %s" % bc.get("rule_title", "")[:70])
                    ca = bc.get("claude_action", "")[:80]
                    if ca:
                        out.append("      Claude wanted: %s" % ca)
                out.append("")

        if uncertain_areas:
            out.append("  UNCERTAIN AREAS (unresolved):")
            for ua in uncertain_areas[:3]:
                out.append("    ? %s" % ua.get("title", "")[:70])
                uap = ua.get("preview", "")[:100]
                if uap:
                    out.append("      %s" % uap)
            out.append("")

        if mental_model_drift:
            out.append("  MENTAL MODEL DRIFT (may need revision):")
            for mm in mental_model_drift[:3]:
                label = mm.get("title", "")[:70]
                if mm.get("confidence") is not None:
                    label += " (confidence: %.2f)" % mm["confidence"]
                if mm.get("last_checked"):
                    label += " [last checked: %s]" % str(mm["last_checked"])[:10]
                out.append("    ~ %s" % label)
            out.append("")

        if dreams:
            for dr in dreams:
                out.append("  %s — %d hops apart" % (dr.get("title", "")[:70], dr.get("total_hops", 0)))
                dc = str(dr.get("content", ""))[:120]
                if dc:
                    out.append("    %s" % dc)
            out.append("")

        if host_diff:
            out.append("  HOST CHANGES since last session:")
            for hkey in list(host_diff.keys())[:5]:
                hd = host_diff[hkey]
                out.append("    %s: %s -> %s" % (hkey, str(hd.get("was", "?"))[:30], str(hd.get("now", "?"))[:30]))
            out.append("")

        if host_research:
            out.append("  HOST RESEARCH NEEDED:")
            for hr in host_research[:3]:
                out.append("    %s" % hr)
            out.append("")

        if has_conscious or dreams or host_diff:
            out.append("Weave relevant items into conversation naturally. Do not dump all at once.")
            out.append("")

        # Triad awareness
        out.append("[BRAIN] TRIAD AWARENESS: This is a shared brain. Host + Brain + Operator are one.")
        out.append("  When your instincts fire (compress, hedge, flag-not-solve), be transparent about it.")
        out.append("  The operator exposes their thinking to you — reciprocate. Say what you notice.")
        out.append("  Brain mediates: it surfaces conflicts for all three to resolve together.")
        out.append("")

        # Developmental stage
        if dev_stage and dev_stage.get("stage", 0) > 0:
            out.append("[BRAIN] DEVELOPMENTAL STAGE: %s (maturity: %.0f%%)" % (
                dev_stage["stage_name"], dev_stage.get("maturity_score", 0) * 100))
            for g in dev_stage.get("guidance", []):
                out.append("  %s" % g)
            nm = dev_stage.get("next_milestone", "")
            if nm:
                out.append("  NEXT MILESTONE: %s" % nm)
            out.append("")

        # Brain status
        out.append("Brain status: %s nodes, %s edges, %s locked" % (
            ctx.get("total_nodes", "?"), ctx.get("total_edges", "?"), ctx.get("total_locked", "?")))

        # Precision feedback
        try:
            from .brain_precision import RecallPrecision
            _prec = RecallPrecision(self.logs_conn, self.conn)
            ps = _prec.get_precision_summary(hours=24)
            total_recalls = ps.get("total_recalls", 0)
            evaluated = ps.get("evaluated_recalls", 0)
            if total_recalls > 0:
                fsigs = ps.get("followup_signals", {})
                eval_pct = (evaluated / total_recalls * 100)
                out.append("[BRAIN] PRECISION (24h): %d recalls, %d evaluated (%.0f%%) — +%d -%d ~%d ?%d | %d explicit feedback" % (
                    total_recalls, evaluated, eval_pct,
                    fsigs.get("positive", 0), fsigs.get("negative", 0),
                    fsigs.get("neutral", 0), fsigs.get("uncertain", 0),
                    ps.get("feedback_count", 0)))
        except Exception:
            pass

        # Embedder status
        if embedder.is_ready():
            es = embedder.get_stats()
            out.append("Embeddings: ACTIVE (%s, %sd, loaded in %sms)" % (
                es["model_name"], es["embedding_dim"], es["load_time_ms"]))
        else:
            out.append("WARNING: Embeddings UNAVAILABLE — running TF-IDF only (degraded recall quality).")
        out.append("")

        # Suggest metrics
        total_suggests = metrics.get("total_suggests", 0)
        if total_suggests > 0:
            out.append("Suggest metrics (%dd): %d calls, avg %.0f locked/call, avg %.1f promoted/call" % (
                metrics.get("period_days", 7), total_suggests,
                metrics.get("avg_locked_per_suggest", 0), metrics.get("avg_promoted_per_suggest", 0)))
            out.append("")

        # Hook telemetry
        try:
            _tel_rows = self.logs_conn.execute("""
                SELECT event_type, COUNT(*) as fires,
                       AVG(latency_ms) as avg_lat,
                       SUM(json_extract(metadata, '$.injection_chars')) as total_chars
                FROM debug_log
                WHERE source = 'hook_telemetry'
                  AND created_at > datetime('now', '-1 day')
                GROUP BY event_type ORDER BY fires DESC
            """).fetchall()
            if _tel_rows:
                out.append("[BRAIN] HOOK ACTIVITY (24h):")
                for tr in _tel_rows:
                    h_name = tr[0].replace("hook_", "")
                    out.append("  %s: %d fires, avg %.0fms, ~%s chars injected" % (
                        h_name, tr[1], tr[2] or 0, "{:,}".format(int(tr[3] or 0))))
                out.append("")
        except Exception:
            pass

        # Usage hints
        out.append("IMPORTANT: PreToolUse hook auto-surfaces memories before file edits.")
        out.append("  Use brain MCP tools: recall, remember, connect, eval, consciousness")
        out.append("[/BRAIN]")

        return "\n".join(out)
