"""
brain — BrainSurface Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from .schema import BRAIN_VERSION, BRAIN_VERSION_KEY, NODE_TYPES
from .text_processing import split_identifier
from .clock import iso_cutoff
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


class BrainAssemblyMixin:
    """Assembly methods — gathers context for hooks (boot, pre-edit, procedures)."""

    def suggest(self, context: Optional[str] = None, file: Optional[str] = None,
               screen: Optional[str] = None, action: Optional[str] = None,
               project: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Multi-query recall with type boosts, locked node boost, file-term relevance.
        Returns dict with suggestions list and query_count.
        """
        if limit is None:
            limit = self.get_config('suggestion_limit', 5)

        # Pre_edit cost surgery (2026-05-08): suggest() previously fanned out
        # the inputs into up to 7 separate full recalls per call (context +
        # joined filename tokens + each individual token > 2 chars + screen +
        # action). Single-word tokens like 'servers', 'scales', 'edit'
        # produced thousands of low-signal matches per query, and 7×O(N)
        # cosine scans per Edit was the dominant CPU + memory pressure
        # source (watchdog confirmed). Now: ONE query — the most-specific
        # input available. context (e.g., "editing path/to/file.py") wins
        # because it's the whole signal in one string; the fan-out lost
        # precision while burning compute.
        queries = []
        if context:
            queries.append(context)
        elif file:
            # Fallback when caller didn't compose a context — use the joined
            # tokens. Skip the per-token fan-out for the same precision-vs-
            # cost reason.
            file_tokens = split_identifier(file)
            if file_tokens:
                queries.append(' '.join(file_tokens))
        elif screen:
            queries.append(screen)
        elif action:
            queries.append(action)

        if not queries:
            return {'suggestions': [], 'reason': 'no context provided'}

        # Run recall for each query
        seen = set()
        all_results = []
        pool_multiplier = self.get_config('recall_pool_multiplier', 2)
        recall_limit = max(limit * pool_multiplier, 15)

        for q in queries:
            result = self.recall(query=q, limit=recall_limit, source='internal')
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
                    SELECT DISTINCT n.id, n.type, n.title, n.content, n.activation,
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
                            'activation': row[4], 'stability': row[5],
                            'access_count': row[6], 'locked': row[7] == 1, 'archived': row[8] == 1,
                            'last_accessed': row[9], 'created_at': row[10],
                            '_edge_neighbor': True
                        })
        except Exception as e:
            self._log_error('suggest_edge_neighbors', e, 'fetching edge neighbors for suggestions')

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

            # File relevance — match against title + content snippet
            # (keywords column dropped in schema v28)
            file_relevance = 0
            if file_terms:
                node_text = f"{r.get('title', '')} {(r.get('content') or '')[:300]}".lower()
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
                   any(t in f"{r.get('title', '')} {(r.get('content') or '')[:300]}".lower() for t in file_terms)
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

        # suggest_log writes REMOVED 2026-04-05 — table dropped

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
            SELECT id, type, title, content FROM nodes
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
                'content': r[3],
                '_critical': True
            })

        # 1. Get locked nodes with full content for top N
        # Project-scoped: return nodes for this project + global (NULL project)
        locked = self.conn.execute('''
            SELECT id, type, title, content FROM nodes
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
                'content': r[3]
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
            SELECT id, type, title, content, activation, last_accessed FROM nodes
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
            recall_result = self.recall(query=query, limit=max_recall, project=project, source='internal')
            recalled = recall_result.get('results', recall_result) if isinstance(recall_result, dict) else recall_result
            for r in recalled:
                if r['id'] not in seen:
                    seen.add(r['id'])
                    results['recalled'].append({
                        'id': r['id'], 'type': r.get('type'),
                        'title': r.get('title'), 'content': r.get('content')
                    })

        # Get total locked count
        total_locked = self._nodes.count_locked()

        return {
            'brain_version': BRAIN_VERSION,
            'total_nodes': self._get_node_count(),
            'total_edges': self._get_edge_count(),
            'total_locked': total_locked,
            'locked_shown': len(results['locked']),
            'has_more_locked': total_locked > max_locked,
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
        except Exception as e:
            self._log_error('validate_schema_version', e, 'checking schema version')

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
        except Exception as e:
            self._log_error('validate_db_sizes', e, 'checking database file sizes')

        return warnings

    def health_check(self, session_id: str = 'boot', auto_fix: bool = True) -> Dict[str, Any]:
        """
        Check brain health: orphaned locked nodes, stale contexts, stale staged learnings.
        Auto-fix: enrich missed nodes, promote staged learnings.
        """
        issues = []
        actions = []
        ts = self.now()

        # compaction_boundary node check REMOVED 2026-05-03 — pre/post-compact
        # hooks deleted; legacy node-style boundaries (pre-2026-04-13) no
        # longer produced. miss_log check REMOVED 2026-04-06.

        # Check for orphaned locked nodes
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
            AND created_at < ?
        ''', (iso_cutoff(days=7),)).fetchone()
        stale_count = stale_count_row[0] if stale_count_row else 0

        if stale_count > 10:
            issues.append({
                'type': 'stale_contexts',
                'severity': 'low',
                'message': f'{stale_count} context nodes older than 7 days.'
            })
            if auto_fix:
                stale_ids = [r[0] for r in self.conn.execute('''
                    SELECT id FROM nodes
                    WHERE type = 'context' AND locked = 0 AND archived = 0
                    AND created_at < ?
                ''', (iso_cutoff(days=14),)).fetchall()]
                for sid in stale_ids:
                    self.archive_node(sid, archived_by='hook:integrity',
                                      reason='context node older than 14 days')
                actions.append('Auto-archived %d context nodes older than 14 days' % len(stale_ids))

        # 5-7: miss_log auto-enrich, staged_learnings auto-promote, stale staged check
        # ALL REMOVED 2026-04-06 — tables dropped

        return {
            'healthy': not any(i['severity'] == 'high' for i in issues),
            'issues': issues,
            'actions': actions,
            'checked_at': ts
        }

    # list_staged, confirm_staged, dismiss_staged, auto_promote_staged
    # REMOVED 2026-04-06 — staged_learnings table dropped

    def pre_edit(self, file: str, tool_name: str = 'Edit',
                 ctx=None) -> dict:
        """
        Batch pre-edit call combining all lookups into one.
        Replaces 8 sequential HTTP calls from the old architecture.

        Args:
            file: Filename being edited
            tool_name: 'Edit' or 'Write'

        Returns:
            Dict with suggestions, procedures, context_files, encoding health, timings

        Note (2026-05-08): the pre_edit-level cache that lived here was
        removed in favor of a recall-layer cache (brain.recall now has a
        result cache + single-flight gate). The recall-layer cache covers
        every recall caller (pre_edit, pre_bash_safety, hook_recall,
        manual MCP) instead of just pre_edit, and the cache is keyed by
        the actual recall query — more correct than caching by filename.
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

        # 3. Encoding health — per-session counters via SessionContext
        if ctx is not None:
            boot_time = ctx.boot_time or self.now()
            remembers = ctx.remember_count
            edits_checked = ctx.edit_check_count
        else:
            boot_time = self.now()
            remembers = 0
            edits_checked = 0

        # Compute session minutes
        try:
            from datetime import datetime as _dt
            boot_dt = _dt.fromisoformat(boot_time.replace('Z', '+00:00'))
            now_dt = _dt.now(boot_dt.tzinfo) if boot_dt.tzinfo else _dt.utcnow()
            session_minutes = (now_dt - boot_dt).total_seconds() / 60
        except Exception as e:
            self._log_error('pre_edit_session_minutes', e, 'computing session minutes from boot time')
            session_minutes = 0

        # Determine encoding health status.
        # Note: minutes_since_last_remember was historically computed from
        # `last_remember_at` — a key never written by any writer. The STALE
        # branch never fired. Health collapses to NONE-or-OK by edit_check
        # count only.
        edits_since = edits_checked  # approximate — reset on each remember
        if remembers == 0 and session_minutes > 3:
            encoding_health = 'NONE'
        elif edits_since > 8:
            encoding_health = 'STALE'
        else:
            encoding_health = 'OK'

        # 4. Context files (nodes of type 'file' matching the edited filename)
        context_files = []
        try:
            # Match against title only (keywords column dropped in v28).
            # File-type nodes encode their topic in the title; content
            # snippet carries the body.
            cursor = self.conn.execute(
                "SELECT id, title, content, updated_at FROM nodes WHERE type = 'file' AND archived = 0 AND title LIKE ? LIMIT 3",
                (f'%{file}%',)
            )
            for row in cursor.fetchall():
                context_files.append({
                    'id': row[0], 'title': row[1], 'summary': (row[2] or '')[:200],
                    'topic': '', 'last_updated': row[3],
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
            recall_result = self.recall(command, limit=5, source='internal')
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
            # Procedure-trigger matching: keywords column dropped in v28
            # along with the broken auto-extractor. Trigger types
            # ("session_start", "pre_edit", "pre_compact") and file_name
            # filters now match against title + content only. Procedure
            # nodes that need explicit tagging should include the trigger
            # in their content body (e.g. "Run on session_start").
            cursor = self.conn.execute(
                "SELECT id, title, content FROM nodes WHERE type = 'procedure' AND archived = 0 AND locked = 1"
            )
            for row in cursor.fetchall():
                node_id, title, content = row
                content_lower = (content or '').lower()
                title_lower = (title or '').lower()
                # Combined search surface for trigger matching.
                search_text = content_lower + ' ' + title_lower

                # Check if procedure matches trigger type
                if trigger_type in search_text:
                    # Parse procedure content for steps
                    steps = content or ''
                    category = 'general'
                    if 'session_start' in search_text:
                        category = 'session_start'
                    elif 'pre_edit' in search_text:
                        category = 'pre_edit'
                    elif 'pre_compact' in search_text:
                        category = 'pre_compact'

                    # Check file-specific procedures
                    if trigger_type == 'pre_edit' and 'file' in context:
                        file_name = context['file'].lower()
                        if file_name not in search_text:
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

    # ── Self-knowledge for boot ──────────────

    def fetch_boot_nodes(self, limit: int = 3) -> list:
        """Fetch boot nodes — messages from previous sessions' Claude to this one."""
        rows = self.conn.execute('''
            SELECT id, title, content, created_at FROM nodes
            WHERE type = 'boot' AND archived = 0
            ORDER BY created_at DESC LIMIT ?
        ''', (limit,)).fetchall()
        return [{'id': r[0], 'title': r[1], 'content': r[2], 'created_at': r[3]} for r in rows]

    # 2026-05-02 (Frame Phase 2.5): fetch_self_knowledge removed — was the
    # data source for the old boot's "PATTERNS YOU FALL INTO" section.
    # Frame's Operator section (read via brain.aspects.identity_bearing)
    # covers the same need with a cleaner abstraction. If a future consumer
    # wants self-reflective node lookup, prefer filter_nodes(field='type',
    # include=brain.aspects.identity_bearing.node_types) or similar via
    # AspectRegistry.




    # ── Formatted boot context ──────────────

    def format_boot_context(self, user: str = 'User', project: str = 'default',
                            db_dir: str = '', session_id: str = '') -> str:
        """
        Gather all boot data and return formatted text for Claude's context window.
        Delegates to BrainVoice.render_boot() — thin wrapper for backwards compat.

        Returns a single string with both [BRAIN] and [BRAIN-To-*] channels merged.
        For raw channel dict, call BrainVoice(self).render_boot() directly.

        2026-05-02 (Frame Phase 2.5): session_id threaded through so
        render_boot_v2 can build the Frame for THIS session via
        ctx.get_frame(brain).
        """
        from .brain_voice import BrainVoice
        voice = BrainVoice(self)
        rendered = voice.render_boot_v2(user, project, db_dir, session_id=session_id)
        return voice.wrap_for_hook(rendered['for_claude'], rendered.get('for_operator'))


    # auto_encode, track_response, detect_vocab_gaps: deleted 2026-03-28
    # Replaced by Stop agent hook + daemon gating (see daemon_hooks.py)

    # ═══════════════════════════════════════════════════════════════
    # v8: Invisible encoding — conversation stream
    # ═══════════════════════════════════════════════════════════════

    # store_exchange + resolve_recent_pending REMOVED 2026-04-05
    # message_stream table deleted. S0 traces capture full content.
    # Escalation tracking was redundant (encoding agent reads from traces).

    # ═══════════════════════════════════════════════════════════════
    # v8: Consolidation detection — find overlapping nodes
    # ═══════════════════════════════════════════════════════════════

    def detect_consolidation_candidates(self, similarity_threshold: float = 0.85,
                                         min_age_hours: int = 24,
                                         max_pairs: int = 10) -> int:
        """Scan for duplicate/overlapping nodes. Queue for LLM consolidation.

        Called by idle_maintenance. Returns count of new pairs found.

        Scoping rules:
        - Only compare nodes of the same type (reduces O(n²))
        - Skip pairs where both nodes were created within min_age_hours
        - Skip archived nodes
        - Skip pairs already in pending_consolidation
        """
        from . import embedder
        from .dal import VectorDAL, LogsDAL
        from datetime import datetime, timezone, timedelta

        if not embedder.is_ready():
            self._log_error("consolidation_detect", Exception("Embedder not ready"),
                            "Cannot detect consolidation without embedder")
            return 0

        # Get all primary embeddings for the active model — stale-model rows
        # would pollute the consolidation cluster scores.
        vdal = self._vec_dal
        _active_model = embedder.stats.get('model_name') or None
        all_embeddings = [{'node_id': r['node_id'], 'embedding': r['embedding']}
                          for r in vdal.get_all_vectors(
                              vector_types=['_primary'],
                              model=_active_model)]
        if not all_embeddings:
            return 0

        # Build node_id → embedding map
        emb_map = {}
        for item in all_embeddings:
            emb_map[item['node_id']] = item['embedding']

        # Get node types and creation dates
        rows = self.conn.execute(
            'SELECT id, type, created_at FROM nodes WHERE archived = 0'
        ).fetchall()

        # Group by type
        type_groups = {}
        node_dates = {}
        for nid, ntype, created_at in rows:
            if nid not in emb_map:
                continue  # no embedding, skip
            type_groups.setdefault(ntype, []).append(nid)
            node_dates[nid] = created_at

        # Compare within each type group
        logs_dal = LogsDAL(self.logs_conn)
        new_pairs = 0
        cutoff_hours = min_age_hours

        for ntype, node_ids in type_groups.items():
            if len(node_ids) < 2:
                continue

            for i in range(len(node_ids)):
                if new_pairs >= max_pairs:
                    break
                for j in range(i + 1, len(node_ids)):
                    if new_pairs >= max_pairs:
                        break

                    nid_a, nid_b = node_ids[i], node_ids[j]

                    # Skip if created too close together (same session)
                    date_a = node_dates.get(nid_a, '')
                    date_b = node_dates.get(nid_b, '')
                    if date_a and date_b:
                        try:
                            dt_a = datetime.fromisoformat(date_a.replace('Z', '+00:00'))
                            dt_b = datetime.fromisoformat(date_b.replace('Z', '+00:00'))
                            if abs((dt_a - dt_b).total_seconds()) < cutoff_hours * 3600:
                                continue
                        except (ValueError, TypeError) as e:
                            self._log_error('consolidation_date_parse', e, 'nodes %s/%s' % (nid_a[:8], nid_b[:8]))

                    # Compute similarity
                    emb_a = emb_map[nid_a]
                    emb_b = emb_map[nid_b]
                    sim = embedder.cosine_similarity(emb_a, emb_b)

                    if sim >= similarity_threshold:
                        # Order by creation date (older = a, newer = b)
                        if date_a > date_b:
                            nid_a, nid_b = nid_b, nid_a
                        if logs_dal.queue_consolidation(nid_a, nid_b, sim):
                            new_pairs += 1

        return new_pairs

    # _get_or_create_precision REMOVED 2026-04-05 — brain_precision.py deleted
