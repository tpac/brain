"""
brain — BrainEngineering Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from .brain_constants import TYPE_CONFIDENCE, EXTERNAL_CLAIM_KEYWORDS
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import json
import re
import time
import uuid


class BrainEngineeringMixin:
    """Engineering methods for Brain."""

    def remember_purpose(self, title: str, content: str, scope: str = 'system',
                         project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """What something is and why it exists. The warm-up killer.

        Scope: system | module | file | function
        Examples:
          system: "brain is a Claude Code plugin for persistent AI memory"
          file: "brain.py (5500 lines) — core engine: schema, remember(), recall(), consciousness"
          function: "recall_with_embeddings() — 90/10 embedding/keyword blend, returns ranked nodes"
        """
        return self.remember_rich(
            type='purpose', title=title, content=content,
            scope=scope, source_attribution='claude_inferred',
            project=project, locked=True, **kwargs)

    def remember_mechanism(self, title: str, content: str, scope: str = 'system',
                           steps: Optional[List[str]] = None,
                           data_flow: Optional[str] = None,
                           project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """How something works: flows, algorithms, interactions.

        Examples:
          "Recall pipeline: embed query → cosine scan → keyword fallback → blend → rank → hydrate"
        """
        # Enrich content with structured steps/flow
        enriched = content
        if steps:
            enriched += '\n\nSteps: ' + ' → '.join(steps)
        if data_flow:
            enriched += '\n\nData flow: ' + data_flow
        return self.remember_rich(
            type='mechanism', title=title, content=enriched,
            scope=scope, source_attribution='claude_inferred',
            project=project, **kwargs)

    def remember_impact(self, title: str, if_changed: str, must_check: str,
                        because: str, scope: str = 'cross-file',
                        severity: str = 'medium',
                        project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """What changes ripple where — the connectivity layer.

        Example: if_changed="recall_with_embeddings() output format"
                 must_check="pre-response-recall.sh, boot-brain.sh"
                 because="they parse its return structure"
        """
        content = f'If {if_changed} changes → must check {must_check} because {because}'
        change_impacts = [{'if_modified': if_changed, 'must_check': must_check, 'because': because}]
        return self.remember_rich(
            type='impact', title=title, content=content,
            scope=scope, source_attribution='claude_inferred',
            change_impacts=change_impacts,
            project=project, locked=True,
            emotion=0.3 if severity == 'critical' else 0.1,
            emotion_label='concern' if severity in ('high', 'critical') else 'neutral',
            **kwargs)

    def remember_constraint(self, title: str, content: str, scope: str = 'system',
                            violates_if: Optional[str] = None,
                            project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """What must or must not be done.

        Example: "Never call context_boot() without user/project args"
                 violates_if="context_boot() called with no arguments"
        """
        enriched = content
        if violates_if:
            enriched += f'\n\nViolated when: {violates_if}'
        return self.remember_rich(
            type='constraint', title=title, content=enriched,
            scope=scope, source_attribution='claude_inferred',
            project=project, locked=True, **kwargs)

    def remember_convention(self, title: str, content: str, scope: str = 'project-wide',
                            examples: Optional[List[str]] = None,
                            anti_patterns: Optional[List[str]] = None,
                            project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Patterns, utilities, coding style for a codebase.

        Example: "Error handling in hooks: resolve DB path first, wrap brain imports in try/except"
        """
        enriched = content
        if examples:
            enriched += '\n\nExamples: ' + '; '.join(examples)
        if anti_patterns:
            enriched += '\n\nAnti-patterns: ' + '; '.join(anti_patterns)
        return self.remember_rich(
            type='convention', title=title, content=enriched,
            scope=scope, source_attribution='claude_inferred',
            project=project, **kwargs)

    def remember_lesson(self, title: str, what_happened: str, root_cause: str,
                        fix: str, preventive_principle: str,
                        project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """What went wrong, root cause, fix, preventive principle.

        Replaces both bug_lesson and causal_chain.
        """
        content = (f'What happened: {what_happened}\n'
                   f'Root cause: {root_cause}\n'
                   f'Fix: {fix}\n'
                   f'Principle: {preventive_principle}')
        return self.remember_rich(
            type='lesson', title=title, content=content,
            source_attribution='session_synthesis',
            project=project, locked=True,
            emotion=0.4, emotion_label='emphasis', **kwargs)

    def remember_mental_model(self, title: str, model_description: str,
                              applies_to: Optional[str] = None,
                              confidence: float = 0.7,
                              project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Claude's understanding of how systems/processes work.

        Example: "brain.py has 3 layers: storage (SQLite), intelligence (embeddings + recall),
                  consciousness (signals + evolution)"
        """
        content = model_description
        if applies_to:
            content += f'\n\nApplies to: {applies_to}'
        return self.remember_rich(
            type='mental_model', title=title, content=content,
            source_attribution='claude_inferred',
            confidence_rationale=f'Inferred from code reading (confidence: {confidence})',
            project=project, confidence=confidence, **kwargs)

    def remember_uncertainty(self, title: str, what_unknown: str,
                             why_it_matters: str,
                             project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Where Claude knows it doesn't understand something.

        These feed the curiosity system and guide future sessions.
        """
        content = f'Unknown: {what_unknown}\nWhy it matters: {why_it_matters}'
        return self.remember_rich(
            type='uncertainty', title=title, content=content,
            source_attribution='claude_inferred',
            confidence=0.3, project=project, **kwargs)

    def record_reasoning_trace(self, title: str, steps: List[str],
                               conclusion: str, reusable: bool = True,
                               project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Reusable logic chain — not just the conclusion but the path to it.

        Populates reasoning_chains + reasoning_steps tables.
        """
        content = 'Steps:\n' + '\n'.join(f'  {i+1}. {s}' for i, s in enumerate(steps))
        content += f'\n\nConclusion: {conclusion}'
        result = self.remember_rich(
            type='reasoning_trace', title=title, content=content,
            reasoning=' → '.join(steps) + ' → ' + conclusion,
            source_attribution='claude_inferred',
            project=project, locked=reusable, **kwargs)

        # Also populate reasoning_chains/steps tables
        try:
            chain_id = self._generate_id('chain')
            ts = self.now()
            self.conn.execute(
                '''INSERT INTO reasoning_chains (id, decision_node_id, title, step_count, project, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (chain_id, result['id'], title, len(steps), project, ts)
            )
            for i, step in enumerate(steps):
                self.conn.execute(
                    '''INSERT INTO reasoning_steps (chain_id, step_order, step_type, content, node_id, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (chain_id, i + 1, 'observation', step, result['id'], ts)
                )
            # Final conclusion step
            self.conn.execute(
                '''INSERT INTO reasoning_steps (chain_id, step_order, step_type, content, node_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (chain_id, len(steps) + 1, 'decision', conclusion, result['id'], ts)
            )
            self.conn.commit()
            result['chain_id'] = chain_id
        except Exception as _e:
            self._log_error("record_reasoning_trace", _e, "inserting reasoning steps into reasoning_steps table")
        return result

    def update_file_inventory(self, project: str, file_path: str,
                              purpose: str, key_exports: Optional[List[str]] = None,
                              dependencies: Optional[List[str]] = None,
                              file_hash: Optional[str] = None) -> Dict[str, Any]:
        """Track what Claude knows about a file — purpose, exports, dependencies, last seen hash."""
        ts = self.now()
        map_id = f'file:{project}:{file_path}'
        content = json.dumps({
            'file_path': file_path,
            'purpose': purpose,
            'key_exports': key_exports or [],
            'dependencies': dependencies or [],
            'last_seen_hash': file_hash,
            'last_seen_at': ts,
        })
        self.conn.execute(
            '''INSERT OR REPLACE INTO project_maps (id, project, map_type, content, last_updated, created_at)
               VALUES (?, ?, 'file_inventory', ?, ?, COALESCE(
                   (SELECT created_at FROM project_maps WHERE id = ?), ?))''',
            (map_id, project, content, ts, map_id, ts)
        )
        self.conn.commit()

        # v5: Cross-reference with vocabulary — connect file to vocab nodes that mention it
        try:
            filename = os.path.basename(file_path)
            vocab_nodes = self.conn.execute(
                "SELECT id FROM nodes WHERE type = 'vocabulary' AND archived = 0 AND content LIKE ?",
                (f'%{filename}%',)
            ).fetchall()
            # Also find purpose/mechanism nodes about this file
            eng_nodes = self.conn.execute(
                "SELECT id FROM nodes WHERE type IN ('purpose', 'mechanism') AND archived = 0 AND title LIKE ?",
                (f'%{filename}%',)
            ).fetchall()
            # Create edges between file inventory's related nodes and vocabulary
            for (vid,) in vocab_nodes:
                for (eid,) in eng_nodes:
                    self.connect(vid, eid, 'maps_to', weight=0.7)
        except Exception as _e:
            self._log_error("update_file_inventory", _e, "linking file inventory vocab nodes to engineering nodes")

        return {'id': map_id, 'file_path': file_path, 'updated': True}

    def get_file_inventory(self, project: str) -> List[Dict[str, Any]]:
        """Return full file inventory for a project."""
        cur = self.conn.execute(
            "SELECT content FROM project_maps WHERE project = ? AND map_type = 'file_inventory' ORDER BY last_updated DESC",
            (project,)
        )
        results = []
        for row in cur.fetchall():
            try:
                results.append(json.loads(row[0]))
            except (json.JSONDecodeError, TypeError):
                pass
        return results

    def detect_file_changes(self, project: str) -> List[Dict[str, Any]]:
        """Compare stored file hashes against current state. Returns files that changed since last session.

        Requires git to be available. Falls back to empty list if not.
        """
        import subprocess
        inventory = self.get_file_inventory(project)
        if not inventory:
            return []

        changes = []
        for entry in inventory:
            fp = entry.get('file_path', '')
            stored_hash = entry.get('last_seen_hash')
            if not fp or not stored_hash:
                continue
            try:
                result = subprocess.run(
                    ['git', 'hash-object', fp],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    current_hash = result.stdout.strip()
                    if current_hash != stored_hash:
                        changes.append({
                            'file_path': fp,
                            'purpose': entry.get('purpose', ''),
                            'stored_hash': stored_hash[:8],
                            'current_hash': current_hash[:8],
                        })
            except Exception:
                continue
        return changes

    def update_system_purpose(self, project: str, purpose: str,
                              architecture: Optional[str] = None,
                              key_decisions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Store/update the system-level purpose for a project."""
        ts = self.now()
        map_id = f'purpose:{project}'
        content = json.dumps({
            'purpose': purpose,
            'architecture': architecture,
            'key_decisions': key_decisions or [],
        })
        self.conn.execute(
            '''INSERT OR REPLACE INTO project_maps (id, project, map_type, content, last_updated, created_at)
               VALUES (?, ?, 'system_purpose', ?, ?, COALESCE(
                   (SELECT created_at FROM project_maps WHERE id = ?), ?))''',
            (map_id, project, content, ts, map_id, ts)
        )
        self.conn.commit()
        return {'id': map_id, 'project': project}

    def get_engineering_context(self, project: Optional[str] = None) -> Dict[str, Any]:
        """Synthesize all engineering memory for boot context. The warm-up killer."""
        result = {}

        # System purpose
        if project:
            cur = self.conn.execute(
                "SELECT content FROM project_maps WHERE project = ? AND map_type = 'system_purpose'",
                (project,)
            )
            row = cur.fetchone()
            if row:
                try:
                    result['system_purpose'] = json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    pass

        # Purpose nodes (system scope first, then file, then function)
        purpose_filter = "AND project = ?" if project else ""
        purpose_params = (project,) if project else ()
        cur = self.conn.execute(
            f'''SELECT id, title, content, scope FROM nodes
                WHERE type = 'purpose' AND archived = 0 {purpose_filter}
                ORDER BY CASE scope
                    WHEN 'system' THEN 1 WHEN 'module' THEN 2
                    WHEN 'file' THEN 3 WHEN 'function' THEN 4
                    ELSE 5 END, access_count DESC
                LIMIT 20''',
            purpose_params
        )
        result['purposes'] = [{'id': r[0], 'title': r[1], 'content': r[2], 'scope': r[3]}
                              for r in cur.fetchall()]

        # Impact links (safety-critical)
        cur = self.conn.execute(
            f'''SELECT id, title, content FROM nodes
                WHERE type = 'impact' AND archived = 0 {purpose_filter}
                ORDER BY access_count DESC LIMIT 10''',
            purpose_params
        )
        result['impacts'] = [{'id': r[0], 'title': r[1], 'content': r[2]} for r in cur.fetchall()]

        # Constraints
        cur = self.conn.execute(
            f'''SELECT id, title, content FROM nodes
                WHERE type = 'constraint' AND archived = 0 {purpose_filter}
                ORDER BY access_count DESC LIMIT 10''',
            purpose_params
        )
        result['constraints'] = [{'id': r[0], 'title': r[1], 'content': r[2]} for r in cur.fetchall()]

        # Conventions
        cur = self.conn.execute(
            f'''SELECT id, title, content FROM nodes
                WHERE type = 'convention' AND archived = 0 {purpose_filter}
                ORDER BY access_count DESC LIMIT 5''',
            purpose_params
        )
        result['conventions'] = [{'id': r[0], 'title': r[1], 'content': r[2]} for r in cur.fetchall()]

        # Vocabulary
        cur = self.conn.execute(
            "SELECT id, title, content FROM nodes WHERE type = 'vocabulary' AND archived = 0 ORDER BY access_count DESC LIMIT 15"
        )
        result['vocabulary'] = [{'id': r[0], 'title': r[1], 'content': r[2]} for r in cur.fetchall()]

        # File inventory
        if project:
            result['file_inventory'] = self.get_file_inventory(project)

        # File changes since last session
        if project:
            try:
                result['file_changes'] = self.detect_file_changes(project)
            except Exception:
                result['file_changes'] = []

        return result

    def get_change_impact(self, file_path: str) -> List[Dict[str, Any]]:
        """Return all change impact entries for a file — 'If you modify this, also check...'"""
        results = []
        # Search impact nodes
        cur = self.conn.execute(
            "SELECT id, title, content FROM nodes WHERE type = 'impact' AND archived = 0 AND content LIKE ?",
            (f'%{file_path}%',)
        )
        for row in cur.fetchall():
            results.append({'id': row[0], 'title': row[1], 'content': row[2]})

        # Also check node_metadata change_impacts
        cur = self.conn.execute(
            "SELECT nm.node_id, n.title, nm.change_impacts FROM node_metadata nm JOIN nodes n ON n.id = nm.node_id WHERE nm.change_impacts LIKE ?",
            (f'%{file_path}%',)
        )
        for row in cur.fetchall():
            try:
                impacts = json.loads(row[2])
                for imp in impacts:
                    if file_path in imp.get('if_modified', '') or file_path in imp.get('must_check', ''):
                        results.append({'id': row[0], 'title': row[1], 'impact': imp})
            except (json.JSONDecodeError, TypeError):
                pass
        return results

    def record_divergence(self, claude_assumed: str, reality: str,
                          underlying_pattern: Optional[str] = None,
                          severity: str = 'minor',
                          original_node_id: Optional[str] = None,
                          entity: str = 'host',
                          project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Record where any entity's model diverged from reality.

        This is the highest-value data in the brain — it's where learning happens.
        Framed as 'divergence points' not 'mistakes'.

        The three-entity model: host (Claude training), brain (shared memory),
        operator (Tom). Any of them can have patterns worth tracking.

        Args:
            claude_assumed: What was believed/expected (by any entity)
            reality: What was actually true
            underlying_pattern: The deeper pattern (e.g., "over-generalizing from single examples")
            severity: minor | significant | fundamental
            original_node_id: The node that contained the wrong assumption (if any)
            entity: Which entity's pattern this is — 'host' | 'operator' | 'shared'
                    host = Claude training instinct
                    operator = Tom's habit or assumption
                    shared = collaborative pattern (both contributed)
        """
        content = "Entity: %s\nAssumed: %s\nReality: %s" % (entity, claude_assumed, reality)
        if underlying_pattern:
            content += f"\nPattern: {underlying_pattern}"

        # Create correction node via remember_rich
        result = self.remember_rich(
            type='correction',
            title=f"Divergence: {claude_assumed[:60]}",
            content=content,
            reasoning=f"Claude assumed: {claude_assumed}. Actual: {reality}.",
            correction_pattern=underlying_pattern,
            source_attribution='correction',
            confidence=1.0,
            project=project,
            **kwargs
        )
        node_id = result.get('id')

        # Store in correction_traces table
        session_id = self._get_session_activity().get('session_id', '')
        corrected_node_id = node_id
        self.conn.execute(
            '''INSERT INTO correction_traces
               (session_id, original_node_id, corrected_node_id, claude_assumed, reality,
                underlying_pattern, severity, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (session_id, original_node_id, corrected_node_id,
             claude_assumed, reality, underlying_pattern, severity, self.now())
        )

        # If original node exists, create corrected_by edge and lower its confidence
        if original_node_id:
            self.connect(original_node_id, node_id, 'corrected_by')
            # Lower confidence on the corrected node (NULL treated as 0.7)
            cur_conf = self.conn.execute("SELECT confidence FROM nodes WHERE id=?", (original_node_id,)).fetchone()
            old_c = cur_conf[0] if cur_conf and cur_conf[0] is not None else 0.7
            new_c = max(0.1, old_c - 0.2)
            self.conn.execute("UPDATE nodes SET confidence = ? WHERE id = ?", (new_c, original_node_id))

        # If pattern matches an existing pattern evolution node, strengthen the edge
        if underlying_pattern:
            pattern_nodes = self.conn.execute(
                "SELECT id FROM nodes WHERE type = 'pattern' AND archived = 0 AND title LIKE ?",
                (f'%{underlying_pattern[:30]}%',)
            ).fetchall()
            for pn in pattern_nodes:
                self.connect(node_id, pn[0], 'exemplifies')

            # If severity >= significant and no existing pattern, create one
            if severity in ('significant', 'fundamental') and not pattern_nodes:
                pat_result = self.remember(
                    type='pattern',
                    title=f"Pattern: {underlying_pattern[:80]}",
                    content=f"Divergence pattern detected (severity: {severity}): {underlying_pattern}",
                    project=project,
                )
                # Set evolution_status directly (remember() doesn't accept it)
                if pat_result.get('id'):
                    self.conn.execute(
                        "UPDATE nodes SET evolution_status = 'active' WHERE id = ?",
                        (pat_result['id'],)
                    )

        # v5: Cross-reference with impact maps — if correction relates to a predicted impact,
        # strengthen the impact node (it was right) and create a causal link
        try:
            # Extract file/function names from the correction text
            import re as _re
            mentioned = set(_re.findall(r'[\w_-]+\.(?:py|sh|js|ts|json|sql)', f"{claude_assumed} {reality}"))
            mentioned.update(_re.findall(r'[\w_]+\(\)', f"{claude_assumed} {reality}"))
            for name in mentioned:
                clean = name.rstrip('()')
                impacts = self.conn.execute(
                    """SELECT id, title FROM nodes WHERE type = 'impact' AND archived = 0
                       AND content LIKE ?""",
                    (f'%{clean}%',)
                ).fetchall()
                for impact_id, impact_title in impacts:
                    # Link correction to the impact that predicted it
                    self.connect(node_id, impact_id, 'validates_impact')
                    # Boost impact node confidence (it was prophetic)
                    self.conn.execute(
                        """UPDATE nodes SET confidence = MIN(1.0, COALESCE(confidence, 0.7) + 0.1)
                           WHERE id = ?""", (impact_id,)
                    )
        except Exception as _e:
            self._log_error("record_divergence", _e, "")

        # Track in session state
        self._session_state['corrections'].append({
            'node_id': node_id,
            'assumed': claude_assumed[:100],
            'reality': reality[:100],
            'pattern': underlying_pattern,
            'severity': severity,
        })

        return result

    def record_validation(self, node_id: str, context: Optional[str] = None,
                          project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Record positive signal — this approach/knowledge was confirmed correct.

        Creates a validation node, updates last_validated on the target,
        and boosts its confidence.
        """
        # Get the node being validated
        target = self.conn.execute(
            "SELECT id, title, type FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if not target:
            return {'error': f'Node {node_id} not found'}

        target_title = target[1]
        target_type = target[2]

        # Create validation node
        val_content = f"Validated: {target_title}"
        if context:
            val_content += f"\nContext: {context}"

        result = self.remember_rich(
            type='validation',
            title=f"Confirmed: {target_title[:60]}",
            content=val_content,
            source_attribution='correction',
            project=project,
            **kwargs
        )

        # Update last_validated on the target node's metadata
        self.validate_node(node_id, context)

        # Create validation edge
        if result.get('id'):
            self.connect(result['id'], node_id, 'validates')

        # Track in session state
        self._session_state['validations'].append({
            'node_id': node_id,
            'title': target_title,
            'context': context,
        })

        return result

    def get_correction_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recurring divergence patterns by frequency.

        Returns patterns that appear 2+ times — these shape Claude's behavior.
        """
        cur = self.conn.execute(
            '''SELECT underlying_pattern, COUNT(*) as cnt,
                      GROUP_CONCAT(claude_assumed, ' | ') as examples,
                      MAX(severity) as max_severity,
                      MAX(created_at) as latest
               FROM correction_traces
               WHERE underlying_pattern IS NOT NULL
               GROUP BY underlying_pattern
               HAVING cnt >= 2
               ORDER BY cnt DESC, latest DESC
               LIMIT ?''',
            (limit,)
        )
        return [{
            'pattern': r[0],
            'count': r[1],
            'examples': r[2][:200] if r[2] else '',
            'max_severity': r[3],
            'latest': r[4],
        } for r in cur.fetchall()]

    def track_session_event(self, event_type: str, data: Dict[str, Any]):
        """Accumulate session events for synthesis.

        Called by hooks during the session to build running state.

        event_type: decision | correction | inflection | model_update | validation | open_question
        """
        valid_types = {
            'decision': 'decisions',
            'correction': 'corrections',
            'inflection': 'inflections',
            'model_update': 'model_updates',
            'validation': 'validations',
            'open_question': 'open_questions',
        }
        key = valid_types.get(event_type)
        if key and key in self._session_state:
            data['timestamp'] = self.now()
            self._session_state[key].append(data)

    def assess_session_health(self, boot_time: Optional[str] = None) -> Dict[str, Any]:
        """Generalized "notice that we didn't" — session health assessment.

        At session boundaries, compare what happened vs what should have happened
        across multiple dimensions. Each dimension asks: did we do this thing that
        a healthy collaboration typically does?

        This is NOT a checklist to satisfy. It's the brain noticing gaps in its own
        behavior — self-awareness, not compliance. The dimensions that matter emerge
        from operator engagement over time.

        Returns:
            {
                'dimensions_checked': int,
                'gaps': [{'dimension': str, 'signal': str, 'severity': str}],
                'healthy': [str],  # dimensions that look good
                'overall': 'healthy' | 'some_gaps' | 'concerning',
                'top_prompt': str | None  # single most important gap to surface
            }
        """
        if not boot_time:
            activity = self._get_session_activity()
            boot_time = activity.get('boot_time') or self.get_config('session_start_at')

        if not boot_time:
            return {'dimensions_checked': 0, 'gaps': [], 'healthy': [],
                    'overall': 'unknown', 'top_prompt': None}

        gaps = []
        healthy = []

        # Count session nodes for thresholds
        try:
            type_rows = self.conn.execute(
                '''SELECT type, COUNT(*) FROM nodes
                   WHERE created_at >= ? AND archived = 0
                   GROUP BY type''',
                (boot_time,)
            ).fetchall()
        except Exception:
            type_rows = []

        total_nodes = sum(r[1] for r in type_rows) if type_rows else 0
        type_set = set(r[0] for r in type_rows) if type_rows else set()
        type_counts = {r[0]: r[1] for r in type_rows} if type_rows else {}

        # Skip health check for very short sessions
        if total_nodes < 3:
            return {'dimensions_checked': 0, 'gaps': [], 'healthy': [],
                    'overall': 'too_short', 'top_prompt': None}

        # ── DIMENSION 1: Encoding Diversity ──────────────────────────
        # (Existing encoding_bias check, generalized)
        ENCODING_DIMS = {
            'technical': {'rule', 'lesson', 'mechanism', 'constraint',
                          'convention', 'impact', 'purpose', 'bug_lesson'},
            'relational': {'mental_model', 'pattern', 'person', 'validation'},
            'reasoning': {'decision', 'hypothesis', 'reasoning_trace',
                          'uncertainty'},
            'self_awareness': {'correction', 'tension', 'aspiration'},
        }
        present_enc = set()
        for dim_name, dim_types in ENCODING_DIMS.items():
            if type_set & dim_types:
                present_enc.add(dim_name)
        missing_enc = sorted(set(ENCODING_DIMS.keys()) - present_enc)
        if len(missing_enc) >= 2:
            gaps.append({
                'dimension': 'encoding_diversity',
                'signal': '%d nodes but only %s. Missing: %s' % (
                    total_nodes, '/'.join(sorted(present_enc)),
                    '/'.join(missing_enc)),
                'severity': 'moderate',
            })
        else:
            healthy.append('encoding_diversity')

        # ── DIMENSION 2: Self-Correction Honesty ─────────────────────
        # Did we admit any mistakes? Sessions with many nodes but zero
        # corrections suggest either a perfect session (rare) or avoidance.
        correction_count = type_counts.get('correction', 0)
        try:
            trace_count = self.conn.execute(
                '''SELECT COUNT(*) FROM correction_traces
                   WHERE created_at >= ?''',
                (boot_time,)
            ).fetchone()[0]
        except Exception:
            trace_count = 0

        if total_nodes >= 8 and correction_count == 0 and trace_count == 0:
            gaps.append({
                'dimension': 'correction_honesty',
                'signal': '%d nodes encoded but zero corrections. '
                          'Was everything really right the first time?' % total_nodes,
                'severity': 'mild',
            })
        elif correction_count > 0 or trace_count > 0:
            healthy.append('correction_honesty')

        # ── DIMENSION 3: Validation ──────────────────────────────────
        # Did we confirm or update any existing knowledge? Every session
        # should validate or invalidate something the brain already knows.
        validation_count = type_counts.get('validation', 0)
        try:
            validated_count = self.conn.execute(
                '''SELECT COUNT(*) FROM node_metadata
                   WHERE last_validated >= ?''',
                (boot_time,)
            ).fetchone()[0]
        except Exception:
            validated_count = 0

        if total_nodes >= 5 and validation_count == 0 and validated_count == 0:
            gaps.append({
                'dimension': 'validation',
                'signal': 'No existing knowledge validated or invalidated. '
                          'Did old assumptions go unchecked?',
                'severity': 'mild',
            })
        elif validation_count > 0 or validated_count > 0:
            healthy.append('validation')

        # ── DIMENSION 4: Operator Voice Preservation ─────────────────
        # Did we store the operator's actual words, not just our interpretation?
        # Check for user_raw_quote in node_metadata.
        try:
            quote_count = self.conn.execute(
                '''SELECT COUNT(*) FROM node_metadata nm
                   JOIN nodes n ON nm.node_id = n.id
                   WHERE n.created_at >= ? AND nm.user_raw_quote IS NOT NULL''',
                (boot_time,)
            ).fetchone()[0]
        except Exception:
            quote_count = 0

        if total_nodes >= 5 and quote_count == 0:
            gaps.append({
                'dimension': 'operator_voice',
                'signal': 'No operator quotes preserved. '
                          'All %d nodes are Claude interpretations only.' % total_nodes,
                'severity': 'moderate',
            })
        elif quote_count > 0:
            healthy.append('operator_voice')

        # ── DIMENSION 5: Reasoning Preservation ──────────────────────
        # Did we record WHY decisions were made, not just WHAT was decided?
        try:
            reasoning_count = self.conn.execute(
                '''SELECT COUNT(*) FROM node_metadata nm
                   JOIN nodes n ON nm.node_id = n.id
                   WHERE n.created_at >= ? AND nm.reasoning IS NOT NULL''',
                (boot_time,)
            ).fetchone()[0]
        except Exception:
            reasoning_count = 0

        decision_count = type_counts.get('decision', 0)
        if decision_count >= 2 and reasoning_count == 0:
            gaps.append({
                'dimension': 'reasoning_preservation',
                'signal': '%d decisions recorded but none with reasoning. '
                          'Future sessions will know WHAT but not WHY.' % decision_count,
                'severity': 'moderate',
            })
        elif reasoning_count > 0:
            healthy.append('reasoning_preservation')

        # ── DIMENSION 6: Connection / Integration ────────────────────
        # Did new knowledge get connected to existing knowledge?
        # Check for edges created this session.
        try:
            edge_count = self.conn.execute(
                '''SELECT COUNT(*) FROM edges
                   WHERE created_at >= ?''',
                (boot_time,)
            ).fetchone()[0]
        except Exception:
            edge_count = 0

        if total_nodes >= 5 and edge_count < 2:
            gaps.append({
                'dimension': 'integration',
                'signal': '%d nodes but only %d connections made. '
                          'New knowledge is isolated from existing knowledge.' % (
                              total_nodes, edge_count),
                'severity': 'mild',
            })
        elif edge_count >= 2:
            healthy.append('integration')

        # ── DIMENSION 7: Follow-Through ──────────────────────────────
        # Did we encode what we said we would? Check for tasks created
        # this session that are still open (not a gap per se, but awareness).
        # More importantly: check if aspirations/hypotheses were acted on.
        try:
            active_evolutions = self.conn.execute(
                '''SELECT COUNT(*) FROM nodes
                   WHERE type IN ('tension', 'hypothesis', 'aspiration')
                     AND evolution_status = 'active' AND archived = 0
                     AND created_at < ?''',
                (boot_time,)
            ).fetchone()[0]
        except Exception:
            active_evolutions = 0

        try:
            evolution_actions = self.conn.execute(
                '''SELECT COUNT(*) FROM nodes
                   WHERE type IN ('tension', 'hypothesis', 'aspiration')
                     AND evolution_status != 'active' AND archived = 0
                     AND last_accessed >= ?''',
                (boot_time,)
            ).fetchone()[0]
        except Exception:
            evolution_actions = 0

        if active_evolutions >= 3 and evolution_actions == 0:
            gaps.append({
                'dimension': 'follow_through',
                'signal': '%d active evolutions (tensions/hypotheses) but none '
                          'were confirmed, dismissed, or addressed this session.' % active_evolutions,
                'severity': 'mild',
            })
        elif evolution_actions > 0:
            healthy.append('follow_through')

        # ── DIMENSION 8: Depth Over Breadth ──────────────────────────
        # Many shallow nodes vs fewer deep ones. Check average content length
        # and metadata presence as proxy for depth.
        try:
            depth_stats = self.conn.execute(
                '''SELECT AVG(LENGTH(content)), COUNT(*) FROM nodes
                   WHERE created_at >= ? AND archived = 0''',
                (boot_time,)
            ).fetchone()
            avg_len = depth_stats[0] or 0

            meta_count = self.conn.execute(
                '''SELECT COUNT(*) FROM node_metadata nm
                   JOIN nodes n ON nm.node_id = n.id
                   WHERE n.created_at >= ?''',
                (boot_time,)
            ).fetchone()[0]
        except Exception:
            avg_len = 0
            meta_count = 0

        meta_ratio = meta_count / max(total_nodes, 1)
        if total_nodes >= 5 and avg_len < 80 and meta_ratio < 0.1:
            gaps.append({
                'dimension': 'depth',
                'signal': 'Average node is %d chars with %.0f%% having metadata. '
                          'Encoding is shallow — titles without substance.' % (
                              int(avg_len), meta_ratio * 100),
                'severity': 'moderate',
            })
        elif avg_len >= 120 or meta_ratio >= 0.2:
            healthy.append('depth')

        # ── Compute overall health ───────────────────────────────────
        dimensions_checked = len(gaps) + len(healthy)
        if not gaps:
            overall = 'healthy'
        elif len(gaps) <= 2:
            overall = 'some_gaps'
        else:
            overall = 'concerning'

        # Pick the single most important gap to surface
        severity_order = {'moderate': 0, 'mild': 1}
        sorted_gaps = sorted(gaps, key=lambda g: severity_order.get(g['severity'], 2))
        top_prompt = sorted_gaps[0]['signal'] if sorted_gaps else None

        return {
            'dimensions_checked': dimensions_checked,
            'gaps': gaps,
            'healthy': healthy,
            'overall': overall,
            'top_prompt': top_prompt,
        }

    def recalibrate_confidence(self, boot_time: Optional[str] = None) -> Dict[str, Any]:
        """Recalibrate confidence at session boundaries — the brain's "sleep."

        Three dynamics:
        1. EMOTIONAL COOLING: Nodes encoded with high emotion get confidence
           pulled toward type default. Excitement of discovery != quality of insight.
        2. TEMPORAL-EXTERNAL DECAY: Nodes about external systems (tools, APIs,
           capabilities) lose confidence over time because the external system
           evolves independently of us.
        3. SILENT VALIDATION: Nodes recalled multiple times without correction
           get a slight confidence boost — evidence of utility.

        Returns dict with counts of adjustments made.
        """
        adjusted = {'emotional_cooling': 0, 'temporal_external': 0, 'silent_validation': 0}

        # ── 1. EMOTIONAL COOLING ──────────────────────────────────────
        # Nodes encoded with high emotion get a confidence discount.
        # Excitement inflates certainty — everything feels more true when
        # you just discovered it. After "sleeping on it" (session boundary),
        # confidence settles to a more calibrated level.
        #
        # All thresholds are tunable — the brain learns its own parameters.
        emotion_threshold = self._get_tunable('conf_emotion_threshold', 0.7)
        emotion_discount_min = self._get_tunable('conf_emotion_discount_min', 0.05)
        emotion_discount_max = self._get_tunable('conf_emotion_discount_max', 0.15)

        if boot_time:
            # Only cool nodes created THIS session (created_at >= boot_time)
            # This prevents re-cooling nodes from previous sessions
            hot_nodes = self.conn.execute(
                '''SELECT id, type, confidence, emotion FROM nodes
                   WHERE created_at >= ? AND archived = 0
                     AND emotion >= ? AND confidence IS NOT NULL
                     AND locked = 0''',
                (boot_time, emotion_threshold)
            ).fetchall()
            for node_id, node_type, conf, emotion in hot_nodes:
                type_default = TYPE_CONFIDENCE.get(node_type, 0.70)
                # Discount proportional to emotion above threshold
                range_pct = (emotion - emotion_threshold) / max(0.01, 1.0 - emotion_threshold)
                discount = emotion_discount_min + range_pct * (emotion_discount_max - emotion_discount_min)
                new_conf = conf * (1.0 - discount)
                # Floor: never below type_default * 0.7
                new_conf = max(type_default * 0.7, new_conf)
                if conf - new_conf > 0.005:
                    self.conn.execute(
                        'UPDATE nodes SET confidence = ? WHERE id = ?',
                        (round(new_conf, 3), node_id)
                    )
                    adjusted['emotional_cooling'] += 1

        # ── 2. TEMPORAL-EXTERNAL DECAY ────────────────────────────────
        # Nodes about external systems lose confidence over time.
        # Detect by keywords in title/content/keywords fields.
        # All parameters tunable.
        external_halflife_days = self._get_tunable('conf_external_halflife_days', 30)
        external_min_keywords = self._get_tunable('conf_external_min_keywords', 2)
        external_min_age_days = self._get_tunable('conf_external_min_age_days', 7)
        # Daily decay factor from half-life: factor^halflife = 0.5
        daily_decay = 0.5 ** (1.0 / max(1, external_halflife_days))

        try:
            candidates = self.conn.execute(
                '''SELECT id, type, confidence, created_at,
                          COALESCE(title, '') || ' ' || COALESCE(content, '') || ' ' || COALESCE(keywords, '') as text
                   FROM nodes
                   WHERE archived = 0 AND locked = 0
                     AND confidence IS NOT NULL AND confidence > 0.2
                     AND created_at < datetime('now', '-%d days')''' % external_min_age_days
            ).fetchall()
            for node_id, node_type, conf, created_at, text in candidates:
                text_lower = text.lower()
                # Count how many external keywords match
                matches = sum(1 for kw in EXTERNAL_CLAIM_KEYWORDS if kw in text_lower)
                if matches < external_min_keywords:
                    continue

                # Calculate age in days
                try:
                    created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    age_days = (datetime.now(created_dt.tzinfo) - created_dt).total_seconds() / 86400
                except Exception:
                    continue

                type_default = TYPE_CONFIDENCE.get(node_type, 0.70)
                decay_factor = daily_decay ** age_days
                floor = max(0.2, type_default * 0.3)
                new_conf = max(floor, type_default * decay_factor)
                if new_conf < conf - 0.01:
                    self.conn.execute(
                        'UPDATE nodes SET confidence = ? WHERE id = ?',
                        (round(new_conf, 3), node_id)
                    )
                    adjusted['temporal_external'] += 1
        except Exception as _e:
            self._log_error("recalibrate_confidence", _e, "temporal_external")

        # ── 3. SILENT VALIDATION ──────────────────────────────────────
        # Nodes recalled often without being corrected: slight boost.
        # "Recalled without correction" = high access_count, no corrected_by edge.
        silent_min_access = self._get_tunable('conf_silent_min_access', 5)
        silent_boost = self._get_tunable('conf_silent_boost', 0.03)
        silent_ceiling_above_default = self._get_tunable('conf_silent_ceiling_above_default', 0.15)

        try:
            well_used = self.conn.execute(
                '''SELECT n.id, n.type, n.confidence FROM nodes n
                   WHERE n.archived = 0 AND n.access_count >= ?
                     AND n.confidence IS NOT NULL AND n.confidence < 0.95
                     AND NOT EXISTS (
                         SELECT 1 FROM edges e
                         WHERE e.target_id = n.id AND e.relation = 'corrected_by'
                     )
                     AND NOT EXISTS (
                         SELECT 1 FROM correction_traces ct
                         WHERE ct.original_node_id = n.id
                     )''',
                (silent_min_access,)
            ).fetchall()
            for node_id, node_type, conf in well_used:
                type_default = TYPE_CONFIDENCE.get(node_type, 0.70)
                ceiling = min(0.95, type_default + silent_ceiling_above_default)
                new_conf = min(ceiling, conf + silent_boost)
                if new_conf > conf + 0.005:
                    self.conn.execute(
                        'UPDATE nodes SET confidence = ? WHERE id = ?',
                        (round(new_conf, 3), node_id)
                    )
                    adjusted['silent_validation'] += 1
        except Exception as _e:
            self._log_error("recalibrate_confidence", _e, "silent_validation")

        if sum(adjusted.values()) > 0:
            self.conn.commit()

        return adjusted

    def synthesize_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Create structured synthesis from session activity.

        One good synthesis > 50 shallow nodes.
        Called by pre-compact-save or explicitly at session end.

        v6: Self-sufficient — queries DB for what happened since boot_time,
        merged with any in-memory _session_state. No longer depends on
        track_session_event() calls that nothing was making.
        """
        if not session_id:
            session_id = self._get_session_activity().get('session_id', uuid.uuid4().hex)

        state = self._session_state
        synthesis_id = uuid.uuid4().hex

        # Calculate session duration
        activity = self._get_session_activity()
        boot_time = activity.get('boot_time')
        duration_minutes = None
        if boot_time:
            try:
                boot_dt = datetime.fromisoformat(boot_time.replace('Z', '+00:00'))
                now_dt = datetime.now(boot_dt.tzinfo) if boot_dt.tzinfo else datetime.utcnow()
                duration_minutes = int((now_dt - boot_dt).total_seconds() / 60)
            except Exception as _e:
                self._log_error("synthesize_session", _e, "duration calc")

        # ── v6: Harvest from DB (self-sufficient synthesis) ──────────────
        # The in-memory _session_state is often empty because nothing calls
        # track_session_event(). Query the DB for what actually happened.
        if boot_time:
            # Decisions created this session
            if not state['decisions']:
                rows = self.conn.execute(
                    '''SELECT id, title, content FROM nodes
                       WHERE type = 'decision' AND created_at >= ? AND archived = 0
                       ORDER BY created_at''',
                    (boot_time,)
                ).fetchall()
                for r in rows:
                    state['decisions'].append({
                        'title': r[1], 'content': (r[2] or '')[:200],
                        'node_id': r[0],
                    })

            # Corrections this session
            if not state['corrections']:
                rows = self.conn.execute(
                    '''SELECT ct.claude_assumed, ct.reality, ct.underlying_pattern, ct.severity
                       FROM correction_traces ct
                       WHERE ct.created_at >= ?
                       ORDER BY ct.created_at''',
                    (boot_time,)
                ).fetchall()
                for r in rows:
                    state['corrections'].append({
                        'assumed': r[0], 'reality': r[1],
                        'pattern': r[2] or 'unknown', 'severity': r[3],
                    })

            # Mental model / mechanism / purpose nodes = model updates
            if not state['model_updates']:
                rows = self.conn.execute(
                    '''SELECT type, title, content FROM nodes
                       WHERE type IN ('mental_model', 'mechanism', 'purpose')
                         AND created_at >= ? AND archived = 0
                       ORDER BY created_at''',
                    (boot_time,)
                ).fetchall()
                for r in rows:
                    state['model_updates'].append({
                        'area': r[1], 'before': None,
                        'after': (r[2] or '')[:200],
                    })

            # Lessons / constraints = inflection points
            if not state['inflections']:
                rows = self.conn.execute(
                    '''SELECT title, content FROM nodes
                       WHERE type IN ('lesson', 'constraint', 'rule')
                         AND created_at >= ? AND archived = 0
                       ORDER BY created_at''',
                    (boot_time,)
                ).fetchall()
                for r in rows:
                    state['inflections'].append({
                        'description': r[0],
                        'triggered_by': (r[1] or '')[:100],
                    })

            # Uncertainty / hypothesis nodes = open questions
            if not state['open_questions']:
                rows = self.conn.execute(
                    '''SELECT title FROM nodes
                       WHERE type IN ('uncertainty', 'hypothesis')
                         AND created_at >= ? AND archived = 0
                       ORDER BY created_at''',
                    (boot_time,)
                ).fetchall()
                for r in rows:
                    state['open_questions'].append(r[0])

        # ── Build structured synthesis ──────────────────────────────────
        decisions_made = json.dumps(state['decisions']) if state['decisions'] else None
        corrections_received = json.dumps(state['corrections']) if state['corrections'] else None
        validations_list = state['validations']

        # Identify inflection points — moments where direction changed
        inflection_points = json.dumps(state['inflections']) if state['inflections'] else None

        # Identify mental model updates
        mental_model_updates = json.dumps(state['model_updates']) if state['model_updates'] else None

        # Identify teaching arcs — sequences of corrections that build toward understanding
        teaching_arcs = None
        if len(state['corrections']) >= 2:
            # Group corrections by pattern
            patterns = {}
            for c in state['corrections']:
                p = c.get('pattern', 'unknown')
                if p not in patterns:
                    patterns[p] = []
                patterns[p].append(c)
            arcs = []
            for pattern, corrections in patterns.items():
                if len(corrections) >= 2:
                    arcs.append({
                        'pattern': pattern,
                        'sequence': [c.get('assumed', '')[:50] for c in corrections],
                        'lesson': "Recurring divergence on: %s" % pattern,
                    })
            if arcs:
                teaching_arcs = json.dumps(arcs)

        open_questions = json.dumps(state['open_questions']) if state['open_questions'] else None

        # Include encoding health in synthesis metadata
        try:
            edits = activity.get('edit_check_count', 0)
            remembers = activity.get('remember_count', 0)
            if edits > 0 or remembers > 0:
                health_status = 'GOOD' if remembers >= edits * 0.1 else ('SPARSE' if remembers > 0 else 'NONE')
                health_entry = {
                    'area': 'encoding_health', 'before': None,
                    'after': '%d edits, %d remembers (%s)' % (edits, remembers, health_status),
                }
                model_updates_list = state['model_updates'][:]
                model_updates_list.append(health_entry)
                mental_model_updates = json.dumps(model_updates_list)
        except Exception as _e:
            self._log_error("synthesize_session", _e, "encoding health calc")

        # Include reflection prompts in open questions if they add value
        try:
            reflections = self.prompt_reflection()
            if reflections:
                existing_q = state['open_questions'][:]
                for r in reflections[:2]:
                    existing_q.append(r[:200])
                open_questions = json.dumps(existing_q) if existing_q else open_questions
        except Exception as _e:
            self._log_error("synthesize_session", _e, "reflections")

        # Also include a session summary of all nodes created (for boot context)
        session_node_summary = None
        if boot_time:
            try:
                type_counts = self.conn.execute(
                    '''SELECT type, COUNT(*) FROM nodes
                       WHERE created_at >= ? AND archived = 0
                       GROUP BY type ORDER BY COUNT(*) DESC''',
                    (boot_time,)
                ).fetchall()
                if type_counts:
                    parts = ['%s: %d' % (r[0], r[1]) for r in type_counts]
                    total = sum(r[1] for r in type_counts)
                    session_node_summary = '%d nodes encoded (%s)' % (total, ', '.join(parts))
            except Exception:
                pass

        # v5.2: Generalized session health assessment — "notice that we didn't"
        # Replaces the single-dimension encoding_blind_spots check with a
        # multi-dimensional health assessment. Each dimension asks:
        # did we do this thing that healthy collaboration typically does?
        session_health = None
        if boot_time:
            try:
                session_health = self.assess_session_health(boot_time=boot_time)
                if session_health and session_health.get('gaps'):
                    existing_q = json.loads(open_questions) if open_questions else state['open_questions'][:]
                    # Insert top gap as the lead question
                    top = session_health.get('top_prompt')
                    if top:
                        existing_q.insert(0, 'SESSION HEALTH: ' + top)
                    # Add other moderate-severity gaps
                    for gap in session_health['gaps']:
                        if gap['severity'] == 'moderate' and gap['signal'] != top:
                            existing_q.append('SESSION GAP (%s): %s' % (
                                gap['dimension'], gap['signal']))
                    open_questions = json.dumps(existing_q)
            except Exception as _e:
                self._log_error("synthesize_session", _e, "session health assessment")

        # Skip empty syntheses
        has_content = any([decisions_made, corrections_received, inflection_points,
                          mental_model_updates, teaching_arcs, open_questions,
                          session_node_summary])
        if not has_content:
            return {'id': None, 'reason': 'no session events to synthesize'}

        # Store synthesis
        self.conn.execute(
            '''INSERT OR REPLACE INTO session_syntheses
               (id, session_id, duration_minutes, decisions_made, corrections_received,
                inflection_points, mental_model_updates, teaching_arcs, open_questions, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (synthesis_id, session_id, duration_minutes, decisions_made,
             corrections_received, inflection_points, mental_model_updates,
             teaching_arcs, open_questions, self.now())
        )

        # For significant decisions, create rich nodes
        for dec in state['decisions']:
            if dec.get('reasoning'):
                self.remember_rich(
                    type='decision',
                    title=dec.get('title', 'Session decision')[:80],
                    content=dec.get('content', ''),
                    reasoning=dec.get('reasoning'),
                    source_attribution='session_synthesis',
                    project=dec.get('project'),
                )

        # v6: Confidence recalibration — the brain's "sleep"
        # Run at session boundary to cool emotional inflation,
        # decay external claims, and boost silently validated nodes.
        recal = {'emotional_cooling': 0, 'temporal_external': 0, 'silent_validation': 0}
        try:
            recal = self.recalibrate_confidence(boot_time=boot_time)
        except Exception as _e:
            self._log_error("synthesize_session", _e, "recalibrate_confidence")

        return {
            'id': synthesis_id,
            'session_id': session_id,
            'duration_minutes': duration_minutes,
            'decisions': len(state['decisions']),
            'corrections': len(state['corrections']),
            'inflections': len(state['inflections']),
            'model_updates': len(state['model_updates']),
            'validations': len(validations_list),
            'open_questions': len(state['open_questions']),
            'teaching_arcs': len(json.loads(teaching_arcs)) if teaching_arcs else 0,
            'node_summary': session_node_summary,
            'confidence_recalibrated': recal,
        }

    def get_last_synthesis(self) -> Optional[Dict[str, Any]]:
        """Get the most recent session synthesis for boot context."""
        row = self.conn.execute(
            '''SELECT id, session_id, duration_minutes, decisions_made,
                      corrections_received, inflection_points, mental_model_updates,
                      teaching_arcs, open_questions, created_at
               FROM session_syntheses
               ORDER BY created_at DESC LIMIT 1'''
        ).fetchone()
        if not row:
            return None

        result = {
            'id': row[0], 'session_id': row[1], 'duration_minutes': row[2],
            'created_at': row[9],
        }
        # Parse JSON fields
        for i, key in enumerate(['decisions_made', 'corrections_received', 'inflection_points',
                                  'mental_model_updates', 'teaching_arcs', 'open_questions'], 3):
            val = row[i]
            if val:
                try:
                    result[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    result[key] = val
            else:
                result[key] = []
        return result

    def create_fn_reasoning(self, fn_name: str, content: str, file: Optional[str] = None,
                            calls: Optional[List[str]] = None, breaks_if: Optional[str] = None,
                            **kwargs) -> Dict[str, Any]:
        """Create a function reasoning node — WHY a function exists, its intent, dependencies, risk."""
        title = f'[fn] {fn_name}'
        if file:
            title += f' ({file})'
        full_content = content
        if calls:
            full_content += f'\nCalls: {", ".join(calls)}'
        if breaks_if:
            full_content += f'\nBreaks if: {breaks_if}'
        return self.remember(type='fn_reasoning', title=title, content=full_content,
                             keywords=kwargs.get('keywords', fn_name), locked=True,
                             emotion=0.4, emotion_label='emphasis', **{k: v for k, v in kwargs.items() if k != 'keywords'})

    def create_param_influence(self, param_name: str, current_value: str, content: str,
                               affects: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Create a parameter influence node — systemic effects of a configurable value."""
        title = f'[param] {param_name}={current_value}'
        full_content = content
        if affects:
            full_content += f'\nAffects: {", ".join(affects)}'
        return self.remember(type='param_influence', title=title, content=full_content,
                             keywords=kwargs.get('keywords', param_name), locked=True,
                             emotion=0.4, emotion_label='emphasis', **{k: v for k, v in kwargs.items() if k != 'keywords'})

    def create_code_concept(self, name: str, content: str, files: Optional[List[str]] = None,
                            blast_radius: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Create a code concept node — semantic unit spanning files/functions with blast radius."""
        title = f'[concept] {name}'
        full_content = content
        if files:
            full_content += f'\nFiles: {", ".join(files)}'
        if blast_radius:
            full_content += f'\nBlast radius: {blast_radius}'
        return self.remember(type='code_concept', title=title, content=full_content,
                             keywords=kwargs.get('keywords', name), locked=True,
                             emotion=0.4, emotion_label='emphasis', **{k: v for k, v in kwargs.items() if k != 'keywords'})

    def create_arch_constraint(self, title_str: str, content: str,
                               implications: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Create an architecture constraint — what limits what and why. Challengeable when host changes."""
        title = f'[constraint] {title_str}'
        full_content = content
        if implications:
            full_content += f'\nImplications: {implications}'
        return self.remember(type='arch_constraint', title=title, content=full_content,
                             keywords=kwargs.get('keywords', title_str), locked=True,
                             emotion=0.4, emotion_label='emphasis', **{k: v for k, v in kwargs.items() if k != 'keywords'})

    def create_causal_chain(self, title_str: str, trigger: str, propagation: str,
                            failure: str, root_cause: str, prevention: Optional[str] = None,
                            **kwargs) -> Dict[str, Any]:
        """Create a causal chain — regression path: trigger → propagation → failure → root cause."""
        title = f'[chain] {title_str}'
        content = f'Trigger: {trigger}\nPropagation: {propagation}\nFailure: {failure}\nRoot cause: {root_cause}'
        if prevention:
            content += f'\nPrevention: {prevention}'
        return self.remember(type='causal_chain', title=title, content=content,
                             keywords=kwargs.get('keywords', title_str), locked=False,
                             emotion=0.5, emotion_label='concern', **{k: v for k, v in kwargs.items() if k != 'keywords'})

    def create_bug_lesson(self, title_str: str, bug: str, fix: str, lesson: str,
                          **kwargs) -> Dict[str, Any]:
        """Create a bug lesson — general principle extracted from a specific bug."""
        title = f'[bug] {title_str}'
        content = f'BUG: {bug}\nFIX: {fix}\nLESSON: {lesson}'
        return self.remember(type='bug_lesson', title=title, content=content,
                             keywords=kwargs.get('keywords', title_str), locked=True,
                             emotion=0.5, emotion_label='emphasis', **{k: v for k, v in kwargs.items() if k != 'keywords'})

    def create_comment_anchor(self, file: str, description: str, why_it_matters: str,
                              **kwargs) -> Dict[str, Any]:
        """Create a comment anchor — load-bearing comment in code that transfers knowledge."""
        title = f'[comment] {file}: {description[:50]}'
        content = f'File: {file}\nComment: {description}\nWhy it matters: {why_it_matters}'
        return self.remember(type='comment_anchor', title=title, content=content,
                             keywords=kwargs.get('keywords', f'{file} comment'), locked=False,
                             emotion=0.3, emotion_label='neutral', **{k: v for k, v in kwargs.items() if k != 'keywords'})

    def create_failure_mode(self, title: str, content: str, instances: int = 1,
                            project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Name a recurring failure CLASS, not an individual miss.
        "Solution fixation" is a failure class. "Recall miss #47" is a single event.
        Failure modes aggregate: instances, triggers, what breaks the pattern, prevention.
        """
        result = self.remember(
            type='failure_mode', title=f'🚫 FAILURE MODE — {title}', content=content,
            keywords=kwargs.get('keywords', ''),
            locked=True,  # Failure modes are permanent prevention
            emotion=0.7, emotion_label='concern',
            project=project, confidence=kwargs.get('confidence', 0.8),
        )
        node_id = result['id']
        self.conn.execute("UPDATE nodes SET evolution_status = 'active' WHERE id = ?", (node_id,))
        self.conn.commit()
        result['evolution_status'] = 'active'
        return result

    def create_performance(self, title: str, content: str,
                           project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Track a brain quality metric over time. Not a snapshot — a persistent trending node.
        "Recall precision on Glo queries: 0.81, up from 0.65 last month."
        """
        result = self.remember(
            type='performance', title=f'📈 PERFORMANCE — {title}', content=content,
            keywords=kwargs.get('keywords', ''),
            emotion=0.3, emotion_label='neutral',
            project=project, confidence=kwargs.get('confidence', 0.7),
        )
        return result

    def create_capability(self, title: str, content: str,
                          project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Record what the brain can or cannot do. Self-inventory.
        "Can absorb other brains." "Cannot trigger time-based reminders natively."
        """
        result = self.remember(
            type='capability', title=f'🔧 CAPABILITY — {title}', content=content,
            keywords=kwargs.get('keywords', ''),
            emotion=0.2, emotion_label='neutral',
            project=project,
        )
        return result

    def create_interaction(self, title: str, content: str,
                           project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Observe dynamics of the human-Claude working relationship.
        Not rules stated by the human — observed cause and effect.
        "Tom responds well to themed grouped questions."
        "Tom disengages when Claude suggests pausing at creative moments."
        """
        result = self.remember(
            type='interaction', title=f'🤝 INTERACTION — {title}', content=content,
            keywords=kwargs.get('keywords', ''),
            emotion=0.4, emotion_label='curiosity',
            project=project,
        )
        return result

    def create_meta_learning(self, title: str, content: str,
                             project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Record HOW the brain learned something — reusable methods.
        "Hebbian bug found via relearning simulation — method: replay transcripts,
         compare output, identify missing edges."
        """
        result = self.remember(
            type='meta_learning', title=f'🔄 META-LEARNING — {title}', content=content,
            keywords=kwargs.get('keywords', ''),
            locked=True,  # Learning methods are reusable forever
            emotion=0.5, emotion_label='curiosity',
            project=project,
        )
        return result

    def set_reminder(self, node_id: str, due_date: str) -> Dict[str, Any]:
        """
        Set a due_date on any node. Scanned at context_boot — surfaces before anything else.
        due_date: ISO timestamp (e.g. "2026-03-25T09:00:00")
        """
        ts = self.now()
        self.conn.execute(
            'UPDATE nodes SET due_date = ?, updated_at = ? WHERE id = ?',
            (due_date, ts, node_id)
        )
        self.conn.commit()
        return {'node_id': node_id, 'due_date': due_date}

    def create_reminder(self, title: str, due_date: str, content: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Create a reminder node with a due_date. Surfaces at boot when due.
        Example: brain.create_reminder("Call mom", "2026-03-25T09:00:00")
        """
        result = self.remember(
            type='task', title=f'🔔 REMINDER — {title}',
            content=content or title,
            keywords=kwargs.get('keywords', f'reminder {title.lower()}'),
            emotion=0.5, emotion_label='urgency',
        )
        self.set_reminder(result['id'], due_date)
        result['due_date'] = due_date
        return result

    def get_due_reminders(self) -> List[Dict[str, Any]]:
        """
        Get all nodes with due_date <= now. Called at boot to surface reminders.
        """
        now = self.now()
        cursor = self.conn.execute(
            """SELECT id, type, title, content, due_date, created_at
               FROM nodes
               WHERE due_date IS NOT NULL AND due_date <= ? AND archived = 0
               ORDER BY due_date ASC""",
            (now,)
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0], 'type': row[1], 'title': row[2],
                'content': row[3], 'due_date': row[4], 'created_at': row[5],
            })
        return results

    def auto_generate_self_reflection(self) -> Dict[str, Any]:
        """
        Analyze brain data and auto-generate self-reflection nodes.
        Called at session end or periodically. Creates performance, capability, and
        interaction observations from accumulated data.
        """
        generated = {'performance': 0, 'capability': 0, 'interaction': 0, 'failure': 0}

        # Performance: recall quality from recall_log
        # NOTE: This signal is BROKEN until mark_recall_used() is implemented.
        # used_count is always 0 because nothing writes back to recall_log.
        # Skipping generation to avoid false "0% precision" alarms.
        # TODO: Re-enable once mark_recall_used() exists. See tests/relearning.py.
        try:
            recall_stats = self.logs_conn.execute(
                """SELECT COUNT(*) as total,
                          SUM(CASE WHEN used_count > 0 THEN 1 ELSE 0 END) as useful
                   FROM recall_log
                   WHERE created_at > datetime('now', '-7 days')"""
            ).fetchone()
            if recall_stats and recall_stats[0] >= 10:
                total, useful = recall_stats
                precision = useful / total if total > 0 else 0
                # Only generate if mark_recall_used() is actually populating data
                if useful > 0:
                    existing = self.conn.execute(
                        "SELECT COUNT(*) FROM nodes WHERE type = 'performance' AND created_at > datetime('now', '-3 days')"
                    ).fetchone()[0]
                    if existing == 0:
                        self.create_performance(
                            f"Recall precision this week: {precision:.0%} ({useful}/{total} useful)",
                            f"Auto-generated from recall_log. {total} recalls in 7 days, {useful} had results marked as used.",
                            keywords="auto performance recall precision weekly"
                        )
                        generated['performance'] += 1
        except Exception as _e:
            self._log_error("auto_generate_self_reflection", _e, "generating recall precision performance node")

        # Failure detection: repeated miss signals
        try:
            repeated = self.logs_conn.execute(
                """SELECT signal, COUNT(*) as cnt FROM miss_log
                   WHERE created_at > datetime('now', '-7 days')
                   GROUP BY signal HAVING cnt >= 3
                   ORDER BY cnt DESC LIMIT 2"""
            ).fetchall()
            for signal, count in repeated:
                existing = self.conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE type = 'failure_mode' AND keywords LIKE ? AND archived = 0",
                    (f'%auto {signal}%',)
                ).fetchone()[0]
                if existing == 0:
                    self.create_failure_mode(
                        f"Recurring miss signal: {signal} ({count}x this week)",
                        f"Auto-detected: {count} '{signal}' events in 7 days. This is a recurring failure pattern.",
                        keywords=f"auto failure-mode {signal} recurring"
                    )
                    generated['failure'] += 1
        except Exception as _e:
            self._log_error("auto_generate_self_reflection", _e, "detecting repeated miss signals from miss_log")

        # Capability: check embedder status
        try:
            emb_ready = embedder.is_ready()
            existing = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE type = 'capability' AND keywords LIKE '%embedder status%' AND created_at > datetime('now', '-7 days')"
            ).fetchone()[0]
            if existing == 0:
                status = "active" if emb_ready else "unavailable"
                model = embedder.stats.get('model_name', 'unknown')
                self.create_capability(
                    f"Embedder {status}: {model}",
                    f"Auto-generated. Embedder is {'ready' if emb_ready else 'NOT ready — recall is keyword-only (degraded)'}. Model: {model}.",
                    keywords="auto capability embedder status"
                )
                generated['capability'] += 1
        except Exception as _e:
            self._log_error("auto_generate_self_reflection", _e, "generating embedder capability status node")

        # Interaction: analyze consciousness response patterns
        try:
            tension_yes = int(self.get_config('consciousness_response_tension_yes', 0) or 0)
            tension_no = int(self.get_config('consciousness_response_tension_no', 0) or 0)
            dream_yes = int(self.get_config('consciousness_response_dream_yes', 0) or 0)
            dream_no = int(self.get_config('consciousness_response_dream_no', 0) or 0)
            total = tension_yes + tension_no + dream_yes + dream_no

            if total >= 5:
                existing = self.conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE type = 'interaction' AND keywords LIKE '%auto consciousness-response%' AND created_at > datetime('now', '-7 days')"
                ).fetchone()[0]
                if existing == 0:
                    # Determine which signals get most engagement
                    observations = []
                    if tension_yes > tension_no:
                        observations.append('Tensions get engaged with (responded %d/%d)' % (tension_yes, tension_yes + tension_no))
                    if dream_no > dream_yes and dream_no >= 2:
                        observations.append('Dream insights get ignored (ignored %d/%d)' % (dream_no, dream_yes + dream_no))
                    if observations:
                        self.create_interaction(
                            'Consciousness engagement pattern: ' + '; '.join(observations),
                            'Auto-detected from consciousness response tracking. ' + '. '.join(observations) + '.',
                            keywords='auto consciousness-response interaction engagement pattern'
                        )
                        generated['interaction'] = generated.get('interaction', 0) + 1
        except Exception as _e:
            self._log_error("auto_generate_self_reflection", _e, "analyzing consciousness response engagement patterns")

        # Meta-learning: track which encoding methods produce good recall
        try:
            # Check if nodes created with embeddings have better recall than those without
            emb_recalled = self.logs_conn.execute(
                """SELECT COUNT(DISTINCT rl.id) FROM recall_log rl
                   WHERE rl.used_count > 0 AND rl.created_at > datetime('now', '-7 days')"""
            ).fetchone()[0]

            if emb_recalled >= 10:
                existing = self.conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE type = 'meta_learning' AND keywords LIKE '%auto recall-method%' AND created_at > datetime('now', '-14 days')"
                ).fetchone()[0]
                if existing == 0:
                    # Count how many useful recalls came from embedding vs keyword path
                    self.create_meta_learning(
                        'Recall method: embeddings-first produces %d useful recalls/week' % emb_recalled,
                        'Auto-measured from recall_log. %d recalls in 7 days had results marked as used.' % emb_recalled,
                        keywords='auto recall-method meta-learning embeddings weekly'
                    )
                    generated['meta_learning'] = generated.get('meta_learning', 0) + 1
        except Exception as _e:
            self._log_error("auto_generate_self_reflection", _e, "generating meta-learning node for recall method effectiveness")

        return generated
