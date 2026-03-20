"""
brain — BrainEvolution Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import re
from .brain_constants import (
    CONTEXT_BOOT_LOCKED_LIMIT,
    CONTEXT_BOOT_RECALL_LIMIT,
    CONTEXT_BOOT_RECENT_LIMIT,
    CURIOSITY_CHAIN_GAP_THRESHOLD,
    CURIOSITY_DECAY_WARNING_HOURS,
    CURIOSITY_MAX_PROMPTS,
    DECAY_HALF_LIFE,
    DREAM_COUNT,
    DREAM_MIN_NOVELTY,
    DREAM_WALK_LENGTH,
    EMBEDDING_PRIMARY_WEIGHT,
    EMOTION_WEIGHT,
    FREQUENCY_WEIGHT,
    KEYWORD_FALLBACK_WEIGHT,
    REASONING_STEP_TYPES,
    RECENCY_WEIGHT,
    RELEVANCE_WEIGHT,
    STABILITY_BOOST,
)


class BrainEvolutionMixin:
    """Evolution methods for Brain."""

    def _safe_serialize_half_lives(self, half_lives: dict) -> dict:
        """Serialize decay half-lives for JSON storage, handling float('inf') and NaN.

        float('inf') → 999999999 (sentinel: effectively infinite in hours = 114,000 years)
        float('nan') → 168 (default: 7 days)
        """
        import math
        result = {}
        for k, v in half_lives.items():
            if isinstance(v, float):
                if math.isinf(v):
                    result[k] = 999999999
                elif math.isnan(v):
                    result[k] = 168  # safe default
                else:
                    result[k] = v
            elif isinstance(v, (int, str)):
                # Handle legacy string "inf" from previous bug
                if isinstance(v, str) and v.lower() in ('inf', 'infinity'):
                    result[k] = 999999999
                else:
                    result[k] = v
            else:
                result[k] = v
        return result

    def auto_heal(self) -> Dict[str, Any]:
        """
        Self-healing: resolve discoveries, tune parameters, clean graph.
        Runs after auto_discover_evolutions() during idle.

        Three categories:
        1. Resolution healing — act on discovered evolutions (merge duplicates, auto-lock, etc.)
        2. Adaptive tuning — adjust brain parameters based on observed behavior
        3. Graph hygiene — clean orphan nodes, normalize edge weights, archive stale evolutions
        """
        ts = self.now()
        results = {
            'resolved': [],
            'tuned': [],
            'cleaned': {'archived': 0, 'edges_created': 0, 'edges_normalized': 0, 'merged': 0, 'locked': 0},
        }

        # ══════════════════════════════════════════════════════
        # CATEGORY 1: Resolution Healing
        # ══════════════════════════════════════════════════════

        # 1a. Merge near-duplicates (sim > 0.85)
        if embedder.is_ready():
            try:
                sim_thresholds = self._get_tunable('similarity_thresholds', {'merge': 0.85})
                merge_threshold = sim_thresholds.get('merge', 0.85) if isinstance(sim_thresholds, dict) else 0.85

                # Find locked node pairs with very high similarity
                cursor = self.conn.execute(
                    """SELECT n.id, n.type, n.title, n.created_at, ne.embedding
                       FROM nodes n
                       JOIN node_embeddings ne ON n.id = ne.node_id
                       WHERE n.locked = 1 AND n.archived = 0
                         AND n.type IN ('rule', 'decision', 'arch_constraint',
                                        'constraint', 'lesson', 'impact', 'purpose',
                                        'vocabulary', 'convention', 'mechanism')
                       ORDER BY RANDOM() LIMIT 60"""
                )
                candidates = cursor.fetchall()

                merged_this_cycle = 0
                for i in range(len(candidates)):
                    if merged_this_cycle >= 3:
                        break
                    for j in range(i + 1, len(candidates)):
                        if merged_this_cycle >= 3:
                            break
                        sim = embedder.cosine_similarity(candidates[i][4], candidates[j][4])
                        if sim > merge_threshold:
                            id_a, type_a, title_a, created_a, _ = candidates[i]
                            id_b, type_b, title_b, created_b, _ = candidates[j]

                            # Keep newer (or longer content), archive older
                            keep_id, archive_id = (id_b, id_a) if created_b > created_a else (id_a, id_b)
                            keep_title = title_b if keep_id == id_b else title_a
                            archive_title = title_a if archive_id == id_a else title_b

                            # Archive the duplicate
                            self.conn.execute(
                                "UPDATE nodes SET archived = 1, updated_at = ? WHERE id = ?",
                                (ts, archive_id)
                            )
                            # Transfer edges from archived → surviving node
                            self.conn.execute(
                                """UPDATE OR IGNORE edges SET source_id = ?
                                   WHERE source_id = ? AND target_id != ?""",
                                (keep_id, archive_id, keep_id)
                            )
                            self.conn.execute(
                                """UPDATE OR IGNORE edges SET target_id = ?
                                   WHERE target_id = ? AND source_id != ?""",
                                (keep_id, archive_id, keep_id)
                            )
                            # Clean up any self-referential edges created by transfer
                            self.conn.execute(
                                "DELETE FROM edges WHERE source_id = target_id"
                            )
                            # Merge node_metadata: preserve best data from both
                            try:
                                keep_meta = self.conn.execute(
                                    "SELECT * FROM node_metadata WHERE node_id = ?", (keep_id,)
                                ).fetchone()
                                archive_meta = self.conn.execute(
                                    "SELECT * FROM node_metadata WHERE node_id = ?", (archive_id,)
                                ).fetchone()
                                if archive_meta and not keep_meta:
                                    # Surviving node has no metadata — adopt archived node's
                                    self.conn.execute(
                                        "UPDATE node_metadata SET node_id = ? WHERE node_id = ?",
                                        (keep_id, archive_id)
                                    )
                                elif archive_meta and keep_meta:
                                    # Both have metadata — merge: take higher validation count,
                                    # most recent last_validated, combine reasoning
                                    cols = [d[1] for d in self.conn.execute(
                                        "PRAGMA table_info(node_metadata)"
                                    ).fetchall()]
                                    arch_dict = dict(zip(cols, archive_meta))
                                    keep_dict = dict(zip(cols, keep_meta))
                                    # Merge validation counts
                                    arch_vc = arch_dict.get('validation_count') or 0
                                    keep_vc = keep_dict.get('validation_count') or 0
                                    merged_vc = arch_vc + keep_vc
                                    # Take most recent last_validated
                                    arch_lv = arch_dict.get('last_validated') or ''
                                    keep_lv = keep_dict.get('last_validated') or ''
                                    best_lv = max(arch_lv, keep_lv) or None
                                    # Combine reasoning if surviving is empty
                                    merged_reasoning = keep_dict.get('reasoning') or arch_dict.get('reasoning')
                                    merged_quote = keep_dict.get('user_raw_quote') or arch_dict.get('user_raw_quote')
                                    merged_impacts = keep_dict.get('change_impacts') or arch_dict.get('change_impacts')
                                    self.conn.execute(
                                        """UPDATE node_metadata SET
                                           validation_count = ?, last_validated = ?,
                                           reasoning = COALESCE(reasoning, ?),
                                           user_raw_quote = COALESCE(user_raw_quote, ?),
                                           change_impacts = COALESCE(change_impacts, ?)
                                           WHERE node_id = ?""",
                                        (merged_vc, best_lv, merged_reasoning,
                                         merged_quote, merged_impacts, keep_id)
                                    )
                            except Exception:
                                pass  # metadata merge is best-effort
                            # Create audit trail edge
                            self.conn.execute(
                                """INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, weight, relation, description, created_at)
                                   VALUES (?, ?, 'contradicts', 0.9, 'merged_duplicate',
                                           'Auto-healed: merged near-duplicate (sim > ' || ? || ')', ?)""",
                                (keep_id, archive_id, str(round(sim, 2)), ts)
                            )
                            results['resolved'].append({
                                'action': 'merge_duplicate',
                                'kept': keep_title[:60],
                                'archived': archive_title[:60],
                                'sim': round(sim, 2)
                            })
                            results['cleaned']['merged'] += 1
                            merged_this_cycle += 1
            except Exception as _e:
                self._log_error("auto_heal", _e, "")

        # 1c. Auto-lock orphan beliefs (access_count >= 10)
        EXCLUDE_ORPHAN_TYPES = ('task', 'context', 'file', 'intuition', 'person', 'project', 'decision')
        try:
            placeholders = ','.join('?' * len(EXCLUDE_ORPHAN_TYPES))
            high_access_unlocked = self.conn.execute(
                f"""SELECT id, title, type, access_count FROM nodes
                    WHERE locked = 0 AND archived = 0
                      AND access_count >= 10
                      AND type NOT IN ({placeholders})
                    ORDER BY access_count DESC LIMIT 5""",
                EXCLUDE_ORPHAN_TYPES
            ).fetchall()

            for node_id, title, node_type, access_count in high_access_unlocked:
                self.conn.execute(
                    "UPDATE nodes SET locked = 1, updated_at = ? WHERE id = ?",
                    (ts, node_id)
                )
                results['resolved'].append({
                    'action': 'auto_lock',
                    'title': title[:60],
                    'type': node_type,
                    'access_count': access_count
                })
                results['cleaned']['locked'] += 1
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # 1d. Create missing edges from co-access (count >= 5)
        try:
            co_pairs = self.conn.execute(
                """SELECT e.source_id, e.target_id, e.co_access_count
                   FROM edges e
                   WHERE e.co_access_count >= 5
                     AND e.edge_type NOT IN ('related', 'part_of', 'contradicts', 'corrected_by',
                                             'exemplifies', 'produced', 'reasoning_step', 'depends_on')
                   ORDER BY e.co_access_count DESC LIMIT 10"""
            ).fetchall()

            for src_id, tgt_id, co_count in co_pairs:
                # Check no explicit semantic edge exists
                has_explicit = self.conn.execute(
                    """SELECT COUNT(*) FROM edges
                       WHERE ((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))
                         AND edge_type IN ('related', 'part_of', 'exemplifies', 'depends_on')""",
                    (src_id, tgt_id, tgt_id, src_id)
                ).fetchone()[0]
                if has_explicit > 0:
                    continue

                weight = 0.6 if co_count >= 10 else 0.4
                self.conn.execute(
                    """INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, weight, relation, description, created_at)
                       VALUES (?, ?, 'related', ?, 'co_access_promoted',
                               'Auto-healed: promoted from co-access (' || ? || 'x)', ?)""",
                    (src_id, tgt_id, weight, str(co_count), ts)
                )
                results['cleaned']['edges_created'] += 1
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # 1f. Resolve confirmed/dismissed evolutions
        try:
            # Auto-archive evolutions dismissed 2+ times (from consciousness tracking)
            dismissed = self.conn.execute(
                """SELECT bm.key, bm.value FROM brain_meta bm
                   WHERE bm.key LIKE 'consciousness_response_%_no'
                     AND CAST(bm.value AS INTEGER) >= 2"""
            ).fetchall()
            for key, count in dismissed:
                # Extract signal type from key
                signal = key.replace('consciousness_response_', '').replace('_no', '')
                if signal == 'evolutions':
                    # Archive all active evolutions that have been repeatedly dismissed
                    self.conn.execute(
                        """UPDATE nodes SET evolution_status = 'dismissed', archived = 1, updated_at = ?
                           WHERE type IN ('tension', 'hypothesis', 'pattern', 'aspiration')
                             AND evolution_status = 'active' AND archived = 0
                             AND keywords LIKE '%auto-discovered%'""",
                        (ts,)
                    )
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # 1g. Consolidate correction clusters — 3+ corrections with same pattern → locked rule
        try:
            patterns = self.conn.execute(
                '''SELECT underlying_pattern, COUNT(*) as cnt
                   FROM correction_traces
                   WHERE underlying_pattern IS NOT NULL
                   GROUP BY underlying_pattern HAVING cnt >= 3
                   ORDER BY cnt DESC LIMIT 3'''
            ).fetchall()
            for pattern, count in patterns:
                # Check if a rule already exists for this pattern
                existing = self.conn.execute(
                    "SELECT id FROM nodes WHERE type IN ('rule', 'lesson') AND archived = 0 AND title LIKE ?",
                    (f'%{pattern[:30]}%',)
                ).fetchone()
                if existing:
                    continue
                # Get examples
                examples = self.conn.execute(
                    "SELECT claude_assumed, reality FROM correction_traces WHERE underlying_pattern = ? ORDER BY created_at DESC LIMIT 3",
                    (pattern,)
                ).fetchall()
                example_text = "; ".join([f"assumed '{r[0][:40]}' but was '{r[1][:40]}'" for r in examples])
                # Create locked lesson node
                self.remember_lesson(
                    title=f"Recurring divergence: {pattern[:60]}",
                    what_happened=f"Pattern appeared {count} times in correction traces",
                    root_cause=pattern,
                    fix=f"Be aware of this tendency and verify assumptions",
                    preventive_principle=f"Examples: {example_text[:200]}",
                )
                results['resolved'].append({
                    'action': 'correction_consolidated',
                    'pattern': pattern[:60],
                    'count': count,
                })
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # 1h. Backfill missing embeddings
        try:
            if embedder.is_ready():
                missing = self.conn.execute(
                    """SELECT n.id, n.title, n.content FROM nodes n
                       LEFT JOIN node_embeddings ne ON n.id = ne.node_id
                       WHERE ne.node_id IS NULL AND n.archived = 0
                       LIMIT 20"""
                ).fetchall()
                for nid, ntitle, ncontent in missing:
                    try:
                        text = f"{ntitle} {ncontent or ''}"
                        vec = embedder.embed(text)
                        if vec is not None:
                            self.conn.execute(
                                "INSERT OR REPLACE INTO node_embeddings (node_id, embedding, model_name, created_at) VALUES (?, ?, ?, ?)",
                                (nid, vec, embedder.get_stats().get('model_name', ''), ts)
                            )
                    except Exception as _e:
                        self._log_error("auto_heal", _e, "backfilling embedding for single node")
        except Exception as _e:
            self._log_error("auto_heal", _e, "backfilling missing embeddings for nodes without embeddings")

        # 1i. Consciousness signal → automated confidence adjustments
        # Stale reasoning and recurring divergence should affect scoring, not just display.
        try:
            # Stale reasoning: nodes with reasoning metadata but never validated (>21 days old)
            stale_nodes = self.conn.execute(
                """SELECT nm.node_id, n.confidence FROM node_metadata nm
                   JOIN nodes n ON n.id = nm.node_id
                   WHERE nm.reasoning IS NOT NULL
                     AND (nm.last_validated IS NULL OR nm.last_validated < datetime('now', '-21 days'))
                     AND n.archived = 0 AND n.confidence IS NOT NULL AND n.confidence > 0.5
                     AND nm.created_at < datetime('now', '-21 days')
                   LIMIT 10"""
            ).fetchall()
            for nid, conf in stale_nodes:
                # Mild confidence decay: -0.05 per cycle, floor at 0.5
                new_conf = max(0.5, (conf or 0.7) - 0.05)
                if new_conf < (conf or 0.7):
                    self.conn.execute(
                        "UPDATE nodes SET confidence = ? WHERE id = ?", (new_conf, nid)
                    )

            # Recurring divergence patterns: lower confidence on nodes that keep getting corrected
            recurring = self.conn.execute(
                """SELECT original_node_id, COUNT(*) as cnt FROM correction_traces
                   WHERE original_node_id IS NOT NULL
                   GROUP BY original_node_id HAVING cnt >= 3"""
            ).fetchall()
            for nid, cnt in recurring:
                cur = self.conn.execute(
                    "SELECT confidence FROM nodes WHERE id = ? AND archived = 0", (nid,)
                ).fetchone()
                if cur:
                    old_conf = cur[0] if cur[0] is not None else 0.7
                    # Floor at 0.3 for repeatedly-corrected nodes
                    new_conf = max(0.3, old_conf - 0.1)
                    if new_conf < old_conf:
                        self.conn.execute(
                            "UPDATE nodes SET confidence = ? WHERE id = ?", (new_conf, nid)
                        )
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # ══════════════════════════════════════════════════════
        # CATEGORY 2: Adaptive Tuning
        # ══════════════════════════════════════════════════════

        # 2a. Decay rates — if a type keeps getting recreated after decay, increase half-life
        try:
            current_half_lives = self._get_tunable('decay_half_lives', DECAY_HALF_LIFE)
            if not isinstance(current_half_lives, dict):
                current_half_lives = dict(DECAY_HALF_LIFE)
            updated_half_lives = dict(current_half_lives)
            changed = False

            # Find types with high re-creation rate (archived > active in last 30 days)
            decay_stats = self.conn.execute(
                """SELECT type,
                          SUM(CASE WHEN archived = 1 AND created_at > datetime('now', '-30 days') THEN 1 ELSE 0 END) as arc,
                          SUM(CASE WHEN archived = 0 AND created_at > datetime('now', '-30 days') THEN 1 ELSE 0 END) as act
                   FROM nodes
                   WHERE created_at > datetime('now', '-30 days')
                   GROUP BY type
                   HAVING arc > act AND arc >= 3"""
            ).fetchall()

            EPHEMERAL_TYPES = ('context', 'intuition', 'thought')  # expected to decay
            for node_type, archived, active in decay_stats:
                if node_type in EPHEMERAL_TYPES:
                    continue
                old_hl = updated_half_lives.get(node_type, 168)
                if old_hl == float('inf'):
                    continue
                new_hl = min(2160, old_hl * 1.5)  # cap at 90 days
                if new_hl != old_hl:
                    updated_half_lives[node_type] = new_hl
                    changed = True
                    results['tuned'].append({
                        'param': f'decay_half_life.{node_type}',
                        'old': old_hl, 'new': new_hl,
                        'reason': f'{archived} archived vs {active} active in 30d — slowing decay'
                    })

            # Find types where nodes are never accessed after creation
            never_accessed = self.conn.execute(
                """SELECT type, COUNT(*) as cnt FROM nodes
                   WHERE access_count <= 1 AND archived = 0
                     AND created_at < datetime('now', '-7 days')
                     AND locked = 0
                   GROUP BY type HAVING cnt >= 5"""
            ).fetchall()
            for node_type, count in never_accessed:
                old_hl = updated_half_lives.get(node_type, 168)
                if old_hl == float('inf') or old_hl <= 1:
                    continue
                new_hl = max(1, old_hl * 0.75)  # floor at 1 hour
                if new_hl != old_hl:
                    updated_half_lives[node_type] = new_hl
                    changed = True
                    results['tuned'].append({
                        'param': f'decay_half_life.{node_type}',
                        'old': old_hl, 'new': new_hl,
                        'reason': f'{count} nodes never re-accessed — speeding decay'
                    })

            if changed:
                serializable = self._safe_serialize_half_lives(updated_half_lives)
                self._set_tunable('decay_half_lives', serializable,
                                  f'Auto-tuned {len(results["tuned"])} decay rates')
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # 2b. Recall weights — track which signal best predicts re-access
        try:
            # Simple heuristic: if old nodes (low recency) are frequently re-accessed,
            # recency weight is too high
            # Two-step: get accessed IDs from logs DB, then filter in main DB
            accessed_ids = self.logs_conn.execute(
                """SELECT DISTINCT node_id FROM access_log
                   WHERE timestamp > datetime('now', '-7 days')"""
            ).fetchall()
            if accessed_ids:
                id_list = ','.join("'%s'" % r[0].replace("'", "''") for r in accessed_ids)
                old_reaccessed = self.conn.execute(
                    """SELECT COUNT(*) FROM nodes
                       WHERE id IN (%s) AND created_at < datetime('now', '-30 days')
                         AND locked = 0""" % id_list
                ).fetchone()[0]
            else:
                old_reaccessed = 0
            total_accessed = self.logs_conn.execute(
                """SELECT COUNT(*) FROM access_log
                   WHERE timestamp > datetime('now', '-7 days')"""
            ).fetchone()[0]

            if total_accessed > 20:
                old_ratio = old_reaccessed / total_accessed
                current_weights = self._get_tunable('recall_weights', {
                    'relevance': RELEVANCE_WEIGHT, 'recency': RECENCY_WEIGHT,
                    'frequency': FREQUENCY_WEIGHT, 'emotion': EMOTION_WEIGHT
                })
                if not isinstance(current_weights, dict):
                    current_weights = {'relevance': RELEVANCE_WEIGHT, 'recency': RECENCY_WEIGHT,
                                       'frequency': FREQUENCY_WEIGHT, 'emotion': EMOTION_WEIGHT}

                # If >40% of re-accessed nodes are old, recency is over-weighted
                if old_ratio > 0.4 and current_weights.get('recency', RECENCY_WEIGHT) > 0.15:
                    new_weights = dict(current_weights)
                    delta = 0.05
                    new_weights['recency'] = round(new_weights.get('recency', RECENCY_WEIGHT) - delta, 2)
                    new_weights['relevance'] = round(new_weights.get('relevance', RELEVANCE_WEIGHT) + delta, 2)
                    self._set_tunable('recall_weights', new_weights,
                                      f'Old nodes re-accessed at {old_ratio:.0%} — shifting weight from recency to relevance')
                    results['tuned'].append({
                        'param': 'recall_weights',
                        'old': current_weights, 'new': new_weights,
                        'reason': f'{old_ratio:.0%} of re-accessed nodes are old'
                    })
                # If <10% are old, recency is under-weighted
                elif old_ratio < 0.1 and current_weights.get('recency', RECENCY_WEIGHT) < 0.45:
                    new_weights = dict(current_weights)
                    delta = 0.05
                    new_weights['recency'] = round(new_weights.get('recency', RECENCY_WEIGHT) + delta, 2)
                    new_weights['relevance'] = round(new_weights.get('relevance', RELEVANCE_WEIGHT) - delta, 2)
                    self._set_tunable('recall_weights', new_weights,
                                      f'Old nodes rarely re-accessed ({old_ratio:.0%}) — boosting recency')
                    results['tuned'].append({
                        'param': 'recall_weights',
                        'old': current_weights, 'new': new_weights,
                        'reason': f'Only {old_ratio:.0%} of re-accessed nodes are old'
                    })
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # 2c. Similarity thresholds — adjust based on evolution dismiss/confirm rates
        try:
            confirmed = self.conn.execute(
                """SELECT COUNT(*) FROM nodes
                   WHERE type IN ('tension', 'hypothesis', 'pattern', 'aspiration')
                     AND evolution_status IN ('confirmed', 'validated', 'resolved')
                     AND keywords LIKE '%auto-discovered%'"""
            ).fetchone()[0]
            dismissed = self.conn.execute(
                """SELECT COUNT(*) FROM nodes
                   WHERE type IN ('tension', 'hypothesis', 'pattern', 'aspiration')
                     AND evolution_status IN ('dismissed', 'disproven')
                     AND keywords LIKE '%auto-discovered%'"""
            ).fetchone()[0]
            total_evolutions = confirmed + dismissed

            if total_evolutions >= 5:
                dismiss_rate = dismissed / total_evolutions
                current_thresholds = self._get_tunable('similarity_thresholds', {
                    'tension': 0.65, 'temporal': 0.70, 'cluster': 0.60,
                    'orphan_backing': 0.70, 'emotion_cluster': 0.65, 'merge': 0.85
                })
                if isinstance(current_thresholds, dict):
                    new_thresholds = dict(current_thresholds)
                    if dismiss_rate > 0.6:
                        # Too many false positives — raise thresholds
                        for k in ('tension', 'temporal', 'cluster'):
                            new_thresholds[k] = min(0.90, round(new_thresholds.get(k, 0.65) + 0.05, 2))
                        self._set_tunable('similarity_thresholds', new_thresholds,
                                          f'High dismiss rate ({dismiss_rate:.0%}) — raising thresholds')
                        results['tuned'].append({
                            'param': 'similarity_thresholds',
                            'old': current_thresholds, 'new': new_thresholds,
                            'reason': f'{dismiss_rate:.0%} dismiss rate'
                        })
                    elif dismiss_rate < 0.2 and total_evolutions >= 10:
                        # Very few false positives — can lower thresholds
                        for k in ('tension', 'temporal', 'cluster'):
                            new_thresholds[k] = max(0.50, round(new_thresholds.get(k, 0.65) - 0.03, 2))
                        self._set_tunable('similarity_thresholds', new_thresholds,
                                          f'Low dismiss rate ({dismiss_rate:.0%}) — lowering thresholds')
                        results['tuned'].append({
                            'param': 'similarity_thresholds',
                            'old': current_thresholds, 'new': new_thresholds,
                            'reason': f'{dismiss_rate:.0%} dismiss rate — room to discover more'
                        })
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # 2d. Stability boost — check for stability inflation or excessive decay
        try:
            avg_stability = self.conn.execute(
                "SELECT AVG(stability) FROM nodes WHERE locked = 0 AND archived = 0"
            ).fetchone()[0] or 1.0
            current_boost = self._get_tunable('stability_boost', STABILITY_BOOST)
            if not isinstance(current_boost, (int, float)):
                current_boost = STABILITY_BOOST

            if avg_stability > 5.0 and current_boost > 1.1:
                # Stability inflation — reduce boost
                new_boost = round(max(1.1, current_boost - 0.1), 2)
                self._set_tunable('stability_boost', new_boost,
                                  f'Avg stability {avg_stability:.1f} > 5.0 — reducing boost')
                results['tuned'].append({
                    'param': 'stability_boost', 'old': current_boost, 'new': new_boost,
                    'reason': f'Stability inflation (avg {avg_stability:.1f})'
                })
            elif avg_stability < 0.5 and current_boost < 3.0:
                # Nodes decaying too fast — increase boost
                new_boost = round(min(3.0, current_boost + 0.2), 2)
                self._set_tunable('stability_boost', new_boost,
                                  f'Avg stability {avg_stability:.1f} < 0.5 — increasing boost')
                results['tuned'].append({
                    'param': 'stability_boost', 'old': current_boost, 'new': new_boost,
                    'reason': f'Excessive decay (avg stability {avg_stability:.1f})'
                })
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # 2e. Embedding blend — shift toward keywords when embedder is degraded
        try:
            if not embedder.is_ready():
                current_blend = self._get_tunable('embedding_blend', {
                    'embedding': EMBEDDING_PRIMARY_WEIGHT, 'keyword': KEYWORD_FALLBACK_WEIGHT
                })
                if isinstance(current_blend, dict) and current_blend.get('embedding', 0.9) > 0.5:
                    new_blend = {'embedding': 0.0, 'keyword': 1.0}
                    self._set_tunable('embedding_blend', new_blend,
                                      'Embedder offline — shifting to keyword-only')
                    results['tuned'].append({
                        'param': 'embedding_blend', 'old': current_blend, 'new': new_blend,
                        'reason': 'Embedder not ready'
                    })
            else:
                # Embedder is healthy — restore if previously degraded
                current_blend = self._get_tunable('embedding_blend', None)
                if isinstance(current_blend, dict) and current_blend.get('embedding', 0.9) < 0.5:
                    new_blend = {'embedding': EMBEDDING_PRIMARY_WEIGHT, 'keyword': KEYWORD_FALLBACK_WEIGHT}
                    self._set_tunable('embedding_blend', new_blend,
                                      'Embedder restored — reverting to embedding-primary')
                    results['tuned'].append({
                        'param': 'embedding_blend', 'old': current_blend, 'new': new_blend,
                        'reason': 'Embedder recovered'
                    })
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # 2f. Hub dampening — adjust based on hub recall patterns
        try:
            # If high-degree nodes dominate recall results, lower the threshold
            # Two-step: get hub IDs from main DB, then count in logs DB
            hub_ids = self.conn.execute(
                """SELECT source_id FROM edges GROUP BY source_id HAVING COUNT(*) > 30"""
            ).fetchall()
            hub_in_recent = 0
            if hub_ids:
                hub_id_list = ','.join("'%s'" % r[0].replace("'", "''") for r in hub_ids)
                hub_in_recent = self.logs_conn.execute(
                    """SELECT COUNT(*) FROM access_log
                       WHERE node_id IN (%s) AND timestamp > datetime('now', '-7 days')""" % hub_id_list
                ).fetchone()[0]
            total_recent = self.logs_conn.execute(
                "SELECT COUNT(*) FROM access_log WHERE timestamp > datetime('now', '-7 days')"
            ).fetchone()[0]

            if total_recent > 20:
                hub_ratio = hub_in_recent / total_recent
                current_hub = self._get_tunable('hub_dampening', {'threshold': 40, 'penalty': 0.5})
                if isinstance(current_hub, dict):
                    if hub_ratio > 0.5:
                        # Hubs dominating — lower threshold (dampen more aggressively)
                        new_hub = dict(current_hub)
                        new_hub['threshold'] = max(10, current_hub.get('threshold', 40) - 5)
                        self._set_tunable('hub_dampening', new_hub,
                                          f'Hub nodes at {hub_ratio:.0%} of recalls — dampening more')
                        results['tuned'].append({
                            'param': 'hub_dampening', 'old': current_hub, 'new': new_hub,
                            'reason': f'Hubs dominate at {hub_ratio:.0%}'
                        })
                    elif hub_ratio < 0.1:
                        # Hubs under-recalled — raise threshold (dampen less)
                        new_hub = dict(current_hub)
                        new_hub['threshold'] = min(100, current_hub.get('threshold', 40) + 5)
                        self._set_tunable('hub_dampening', new_hub,
                                          f'Hub nodes at {hub_ratio:.0%} of recalls — dampening less')
                        results['tuned'].append({
                            'param': 'hub_dampening', 'old': current_hub, 'new': new_hub,
                            'reason': f'Hubs under-recalled at {hub_ratio:.0%}'
                        })
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # 2g. Dream params — adjust based on dream utility
        try:
            dreams_promoted = self.conn.execute(
                """SELECT COUNT(*) FROM nodes
                   WHERE type = 'hypothesis' AND keywords LIKE '%auto-discovered%'
                     AND title LIKE 'Dream promoted%'"""
            ).fetchone()[0]
            total_dreams = self.logs_conn.execute(
                "SELECT COUNT(*) FROM dream_log WHERE created_at > datetime('now', '-30 days')"
            ).fetchone()[0]

            current_dream = self._get_tunable('dream_params', {
                'count': DREAM_COUNT, 'walk_length': DREAM_WALK_LENGTH, 'min_novelty': DREAM_MIN_NOVELTY
            })
            if isinstance(current_dream, dict) and total_dreams > 10:
                promotion_rate = dreams_promoted / max(1, total_dreams)
                if promotion_rate > 0.1:
                    # Dreams are useful — more dreams, longer walks
                    new_dream = dict(current_dream)
                    new_dream['count'] = min(10, current_dream.get('count', DREAM_COUNT) + 1)
                    new_dream['walk_length'] = min(10, current_dream.get('walk_length', DREAM_WALK_LENGTH) + 1)
                    self._set_tunable('dream_params', new_dream,
                                      f'Dream promotion rate {promotion_rate:.0%} — more dreaming')
                    results['tuned'].append({
                        'param': 'dream_params', 'old': current_dream, 'new': new_dream,
                        'reason': f'{promotion_rate:.0%} dream promotion rate'
                    })
                elif promotion_rate == 0 and total_dreams > 30:
                    # Dreams are never useful — fewer dreams
                    new_dream = dict(current_dream)
                    new_dream['count'] = max(1, current_dream.get('count', DREAM_COUNT) - 1)
                    self._set_tunable('dream_params', new_dream,
                                      f'No dream promotions in {total_dreams} dreams — less dreaming')
                    results['tuned'].append({
                        'param': 'dream_params', 'old': current_dream, 'new': new_dream,
                        'reason': f'0 promotions from {total_dreams} dreams'
                    })
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # 2h. Boot limits — check compaction patterns
        # (Simple heuristic: if session_minutes < 30 before first compaction, boot is too heavy)
        try:
            session_min = float(self.get_config('total_session_minutes', 0) or 0)
            compaction_count = self.conn.execute(
                """SELECT COUNT(*) FROM brain_meta
                   WHERE key LIKE 'compaction_%' AND updated_at > datetime('now', '-7 days')"""
            ).fetchone()[0]

            current_limits = self._get_tunable('boot_limits', {
                'locked': CONTEXT_BOOT_LOCKED_LIMIT,
                'recall': CONTEXT_BOOT_RECALL_LIMIT,
                'recent': CONTEXT_BOOT_RECENT_LIMIT
            })
            if isinstance(current_limits, dict) and compaction_count > 3 and session_min < 30:
                # Frequent compaction, short sessions — boot context too heavy
                new_limits = dict(current_limits)
                new_limits['locked'] = max(10, current_limits.get('locked', 50) - 10)
                new_limits['recall'] = max(5, current_limits.get('recall', 15) - 3)
                self._set_tunable('boot_limits', new_limits,
                                  f'{compaction_count} compactions in <30min sessions — reducing boot')
                results['tuned'].append({
                    'param': 'boot_limits', 'old': current_limits, 'new': new_limits,
                    'reason': f'Frequent compaction ({compaction_count}x) in short sessions'
                })
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # ══════════════════════════════════════════════════════
        # CATEGORY 3: Graph Hygiene
        # ══════════════════════════════════════════════════════

        # 3a. Orphan node cleanup — 0 edges, 0 access in 30 days, not locked
        try:
            orphans = self.conn.execute(
                """SELECT n.id FROM nodes n
                   LEFT JOIN edges e1 ON n.id = e1.source_id
                   LEFT JOIN edges e2 ON n.id = e2.target_id
                   WHERE n.locked = 0 AND n.archived = 0
                     AND n.access_count <= 1
                     AND n.created_at < datetime('now', '-30 days')
                     AND e1.source_id IS NULL AND e2.target_id IS NULL
                   LIMIT 20"""
            ).fetchall()
            if orphans:
                orphan_ids = [r[0] for r in orphans]
                placeholders = ','.join('?' * len(orphan_ids))
                self.conn.execute(
                    f"UPDATE nodes SET archived = 1, updated_at = ? WHERE id IN ({placeholders})",
                    [ts] + orphan_ids
                )
                results['cleaned']['archived'] += len(orphan_ids)
        except Exception as _e:
            self._log_error("auto_heal", _e, "")

        # 3b. Edge weight normalization — cap non-structural edges at 0.9
        try:
            normalized = self.conn.execute(
                """UPDATE edges SET weight = 0.9
                   WHERE weight > 0.95
                     AND edge_type NOT IN ('reasoning_step', 'produced', 'corrected_by')"""
            ).rowcount
            results['cleaned']['edges_normalized'] = normalized or 0
        except Exception as _e:
            self._log_error("auto_heal", _e, "normalized = self.conn.execute(")

        # 3d. Stale evolution cleanup — >90 days, no engagement
        try:
            stale = self.conn.execute(
                """UPDATE nodes SET archived = 1, evolution_status = 'dismissed', updated_at = ?
                   WHERE type IN ('tension', 'hypothesis', 'pattern', 'aspiration')
                     AND evolution_status = 'active' AND archived = 0
                     AND created_at < datetime('now', '-90 days')
                     AND (last_accessed IS NULL OR last_accessed < datetime('now', '-60 days'))"""
                , (ts,)
            ).rowcount
            results['cleaned']['archived'] += (stale or 0)
        except Exception as _e:
            self._log_error("auto_heal", _e, "stale = self.conn.execute(")

        self.conn.commit()
        return results

    def prune_irrelevant_quotes(self, batch_size: int = 30,
                                 threshold: float = 0.50) -> Dict[str, Any]:
        """Prune auto-captured operator quotes that don't match their node.

        When remember_rich() auto-captures the last user message as user_raw_quote,
        the quote may be about a completely different topic than the node being created.
        E.g., user says "fix the CSS bug" then Claude encodes a node about embeddings —
        the CSS quote gets attached to the embedding node.

        This method uses embedding similarity to detect mismatches and removes the quote
        (preserving source_context as a record that a quote was pruned).

        Runs during idle. Threshold calibrated against Snowflake Arctic Embed:
        unrelated pairs score 0.46-0.62, related pairs score 0.74+.
        Default 0.50 catches clear mismatches while preserving tangentially related quotes.

        Returns:
            {'checked': int, 'pruned': int, 'pruned_nodes': [{'id': str, 'title': str, 'quote': str}]}
        """
        result = {'checked': 0, 'pruned': 0, 'pruned_nodes': []}

        if not embedder.is_ready():
            return result

        # Find nodes with auto-captured quotes (source_context starts with 'Auto-captured')
        rows = self.conn.execute(
            '''SELECT nm.node_id, nm.user_raw_quote, nm.source_context,
                      n.title, n.content, ne.embedding
               FROM node_metadata nm
               JOIN nodes n ON nm.node_id = n.id
               LEFT JOIN node_embeddings ne ON n.id = ne.node_id
               WHERE nm.user_raw_quote IS NOT NULL
                 AND nm.source_context LIKE 'Auto-captured%'
                 AND n.archived = 0
               ORDER BY n.created_at DESC
               LIMIT ?''',
            (batch_size,)
        ).fetchall()

        for row in rows:
            node_id, quote, source_ctx, title, content, node_emb = row
            result['checked'] += 1

            if not node_emb or not quote:
                continue

            # Embed the quote and compare to the node embedding
            quote_emb = embedder.embed(quote)
            if not quote_emb:
                continue

            sim = embedder.cosine_similarity(quote_emb, node_emb)

            if sim < threshold:
                # Clear mismatch — prune the quote but leave a trace
                self.conn.execute(
                    '''UPDATE node_metadata
                       SET user_raw_quote = NULL,
                           source_context = ?
                       WHERE node_id = ?''',
                    ('Pruned auto-quote (sim=%.2f, below %.2f): "%s"' % (
                        sim, threshold, quote[:100]),
                     node_id)
                )
                result['pruned'] += 1
                result['pruned_nodes'].append({
                    'id': node_id,
                    'title': title[:60] if title else '',
                    'quote': quote[:80],
                    'similarity': round(sim, 3),
                })

        if result['pruned'] > 0:
            self.conn.commit()

        return result

    def auto_tune(self, eval_period_days: int = 7) -> Dict[str, Any]:
        """
        Standalone self-tuning method. Adjusts brain parameters based on observed behavior.
        Can be called independently or as part of auto_heal.

        Returns dict of parameter changes made.
        Safe: never changes values by >10% per cycle.
        """
        # auto_heal already contains comprehensive tuning (categories 2a-2h).
        # This method provides a clean entry point and adds v5-specific tuning.
        results = {'tuned': []}

        # v5: Tune engineering memory boot limits based on token usage patterns
        try:
            purpose_count = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE type = 'purpose' AND archived = 0"
            ).fetchone()[0]
            impact_count = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE type = 'impact' AND archived = 0"
            ).fetchone()[0]
            vocab_count = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE type = 'vocabulary' AND archived = 0"
            ).fetchone()[0]

            current_eng_limits = self._get_tunable('engineering_boot_limits', {
                'purposes': 20, 'impacts': 10, 'constraints': 10,
                'conventions': 5, 'vocabulary': 15, 'file_inventory': 30
            })
            if isinstance(current_eng_limits, dict):
                new_limits = dict(current_eng_limits)
                changed = False

                # If we have many purpose nodes, increase boot limit
                if purpose_count > 15 and new_limits.get('purposes', 20) < 30:
                    new_limits['purposes'] = min(30, new_limits.get('purposes', 20) + 5)
                    changed = True
                # If vocabulary is growing, increase limit
                if vocab_count > 10 and new_limits.get('vocabulary', 15) < 25:
                    new_limits['vocabulary'] = min(25, new_limits.get('vocabulary', 15) + 5)
                    changed = True
                # If impacts are growing, ensure they're all visible (safety-critical)
                if impact_count > 8 and new_limits.get('impacts', 10) < impact_count:
                    new_limits['impacts'] = min(20, impact_count)
                    changed = True

                if changed:
                    self._set_tunable('engineering_boot_limits', new_limits,
                                      f'Adjusted for {purpose_count} purposes, {impact_count} impacts, {vocab_count} vocab')
                    results['tuned'].append({
                        'param': 'engineering_boot_limits',
                        'old': current_eng_limits, 'new': new_limits,
                        'reason': f'Growing engineering memory ({purpose_count}p/{impact_count}i/{vocab_count}v)'
                    })
        except Exception as _e:
            self._log_error("auto_tune", _e, "tuning engineering_boot_limits based on memory sizes")

        # v5: Tune correction sensitivity based on correction frequency
        try:
            recent_corrections = self.conn.execute(
                f"SELECT COUNT(*) FROM correction_traces WHERE created_at > datetime('now', '-{eval_period_days} days')"
            ).fetchone()[0]
            recent_validations = self.conn.execute(
                f"""SELECT COUNT(*) FROM node_metadata
                    WHERE last_validated > datetime('now', '-{eval_period_days} days')"""
            ).fetchone()[0]

            if recent_corrections + recent_validations >= 5:
                correction_ratio = recent_corrections / (recent_corrections + recent_validations)
                # If corrections greatly outnumber validations, the brain is frequently wrong
                # → surface more corrections at boot, be more cautious
                current_correction_boot = self._get_tunable('correction_boot_limit', 3)
                if not isinstance(current_correction_boot, (int, float)):
                    current_correction_boot = 3

                if correction_ratio > 0.7 and current_correction_boot < 5:
                    new_limit = min(5, int(current_correction_boot) + 1)
                    self._set_tunable('correction_boot_limit', new_limit,
                                      f'High correction ratio ({correction_ratio:.0%}) — showing more patterns at boot')
                    results['tuned'].append({
                        'param': 'correction_boot_limit',
                        'old': current_correction_boot, 'new': new_limit,
                        'reason': f'{correction_ratio:.0%} correction ratio'
                    })
                elif correction_ratio < 0.3 and current_correction_boot > 2:
                    new_limit = max(2, int(current_correction_boot) - 1)
                    self._set_tunable('correction_boot_limit', new_limit,
                                      f'Low correction ratio ({correction_ratio:.0%}) — fewer patterns needed')
                    results['tuned'].append({
                        'param': 'correction_boot_limit',
                        'old': current_correction_boot, 'new': new_limit,
                        'reason': f'{correction_ratio:.0%} correction ratio — improving'
                    })
        except Exception as _e:
            self._log_error("auto_tune", _e, "tuning correction_boot_limit based on correction frequency")

        # v5: Tune session synthesis sensitivity
        try:
            total_syntheses = self.conn.execute(
                "SELECT COUNT(*) FROM session_syntheses"
            ).fetchone()[0]
            empty_syntheses = self.conn.execute(
                "SELECT COUNT(*) FROM session_syntheses WHERE decisions_made IS NULL AND corrections_received IS NULL"
            ).fetchone()[0]

            if total_syntheses >= 5 and empty_syntheses > total_syntheses * 0.5:
                # Most syntheses are empty — tracking is too sparse
                results['tuned'].append({
                    'param': 'synthesis_observation',
                    'note': '%d/%d syntheses empty — need more track_session_event calls' % (
                        empty_syntheses, total_syntheses),
                })
        except Exception as _e:
            self._log_error("auto_tune", _e, "evaluating session synthesis density for tuning")

        # v5.2: Embedding model calibration — measure actual similarity baseline
        # so thresholds adapt to the model's operating range, not hardcoded assumptions.
        # Samples random node pairs to estimate the floor (unrelated) and ceiling (similar).
        # Runs at most once per day (cached in brain_meta).
        try:
            if embedder.is_ready():
                last_calibration = self.get_config('embedding_calibration_at')
                should_calibrate = True
                if last_calibration:
                    from datetime import datetime as _dt, timezone as _tz
                    try:
                        cal_dt = _dt.fromisoformat(last_calibration.replace('Z', '+00:00'))
                        age_hours = (_dt.now(_tz.utc) - cal_dt).total_seconds() / 3600
                        should_calibrate = age_hours > 24
                    except Exception:
                        pass

                if should_calibrate:
                    import random as _random
                    # Sample diverse nodes (different types, different ages)
                    sample_rows = self.conn.execute(
                        """SELECT ne.embedding FROM node_embeddings ne
                           JOIN nodes n ON n.id = ne.node_id
                           WHERE n.archived = 0 AND n.content IS NOT NULL
                             AND LENGTH(n.content) > 50
                           ORDER BY RANDOM() LIMIT 40"""
                    ).fetchall()

                    if len(sample_rows) >= 20:
                        embeddings = [r[0] for r in sample_rows]
                        # Compute pairwise similarities for a random subset
                        sims = []
                        indices = list(range(len(embeddings)))
                        pairs = []
                        for i in range(len(indices)):
                            for j in range(i + 1, len(indices)):
                                pairs.append((i, j))
                        # Sample up to 100 pairs
                        if len(pairs) > 100:
                            pairs = _random.sample(pairs, 100)
                        for i, j in pairs:
                            sim = embedder.cosine_similarity(embeddings[i], embeddings[j])
                            sims.append(sim)

                        if sims:
                            sims.sort()
                            floor = sims[int(len(sims) * 0.10)]  # 10th percentile
                            median = sims[len(sims) // 2]
                            p90 = sims[int(len(sims) * 0.90)]  # 90th percentile
                            mean = sum(sims) / len(sims)

                            calibration = {
                                'floor_p10': round(floor, 4),
                                'median': round(median, 4),
                                'p90': round(p90, 4),
                                'mean': round(mean, 4),
                                'sample_pairs': len(sims),
                                'sample_nodes': len(embeddings),
                            }

                            import json as _json
                            self.set_config('embedding_calibration',
                                            _json.dumps(calibration))
                            self.set_config('embedding_calibration_at', self.now())

                            # Now adjust thresholds relative to measured distribution
                            current_thresholds = self._get_tunable('similarity_thresholds', {})
                            if isinstance(current_thresholds, dict):
                                new_thresholds = dict(current_thresholds)
                                changed = False

                                # Contradiction: should be well above median (genuine similarity)
                                # Target: median + 60% of (p90 - median)
                                ideal_contradiction = round(median + 0.6 * (p90 - median), 2)
                                current_contradiction = new_thresholds.get('contradiction', 0.65)
                                if abs(ideal_contradiction - current_contradiction) > 0.03:
                                    # Clamp to safe range and limit change to 10%
                                    new_val = max(0.55, min(0.80, ideal_contradiction))
                                    delta = new_val - current_contradiction
                                    if abs(delta) > current_contradiction * 0.10:
                                        new_val = round(current_contradiction + 0.10 * (1 if delta > 0 else -1), 2)
                                    new_thresholds['contradiction'] = new_val
                                    changed = True

                                # Correction clustering: slightly below contradiction
                                ideal_correction = round(median + 0.4 * (p90 - median), 2)
                                current_correction = new_thresholds.get('correction_cluster', 0.60)
                                if abs(ideal_correction - current_correction) > 0.03:
                                    new_val = max(0.45, min(0.75, ideal_correction))
                                    delta = new_val - current_correction
                                    if abs(delta) > current_correction * 0.10:
                                        new_val = round(current_correction + 0.10 * (1 if delta > 0 else -1), 2)
                                    new_thresholds['correction_cluster'] = new_val
                                    changed = True

                                # Orphan backing: high — need genuine similarity
                                ideal_orphan = round(median + 0.7 * (p90 - median), 2)
                                current_orphan = new_thresholds.get('orphan_backing', 0.70)
                                if abs(ideal_orphan - current_orphan) > 0.03:
                                    new_val = max(0.55, min(0.85, ideal_orphan))
                                    delta = new_val - current_orphan
                                    if abs(delta) > current_orphan * 0.10:
                                        new_val = round(current_orphan + 0.10 * (1 if delta > 0 else -1), 2)
                                    new_thresholds['orphan_backing'] = new_val
                                    changed = True

                                if changed:
                                    self._set_tunable('similarity_thresholds', new_thresholds,
                                                      'Model calibration: floor=%.3f median=%.3f p90=%.3f' % (
                                                          floor, median, p90))
                                    results['tuned'].append({
                                        'param': 'similarity_thresholds (calibrated)',
                                        'calibration': calibration,
                                        'old': current_thresholds,
                                        'new': new_thresholds,
                                        'reason': 'Model baseline: floor=%.3f median=%.3f p90=%.3f' % (
                                            floor, median, p90),
                                    })
                                else:
                                    results['tuned'].append({
                                        'param': 'embedding_calibration',
                                        'calibration': calibration,
                                        'reason': 'Calibrated — thresholds already aligned',
                                    })
        except Exception as _e:
            self._log_error("auto_tune", _e, "embedding model calibration")

        self.conn.commit()
        return results

    def prompt_reflection(self) -> List[str]:
        """Generate reflection prompts based on recent session activity.

        Called during idle maintenance. Surfaces questions the host should consider
        encoding as lessons, corrections, or mental model updates.

        Returns a list of reflection prompts (strings) for the host to act on.
        """
        prompts = []

        # 1. New node types added this session without lifecycle audit
        try:
            recent_types = self.conn.execute(
                """SELECT DISTINCT type FROM nodes
                   WHERE created_at > datetime('now', '-2 hours')
                     AND type NOT IN (SELECT DISTINCT type FROM nodes
                                      WHERE created_at < datetime('now', '-2 hours'))"""
            ).fetchall()
            for (new_type,) in recent_types:
                prompts.append(
                    "NEW NODE TYPE '%s' introduced this session. "
                    "Lifecycle check: does it connect at birth? participate in consolidation? "
                    "get merged when duplicated? have the right decay rate? "
                    "If not, encode a constraint or fix the gap." % new_type
                )
        except Exception:
            pass

        # 2. High edit-to-remember ratio — lots of work, few learnings encoded
        try:
            activity = self._get_session_activity()
            recent_edits = activity.get('edit_check_count', 0)
            recent_remembers = activity.get('remember_count', 0)
            if recent_edits > 10 and recent_remembers < 2:
                prompts.append(
                    "HIGH EDIT-TO-REMEMBER RATIO: %d edits but only %d remembers. "
                    "Were there decisions, corrections, or patterns worth encoding? "
                    "The brain can only learn from what gets stored." % (recent_edits, recent_remembers)
                )
        except Exception:
            pass

        # 3. Corrections without underlying patterns extracted
        try:
            unpattern = self.conn.execute(
                """SELECT COUNT(*) FROM correction_traces
                   WHERE underlying_pattern IS NULL
                     AND created_at > datetime('now', '-24 hours')"""
            ).fetchone()[0]
            if unpattern >= 2:
                prompts.append(
                    "%d recent corrections have no underlying_pattern. "
                    "Look for the common thread — is there a systemic issue? "
                    "A pattern encoded once prevents the same mistake across all future sessions." % unpattern
                )
        except Exception:
            pass

        # 4. Features built without engineering memory
        try:
            recent_files = self.conn.execute(
                """SELECT DISTINCT title FROM nodes
                   WHERE type = 'file' AND archived = 0
                     AND last_accessed > datetime('now', '-2 hours')
                     AND access_count >= 3"""
            ).fetchall()
            for (fname,) in recent_files:
                has_purpose = self.conn.execute(
                    "SELECT 1 FROM nodes WHERE type IN ('purpose', 'mechanism') AND archived = 0 AND title LIKE ?",
                    (f'%{fname}%',)
                ).fetchone()
                if not has_purpose:
                    prompts.append(
                        "FILE '%s' was heavily accessed but has no purpose or mechanism node. "
                        "What does it do? Why does it exist? "
                        "This context eliminates warm-up time in future sessions." % fname
                    )
        except Exception:
            pass

        # 5. Session events tracked but not synthesized
        try:
            total_events = sum(len(v) for v in self._session_state.values())
            if total_events >= 3:
                event_types = [k for k, v in self._session_state.items() if v]
                prompts.append(
                    "SESSION HAS %d unprocessed events (%s). "
                    "Consider: what's the transferable insight from this session? "
                    "What would help a fresh Claude hit the ground running?" % (
                        total_events, ', '.join(event_types))
                )
        except Exception:
            pass

        return prompts

    def auto_discover_evolutions(self) -> Dict[str, Any]:
        """
        Graph-aware auto-discovery of evolutions from the brain's own structure.
        Analyzes embeddings, edges, access patterns, and emotion to find:
          - Tensions: semantic contradictions between locked nodes
          - Patterns: correction clusters, co-access patterns, decay patterns
          - Hypotheses: orphan beliefs, dream promotions, implicit assumptions
          - Aspirations: emotional trajectories, recurring catalysts

        Returns structured dict (designed for future self-model consumption):
        {
            'tensions': [created tension dicts],
            'patterns': [created pattern dicts],
            'hypotheses': [created hypothesis dicts],
            'aspirations': [created aspiration dicts],
            '_stats': { analysis metadata for self-model }
        }
        """
        result = {
            'tensions': [], 'patterns': [], 'hypotheses': [], 'aspirations': [],
            '_stats': {
                'locked_rules_scanned': 0, 'pairs_compared': 0,
                'corrections_analyzed': 0, 'orphan_beliefs_found': 0,
                'emotion_trends': {},
            }
        }

        # ══════════════════════════════════════════════════
        # 1. TENSION DISCOVERY
        # ══════════════════════════════════════════════════

        # 1a. Semantic contradictions — locked directive pairs with high similarity
        if embedder.is_ready():
            try:
                cursor = self.conn.execute(
                    """SELECT n.id, n.type, n.title, n.content, ne.embedding
                       FROM nodes n
                       JOIN node_embeddings ne ON n.id = ne.node_id
                       WHERE n.locked = 1 AND n.archived = 0
                         AND n.type IN ('rule', 'decision', 'arch_constraint')
                       ORDER BY RANDOM() LIMIT 50"""
                )
                candidates = [
                    {'id': r[0], 'type': r[1], 'title': r[2], 'content': r[3] or '', 'embedding': r[4]}
                    for r in cursor.fetchall()
                ]
                result['_stats']['locked_rules_scanned'] = len(candidates)

                # Compare pairs — threshold tunable, calibrated against model baseline
                _sim_t = self._get_tunable('similarity_thresholds', {})
                contradiction_threshold = _sim_t.get('contradiction', 0.65) if isinstance(_sim_t, dict) else 0.65
                close_pairs = []
                pairs_checked = 0
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        pairs_checked += 1
                        sim = embedder.cosine_similarity(candidates[i]['embedding'], candidates[j]['embedding'])
                        if sim > contradiction_threshold:
                            # Check no existing contradicts edge
                            existing = self.conn.execute(
                                """SELECT COUNT(*) FROM edges
                                   WHERE relation = 'contradicts'
                                     AND ((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))""",
                                (candidates[i]['id'], candidates[j]['id'], candidates[j]['id'], candidates[i]['id'])
                            ).fetchone()[0]
                            if existing == 0:
                                close_pairs.append((candidates[i], candidates[j], sim))
                result['_stats']['pairs_compared'] = pairs_checked

                # Create tensions (max 2 per cycle)
                for node_a, node_b, sim in sorted(close_pairs, key=lambda x: -x[2])[:2]:
                    tension = self.create_tension(
                        title="%s vs %s" % (node_a['title'][:40], node_b['title'][:40]),
                        content="Auto-discovered: these locked nodes are semantically similar (cosine %.2f) but may prescribe different approaches. Review whether they conflict or complement." % sim,
                        node_a_id=node_a['id'],
                        node_b_id=node_b['id'],
                        keywords="auto-discovered tension semantic %s %s" % (node_a['title'][:15], node_b['title'][:15])
                    )
                    result['tensions'].append(tension)
            except Exception:
                pass

        # 1b. Temporal contradictions — same topic, different conclusions over time
        if embedder.is_ready():
            try:
                # Find decisions created >7 days apart on same project
                decisions = self.conn.execute(
                    """SELECT n.id, n.title, n.content, n.created_at, n.project, ne.embedding
                       FROM nodes n
                       JOIN node_embeddings ne ON n.id = ne.node_id
                       WHERE n.type = 'decision' AND n.locked = 1 AND n.archived = 0
                       ORDER BY n.created_at DESC LIMIT 40"""
                ).fetchall()

                for i in range(len(decisions)):
                    for j in range(i + 1, len(decisions)):
                        # Same project, >7 days apart
                        if decisions[i][4] != decisions[j][4]:
                            continue
                        try:
                            from datetime import datetime as _dt
                            dt_i = _dt.fromisoformat(decisions[i][3].replace('Z', '+00:00'))
                            dt_j = _dt.fromisoformat(decisions[j][3].replace('Z', '+00:00'))
                            days_apart = abs((dt_i - dt_j).days)
                        except Exception:
                            continue
                        if days_apart < 7:
                            continue

                        sim = embedder.cosine_similarity(decisions[i][5], decisions[j][5])
                        if sim > 0.70:
                            # High similarity + time gap = potential evolution/contradiction
                            existing = self.conn.execute(
                                """SELECT COUNT(*) FROM edges
                                   WHERE relation = 'contradicts'
                                     AND ((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))""",
                                (decisions[i][0], decisions[j][0], decisions[j][0], decisions[i][0])
                            ).fetchone()[0]
                            if existing == 0:
                                newer = decisions[i] if decisions[i][3] > decisions[j][3] else decisions[j]
                                older = decisions[j] if newer == decisions[i] else decisions[i]
                                tension = self.create_tension(
                                    title="Earlier '%s' vs later '%s'" % (older[1][:35], newer[1][:35]),
                                    content="Auto-discovered: these decisions are %d days apart on the same topic (cosine %.2f). The later one may supersede or contradict the earlier. Review and resolve." % (days_apart, sim),
                                    node_a_id=older[0],
                                    node_b_id=newer[0],
                                    keywords="auto-discovered tension temporal %s" % (older[4] or 'global')
                                )
                                result['tensions'].append(tension)
                                if len(result['tensions']) >= 3:
                                    break
                    if len(result['tensions']) >= 3:
                        break
            except Exception:
                pass

        # ══════════════════════════════════════════════════
        # 2. PATTERN DISCOVERY
        # ══════════════════════════════════════════════════

        # 2a. Correction clusters — group corrections/bug_lessons by embedding similarity
        if embedder.is_ready():
            try:
                corrections = self.conn.execute(
                    """SELECT n.id, n.title, n.content, ne.embedding
                       FROM nodes n
                       JOIN node_embeddings ne ON n.id = ne.node_id
                       WHERE (n.title LIKE 'Correction:%' OR n.type = 'bug_lesson')
                         AND n.archived = 0
                       ORDER BY n.created_at DESC LIMIT 30"""
                ).fetchall()
                result['_stats']['corrections_analyzed'] = len(corrections)

                # Simple clustering: threshold tunable, calibrated against model baseline
                _sim_t = self._get_tunable('similarity_thresholds', {})
                correction_cluster_threshold = _sim_t.get('correction_cluster', 0.60) if isinstance(_sim_t, dict) else 0.60
                clusters = []  # list of lists of correction indices
                assigned = set()
                for i in range(len(corrections)):
                    if i in assigned:
                        continue
                    cluster = [i]
                    assigned.add(i)
                    for j in range(i + 1, len(corrections)):
                        if j in assigned:
                            continue
                        sim = embedder.cosine_similarity(corrections[i][3], corrections[j][3])
                        if sim > correction_cluster_threshold:
                            cluster.append(j)
                            assigned.add(j)
                    if len(cluster) >= 3:
                        clusters.append(cluster)

                for cluster in clusters[:2]:
                    titles = [corrections[idx][1][:40] for idx in cluster]
                    # Check no existing pattern about this cluster
                    check_kw = titles[0][:20]
                    existing = self.conn.execute(
                        "SELECT COUNT(*) FROM nodes WHERE type = 'pattern' AND keywords LIKE ? AND archived = 0",
                        ('%auto-discovered pattern correction%' + check_kw + '%',)
                    ).fetchone()[0]
                    if existing > 0:
                        continue

                    pattern = self.create_pattern(
                        title="Corrections cluster: %s (%d instances)" % (titles[0][:30], len(cluster)),
                        content="Auto-discovered: %d corrections/bug lessons cluster together semantically. Titles: %s. This area may need a locked rule." % (
                            len(cluster), '; '.join(titles)),
                        evidence='; '.join(titles),
                        keywords="auto-discovered pattern correction-cluster %s" % check_kw
                    )
                    result['patterns'].append(pattern)
            except Exception:
                pass

        # 2b. Co-access patterns — heavily co-accessed but never explicitly connected
        try:
            strong_coaccesses = self.conn.execute(
                """SELECT e.source_id, e.target_id, e.weight, n1.title, n2.title
                   FROM edges e
                   JOIN nodes n1 ON e.source_id = n1.id
                   JOIN nodes n2 ON e.target_id = n2.id
                   WHERE e.relation = 'co_accessed' AND e.weight > 0.6
                     AND n1.archived = 0 AND n2.archived = 0
                     AND NOT EXISTS (
                       SELECT 1 FROM edges e2
                       WHERE e2.source_id = e.source_id AND e2.target_id = e.target_id
                         AND e2.relation != 'co_accessed'
                     )
                   ORDER BY e.weight DESC LIMIT 5"""
            ).fetchall()

            for src, tgt, weight, title_a, title_b in strong_coaccesses[:2]:
                existing = self.conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE type = 'pattern' AND keywords LIKE ? AND archived = 0",
                    ('%auto-discovered pattern co-access%' + src[:12] + '%',)
                ).fetchone()[0]
                if existing > 0:
                    continue

                pattern = self.create_pattern(
                    title="Always used together: '%s' + '%s'" % (title_a[:30], title_b[:30]),
                    content="Auto-discovered: these concepts are co-accessed with weight %.2f but have no explicit relationship. Consider connecting them or creating a unifying concept." % weight,
                    evidence="co_accessed edge weight: %.2f" % weight,
                    keywords="auto-discovered pattern co-access %s" % src[:12]
                )
                result['patterns'].append(pattern)
        except Exception:
            pass

        # 2c. Decay patterns — node types that get created and consistently archived
        try:
            decay_stats = self.conn.execute(
                """SELECT type,
                          SUM(CASE WHEN archived = 1 THEN 1 ELSE 0 END) as archived_count,
                          SUM(CASE WHEN archived = 0 THEN 1 ELSE 0 END) as active_count
                   FROM nodes
                   WHERE type NOT IN ('context', 'thought', 'intuition')
                   GROUP BY type
                   HAVING archived_count > active_count AND archived_count >= 3"""
            ).fetchall()

            for ntype, archived_count, active_count in decay_stats[:1]:
                existing = self.conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE type = 'pattern' AND keywords LIKE ? AND archived = 0",
                    ('%auto-discovered pattern decay-' + ntype + '%',)
                ).fetchone()[0]
                if existing > 0:
                    continue

                pattern = self.create_pattern(
                    title="Retention issue: %s nodes (%d archived vs %d active)" % (ntype, archived_count, active_count),
                    content="Auto-discovered: the brain creates %s nodes but they consistently decay. More are archived (%d) than active (%d). Consider locking important ones or adjusting decay rates." % (ntype, archived_count, active_count),
                    keywords="auto-discovered pattern decay-%s retention" % ntype
                )
                result['patterns'].append(pattern)
        except Exception:
            pass

        # ══════════════════════════════════════════════════
        # 3. HYPOTHESIS DISCOVERY
        # ══════════════════════════════════════════════════

        # 3a. Orphan beliefs — high-access unlocked nodes with no locked backing
        if embedder.is_ready():
            try:
                orphans = self.conn.execute(
                    """SELECT n.id, n.title, n.content, n.access_count, ne.embedding
                       FROM nodes n
                       JOIN node_embeddings ne ON n.id = ne.node_id
                       WHERE n.locked = 0 AND n.archived = 0
                         AND n.access_count >= 5
                         AND n.type NOT IN ('context', 'thought', 'intuition', 'tension', 'hypothesis', 'pattern', 'aspiration')
                       ORDER BY n.access_count DESC LIMIT 10"""
                ).fetchall()

                locked_embeddings = self.conn.execute(
                    """SELECT ne.embedding FROM node_embeddings ne
                       JOIN nodes n ON n.id = ne.node_id
                       WHERE n.locked = 1 AND n.archived = 0
                       LIMIT 100"""
                ).fetchall()

                _sim_t = self._get_tunable('similarity_thresholds', {})
                orphan_backing_threshold = _sim_t.get('orphan_backing', 0.70) if isinstance(_sim_t, dict) else 0.70
                orphan_count = 0
                for nid, title, content, access_count, emb in orphans:
                    # Check if any locked node is similar (backing this belief)
                    has_backing = False
                    for (locked_emb,) in locked_embeddings:
                        sim = embedder.cosine_similarity(emb, locked_emb)
                        if sim > orphan_backing_threshold:
                            has_backing = True
                            break

                    if not has_backing:
                        existing = self.conn.execute(
                            "SELECT COUNT(*) FROM nodes WHERE type = 'hypothesis' AND keywords LIKE ? AND archived = 0",
                            ('%auto-discovered hypothesis orphan%' + nid[:12] + '%',)
                        ).fetchone()[0]
                        if existing > 0:
                            continue

                        hyp = self.create_hypothesis(
                            title="'%s' relied on %dx but never locked" % (title[:50], access_count),
                            content="Auto-discovered: this node is accessed frequently (%d times) but no locked rule or decision backs it. Should it be promoted to a rule? Original: %s" % (access_count, (content or '')[:200]),
                            confidence=0.4,
                            keywords="auto-discovered hypothesis orphan-belief %s" % nid[:12]
                        )
                        result['hypotheses'].append(hyp)
                        orphan_count += 1
                        if orphan_count >= 2:
                            break

                result['_stats']['orphan_beliefs_found'] = orphan_count
            except Exception:
                pass

        # 3b. Dream promotions — intuitions that gained traction
        try:
            popular_intuitions = self.conn.execute(
                """SELECT id, title, content FROM nodes
                   WHERE type = 'intuition' AND archived = 0 AND access_count >= 2
                   LIMIT 3"""
            ).fetchall()

            for nid, title, content in popular_intuitions:
                # Check not already promoted
                existing = self.conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE type = 'hypothesis' AND keywords LIKE ? AND archived = 0",
                    ('%auto-discovered hypothesis dream-promotion%' + nid[:12] + '%',)
                ).fetchone()[0]
                if existing > 0:
                    continue

                clean_title = title.replace('Dream connection: ', '').replace('Dream observation: ', '')
                hyp = self.create_hypothesis(
                    title="Dream insight gaining traction: %s" % clean_title[:50],
                    content="Auto-discovered: this dream intuition was accessed 2+ times, suggesting it resonated. Original dream: %s" % (content or '')[:300],
                    confidence=0.3,
                    keywords="auto-discovered hypothesis dream-promotion %s" % nid[:12]
                )
                result['hypotheses'].append(hyp)

                # Archive the intuition (promoted)
                self.conn.execute("UPDATE nodes SET archived = 1 WHERE id = ?", (nid,))
        except Exception:
            pass

        # 3c. Implicit assumptions — high in-degree unlocked nodes
        try:
            load_bearing = self.conn.execute(
                """SELECT n.id, n.title, COUNT(e.source_id) as in_degree
                   FROM nodes n
                   JOIN edges e ON e.target_id = n.id
                   WHERE n.locked = 0 AND n.archived = 0
                     AND n.type NOT IN ('context', 'thought', 'intuition')
                   GROUP BY n.id
                   HAVING in_degree >= 5
                   ORDER BY in_degree DESC LIMIT 3"""
            ).fetchall()

            for nid, title, in_degree in load_bearing[:1]:
                existing = self.conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE type = 'hypothesis' AND keywords LIKE ? AND archived = 0",
                    ('%auto-discovered hypothesis implicit%' + nid[:12] + '%',)
                ).fetchone()[0]
                if existing > 0:
                    continue

                hyp = self.create_hypothesis(
                    title="'%s' has %d dependents but isn't locked" % (title[:45], in_degree),
                    content="Auto-discovered: this node is load-bearing (%d edges point to it) but unlocked. If it decays, dependent knowledge may become orphaned. Should it be locked?" % in_degree,
                    confidence=0.5,
                    keywords="auto-discovered hypothesis implicit-assumption %s" % nid[:12]
                )
                result['hypotheses'].append(hyp)
        except Exception:
            pass

        # ══════════════════════════════════════════════════
        # 4. ASPIRATION DISCOVERY
        # ══════════════════════════════════════════════════

        # 4a. Emotional trajectory — rising excitement about a topic
        try:
            # Compare avg emotion by project: last 7 days vs prior 30 days
            recent = self.conn.execute(
                """SELECT project, AVG(emotion), COUNT(*) FROM nodes
                   WHERE emotion > 0 AND archived = 0
                     AND created_at > datetime('now', '-7 days')
                     AND project IS NOT NULL AND project != ''
                   GROUP BY project HAVING COUNT(*) >= 3"""
            ).fetchall()

            older = self.conn.execute(
                """SELECT project, AVG(emotion) FROM nodes
                   WHERE emotion > 0 AND archived = 0
                     AND created_at BETWEEN datetime('now', '-37 days') AND datetime('now', '-7 days')
                     AND project IS NOT NULL AND project != ''
                   GROUP BY project"""
            ).fetchall()
            older_map = {r[0]: r[1] for r in older}

            for project, recent_avg, count in recent:
                old_avg = older_map.get(project, 0.3)  # default baseline
                trend = recent_avg - old_avg
                result['_stats']['emotion_trends'][project] = {
                    'recent': round(recent_avg, 2), 'older': round(old_avg, 2), 'delta': round(trend, 2)
                }

                if trend > 0.2:
                    existing = self.conn.execute(
                        "SELECT COUNT(*) FROM nodes WHERE type = 'aspiration' AND keywords LIKE ? AND archived = 0",
                        ('%auto-discovered aspiration emotion%' + project[:15] + '%',)
                    ).fetchone()[0]
                    if existing > 0:
                        continue

                    asp = self.create_aspiration(
                        title="Growing energy around %s (emotion +%.2f)" % (project, trend),
                        content="Auto-discovered: emotional intensity for %s has increased from %.2f to %.2f over the last week (%d recent nodes). This sustained excitement may indicate an emerging goal or aspiration." % (
                            project, old_avg, recent_avg, count),
                        project=project,
                        keywords="auto-discovered aspiration emotion-trajectory %s" % project[:15]
                    )
                    result['aspirations'].append(asp)
        except Exception:
            pass

        # 4b. Recurring catalysts — repeated high-emotion events around same topic
        try:
            high_emotion = self.conn.execute(
                """SELECT n.id, n.title, n.keywords, n.emotion, ne.embedding
                   FROM nodes n
                   LEFT JOIN node_embeddings ne ON n.id = ne.node_id
                   WHERE n.emotion >= 0.7 AND n.archived = 0
                     AND n.type IN ('decision', 'catalyst', 'rule')
                     AND n.created_at > datetime('now', '-30 days')
                   ORDER BY n.emotion DESC LIMIT 15"""
            ).fetchall()

            if len(high_emotion) >= 2 and embedder.is_ready():
                # Cluster high-emotion nodes
                clusters = []
                assigned = set()
                for i in range(len(high_emotion)):
                    if i in assigned or not high_emotion[i][4]:
                        continue
                    cluster = [i]
                    assigned.add(i)
                    for j in range(i + 1, len(high_emotion)):
                        if j in assigned or not high_emotion[j][4]:
                            continue
                        sim = embedder.cosine_similarity(high_emotion[i][4], high_emotion[j][4])
                        if sim > 0.55:
                            cluster.append(j)
                            assigned.add(j)
                    if len(cluster) >= 2:
                        clusters.append(cluster)

                for cluster in clusters[:1]:
                    titles = [high_emotion[idx][1][:40] for idx in cluster]
                    avg_emotion = sum(high_emotion[idx][3] for idx in cluster) / len(cluster)
                    kw_check = titles[0][:15]

                    existing = self.conn.execute(
                        "SELECT COUNT(*) FROM nodes WHERE type = 'aspiration' AND keywords LIKE ? AND archived = 0",
                        ('%auto-discovered aspiration catalyst%' + kw_check + '%',)
                    ).fetchone()[0]
                    if existing > 0:
                        continue

                    asp = self.create_aspiration(
                        title="Repeated energy: %s (%d events, avg emotion %.1f)" % (titles[0][:30], len(cluster), avg_emotion),
                        content="Auto-discovered: %d high-emotion events cluster around this theme. Titles: %s. Sustained emotional investment suggests an underlying aspiration." % (
                            len(cluster), '; '.join(titles)),
                        keywords="auto-discovered aspiration catalyst-cluster %s" % kw_check
                    )
                    result['aspirations'].append(asp)
        except Exception:
            pass

        self.save()
        return result

    def create_tension(self, title: str, content: str, node_a_id: str, node_b_id: str,
                       project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Create a tension — a contradiction between two existing nodes.

        The brain noticed that node_a and node_b conflict. Tensions never decay
        until resolved. Resolution produces a decision or rule.

        Args:
            title: Description of the contradiction (e.g. "One-tap simplicity contradicts tiered pricing")
            content: Detailed explanation of the conflict
            node_a_id: First conflicting node
            node_b_id: Second conflicting node
            project: Optional project scope

        Returns:
            Dict with id, type, title, evolution_status, connected nodes
        """
        result = self.remember(
            type='tension', title=f'⚡ TENSION — {title}', content=content,
            keywords=kwargs.get('keywords', ''),
            locked=True,  # Tensions never decay until resolved
            emotion=0.6, emotion_label='concern',
            project=project, **{k: v for k, v in kwargs.items() if k not in ('keywords',)}
        )
        node_id = result['id']

        # Set evolution status
        self.conn.execute(
            "UPDATE nodes SET evolution_status = 'active' WHERE id = ?", (node_id,))
        self.conn.commit()

        # Connect to the conflicting nodes
        self.connect(node_id, node_a_id, 'contradicts', 0.9)
        self.connect(node_id, node_b_id, 'contradicts', 0.9)

        result['evolution_status'] = 'active'
        result['connected'] = [node_a_id, node_b_id]
        return result

    def create_hypothesis(self, title: str, content: str, confidence: float = 0.5,
                          project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Create a hypothesis — an untested belief with a confidence score.

        Confidence adjusts with evidence: >0.9 → validated (becomes decision),
        <0.2 → disproven (archived with lesson).

        Args:
            title: The belief (e.g. "Embedding blend weight should be higher for technical queries")
            content: Reasoning behind the hypothesis
            confidence: Initial confidence (0.0-1.0, default 0.5)
            project: Optional project scope
        """
        result = self.remember(
            type='hypothesis', title=f'🔮 HYPOTHESIS — {title}', content=content,
            keywords=kwargs.get('keywords', ''),
            confidence=max(0.0, min(1.0, confidence)),
            emotion=0.4, emotion_label='curiosity',
            project=project, **{k: v for k, v in kwargs.items() if k not in ('keywords',)}
        )
        node_id = result['id']

        self.conn.execute(
            "UPDATE nodes SET evolution_status = 'active' WHERE id = ?", (node_id,))
        self.conn.commit()

        result['evolution_status'] = 'active'
        result['confidence'] = confidence
        return result

    def create_pattern(self, title: str, content: str, evidence: Optional[str] = None,
                       project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Create a pattern — a meta-observation about recurring behavior.

        Detected from data (recall logs, miss logs, correction events).
        Surfaced to user for confirmation: "I noticed this — is it real?"

        Args:
            title: The observation (e.g. "Tom corrects UI decisions 3x more than architecture")
            content: Evidence and data supporting the pattern
            evidence: Specific data points
            project: Optional project scope
        """
        full_content = content
        if evidence:
            full_content += f'\n\nEvidence: {evidence}'

        result = self.remember(
            type='pattern', title=f'📊 PATTERN — {title}', content=full_content,
            keywords=kwargs.get('keywords', ''),
            confidence=0.3,  # Patterns start low — need confirmation
            emotion=0.3, emotion_label='curiosity',
            project=project, **{k: v for k, v in kwargs.items() if k not in ('keywords',)}
        )
        node_id = result['id']

        self.conn.execute(
            "UPDATE nodes SET evolution_status = 'active' WHERE id = ?", (node_id,))
        self.conn.commit()

        result['evolution_status'] = 'active'
        return result

    def create_catalyst(self, title: str, content: str, resulting_decision_ids: Optional[List[str]] = None,
                        project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Create a catalyst — an emotional inflection point that changed direction.

        High-emotion, never-decay. Connected to the decisions that resulted.
        Teaches the brain to recognize similar inflection moments.

        Args:
            title: What happened (e.g. "Tom's frustration with repeated failure triggered optimization reframe")
            content: The full story — what was said, what changed, what was learned
            resulting_decision_ids: Decisions/rules that resulted from this catalyst
            project: Optional project scope
        """
        result = self.remember(
            type='catalyst', title=f'🔥 CATALYST — {title}', content=content,
            keywords=kwargs.get('keywords', ''),
            locked=True,  # Catalysts are permanent
            emotion=0.8, emotion_label='emphasis',
            project=project, **{k: v for k, v in kwargs.items() if k not in ('keywords',)}
        )
        node_id = result['id']

        self.conn.execute(
            "UPDATE nodes SET evolution_status = 'active' WHERE id = ?", (node_id,))
        self.conn.commit()

        # Connect to resulting decisions
        if resulting_decision_ids:
            for dec_id in resulting_decision_ids:
                self.connect(node_id, dec_id, 'caused', 0.85)

        result['evolution_status'] = 'active'
        return result

    def create_aspiration(self, title: str, content: str,
                          project: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Create an aspiration — a directional goal without a finish line.

        Slow decay (90 days), refreshed on every access. Acts as a compass —
        when making decisions, the brain checks active aspirations for relevance.

        Args:
            title: The vision (e.g. "Brain should detect stuck patterns and trigger reframing")
            content: Why this matters, what it looks like when achieved
            project: Optional project scope
        """
        result = self.remember(
            type='aspiration', title=f'🌱 ASPIRATION — {title}', content=content,
            keywords=kwargs.get('keywords', ''),
            emotion=0.5, emotion_label='excitement',
            project=project, **{k: v for k, v in kwargs.items() if k not in ('keywords',)}
        )
        node_id = result['id']

        self.conn.execute(
            "UPDATE nodes SET evolution_status = 'active' WHERE id = ?", (node_id,))
        self.conn.commit()

        result['evolution_status'] = 'active'
        return result

    def resolve_evolution(self, node_id: str, status: str,
                          resolved_by: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve an evolution node — change its status and optionally link to the resolution.

        Args:
            node_id: The evolution node to resolve
            status: New status — 'resolved' (tension), 'validated'/'disproven' (hypothesis),
                    'confirmed'/'dismissed' (pattern)
            resolved_by: Optional node_id of the decision/rule that resolves it

        Returns:
            Updated node info
        """
        valid_statuses = ('resolved', 'validated', 'disproven', 'confirmed', 'dismissed')
        if status not in valid_statuses:
            return {'error': f'Invalid status: {status}. Use: {valid_statuses}'}

        ts = self.now()
        self.conn.execute(
            'UPDATE nodes SET evolution_status = ?, resolved_at = ?, resolved_by = ?, updated_at = ? WHERE id = ?',
            (status, ts, resolved_by, ts, node_id)
        )

        # If resolved/validated, unlock so it can decay naturally
        # (resolved tensions and validated hypotheses become historical records)
        if status in ('resolved', 'validated', 'confirmed'):
            self.conn.execute('UPDATE nodes SET locked = 0 WHERE id = ?', (node_id,))

        # If disproven/dismissed, archive it (keep the lesson in content)
        if status in ('disproven', 'dismissed'):
            self.conn.execute('UPDATE nodes SET archived = 1 WHERE id = ?', (node_id,))

        self.conn.commit()

        # Connect to resolving node
        if resolved_by:
            self.connect(node_id, resolved_by, 'resolved_by', 0.85)

        cursor = self.conn.execute(
            'SELECT type, title, evolution_status, resolved_at, resolved_by, locked, archived FROM nodes WHERE id = ?',
            (node_id,))
        row = cursor.fetchone()
        if not row:
            return {'error': f'Node {node_id} not found'}

        return {
            'node_id': node_id, 'type': row[0], 'title': row[1],
            'evolution_status': row[2], 'resolved_at': row[3],
            'resolved_by': row[4], 'locked': row[5] == 1, 'archived': row[6] == 1,
        }

    def get_active_evolutions(self, types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get all active (unresolved) evolution nodes.

        Args:
            types: Filter by evolution type(s), e.g. ['tension', 'hypothesis']

        Returns:
            List of active evolution nodes
        """
        evolution_types = types or ['tension', 'hypothesis', 'pattern', 'catalyst', 'aspiration']
        placeholders = ','.join('?' * len(evolution_types))
        cursor = self.conn.execute(
            f"""SELECT id, type, title, content, confidence, evolution_status,
                       emotion, created_at, last_accessed
                FROM nodes
                WHERE type IN ({placeholders}) AND evolution_status = 'active'
                  AND archived = 0
                ORDER BY emotion DESC, created_at DESC""",
            evolution_types
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0], 'type': row[1], 'title': row[2],
                'content': row[3], 'confidence': row[4],
                'evolution_status': row[5], 'emotion': row[6],
                'created_at': row[7], 'last_accessed': row[8],
            })
        return results

    def confirm_evolution(self, node_id: str, feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Human confirms a conscious item. For hypotheses: bump confidence by 0.15.
        For patterns: promote from hypothesis to pattern if confidence >= 0.7.
        Records the feedback for consciousness adaptation.
        """
        cursor = self.conn.execute(
            'SELECT type, title, confidence, evolution_status FROM nodes WHERE id = ?', (node_id,))
        row = cursor.fetchone()
        if not row:
            return {'error': f'Node {node_id} not found'}

        ntype, title, confidence, status = row
        ts = self.now()
        result = {'node_id': node_id, 'type': ntype, 'title': title, 'action': 'confirmed'}

        if ntype == 'hypothesis':
            new_conf = min(1.0, (confidence or 0.3) + 0.15)
            self.conn.execute('UPDATE nodes SET confidence = ?, updated_at = ? WHERE id = ?',
                              (new_conf, ts, node_id))
            result['confidence'] = new_conf
            # Promote to decision if confidence >= 0.9
            if new_conf >= 0.9:
                self.resolve_evolution(node_id, 'validated')
                result['promoted'] = 'decision'
            # Promote hypothesis to pattern if >= 0.7
            elif new_conf >= 0.7 and ntype == 'hypothesis':
                self.conn.execute("UPDATE nodes SET type = 'pattern', title = REPLACE(title, '🔮 HYPOTHESIS', '📊 PATTERN'), updated_at = ? WHERE id = ?", (ts, node_id))
                result['promoted'] = 'pattern'

        if feedback:
            self.conn.execute('UPDATE nodes SET content = content || ? WHERE id = ?',
                              (f'\n\nHuman feedback: {feedback}', node_id))

        self.conn.commit()
        self.log_consciousness_response(ntype, True)
        return result

    def dismiss_evolution(self, node_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Human dismisses a conscious item. For hypotheses: drop confidence by 0.2.
        If confidence < 0.2: disprove and archive.
        """
        cursor = self.conn.execute(
            'SELECT type, title, confidence FROM nodes WHERE id = ?', (node_id,))
        row = cursor.fetchone()
        if not row:
            return {'error': f'Node {node_id} not found'}

        ntype, title, confidence = row
        ts = self.now()
        result = {'node_id': node_id, 'type': ntype, 'title': title, 'action': 'dismissed'}

        if ntype in ('hypothesis', 'pattern'):
            new_conf = max(0.0, (confidence or 0.5) - 0.2)
            self.conn.execute('UPDATE nodes SET confidence = ?, updated_at = ? WHERE id = ?',
                              (new_conf, ts, node_id))
            result['confidence'] = new_conf
            if new_conf < 0.2:
                status = 'disproven' if ntype == 'hypothesis' else 'dismissed'
                self.resolve_evolution(node_id, status)
                result['archived'] = True
        elif ntype == 'tension':
            # Mark as dismissed, not resolved
            self.resolve_evolution(node_id, 'dismissed')
            result['archived'] = True

        if reason:
            self.conn.execute('UPDATE nodes SET content = content || ? WHERE id = ?',
                              (f'\n\nDismissed: {reason}', node_id))

        self.conn.commit()
        self.log_consciousness_response(ntype, False)
        return result

    def get_relevant_aspirations(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """
        Aspiration compass: find active aspirations relevant to current conversation.
        Used during decision-making to check if any aspiration should influence the choice.
        """
        aspirations = self.get_active_evolutions(['aspiration'])
        if not aspirations or not embedder.is_ready():
            return aspirations[:limit]

        # Score each aspiration by embedding similarity to query
        query_vec = embedder.embed(query)
        if not query_vec:
            return aspirations[:limit]

        scored = []
        for asp in aspirations:
            asp_text = asp.get('title', '') + ' ' + (asp.get('content', '') or '')
            asp_vec = embedder.embed(asp_text)
            if asp_vec:
                sim = embedder.cosine_similarity(query_vec, asp_vec)
                if sim > 0.3:
                    asp['_relevance'] = sim
                    scored.append(asp)

        scored.sort(key=lambda x: -x.get('_relevance', 0))
        return scored[:limit]

    def check_hypothesis_relevance(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Hypothesis validation: check if any active hypothesis is relevant to current query.
        If relevant, surface it for conversational validation: "I had a hypothesis that X.
        Does this conversation confirm or deny it?"
        """
        hypotheses = self.get_active_evolutions(['hypothesis'])
        if not hypotheses or not embedder.is_ready():
            return None

        query_vec = embedder.embed(query)
        if not query_vec:
            return None

        for hyp in hypotheses:
            hyp_text = hyp.get('title', '') + ' ' + (hyp.get('content', '') or '')
            hyp_vec = embedder.embed(hyp_text)
            if hyp_vec:
                sim = embedder.cosine_similarity(query_vec, hyp_vec)
                if sim > 0.5:
                    hyp['_relevance'] = sim
                    return hyp

        return None

    def detect_catalyst(self, emotion: float, emotion_label: str,
                        context: str) -> Optional[Dict[str, Any]]:
        """
        Catalyst recognition: detect if current emotional signal + context
        represents an inflection point worth recording.
        Triggers when: emotion > 0.7 AND label is frustration/excitement/concern
        AND the topic connects to existing decisions.
        """
        if emotion < 0.7:
            return None
        if emotion_label not in ('frustration', 'excitement', 'concern', 'breakthrough'):
            return None

        # This is a candidate catalyst. Don't auto-create — flag for Claude to create
        # with proper context and resulting_decision_ids.
        return {
            'detected': True,
            'emotion': emotion,
            'emotion_label': emotion_label,
            'context': context,
            'instruction': 'High emotional signal detected. Consider creating a catalyst node if this moment changes direction.',
        }
