"""
brain — BrainReminders Mixin

Reminders + change impact lookup. Renamed from brain_engineering.py 2026-04-13
after removing record_divergence, synthesize_session, assess_session_health,
recalibrate_confidence, auto_generate_self_reflection, reflect_for_next_claude.
"""

from typing import Any, Dict, List, Optional
import json


class BrainRemindersMixin:
    """Reminders and change impact methods for Brain."""

    # REMOVED 2026-04-05: remember_purpose, remember_mechanism, remember_impact,
    # remember_constraint, remember_convention, remember_lesson, remember_mental_model,
    # remember_uncertainty, record_reasoning_trace, update_file_inventory,
    # get_file_inventory, detect_file_changes, update_system_purpose.
    # All were thin wrappers around remember(type='X'). Use remember() directly.

    # get_engineering_context removed 2026-04-13 — was already a stub returning {}.

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

        # Also check change_impacts in metadata KV
        cur = self.conn.execute(
            "SELECT kv.node_id, n.title, kv.value FROM node_metadata_kv kv "
            "JOIN nodes n ON n.id = kv.node_id "
            "WHERE kv.key = 'change_impacts' AND kv.value LIKE ?",
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

    # record_divergence, record_validation, track_session_event, get_correction_patterns
    # removed 2026-04-13 — dead code. correction_traces table also dropped.

    # assess_session_health, recalibrate_confidence, synthesize_session,
    # get_last_synthesis removed 2026-04-13 — direct DB writes bypassing revise(),
    # queried deprecated tables, 0 syntheses ever produced.

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

    # auto_generate_self_reflection, reflect_for_next_claude removed 2026-04-13
    # — created noise nodes (type='boot', type='capability') that nothing read.
