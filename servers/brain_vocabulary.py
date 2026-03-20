"""
brain — BrainVocabulary Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from typing import Any, Dict, List, Optional
import json
import re


class BrainVocabularyMixin:
    """Vocabulary methods for Brain."""

    def learn_vocabulary(self, term: str, maps_to: List[str],
                         context: Optional[str] = None,
                         project: Optional[str] = None) -> Dict[str, Any]:
        """How the operator refers to things → code mapping.

        The same term can map to different things in different contexts.
        Example: term="the hook", maps_to=["pre-response-recall.sh"], context="recall/memory"
                 term="the hook", maps_to=["pre-edit-suggest.sh"], context="editing files"

        Example: term="the recall hook"
                 maps_to=["pre-response-recall.sh", "recall_with_embeddings()"]
        """
        maps_str = ', '.join(maps_to)
        content = '"%s" \u2192 %s' % (term, maps_str)
        if context:
            content += '\nContext: %s' % context
        title = '[vocab] %s' % term
        if context:
            title += ' (%s)' % context
        result = self.remember_rich(
            type='vocabulary', title=title, content=content,
            source_attribution='user_stated',
            project=project, locked=True)
        # Connect vocabulary node to existing nodes that match maps_to targets
        vocab_id = result.get('id')
        if vocab_id:
            self._connect_vocabulary(vocab_id, maps_to, term)
        # Clear this term from vocabulary_gaps if present
        self._clear_vocabulary_gap(term)
        return result

    def _clear_vocabulary_gap(self, term: str):
        """Remove a learned term from the vocabulary_gaps store."""
        try:
            import json as _json
            gaps_json = self.get_config('vocabulary_gaps', '[]')
            gaps = _json.loads(gaps_json) if gaps_json else []
            gaps = [g for g in gaps if (g.get('term') if isinstance(g, dict) else g) != term.lower()]
            self.set_config('vocabulary_gaps', _json.dumps(gaps))
        except Exception as _e:
            self._log_error("_clear_vocabulary_gap", _e, "removing term from vocabulary_gaps config")

    def _connect_vocabulary(self, vocab_id: str, maps_to: List[str], term: str):
        """Connect a vocabulary node to existing nodes matching its maps_to targets.

        Searches for nodes whose title contains any of the maps_to strings
        (file names, function names, concepts) and creates edges.
        Also connects to nodes whose title/keywords match the term itself.
        """
        try:
            connected = set()
            for target in maps_to:
                # Clean target — strip parens from function names like "recall_with_embeddings()"
                clean = target.strip().rstrip('()')
                if not clean or len(clean) < 3:
                    continue
                # Find nodes matching the target (file nodes, purpose nodes, mechanism nodes, etc.)
                rows = self.conn.execute(
                    """SELECT id, type FROM nodes
                       WHERE archived = 0 AND id != ?
                         AND (title LIKE ? OR title LIKE ?)
                       LIMIT 5""",
                    (vocab_id, f'%{clean}%', f'%{target}%')
                ).fetchall()
                for row in rows:
                    nid, ntype = row[0], row[1]
                    if nid not in connected:
                        self.connect(vocab_id, nid, 'maps_to', weight=0.8)
                        connected.add(nid)
            # Also connect to nodes matching the term itself
            if term and len(term) >= 3:
                term_rows = self.conn.execute(
                    """SELECT id FROM nodes
                       WHERE archived = 0 AND id != ? AND type != 'vocabulary'
                         AND (title LIKE ? OR keywords LIKE ?)
                       LIMIT 3""",
                    (vocab_id, f'%{term}%', f'%{term}%')
                ).fetchall()
                for row in term_rows:
                    if row[0] not in connected:
                        self.connect(vocab_id, row[0], 'refers_to', weight=0.6)
                        connected.add(row[0])
        except Exception as _e:
            self._log_error("_connect_vocabulary", _e, "connecting vocabulary node to related nodes via edges")

    def resolve_vocabulary(self, term: str) -> Optional[Dict[str, Any]]:
        """Look up what an operator term maps to.

        Returns all matches if the term has context-dependent mappings.
        Single match → returns dict. Multiple matches → returns dict with 'mappings' list.
        If ambiguous, the caller should ask the user which context applies.
        """
        rows = self.conn.execute(
            "SELECT id, title, content FROM nodes WHERE type = 'vocabulary' AND archived = 0 AND title LIKE ?",
            (f'%{term}%',)
        ).fetchall()
        if not rows:
            return None
        if len(rows) == 1:
            return {'id': rows[0][0], 'title': rows[0][1], 'content': rows[0][2]}
        # Multiple mappings — same term, different contexts
        return {
            'term': term,
            'ambiguous': True,
            'mappings': [
                {'id': r[0], 'title': r[1], 'content': r[2]}
                for r in rows
            ]
        }
