"""
brain — BrainEvolution Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.

Proto-S2 methods (auto_heal, auto_tune, auto_discover_evolutions, evolution lifecycle)
removed 2026-04-13 — S2 integration units replace them. See git history for reference.
"""

from . import embedder
from typing import Any, Dict, List, Optional


class BrainEvolutionMixin:
    """Evolution methods for Brain."""

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
        # Find nodes with auto-captured quotes via KV store
        rows = self.conn.execute(
            '''SELECT kv_q.node_id, kv_q.value as quote, kv_c.value as source_ctx,
                      n.title, n.content, ne.embedding
               FROM node_metadata_kv kv_q
               JOIN nodes n ON kv_q.node_id = n.id
               LEFT JOIN node_embeddings ne ON n.id = ne.node_id
               LEFT JOIN node_metadata_kv kv_c ON kv_q.node_id = kv_c.node_id AND kv_c.key = 'source_context'
               WHERE kv_q.key = 'user_raw_quote'
                 AND kv_c.value LIKE 'Auto-captured%%'
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
                from .dal_metadata import MetadataDAL
                _mdal = MetadataDAL(self.conn)
                _mdal.delete(node_id, 'user_raw_quote')
                _mdal.set(node_id, 'source_context',
                          'Pruned auto-quote (sim=%.2f, below %.2f): "%s"' % (
                              sim, threshold, quote[:100]))
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
