"""S2 Relation Reclassification — one-time migration of generic 'related' edges.

Reads edges with relation='related' or 'related_to' that have descriptions,
sends them to Sonnet in batches, updates the relation type based on the
description content.

This is an S2 operation: it reads S1's accumulated output (the graph) and
improves it. encoding_source='s2:relation_migration' marks everything it touches.
"""

import json
import time

from .base import IntegrationUnit


RECLASSIFY_PROMPT = """You are reclassifying edge relationships in a knowledge graph.

Each edge connects two nodes and has a description explaining the relationship.
The current relation type is generic ("related" or "related_to"). Your job: read the
description and assign a specific relation type that captures HOW these nodes relate.

Use whatever relation type fits accurately. Common patterns:
extends, corrects, depends_on, implements, contradicts, resolves, caused_by,
enables, validates, refines, challenges, supersedes, exemplifies, contextualizes,
produced, configures, part_of, constrains, addresses, elaborates, questions

These are examples, not a closed list. If "prerequisite_for" or "diagnosed_during"
is more accurate, use that. The relation should read naturally as:
  source [relation] target — e.g. "Boot layer" implements "recognition principle"

If the description is too vague to determine a specific type, keep "related".

Return ONLY a JSON array:
[{"id": 1, "relation": "extends"}, {"id": 2, "relation": "implements"}, ...]
"""

BATCH_SIZE = 50


class RelationReclassifier(IntegrationUnit):
    NAME = 'relation_reclassify'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:relation_migration'
    O_SOURCES = ['edge_relations']
    K_SOURCES = ['sonnet_classification']

    def __init__(self, brain, dispatch_fn=None):
        super().__init__(brain, dispatch_fn)

    def run(self):
        """Reclassify all generic relations that have descriptions.

        Returns: {total, reclassified, kept_related, batches, errors}
        """
        # 1. Gather candidates
        candidates = self._gather_candidates()
        if not candidates:
            return {'total': 0, 'reclassified': 0, 'details': 'no candidates'}

        self.trace('O', 'graph_structure',
                   '%d generic relations with descriptions to reclassify' % len(candidates))

        # 2. Process in batches
        total_reclassified = 0
        total_kept = 0
        total_errors = 0
        batch_count = 0

        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i:i + BATCH_SIZE]
            batch_count += 1

            results = self._reclassify_batch(batch)
            if results is None:
                total_errors += len(batch)
                continue

            # 3. Apply results
            for item in results:
                idx = item.get('id')
                new_rel = item.get('relation', 'related')

                if idx is None or idx < 0 or idx >= len(batch):
                    total_errors += 1
                    continue

                edge_id = batch[idx]['edge_id']
                old_rel = batch[idx]['old_relation']

                if new_rel and new_rel != old_rel and new_rel != 'related':
                    self.brain.conn.execute(
                        "UPDATE edge_relations SET relation = ?, encoding_source = ? "
                        "WHERE edge_id = ? AND relation = ?",
                        (new_rel, self.ENCODING_SOURCE, edge_id, old_rel))
                    total_reclassified += 1
                else:
                    total_kept += 1

            self.brain.conn.commit()
            print('[s2-reclassify] batch %d/%d: %d reclassified' % (
                batch_count, (len(candidates) + BATCH_SIZE - 1) // BATCH_SIZE,
                sum(1 for r in results if r.get('relation', 'related') != 'related')),
                flush=True)

        self.trace('delta', 'community_assignments',
                   'Reclassified %d/%d relations (%d kept, %d errors)' % (
                       total_reclassified, len(candidates), total_kept, total_errors))

        return {
            'total': len(candidates),
            'reclassified': total_reclassified,
            'kept_related': total_kept,
            'errors': total_errors,
            'batches': batch_count,
        }

    def _gather_candidates(self):
        """Find all generic relations with descriptions worth reclassifying."""
        rows = self.brain.conn.execute("""
            SELECT er.edge_id, er.relation, er.description,
                   e.source_id, e.target_id,
                   ns.title, ns.type, nt.title, nt.type
            FROM edge_relations er
            JOIN edges e ON e.edge_id = er.edge_id
            JOIN nodes ns ON ns.id = e.source_id
            JOIN nodes nt ON nt.id = e.target_id
            WHERE er.relation IN ('related', 'related_to')
            AND er.description IS NOT NULL AND er.description != ''
            AND LENGTH(er.description) > 5
            AND er.encoding_source != 's2:relation_migration'
        """).fetchall()

        return [{
            'edge_id': r[0], 'old_relation': r[1], 'description': r[2],
            'source_id': r[3], 'target_id': r[4],
            'source_title': r[5], 'source_type': r[6],
            'target_title': r[7], 'target_type': r[8],
        } for r in rows]

    def _reclassify_batch(self, batch):
        """Send a batch to Sonnet for reclassification. Returns list of {id, relation}."""
        import anthropic

        lines = []
        for i, c in enumerate(batch):
            lines.append('[%d] "%s" [%s] → "%s" [%s]' % (
                i, c['source_title'][:50], c['source_type'],
                c['target_title'][:50], c['target_type']))
            lines.append('    Description: %s' % c['description'][:150])

        user_content = 'EDGES TO RECLASSIFY:\n\n' + '\n'.join(lines)

        try:
            import os
            # Load API key — check .env if not in environment
            if not os.environ.get('ANTHROPIC_API_KEY'):
                from ..dispatch import load_env
                load_env()
            client = anthropic.Anthropic()
            response = client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=2048,
                system=RECLASSIFY_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )

            text = response.content[0].text.strip()
            # Extract JSON from response (might have markdown fences)
            if '```' in text:
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            print('[s2-reclassify] batch error: %s' % e, flush=True)
            self.brain._log_error('s2_reclassify', e, 'batch of %d edges' % len(batch))
            return None
