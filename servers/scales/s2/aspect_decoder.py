"""S2 Aspect Decoder — finds unclassified node types and edge relations.

Scans the brain for distinct types/relations not yet present in any
aspect's member list. Filters by count threshold (singletons skipped).
Sorts by count DESC and takes the top N as candidates. For each
candidate, loads N example records (nodes for types, edges for relations)
to give the encoder real evidence of usage.

The encoder reads the aspects_v1.json menu directly — decoder only
produces the candidate list + examples. No suppression, no clustering,
no batching across cycles — closed-list classification means every
string gets a home and re-running is cheap.
"""

import json
import os

from .base import IntegrationUnit
from .aspect_contract import ASPECT, ASPECTS_JSON_PATH


class AspectDecoder(IntegrationUnit):
    NAME = 'aspect_integration'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:aspect_integration'

    O_SOURCES = ['nodes.type', 'edge_relations.relation', 'aspects_v1.json']
    K_SOURCES = ['count_threshold', 'examples_per_candidate']

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or ASPECT

    def run(self):
        """Find unclassified strings and build proposals.

        Returns: {proposals: [...], stats: {...}, skipped: str|None}
        """
        classified = self._load_classified_strings()
        type_rows, relation_rows = self._scan_distinct()

        unclassified_types = [
            r for r in type_rows
            if r['value'] not in classified['node_types']
            and r['count'] >= self.config['min_count_threshold']
        ]
        unclassified_relations = [
            r for r in relation_rows
            if r['value'] not in classified['edge_relations']
            and r['count'] >= self.config['min_count_threshold']
        ]

        stats = {
            'distinct_types': len(type_rows),
            'distinct_relations': len(relation_rows),
            'classified_types': len(classified['node_types']),
            'classified_relations': len(classified['edge_relations']),
            'unclassified_types': len(unclassified_types),
            'unclassified_relations': len(unclassified_relations),
        }

        # Merge + sort by count DESC, take top N. Sorting both categories
        # together lets the encoder see the highest-evidence candidates
        # first regardless of category.
        all_unclassified = (
            [{'category': 'node_types', **r} for r in unclassified_types]
            + [{'category': 'edge_relations', **r} for r in unclassified_relations]
        )
        all_unclassified.sort(key=lambda r: r['count'], reverse=True)

        batch_size = self.config['max_candidates_per_call']
        batch = all_unclassified[:batch_size]

        self.trace('O', 'aspect_scan',
                   '%d unclassified types + %d unclassified relations → batch of %d' % (
                       stats['unclassified_types'],
                       stats['unclassified_relations'],
                       len(batch)),
                   metadata=stats)

        if not batch:
            return {'proposals': [], 'stats': stats, 'skipped': 'nothing unclassified'}

        # Attach example records to each candidate
        proposals = self._build_proposals(batch)

        self.trace('K', 'aspect_proposals',
                   '%d proposals built with %d examples each' % (
                       len(proposals), self.config['examples_per_candidate']),
                   metadata={
                       'proposal_count': len(proposals),
                       'examples_per_proposal': self.config['examples_per_candidate'],
                       'remaining_unclassified': max(0, len(all_unclassified) - len(batch)),
                   })

        return {
            'proposals': proposals,
            'stats': stats,
            'remaining': len(all_unclassified) - len(batch),
        }

    # ─── helpers ─────────────────────────────────────────────────────

    def _load_classified_strings(self):
        """Read aspects_v1.json and return {node_types: set, edge_relations: set}.

        Strings that already appear in any aspect's member list are
        considered classified and won't be re-proposed.
        """
        try:
            with open(ASPECTS_JSON_PATH, 'r') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.brain._log_error(
                self.NAME, e,
                'failed to read %s — treating all strings as unclassified' % ASPECTS_JSON_PATH)
            return {'node_types': set(), 'edge_relations': set()}

        node_types = set()
        edge_relations = set()
        for aspect_def in data.values():
            for t in aspect_def.get('node_types', []):
                node_types.add(t)
            for r in aspect_def.get('edge_relations', []):
                edge_relations.add(r)
        return {'node_types': node_types, 'edge_relations': edge_relations}

    def _scan_distinct(self):
        """Return (type_rows, relation_rows) sorted by count DESC.

        Each row: {value: str, count: int}. Archived rows excluded.
        """
        type_rows = self.brain.conn.execute("""
            SELECT type, COUNT(*)
            FROM nodes
            WHERE archived = 0 AND type IS NOT NULL AND type != ''
            GROUP BY type
            ORDER BY COUNT(*) DESC
        """).fetchall()

        relation_rows = self.brain.conn.execute("""
            SELECT relation, COUNT(*)
            FROM edge_relations
            WHERE archived = 0 AND relation IS NOT NULL AND relation != ''
            GROUP BY relation
            ORDER BY COUNT(*) DESC
        """).fetchall()

        return (
            [{'value': r[0], 'count': r[1]} for r in type_rows],
            [{'value': r[0], 'count': r[1]} for r in relation_rows],
        )

    def _build_proposals(self, batch):
        """Attach example records to each candidate.

        For node_types: 3 nodes that have that type — title + content snippet.
        For edge_relations: 3 edges with that relation — source/target titles
            + edge description.

        Examples are picked deterministically (highest confidence first)
        for reproducibility across cycles.
        """
        n_examples = self.config['examples_per_candidate']
        proposals = []
        for c in batch:
            if c['category'] == 'node_types':
                examples = self._sample_node_examples(c['value'], n_examples)
            else:
                examples = self._sample_relation_examples(c['value'], n_examples)
            proposals.append({
                'category': c['category'],
                'value': c['value'],
                'count': c['count'],
                'examples': examples,
            })
        return proposals

    def _sample_node_examples(self, type_value, n):
        """Pick examples spanning utility range — strong/typical/edge.

        Ordering: access_count DESC (how often the brain reaches for it)
        then length(content) DESC (richer text). Confidence dropped as
        ordering signal — it tracks content truthness, not how
        representative an example is of its type.

        Loads situation alongside content because situation often signals
        the type's role more clearly than content (e.g., "When designing
        the recall pipeline" → architecture/design type).
        """
        rows = self.brain.conn.execute("""
            SELECT n.id, n.title, n.content, n.confidence, n.type,
                   (SELECT value FROM node_metadata_kv
                    WHERE node_id = n.id AND key = 'situation' LIMIT 1)
            FROM nodes n
            WHERE n.archived = 0 AND n.type = ?
            ORDER BY n.access_count DESC, LENGTH(n.content) DESC
        """, (type_value,)).fetchall()
        # Indices: 0=id, 1=title, 2=content, 3=confidence, 4=type, 5=situation
        picked = self._pick_diverse(rows, n)
        return [
            {
                'tier': tier,
                'id': (r[0] or '')[:8],
                'type': r[4] or '',
                'title': r[1] or '',
                'content_snippet': (r[2] or '')[:400],
                'situation': (r[5] or '')[:300],
                'confidence': r[3],
            }
            for tier, r in picked
        ]

    def _sample_relation_examples(self, relation_value, n):
        """Pick relation examples spanning weight range — strong/typical/edge.

        Ordering: weight DESC (Hebbian utility) then length(description)
        DESC (richer text). Includes src + tgt content snippets so the
        encoder can reason about what the relation actually expresses,
        not just what its name suggests.
        """
        rows = self.brain.conn.execute("""
            SELECT er.description, er.weight,
                   src.title, src.type, src.content,
                   tgt.title, tgt.type, tgt.content
            FROM edge_relations er
            JOIN edges e ON er.edge_id = e.edge_id
            JOIN nodes src ON e.source_id = src.id
            JOIN nodes tgt ON e.target_id = tgt.id
            WHERE er.archived = 0 AND er.relation = ?
              AND src.archived = 0 AND tgt.archived = 0
            ORDER BY er.weight DESC, LENGTH(er.description) DESC
        """, (relation_value,)).fetchall()
        # Indices: 0=desc, 1=weight, 2=src_title, 3=src_type, 4=src_content,
        #          5=tgt_title, 6=tgt_type, 7=tgt_content
        picked = self._pick_diverse(rows, n)
        return [
            {
                'tier': tier,
                'src_title': r[2] or '',
                'src_type': r[3] or '',
                'src_content_snippet': (r[4] or '')[:150],
                'tgt_title': r[5] or '',
                'tgt_type': r[6] or '',
                'tgt_content_snippet': (r[7] or '')[:150],
                'description': (r[0] or '')[:300],
                'weight': r[1],
            }
            for tier, r in picked
        ]

    @staticmethod
    def _pick_diverse(rows, n):
        """Pick n rows spanning the input list's range, with tier labels.

        Input rows are pre-sorted (highest score first). For n=3, picks:
        - rows[0]            → 'strong'
        - rows[len/2]        → 'typical'
        - rows[-1]           → 'edge'

        For shorter inputs, returns all available with appropriate tiers.
        """
        if not rows:
            return []
        if len(rows) <= n:
            tiers = ['strong', 'typical', 'edge'][:len(rows)]
            return list(zip(tiers, rows))
        if n == 3:
            return [
                ('strong', rows[0]),
                ('typical', rows[len(rows) // 2]),
                ('edge', rows[-1]),
            ]
        # Generic n: evenly spaced indices including endpoints
        step = (len(rows) - 1) / (n - 1)
        indices = [int(round(i * step)) for i in range(n)]
        tiers = ['strong'] + ['typical'] * (n - 2) + ['edge']
        return list(zip(tiers, [rows[i] for i in indices]))
