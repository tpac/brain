"""S2 Edge Family Integration — classifies edge relation types into semantic families.

S1E writes edges with open-text relation types (224+ unique types).
This unit groups them into semantic families so other S2 units (community
detection, dedup, etc.) can reason about relational patterns.

Mapping stored in interactions table as 's2_edge_families' — versioned,
S3-editable. Any S2 unit reads it via _get_interaction_config('s2_edge_families').

Runs when new unclassified relation types appear (detected from edge_relations).
"""

import json
from collections import Counter

from .base import IntegrationUnit


class EdgeFamilyIntegration(IntegrationUnit):
    NAME = 'edge_family_integration'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:edge_families'

    O_SOURCES = ['edge_relations']
    K_SOURCES = ['llm_classification']

    def run(self):
        """Check for unclassified relation types, classify via LLM if needed."""

        # Load current mapping
        current = self._get_interaction_config('s2_edge_families') or {}
        classified = set()
        for members in current.values():
            if isinstance(members, list):
                classified.update(members)

        # Find all relation types in the graph
        rows = self.brain.conn.execute("""
            SELECT relation, COUNT(*) as cnt FROM edge_relations
            GROUP BY relation ORDER BY cnt DESC
        """).fetchall()
        all_types = {r[0]: r[1] for r in rows}

        # Find unclassified (excluding noise we always skip)
        noise = {'co_accessed', 'emergent_bridge', 'community_member'}
        unclassified = []
        for rel, cnt in all_types.items():
            if rel not in classified and rel not in noise:
                unclassified.append((rel, cnt))

        if not unclassified:
            return {'actions': 0, 'unclassified': 0,
                    'families': len(current)}

        self.trace('O', 'graph_structure',
                   '%d unclassified relation types (of %d total)' % (
                       len(unclassified), len(all_types)),
                   metadata={'unclassified_count': len(unclassified),
                             'total_types': len(all_types)})

        # Format for LLM
        existing_text = ""
        if current:
            existing_text = "\n\nEXISTING FAMILIES:\n"
            for family, members in sorted(current.items()):
                existing_text += "  %s: %s\n" % (
                    family, ', '.join(sorted(members)[:10]))

        new_types_text = "\n".join(
            "  %4d  %s" % (cnt, rel) for rel, cnt in unclassified)

        prompt = """Classify these edge relation types into semantic families.
Each family represents a KIND of relationship pattern in a knowledge graph.

%sNEW UNCLASSIFIED TYPES:
%s

Rules:
- Assign each type to an EXISTING family if it fits, or create a NEW family
- Family names are lowercase_with_underscores, descriptive of the relational pattern
- "related_to" and "related" go in "generic_relation"
- Return ONLY JSON: {"family_name": ["type1", "type2"], ...}
- Only include the NEW types in your response, not the existing ones""" % (
            existing_text, new_types_text)

        result = self._call_llm('s2_edge_families', prompt)

        if not result or not isinstance(result, dict):
            self.trace('delta', 'community_assignments',
                       'LLM classification failed')
            return {'actions': 0, 'error': 'classification failed',
                    'unclassified': len(unclassified)}

        # Merge into existing mapping
        updated = dict(current)
        new_assignments = 0
        for family, members in result.items():
            if not isinstance(members, list):
                continue
            if family not in updated:
                updated[family] = []
            for m in members:
                if m not in classified:
                    updated[family].append(m)
                    new_assignments += 1

        # Store updated mapping as new interaction version
        self.brain._interaction_dal.register(
            's2_edge_families',
            template='',
            parameters=json.dumps(updated, indent=2),
            created_by=self.ENCODING_SOURCE)

        self.trace('delta', 'community_assignments',
                   'Classified %d new relation types into %d families' % (
                       new_assignments, len(updated)),
                   metadata={'new_assignments': new_assignments,
                             'total_families': len(updated)})

        return {
            'actions': new_assignments,
            'families': len(updated),
            'unclassified': len(unclassified),
            'classified': new_assignments,
        }

    @staticmethod
    def get_family(relation, families_config):
        """Look up which family a relation type belongs to.

        Args:
            relation: Edge relation type string
            families_config: Dict from _get_interaction_config('s2_edge_families')

        Returns:
            Family name string, or 'unclassified' if not found.
        """
        for family, members in families_config.items():
            if isinstance(members, list) and relation in members:
                return family
        return 'unclassified'

    @staticmethod
    def get_reverse_map(families_config):
        """Build relation → family lookup dict.

        Returns: {relation_type: family_name, ...}
        """
        reverse = {}
        for family, members in families_config.items():
            if isinstance(members, list):
                for m in members:
                    reverse[m] = family
        return reverse
