"""S2 Edge Family Integration — classifies edge relation types into semantic families.

S1E writes edges with open-text relation types (224+ unique types). This unit
groups them into semantic families so S2 can reason over relational patterns
at a level above individual relation names. Primary consumers are other S2
units — community detection uses families as cluster dimensions; consolidation
uses families to reason about merge candidates.

S1 Surface consumes this module incidentally: `family_meaning` gets composed
into the enriched edge text that Surface embeds for query-time activation.
When the relation name is specific (`contextualizes`, `corrects`), the family
meaning adds little; when the relation name is generic (`related`, `related_to`),
the family meaning carries most of the semantic signal. Either way, Surface
does NOT route on family names — it just lets the composed text's embedding
geometry do the work.

Mapping stored in interactions table as 's2_edge_families' — versioned,
S3-editable. Any S2/S1 unit reads it via `_get_interaction_config('s2_edge_families')`.

Config shape (v2 — nested per family):
    {
      "family_name": {
        "members": ["relation1", "relation2", ...],
        "meaning": "1-2 sentence semantic description of the family"
      }
    }

Legacy shape (v1 — flat list per family) is still accepted on read:
    {"family_name": ["relation1", "relation2", ...]}

New writes always use v2. Use the `iter_families()` helper to read either
shape uniformly.

Runs when new unclassified relation types appear (detected from edge_relations).
"""

import json

from .base import IntegrationUnit


# ─────────────────────────────────────────────────────────────────
# Shape-handling helpers (public — called from consumers across S2/S1)
# ─────────────────────────────────────────────────────────────────

def iter_families(config):
    """Yield (family_name, members_list, meaning_str) for each family.

    Handles both legacy list shape and new nested-object shape:
      - list  → (name, list, '')
      - dict  → (name, dict.get('members', []), dict.get('meaning', ''))

    Silently skips malformed entries and top-level metadata keys
    (names starting with '__').
    """
    if not config:
        return
    for family, value in config.items():
        if not isinstance(family, str) or family.startswith('__'):
            continue
        if isinstance(value, list):
            yield family, list(value), ''
        elif isinstance(value, dict):
            members = value.get('members')
            meaning = value.get('meaning', '') or ''
            if isinstance(members, list):
                yield family, list(members), str(meaning)


def get_reverse_map(families_config):
    """Build relation → family lookup dict.

    Returns: {relation_type: family_name, ...}
    """
    reverse = {}
    for family, members, _meaning in iter_families(families_config):
        for m in members:
            reverse[m] = family
    return reverse


# ─────────────────────────────────────────────────────────────────
# Integration unit (runs periodically from coordinator)
# ─────────────────────────────────────────────────────────────────


class EdgeFamilyIntegration(IntegrationUnit):
    NAME = 'edge_family_integration'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:edge_families'

    O_SOURCES = ['edge_relations']
    K_SOURCES = ['llm_classification']

    def run(self):
        """Check for unclassified relation types, classify via LLM if needed.

        Also fills in missing `meaning` fields for existing families (one-shot
        migration + ongoing drift coverage).
        """

        # Load current mapping — may be legacy list-shape or new nested-dict shape
        current = self._get_interaction_config('s2_edge_families') or {}
        classified = set()
        for _fam, members, _mean in iter_families(current):
            classified.update(members)

        families_missing_meaning = [
            family for family, _members, meaning in iter_families(current)
            if not meaning
        ]

        # Find all relation types in the graph.
        # GraphDAL.count_by_relation defaults archived=0 (v25); for family
        # vocabulary we want only active relations — archived ones are
        # history, not the current classification target.
        from servers.dal import GraphDAL
        all_types = GraphDAL(self.brain.conn).count_by_relation()

        # Find unclassified (excluding noise we always skip upstream)
        noise = {'co_accessed', 'emergent_bridge', 'community_member'}
        unclassified = []
        for rel, cnt in all_types.items():
            if rel not in classified and rel not in noise:
                unclassified.append((rel, cnt))

        # Nothing to do: no new relations AND no missing meanings
        if not unclassified and not families_missing_meaning:
            return {'actions': 0, 'unclassified': 0,
                    'families': sum(1 for _ in iter_families(current))}

        self.trace('O', 'graph_structure',
                   '%d unclassified relation types (of %d total), '
                   '%d families missing meaning' % (
                       len(unclassified), len(all_types),
                       len(families_missing_meaning)),
                   metadata={'unclassified_count': len(unclassified),
                             'total_types': len(all_types),
                             'families_missing_meaning':
                                 len(families_missing_meaning)})

        # Format existing families for the prompt — members + meanings
        existing_text = ""
        if current:
            existing_text = "\n\nEXISTING FAMILIES (with current meanings):\n"
            for family, members, meaning in sorted(iter_families(current)):
                sample = ', '.join(sorted(members)[:10])
                meaning_note = (' — MEANING: %s' % meaning) if meaning \
                    else ' — MEANING: (MISSING — please provide)'
                existing_text += "  %s: %s%s\n" % (family, sample, meaning_note)

        new_types_text = (
            "\n".join("  %4d  %s" % (cnt, rel) for rel, cnt in unclassified)
            or "  (none — only meanings to fill)")

        prompt = """Classify edge relation types into semantic families and provide a meaning for each family.

This graph stores knowledge from an AI-human collaboration — decisions, lessons, corrections, mechanisms, rules, concepts. The relation types were written by an encoding agent (open text, no closed list). Each family groups relations that share a semantic role in the graph.

The `meaning` field will be embedded and used by the recall system to match queries to families — it should be specific enough to distinguish this family's relational pattern from others, and written in natural language a reader (or an embedding model) would recognize.

%sNEW UNCLASSIFIED TYPES:
%s

Rules:
- Assign each type to an EXISTING family if it fits, or create a NEW family.
- Provide a `meaning` for every family you mention in the output (1-2 sentences).
- When an existing family's current MEANING is marked MISSING, include a meaning for it too.
- Family names are lowercase_with_underscores, descriptive of the relational pattern.
- "related_to" and "related" go in "generic_relation".
- Noise types (co_accessed, emergent_bridge, dreamed_from, dream_observation) go in "noise".
- If a type's descriptions show inconsistent usage, put it in the family matching MAJORITY usage.
- Aim for 15-25 families overall — not 3 mega-groups, not 50 singletons.

Return ONLY JSON in this shape:
{
  "family_name": {
    "members": ["type1", "type2"],
    "meaning": "Short semantic description of what this family represents."
  }
}

Only include families you TOUCHED in the output (new families OR existing families you added members to OR existing families with missing meaning).""" % (
            existing_text, new_types_text)

        result = self._call_llm('s2_edge_families', prompt)

        if not result or not isinstance(result, dict):
            self.trace('delta', 'community_assignments',
                       'LLM classification failed')
            return {'actions': 0, 'error': 'classification failed',
                    'unclassified': len(unclassified)}

        # Merge into existing mapping — always write new shape.
        # Start by normalizing `current` to new shape.
        updated = {}
        for family, members, meaning in iter_families(current):
            updated[family] = {'members': list(members), 'meaning': meaning}

        new_assignments = 0
        meaning_updates = 0
        for family, value in result.items():
            if not isinstance(family, str) or family.startswith('__'):
                continue
            # LLM may output either shape — be tolerant
            if isinstance(value, list):
                new_members = list(value)
                new_meaning = None
            elif isinstance(value, dict):
                new_members = value.get('members', []) or []
                new_meaning = value.get('meaning')
            else:
                continue

            if family not in updated:
                updated[family] = {'members': [], 'meaning': ''}

            # Append new members (dedup within family)
            existing_members = set(updated[family]['members'])
            for m in new_members:
                if m not in classified and m not in existing_members:
                    updated[family]['members'].append(m)
                    existing_members.add(m)
                    new_assignments += 1

            # Update meaning if provided and current is empty/different
            if new_meaning and not updated[family].get('meaning'):
                updated[family]['meaning'] = str(new_meaning)
                meaning_updates += 1
            elif new_meaning and updated[family].get('meaning') != new_meaning:
                # LLM provided refinement — accept it (trust the classifier)
                updated[family]['meaning'] = str(new_meaning)
                meaning_updates += 1

        # Store updated mapping — preserve existing template
        existing = self.brain._interaction_dal.get_latest('s2_edge_families')
        existing_template = existing['template'] if existing else ''
        self.brain._interaction_dal.register(
            's2_edge_families',
            template=existing_template,
            parameters=json.dumps(updated, indent=2),
            created_by=self.ENCODING_SOURCE)

        self.trace('delta', 'community_assignments',
                   'Classified %d new relations; updated %d meanings across %d families' % (
                       new_assignments, meaning_updates, len(updated)),
                   metadata={'new_assignments': new_assignments,
                             'meaning_updates': meaning_updates,
                             'total_families': len(updated)})

        return {
            'actions': new_assignments + meaning_updates,
            'families': len(updated),
            'unclassified': len(unclassified),
            'classified': new_assignments,
            'meaning_updates': meaning_updates,
        }
