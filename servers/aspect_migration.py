"""Aspect migration — one-shot for existing brains + reusable seed for fresh brains.

Three operations:

1. seed_required_aspects(brain): create the 14 required aspect-nodes from
   aspects_v1.json if absent. Idempotent. Writes locked=True via
   encoding_source='anchor:seed_aspects' (constitution allows anchor-prefixed
   sources to lock). Used by:
     - Fresh brain bootstrap (Step 6 wires this in)
     - Boot-time auto-heal when a required aspect is missing
     - migrate_to_aspects orchestrator below

2. migrate_emergent_from_legacy(brain): read existing s2_node_families and
   s2_edge_families interactions, create aspect-nodes for any non-required
   names found. Locked=False (the constitution + the 'migration:*' prefix
   would force-clear locked anyway — emergent aspects shouldn't be locked
   automatically). Preserves legacy classifications. Used only by the
   orchestrator on existing brains.

3. migrate_to_aspects(brain): full orchestrator. seed required + migrate
   emergent. Idempotent (skips already-existing aspect-nodes by title).

Concept overlap during the merge: same name in both legacy interactions
(generally rare) → one aspect with both slots populated. One explicit
rename: legacy 'correction_supersession' (node-side) maps to canonical
'correction_improvement' (the unified name).
"""

import json
import os
from typing import Any, Dict, Tuple

from servers.aspects import REQUIRED_ASPECTS


SEED_PATH = os.path.join(os.path.dirname(__file__), 'scales', 's2', 'aspects_v1.json')


# Legacy node-side names that map onto canonical aspect names.
# Discovered during the unification: legacy node_families had
# `correction_supersession` for the same concept that legacy edge_families
# called `correction_improvement`. We canonicalize on the edge-side name
# because it's what code already references via REQUIRED_ASPECTS.
NAME_RENAMES = {
    'correction_supersession': 'correction_improvement',
}


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _load_seed() -> Dict[str, Dict[str, Any]]:
    """Load aspects_v1.json — single source of truth for required aspects."""
    with open(SEED_PATH) as f:
        return json.load(f)


def _existing_aspect_titles(brain) -> Dict[str, str]:
    """Return {title: node_id} for current type='aspect' nodes in the brain."""
    res = brain.filter_nodes(field='type', include=['aspect'], rich=False, limit=500)
    nodes = res.get('nodes', []) if isinstance(res, dict) else []
    return {n['title']: n['id'] for n in nodes if n.get('title') and n.get('id')}


def _build_remember_kwargs(name: str, spec: Dict[str, Any],
                           encoding_source: str, locked: bool) -> Dict[str, Any]:
    """Build the kwargs dict for brain.remember(type='aspect', ...).

    Member lists ride through as native lists — MetadataDAL.set_many
    JSON-encodes them as of Step 5a. AspectRegistry._load (Step 6)
    decode_value's them back to lists.
    """
    keywords_parts = ['aspect', name]
    keywords_parts.extend(spec.get('node_types', [])[:5])
    keywords_parts.extend(spec.get('edge_relations', [])[:5])

    kwargs = {
        'type': 'aspect',
        'title': name,
        'content': spec.get('meaning') or 'Aspect %s' % name,
        'situation': (
            'When determining what semantic role a node type or edge relation '
            'plays — Frame routing, S2 family-aware classification, healer '
            'display, MCP filter_nodes(field=\'aspect\').'
        ),
        'keywords': ' '.join(keywords_parts),
        'locked': locked,
        'encoding_source': encoding_source,
        # Member lists + dimension + display_label flow into node_metadata_kv
        # via remember()'s **extra_fields routing.
        'node_types': list(spec.get('node_types', [])),
        'edge_relations': list(spec.get('edge_relations', [])),
        'dimension': spec.get('dimension', 'semantic'),
    }

    # Optional metadata that's per-aspect (display_label for healer, etc.)
    metadata = spec.get('metadata') or {}
    if isinstance(metadata, dict) and 'display_label' in metadata:
        kwargs['display_label'] = metadata['display_label']

    return kwargs


def _members_meaning_from_legacy(spec: Any) -> Tuple[list, str]:
    """Extract (members_list, meaning_str) from a legacy family entry.

    Handles both legacy shapes:
      - list  (v1, e.g. node_families_v1 early form): just members
      - dict (v2, current edge_families, node_families_v1): {members, meaning}
    """
    if isinstance(spec, list):
        return list(spec), ''
    if isinstance(spec, dict):
        return list(spec.get('members') or []), str(spec.get('meaning') or '')
    return [], ''


# ─────────────────────────────────────────────────────────────────
# Operation 1: seed required aspects from JSON
# ─────────────────────────────────────────────────────────────────


def seed_required_aspects(brain) -> Dict[str, Any]:
    """Ensure all REQUIRED_ASPECTS exist as aspect-nodes in the brain.

    Idempotent: skips any required name whose aspect-node already exists
    (matched by title). Returns a summary.

    Locks via encoding_source='anchor:seed_aspects' — this is the operator's
    declared bootstrap, so the constitution permits the lock=True. Reseeded
    aspects get locked from day one because code references them by name.
    """
    seed = _load_seed()
    existing = _existing_aspect_titles(brain)

    created, skipped, errors = [], [], []

    for name in REQUIRED_ASPECTS:
        if name in existing:
            skipped.append(name)
            continue
        spec = seed.get(name)
        if not spec:
            errors.append('Required aspect %s missing from aspects_v1.json' % name)
            continue
        try:
            kwargs = _build_remember_kwargs(
                name, spec,
                encoding_source='anchor:seed_aspects',
                locked=True)
            brain.remember(**kwargs)
            created.append(name)
        except Exception as e:
            errors.append('%s: %s' % (name, e))

    return {
        'phase': 'seed_required',
        'created': created,
        'skipped': skipped,
        'errors': errors,
        'total_required': len(REQUIRED_ASPECTS),
    }


# ─────────────────────────────────────────────────────────────────
# Operation 2: migrate emergent from legacy interactions
# ─────────────────────────────────────────────────────────────────


def migrate_emergent_from_legacy(brain) -> Dict[str, Any]:
    """Create aspect-nodes for emergent (non-required) names from old interactions.

    Reads:
      - s2_node_families (the old node-classification interaction)
      - s2_edge_families (the old edge-classification interaction)

    For each name found that is NOT in REQUIRED_ASPECTS:
      - Create a single aspect-node, populated from whichever side(s) had it
      - If the same name appears in both, merge node_types + edge_relations
      - Apply NAME_RENAMES (e.g. correction_supersession → correction_improvement)
        — but the renamed target IS required, so it goes through seed_required
        not here. We still de-dup so we don't double-create on rename.

    Locked=False — emergent aspects are unlocked by the constitution
    (encoding_source='migration:aspects_emergent' doesn't start with 'anchor').
    """
    node_fams = brain.get_interaction_config('s2_node_families') or {}
    edge_fams = brain.get_interaction_config('s2_edge_families') or {}

    # Merge into unified shape keyed by canonical name
    unified: Dict[str, Dict[str, Any]] = {}

    def _add_node_members(canonical: str, members: list, meaning: str) -> None:
        if canonical not in unified:
            unified[canonical] = {'node_types': [], 'edge_relations': [],
                                  'meaning': '', 'metadata': {}}
        for m in members:
            if m not in unified[canonical]['node_types']:
                unified[canonical]['node_types'].append(m)
        if meaning and not unified[canonical]['meaning']:
            unified[canonical]['meaning'] = meaning

    def _add_edge_members(canonical: str, members: list, meaning: str) -> None:
        if canonical not in unified:
            unified[canonical] = {'node_types': [], 'edge_relations': [],
                                  'meaning': '', 'metadata': {}}
        for m in members:
            if m not in unified[canonical]['edge_relations']:
                unified[canonical]['edge_relations'].append(m)
        if meaning and not unified[canonical]['meaning']:
            unified[canonical]['meaning'] = meaning

    for name, spec in (node_fams or {}).items():
        if not isinstance(name, str) or name.startswith('__'):
            continue
        canonical = NAME_RENAMES.get(name, name)
        members, meaning = _members_meaning_from_legacy(spec)
        _add_node_members(canonical, members, meaning)

    for name, spec in (edge_fams or {}).items():
        if not isinstance(name, str) or name.startswith('__'):
            continue
        members, meaning = _members_meaning_from_legacy(spec)
        _add_edge_members(name, members, meaning)

    # Filter out required ones — they go through seed_required_aspects
    emergent_only = {n: spec for n, spec in unified.items()
                     if n not in REQUIRED_ASPECTS}

    existing = _existing_aspect_titles(brain)
    created, skipped, errors = [], [], []

    for name, spec in emergent_only.items():
        if name in existing:
            skipped.append(name)
            continue
        # Default meaning if legacy didn't have one
        if not spec.get('meaning'):
            spec['meaning'] = (
                'Emergent aspect migrated from legacy s2_*_families. '
                'AspectIntegration may refine member lists and meaning over time.'
            )
        spec['dimension'] = 'semantic'  # only dimension we have in v1
        try:
            kwargs = _build_remember_kwargs(
                name, spec,
                encoding_source='migration:aspects_emergent',
                locked=False)
            brain.remember(**kwargs)
            created.append(name)
        except Exception as e:
            errors.append('%s: %s' % (name, e))

    return {
        'phase': 'migrate_emergent',
        'created': created,
        'skipped': skipped,
        'errors': errors,
        'total_unified': len(unified),
        'emergent_count': len(emergent_only),
    }


# ─────────────────────────────────────────────────────────────────
# Operation 3: orchestrator
# ─────────────────────────────────────────────────────────────────


def migrate_to_aspects(brain) -> Dict[str, Any]:
    """Full migration — seed required + migrate emergent.

    Idempotent: re-running is safe (skips already-present aspect-nodes).
    Doesn't archive the legacy interactions yet — Step 12 cleans those up
    once all consumers (Steps 7-11) have moved off them.
    """
    seed_result = seed_required_aspects(brain)
    emergent_result = migrate_emergent_from_legacy(brain)

    return {
        'phase': 'complete',
        'required': seed_result,
        'emergent': emergent_result,
        'aspect_node_count': len(_existing_aspect_titles(brain)),
    }
