"""Aspect Registry — first-class roles/groupings for nodes and edges.

An aspect is a semantic role (correction, identity_bearing, active_thread, ...)
that groups node TYPES and/or edge RELATIONS under a shared meaning. Aspects
are themselves brain nodes (type='aspect'), with member lists in metadata.

Two tiers:
  - Required aspects (locked): code routes on these by string. Must always
    exist. Listed in REQUIRED_ASPECTS — single source of truth, mirrored by
    aspects_v1.json (test enforces equivalence). On boot the registry
    validates and auto-heals from seed if any required aspect is missing.
  - Emergent aspects (unlocked): discovered by S2's AspectIntegration unit
    as the brain accumulates new types/relations. Created freely.

Consumers (every name string in the codebase flows through here):
  - Frame: brain.aspects.identity_bearing.node_types (etc.)
  - S2 community/consolidation: brain.aspects.relations_in(['noise', ...])
  - Healer: brain.aspects.<name>.metadata.display_label
  - Surface: brain.aspects.relation_meaning_map() for embedding composition
  - MCP: list_aspects, filter_nodes(field='aspect'), recall(filter={'aspect':...})

Step 2 ships the typed shapes and from_dict() construction for tests/seeding.
The _load_from_brain() path (filter_nodes(type='aspect')) is wired in Step 6.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional


# ─────────────────────────────────────────────────────────────────
# Required aspects — code routes on these by string. Must always exist
# in the brain. Test asserts equivalence with aspects_v1.json keys.
# ─────────────────────────────────────────────────────────────────
REQUIRED_ASPECTS: tuple = (
    # Node-facing (used by Frame)
    'identity_bearing',     # principle / identity / vision / rule / operator types
    'episodic_anchor',      # moment / anchor_quote / user_quote / quote types
    'active_thread',        # open / tension / hypothesis / aspiration types
    'lesson_insight',       # lesson / insight / validation / reflection types

    # Edge-facing (used by S2 community / consolidation / healer)
    'generic_relation',         # related, related_to (skip set in community/consolidation)
    'noise',                    # co_accessed, emergent_bridge (skip set + structural)
    'correction_improvement',   # corrects, supersedes
    'extension_refinement',     # extends, refines, elaborates
    'explanation_causation',    # explains, causes
    'dependency_flow',          # depends_on, enables
    'contradiction_conflict',   # contradicts, challenges
    'validation_evidence',      # validates, demonstrates
    'hierarchical_structure',   # part_of, supersedes_structurally
    'temporal_sequence',        # follows_from, leads_to
)


class AspectContractError(Exception):
    """Raised when a required aspect is missing from the brain.

    The auto-heal path catches this on boot and reseeds from
    aspects_v1.json. Code that calls `registry.<aspect_name>` for a
    non-required (emergent) aspect should use registry.by_name() instead,
    which returns Optional[Aspect].
    """


# ─────────────────────────────────────────────────────────────────
# Aspect value object
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Aspect:
    """A single aspect — a named semantic role grouping types and/or relations.

    Frozen so callers can pass it around without worrying about mutation.
    Member tuples are tuples (not lists) for hashability + immutability.
    `metadata` is a dict by convention treated as read-only — don't mutate
    after construction. Whole-field reassignment is blocked by the frozen
    dataclass; in-place dict mutation is on the convention layer.
    """

    name: str
    node_types: tuple = ()
    edge_relations: tuple = ()
    meaning: str = ''
    dimension: str = 'semantic'
    locked: bool = False
    metadata: dict = field(default_factory=dict)

    def __contains__(self, item: str) -> bool:
        return item in self.node_types or item in self.edge_relations

    def is_node_only(self) -> bool:
        return bool(self.node_types) and not self.edge_relations

    def is_edge_only(self) -> bool:
        return bool(self.edge_relations) and not self.node_types

    def is_empty(self) -> bool:
        return not self.node_types and not self.edge_relations

    @property
    def member_count(self) -> int:
        return len(self.node_types) + len(self.edge_relations)


# ─────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────


class AspectRegistry:
    """First-class API for the aspects system.

    Production: instantiated once at Brain.__init__, exposed as brain.aspects.
    Loads from filter_nodes(type='aspect') eagerly; validates required aspects
    are present; auto-heals from aspects_v1.json if any are missing. The
    cache invalidates on a dirty flag set by the maintenance unit after writes.

    Testing/seeding: use from_dict() to construct without a brain — bypasses
    the load path, takes a pre-built dict of aspect specs.
    """

    def __init__(self, brain):
        self._brain = brain
        self._aspects: dict = {}
        self._reverse_node: dict = {}   # type_string → aspect_name
        self._reverse_edge: dict = {}   # relation_string → aspect_name
        self._dirty: bool = True
        self._load()
        self._validate()

    # ── Loading + invalidation ──

    def _load(self) -> None:
        """Load aspects from aspects_v1.json — single source of truth.

        Migrated 2026-05-08: was previously reading brain aspect-nodes
        (type='aspect') with member lists in metadata. Now reads JSON file
        directly. The brain aspect-nodes are legacy and slated for archive.

        Per-operator state (2026-05-17): the working file lives next to
        brain.db ($BRAIN_DB_DIR/aspects_v1.json), not in the repo. On
        first boot the repo seed is copied to the user dir; all subsequent
        encoder writes stay there. The repo file is the shipped baseline,
        never touched by runtime.

        Multi-membership: a string can appear in multiple aspects' member
        lists. Reverse maps (_reverse_node, _reverse_edge) store the FIRST
        aspect that claimed the string in JSON-iteration order — preserves
        the prior single-aspect API contract for `by_node_type`/`by_edge_relation`
        while letting the underlying data carry richer multi-aspect membership.
        """
        import json
        import os
        from servers.scales.s2.aspect_contract import (
            ASPECTS_JSON_PATH, ensure_aspects_user_copy)

        self._aspects = {}
        self._reverse_node = {}
        self._reverse_edge = {}

        # First-boot: seed user-dir copy from the repo baseline if missing
        try:
            ensure_aspects_user_copy()
        except Exception as e:
            try:
                self._brain._log_warning(
                    'aspect_registry_seed',
                    'failed to seed user-dir aspects_v1.json from repo baseline',
                    repr(e))
            except Exception:
                pass

        if not os.path.exists(ASPECTS_JSON_PATH):
            try:
                self._brain._log_warning(
                    'aspect_registry_load',
                    'aspects_v1.json missing — registry empty',
                    ASPECTS_JSON_PATH)
            except Exception:
                pass
            self._dirty = False
            return

        try:
            with open(ASPECTS_JSON_PATH, 'r') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            try:
                self._brain._log_warning(
                    'aspect_registry_load',
                    'failed to parse aspects_v1.json — registry empty',
                    repr(e))
            except Exception:
                pass
            self._dirty = False
            return

        for name, entry in data.items():
            if not isinstance(entry, dict):
                continue
            node_types = entry.get('node_types', []) or []
            edge_relations = entry.get('edge_relations', []) or []
            if not isinstance(node_types, list):
                node_types = []
            if not isinstance(edge_relations, list):
                edge_relations = []

            aspect = Aspect(
                name=name,
                node_types=tuple(node_types),
                edge_relations=tuple(edge_relations),
                meaning=entry.get('meaning', '') or '',
                dimension=entry.get('dimension', 'semantic') or 'semantic',
                locked=bool(entry.get('locked', False)),
                metadata=entry.get('metadata') or {},
            )
            self._aspects[name] = aspect
            for t in aspect.node_types:
                # First aspect to list a string wins for reverse lookup —
                # deterministic given JSON ordering.
                self._reverse_node.setdefault(t, name)
            for r in aspect.edge_relations:
                self._reverse_edge.setdefault(r, name)

        self._dirty = False

    def _validate(self) -> None:
        """Check REQUIRED_ASPECTS are present. Log loudly if any are missing.

        With JSON-source loading, "auto-heal" is no longer meaningful —
        the JSON file IS the spec. Missing required aspects in the JSON
        is a deployment error (or local edit mistake) that should surface
        clearly, not silently self-repair.
        """
        missing = [n for n in REQUIRED_ASPECTS if n not in self._aspects]
        if not missing:
            return
        try:
            self._brain._log_warning(
                'aspect_contract',
                'Required aspects missing from aspects_v1.json — registry will not be self-consistent',
                'missing=%s' % missing)
        except Exception:
            pass  # never let logging block validation

    def invalidate(self) -> None:
        """Mark cache stale. Next access reloads from brain.

        Called by AspectIntegration after writing new/revised aspect-nodes.
        """
        self._dirty = True

    def _refresh_if_dirty(self) -> None:
        if self._dirty:
            self._load()

    # ── Per-aspect access ──

    def __getattr__(self, name: str) -> Aspect:
        """brain.aspects.<aspect_name> → Aspect.

        Raises AspectContractError if the name isn't present. Required
        aspects always resolve (validated at boot). For emergent aspects
        whose existence isn't guaranteed, prefer by_name() which returns
        Optional[Aspect].

        Skip dunder-name lookups so Python internals (pickling, repr, etc.)
        don't trigger spurious AspectContractError raises.
        """
        if name.startswith('_'):
            raise AttributeError(name)
        # Bypass _refresh_if_dirty + _aspects access via getattr to avoid
        # recursion when the registry is partly constructed.
        aspects = self.__dict__.get('_aspects')
        if aspects is None:
            raise AttributeError(name)
        if self.__dict__.get('_dirty'):
            self._refresh_if_dirty()
            aspects = self._aspects
        if name in aspects:
            return aspects[name]
        raise AspectContractError(
            "No aspect named '%s'. Required aspects: %s. "
            "Use registry.by_name() for emergent aspects." % (name, REQUIRED_ASPECTS)
        )

    def by_name(self, name: str) -> Optional[Aspect]:
        """Return aspect by name, or None if not present.

        Use this for emergent aspects whose existence isn't guaranteed by
        the contract. Required aspects can use the attribute form safely.
        """
        self._refresh_if_dirty()
        return self._aspects.get(name)

    # ── Reverse lookups ──

    def by_node_type(self, t: str) -> Optional[Aspect]:
        """Find the aspect whose node_types contains this type, or None."""
        self._refresh_if_dirty()
        name = self._reverse_node.get(t)
        return self._aspects.get(name) if name else None

    def by_edge_relation(self, r: str) -> Optional[Aspect]:
        """Find the aspect whose edge_relations contains this relation, or None."""
        self._refresh_if_dirty()
        name = self._reverse_edge.get(r)
        return self._aspects.get(name) if name else None

    # ── Cross-aspect unions ──

    def types_in(self, names: Iterable[str]) -> tuple:
        """Union of node_types across the named aspects (insertion-ordered, deduped)."""
        self._refresh_if_dirty()
        seen = []
        for n in names:
            a = self._aspects.get(n)
            if not a:
                continue
            for t in a.node_types:
                if t not in seen:
                    seen.append(t)
        return tuple(seen)

    def relations_in(self, names: Iterable[str]) -> tuple:
        """Union of edge_relations across the named aspects (insertion-ordered, deduped)."""
        self._refresh_if_dirty()
        seen = []
        for n in names:
            a = self._aspects.get(n)
            if not a:
                continue
            for r in a.edge_relations:
                if r not in seen:
                    seen.append(r)
        return tuple(seen)

    # ── Discovery + enumeration ──

    def all(self) -> dict:
        """All aspects keyed by name. Returns a fresh dict — safe to iterate."""
        self._refresh_if_dirty()
        return dict(self._aspects)

    def all_with_counts(self) -> list:
        """Aspect summaries with member counts + previews — for list_aspects MCP.

        Each entry: {name, meaning, node_types_count, edge_relations_count,
        node_types_preview, edge_relations_preview, dimension, locked}.
        """
        self._refresh_if_dirty()
        out = []
        for name, a in self._aspects.items():
            out.append({
                'name': name,
                'meaning': a.meaning,
                'node_types_count': len(a.node_types),
                'edge_relations_count': len(a.edge_relations),
                'node_types_preview': list(a.node_types[:5]),
                'edge_relations_preview': list(a.edge_relations[:5]),
                'dimension': a.dimension,
                'locked': a.locked,
            })
        return out

    def required(self) -> dict:
        """Required aspects only (those named in REQUIRED_ASPECTS)."""
        self._refresh_if_dirty()
        return {n: a for n, a in self._aspects.items() if n in REQUIRED_ASPECTS}

    def emergent(self) -> dict:
        """Emergent (non-required) aspects only."""
        self._refresh_if_dirty()
        return {n: a for n, a in self._aspects.items() if n not in REQUIRED_ASPECTS}

    def by_dimension(self, dim: str) -> dict:
        """Aspects in one dimension (e.g., 'semantic', 'temporal' once present)."""
        self._refresh_if_dirty()
        return {n: a for n, a in self._aspects.items() if a.dimension == dim}

    def dimensions(self) -> set:
        """Set of all dimensions present in the brain right now."""
        self._refresh_if_dirty()
        return {a.dimension for a in self._aspects.values()}

    # ── Surface-specific (edge enrichment for embeddings) ──

    def relation_meaning_map(self) -> dict:
        """{relation_string: meaning_text} for edge enrichment in surface."""
        self._refresh_if_dirty()
        out = {}
        for a in self._aspects.values():
            for r in a.edge_relations:
                out[r] = a.meaning
        return out

    def compose_edge_text(self, relation: str, description: str) -> str:
        """Compose the canonical text embedded as an edge's semantic identity.

        Pattern: "[<relation>] <description> family: <meaning>"

        INTRINSIC to the edge — does NOT include partner node title. The
        partner's content lives in its own stored embedding; including it
        here would (a) couple the edge embedding to the partner's title
        (cascade-stale on node revise) and (b) double-count partner signal.

        Stable per `(relation, description, family meaning)` triple, so a
        single embedding can be stored on `edge_relations.embedding`
        (schema v26+) and reused across partner revisions.

        Used by:
          - `GraphDAL.add_relation` (write path) — compute + store at write
          - `surface_contract._compose_enriched_edge_text` (legacy read
            path, falls through to live compose when stored embedding NULL)
          - `scripts/backfill_edge_embeddings.py` (one-shot migration)
        """
        rel = (relation or '').strip()
        desc = (description or '').strip()
        family_aspect = self.by_edge_relation(rel) if rel else None
        meaning = (family_aspect.meaning or '') if family_aspect else ''

        parts = []
        if rel:
            parts.append('[%s]' % rel)
        if desc:
            parts.append(desc)
        if meaning:
            parts.append('family: ' + meaning)
        return ' '.join(parts)

    def type_meaning_map(self) -> dict:
        """{type_string: meaning_text} — symmetric to relation_meaning_map."""
        self._refresh_if_dirty()
        out = {}
        for a in self._aspects.values():
            for t in a.node_types:
                out[t] = a.meaning
        return out

    # ── Construction for tests / seeding ──

    @classmethod
    def from_dict(cls, brain, data: dict) -> 'AspectRegistry':
        """Construct directly from a dict — bypasses _load.

        For tests and seeding paths. The data shape mirrors aspects_v1.json:
            {name: {node_types: [...], edge_relations: [...],
                    meaning: '...', dimension: 'semantic',
                    locked: bool, metadata: {...}}}
        """
        instance = cls.__new__(cls)
        instance._brain = brain
        instance._aspects = {}
        instance._reverse_node = {}
        instance._reverse_edge = {}
        instance._dirty = False
        for name, spec in data.items():
            aspect = Aspect(
                name=name,
                node_types=tuple(spec.get('node_types', [])),
                edge_relations=tuple(spec.get('edge_relations', [])),
                meaning=spec.get('meaning', ''),
                dimension=spec.get('dimension', 'semantic'),
                locked=spec.get('locked', False),
                metadata=dict(spec.get('metadata', {})),
            )
            instance._aspects[name] = aspect
            for t in aspect.node_types:
                instance._reverse_node[t] = name
            for r in aspect.edge_relations:
                instance._reverse_edge[r] = name
        return instance
