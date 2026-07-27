"""Aspect Registry — first-class roles/groupings for nodes and edges.

An aspect is a semantic role (correction, identity_bearing, active_thread, ...)
that groups node TYPES and/or edge RELATIONS under a shared meaning. Aspects
live in aspects_v1.json — each entry carries its own member lists (the
node_types / edge_relations it claims). The registry reads that file directly;
aspects are NOT brain nodes.

Two tiers:
  - Required aspects (locked): code routes on these by string. Must always
    exist. Listed in REQUIRED_ASPECTS — single source of truth for the
    required NAMES, mirrored by aspects_v1.json keys (test enforces
    equivalence). On boot the registry validates they are present in the JSON
    and logs loudly if any are missing (no auto-heal — the JSON is the spec).
  - Emergent aspects (unlocked): discovered by S2's AspectIntegration unit
    as the brain accumulates new types/relations. Created freely.

Consumers (every name string in the codebase flows through here):
  - Frame: brain.aspects.identity_bearing.node_types (etc.)
  - S2 community/consolidation: brain.aspects.relations_in(['noise', ...])
  - Healer: brain.aspects.<name>.metadata.display_label
  - Surface: brain.aspects.relation_meaning_map() for embedding composition
  - MCP: list_aspects, filter_nodes(field='aspect'), recall(filter={'aspect':...})

from_dict() constructs a registry without a brain (for tests/seeding); the
production path is _load(), which reads aspects_v1.json directly.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional

# Required-name contract lives in the byte-level core; re-exported here so
# consumers keep one import surface (servers.aspects) for taxonomy names.
from .aspect_store import (
    REQUIRED_ASPECTS,  # noqa: F401 — re-export
    SEED_ASPECTS_JSON_PATH,
    aspects_json_path,
    atomic_json_write,
)


# ─────────────────────────────────────────────────────────────────
# Edge-aspect prompt block — single source for every encoder that offers the
# edge-relation vocabulary in its prompt (S2 consolidation + community via
# IntegrationUnit._inject_edge_aspects; S1E/surface may adopt). Mirrors the
# journal block's render-fn-+-base-method split.
# ─────────────────────────────────────────────────────────────────
EDGE_ASPECT_PROMPT_SKIP = ('generic_relation', 'noise', 'survivor_lineage')


def render_edge_aspects_block(aspects):
    """Render the edge-relation aspect vocabulary as a prompt block so an encoder
    picks specific relations over generic ones. `aspects` is the name→Aspect dict
    from brain.aspects.all(). Skips structural/system aspects (EDGE_ASPECT_PROMPT_SKIP)
    and node-only aspects (no edge relations); each shown aspect lists its first 8
    relations. Returns '' when there's nothing to show.
    """
    lines = []
    for name, aspect in sorted(aspects.items()):
        if name in EDGE_ASPECT_PROMPT_SKIP or not aspect.edge_relations:
            continue
        lines.append('- **%s**: %s' % (
            name, ', '.join(list(aspect.edge_relations[:8]))))
    if not lines:
        return ''
    return ('## Edge Aspects (%d from brain.aspects)\n\n%s\n\n'
            'Avoid `related_to` — pick a specific relation.' % (
                len(lines), '\n'.join(lines)))


def reconcile_working_copy(log_fn=None) -> bool:
    """Seed the working aspects file from the repo seed, and SELF-HEAL.

    Three jobs, all idempotent and safe to call on every boot:
      1. First boot — working copy missing → copy the whole seed (atomic).
      2. Missing aspect — the seed has a REQUIRED aspect the working copy
         lacks → add the whole aspect from the seed.
      3. Missing member — a REQUIRED aspect's seed `node_types` /
         `edge_relations` list names a string the working copy's list lacks
         → APPEND it. This is how a curated membership fix (e.g. multi-homing
         a replacement verb into correction_improvement, which recall walks)
         reaches an existing brain. Without it a seed member edit propagates
         to fresh installs only, and every existing brain keeps the defect
         while the seed-based contract tests pass.

    An UNPARSEABLE working copy is quarantined (renamed aside, preserving the
    operator's classifier-grown members for manual recovery) and re-seeded —
    the old return-False path left the registry permanently empty on every
    subsequent boot, which its own comments called catastrophic.

    ADDITIVE ONLY, in both directions of scope:
      · Members are appended, never reordered or removed — operator- and
        AspectIntegration-grown lists survive, and append-at-end cannot
        evict anything from the first-8 window `render_edge_aspects_block`
        shows the encoders.
      · A seed REMOVAL does not propagate. Retiring a member (or moving one
        between aspects) still needs a supervised migration; this heals
        omissions, not disagreements.
      · Only REQUIRED aspects are touched. Emergent/unlocked aspects are the
        classifier's to own.

    Never writes the seed itself (tests may point the working path at it).
    Returns True if the file was created or modified. `log_fn(message)` — when
    given — is called with a one-line summary of what was healed: a member heal
    silently changes which edges `correction_enrich` walks, so it announces
    itself rather than being inferred from behaviour later.
    """
    json_path = aspects_json_path()
    if not os.path.exists(SEED_ASPECTS_JSON_PATH):
        return False
    # Never heal the seed into itself.
    if os.path.abspath(json_path) == os.path.abspath(SEED_ASPECTS_JSON_PATH):
        return False
    try:
        with open(SEED_ASPECTS_JSON_PATH) as f:
            seed = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        # Broken plugin install — nothing trustworthy to reconcile from.
        if log_fn:
            try:
                log_fn('repo seed unreadable — reconcile skipped: %r' % e)
            except Exception:
                pass
        return False
    if not os.path.exists(json_path):
        atomic_json_write(json_path, seed)
        return True

    # Working copy exists — self-heal any missing REQUIRED aspect from the seed.
    try:
        with open(json_path) as f:
            cur = json.load(f)
    except json.JSONDecodeError:
        # Corrupt working copy (e.g. a partial first-boot copy from before the
        # write was atomic). Quarantine it — the operator's classifier-grown
        # members live in there and deserve a recovery path — then re-seed.
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')  # clock-ok — quarantine filename
        quarantine = '%s.corrupt-%s' % (json_path, stamp)
        os.replace(json_path, quarantine)
        atomic_json_write(json_path, seed)
        summary = ('working copy unparseable — quarantined to %s and re-seeded '
                   'from the repo baseline. Classifier-grown members are in the '
                   'quarantined file; recover manually.' % quarantine)
        print('[aspects] %s' % summary, flush=True)
        if log_fn:
            try:
                log_fn(summary)
            except Exception:
                pass
        return True
    except OSError:
        return False
    missing = [n for n in REQUIRED_ASPECTS if n in seed and n not in cur]
    # Missing MEMBERS of required aspects that both files carry (job 3).
    # Shape-guarded the same way AspectRegistry._load guards: a working copy
    # can carry a malformed entry (hand-edit, partial write), and an unguarded
    # deref here would abort the WHOLE heal — including job 2 above, whose
    # failure leaves survivor_lineage empty and silently disables the
    # absorbed_into archive exemption. A malformed entry is skipped, not fatal.
    member_heals = []          # (aspect, category, [strings]) — for the caller's log
    skipped_malformed = []
    for n in REQUIRED_ASPECTS:
        if n not in seed or n not in cur:
            continue           # job 1/2 territory, or not seeded at all
        if not isinstance(cur[n], dict) or not isinstance(seed[n], dict):
            skipped_malformed.append(n)
            continue
        for category in ('node_types', 'edge_relations'):
            have = cur[n].get(category)
            want = seed[n].get(category)
            if have is None:
                have = []      # absent key is a legitimate empty list
            if not isinstance(have, list) or not isinstance(want or [], list):
                skipped_malformed.append('%s.%s' % (n, category))
                continue
            gap = [s for s in (want or []) if s not in have]
            if gap:
                member_heals.append((n, category, gap))
    if not missing and not member_heals:
        if skipped_malformed and log_fn:
            try:
                log_fn('working copy has malformed entries, skipped: %s'
                       % ', '.join(skipped_malformed))
            except Exception:
                pass
        return False
    for n in missing:
        cur[n] = seed[n]
    for n, category, gap in member_heals:
        cur[n].setdefault(category, []).extend(gap)   # append — never reorder
    parts = ['+aspect %s' % n for n in missing]
    parts += ['%s.%s += %s' % (n, category, ','.join(gap))
              for n, category, gap in member_heals]
    if skipped_malformed:
        parts.append('SKIPPED malformed: %s' % ', '.join(skipped_malformed))
    summary = 'healed working copy from seed: ' + '; '.join(parts)
    atomic_json_write(json_path, cur)
    # Announce AFTER the write lands, so a failed write can't leave a log line
    # claiming a heal that didn't happen. Two channels on purpose: stdout is the
    # only one that reliably reaches a log at Brain.__init__ time (a short-lived
    # process whose logs-db write hits `database is locked` degrades to a stderr
    # nobody reads), and log_fn gives the daemon a queryable row.
    print('[aspects] %s' % summary, flush=True)
    if log_fn:
        try:
            log_fn(summary)
        except Exception:
            pass                                      # never break boot on a log
    return True


class AspectContractError(Exception):
    """Raised by __getattr__ when `brain.aspects.<name>` references an aspect
    not present in aspects_v1.json.

    Code that calls `registry.<aspect_name>` for a non-required (emergent)
    aspect should use registry.by_name() instead, which returns
    Optional[Aspect].
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
    """First-class API for the aspects system — the ONE door, reads AND writes.

    Production: instantiated once at Brain.__init__, exposed as brain.aspects.
    Construction reconciles the working copy with the repo seed (first-boot
    copy + additive heal — see reconcile_working_copy), then loads and
    validates. All writes to aspects_v1.json go through this object
    (add_members / reconcile_with_seed); every write re-derives the in-memory
    maps, so the cache cannot go stale and needs no invalidation protocol.

    Testing/seeding: use from_dict() to construct without a brain — bypasses
    the load path, takes a pre-built dict of aspect specs.
    """

    def __init__(self, brain):
        self._brain = brain
        self._aspects: dict = {}
        self._reverse_node: dict = {}   # type_string → aspect_name
        self._reverse_edge: dict = {}   # relation_string → aspect_name
        try:
            reconcile_working_copy(log_fn=self._heal_log)
        except Exception as e:
            try:
                self._brain._log_warning(
                    'aspect_registry_seed',
                    'failed to seed user-dir aspects_v1.json from repo baseline',
                    repr(e))
            except Exception:
                pass
        self._load()
        self._validate()

    def _heal_log(self, msg: str) -> None:
        try:
            self._brain._log_warning('aspect_registry_heal', msg)
        except Exception:
            pass

    # ── Loading + writing (the door) ──

    def _load(self) -> None:
        """Load aspects from aspects_v1.json — a PURE read, no side effects.

        Seed materialization / self-heal is NOT here — it happens once at
        __init__ (and on explicit reconcile_with_seed() calls), so the read
        path never writes the file it reads.

        Per-operator state (2026-05-17): the working file lives next to
        brain.db ($BRAIN_DB_DIR/aspects_v1.json), not in the repo. On
        first boot the repo seed is copied to the user dir; all subsequent
        writes stay there. The repo file is the shipped baseline, never
        touched by runtime.

        Multi-membership: a string can appear in multiple aspects' member
        lists. Reverse maps (_reverse_node, _reverse_edge) store the FIRST
        aspect that claimed the string in JSON-iteration order — preserves
        the prior single-aspect API contract for `by_node_type`/`by_edge_relation`
        while letting the underlying data carry richer multi-aspect membership.
        """
        ASPECTS_JSON_PATH = aspects_json_path()

        self._adopt({})

        if not os.path.exists(ASPECTS_JSON_PATH):
            try:
                self._brain._log_warning(
                    'aspect_registry_load',
                    'aspects_v1.json missing — registry empty',
                    ASPECTS_JSON_PATH)
            except Exception:
                pass
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
            return

        self._adopt(data)

    def _adopt(self, data: dict) -> None:
        """Build aspects + reverse-lookup maps from a taxonomy dict — the ONE
        constructor body behind both _load (file) and from_dict (tests).

        Reverse maps use setdefault: the FIRST aspect to claim a string in
        dict-iteration order wins — the documented multi-membership contract.
        The two constructors used to disagree here (from_dict let the LAST
        claimant win), which flipped the reported primary for every
        multi-homed string (`supersedes`, `absorbed_into`, the wisdom types).

        Shape-guarded: non-dict entries are skipped, malformed member lists
        coerce to empty — a hand-edited working copy degrades per-entry, not
        wholesale. This is also the seam for load-time derivations (Step 6's
        precomputed exclusion policies belong here, not per-read).
        """
        self._aspects = {}
        self._reverse_node = {}
        self._reverse_edge = {}
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

    def _validate(self) -> None:
        """Check structural invariants on the loaded aspects; log loudly on any break.

        With JSON-source loading, "auto-heal" is no longer meaningful — the JSON
        file IS the spec. An invariant break is a deployment / local-edit mistake
        that must surface clearly, not silently self-repair.
        """
        # (1) Required-aspect presence — config/deployment issue → warning.
        missing = [n for n in REQUIRED_ASPECTS if n not in self._aspects]
        if missing:
            try:
                self._brain._log_warning(
                    'aspect_contract',
                    'Required aspects missing from aspects_v1.json — registry will not be self-consistent',
                    'missing=%s' % missing)
            except Exception:
                pass  # never let logging block validation

        # (2) Noise-exclusivity invariant — data-integrity break → error.
        # `noise` is the "no semantic claim" bucket. A string that lives in noise
        # AND a semantic aspect would make "not in noise" exclusion drop real
        # knowledge. The encoder strips this at write time
        # (aspect_encoder._validate_classifications), but a hand-edit, a
        # migration, or a direct member-list write could reintroduce it — so
        # surface it loudly here, on every load, regardless of source.
        noise = self._aspects.get('noise')
        if noise is not None:
            noise_e, noise_n = set(noise.edge_relations), set(noise.node_types)
            for name, aspect in self._aspects.items():
                if name == 'noise':
                    continue
                edge_overlap = noise_e & set(aspect.edge_relations)
                node_overlap = noise_n & set(aspect.node_types)
                if edge_overlap or node_overlap:
                    try:
                        self._brain._log_error(
                            'aspect_contract',
                            Exception('noise overlaps a semantic aspect — exclusion invariant broken'),
                            'aspect=%s edge_overlap=%s node_overlap=%s' % (
                                name, sorted(edge_overlap), sorted(node_overlap)))
                    except Exception:
                        pass  # never let logging block validation

    def reconcile_with_seed(self) -> bool:
        """Re-run the seed reconcile (first-boot copy + additive heal) and
        reload if it changed anything. Runs automatically at construction;
        public for callers that update the seed mid-process (tests, redeploy
        flows). Returns True if the working copy was created or modified.
        """
        changed = reconcile_working_copy(log_fn=self._heal_log)
        if changed:
            self._load()
            self._validate()
        return changed

    def add_members(self, classifications, source: str = '') -> int:
        """The single write door for classifier output.

        `classifications`: [{'category': 'node_types'|'edge_relations',
        'value': str, 'aspects': [names, primary first]}] — the shape
        AspectEncoder._validate_classifications emits. Each value is appended
        to each listed aspect's member list (idempotent — duplicates skipped).

        Reads the file fresh at write time, writes atomically, then re-derives
        the in-memory maps from what was just written — the writer owns the
        cache, so no invalidation protocol exists or is needed.

        NOTE (accepted, narrow): a boot-time reconcile in a concurrent
        short-lived process can still interleave with this read-modify-write.
        The reconcile only writes when the working copy is behind the seed,
        and it re-heals idempotently on the next boot — self-correcting, not
        durable loss. Both writers now live in this file; if locking is ever
        needed, this is where it goes.

        Returns the number of (aspect, member) additions actually written.
        Raises on an unreadable working copy — a write must never proceed
        from (and then clobber the file with) an empty in-memory guess.
        """
        path = aspects_json_path()
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            try:
                self._brain._log_error(
                    'aspect_registry_write', e,
                    'add_members: cannot read %s — refusing to write' % path)
            except Exception:
                pass
            raise
        added = 0
        for c in classifications:
            category = c['category']  # 'node_types' or 'edge_relations'
            for aspect_name in c['aspects']:
                entry = data.setdefault(aspect_name, {})
                members = entry.setdefault(category, [])
                if c['value'] not in members:
                    members.append(c['value'])
                    added += 1
        if not added:
            return 0
        atomic_json_write(path, data)
        self._load()
        self._validate()
        print('[aspects] add_members(%s): +%d memberships' % (
            source or 'unattributed', added), flush=True)
        return added

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
        # __dict__ access (not getattr) to avoid recursion when the registry
        # is partly constructed.
        aspects = self.__dict__.get('_aspects')
        if aspects is None:
            raise AttributeError(name)
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
        return self._aspects.get(name)

    # ── Reverse lookups ──

    def by_node_type(self, t: str) -> Optional[Aspect]:
        """Find the aspect whose node_types contains this type, or None."""
        name = self._reverse_node.get(t)
        return self._aspects.get(name) if name else None

    def by_edge_relation(self, r: str) -> Optional[Aspect]:
        """Find the aspect whose edge_relations contains this relation, or None."""
        name = self._reverse_edge.get(r)
        return self._aspects.get(name) if name else None

    # ── Cross-aspect unions ──

    def types_in(self, names: Iterable[str]) -> tuple:
        """Union of node_types across the named aspects (insertion-ordered, deduped)."""
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
        return dict(self._aspects)

    def all_with_counts(self) -> list:
        """Aspect summaries with member counts + previews — for list_aspects MCP.

        Each entry: {name, meaning, node_types_count, edge_relations_count,
        node_types_preview, edge_relations_preview, dimension, locked}.
        """
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
        return {n: a for n, a in self._aspects.items() if n in REQUIRED_ASPECTS}

    def emergent(self) -> dict:
        """Emergent (non-required) aspects only."""
        return {n: a for n, a in self._aspects.items() if n not in REQUIRED_ASPECTS}

    def by_dimension(self, dim: str) -> dict:
        """Aspects in one dimension (e.g., 'semantic', 'temporal' once present)."""
        return {n: a for n, a in self._aspects.items() if a.dimension == dim}

    def dimensions(self) -> set:
        """Set of all dimensions present in the brain right now."""
        return {a.dimension for a in self._aspects.values()}

    # ── Surface-specific (edge enrichment for embeddings) ──

    def relation_meaning_map(self) -> dict:
        """{relation_string: meaning_text} for edge enrichment in surface."""
        out = {}
        for a in self._aspects.values():
            for r in a.edge_relations:
                out[r] = a.meaning
        return out

    def compose_edge_text(self, relation: str, description: str) -> str:
        """Compose the canonical text embedded as an edge's semantic identity.

        Pattern: "[<relation>] <description>"

        INTRINSIC to the edge — does NOT include:
          - the partner node title: would couple the edge embedding to the
            partner (cascade-stale on node revise) and double-count its signal.
          - the relation's aspect-family `meaning`: that text is verbose
            classifier guidance authored for AspectIntegration; baked into
            every edge it dominated the (much shorter) description in the
            embedded string and blunted the per-edge disambiguation that
            embedding the description was meant to provide. The family meaning
            still lives in `aspects_v1.json` and feeds the classifier — it is
            simply not part of the edge's embedding geometry.

        Stable per `(relation, description)` pair, so a single embedding can be
        stored on `edge_relations.embedding` (schema v26+) and reused across
        partner revisions.

        Used by:
          - `GraphDAL.add_relation` (write path) — invalidate + async re-embed
          - `surface_contract` (read-path embed fallback when stored blob NULL)
          - `scripts/backfill_edge_embeddings.py` (one-shot migration)
        """
        rel = (relation or '').strip()
        desc = (description or '').strip()
        parts = []
        if rel:
            parts.append('[%s]' % rel)
        if desc:
            parts.append(desc)
        return ' '.join(parts)

    def type_meaning_map(self) -> dict:
        """{type_string: meaning_text} — symmetric to relation_meaning_map."""
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
        instance._adopt(data)
        return instance
