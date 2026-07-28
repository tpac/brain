"""Aspect store — byte-level core + structural contract for aspects_v1.json.

Path resolution, the required-name contract, the single atomic JSON writer,
and the structural validator every write is gated on. The semantic layer
(AspectRegistry in servers/aspects.py) is the ONE public door for reading
and writing taxonomy content — everything here is its plumbing.

This module imports nothing from servers/ so the dependency is strictly
one-directional (aspects.py → here, scales/s2 → here). The import cycle the
old placement forced (aspects.py ↔ aspect_contract.py, both via
function-local imports) is dissolved by this file existing.
"""

import json
import os
import tempfile


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
    'wisdom',               # generative subset (insight/lesson/principle/vision/reflection/meta_learning/philosophy) — Frame's "What I've learned"

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
    'survivor_lineage',         # absorbed_into — archived→living-descendant redirect
)


# Runtime aspect state lives next to brain.db (per-operator, not in repo).
# Repo-bundled seed (SEED_ASPECTS_JSON_PATH) is the first-boot baseline;
# the registry copies seed → user dir on first load when missing. After
# that, all writes stay in the user dir — the repo seed is read-only and
# never touched by runtime.
#
# Paths are resolved at CALL time, not import time. BRAIN_DB_DIR (and the
# explicit overrides) are read on every call so a later `os.environ` change
# takes effect — IsolatedBrain sets BRAIN_DB_DIR in __enter__, AFTER this
# module is imported, and a module-level constant would freeze the live
# user-dir path and leak heals into it (observed 2026-06-16). Pinned by
# tests/test_aspects_path_isolation.py.
_DEFAULT_DB_DIR = os.path.join(os.path.expanduser('~'), 'AgentsContext', 'brain')


def aspects_json_path() -> str:
    """Path to the per-operator working aspects file (call-time resolved)."""
    return os.environ.get(
        'ASPECTS_JSON_PATH',
        os.path.join(os.environ.get('BRAIN_DB_DIR', _DEFAULT_DB_DIR),
                     'aspects_v1.json'))


def aspects_proposed_path() -> str:
    """Path to the per-cycle audit artifact (call-time resolved)."""
    return os.environ.get(
        'ASPECTS_PROPOSED_PATH',
        os.path.join(os.environ.get('BRAIN_DB_DIR', _DEFAULT_DB_DIR),
                     'aspects_proposed.json'))


# Repo seed — frozen baseline shipped with the plugin. Never written.
SEED_ASPECTS_JSON_PATH = os.path.join(
    os.path.dirname(__file__), 'scales', 's2', 'aspects_v1.json')


def validate_taxonomy(data) -> list:
    """Structural invariants any taxonomy copy must satisfy. Returns a list of
    violation strings — empty means valid.

    This is the STRUCTURAL set, shared by the seed and the working copy: it
    is what a write must never produce, so the registry's door refuses on it
    and the boot load reports it. Seed-only CURATION standards (locked flags,
    meaning length, display labels, no-extra-aspects) live in
    tests/test_aspects_contract.py — the working copy legitimately grows
    emergent unlocked aspects, so the gate must not enforce those.

    Invariants:
      1. every entry is an object
      2. node_types / edge_relations, when present, are lists of unique,
         non-empty strings
      3. every REQUIRED_ASPECTS name is present
      4. noise-exclusivity: a string in `noise` appears in NO other aspect
         (either category) — "not in noise" exclusion filters trust this
      5. per-aspect fact fields, when present, are well-shaped: `accepts`
         is a non-empty list drawn from {node_types, edge_relations};
         `routable` / `prompt_visible` / `structural_lineage` are booleans.
         Presence itself is a seed-curation standard (tests), not a gate —
         emergent aspects and pre-heal working copies legitimately lack them.

    Keys starting with '_' are reserved for in-file documentation
    (`_schema`) — JSON has no comments. Skipped here and by every reader
    (registry _adopt, dashboard); refused as write targets by add_members.
    """
    violations = []
    if not isinstance(data, dict):
        return ['taxonomy root is %s, expected object' % type(data).__name__]

    entries = {}
    for name, entry in data.items():
        if name.startswith('_'):
            continue
        if not isinstance(entry, dict):
            violations.append("aspect '%s': entry is %s, expected object"
                              % (name, type(entry).__name__))
            continue
        entries[name] = entry
        for category in ('node_types', 'edge_relations'):
            members = entry.get(category)
            if members is None:
                continue
            if not isinstance(members, list):
                violations.append("aspect '%s'.%s: %s, expected list"
                                  % (name, category, type(members).__name__))
                continue
            seen = set()
            for m in members:
                if not isinstance(m, str) or not m:
                    violations.append("aspect '%s'.%s: malformed member %r"
                                      % (name, category, m))
                elif m in seen:
                    violations.append("aspect '%s'.%s: duplicate member '%s'"
                                      % (name, category, m))
                else:
                    seen.add(m)
        accepts = entry.get('accepts')
        if accepts is not None:
            if (not isinstance(accepts, list) or not accepts
                    or any(a not in ('node_types', 'edge_relations')
                           for a in accepts)
                    or len(set(accepts)) != len(accepts)):
                violations.append(
                    "aspect '%s'.accepts: %r — expected a non-empty unique "
                    "subset of ['node_types', 'edge_relations']" % (name, accepts))
        for flag in ('routable', 'prompt_visible', 'structural_lineage'):
            val = entry.get(flag)
            if val is not None and not isinstance(val, bool):
                violations.append("aspect '%s'.%s: %r, expected boolean"
                                  % (name, flag, val))

    missing = [n for n in REQUIRED_ASPECTS if n not in data]
    if missing:
        violations.append('required aspects missing: %s' % missing)

    noise = entries.get('noise')
    if noise is not None:
        for category in ('node_types', 'edge_relations'):
            members = noise.get(category)
            noise_set = set(members) if isinstance(members, list) else set()
            if not noise_set:
                continue
            for name, entry in entries.items():
                if name == 'noise':
                    continue
                other = entry.get(category)
                overlap = noise_set & set(other) if isinstance(other, list) else set()
                if overlap:
                    violations.append(
                        "noise-exclusivity broken: noise shares %s with '%s': %s"
                        % (category, name, sorted(overlap)))
    return violations


def atomic_json_write(path: str, data) -> None:
    """The ONE atomic JSON writer for aspect files: temp file + os.replace.

    A crash mid-write can't leave a truncated/0-byte file — a corrupt working
    copy loads as an empty registry → relations_in(['survivor_lineage'])
    returns () → the absorbed_into exemption silently disables and the reaper
    scrubs redirect edges.

    One canonical dump shape (indent=2, ensure_ascii=False, trailing newline)
    so the file stops interleaving two writers' escaping styles.
    """
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '_',
                               suffix='.tmp', dir=d)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
