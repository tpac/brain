"""Aspect store — byte-level core for aspects_v1.json.

Touches bytes; never interprets them. The semantic layer (AspectRegistry in
servers/aspects.py) is the ONE public door for reading and writing taxonomy
content — everything here is its plumbing: path resolution, the required-name
contract, and the single atomic JSON writer.

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
