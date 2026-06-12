"""S2 Aspect Integration — Contract and Configuration.

Classifies distinct node types and edge relations into the 14 required
aspects defined in `aspects_v1.json`. Closed-list classification — encoder
can only route to existing aspects (or `noise`/`generic_relation` as
catch-alls), never proposes new ones. New aspects are added by humans
editing `aspects_v1.json`.

The unit is self-contained: reads + writes JSON files only, never mutates
brain state. No suppression machinery — closed list means every string
gets a home, so SKIP isn't a valid encoder action.
"""

import os


ASPECT = {
    # LLM config
    'model': 'claude-sonnet-4-6',  # Sonnet for classification quality
    'max_tokens': 8192,

    # Batch sizing — chosen for clean per-item attention without losing
    # the cross-string visibility that helps consistency (e.g., `corrects`
    # and `corrected_by` classified together).
    'max_candidates_per_call': 30,

    # Decoder filters
    'min_count_threshold': 2,        # ignore singletons (typos, one-offs)
    'examples_per_candidate': 3,     # nodes/edges shown per candidate string
}


# Runtime aspect state lives next to brain.db (per-operator, not in repo).
# Repo-bundled seed (SEED_ASPECTS_JSON_PATH) is the first-boot baseline;
# AspectRegistry copies seed → user dir on first load when missing. After
# that, all encoder writes stay in the user dir — the repo seed is read-
# only and never touched by runtime.
_DEFAULT_DB_DIR = os.path.join(os.path.expanduser('~'), 'AgentsContext', 'brain')
_BRAIN_DB_DIR = os.environ.get('BRAIN_DB_DIR', _DEFAULT_DB_DIR)

ASPECTS_JSON_PATH = os.environ.get(
    'ASPECTS_JSON_PATH',
    os.path.join(_BRAIN_DB_DIR, 'aspects_v1.json'))

# Per-cycle audit artifact. Same per-operator location.
ASPECTS_PROPOSED_PATH = os.environ.get(
    'ASPECTS_PROPOSED_PATH',
    os.path.join(_BRAIN_DB_DIR, 'aspects_proposed.json'))

# Repo seed — frozen baseline shipped with the plugin. Never written.
SEED_ASPECTS_JSON_PATH = os.path.join(
    os.path.dirname(__file__), 'aspects_v1.json')


def ensure_aspects_user_copy() -> bool:
    """Seed the user-dir aspects file from the repo on first boot.

    Returns True if a copy happened, False otherwise (already exists,
    or seed missing). Idempotent — safe to call on every boot.
    """
    import shutil
    if os.path.exists(ASPECTS_JSON_PATH):
        return False
    if not os.path.exists(SEED_ASPECTS_JSON_PATH):
        return False
    os.makedirs(os.path.dirname(ASPECTS_JSON_PATH), exist_ok=True)
    shutil.copy2(SEED_ASPECTS_JSON_PATH, ASPECTS_JSON_PATH)
    return True
