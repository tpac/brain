"""S2 Aspect Integration — Contract and Configuration.

Classifies distinct node types and edge relations into the 15 required
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
    """Seed the user-dir aspects file from the repo seed, and SELF-HEAL.

    Two jobs, both idempotent and safe to call on every boot:
      1. First boot — working copy missing → copy the whole seed.
      2. Self-heal — working copy exists but the seed has gained a new
         REQUIRED aspect → add the missing required aspect(s) from the seed.
         This is how a deliberate seed addition (e.g. survivor_lineage)
         propagates to an existing brain without a manual migration.

    Scoped tightly: only adds WHOLE missing required aspects. Never writes
    the seed itself (tests may point ASPECTS_JSON_PATH at it); never touches
    existing entries, so operator/AspectIntegration-grown member lists and
    emergent aspects are preserved. Returns True if the file was created or
    modified.
    """
    import shutil
    if not os.path.exists(SEED_ASPECTS_JSON_PATH):
        return False
    # Never heal the seed into itself.
    if os.path.abspath(ASPECTS_JSON_PATH) == os.path.abspath(SEED_ASPECTS_JSON_PATH):
        return False
    if not os.path.exists(ASPECTS_JSON_PATH):
        os.makedirs(os.path.dirname(ASPECTS_JSON_PATH), exist_ok=True)
        shutil.copy2(SEED_ASPECTS_JSON_PATH, ASPECTS_JSON_PATH)
        return True

    # Working copy exists — self-heal any missing REQUIRED aspect from the seed.
    import json
    from servers.aspects import REQUIRED_ASPECTS  # local: avoid import cycle
    try:
        with open(SEED_ASPECTS_JSON_PATH) as f:
            seed = json.load(f)
        with open(ASPECTS_JSON_PATH) as f:
            cur = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    missing = [n for n in REQUIRED_ASPECTS if n in seed and n not in cur]
    if not missing:
        return False
    # NOTE (accepted, narrow): this read-modify-write can lose to a concurrent
    # AspectIntegration os.replace that carries a pre-heal snapshot. The window
    # is tiny — self-heal only writes when a required aspect is MISSING (the
    # first boot after a seed addition), and AspectIntegration runs during S2
    # idle maintenance, not at Brain.__init__. If it ever loses, the next boot
    # self-heals again (idempotent), so it's self-correcting, not durable loss.
    for n in missing:
        cur[n] = seed[n]
    # Atomic write — temp file + os.replace, so a crash mid-write can't leave a
    # truncated/0-byte aspects file. A corrupt working copy loads as an empty
    # registry → relations_in(['survivor_lineage']) returns () → the
    # absorbed_into exemption silently disables and the reaper scrubs redirect
    # edges. Mirrors AspectEncoder._write_aspects.
    import tempfile
    d = os.path.dirname(ASPECTS_JSON_PATH)
    fd, tmp = tempfile.mkstemp(prefix='aspects_v1_', suffix='.json.tmp', dir=d)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(cur, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.replace(tmp, ASPECTS_JSON_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return True
