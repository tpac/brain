"""S2 Aspect Integration — Contract and Configuration.

Classifies distinct node types and edge relations into the required
aspects defined in `aspects_v1.json`. Closed-list classification — encoder
can only route to existing aspects (or `noise`/`generic_relation` as
catch-alls), never proposes new ones. New aspects are added by humans
editing `aspects_v1.json`.

The unit reads the taxonomy via brain.aspects and writes through its door
(AspectRegistry.add_members); it never mutates brain state. No suppression
machinery — closed list means every string gets a home, so SKIP isn't a
valid encoder action.

Path resolution and the seed-reconcile live in servers/aspect_store.py and
servers/aspects.py (re-exported here for existing importers).
"""

from servers.aspect_store import (  # noqa: F401 — canonical home; re-exported
    REQUIRED_ASPECTS,
    SEED_ASPECTS_JSON_PATH,
    aspects_json_path,
    aspects_proposed_path,
)


ASPECT = {
    # LLM config
    'model': 'claude-sonnet-4-6',  # Sonnet for classification quality
    'max_tokens': 8192,

    # Batch sizing — chosen for clean per-item attention without losing
    # the cross-string visibility that helps consistency (e.g., `corrects`
    # and `corrected_by` classified together).
    'max_candidates_per_call': 30,

    # Decoder filters
    # 1 = classify EVERY string, singletons included. An unaspected string is
    # invisible to every aspect-driven consumer (correction_enrich, the wisdom
    # Frame pull, noise filtering); a one-off typo filed under `noise` is the
    # cheaper failure. Most strings arrive exactly once, so this is the floor,
    # not a tuning knob.
    'min_count_threshold': 1,
    'examples_per_candidate': 3,     # nodes/edges shown per candidate string
}

# Interaction config default for the `s2_aspects` K — sliced from ASPECT so
# model/max_tokens have exactly one home in this file.
ASPECT_INTERACTION_DEFAULT = {
    'model': ASPECT['model'],
    'max_tokens': ASPECT['max_tokens'],
}
