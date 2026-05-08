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
    'model': 'claude-sonnet-4-5-20250929',  # Sonnet for classification quality
    'max_tokens': 8192,

    # Batch sizing — chosen for clean per-item attention without losing
    # the cross-string visibility that helps consistency (e.g., `corrects`
    # and `corrected_by` classified together).
    'max_candidates_per_call': 30,

    # Decoder filters
    'min_count_threshold': 2,        # ignore singletons (typos, one-offs)
    'examples_per_candidate': 3,     # nodes/edges shown per candidate string
}


# Path to the aspects spec + working state. Single file, descriptions +
# growing member lists. Override via env var for eval/clone runs.
ASPECTS_JSON_PATH = os.environ.get(
    'ASPECTS_JSON_PATH',
    os.path.join(os.path.dirname(__file__), 'aspects_v1.json'))


# Path to the proposed-this-cycle audit file. Encoder writes here before
# auto-merging into ASPECTS_JSON_PATH. Useful for debugging which cycle
# classified what, and for operator review when auto-merge is disabled.
ASPECTS_PROPOSED_PATH = os.environ.get(
    'ASPECTS_PROPOSED_PATH',
    os.path.join(os.path.dirname(__file__), 'aspects_proposed.json'))
