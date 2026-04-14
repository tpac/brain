"""
brain — BrainDreams Mixin

dream(), consolidate(), _spawn_thought(), and related methods removed 2026-04-13.
Dreams created noise nodes (intuition, thought) with dreamed_from edges that S2
consolidation had to filter out. consolidate() wrote to deprecated stability field.
S2 integration units replace both. See git history for reference.

This mixin is now empty. It remains as a placeholder until Brain class
inheritance is cleaned up.
"""


class BrainDreamsMixin:
    """Dreams methods for Brain — currently empty after proto-S2 removal."""
    pass
