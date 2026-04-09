"""Community Detection contract — config and constants.

All tunable parameters for the community detection unit.
Stored in the interactions table as a learnable boundary.
"""

# Algorithm and threshold parameters
COMMUNITY_DETECTION = {
    # Graph size gate — don't run on tiny graphs
    'min_graph_nodes': 20,

    # Ignore edges below this weight (noise filtering)
    'edge_weight_threshold': 0.05,

    # Leiden resolution parameter — higher = more, smaller communities
    'resolution': 1.0,

    # Don't create a community node for communities with fewer members
    'min_community_size': 3,

    # Naming heuristic — how many keywords in community title
    'max_community_name_keywords': 5,

    # Max member titles to include in community content description
    'max_member_titles_in_content': 5,

    # Minimum hours between runs (cooldown)
    'cooldown_hours': 6,

    # Stability threshold — skip update if < N% of nodes changed communities
    'stability_threshold_pct': 10,
}
