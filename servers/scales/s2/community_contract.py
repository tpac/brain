"""S2 Community Detection contract — config, metadata schema, rendering.

Three contracts in one file:
1. COMMUNITY_DETECTION — decoder config (z-scores, thresholds)
2. COMMUNITY_METADATA_KEYS — flat keys for community nodes
3. S2CE_NODE_FORMAT — how representative nodes are rendered for the encoder
"""


# ═══════════════════════════════════════════════════════════════
# DECODER CONFIG
# ═══════════════════════════════════════════════════════════════

COMMUNITY_DETECTION = {
    # Minimum nodes to form a community proposal
    'min_community_size': 3,

    # ── Z-score pair scoring ──
    # z = (raw_shared - bucket_mean) / bucket_std
    # z≥1.0 = 1 std above average for pairs at this degree
    'z_seed_threshold': 1.0,

    # ── Adaptive grow ──
    # Threshold = p50 of all node-to-cluster affinities (computed, not static)
    'grow_threshold_fallback': 0.20,

    # ── Ratio-based overlap ──
    # secondary_affinity / primary_affinity ≥ this
    'overlap_min_ratio': 0.5,

    # ── Add-to-existing affinity threshold ──
    # Fraction of node's typed neighbors that must be in the community.
    # 15% was too low (noise). 25% = at least 1 in 4 neighbors.
    'add_to_existing_min_affinity': 0.25,

    # ── Embedding placement for orphans ──
    'embedding_placement_threshold': 0.50,

    # ── Cluster-to-community overlap (duplicate prevention) ──
    # Before proposing new_community, check if cluster members mostly
    # connect to an existing community. If fraction of cluster nodes
    # with neighbors in the same existing community >= this threshold,
    # convert to add_to_existing instead.
    'cluster_overlap_threshold': 0.60,

    # ── Drift detection ──
    # foreign_affinity > home_affinity × drift_ratio triggers a drift proposal.
    # Per-node override: _sys_drift_threshold in metadata_kv. Encoder raises
    # this on rejection (e.g. 1.6, 1.7) so weak drift doesn't re-propose.
    'drift_ratio': 1.5,
    'drift_ratio_step': 0.1,        # how much encoder raises per rejection
    'drift_min_foreign_affinity': 0.15,  # absolute floor for foreign affinity

    # ── Community merge detection ──
    # Merge when overlap >= this AND unique members in smaller < min_unique
    'merge_overlap_threshold': 0.80,    # % of smaller community's members shared
    'merge_min_unique_members': 3,      # smaller must have < this many unique to merge

    # ── Cross-cutting detection ──
    'cross_cutting_min_degree': 15,
    'cross_cutting_max_top_affinity': 0.35,

    # ── Encoder ──
    # Haiku 4.5 after A/B eval proved it outperforms Sonnet 4.6 on this task
    # (Sonnet 4.6 couldn't complete 2 rounds without max_tokens truncation or
    # running out of rounds on inspection. Haiku produced 10 communities
    # with structured narratives in ~60s vs Sonnet's 0 writes in 2 rounds.)
    'model': 'claude-haiku-4-5-20251001',
    'max_tokens': 32768,
    'max_proposals_per_call': 8,    # Smaller batches = richer per-community
                                    # context AND lower round-2 generation
                                    # load (10 was hitting the 180s client
                                    # timeout on batch 1 consistently).
    'max_actionable_per_run': 30,   # Cap per idle cycle (was 60 for Sonnet)
    # Per-type quotas — priority order: merge > new_community > add_to_existing > health > drift
    'type_quotas': {
        'merge_communities': 3,
        'new_community': 10,
        'add_to_existing': 12,
        'health_update': 3,
        'drift': 2,
    },
    'max_rounds': 2,                # LLM rounds per batch — get_nodes then brain_batch, done
    'journal_max_chars': 14000,
}


# ═══════════════════════════════════════════════════════════════
# COMMUNITY ENRICHMENT LLM CONFIG
# (stored in interactions table as 's2_community_enrichment' parameters)
# ═══════════════════════════════════════════════════════════════

COMMUNITY_ENRICHMENT = {
    'model': 'claude-haiku-4-5-20251001',
    'max_tokens': 32768,
}


# ═══════════════════════════════════════════════════════════════
# PROPOSAL TYPE PRIORITY
# ═══════════════════════════════════════════════════════════════
#
# Order proposals by structural impact so the encoder sees the
# highest-leverage work first:
#   1. merges simplify the community landscape
#   2. new_community creates frontier for future inside-out scans
#   3. add_to_existing extends communities at the frontier
#   4. health_update corrects community metadata
#   5. drift moves a single node between communities (lowest impact)
#
# Used by rejection_table.sort_proposals_by_priority() when capping
# quotas and sorting the batch sent to the encoder.

TYPE_PRIORITY = {
    'merge_communities': 0,
    'new_community': 1,
    'add_to_existing': 2,
    'health_update': 3,
    'drift': 4,
}


# ═══════════════════════════════════════════════════════════════
# COMMUNITY NODE METADATA KEYS
# Flat keys in node_metadata_kv — each independently queryable via SQL.
#
# Source of truth for membership: community_member edges.
# Metadata is denormalized cache, rebuilt on write.
#
# Query examples:
#   SELECT node_id, value FROM node_metadata_kv
#   WHERE key = 'community_internal_fraction' AND CAST(value AS REAL) < 0.2
#
#   SELECT node_id, value FROM node_metadata_kv
#   WHERE key = 'community_maturity' AND value = 'active'
# ═══════════════════════════════════════════════════════════════

COMMUNITY_METADATA_KEYS = {
    # ── Membership (denormalized from community_member edges) ──
    'community_members',              # JSON list of node IDs
    'community_size',                 # Integer

    # ── Structure (from decoder) ──
    'community_edge_signature',       # JSON dict {family: proportion}
    'community_centroid',             # Base64-encoded embedding bytes
    'community_internal_edges',       # Integer
    'community_external_edges',       # Integer
    'community_internal_fraction',    # Float 0-1
    'community_is_corridor',          # "true" / "false"

    # ── Health (tracked across runs) ──
    'community_growth_rate',          # Float: members added per run (rolling)
    'community_last_change',          # ISO timestamp
    'community_last_active',          # ISO timestamp of newest member's created_at
    'community_created_at_range',     # "2026-03-20 to 2026-04-09"
    'community_run_count',            # Integer: how many S2 runs touched this

    # ── Encoder-produced (from LLM) ──
    # community_narrative / community_open_questions REMOVED 2026-04-17:
    # - narrative was a duplicate of content (the node's own content field)
    # - open_questions migrated to the standard `question` enrichment path
    #   (embedded, participates in recall like any other node's question)
    'community_key_decisions',        # JSON list of node IDs (3-5 defining nodes)
    'community_correction_count',     # Integer: correction edges inside community
    'community_dominant_type',        # Most common node type among members
    'community_maturity',             # "forming" / "active" / "settled" / "corridor"

    # ── Encoder-produced (human-readable) ──
    'community_latest_development',   # One sentence: most recent change

    # ── Diagnostics ──
    'community_recall_signals',       # JSON list of diagnostic dicts
    'community_overlap_reasons',      # JSON dict {node_id: reason}
}


# ═══════════════════════════════════════════════════════════════
# S2CE NODE RENDERING FORMAT
#
# How representative nodes are rendered for the community encoder.
# The encoder needs to understand what the community IS about —
# not every detail of every member, but enough to write a narrative.
#
# Compare to S1E's format (full content, 5 edges, all metadata):
#   S1E writes about individual nodes → needs full depth
#   S2CE writes about communities → needs the gist of 5 representatives
#
# Per representative: ~600-800 chars
# 5 reps per community: ~3-4K chars
# 15 communities per batch: ~45-60K chars for representatives alone
# Plus edge signatures, sample edges, metadata → fits in Sonnet 200K
# ═══════════════════════════════════════════════════════════════

# Used when encoder calls get_nodes() to inspect specific members
S2CE_NODE_FORMAT = {
    'content_limit': 300,       # Gist, not full content
    'edge_limit': 4,            # Relations matter — keep 4 with descriptions
    'metadata_limit': 150,      # Key metadata only
    'time_format': 'relative',  # "2d ago" not "2026-04-09"
}

# Compact format for existing communities in the context listing.
# Just enough for the encoder to check "does my proposal overlap with this?"
S2CE_COMMUNITY_FORMAT = {
    'content_limit': 150,       # Brief narrative gist
    'edge_limit': 0,            # No edges
    'metadata_limit': 0,        # No metadata — content is the narrative
    'time_format': 'relative',
}

# Fields the encoder needs to see per representative node:
# title          — always (what is this node)
# type           — always (what kind of knowledge)
# content        — 300 chars (the gist)
# situation      — always (when is this relevant — helps write community situation)
# confidence     — always (stable/uncertain)
# created_at     — relative (temporal context)
# _corrections   — always (story arc: what corrected what)
# connections    — 4 edges with relation + description (the story between nodes)
# reasoning      — if present (why this was encoded — helps understand intent)
# anchor/user_raw_quote — if present (exact words carry weight)
#
# Fields the encoder does NOT need per representative:
# activation, stability, recency_score — internal scoring
# emotion, emotion_label — only if building identity communities
# encoding_version, encoding_source — provenance
# content_summary — redundant
# access_count — doesn't help narrative writing
# last_accessed, updated_at — internal timestamps

# What the encoder sees per community (assembled by _format_proposals):
#
# [N] NEW COMMUNITY — 18 seed members
#     Internal fraction: 63%  |  Corridor: no
#     Created: 2026-03-25 to 2026-04-09  |  Newest member: 1d ago
#     Relational signature: implementation_execution(50%), validation_evidence(50%)
#     Correction chains: 3 internal corrections
#     Type distribution: decision(6), mechanism(4), lesson(3), research(2)
#
#     Timeline (chronological — shows how the story developed):
#       Origin (2026-03-25):
#         [finding] "Haiku judge timeout: hook kills judge" (id:abc, conf:0.8)
#           Content: ...
#       Transition (2026-03-28):
#         [decision] "Judge moved into daemon" (id:def, conf:0.9)
#           ⚠ Corrects: "Haiku judge timeout" (id:abc)
#           Content: ...
#       Latest (2026-04-07):
#         [finding] "Hook pipeline latency: 500ms is our code" (id:ghi, conf:0.9)
#           Content: ...
#
#     Most connected (structural hubs):
#       [mechanism] "Recall pipeline: 10-step decode path" (id:jkl, 6 internal edges)
#         Content: ...
#         Edges: ...
#
#     Sample internal edges:
#       - "brain_surface.py" produced "Monolith split" — Split decision produced this module
#
#     Existing communities this run is aware of: (from journal)
#       C1 "Hook Architecture Evolution" (25 members, active)
#       C2 "Recall Pipeline Journey" (18 members, active)
#       ...


# EDGE_TYPE_GROUPS + RELATION_TO_GROUP — REMOVED 2026-05-04
# (Step 12 cleanup of unified-aspects). Were "legacy reference" dicts
# pre-dating the s2_edge_families interaction; never read by runtime
# since that interaction landed. brain.aspects now owns the equivalent
# data — consumers iterate brain.aspects.all() and read .edge_relations.


# ═══════════════════════════════════════════════════════════════
# PROPOSAL TYPES
# ═══════════════════════════════════════════════════════════════

PROPOSAL_TYPES = {
    'new_community',    # Cluster of nodes forming a community
    'node_affinities',  # Node's ranked affinities to clusters
    'cross_cutting',    # High-degree thin-spread principle
}
