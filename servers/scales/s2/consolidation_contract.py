"""S2 Consolidation contract — cluster shape, config, and rendering.

Three contracts:
1. CLUSTER_SHAPE — what the decoder produces, what the encoder can rely on
2. CONSOLIDATION — config (thresholds, encoder settings)
3. CONSOLIDATION_NODE_FORMAT — how cluster member nodes are rendered for the encoder
"""


# ═══════════════════════════════════════════════════════════════
# CLUSTER SHAPE — decoder→encoder contract
#
# Each cluster is a dict with these fields. The encoder RELIES on
# this shape — if the decoder changes what it produces, update here
# and the encoder's _format_clusters() must adapt.
#
# This is the equivalent of S1's CANDIDATES_FILE contract.
# ═══════════════════════════════════════════════════════════════

CLUSTER_SHAPE = {
    # ── Identity ──
    'nodes': 'list[str]',              # Node IDs in the cluster (sorted)
    'size': 'int',                     # len(nodes)

    # ── Similarity (two independent dimensions) ──
    'content_cosine_max': 'float',     # Highest content embedding similarity
    'content_cosine_avg': 'float',     # Average content similarity
    'title_cosine_max': 'float',       # Highest title-only embedding similarity
    'title_cosine_avg': 'float',       # Average title similarity
    'pair_scores': 'dict[str, dict]',  # "idA-idB" → {content: float, title: float}

    # ── Pre-classification (decoder's algorithmic guess) ──
    'pre_class': 'str',               # likely_consolidate | likely_evolve | likely_keep | needs_judgment

    # ── Node details (per member) ──
    'node_details': 'dict[str, dict]',  # node_id → {
    #   title, type, content, confidence, encoding_source,
    #   keywords, locked, critical, created_at, updated_at,
    #   reasoning*, user_raw_quote*, anchor_raw_quote*,
    #   situation*, correction_of*
    #   (* = from node_metadata_kv, may be absent)
    # }

    # ── Behavioral evidence (from S1 traces) ──
    'co_recall_count': 'int',          # Times pair appeared as candidates for same query
    'judge_preference': 'dict[str, int]',  # node_id → times selected by surfacer
    'recall_counts': 'dict[str, int]',     # node_id → total times appeared as candidate
    'query_coverage': 'dict[str, list]',   # node_id → [query strings that found it]

    # ── Catalog blindness (from S1E traces) ──
    'catalog_blind': 'dict[str, bool]',  # node_id → was created without seeing cluster mates?

    # ── Graph structure ──
    'shared_edge_count': 'int',        # Neighbors shared by ALL cluster members
    'unique_edges': 'dict[str, int]',  # node_id → edges not in shared set
    'edge_details': 'dict[str, dict]', # node_id → {neighbor_id: [{relation, description, title, type}]}
    'communities': 'dict[str, list]',  # node_id → [{id, title}] community memberships
    'same_community': 'bool',          # Any pair shares a community?
    'shared_community_ids': 'list[str]',

    # ── Correction ──
    'has_correction_edge': 'bool',     # Correction/supersedes edge between members?

    # ── Tension ──
    'has_tension_edge': 'bool',        # Contradicts/challenges edge between members?
                                        # NEVER consolidate — tensions are productive
}

# Fields the encoder MUST see to make good decisions.
# If any of these are missing, the formatter should warn.
CLUSTER_REQUIRED_FIELDS = {
    'nodes', 'size', 'pre_class',
    'content_cosine_max', 'title_cosine_max',
    'node_details',
    'co_recall_count', 'judge_preference', 'catalog_blind',
    'shared_edge_count', 'same_community', 'has_correction_edge',
}


# ═══════════════════════════════════════════════════════════════
# NODE RENDERING FORMAT
#
# How cluster member nodes are rendered for the consolidation encoder.
# Consolidation needs FULL depth for 2-5 nodes — Sonnet must read
# the actual content to decide synthesize vs keep.
#
# Compare to community format (gist of 5 reps → 300 char content):
#   S2CE writes about communities → needs the gist
#   Consolidation decides per-node fate → needs the substance
# ═══════════════════════════════════════════════════════════════

CONSOLIDATION_NODE_FORMAT = {
    'content_limit': 600,       # More depth than community (300)
    'edge_limit': 5,            # Full edge context
    'metadata_limit': 300,      # Full metadata — reasoning, raw quotes
    'time_format': 'relative',
}


# ═══════════════════════════════════════════════════════════════
# CONSOLIDATION CONFIG
# ═══════════════════════════════════════════════════════════════

CONSOLIDATION = {
    # ── Embedding scan (decoder) ──
    'similarity_threshold': 0.89,       # Minimum cosine for candidate pair.
                                        # Raised from 0.82 after Nomic-Q migration: at 0.82 the
                                        # similarity graph collapsed into a 2165-node giant
                                        # component (86% of corpus) — unprocessable as clusters.
                                        # 0.89 yields ~250 well-formed clusters with biggest ~12.
    'max_cluster_size': 10,             # Larger clusters get dropped + logged.
                                        # Historical note: 5 was too tight —
                                        # genuine convergence across sessions
                                        # routinely produced 6-9 member clusters
                                        # that were silently lost. 10 covers the
                                        # observed backlog; if clusters hit 15+,
                                        # threshold is probably too permissive.

    # ── Behavioral trace lookback (decoder) ──
    'co_recall_lookback_hours': 168,    # 7 days of S1R traces
    'encoding_lookback_hours': 168,     # 7 days of S1E traces

    # ── Suppression (decoder) ──
    # Pairs with these edge relations are skipped (already reviewed)
    'suppression_relations': {'similar_to', 'consolidated_into'},

    # ── Pre-classification thresholds (decoder) ──
    'likely_consolidate_cosine': 0.90,  # Above this + structural signal = likely consolidate

    # ── Encoder ──
    'model': 'claude-sonnet-4-20250514',
    'max_tokens': 32768,
    'max_proposals_per_call': 10,       # Clusters per Sonnet call
    'max_rounds': 2,                    # Tool-use rounds per call — read then write, done
    'journal_max_chars': 14000,

    # ── Cold start / run cap ──
    # Max clusters the encoder processes per idle cycle.
    # Cold start (164 clusters) spreads across ~6 cycles.
    # Each cycle: suppression edges prevent re-processing.
    # Order: likely_consolidate → likely_evolve → likely_keep → needs_judgment
    # Easy cases first — validates prompt quality before harder decisions.
    'max_clusters_per_run': 10,
}
