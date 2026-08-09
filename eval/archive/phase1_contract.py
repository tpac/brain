"""Phase 1 Redesign — Constants and Configuration.

All tunable parameters for the recognition-based recall pipeline.
Follows the brain's contract-first pattern: constants live here, not in algorithm code.
"""

# ═══════════════════════════════════════════════════
# STEP 1: Query Understanding
# ═══════════════════════════════════════════════════

QUERY = {
    # Garbage detection — skip pipeline for low-value messages
    'garbage_max_chars': 50,       # Messages shorter than this get garbage check
    'garbage_floor': 0.45,         # Best community cosine below this = skip (tight)

    # Community vector — long conversation arc
    'community_window': 10,        # User turns to blend for community matching
    'community_weights': [0.25, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03],

    # Node vector — immediate topic
    'node_window': 3,              # User turns to blend for node matching
    'node_weights': [0.6, 0.3, 0.1],  # Matches existing TURN_WEIGHTS
}

# ═══════════════════════════════════════════════════
# STEP 2: Candidate Discovery
# ═══════════════════════════════════════════════════

DISCOVERY = {
    # Community channel
    'community_floor': 0.45,       # Min community cosine to activate (tight — want 3-8, not 20)
    'member_floor': 0.30,          # Community members must individually score above this

    # Node channel
    'node_floor': 0.35,            # Min node cosine to include (tighter than current 0.25)

    # Situation matching
    'situation_threshold': 0.40,   # Min situation cosine to boost
    'situation_weight': 0.15,      # Additive boost from situation match
}

# ═══════════════════════════════════════════════════
# STEP 3: Enrichment Scoring
# ═══════════════════════════════════════════════════

ENRICHMENT = {
    # Reuses existing constants from brain_constants.py:
    # TITLE_MATCH_BOOST = 0.3
    # CRITICAL_BOOST = 3.0
    # NOISE_FLOOR_THRESHOLD = 0.15
    # Plus z-weighted enrichment from pipeline_contract.EMBEDDING_GROUPS
}

# ═══════════════════════════════════════════════════
# STEP 3.5: Chain Detection
# ═══════════════════════════════════════════════════

# Edge family weights for chain scoring.
# Higher weight = stronger chain signal when this family connects candidates.
# Based on semantic meaningfulness: evolution/correction chains carry more
# signal than generic "related_to" connections.
CHAIN_EDGE_FAMILY_WEIGHTS = {
    'extension_refinement':    0.9,   # Knowledge deepening: extends, refines, evolves
    'correction_improvement':  0.85,  # Getting better: corrects, improves, reframes
    'explanation_causation':   0.8,   # Why chain: causes, explains, motivates
    'dependency_flow':         0.8,   # Prerequisites: depends_on, requires, blocks
    'contradiction_conflict':  0.75,  # Tensions (both sides matter): contradicts, challenges
    'validation_evidence':     0.7,   # Proof: validates, demonstrates, supports
    'temporal_sequence':       0.65,  # Event ordering: leads_to, follows_from
    'hierarchical_structure':  0.6,   # Composition: part_of, supersedes
    'modification_change':     0.6,   # Altering: modifies, revises, replaces
    'discovery_revelation':    0.55,  # Finding: reveals, discovered_alongside
    'similarity_complement':   0.5,   # Related: parallels, example_of
    'contextual_information':  0.4,   # Framing: contextualizes, informs
    'generic_relation':        0.3,   # Catch-all: related_to
}
DEFAULT_EDGE_WEIGHT = 0.3  # For relations not in any known family

CHAIN = {
    'min_chain_size': 2,           # Minimum nodes to count as a chain
    'max_chain_size': 12,          # Cap to prevent mega-chains
    'community_cooccurrence': 0.4, # Weight for "both in same community" (moderate signal)

    # Edges to exclude from chain detection (noise)
    'excluded_relations': {'co_accessed', 'emergent_bridge', 'community_member'},
}

# ═══════════════════════════════════════════════════
# STEP 4: Unified Ranking
# ═══════════════════════════════════════════════════

RANKING = {
    'final_limit': 25,             # Top N to return
    'relevance_floor': 0.20,       # Min final score to include
}
