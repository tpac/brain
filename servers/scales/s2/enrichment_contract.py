"""S2 Enrichment Unit — Contract and Configuration.

Generates V5 enrichment vectors (question, anchor, bridge, keywords) for nodes
that lack them. Uses Haiku with S0/S1 trace context for high-quality generation.

The enrichment gap: 1242 nodes (80% of recall misses) lack V5 vectors because
the enrichment pipeline was never wired to the encoding agent. This S2 unit
fixes the gap in batch and handles ongoing enrichment for new/revised nodes.
"""

ENRICHMENT = {
    # LLM config
    'model': 'claude-haiku-4-5',
    'max_tokens': 2048,

    # Batch sizing
    'max_nodes_per_call': 10,        # Nodes per Haiku call
    'max_nodes_per_run': 50,         # Total nodes per S2 run (graduated: increase over time)
    'min_edges_for_enrichment': 0,   # 0 = enrich even orphans (question + keywords don't need neighbors)

    # V5 vector types to generate
    'vectors_required': ['question', 'anchor', 'bridge', 'keywords'],

    # What triggers staleness (node needs re-enrichment)
    'staleness_triggers': ['revise', 'connect'],

    # Metadata flag key
    'needs_enrichment_key': 'needs_enrichment',

    # Trace query context
    'max_trace_queries': 3,          # Real queries from S1R traces to include in prompt
    'trace_lookback_hours': 720,     # 30 days of traces to search
}

# Edge family context for prompt: which families produce distinctive anchor/bridge text
ENRICHMENT_EDGE_FAMILIES = {
    'correction_improvement':  'corrects/improves',
    'extension_refinement':    'extends/refines',
    'explanation_causation':   'explains/causes',
    'dependency_flow':         'depends on/enables',
    'contradiction_conflict':  'contradicts/challenges',
    'validation_evidence':     'validates/demonstrates',
    'hierarchical_structure':  'part of/supersedes',
    'temporal_sequence':       'follows from/leads to',
}
