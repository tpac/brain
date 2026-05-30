"""S2 Healer — Contract and Configuration.

Generates missing findability fields (question, anchor, bridge, keywords) for
nodes that lack them. Uses Haiku with S0/S1 trace context for high-quality
generation.

The healer gap: nodes created before the current field contract, or by paths
that skipped field population, end up missing findability context. This unit
scans for those gaps and fills them in batch; ongoing healing also fires on
revise/connect so the graph stays dense.
"""

HEALER = {
    # LLM config
    'model': 'claude-haiku-4-5',
    'max_tokens': 2048,

    # Batch sizing
    'max_nodes_per_call': 10,        # Nodes per Haiku call
    'max_nodes_per_run': 50,         # Total nodes per S2 run (graduated: increase over time)
    'min_edges_for_healing': 0,      # 0 = heal even orphans (question + keywords don't need neighbors)

    # Field types to generate
    'fields_required': ['question', 'anchor', 'bridge', 'keywords'],

    # What triggers staleness (node needs re-healing)
    'staleness_triggers': ['revise', 'connect'],

    # Metadata flag key
    'needs_healing_key': 'needs_healing',

    # Trace query context
    'max_trace_queries': 3,          # Real queries from S1R traces to include in prompt
    'trace_lookback_hours': 720,     # 30 days of traces to search
}

# HEALER_EDGE_FAMILIES — REMOVED 2026-05-04 (Step 10 of unified-aspects).
# Was a dict of {family_name: display_label} for the healer prompt. Per a
# pre-removal grep, nothing imported it — the dict was already dead code.
# Display-label data now lives on each aspect's metadata (from aspects_v1.json):
#     brain.aspects.correction_improvement.metadata.get('display_label')
#       → 'corrects/improves'
# Healer prompts that want the labels back should iterate brain.aspects.all()
# and read each aspect's metadata.
