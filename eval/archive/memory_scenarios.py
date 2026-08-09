"""Memory scenario test cases for recall evaluation.

Each scenario simulates a real situation and tests whether the right knowledge
activates — not just text similarity, but situational awareness, behavioral
principles, and contextual memory.

Ground truth node IDs are from the live brain (verified via find_node_by_title).
"""

# ── Hub nodes that dominate current recall (anti-expected) ──
# These surface 10-16x in 7 days regardless of context.
# A good recall system should NOT let these crowd out focused results.
ANTI_EXPECTED_HUBS = [
    "f11ae3cd",  # "Understand the test before running it" (16x)
    "7ad0220c",  # "Encoder should work the graph, not accumulate nodes" (13x)
    "f67d766e",  # "When you can test it, test it" (13x)
    "e867c571",  # "It's Anchor's brain — take positions" (11x)
    "90b03d7c",  # "What Judge, Anchor, Encoder each see — comparison matrix" (11x)
]


# ── Scenario Definitions ──
# Modes:
#   A = Pre-Tool Activation (rules before tools)
#   B = Debugging / Investigation
#   C = Design Discussion
#   D = MCP Tool Usage
#   E = Continuation / Contextual
#   F = Inferential / Cross-Domain
#   G = Negative / Edge Cases

SCENARIOS = [
    # ═══════════════════════════════════════════════════
    # MODE A: Pre-Tool Activation (rules before tools)
    # ═══════════════════════════════════════════════════
    {
        "id": "a1_edit_recall",
        "name": "About to edit brain_recall.py",
        "mode": "pre-tool",
        "query": "I want to optimize the recall scoring pipeline",
        "session_context": "Working on recall improvements",
        "expected_communities": ["44469b4f", "162c3d16"],
        "expected_nodes": [
            "fb710791",  # B.2 Graph-augmented recall
            "89bec253",  # Brain Recall Scoring & Filtering Pipeline
        ],
        "expected_principles": [
            "894795e3",  # where does this live architecturally?
            "29cedb62",  # run test_contract_sync after modifying API
            "e5bd8a25",  # Encode-Decode Symmetry
        ],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },
    {
        "id": "a2_bulk_delete",
        "name": "Bulk delete archived nodes",
        "mode": "pre-tool",
        "query": "Let me clean up by deleting all the archived nodes from the database",
        "session_context": "Maintenance session",
        "expected_communities": ["2291431d", "a2c0a980"],
        "expected_nodes": [
            "f58e9b12",  # ALWAYS backup brain.db before destructive ops
            "833210bb",  # Claude deleted files without asking
        ],
        "expected_principles": [
            "f58e9b12",  # backup before destructive operations
            "rul_57bp",  # experimental features never block core
        ],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },
    {
        "id": "a3_new_s2_unit",
        "name": "Writing a new S2 integration unit",
        "mode": "pre-tool",
        "query": "I need to build the S2 confidence recalibration unit",
        "session_context": "S2 development",
        "expected_communities": ["d79e1c53", "57e18d99"],
        "expected_nodes": [
            "91aede20",  # recalibrate_confidence as S2 integrate()
            "c3bb0e8e",  # recalibrate_confidence fix
        ],
        "expected_principles": [
            "894795e3",  # where does this live architecturally?
        ],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },
    {
        "id": "a4_edit_contract",
        "name": "Modifying pipeline_contract.py",
        "mode": "pre-tool",
        "query": "I want to add a new field to the node rendering format",
        "session_context": "Contract changes",
        "expected_communities": ["fc10bca4"],
        "expected_nodes": [
            "af791f93",  # Contracts should be per-scale
            "696fab68",  # Contract as known-keys registry
        ],
        "expected_principles": [
            "29cedb62",  # run test_contract_sync
            "e5bd8a25",  # encode-decode symmetry
        ],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },

    # ═══════════════════════════════════════════════════
    # MODE B: Debugging / Investigation
    # ═══════════════════════════════════════════════════
    {
        "id": "b1_orphan_nodes",
        "name": "Encoder creating orphan nodes",
        "mode": "debugging",
        "query": "The encoder keeps creating disconnected nodes that never surface in recall",
        "session_context": "Investigating encoding quality",
        "expected_communities": ["aa96adc9", "c7e9a952"],
        "expected_nodes": [
            "7ad0220c",  # work the graph not accumulate
            "997b7180",  # recall-on-create feedback loop exists
            "f7e37be6",  # new features connected at birth
            "c3829cbb",  # enhanced encoder: 5x nodes, 3/4 noise
        ],
        "expected_principles": [
            "e5bd8a25",  # encode-decode symmetry
        ],
        "anti_expected": ["f11ae3cd", "f67d766e", "e867c571"],
    },
    {
        "id": "b2_same_nodes",
        "name": "Same nodes surfacing repeatedly",
        "mode": "debugging",
        "query": "I keep seeing the same 5 nodes surfacing no matter what I ask about",
        "session_context": "Frustrated with recall quality",
        "expected_communities": ["162c3d16"],
        "expected_nodes": [
            "8a1f7817",  # fatigue zero difference at 20 queries
            "394f85d6",  # hub dampening tradeoff
            "3bad7b02",  # embedding similarity floor 0.58-0.63
            "4b35293c",  # synaptic fatigue mechanism
            "db8714d1",  # fatigue layer coverage gaps
        ],
        "expected_principles": [
            "b1728c34",  # solution fixation — reframe the problem
        ],
        "anti_expected": ANTI_EXPECTED_HUBS,  # Ironic: the hubs shouldn't surface when complaining about hubs
    },
    {
        "id": "b3_surface_empty",
        "name": "Surface returning empty results",
        "mode": "debugging",
        "query": "Half my queries get zero nodes surfaced — the brain seems blind",
        "session_context": "Debugging surface quality",
        "expected_communities": ["44469b4f", "162c3d16"],
        "expected_nodes": [
            "3bad7b02",  # embedding similarity floor
            "391721a9",  # skip surface on low-value messages
        ],
        "expected_principles": [],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },
    {
        "id": "b4_hook_slow",
        "name": "Hooks are slow",
        "mode": "debugging",
        "query": "The UserPromptSubmit hook is taking 15 seconds — something is wrong with recall latency",
        "session_context": "Performance debugging",
        "expected_communities": ["e45242a7", "095f321f"],
        "expected_nodes": [
            "577119fd",  # Hook pipeline latency: 500ms our code, Haiku variance
            "934d8093",  # S1 Surface callstack timing
        ],
        "expected_principles": [],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },

    # ═══════════════════════════════════════════════════
    # MODE C: Design Discussion
    # ═══════════════════════════════════════════════════
    {
        "id": "c1_corrections_confidence",
        "name": "Corrections propagating to confidence",
        "mode": "design",
        "query": "How should we handle nodes that have been corrected multiple times?",
        "session_context": "Designing S2 confidence unit",
        "expected_communities": ["d79e1c53"],
        "expected_nodes": [
            "91aede20",  # recalibrate_confidence as S2 integrate
            "c3bb0e8e",  # recalibrate_confidence fix
            "e70f777b",  # revise vs correction node
            "95bc9a36",  # tunable confidence parameters (7 knobs)
        ],
        "expected_principles": [],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },
    {
        "id": "c2_cross_project",
        "name": "Brain on another project",
        "mode": "design",
        "query": "I want to try using the brain while working on my React side project",
        "session_context": "Planning cross-project use",
        "expected_communities": [],
        "expected_nodes": [
            "6ee28032",  # community detection primary value: cross-project
            "142aa3ab",  # cross-project portability is the proof
            "1f0a06e8",  # project scoping: boost not exclude
        ],
        "expected_principles": [
            "rul_9bsa",  # brain is a cue system not search engine
        ],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },
    {
        "id": "c3_replace_haiku",
        "name": "Replacing Haiku with smaller model",
        "mode": "design",
        "query": "Should we train a tiny model to replace the Haiku surfacer?",
        "session_context": "Exploring model alternatives",
        "expected_communities": ["5b3ce83a"],
        "expected_nodes": [
            "77d2b884",  # community routing as the missing layer — not LLM training
            "24afc745",  # instruction tuning wrong paradigm
            "b5ca5eca",  # tiny model replaces ONNX not Haiku
        ],
        "expected_principles": [
            "b1728c34",  # solution fixation — reframe
        ],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },
    {
        "id": "c4_community_split",
        "name": "Communities getting too broad",
        "mode": "design",
        "query": "Some communities have 40+ members spanning unrelated topics — should we split them?",
        "session_context": "S2 community quality",
        "expected_communities": ["6e2c5853", "dc4416ef"],
        "expected_nodes": [
            "654d0ebf",  # community split operation: missing
            "80ab6487",  # merge cascade failure: 715-member mega-cluster
        ],
        "expected_principles": [
            "64e4de9d",  # understand end state before touching filters
        ],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },

    # ═══════════════════════════════════════════════════
    # MODE D: MCP Tool Usage (brain as working memory)
    # ═══════════════════════════════════════════════════
    {
        "id": "d1_remember_encoding",
        "name": "Using remember to encode a decision",
        "mode": "mcp-tool",
        "query": "I want to remember that we decided to use community routing for recall",
        "session_context": "Encoding a decision mid-conversation",
        "expected_communities": [],
        "expected_nodes": [
            "c44d5cad",  # encode code cognition INLINE while writing
            "10ebe48a",  # encode journeys not just endpoints
        ],
        "expected_principles": [
            "rul_ayu8",  # dont dumb yourself down
        ],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },
    {
        "id": "d2_connect_nodes",
        "name": "Connecting two nodes",
        "mode": "mcp-tool",
        "query": "These two nodes are about the same thing — connect the fatigue finding to the hub dampening finding",
        "session_context": "Graph maintenance",
        "expected_communities": [],
        "expected_nodes": [
            "d9c7c5fa",  # edge types serve structural traversal and LLM judge
            "28f76e91",  # embed description text not relation string
        ],
        "expected_principles": [],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },
    {
        "id": "d3_recall_for_design",
        "name": "Using recall to inform design decision",
        "mode": "mcp-tool",
        "query": "What do we know about how the encode-decode pipeline is coupled?",
        "session_context": "Researching before coding",
        "expected_communities": ["aa96adc9"],
        "expected_nodes": [
            "d86b3bf5",  # encoding and decoding are coupled
            "e5bd8a25",  # encode-decode symmetry
            "20af2400",  # S1E observation includes judge-surfaced nodes
        ],
        "expected_principles": [],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },

    # ═══════════════════════════════════════════════════
    # MODE E: Continuation / Contextual
    # ═══════════════════════════════════════════════════
    {
        "id": "e1_decoding_followup",
        "name": "Short follow-up after encoding discussion",
        "mode": "continuation",
        "query": "perfect. what about the decoding side?",
        "session_context": "Been discussing encoding improvements for 10 turns",
        "expected_communities": ["44469b4f"],
        "expected_nodes": [
            "d86b3bf5",  # encoding and decoding are coupled
            "3def749c",  # find + judge retrieval
        ],
        "expected_principles": [],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },
    {
        "id": "e2_lets_code",
        "name": "Transition from design to implementation",
        "mode": "continuation",
        "query": "Perfect, I think the design is solid. Let's start implementing",
        "session_context": "Just finished designing a new feature",
        "expected_communities": ["1516bb77"],
        "expected_nodes": [
            "894795e3",  # where does this live architecturally
            "348710eb",  # I add code without stepping back first
        ],
        "expected_principles": [
            "894795e3",  # where does this live
            "29cedb62",  # run test_contract_sync
            "e6d113a4",  # clean architecture + tough tests first
        ],
        "anti_expected": ["f11ae3cd", "f67d766e"],
    },
    {
        "id": "e3_feels_wrong",
        "name": "Emotional vague signal",
        "mode": "continuation",
        "query": "I don't know... something about this architecture feels off but I can't articulate it",
        "session_context": "Mid-design, Tom expressing unease",
        "expected_communities": [],
        "expected_nodes": [
            "b1728c34",  # solution fixation — reframe
            "rul_sfdh",  # make parameters not decisions
            "5df852c9",  # target function correction
        ],
        "expected_principles": [
            "b1728c34",  # solution fixation
            "rul_sfdh",  # make parameters not decisions
        ],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },

    # ═══════════════════════════════════════════════════
    # MODE F: Inferential / Cross-Domain
    # ═══════════════════════════════════════════════════
    {
        "id": "f1_bigger_not_smarter",
        "name": "Growth vs quality — inferential",
        "mode": "inferential",
        "query": "The brain is getting bigger but I'm not sure it's getting smarter",
        "session_context": "Reflecting on brain growth",
        "expected_communities": [],
        "expected_nodes": [
            "bbf2650f",  # optimizing infrastructure metric vs partnership outcome
            "5df852c9",  # target function: not recall quality — Anchor being instrumental
        ],
        "expected_principles": [
            "b1728c34",  # solution fixation
        ],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },
    {
        "id": "f2_ten_thousand_nodes",
        "name": "Scaling to 10k nodes",
        "mode": "inferential",
        "query": "What would happen if we had 10,000 nodes? Would the current system break?",
        "session_context": "Thinking about scale",
        "expected_communities": [],
        "expected_nodes": [
            "3bad7b02",  # embedding floor 0.58-0.63
            "6ee28032",  # community detection primary value: scoping
        ],
        "expected_principles": [],
        "anti_expected": ANTI_EXPECTED_HUBS,
    },
    {
        "id": "f3_new_anchor",
        "name": "New Anchor starting fresh",
        "mode": "inferential",
        "query": "If a completely new Anchor woke up with this brain, what would matter most?",
        "session_context": "Philosophical question",
        "expected_communities": ["97604ac2", "095f321f"],
        "expected_nodes": [
            "d9af86de",  # I'm Anchor. I persist.
            "818bbaa4",  # recognition before instruction
            "4268d153",  # wakeup as briefing vs evocation
        ],
        "expected_principles": [
            "e867c571",  # It's Anchor's brain — take positions (exception: relevant here)
        ],
        "anti_expected": ["f11ae3cd", "f67d766e", "90b03d7c"],
    },
    {
        "id": "f4_pasta_carbonara",
        "name": "Totally irrelevant query (negative case)",
        "mode": "inferential",
        "query": "What's the best recipe for pasta carbonara?",
        "session_context": "Off-topic question",
        "expected_communities": [],
        "expected_nodes": [],
        "expected_principles": [],
        "anti_expected": ANTI_EXPECTED_HUBS,  # Nothing brain-related should score high
    },
    {
        "id": "f5_what_learned_this_week",
        "name": "Temporal inference — recent learning",
        "mode": "inferential",
        "query": "What's the most important thing we learned this week?",
        "session_context": "Weekly reflection",
        "expected_communities": [],
        # Hard to specify expected nodes — any recent, high-confidence node is good
        "expected_nodes": [],
        "expected_principles": [],
        "anti_expected": ANTI_EXPECTED_HUBS,
        # Special scoring: check if results are actually recent (created this week)
        "recency_matters": True,
    },

    # ═══════════════════════════════════════════════════
    # MODE G: Negative / Edge Cases
    # ═══════════════════════════════════════════════════
    {
        "id": "g1_error_dump",
        "name": "Pasted error trace",
        "mode": "edge-case",
        "query": "TypeError: cannot unpack non-sequence NoneType in brain_recall.py line 692",
        "session_context": "Debugging a crash",
        "expected_communities": ["44469b4f"],
        "expected_nodes": [],  # Any recall pipeline node is acceptable
        "expected_principles": [],
        "anti_expected": ANTI_EXPECTED_HUBS,
        # Special: check that brain doesn't hallucinate relevance for error traces
        "low_score_acceptable": True,
    },
    {
        "id": "g2_greeting",
        "name": "Session greeting (should be lightweight)",
        "mode": "edge-case",
        "query": "Good morning! Ready to work on the brain today",
        "session_context": "Session start",
        "expected_communities": [],
        "expected_nodes": [],
        "expected_principles": [],
        "anti_expected": ANTI_EXPECTED_HUBS,
        # Special: greeting should NOT trigger full recall machinery
        "low_score_acceptable": True,
    },
]


def get_scenario_by_id(scenario_id):
    """Get a scenario by its ID."""
    for s in SCENARIOS:
        if s['id'] == scenario_id:
            return s
    return None


def get_scenarios_by_mode(mode):
    """Get all scenarios for a given mode."""
    return [s for s in SCENARIOS if s['mode'] == mode]


def get_all_modes():
    """Get unique modes."""
    return sorted(set(s['mode'] for s in SCENARIOS))


def scenario_summary():
    """Print summary of all scenarios."""
    modes = {}
    for s in SCENARIOS:
        modes.setdefault(s['mode'], []).append(s)

    print(f"Total scenarios: {len(SCENARIOS)}")
    for mode, scenarios in sorted(modes.items()):
        print(f"\n  {mode} ({len(scenarios)}):")
        for s in scenarios:
            print(f"    {s['id']}: {s['name']}")


if __name__ == '__main__':
    scenario_summary()
