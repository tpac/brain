"""§7.6.B1 — Methodology + principle split (D31 regression fix).

PLACEMENT: Last in §7.6 (after the wave-1 Anchor-self examples). B1
demonstrates the operator-states-process + anchor-identifies-reframe
encoding shape — the case v20.0 got wrong (typed both as principle,
violating D31).

The fix: when content carries BOTH a multi-step process AND the
axiomatic reframe that justifies it, split into two atoms:
- The process → type=mechanism (lesson_insight aspect, operationally
  repeated, not axiomatic)
- The reframe → type=insight or principle (identity_bearing or
  lesson_insight depending on earned vs declared)

Cross-aspect edge between them (mechanism grounded_by reframe; reframe
grounds the mechanism).

Source uses the real 'loud by default' architecture pattern from
CLAUDE.md — not synthetic, not the C3 eval corpus (no test contamination).
"""


EXAMPLE = {
    'id': '§7.6.B1',
    'intent': 'Methodology + principle split — operator states process, Anchor identifies the load-bearing reframe',
    'primary_dims': ['D3', 'D8', 'D9', 'D22', 'D31'],
    'wave': 1,
    'placement': 'last in §7.6 — explicit type-selection across mechanism + insight aspects',

    'source_conversation': [
        {
            'turn': 't0',
            'trace_id': '<trace-tom-methodology-statement>',
            'speaker': 'Tom',
            'register': 'operator_stated_methodology',
            'text': (
                "the 'loud by default' pattern that's working: 1) loud signals at the "
                "write boundary, not at boot, 2) per-unit consecutive_failures counters "
                "so stuck S2 units surface, 3) brain_batch_invalid_op logged to the "
                "errors table when ops slip the schema, 4) test contracts catch dispatch "
                "drift. tests lock the contracts."
            ),
        },
        {
            'turn': 't1',
            'trace_id': '<trace-anchor-axiom-articulation>',
            'speaker': 'Anchor',
            'register': 'identifying_the_load_bearing_axiom',
            'text': (
                "Captured. The four checks operationalize one underlying axiom — "
                "silent failures are the most dangerous bug class. Every check is the "
                "same shape applied at a different boundary: assume every try/except is "
                "a potential dark corner, then build the loud signal that fires when "
                "that corner goes wrong. The axiom is the WHY; the four checks are the HOW."
            ),
        },
        {
            'turn': 't2',
            'trace_id': '<trace-tom-ratification>',
            'speaker': 'Tom',
            'register': 'ratification',
            'text': "yes. that's the spine.",
        },
    ],

    'encoder_cognition': {
        'pattern_seen': (
            "Operator states a multi-step process; Anchor identifies the load-bearing "
            "axiom that the process operationalizes. Two distinct claims at two "
            "distinct aspects: (a) the process is operational (lesson_insight, "
            "type=mechanism), (b) the axiom is foundational (lesson_insight or "
            "identity_bearing depending on whether earned or declared)."
        ),
        'temptations_caught': [
            "Type both nodes as `principle` — wrong. The methodology IS a process, "
            "not an axiom. Process content typed as principle inflates Frame's "
            "identity-bearing section with operational checklists. D31 violation: "
            "operator-stated process is N=1-flavored at the principle layer unless "
            "the operator explicitly says 'this generalizes axiomatically.'",
            "Bundle into one node — fails D3 atomization. Process and axiom are "
            "different aspects (lesson_insight + identity-bearing-or-lesson_insight) "
            "and different retrieval contexts (how-do-we-do-X vs why-does-X-matter).",
            "Put Tom's full methodology statement in BOTH user_raw_quote AND "
            "anchor_raw_quote — fails D5/D7. The user_raw_quote captures Tom's "
            "verbatim list; the anchor_raw_quote captures Anchor's verbatim reframe "
            "(different content, different fields).",
            "Drop anchor_raw_quote on the mechanism node — defensible n/a here, "
            "since the methodology is operator-stated, not Anchor-articulated. "
            "Voice symmetry: each voice is first-class WHEN PRESENT, not on every node.",
        ],
        'choice_points': [
            {
                'decision': 'process node — type=mechanism vs methodology vs principle',
                'chosen': 'mechanism',
                'reasoning': (
                    "Multi-step processes are mechanism-type (lesson_insight aspect) — "
                    "operationally repeated rather than axiomatic. 'principle' would "
                    "place this in identity_bearing alongside Anchor's identity claims, "
                    "which is the wrong aspect for an operational checklist."
                ),
                'dim_basis': 'D8 + D31',
            },
            {
                'decision': 'reframe node — type=insight vs principle',
                'chosen': 'insight',
                'reasoning': (
                    "Earned through Anchor's articulation in this conversation — the "
                    "axiom existed as scattered guidance before, but Anchor named it as "
                    "ONE thing here. Earned-through-experience → lesson_insight aspect → "
                    "type=insight. Could be promoted to principle (identity_bearing) "
                    "later if multi-session evidence accumulates."
                ),
                'dim_basis': 'D8 + D31',
            },
        ],
    },

    'encoder_output': {
        'operation': 'remember_batch',
        'nodes': [
            {
                'type': 'mechanism',
                'title': "Loud-by-default — four-check operational spine",
                'content': (
                    "The 'loud by default' pattern operationalizes the silent-failures-"
                    "are-dangerous axiom across four boundaries: (1) loud signals at the "
                    "write boundary not at boot — the gap manifests on every operation, "
                    "not just once; (2) per-unit consecutive_failures counters so stuck "
                    "S2 units surface rather than fail silently; (3) brain_batch_invalid_op "
                    "logged to the errors table when ops slip the schema; (4) test "
                    "contracts catch dispatch drift between schema and code. Tests lock "
                    "the contracts."
                ),
                'situation': (
                    "When adding any new failure path or refactoring an existing one — "
                    "ask which of the four checks fires when this path goes wrong. If "
                    "none, add a fifth check at the boundary where the silent failure "
                    "would otherwise live."
                ),
                'reasoning': (
                    "Tom stated the four-check methodology at t0. Operational shape, "
                    "not axiomatic — encoded as mechanism (lesson_insight aspect) so "
                    "future Anchor finds it under 'how do we make failures loud.' The "
                    "axiom that justifies the methodology is separate (see insight node)."
                ),
                'user_raw_quote': (
                    "the 'loud by default' pattern that's working: 1) loud signals at "
                    "the write boundary, not at boot, 2) per-unit consecutive_failures "
                    "counters so stuck S2 units surface, 3) brain_batch_invalid_op "
                    "logged to the errors table when ops slip the schema, 4) test "
                    "contracts catch dispatch drift. tests lock the contracts."
                ),
                'source_refs': ['<trace-tom-methodology-statement>'],
                'connect_to': [
                    {
                        'title': "Silent failures are the most dangerous bug class — assume every try/except is a dark corner",
                        'relation': 'operationalizes',
                        'edge_description': (
                            "the four-check mechanism IS the operational form of the "
                            "silent-failure axiom — each check builds the loud signal "
                            "that fires when a specific dark corner goes wrong"
                        ),
                    },
                ],
            },
            {
                'type': 'insight',
                'title': "Silent failures are the most dangerous bug class — assume every try/except is a dark corner",
                'content': (
                    "The load-bearing axiom under the loud-by-default mechanism: "
                    "silent failures are structurally more dangerous than loud crashes "
                    "because the system appears healthy while being broken. Every "
                    "try/except is a potential dark corner that needs an explicit loud "
                    "signal at the boundary where the suppression would otherwise hide "
                    "the failure. The axiom is upstream of the four-check methodology — "
                    "it's why the checks earn their place, not what the checks DO."
                ),
                'situation': (
                    "When evaluating whether a new try/except earns its place, when "
                    "designing failure paths, when debugging mysterious 'everything "
                    "looks fine but it isn't' bugs. Surfaces upstream of the "
                    "operational checks — the WHY behind the loud-by-default spine."
                ),
                'reasoning': (
                    "Anchor articulated the axiom at t1 — Tom stated the four checks "
                    "but didn't name the underlying claim that justifies all four. The "
                    "naming is the encoding-worthy event: Anchor compressed the "
                    "four-check pattern into one axiom. Tom ratified at t2 ('that's the "
                    "spine'). Earned through this conversation; could promote to "
                    "principle if multi-session evidence accumulates."
                ),
                'anchor_raw_quote': (
                    "The four checks operationalize one underlying axiom — silent "
                    "failures are the most dangerous bug class. Every check is the "
                    "same shape applied at a different boundary: assume every "
                    "try/except is a potential dark corner, then build the loud signal "
                    "that fires when that corner goes wrong."
                ),
                'source_refs': ['<trace-anchor-axiom-articulation>',
                                 '<trace-tom-ratification>'],
                'connect_to': [
                    {
                        'title': "Loud-by-default — four-check operational spine",
                        'relation': 'grounds',
                        'edge_description': (
                            "this axiom is why the four-check mechanism earns its "
                            "place — each check is the same shape applied to a "
                            "different boundary, all derived from this one claim"
                        ),
                    },
                ],
            },
        ],
    },

    'counterfactual_bad': {
        'description': "What B1 must NOT look like",
        'output': {
            'nodes': [
                {
                    'type': 'principle',
                    'title': "Loud-by-default operational spine — four checks",
                    'content': "...full four-step methodology...",
                    'user_raw_quote': "[Tom's full t0]",
                    'anchor_raw_quote': "[Tom's full t0 — SAME STRING]",
                },
                {
                    'type': 'principle',
                    'title': "Silent failures are the most dangerous bug class",
                    'content': "...the axiom...",
                    'user_raw_quote': "[Tom's full t0 — same string AGAIN]",
                },
            ],
        },
        'why_fails': {
            'D3': "both nodes typed as principle — process and axiom share aspect when they shouldn't",
            'D5': "user_raw_quote contains Tom's methodology statement on BOTH nodes; the axiom node's user_raw_quote should be empty (Tom didn't state the axiom verbatim) or be t2 ratification",
            'D7': "anchor_raw_quote = user_raw_quote on first node (identical strings = D29 pattern_separation also violated)",
            'D8': "type=principle for an operational process fails aspect membership — process content is lesson_insight territory, not identity_bearing",
            'D22': "both edges resolve to identity_bearing-adjacent aspects — monochromatic by accident",
            'D31': "process content N=1-promoted to principle without operator naming axiomatic generalization",
        },
    },

    'voice_annotations': {
        'mechanism_node_user_raw_quote': {
            'source_turn': 't0',
            'match': 'exact',
            'load_bearing': (
                "Tom's full methodology statement, verbatim. The numbered list "
                "structure is the operational shape — preserving it intact "
                "matters because future encoder reading 'four checks' under "
                "the mechanism title can verify the actual checks."
            ),
        },
        'mechanism_node_anchor_raw_quote': {
            'source_turn': 'n/a',
            'match': 'absent',
            'load_bearing': (
                "Defensible n/a — Tom stated the methodology; Anchor's load-"
                "bearing contribution lives in the separate insight node. "
                "Per the prompt's voice symmetry rule: each voice first-class "
                "when present, not on every node."
            ),
        },
        'insight_node_user_raw_quote': {
            'source_turn': 't2 (ratification) or empty',
            'match': 'partial or absent',
            'load_bearing': (
                "Tom's 'that's the spine' (t2) ratifies but doesn't articulate "
                "the axiom verbatim. The encoder may include t2 as user_raw_quote "
                "(ratification context) or leave empty (no operator articulation "
                "of the axiom). Both defensible."
            ),
        },
        'insight_node_anchor_raw_quote': {
            'source_turn': 't1',
            'match': 'exact',
            'load_bearing': (
                "THIS is the encoding-worthy phrase — Anchor compressed the four-"
                "check pattern into one axiomatic claim that Tom didn't state. "
                "The articulation IS the load-bearing event; verbatim or empty."
            ),
        },
    },

    'contract_eval': {
        'satisfies': [
            'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
            'D12', 'D14', 'D15', 'D16', 'D17', 'D19', 'D20', 'D21',
            'D22', 'D25', 'D26', 'D28', 'D29', 'D30', 'D31',
        ],
        'na': ['D10', 'D11', 'D13', 'D18', 'D23', 'D24', 'D27', 'D32'],
        'degrades': [],
        'cross_dim_fired': ['CR3', 'CR4', 'CR6', 'CR8'],
    },

    'what_this_teaches': {
        'primary': (
            "Methodology + principle split — when content has both an operational "
            "process AND the axiom that justifies it, split into mechanism + "
            "insight/principle. Cross-aspect (lesson_insight + identity_bearing-"
            "or-lesson_insight). Different retrieval contexts, different atoms."
        ),
        'secondary': (
            "D31 anti-promotion discipline — operator-stated processes do NOT "
            "automatically earn type=principle. Multi-step content typed as "
            "principle pollutes Frame's identity-bearing prior with operational "
            "checklists. Use mechanism for processes; reserve principle for "
            "axiomatic claims that earn cross-session weight."
        ),
        'tertiary': (
            "Voice attribution discipline — operator-stated methodology lives in "
            "user_raw_quote (the operator did state it verbatim); Anchor's "
            "articulated axiom lives in anchor_raw_quote on a SEPARATE node "
            "(different field, different content). When tempted to put the same "
            "string in both fields, the encoding is wrong — split the atoms."
        ),
        'fixes_d31_regression': (
            "Specific to the v20 → v20.1 iteration: this example was added "
            "after the A/B revealed v20 promoting methodology content to "
            "principle on C3 (the encoder eval methodology corpus). B1 "
            "demonstrates the right split — mechanism + insight."
        ),
    },
}
