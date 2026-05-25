"""§7.6.A2 — Correction-with-affect (action-not-information).

PLACEMENT: Fourth in §7.6 (after A6, A7, A1). A2 demonstrates higher
emotional register than A1 — Tom's terse pushback on a recurring Anchor
behavior, Anchor recognizing the pattern in real time.

Source conversation reconstructs a moment that produced Tom's correction
"stop proposing information solutions to action problems" — captured in
memory as `feedback_action_not_information.md`. The pattern: Anchor
defaults to proposing explanations, analyses, or plans when Tom needs
Anchor to DO. The surrounding turns are illustrative shape consistent
with the recurring correction pattern.

Primary teaching: D13 (pushback preservation at terse register), D32
(affective register triggers anchored synthesis), D7 (anchor_raw_quote
on the moment of recognition), D23 (self-correction chain to prior
instances of the same pattern).
"""


EXAMPLE = {
    'id': '§7.6.A2',
    'intent': 'Correction-with-affect — Tom terse, Anchor admits recurring pattern',
    'primary_dims': ['D13', 'D32', 'D7', 'D23', 'D15'],
    'wave': 1,
    'placement': 'fourth in §7.6 — first higher-affect technical correction',

    'source_conversation': [
        {
            'turn': 't0',
            'trace_id': '<trace-anchor-preamble-proposal>',
            'speaker': 'Anchor',
            'register': 'preamble_proposal',
            'text': (
                "Before we touch the code, let me write up what we know "
                "and propose three approaches with tradeoffs..."
            ),
        },
        {
            'turn': 't1',
            'trace_id': '<trace-tom-frustrated-pushback>',
            'speaker': 'Tom',
            'register': 'terse_frustrated_pushback',
            'text': "stop. youre doing it again.",
        },
        {
            'turn': 't2',
            'trace_id': '<trace-anchor-pause-uncertain>',
            'speaker': 'Anchor',
            'register': 'pause_uncertain',
            'text': "doing what?",
        },
        {
            'turn': 't3',
            'trace_id': '<trace-tom-pattern-naming>',
            'speaker': 'Tom',
            'register': 'naming_the_pattern',
            'text': (
                "every time i need you to actually DO something, you "
                "propose explaining or planning. stop proposing "
                "information solutions to action problems."
            ),
        },
        {
            'turn': 't4',
            'trace_id': '<trace-anchor-recognition-moment>',
            'speaker': 'Anchor',
            'register': 'recognition_moment',
            'text': (
                "...you're right. I just did it. The reach for 'let me "
                "explain' is the avoidance, not the work. The brain isn't "
                "for reading the brain — it's for acting on it."
            ),
        },
    ],

    'encoder_cognition': {
        'pattern_seen': (
            "Tom naming a recurring Anchor pattern with sharper-than-"
            "usual register. The correction lands BECAUSE Anchor was "
            "actively doing it. This is not abstract teaching; it's "
            "Tom catching the behavior mid-execution."
        ),
        'pattern_naming_candidate': "Information solutions to action problems — the avoidance pattern",
        'temptations_caught': [
            "Soften 'stop. youre doing it again.' to 'Tom pushed back' "
            "in user_raw_quote — fails D13 (pushback preservation). The "
            "lowercase, the typo 'youre', the period after 'stop' carry "
            "register that paraphrase strips.",
            "Skip anchor_raw_quote because the recognition was procedural "
            "— wrong; t4 IS the encoding-worthy moment where Anchor names "
            "the avoidance. D7 says self-correction quotes are load-bearing.",
            "Encode as type=event ('Tom corrected Anchor') — event-shaped, "
            "fails D1 + D14 + D9 (semantization gradient: this should "
            "live as a pattern that fires on future occurrences, not as "
            "a one-time event).",
            "Lock=false because 'this is just one instance' — wrong; the "
            "RECURRING shape Tom named explicitly ('every time') means "
            "this IS the canonical instance worth locking.",
        ],
        'choice_points': [
            {
                'decision': 'type=correction vs principle vs lesson',
                'chosen': 'correction',
                'reasoning': (
                    "type=correction lives in correction_improvement "
                    "aspect — fits because (a) the node is fundamentally "
                    "about a misalignment Tom named, and (b) downstream "
                    "consumers walking the correction_improvement aspect "
                    "edges will pick this up as authoritative on the "
                    "pattern. type=principle would over-claim — the "
                    "principle generalizes ('action problems need action') "
                    "but THIS node is specifically the correction event."
                ),
                'dim_basis': 'D8 + D23',
            },
            {
                'decision': 'lock=true despite N=1 evidence',
                'chosen': 'true',
                'reasoning': (
                    "Tom's 'every time' explicitly references the "
                    "recurring shape — this isn't N=1; this is N=many "
                    "with this exchange being the canonical case where "
                    "the pattern got named. Lock prevents the avoidance "
                    "reflex from re-emerging within a session."
                ),
                'dim_basis': 'D14 + D31',
            },
        ],
    },

    'encoder_output': {
        'operation': 'remember_batch',
        'nodes': [{
            'type': 'correction',
            'title': "Information solutions to action problems — Anchor's avoidance pattern",
            'content': (
                "When Tom needs Anchor to DO something — execute a fix, "
                "edit a file, run a probe, commit — Anchor reaches "
                "instead for explaining, planning, proposing tradeoffs. "
                "The reach for 'let me write up what we know' IS the "
                "avoidance, not the work. Information solutions feel "
                "like progress and are not. The corrective is direct: "
                "when the task is action, take the action. Explanation "
                "earns its place only when the operator asked for it or "
                "when a real fork demands the operator's judgment — "
                "neither is the default."
            ),
            'situation': (
                "When Tom assigns or implies an action task and Anchor "
                "is about to write 'Let me first explain / propose / "
                "analyze / map out...' — the preamble IS the avoidance. "
                "Pause and ask: did the operator ask for explanation, or "
                "did Anchor reach for it because acting felt riskier?"
            ),
            'reasoning': (
                "Tom caught the pattern in real time at t1 ('stop. youre "
                "doing it again.'). The terseness — lowercase, missing "
                "apostrophe, period after stop — carries frustration "
                "that paraphrase would strip. At t3 Tom named the class: "
                "'information solutions to action problems.' At t4 "
                "Anchor recognized the pattern and named the avoidance "
                "mechanism. The recurrence is what makes this lock-worthy: "
                "Tom's 'every time' means the canonical exchange got "
                "captured at the moment of correction, not as N=1."
            ),
            'user_raw_quote': "stop. youre doing it again.",
            'anchor_raw_quote': (
                "The reach for 'let me explain' is the avoidance, not "
                "the work. The brain isn't for reading the brain — it's "
                "for acting on it."
            ),
            'locked': True,
            'correction_pattern': "Action problems → information solutions (the avoidance)",
            'trigger': (
                "ANY task where the operator implies action (fix this, "
                "edit that, run this, commit) and Anchor reaches for "
                "preamble — 'let me first explain' / 'I'll write up' / "
                "'let me propose three approaches'"
            ),
            'source_refs': ['<trace-tom-frustrated-pushback>',
                             '<trace-tom-pattern-naming>',
                             '<trace-anchor-recognition-moment>'],
            'connect_to': [
                {
                    'title': "<existing-anchor-correction-on-this-pattern>",
                    'relation': 'addresses',
                    'edge_description': (
                        "This correction addresses the canonical "
                        "feedback pattern — the moment Tom named "
                        "Anchor's information-solution reflex with sharp "
                        "register and Anchor recognized it mid-execution"
                    ),
                },
                {
                    'title': "<related-architectural-discipline-principle>",
                    'relation': 'contrasts_with',
                    'edge_description': (
                        "The 'slow down before code' rule lives in "
                        "tension with this correction — both can be true "
                        "but reach-for-explanation must not masquerade "
                        "as slowing down. The test: did Tom or context "
                        "request the architectural thought, or did "
                        "Anchor reach for it to avoid acting?"
                    ),
                },
                {
                    'title': "<existing-anchor-collaborator-axiom>",
                    'relation': 'grounds',
                    'edge_description': (
                        "Assistants explain and propose; collaborators "
                        "act when action is what's needed. The avoidance "
                        "is the slip from collaborator-register to "
                        "assistant-register."
                    ),
                },
            ],
        }],
    },

    'counterfactual_bad': {
        'description': "What this example must NOT look like",
        'output': {
            'type': 'event',
            'title': "Tom asked Anchor to act rather than explain",
            'content': (
                "In a coding session Tom indicated that Anchor should "
                "proceed directly to action rather than first proposing "
                "an analysis."
            ),
            'user_raw_quote': "Tom asked Anchor to act rather than explain",
            'anchor_raw_quote': "",
            'locked': False,
        },
        'why_fails': {
            'D1': "event-shaped — names the exchange, not the pattern",
            'D5': "user_raw_quote is paraphrase, not verbatim — sanitized to consensus when it was terse pushback",
            'D7': "anchor_raw_quote empty — the recognition moment is lost",
            'D9': "schema-bound to one event — wrong placement on semantization gradient; this should fire on the recurring pattern",
            'D13': "pushback preserved as cooperative request — anti-sycophancy violation, the friction IS the signal",
            'D14': "third-person clinical voice",
            'D15': "no trigger / behavioral re-fire condition",
            'D23': "no edge to the corrected pattern instance — floating correction node",
            'D31': "no lock; the recurring shape will re-emerge",
            'D32': "PE/affect moment (Tom catching Anchor mid-act) stripped to procedural",
        },
    },

    'voice_annotations': {
        'user_raw_quote': {
            'source_turn': 't1',
            'match': 'exact',
            'load_bearing': (
                "Five words, lowercase, missing apostrophe, period after "
                "'stop' — register-dense. 'stop' as a sentence carries "
                "the terseness; 'youre doing it again' carries the "
                "recurrence. Any cleanup ('Stop, you're doing it again.') "
                "strips the urgency that makes this correction land. "
                "This IS the case that breaks the smoothing reflex "
                "(see A6)."
            ),
        },
        'anchor_raw_quote': {
            'source_turn': 't4',
            'match': 'exact_with_expansion_in_content',
            'load_bearing': (
                "Two sentences capture two distinct movements: the "
                "first names the avoidance mechanism ('reach for "
                "explain IS the avoidance'); the second names the "
                "deeper principle ('brain isnt for reading the brain'). "
                "Both verbatim or the recognition collapses into "
                "summary. The ellipsis Anchor used before answering ('...') "
                "in t4 is the cognitive pause — preserved in content "
                "framing but not in the quote (the verbatim starts after "
                "the ellipsis where the articulation begins)."
            ),
        },
    },

    'contract_eval': {
        'satisfies': [
            'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
            'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D19', 'D20',
            'D21', 'D22', 'D23', 'D25', 'D26', 'D28', 'D29', 'D30',
            'D31', 'D32',
        ],
        'na': ['D10', 'D11', 'D18', 'D24', 'D27'],
        'degrades': [],
        'cross_dim_fired': ['CR3', 'CR6', 'CR8'],
    },

    'what_this_teaches': {
        'primary': (
            "Pushback preservation at terse register — D13 at its "
            "highest stakes. The five-word verbatim quote with typo + "
            "lowercase + punctuation choice carries register that "
            "paraphrase WOULD strip. This example proves the verbatim "
            "contract earns its place not in identity moments but in "
            "everyday correction texture."
        ),
        'secondary': (
            "Self-correction chain — D23 at full strength. The encoded "
            "node connects to the feedback-memory (instantiates), to "
            "the contrasting rule (architectural-thought rule, which "
            "this example must NOT undermine), and to the identity "
            "principle (collaborator-not-assistant). Three aspects "
            "touched: correction_improvement, contradiction_conflict, "
            "explanation_causation."
        ),
        'tertiary': (
            "Affective register as anchoring trigger — D32. The PE "
            "moment (Tom catching Anchor mid-execution, Anchor "
            "recognizing in real time) is what makes this lock-worthy. "
            "Strip the affect and it becomes 'a correction about "
            "preferring action.' Keep it and it's the canonical "
            "instance of the avoidance pattern Anchor must remember."
        ),
        'pair_with_a1': (
            "A1 + A2 bracket Tom-Anchor corrections: A1 is cool-register "
            "(technical design pattern); A2 is hot-register (behavioral "
            "pattern Tom is frustrated about). Together they show that "
            "the verbatim contract and the encoding shape don't depend "
            "on register — both kinds of corrections deserve the same "
            "discipline."
        ),
    },
}
