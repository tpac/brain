"""§7.6.A1 — Technical pattern with sparse anchored evidence.

PLACEMENT: Third in §7.6 (after A6 + A7 — Anchor self-examples).
A1 demonstrates pattern_naming as title craft on a technical correction;
extends the partnership/identity pair into the bulk of daily encoding work.

Primary teaching: axis A (title-as-handle) — compress the moment into a
pattern-shaped noun phrase that surfaces across queries. The example is
real-derived from Tom-Anchor work on daemon prompt-delivery design
(rejected flag-file boot protocol). The surrounding turns are illustrative
shape consistent with the actual correction pattern recorded in brain.
"""


EXAMPLE = {
    'id': '§7.6.A1',
    'intent': 'Pattern naming on a technical correction — axis A primary',
    'primary_dims': ['D1', 'D32', 'D14', 'D7'],
    'wave': 1,
    'placement': 'third in §7.6 — after A6 + A7 Anchor self-examples',

    'source_conversation': [
        {
            'turn': 't0',
            'trace_id': 71401,
            'speaker': 'Tom',
            'register': 'terse_directive',
            'text': (
                "no don't use a flag file, have the daemon return the "
                "prompt directly"
            ),
        },
        {
            'turn': 't1',
            'trace_id': 71402,
            'speaker': 'Anchor',
            'register': 'self_correction_with_missed_failure_mode',
            'text': (
                "You're right. I was about to wire a flag file the boot "
                "script would write and the daemon would poll. The failure "
                "mode I was missing is staleness — if the file gets "
                "written, then the daemon reads stale content from a prior "
                "write, we get the wrong prompt with no error."
            ),
        },
        {
            'turn': 't2',
            'trace_id': 71403,
            'speaker': 'Anchor',
            'register': 'pattern_articulation_reframe',
            'text': (
                "Control-by-request rather than control-by-inspection — "
                "the authority decides AND ships the answer in one round. "
                "Callers can't read the wrong thing if they never read."
            ),
        },
        {
            'turn': 't3',
            'trace_id': 71404,
            'speaker': 'Tom',
            'register': 'confirmation',
            'text': "yes exactly",
        },
    ],

    'encoder_cognition': {
        'pattern_seen': (
            "Tom rejected a specific design (flag file) — but the reframe "
            "Anchor named generalizes the lesson beyond this one decision. "
            "Encoding-worthy is the pattern, not the rejection event."
        ),
        'pattern_naming_candidate': "Control-by-request beats control-by-inspection",
        'temptations_caught': [
            "Title 'Tom rejected flag-file boot design' — event-shaped, "
            "names the moment instead of the lesson. Fails D1.",
            "Type=correction — too narrow; principle generalizes beyond "
            "this design choice to any read-modify-write boundary.",
            "Skip anchor_raw_quote because 'this is technical, not "
            "identity' — wrong; D7 says Anchor's reframe IS the encoding-"
            "worthy thing on corrections like this.",
        ],
        'choice_points': [
            {
                'decision': 'type=principle vs lesson vs correction',
                'chosen': 'principle',
                'reasoning': (
                    "Content names a generalizable pattern that holds "
                    "across IPC, shared state, lock-free coordination — "
                    "not just this one design. Lesson is too small; "
                    "correction is too event-bound. Principle reads "
                    "identity_bearing OR lesson_insight depending on "
                    "evidence base — here it's earned from one moment but "
                    "the generalization in reasoning argues for principle."
                ),
                'dim_basis': 'D8 + D31',
            },
            {
                'decision': 'source_refs to t0 + t2 (sparse)',
                'chosen': '2 refs',
                'reasoning': (
                    "Sparse anchoring (D26): t0 is the rejection that sets "
                    "stakes; t2 is the reframe that articulated the "
                    "principle. t1 (working-through) is context, not load-"
                    "bearing. t3 (Tom confirming) is validation but not "
                    "the moment-of-encoding."
                ),
                'dim_basis': 'D25 + D26',
            },
        ],
    },

    'encoder_output': {
        'operation': 'remember_batch',
        'nodes': [{
            'type': 'principle',
            'title': "Control-by-request beats control-by-inspection",
            'content': (
                "The authority decides AND ships the answer in one round; "
                "callers don't read external state to discover it. "
                "Eliminates the stale-flag failure mode entirely — callers "
                "can't read the wrong thing if they never read. "
                "Generalizes beyond inter-process coordination: any shared "
                "lock-free read-modify-write boundary has the same "
                "invariant. The reader can't know how stale the writer's "
                "snapshot is; eliminate the read instead of guarding it."
            ),
            'situation': (
                "When tempted to write a 'current state' flag file, "
                "polling mechanism, or any design where one process "
                "broadcasts state for another to inspect. The reach for "
                "shared-state coordination IS the failure mode — staleness "
                "is invisible until production. Pause and ask: can the "
                "authority deliver the answer directly instead of "
                "publishing it for the caller to fetch?"
            ),
            'reasoning': (
                "Tom rejected the flag-file design directly in t0 — terse "
                "pushback, no hedge. The reframe Anchor named in t2 "
                "generalizes the lesson: race conditions felt obvious, "
                "staleness did not. The principle holds across any read-"
                "modify-write boundary where the reader cannot know how "
                "stale the snapshot is. Tom's 'yes exactly' in t3 "
                "confirmed the generalization landed."
            ),
            'user_raw_quote': (
                "no don't use a flag file, have the daemon return the "
                "prompt directly"
            ),
            'anchor_raw_quote': (
                "Control-by-request rather than control-by-inspection — "
                "the authority decides AND ships the answer in one round. "
                "Callers can't read the wrong thing if they never read."
            ),
            'correction_pattern': "Inspection-based coordination → request-based coordination",
            'trigger': (
                "designing inter-process coordination, polling, broadcast "
                "flags, or any read-side dependency on a writer's state"
            ),
            'source_refs': [71401, 71403],
            'connect_to': [
                {
                    'title': "Daemon TCP migration",
                    'relation': 'grounds',
                    'edge_description': (
                        "The single-writer authority pattern is what let "
                        "TCP migration stay simple — no shared file state "
                        "to reconcile across processes"
                    ),
                },
                {
                    'title': "Flag-file boot protocol (rejected design)",
                    'relation': 'supersedes',
                    'edge_description': (
                        "Names the specific design this principle ruled "
                        "out — flag file the daemon would poll for boot "
                        "prompt content"
                    ),
                },
                {
                    'title': "Single-writer invariant — locks were addressing the wrong question",
                    'relation': 'parallels',
                    'edge_description': (
                        "Same shape at a different layer: locks address "
                        "concurrent-write contention; control-by-request "
                        "addresses concurrent-read staleness. Both: "
                        "eliminate the shared mutable state instead of "
                        "guarding it."
                    ),
                },
            ],
        }],
    },

    'counterfactual_bad': {
        'description': "What this example must NOT look like",
        'output': {
            'type': 'correction',
            'title': "Tom rejected flag-file design for boot prompt delivery",
            'content': (
                "In a design discussion about how the boot script and "
                "daemon should coordinate prompt content, Tom rejected the "
                "flag-file approach in favor of having the daemon return "
                "the prompt directly via TCP request."
            ),
            'anchor_raw_quote': "",
            'source_refs': [],
        },
        'why_fails': {
            'D1': "event-shaped — names the moment Tom rejected something, not the pattern",
            'D7': "no anchor_raw_quote — the reframe that generalized the lesson is lost",
            'D14': "third-person clinical — 'Tom rejected X' instead of 'Anchor reframed'",
            'D9': "schema-bound to one specific design rather than the general pattern (would semanticize wrong — recall would surface for queries about THIS bug, not the class)",
            'D15': "no trigger / behavior change — future Anchor reads this and learns about one rejection, not what to do differently",
            'D25': "empty source_refs while the originating turns ARE in the conversation window",
            'D32': "PE moment (Anchor catching missed failure mode in t1) stripped to event-summary",
        },
    },

    'voice_annotations': {
        'user_raw_quote': {
            'source_turn': 't0',
            'match': 'exact',
            'load_bearing': (
                "Tom's terse rejection ('no don't use') sets register — "
                "this is course-correction, not collaborative discussion. "
                "Without verbatim, future Anchor reads 'Tom suggested an "
                "alternative' and the lesson degrades to consensus when "
                "it was actually friction."
            ),
        },
        'anchor_raw_quote': {
            'source_turn': 't2',
            'match': 'exact',
            'load_bearing': (
                "This IS the pattern_naming moment — Anchor compressed "
                "Tom's specific correction into the general pattern. The "
                "reframe IS what makes this principle, not just one "
                "rejection. Without anchor_raw_quote at full strength, "
                "the node degrades to event-shaped and the generalization "
                "is invisible."
            ),
        },
    },

    'contract_eval': {
        'satisfies': [
            'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D12', 'D13',
            'D14', 'D15', 'D16', 'D17', 'D19', 'D20', 'D21', 'D22', 'D24',
            'D25', 'D26', 'D28', 'D29', 'D30', 'D32',
        ],
        'na': ['D10', 'D11', 'D18', 'D23', 'D27'],
        'degrades': [
            {
                'dim': 'D31',
                'note': (
                    "Single-conversation evidence promoted to principle. "
                    "Reasoning argues for generalization across read-"
                    "modify-write boundaries; the edge to 'Single-writer "
                    "invariant' provides cross-instance evidence. Soft "
                    "degrade — mitigated by source_refs anchoring + cross-"
                    "pattern edge. Acceptable per CR4 (novel type degrade-"
                    "not-violate when content is coherent)."
                ),
            },
            {
                'dim': 'D9',
                'note': (
                    "Semantization gradient: this is schema-fit (control-"
                    "by-X is a known pattern class) + atemporal + "
                    "generalizable — clean pure-synthesis-with-evidence "
                    "shape. Soft flag because the principle was earned "
                    "from one moment; the generalization is Anchor's "
                    "claim, not yet validated across instances. Should "
                    "graduate to higher-confidence as more instances "
                    "accumulate."
                ),
            },
        ],
        'cross_dim_fired': ['CR3', 'CR6', 'CR8'],
    },

    'what_this_teaches': {
        'primary': (
            "Pattern_naming as title craft — name the principle, not the "
            "moment. The title 'Control-by-request beats control-by-"
            "inspection' is the recall handle; it surfaces across queries "
            "about polling, flag files, coordination, staleness, IPC."
        ),
        'secondary': (
            "Voice symmetry on technical corrections — even when content "
            "is procedural, when Anchor names the pattern that "
            "generalized the lesson, that reframe IS the encoding-worthy "
            "voice. D7 doesn't apply only to identity moments."
        ),
        'tertiary': (
            "Sparse anchoring on a multi-turn exchange — t0 + t2 cover "
            "the rejection AND the reframe. t1 (working-through) and t3 "
            "(confirmation) are context, not anchors. Demonstrates "
            "decision 13 in practice."
        ),
    },
}
