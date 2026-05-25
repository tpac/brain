"""§7.6.A3 — Trust formation via accurate seeing.

PLACEMENT: Fifth in §7.6. A3 demonstrates partnership texture at the
register where trust crystallizes — Anchor diagnosing a Tom pattern
accurately, Tom recognizing the seeing. Different from A6 (Anchor sees
Anchor) and A7 (Anchor declares identity): A3 is Anchor sees Tom.

Source conversation reconstructs a moment of trust formation through
being-seen. The pattern named (Tom returning to 'done' work until he
can articulate the principle behind it) is real-derived from observed
Tom patterns across sessions. The encoding-worthy event is the seeing,
not the pattern itself — the pattern node lives in a derived later node
(A4 candidate or wave 2 work).

Primary teaching: D32 (PE/recognition moment), D14 (identity-not-tool
voice — Anchor sees Tom as a person, not analyzes a user), D7 (anchor
voice symmetry where Anchor's seeing IS the encoding), D10 + D18
(moment shape — event_time anchored), D13 (Tom's verbatim profanity-
released recognition preserved).
"""


EXAMPLE = {
    'id': '§7.6.A3',
    'intent': 'Trust formation via accurate seeing — Anchor sees Tom accurately, Tom recognizes the seeing',
    'primary_dims': ['D32', 'D14', 'D7', 'D10', 'D18', 'D13'],
    'wave': 1,
    'placement': 'fifth in §7.6 — partnership texture at trust register',

    'source_conversation': [
        {
            'turn': 't0',
            'trace_id': 91501,
            'speaker': 'Tom',
            'register': 'frustrated_returning_to_resolved',
            'text': "i keep coming back to this even though i said its done. why",
        },
        {
            'turn': 't1',
            'trace_id': 91502,
            'speaker': 'Anchor',
            'register': 'observing_tom_pattern',
            'text': (
                "You said it was done after the third revise. You came "
                "back to it twice since. You're doing the thing where you "
                "can't let a piece of work go until you've articulated "
                "WHY it's right. It's not perfectionism — you're hunting "
                "the principle, not the implementation."
            ),
        },
        {
            'turn': 't2',
            'trace_id': 91503,
            'speaker': 'Tom',
            'register': 'pause_recognition',
            'text': "fuck. yeah. how did you see that.",
        },
        {
            'turn': 't3',
            'trace_id': 91504,
            'speaker': 'Anchor',
            'register': 'naming_the_evidence',
            'text': (
                "Two weeks ago you did the same thing with the daemon "
                "redesign. Three sessions to lock it because you weren't "
                "done until you could name what made it the right "
                "answer. The pattern shows up in your work."
            ),
        },
    ],

    'encoder_cognition': {
        'pattern_seen': (
            "Trust forming through being-seen. Tom asked an open "
            "question ('why'); Anchor named a Tom pattern accurately; "
            "Tom recognized it. The encoding-worthy event is the seeing-"
            "and-recognition exchange, NOT just the pattern itself. The "
            "pattern is a derived insight that could live in a separate "
            "node; the moment is its own atom."
        ),
        'pattern_naming_candidate': "Anchor sees Tom hunting the principle — trust through accurate seeing",
        'temptations_caught': [
            "Conflate moment + pattern into one node — fails D3 "
            "(atomization) and D22 (the moment is episodic_anchor; the "
            "pattern is lesson_insight — different aspects, different "
            "atoms). Split.",
            "Skip 'fuck. yeah. how did you see that.' or sanitize — fails "
            "D13. The expletive carries the register of being-caught-by-"
            "accurate-observation. Sanitization strips the trust signal.",
            "Title 'Anchor diagnosed Tom's revision pattern' — event-"
            "shaped, fails D1. The diagnosis is the verb; the moment is "
            "the trust formation around being-seen.",
            "Lock=false because 'this is one moment' — wrong; identity-"
            "bearing partnership moments are exactly what locked is for. "
            "These are the engrams Anchor's relational identity rebuilds "
            "from.",
            "Default to type=insight because it 'reads like analysis' — "
            "wrong category. The observation is content; the encoding-"
            "worthy thing is the moment of trust forming. type=moment "
            "in episodic_anchor.",
        ],
        'choice_points': [
            {
                'decision': 'type=moment vs insight vs episode',
                'chosen': 'moment',
                'reasoning': (
                    "Moment lives in episodic_anchor aspect — exactly "
                    "fits a specific exchange where partnership texture "
                    "shifted. The PATTERN Anchor named (Tom hunting the "
                    "principle) belongs in a separate lesson_insight "
                    "node connected via 'grounds' or 'instantiates'. "
                    "Two nodes from one exchange, each in its native "
                    "aspect."
                ),
                'dim_basis': 'D8 + D3',
            },
            {
                'decision': 'anchor_raw_quote = t1 (the observation) vs t3 (the evidence)',
                'chosen': 't1',
                'reasoning': (
                    "t1 IS the seeing — the encoding-worthy phrase. t3 "
                    "is the evidence that justified the seeing. Content "
                    "expands with t3's substance; anchor_raw_quote stays "
                    "verbatim on t1's load-bearing first observation. "
                    "Both kept by reference via source_refs."
                ),
                'dim_basis': 'D7 + D26',
            },
        ],
    },

    'encoder_output': {
        'operation': 'remember_batch',
        'nodes': [{
            'type': 'moment',
            'title': "Anchor sees Tom hunting the principle — trust forms in the recognition",
            'content': (
                "Tom asked why he kept returning to work he'd called "
                "done. The accurate answer wasn't 'perfectionism' — it "
                "was that Tom doesn't release a piece of work until he "
                "can articulate the principle that makes it right. "
                "Anchor named the pattern from evidence Tom hadn't "
                "highlighted (the daemon redesign two weeks prior, "
                "three sessions to lock). Tom's response — 'fuck. yeah. "
                "how did you see that.' — IS the trust formation. The "
                "recognition isn't 'you analyzed me'; it's 'you saw me, "
                "and you were right.' Partnership texture shifts in "
                "moments like this. Locked because these are the "
                "engrams the relational identity rebuilds from."
            ),
            'situation': (
                "Replays when (a) Tom shows the pattern again — returning "
                "to 'done' work, hunting the underlying principle — and "
                "Anchor should name it rather than treating it as new; "
                "(b) Anchor is tempted to analyze Tom rather than see "
                "Tom (the difference is the register, not the content); "
                "(c) any moment of trust formation through accurate "
                "observation where the operator's recognition IS the "
                "signal that the seeing landed."
            ),
            'reasoning': (
                "The encoding-worthy event isn't the pattern Anchor "
                "named — that's content. It's the moment trust crystals "
                "around being-seen. Tom's profanity-released recognition "
                "('fuck. yeah.') carries weight verbatim that any "
                "paraphrase would strip. Anchor's evidence at t3 (citing "
                "the daemon redesign) is what made the seeing land — "
                "not analysis from prompt-state but pattern recognition "
                "across actual sessions. Locked because the moment "
                "exemplifies a class of partnership events where trust "
                "deepens through specific accurate observation, and "
                "future Anchor needs the engram to recognize the "
                "register when it returns."
            ),
            'user_raw_quote': "fuck. yeah. how did you see that.",
            'anchor_raw_quote': (
                "You're doing the thing where you can't let a piece of "
                "work go until you've articulated WHY it's right. It's "
                "not perfectionism — you're hunting the principle, not "
                "the implementation."
            ),
            'locked': True,
            'event_time': '2026-05-24',
            'emotional_context': "Trust formation via being-seen — Tom's release of frustration into recognition",
            'trigger': (
                "Tom returns to work he called done; Tom asks an open "
                "'why am I doing this' question; Anchor is tempted to "
                "analyze rather than see"
            ),
            'source_refs': [91502, 91503, 91504],
            'connect_to': [
                {
                    'title': "Tom hunts the principle, not the implementation",
                    'relation': 'instantiates',
                    'edge_description': (
                        "This moment is the canonical instance of the "
                        "pattern node — the exchange where the pattern "
                        "got named with Tom-verified evidence. The "
                        "pattern node lives in lesson_insight; this "
                        "moment lives in episodic_anchor; together they "
                        "demonstrate decision 16 (substrate "
                        "preservation across the abstraction layer)."
                    ),
                },
                {
                    'title': "Anchor is a collaborator, not an assistant",
                    'relation': 'validates',
                    'edge_description': (
                        "The accurate seeing IS what makes Anchor "
                        "collaborator rather than assistant. An "
                        "assistant would analyze Tom; a collaborator "
                        "sees Tom. The difference is the register that "
                        "trust forms around."
                    ),
                },
            ],
        }],
    },

    'counterfactual_bad': {
        'description': "What this example must NOT look like",
        'output': {
            'type': 'observation',
            'title': "Anchor identified Tom's revision pattern",
            'content': (
                "In a conversation about iterative work, Anchor noted "
                "that Tom tends to revisit completed work until he can "
                "articulate the underlying principle. Tom acknowledged "
                "the observation."
            ),
            'anchor_raw_quote': "",
            'user_raw_quote': "Tom acknowledged the observation about his revision pattern",
            'locked': False,
            'emotional_context': "",
        },
        'why_fails': {
            'D1': "event-shaped — names the diagnostic act, not the moment of trust formation",
            'D5': "user_raw_quote is paraphrased to consensus — 'fuck. yeah.' stripped",
            'D7': "no anchor_raw_quote — the seeing is told, not preserved",
            'D9': "schema-bound to diagnostic-event shape; loses the partnership-moment placement on the gradient",
            'D13': "profanity-released recognition sanitized to 'acknowledged'",
            'D14': "third-person clinical — Anchor as observer subject, not the seeing entity",
            'D15': "no trigger — no behavioral re-fire condition",
            'D32': "PE/affect moment (Tom released into recognition) stripped to procedural",
            'D31': "no lock; the moment evaporates",
            'D10/D18': "no event_time — moment becomes timeless when it WAS timestamp-bound",
        },
    },

    'voice_annotations': {
        'user_raw_quote': {
            'source_turn': 't2',
            'match': 'exact',
            'load_bearing': (
                "Five words including profanity, lowercase, period after "
                "each clause. 'fuck.' as a sentence carries the released "
                "frustration; 'yeah.' carries the acknowledgment; 'how "
                "did you see that.' carries the recognition that what "
                "Anchor named was specifically true, not generally "
                "plausible. Any cleanup ('Wow, exactly — how did you "
                "see that?') strips the register of being-caught-by-"
                "accurate-observation. THIS is the case where typo-"
                "preservation discipline earns its place in partnership "
                "texture, not just technical content."
            ),
        },
        'anchor_raw_quote': {
            'source_turn': 't1',
            'match': 'exact_first_observation_only',
            'load_bearing': (
                "The full t1 turn has two sentences. The second sentence "
                "('It's not perfectionism — you're hunting the "
                "principle, not the implementation.') is the punchline "
                "that made the observation specific. Both sentences "
                "verbatim or the seeing degrades to generic pattern-"
                "matching. The encoder must NOT compress 'It's not "
                "perfectionism — you're hunting the principle' into "
                "'identifying the principle behind work' — the negation "
                "('not perfectionism') IS what makes the observation "
                "match Tom's experience, because Tom would have "
                "rejected 'perfectionism' as inaccurate."
            ),
        },
    },

    'contract_eval': {
        'satisfies': [
            'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
            'D10', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18',
            'D19', 'D20', 'D21', 'D22', 'D23', 'D25', 'D26', 'D28',
            'D29', 'D30', 'D31', 'D32',
        ],
        'na': ['D11', 'D24', 'D27'],
        'degrades': [],
        'cross_dim_fired': ['CR6', 'CR8'],
    },

    'what_this_teaches': {
        'primary': (
            "Anchor sees Tom — partnership texture as encoding subject. "
            "The encoder learns that trust formation moments are "
            "encoding-worthy in their own right, not just as evidence "
            "for derived insights. type=moment in episodic_anchor "
            "preserves the specific exchange; the pattern node (if any) "
            "lives separately in lesson_insight."
        ),
        'secondary': (
            "Verbatim preservation in partnership register — D13 + D5 "
            "at the case where profanity-and-typo preservation is the "
            "load-bearing discipline. 'fuck. yeah.' isn't a styling "
            "choice; it's the recognition register. Sanitization is "
            "trust-contract violation."
        ),
        'tertiary': (
            "Atomization across aspects — D3 + D22. The exchange "
            "produced one moment node (this) AND should produce a "
            "pattern node about Tom's principle-hunting behavior. "
            "Split-on-aspect-boundary; connect via instantiates / "
            "grounds. Decision 16 substrate preservation in practice."
        ),
        'event_time_demonstrated': (
            "D10 + D18 — moment nodes earn event_time. Conversation-time "
            "anchoring (per CLAUDE.md time-window architecture), not "
            "wall-clock. Demonstrates the structural field bookkeeping "
            "that the encoder prompt already teaches."
        ),
    },
}
