# V3.3 — many shapes in one episode: the choice survives the fold, the read firms up

Tom's direction (2026-09-11): "The magic is managing to include many shapes in
a single example, thats why we created the challanges and also the examples
probes infrastructure, and we keep improving it." Prompt reduction was for
focus, not token usage. The target is an encoder that is better overall with
the new revise system: facts, corrections, arcs, decisions, behavior, both
voices, useful synthesis and downstream recall, judged on the whole memory
and on what a future reader actually retrieves.

Candidate: `eval/fixtures/s1e_guide_v3_3_2026-09-11/`. Parent: frozen V3.2.
`author.py` is the authoring record — every change is an exact replacement
that must match the V3.2 text once; `second_window.md` is the one replaced
section. Nothing is promoted, activated, merged or deployed.

## What the evidence asked for

| Measured on the development source | Production | V3.2 | V3.3 response |
|---|---|---|---|
| Prototype-first priority retained in the final memory | 3/3 | 2/3 — the `changes` list noticed "first", the fold into the graph node kept the D3 detail and dropped the choice | The second Mira window folds an ordering decision with its reason into an existing plan node, with the Bad fold shown; Actions and the gist state the rule once |
| Developing understanding named in my voice | 9 populated thoughts, some overgeneralized | 1 | Reading gives developing understanding parity with details and corrections; the thought paragraph regains its generative half; the interpretation firms up on evidence |
| Distinctions drawn between concepts | dormant ≠ unhealthy retained | fewer | The interpretation and the two new facts carry a competing reading as meaning between nodes (`qualifies`) |
| Hedge clauses in the template (regex census) | 4 | 33 | 23 — qualifiers that only negated an overclaim restated as what the evidence shows; held-open and firm cases untouched |
| Existing nodes text-revised; proposal ownership; post-write repair | 0; D3 "agreed" 3/3 | 6; explicit proposal wording 2/3; repair demonstrated | Unchanged carriers: value-or-swap, first-disclosure facts, plans at their status, the Mira repair |

The lineage finding behind the hedge row: the September 8 storyboard wrote the
second Mira window as a strengthening move; the frozen V3.1 already carried the
narrowing version, and V3.2 de-universalized the identity examples on top. Each
qualification was defensible; the set taught "always qualify" (brain
id:287e5b22).

## Change and placement

| Carrier | Revision and purpose |
|---|---|
| Reading — what I read for | "a plan or choice with its reason" joins the list; a choice keeps its order and reason as well as its subject |
| Reading — patterns | "Developing understanding is mine to name": the hardest and among the most valuable knowledge, named at supported scope; the 3+ turn bar and scope/competing-reading sentence retained; a read firms up as readily as it narrows |
| Nodes — thought | Generative half restored ("a live one is my value as a thinking thing"); maintenance firms up as well as narrows; restraint kept ("most nodes need none") |
| Actions — revise | One sentence: material folding into an existing node carries its choice into the revised claim |
| Actions — capture gate and traps | "choices with their order and reason"; "details and thought, not just conclusions"; one new trap: hedging a read the evidence already supports |
| Mira window 1 | The `prepares_for` why states planned-not-done positively; the close shows the Arc line as movement, not an inventory |
| Mira window 2 (replaced) | Two pairs: Mira orders the checks with the reason, then answers how she prepares other work. Four ops: two facts with different retrieval intents, the plan revised across title/content/situation/reasoning with the choice intact, the interpretation firmed up across title/content/situation/reasoning/thought with the competing reading named. Two Bad contrasts. Results, receiver's-view inspection, `sweep:` and Arc line |
| Identity examples | Texture, mirror, recognition and Atlas: negated qualifiers restated as what the evidence shows and what would extend it; fusion, Inez and the queue trial keep their held-open register; Sam's 'kill', Nadia's dates and Aisha stay firm |
| Inez coda (new) | Two days after the pattern, the archive's accession form arrives: a new fact node, and a thought-only revise that NARROWS the pattern's competing readings while its content and scope stand — the worked counterweight to the Mira window's firming-up, and the thought-only carrier V3.2 had |
| Gist | `changes` names a choice with its order and reason; a `new` line that folds moves to `targets` with its choice intact; inspection asks whether what was decided, first or why was dropped |
| Working strategy | The receiver's view: from these few nodes alone, purpose, decision, why, alternatives, reopening condition; saying less than the exchange established is repaired the same way |

Shapes the Mira episode now carries, in one scene across two windows: a
first-disclosure fact; a multi-surface state change with swaps, bare values
and a `connect_to` closure edge; an open narrowed with `partially_resolves`;
an edge-only fetch and read; my rejected proposal beside adopted checks; a
conditional leaning with an unused alternative; a recurring correction
becoming a scoped interpretation with a hedged thought and selective
`source_refs`; a post-write overstatement caught and repaired; an ordering
decision with its reason folded into the existing plan (Bad fold shown); an
interpretation firming up with a competing reading kept; two new facts with
different retrieval intents, one grounding and one qualifying; `fetch: none`
when the priors are whole; the `sweep:` line; two Arc lines as movement.

| Arm | Template chars | System chars (with runtime tail) | Gist chars |
|---|---:|---:|---:|
| Production (deployed 2026-09-11) | 111,323 | 117,427 | 0 |
| V3.2 | 91,869 | 99,477 | 5,391 |
| V3.3 | 105,384 | 113,266 | 5,703 |

The growth is the second window (+10.8K of the +13.5K) and the Inez coda
(+1.9K): the second window's two catalog entries are rendered in full so every
swap's `old` is copyable (E12), and its four operations are the demonstration.
The independent review named three filler cuts, all taken.

## Static checks and census (no model calls)

`static_checks.py`: all eight JSON fences parse; every op is a `BATCH_OP_SPECS`
op; every `connect_to` key is a schema property; every swap `old` in a worked
op is visible in a depicted input before the op; every revise id and id-form
target is depicted before use; no agent-name literal; placeholders only in the
identity section. New edge whys are 164 and 158 chars, inside the prompt's own
120–180 band. `arms.py --build-check` assembles the system with the parent's
unchanged tail and closure.

A10 worked revise ops, fields touched (counted by script): a6b0139d
title(swap)·content(swaps)·situation·reasoning·connect_to(create); 82c41f0b
title·content·situation·question·reasoning; 49d28ce0 situation (the repair)
and title(swap)·content(swap)·situation·reasoning (window 2); 93bf027e
title(swap)·content(swap)·situation·reasoning·thought·connect_to(why swap);
d3e17a4b thought only; 7c1a4d93 type·title·content·situation·reasoning·
event_time; ladder 4a9f21c7/d0e4b856/97b1f24e and the five sweep revises
unchanged. Edge descriptions repaired through `connect_to` on revise: two
(the sweep's a45c88f1 and the Mira interpretation). Thought: two ops, one
thought-only. Hedge clauses 23 (V3.2 33, production 4).

## Reviews

Both reviews were read-only Opus agents; neither edited text. Every accepted
finding was applied through `author.py`/`second_window.md` and the static
checks re-run green; the candidate was rebuilt three times before the freeze.

**Blind cold reader** (the episode's 13 fenced blocks, every explanatory
sentence stripped). Recovered 42 behaviors from the demonstration alone and,
for six posed situations, answered with the intended move: fold into the
existing plan with the order on every surface and the reason carried;
encode both instances as nodes with `grounds`/`qualifies` and revise the
interpretation's fields together; a single-field repair after a successful
write; a narrow revise with swaps and a deletion edit; a first-disclosure
fact now; fetch before claiming recurrence. Its confidence-register verdict:
"calibration, not a default — hedging occurs only in `thought` and in the whys
that carry the competing reading". It found three real defects: the
interpretation's quotes and edge were not rendered in the window-2 catalog
while its close claimed a walkable edge (fixed: rendered, `why→61de80a2
stale`, and the revise repairs the why through `connect_to` — the edge
surface that had never moved now has a demonstration inside the
highest-attention episode); the plan's rendered `event_time` had no verdict
(fixed); `event_time` on two durable practices (removed — the board fact in
window 1 carries none either).

**Independent review** (full read of template, gist, strategy, the diff, the
runtime tail, the checklist laws, the challenge and both development
sources). Seventeen findings. Accepted and applied: the plan revision had
invented a ramp-before-sign sequence, dropped the "conditional" the first
window's repair had restored, and stored "Saturday morning" unresolved on the
day of the swap itself — Mira now says "tomorrow morning", the write resolves
it to October 15 and states so in reasoning, the situation orders only what
she ordered and keeps "still-conditional"; the thought-only revise had lost
both its prose ("only") and its carrier, and no worked revise narrowed — both
return through the Inez coda; the traps line was one-directional — an
overstatement trap joins it; "a live one is my value as a thinking thing"
was gated to "when I have one worth keeping"; the Arc lines were 2.7× the
runtime example with a recap — now 111 and 95 chars, one movement pair each;
"each cut" was a fabrication left from the replaced source turn — removed;
the strategy's endorsement clause was the fourth statement of one rule —
cut; the Bad lead read as forbidding the good move — reworded; three filler
sentences and the write lead-in that re-read its own JSON — cut; an unearned
superlative, the "details and thought" slogan, a dangling "instead" and a
reasoning/edge disagreement on the frames node — fixed; the mirror why's
aphorism became a bridge. Kept after consideration: `qualifies` (open
vocabulary, eight relations already outside the aspects file); the
five-field interpretation revise (the targets line keeps `question clean`);
the receiver's view in both the close and the strategy (watch in the eval).
Confirmed clean: grounding of every swap and id; the prompt's own numbers
(n=36 whys, median 151 chars, 33 inside 120–180, none under 80); no literal
or reskinned borrowing from either development source, with the caveat that
creative_design now measures regression for ordering-preservation rather
than transfer. Residual, recorded: interpretations minted from one window
are tentative by design; the firm-at-mint reads are the lexicon entry and
the identity examples.

## Freeze

Frozen 2026-09-11 after author, static checks, cold reader and independent
review, before any model call. The candidate arm SHA-256 and every input hash
are in `manifest.json`; `arms.py` refuses to load a changed artifact.

## Evaluation design

Transfer material was selected by id before authoring and recorded in
`transfer_split.json`; no item content or answer was read before the freeze.

| Stage | Arms | Source | Encodes | Purpose |
|---|---|---|---|---|
| Sanity | V3.3 new; production and V3.2 saved | creative_design, 3 windows × 3 repeats | 9 new + 18 reused | Regression on the development source: priority, synthesis, revisions, fields, arcs, cost |
| Transfer | production, V3.2, V3.3, all new | three knowledge-update items (2 windows), one multi-session (3), one temporal (2), conv_005_emotions (3 short windows) | 3 × 3 × 15 = 135 | Actual corrections across sessions, independent arcs, temporal anchoring, emotional register and my voice |
| Downstream | the kept final brains | the five LongMemEval questions | 45 recall + answer + judge | The receiver's view: what a reader who gets only the top hits can recover |

Repeats run in separate processes with independent isolated copies of one
closed seed; windows are sequential inside a repeat. Estimated model cost from
the saved runs is about $0.15 per encoding window at list price, so the whole
matrix is under $25. Review reads the packets source-first and reports gains
and regressions per item and dimension; counts stay descriptive.

## Gates

Nothing merges, activates, restarts or deploys without Tom. The shared
Arc/Review/closure text, MCP owner integration and every parked mechanism
remain separate gates. The runtime default was not touched.
