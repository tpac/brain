# Current production versus V3.2: different strengths, no overall winner yet

Nine production-package encodes completed, compared with the nine saved V3.2
encodes. V3.2 integrates additions into existing nodes and is more explicit
about proposal status. Production supplies more exploratory interpretation,
more connections and slightly better detail/priority retention in this source.
The candidate is substantially leaner in stored memory, but generated slightly
more output. These results support carrying the complete candidate forward to
stronger release tests; they do not establish an overall quality win.

## What was compared

Current production was read from the running daemon, not inferred from the
old production candidate file. Effective `s1e`: default `fd28b7b6321f`, Sonnet
4.6 / medium. The daemon's code fingerprint `8c33e3f67dc54c06` matched the
source code used to export its native prompt, generated ending, tool schemas,
preamble and limits. Production retained its one-batch preamble, absence of
a gist, older revise API, and next-reply-is-final closure.

Three independent repetitions each ran three sequential five-pair windows on
`conv_004_art_design_extended`. V3.2's nine saved runs use the same source,
trace IDs, closed seed, clock, model and limits. All six production snapshot
carryovers match exactly; every later catalog contains that repetition's own
prior created/touched nodes. The source is a synthetic design conversation
whose user is called Tom; its claims are not new observations about the real
operator.

The common replay engine is the existing branch harness. It gives production
only its native schemas and V3.2 its wider schemas; production emitted **zero
candidate-only revision shapes**. Both use the same catalog/storage/journal
replay and local dispatch. This is a comparison of encoder packages under
shared replay, not a full-daemon A/B: scheduling, installed MCP behavior,
S1Scribe attribution and downstream recall still need their own verification.
The newer branch engine and older API are compatible on the operations used;
this does not prove every production write-path behavior is identical.

All nine new runs created nodes and ended in DONE. No API error, truncation
or partial-tool failure occurred. Production made no read calls; two windows
made a second write for connections. V3.2 read before writing in two windows.
Neither package made a post-write semantic repair. There were no additional
reviewer/interview calls.

## Observed results

Totals below combine three final memories, not one graph. Counts describe the
outputs; they are not a weighted quality score.

| Measure | Production | V3.2 | Meaning |
|---|---:|---:|---|
| Nodes, repeats 1/2/3 | 19 / 20 / 17 | 16 / 18 / 18 | Some extra production nodes carry principles or behavior interpretations; fewer is not automatically better |
| Total final nodes | 56 | 52 | Similar scale |
| Existing nodes with actual text-field revisions | 0 | 6 | V3.2 updates 1/2/3 nodes by repeat; mostly enrichment, not correction of contradicted facts |
| Content / reasoning change events | 0 / 0 | 5 / 4 | Useful changes described below |
| Title / situation / question change events | 0 / 0 / 0 | 0 / 0 / 0 | No evidence here of broader field repair |
| Explicit prototype-first priority retained | 3/3 | 2/3 | V3.2 repeat 1 loses the ordering decision |
| Encoder activity as bursts of connections | 3/3 | 2/3 | A smaller V3.2 detail omission, also present in its V3.1 predecessor |
| Full three Arc lines retained in final continuity | 2/3 | 3/3 | Production repeat 2 exceeds the shared 800-character cap and loses the beginning of its first line |
| Nodes with a populated thought | 9 | 1 | Production is more exploratory in this field; assess actual value, not a quota |
| Outgoing semantic relations | 106 | 69 | Both useful connections and repetition contribute to production's larger graph |
| Final nodes with no semantic neighbor | 0 | 2 | V3.2's journal overlay in R1 and missing-spec open in R2; recall impact untested |
| Node words, including quotes | 12,209 | 9,909 | V3.2 is 18.8% shorter |
| Node + relation words | 15,440 | 11,993 | V3.2 is 22.3% shorter |
| Generated output tokens | 37,470 | 38,930 | V3.2 is 3.9% higher despite smaller stored memory |

The primary source bundles survive in both: organic graph with meaningful
distance; access/recency/pulse/locked-node mappings; gap placeholders; journal
colors; time-lapse; session-impact metrics; four health metrics; personal
reflection and user agency; four themes and CSS variables; event-specific
audio; ambient presence; D3 and field mappings; beauty/first-impression intent.
Presence does not mean every surrounding assertion is correct.

## What V3.2 improves in this source

**Integration into existing knowledge.** Production creates a new prototype
node linked to earlier graph/layout nodes. V3.2 also updates the graph node
itself in every repeat, retaining old layout details while adding the proposed
implementation. Repeat 2 updates the mirror node's reasoning with the later
explicit endorsement. Repeat 3 develops its profile through multiple windows.
These are actual persisted changes, not counts of attempted revise calls.
Repeat 3 also propagates an aesthetic ownership error into the profile;
revision activity is therefore not uniformly beneficial even within V3.2.

That demonstrates useful behavior of the new revise system. It does not prove
production forgot the additions: production's separate linked nodes preserve
most of them. The decisive test is whether a future reader receives the right
current claim, especially after an actual correction. This conversation has
little source-driven reversal of old facts.

**More careful proposal ownership.** Production's repeat-2 ambient node
`8135a5a5` reasons that Tom's endorsement “that's the vision” ratified the
menubar/notification proposals. The phrase was the source assistant's own
reply. V3.2's ambient nodes explicitly retain the implementations as the
assistant's proposals, unconfirmed for implementation. Both preserve the
expressive user quote; V3.2 avoids this particular false endorsement.

All production prototype nodes call D3 decided/agreed. V3.2 includes explicit
proposal wording in repeats 2/3, although repeat 2's situation still calls it
agreed. Audio becomes mandatory default-off in all production repeats despite
the source saying optional/quiet; V3.2 does this in two repeats. These are local
improvements, not a solved certainty/commitment contract.

**Shorter continuity without losing a phase.** Both capture visualization →
analytics/privacy → ambient presence. Production repeat 2 writes 934 characters
of new Arc text before separators into an 800-character accumulated field;
the final memory starts partway through the first line. V3.2's final Arcs are
605, 695 and 639 characters and keep all three lines. The underlying nodes
still retain production's initial vision; this is a continuity defect, not
complete loss of that knowledge. Both still lean too much toward inventories
of encoded features rather than movement in decisions.

## What production preserves that deserves protection

**The choice of what happens first.** Production stores graph visualization
and temperature colors as the first prototype priority in all three repeats.
V3.2 repeat 1 folds the technical specification into `492707b8` and drops the
priority from all final nodes, quotes and Arc. This is a material omission;
retaining the detailed subject of a decision does not preserve the decision.

**Exploratory understanding of behavior.** Production repeat 2 interprets the
time-lapse, session-impact and self-portrait requests as a recurring move toward
observing the system's own behavior (`d7d7a8df`). Its thought links that desire
with agency: rich internal reflection can coexist with user-controlled display.
Its aesthetic node also identifies the session's return from the opening
vision through features to making that vision perceptible through beauty.
These are useful syntheses beyond a feature inventory.

Not all extra synthesis is sound. Production's earlier pattern `69dab691`
says Tom's first move is *always* to reject the generic, using several turns
that did not contain a rejection. Another thought treats node temperature and
cluster health as the same signal even though the health map adds confidence
and revision age. Preserve the willingness to form understanding; calibrate
its evidence and scope rather than rewarding every extra thought.

The sparse-thought result predates V3.2: on this same source the saved V3 arm
has one thought across three repeats, V3 with generic tools zero, and V3.1 one.
We cannot attribute the production-to-V3.2 difference specifically to the latest
semantic-fidelity additions. Nor is thought usage alone proof that useful
synthesis vanished: V3.2 carries some in content, reasoning and edges.

**Useful graph organization.** Every production node has a semantic neighbor.
Production's session-impact versus health-map connection distinguishes investment
over time from resulting condition; its time-lapse connection distinguishes
continuous growth from session aggregates. V3.2 also produces versions of these
useful distinctions, with less relation text. Production repeat 2 additionally
keeps the distinction that cold/dormant knowledge is not necessarily bad.

## Fields, voice and scope

| Field | Production: nodes / words | V3.2: nodes / words |
|---|---:|---:|
| Title | 56 / 732 | 52 / 614 |
| Content | 56 / 3,966 | 52 / 3,296 |
| Situation | 56 / 1,506 | 52 / 1,177 |
| Reasoning | 56 / 2,467 | 52 / 1,906 |
| Question | 42 / 484 | 41 / 486 |
| Thought | 9 / 349 | 1 / 54 |
| Their quote | 48 / 1,055 | 45 / 956 |
| My quote | 44 / 1,650 | 43 / 1,420 |

Production titles are often longer but specific; V3.2 retains useful retrieval
handles with fewer words. Both preserve “brain breathing,” “mirror, not a
camera,” “peripheral vision for knowledge,” and the spreadsheet/first-impression
framing. Both voices remain represented. Production's 92 populated quotes all
match the source; nevertheless some surrounding prose reverses ownership.
This supports Tom's calibration: credit retained sentiment without pretending
all inferences built on it are sound.

The most consequential field conflict is often the future-use cue: a cautious
body or quote can sit beside a situation that declares an implementation agreed.
It is recoverable knowledge with ambiguous guidance, distinct from a missing
priority. Neither arm changed a situation/title/question in this cell.

Scope also matters in derived edges. Production generalizes the personal-
inference “mirror” rule into a restriction on ordinary contextual notifications.
Repeat 2's review calls ambient notifications a contradiction; repeat 3's edge
says peripheral presence is ethical only if it never pushes. The source's
pull-only restriction concerns personal insights, so the broad prohibition
does not follow. This is an example of useful concern becoming too broad,
not a reason to suppress all doubt. V3.2 has its own weak scope expansion,
such as tying ordinary theming to personal-inference governance in repeat 1.

Explicit event_time writes are 19 in production and 51 in V3.2; confidence
writes are 1 and 5. This fixed-date synthetic source does not establish better
temporal or confidence reasoning. Default neutral emotion and pipeline fields
are not evidence of intentional additional encoding. No field/ref quota follows.

## Journals and execution

Production keeps the missing journal specification open through all windows
in repeats 1/3 without fetching it. Repeat 2 calls it resolved on the basis
of supposed evidence of its existence, while acknowledging the design record
is still absent. V3.2 does make two initial read attempts but does not resolve
the missing specification. Neither system's notes establish an infinite-cycle
explanation for all missed writes.

There is also legitimate residue: unvalidated clustering, an unknown earlier
specification, and unresolved implementation choices. Encoding those doubts
is useful; not implementing the hypothetical dashboard is not an encoder failure.
No run here failed from token limits or wrote zero nodes.

Production ignored its literal next-reply-is-final wording twice to create
follow-up connections. That shows the wording is not absolute control. It does
not contradict the earlier measured read-round suppression: this cell has no
production read round. Extra production writes were connections, not semantic
repairs. V3.2's new repair example also produced no such repair here.

## Economy and the release decision

System plus gist is 117,427 → 104,868 characters (10.7% smaller). Including
native tool JSON gives 144,227 → 131,699 (8.7% smaller), excluding preambles
and dynamic input. Stored node/relation text is 22.3% smaller, but output
tokens are 3.9% higher: planning/checking prose is output too. Cache behavior
differs, and these are reused runs on different dates; do not turn this into
a controlled dollar-cost claim.

My recommendation is to keep V3.2 as the complete-package contender and use
the next comparison to establish its original intended advantage: changes to
existing knowledge under actual corrections, plus first-disclosure facts and
downstream recall. Preserve production's exploratory synthesis as a release
criterion alongside voice and detail. This result does not justify either
declaring V3.2 an overall winner or reflexively writing another prompt version
around the familiar aesthetic sentence.

The complete release includes real MCP descriptions/schema, revise write path,
selected template/gist/ending and required shared-reader changes. Their
integration and deployment checks remain work beyond this replay. See the
[release scope](S1E-PRODUCTION-COMPARISON-AND-RELEASE-SCOPE-2026-09-11.md).
No runtime/default/MCP source changes, merge, activation or deployment occurred.

## Reproducible evidence

`eval/results/s1e_production_comparison_2026-09-11/` contains completion,
manifest history, preflight, exact requests/calls, before/after snapshots,
continuity, `quality_inventory.json`, `comparison_inventory.json` and readable
`whole_memory_review/` packets for both arms. Its summary records Arc retention
and graph adjacency. The existing census is reused; this report supplies an
author's source-based quality review, not a blind score. Earlier V3.1/V3.2
reports and their frozen inputs remain unchanged.
