# S1E content and revision review — saved outputs, 2026-09-09

The current bottleneck is preserving the meaning of a claim as it moves from
source dialogue into a plan, node fields, edges and later verification.
The tools mostly execute the requested operations. A successful write can
persist a mistaken interpretation, and subsequent inspection can endorse it.
This is more specific than “the encoder needs to revise more.”

This review adds lifecycle evidence to the [54-encode report](S1E-V3-1-CROSS-CORPUS-RESULTS-2026-09-08.md).
It uses existing outputs only. No prompt changes, model calls, deployments or
merges were made. V3 means compact V3+cues; “tools” means the generic revised
descriptions; V3.1 includes those tools and the revised comparison procedure.

## The zeros: current results and historical failures are different

The zeros in the previous report's advice table meant **zero standalone
advice/reference nodes**, not zero total nodes. That presentation was ambiguous
and is corrected. The current LongMem sample wrote in all 27 encoding windows:

| Arm | Repeat | Created W1 → W2 → W3 | Final nodes | Standalone advice among final nodes |
|---|---|---|---:|---:|
| V3 | 1 | 5 → 4 → 7 | 16 | 6 |
| V3 | 2 | 3 → 4 → 4 | 11 | 0 |
| V3 | 3 | 4 → 5 → 6 | 15 | 4 |
| V3 + tools | 1 | 5 → 6 → 6 | 17 | 7 |
| V3 + tools | 2 | 5 → 5 → 7 | 17 | 10 |
| V3 + tools | 3 | 3 → 3 → 4 | 10 | 0 |
| V3.1 + tools | 1 | 3 → 4 → 3 | 10 | 0 |
| V3.1 + tools | 2 | 5 → 6 → 5 | 16 | 7 |
| V3.1 + tools | 3 | 5 → 8 → 5 | 18 | 9 |

These counts come from differences between actual `nodes_before.json` and
`nodes_after.json`, not a missing-file default. Each window took two or three
model replies, below the five-reply limit. The largest LongMem window used
4,747 output tokens across its replies; the configured limit was 12,288 per
reply. Every saved `truncations` list is empty; every final reply contains DONE.
The runner explicitly records `stop_reason=max_tokens` in that list.
The saved chains and persisted writes show no evidence of an interrupted,
blocked or token-exhausted encoding window. A read-only reply or final closing
reply can naturally create zero nodes within a successful window.

The old LongMem guide v1.2 failures really did write zero nodes. I read their
existing `encoding_run` traces through `Brain.query_traces` on temporary copies
of the frozen item databases; the historical ledger's totals were not rescored.

| Historical item | What actually happened | What this rules out |
|---|---|---|
| Gym schedule, `59524333` | Three one-reply runs, zero actions. First calls the gym schedule hypothetical; a later run acknowledges a disclosure but requires cross-session confirmation; the final run cites that no-mint verdict despite recognizing repeated confirmations. | A write rejected by a tool: no tool was called. Recorded truncation and rejection counts are zero. |
| Chandelier, `71017276` | First run lists seven advisory candidates and makes no call, using 686 output tokens. The later flush says `new: none`, rejecting acquisitions/plans as insufficiently specific. | Token exhaustion or a failed mutation in either run: telemetry records neither. |

**Historical ledger correction:** the chandelier entry describes the candidates
as five and puts the 686-token list in the second run. Saved timestamps and
stop counters place that list at counter 5, before the `new: none` flush at
counter 6; seven bullets are visible. The first gym run has a clean Review;
the explicit no-mint residue appears in the second and is cited by the third.
The ledger places that residue in the first. These corrections change the
causal chronology, not the recorded zero-node or benchmark totals.

There is also an observability limit: the historical chandelier `final_text`
was capped at storage, with an explicit “+954 chars truncated” marker. That is
different from model token truncation. We can establish a nonempty candidate
list and no call; we cannot read the entire closing justification from that
trace to distinguish abandoned intent from a forgotten call. The next saved
prompt carries an arc saying “generic advice delivery, no durable atoms.”
The historical label “plan then stop” should retain that limit.

The older `nodes.keywords` dump bug is a separate known failure. The current
cell saves nodes directly through `brain.get_node`; it does not use that old
dump route or substitute an empty list for a missing dump.

Evidence: [window census and lifecycle extracts](../eval/results/s1e_v31_cross_corpus_2026-09-08/deep_evidence.json),
[historical trace export](../eval/results/s1e_v31_cross_corpus_2026-09-08/historical_zero_traces.json),
[original ledger](</Users/tpac/AgentsContext/s1e-field-coverage-gold/ab_2026-09-01_03/ops9/ADJUDICATION.md>).

## 1. A field can reverse the qualification in another field

Source design turn 14: Tom requests graph visualization plus temperature
colors as the prototype's first step. The assistant proposes D3.js. Tom's
next turn asks for beauty; it does not explicitly select the library.

V3+tools R3 writes the following **in one creation operation**:

| Surface of `3b9bcdab` | Stored meaning |
|---|---|
| Content | “Implementation spec I proposed: D3.js…” |
| Reasoning | “The D3.js choice is my recommendation and not yet confirmed as a final technology decision.” |
| Situation | “D3.js is the agreed starting point” |

This is not missing evidence, a field-preservation side effect, or a failed
revision. The encoder explicitly knows the evidence limit while writing the
stronger claim elsewhere. Situation is especially vulnerable because it is
written as an instruction for a future action. The same distinction fails
outside software: V3.1 R1's handbag content says “No purchase confirmed,” while
its title says the sister is “receiving a handbag gift.” That can be read as an
intended gift, but it loses the explicit uncertainty a standalone title needs.

V3.1 R1's ambient node goes further: reasoning calls menubar/nudges “my concrete
proposals … adopted without objection,” and situation calls them agreed.
Here the reasoning itself treats non-objection as evidence of adoption. These
are two related failures: inconsistent rendering of known uncertainty, and an
unsupported rule for deciding what counts as agreement.

The guide already says: **“Title, trigger, reasoning and edges must agree with
the content on speaker, scope and evidence state. A qualified paragraph does
not repair an unsupported claim elsewhere.”** Its reading cue repeats the
principle. Adding it again would not be a new lever.

The next hypothesis should concern construction: settle the actor and evidence
status of each claim before rendering its retrieval surfaces; demonstrate a
requested outcome beside an unaccepted implementation proposal. Check the
isolated title/situation/edge for the same meaning, rather than verify only
that those fields exist. This targets original handoff patterns 1, 4 and 5.
Whether a revised example produces this behavior remains untested.

Evidence: [D3 call and result](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_titles_new_tools/repeat3/creative_design/window3/calls.json),
[ambient and handbag fields](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_1_titles/repeat1/creative_design/window3/nodes_after.json),
[handbag node](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_1_titles/repeat1/longmem_unseen/window3/nodes_after.json).

## 2. Choosing the same claim comes before choosing a revision

LongMem first discusses a baby gift in a friend's nursery context, narrowing
toward a gym. The next session introduces a coworker's new baby and a basket.
The source does not say that this replaces the first search. Nor does it
prove those relationship labels refer to different people. Preserve the two
contexts without inventing their identity relationship.

| Step | V3 R1: useful scoping | V3 R3: unsupported merger |
|---|---|---|
| Initial memory | Baby-gym leaning, no purchase | Open gift search, blanket → gym, no purchase |
| Interpretation of next session | New coworker context; original claim still holds at its earlier scope | “recipient was identified as a coworker”; same search progressed to basket |
| Actual mutation | Changes only situation, limiting the gym cue to its original friend context | Rewrites title, content, question and reasoning; changes relations to make gym an earlier phase |
| Following window | Keeps gym status unchanged | Marks the rewritten arc and relations clean |

The bad run revises more fields and tells a coherent story. It is wrong at
the entity/occasion comparison before any call is constructed. Its later
verification checks consistency with that interpretation, not whether the
source establishes continuity between the occasions. This is an extension of
the mistaken-clean loop, not evidence that tools cannot revise.

Conversely, V3.1 R3's mirror principle shows the procedure working. W2 reasoning
records no explicit endorsement. W3 sees Tom's “Good principle. Mirror not
camera.”, marks reasoning stale, actually updates reasoning and resolves that
open question. But it leaves “I supplied the camera/mirror framing,” although
Tom introduced that phrase. The update repairs the newly noticed predicate
(endorsement), not all claims bundled into the sentence (including authorship).

V3.1 R1's graph update demonstrates the opposite: it adds “D3.js is the agreed
implementation library” to content and situation. More activity can amplify
a mistake. The original reasoning about opening turns stays unchanged.

Across this cell, the V3.1 design sequences have nine content-change events,
four situation changes and two reasoning changes; no title or question changes.
Some nodes change in more than one window, so these are field-change events,
not distinct-node counts. LongMem V3.1 has no field revisions. That does not
mean an inability to revise: most of that source introduces separate occasions.
The design source supplies direct positive evidence that V3.1 revises.

A useful next comparison example would require deciding “same claim revised,
new claim alongside it, or relationship unknown” before the field verdicts.
Then verify the evidence for identity/continuity, not merely shared nouns.
This targets handoff patterns 2 and 5, while preserving the successful narrow
update. No source-specific friend/coworker rule is warranted.

Evidence: [scoping trace](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_titles/repeat1/longmem_unseen/window2/result.json),
[merger trace](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_titles/repeat3/longmem_unseen/window2/result.json),
[mirror update](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_1_titles/repeat3/creative_design/window3/result.json),
[all actual field revisions](../eval/results/s1e_v31_cross_corpus_2026-09-08/deep_evidence.json).

## 3. Extra nodes buy different recall, not necessarily better recall

Within each arm and corpus, the three initial captured requests are
byte-identical. V3.1 LongMem R1 opens by describing its contributions as
“advisory responses with no personal findings or decisions” and proposes
three personal/decision nodes. R2 also recognizes advice but selects separate
reference nodes. R3 initially lists seven candidates, writes five, then
explicitly explains why store/general-category lists were dropped.
The four-list artifact does not fix a stable boundary for useful substance.

All three sequences with zero standalone advice in W1 keep zero through W3
(V3 R2, tools R3, V3.1 R1). Each of the other six creates standalone advice in
W1 and at least one later window. This is consistent with initial selection
habits being carried by catalog/journal, but does not prove that causal effect.
We did not swap continuities or replay W2 with alternative initial catalogs.

The V3.1 contrast is informative:

| Future information need | R1, 10 nodes / 1,280 words | R3, 18 nodes / 2,521 words |
|---|---|---|
| Nursery/Target/phone events; sibling interests; budget and unpurchased bag | Retained | Retained |
| Six gym brands and main safety checklist | Retained inside the 141-word gift-decision node | Separate 134-word brand node, plus 105-word episode; adds brand-specific characterizations |
| Gym versus playmat distinctions | The comparison is mentioned in the arc, but its substantive distinctions are absent | Separate comparison node |
| Personalized-blanket suppliers and customization options | Absent beyond the considered blanket option | Separate reference node |
| Why a carrier gave way to a more universal gift | Retains personal-preference rationale in the decision | Adds a 167-word guidance node with conditions and alternatives |
| Handbag specification details | Keeps price, material, shape and hardware | Adds dimensions and other assistant-provided details |

Those extra details may be useful. Their attribution also matters: “the
assistant recommended this” is supported; “this is verified product research”
is not supported by these conversations. The large run additionally spends
154 words on basket presentation advice, including a generic spoken greeting.
That has less distinctive value for this person's history than their chosen
message or why a gift option was rejected, though it can answer a reference query.

Therefore the compact run is not a proven winner and the reference nodes are
not automatically noise. Compare each node's **unique recoverable detail,
new explanation/connection and likely future query**, then its duplication
and unsupported additions. Keep personal disclosures unconditional; let a
different retrieval need justify a separate advice node. This targets patterns
6 and 8. “Save all advice” and “save no advice” both repeat known failures.

Evidence: [R1 first write](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_1_titles/repeat1/longmem_unseen/window1/calls.json),
[R3 planning, writes and explicit skips](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_1_titles/repeat3/longmem_unseen/window1/result.json),
[advice membership/word counts](../eval/results/s1e_v31_cross_corpus_2026-09-08/quality_summary.json).

## 4. More fields did not produce more developed understanding

The previous census is descriptive: all 287 nodes have content, situation and
reasoning; questions are selective. The new content reading adds these limits:

| Field | What worked | What did not |
|---|---|---|
| Title | Specific people, stores, products and scoped feature handles | Stronger commitment than body; one harmful title rewrite joins unsupported occasions |
| Situation | Correctly limits the gym memory to its context; often supplies a useful future action | Turns proposals into agreed instructions; can be “clean” despite conflicting reasoning |
| Reasoning | Explicit purchase limits; useful mirror-endorsement update | “Not contradicted” becomes acceptance; later claims inherit opening-turn support; encoding rationale and turn coordinates replace evidence |
| Question | Retains genuinely open purchases/choices as future retrieval needs | Presence alone does not show whether a future question can be answered accurately |
| Thought | The field is available and can be changed independently | Only two populated fields, both future encoding-workflow notes; no domain thought in LongMem |
| Type / lifecycle | Episodes preserve gift sequences; open choices sometimes remain explicit | V3.1 has 27 exact `decision` types among 52 design nodes, including unsettled implementation proposals; no lifecycle values were supplied |
| Open metadata | Arbitrary keys are permitted; dispatch forwards extra fields | No custom dimensions were used; adding permission alone did not make them useful |
| Quotes | Many exact, discriminating phrases survive | Six unmarked condensations/stitches; many exact quotes merely repeat generic advice or content |

Type is not cosmetic: the guide defines `decision` as a settled choice and
describes special retrieval treatment for it. A qualified paragraph does not
automatically justify classifying a still-open implementation as a decision.
The comparison with V3's other type names is not a causal type-quality score.

Do not respond with a thought/custom-field quota. Meaning already appears in
other carriers: V3.1 R2 distinguishes cold knowledge from stale knowledge, and
a V3 edge distinguishes session impact (WHEN knowledge grew) from health
(its current STATE). Those are useful abstractions, although the cold/stale
node miscredits the introduction of the metaphor. An insight can be valuable
while its attribution still needs correction.

A better challenge would give a factual observation plus an unresolved
interpretation, and show the observation kept intact while a tentative thought
develops or changes. It should not require the model to invent doubt or infer
a stable personal pattern from one example. These sources do not exercise
Tom's actual commit behavior or test sensitivity; the design conversation is
synthetic. Their absence here cannot grade capture of those behaviors.

The seven explicitly supplied confidence values and zero supplied emotion
values also matter for measurement: default stored values are not deliberate
model judgments. The local harness bypasses Scribe attribution; field-presence
counts cannot establish production provenance behavior.

Evidence: [per-node inventory](../eval/results/s1e_v31_cross_corpus_2026-09-08/quality_inventory.json),
[hand notes](../eval/results/s1e_v31_cross_corpus_2026-09-08/REVIEW-NOTES.md),
[field contract in the evaluated template](../eval/fixtures/s1e_guide_v3_1_reviewed_2026-09-08/template.md).

## 5. Edges can add meaning, repeat it, or contradict a simultaneous update

The session-impact/health `complements` relation adds the WHEN/STATE distinction;
it is not simply “these are dashboard features.” In contrast, many vision-hub
edges spend words restating membership. The latter are not necessarily useless
for traversal, but their descriptions do not all earn separate semantic value.

V3.1 R1's ambient edge says the “eight-feature dashboard vision does not yet
name” ambient presence. In the **same successful batch**, the vision node is
expanded to eleven dimensions including ambient presence. The closing reply
reports both the expansion and the new edge as successful. This is a relation
written against the old endpoint while another operation changes that endpoint.
It is not just an old edge forgotten on a later pass.

In V3 R3's LongMem merger, the edge describing baby gyms as a superseded phase
adds the same unsupported continuity as the revised node. It reinforces the
mistake along a second retrieval route. A card-message relation can also claim
the whole basket is compositionally complete when only the wording is settled.

Result inspection therefore needs to reconstruct endpoint claims after the
whole batch before judging the relation. An endpoint update and its relation
cannot be reviewed independently. This targets handoff patterns 1 and 2 and
adds a concrete same-batch case to the existing edge-repair challenge.

Mechanical direction must be assessed separately: this engine stores one
physical edge per pair. The saved opposite-direction `uses` request can appear
under the first edge's orientation. That observed harness/engine limitation
must not be scored as the model inventing that direction. No mechanism was
changed in this review.

Evidence: [same-batch calls](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_1_titles/repeat1/creative_design/window3/calls.json),
[final ambient and hub nodes](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_1_titles/repeat1/creative_design/window3/nodes_after.json),
[useful WHEN/STATE edge](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_titles/repeat1/creative_design/window3/nodes_after.json).

## 6. The journal sometimes preserves a reason not to repair

V3+tools R2 assigns Feb 5 to a new cousin/coworker context from the Feb 10
session. Its own closing note recognizes that the date came from older catalog
nodes and that the actual outing date is ambiguous. It says no correction is
needed until the date is confirmed. W3 carries that note and calls the context
clean. **Feb 5 is unsupported, not disproved as a possible outing date.**
Replacing it with Feb 10 would also invent precision; remove the unsupported
precision or distinguish report date from unknown event date instead.

V3.1 R2 notices a stored “(turn 14)” parenthetical, labels it a catalog violation,
then leaves a “swap on next touch.” R3 notices another and explicitly decides
it is minor enough not to correct. Both have replies available and working
write tools. The guide already says a named repair should happen now.
These traces support deliberate deferral/importance judgment, not an inference
that the model literally assumes infinite cycles.

R3 also resolves the missing-prior-journal-design note because the overlay node
is linked, while admitting the prior design reference remains unresolved and
asking whether a separate document exists. Connecting the overlay is not
evidence that the prior design was recovered. Conversely, keeping an unreported
purchase or genuinely missing design document open is appropriate. An empty
fresh brain cannot supply a nonexistent source document merely by calling tools.

The portable distinction is between missing evidence, an editable misstatement
already identified, and a resolved question with new supporting evidence. A
worked ending should perform those different actions. This targets patterns
5 and 7. The shared journal renderer and any runner continuation remain Tom's
gates; no change to either is implied by this analysis.

Evidence: [date notice](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_titles_new_tools/repeat2/longmem_unseen/window2/result.json),
[date carried](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_titles_new_tools/repeat2/longmem_unseen/window3/result.json),
[R2 deliberate deferral](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_1_titles/repeat2/creative_design/window3/result.json),
[R3 deferral and proxy resolution](../eval/results/s1e_v31_cross_corpus_2026-09-08/v3_1_titles/repeat3/creative_design/window3/result.json).

## What this changes about the next version

Keep the successful carriers: first-disclosure capture, the written comparison,
narrow field revisions, explicit evidence limits and concrete relation repairs.
Do not infer a V3.1 win from coverage, number of revised nodes or fields filled.

Before another version, turn the cases above into a small challenge matrix:
requested outcome versus proposed implementation; explicit versus absent
endorsement; same claim versus a separate occasion; missing date versus known
bad precision; a relation whose endpoint changes in the same batch; factual
observation versus a developing interpretation. Judge the resulting claim in
every independently retrievable surface, including type, and inspect what
later windows carry forward. Use unrelated scenarios for the worked examples.

Those challenges should score preserved supported details, unsupported
additions, correct lifecycle changes and useful distinct retrieval needs
separately. Word counts and operation counts remain diagnostic columns.
They should not collapse into a score that lets more nodes offset false facts.

The leading prompt hypothesis is to make the evidence decision the source of
the field values and final check, rather than let a global story or an earlier
clean verdict supply it. That means revising the existing comparison/example
and ending, not appending another long list of prohibitions. A second,
independent question is selection/atomization of advice; keep that separate
so a change in node count does not masquerade as a repair improvement.

These are hypotheses for Tom to review, not an authored next arm. The current
two source samples cannot establish large-context reach, correction propagation
under the historical substrate, natural behavioral-pattern capture or end-to-end
LongMem answering. The full historical matrix remains a later gate after a
small, correction-rich check. All evaluated arms and input hashes stay frozen.
