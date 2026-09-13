# S1E release objective and regression map

Current next stage: [author and evaluate V3.3](HANDOFF-S1E-V3-3-2026-09-11.md).
V3.3 was authored, reviewed and frozen on 2026-09-11 — see the
[authoring review](S1E-V3-3-AUTHORING-REVIEW-2026-09-11.md); its sanity and
three-arm transfer cells are complete — [results](S1E-V3-3-RESULTS-2026-09-11.md).
Tom requests a fresh session to weave many shapes into examples using the
existing challenges and probes. The working set carries the design proposal,
verified infrastructure and stale-tool cautions. Reduction was for focus, not
token usage; restore useful understanding while preserving revision gains.

Latest evidence: [current production versus V3.2](S1E-CURRENT-PRODUCTION-VS-V3-2-RESULTS-2026-09-11.md)
is complete (nine new production encodes, nine saved V3.2 encodes). Tom explicitly
clarified that the target is the complete MCP/prompt/supporting-code release;
its [integration scope](S1E-PRODUCTION-COMPARISON-AND-RELEASE-SCOPE-2026-09-11.md)
separates package-level replay evidence from remaining release work.

Tom's objective is an encoder that is better overall with the new revise
system: preserve details, facts, corrections, arcs, behavior and both voices,
then demonstrate useful gains on our tests and benchmarks without teaching
their answers. Refinement should converge on a release candidate. An isolated
wording defect must not displace this objective or erase earlier gains.

This restores the direction stated on September 7–8 and reaffirmed on
September 11. It is the current navigation document, not another instruction
block for the encoder. The existing [challenge checklist](S1E-CHECKLIST.md)
remains the detailed inventory. No prompt changes or model calls were made
while assembling this map.

## What exists and what has actually run

| Stage | Contribution to preserve when selecting or revising | Evidence and limits |
|---|---|---|
| Branch revise system and original candidate | Value-or-swap field updates, edges carried on revise, sweep of affected existing claims, pre-timeline gist | Historical candidate: 29/38 gold surfaces and 26/30 LongMem answer repetitions; historical production: 15/38 and 22/30. These are the prior stream's ledger results, not current-production measurements. Edge repair and absent-catalog reach remained weak. |
| Guide v1.2 | Four visible working lists; per-node field roll-call; depicted fetch followed by writes | Strong revise results on specific historical items, but create-side LongMem regression. Its 25/30 scoreable gold total cannot be compared as the same denominator as 29/38. Preserve the revise behavior; do not restore the my-turns-first creation gate. |
| V2 | First-disclosure facts before pattern thresholds; originating mixed conversation → integration → writes → inspection; later thought update and unspoken-pattern teaching | Small sequential diagnostics retain incidental facts, repair stale fields and preserve still-true details. Some unsupported dates, scope and residue decisions remain. These are bounded observations, not universal guarantees. |
| Compact Shapes / V3 with cues | Substantial template compression, retained example roles, section-local cues and ending strategy | Astra authored the compression options. The Shapes diagnostic retained tested coverage with a 30.05% smaller template than V2. Later edits increased size. Cue-package and compression results do not isolate every component's causal effect. |
| Generic tool descriptions | Shorter descriptions consistent with the operations and returned-result lifecycle | Compared in the 54-encode cell; observed nested relation failures were repaired. This does not establish production parity. Tool schemas and shared runtime changes are separate concerns. |
| Reviewed V3.1 | Compare old assertion → new evidence → what holds before marking fields; inspect persisted claims, including initially clean fields; permit repair before close | 54 encodes across three arms, two sources and three repetitions. Broad topic/fact coverage held; some arcs and revisions improved, while agreement inflation and other mistakes survived. |
| Frozen V3.2 | Preserve commitment/uncertainty range; conditional ideas beside firm facts; scoped failed-trial example; actual semantic-repair demonstration; corrected older examples | Nine new Sonnet 4.6 encodes completed against one familiar design conversation, compared with nine saved V3.1 encodes. Mixed result. No post-write repair occurred; one repeat omitted the first-build priority. Negative-trial teaching, wider transfer and historical benchmark parity remain untested. |

Historical gold/LongMem figures come from
`/Users/tpac/AgentsContext/s1e-field-coverage-gold/ab_2026-09-01_03/ops9/ADJUDICATION.md`.
Use its per-item rows and the handoff's linked corrections. In particular,
future-state reads can make runs VOID, and the original guide LongMem capture
did not actually carry the claimed lists-first preamble. Do not rederive or
overwrite old numbers to make the versions look comparable.

Recent evidence is in [compression results](S1E-GUIDE-COMPRESSION-EVAL-2026-09-08.md),
[V3.1 cross-corpus results](S1E-V3-1-CROSS-CORPUS-RESULTS-2026-09-08.md),
[content/lifecycle review](S1E-CONTENT-LIFECYCLE-REVIEW-2026-09-09.md), and
[V3.2 sanity results](S1E-V3-2-SANITY-RESULTS-2026-09-10.md). Those reports retain
the per-run paths, limitations and completed observations.

The [whole-memory reread of V3.2](S1E-V3-2-WHOLE-MEMORY-REVIEW-2026-09-11.md)
applies Tom's September 11 calibration across coverage, arcs, revisions,
voice, synthesis, fields, edges and cost. It supplements the earlier
failure-focused account without changing frozen scores or prompts.

## Capabilities to check together

| Dimension | What a candidate must continue to do | Evidence needed for the comparison |
|---|---|---|
| Facts and concrete detail | Keep first disclosures, incidental details, dates/amounts/names and later corrections. Uncertainty about meaning must not suppress the underlying fact. | Source-to-stored-knowledge coverage, including mundane facts in larger windows. Credit a usable quote as a carrier; do not demand a separate node per detail. |
| Arcs and developing knowledge | Preserve sequence, turning points, motivations, decisions, priorities, and unresolved developments. Keep distinct stories distinct. | Read successive snapshots and the actual Arc, not only the final topic count. Check what survives when one story is revised and another is introduced. |
| Revision and preservation | Reach the correct existing claim, update affected fields/relationships, keep still-valid details and useful history, avoid contradictory twins. | Actual operations and resulting state. Separately report target reach, correct changed fields, untouched valid claims and missed repairs. A greater revise count is not automatically better. |
| Voice, relationship and behavior | Preserve Tom's statements and expressive texture; keep my useful observations, advice, doubt and synthesis in my voice. Learn grounded behavior without requiring every interpretation to be endorsed. | Read the node together with quotes and relevant relations. Count meaningful synthesis as well as literal facts; distinguish my interpretation from Tom's commitment. |
| Evidence strength and scope | Keep firm evidence firm, plans as plans and trials bounded to their conditions. Retain useful unused ideas. | Check material meaning, including supported commitment and uncertainty. Do not reward blanket hedging or make every paraphrase discrepancy a release blocker. |
| Recall usefulness and value | Keep discriminating titles/questions/situations, meaningful relation descriptions and focused memories. Extra fields should add something useful. | Benchmark answers and actual retrieval alongside node review. Report duplication and word/token cost descriptively; avoid field, thought, quote, edge or source-ref quotas. |
| Execution and continuity | Complete needed reads/writes, use real outcomes, and carry genuine uncertainty without turning a no-mint verdict into future authority. | Saved request/result chains and sequential continuity. Distinguish tool errors, legitimate no-work windows, planned-but-unexecuted writes and semantic mistakes. |

These rows expose tradeoffs; they are not an invented weighted score or a
requirement for zero defects. Selection and tolerable regressions remain a
judgment for Tom using the whole evidence. Small tests have not yet established
every capability above for V3.2, especially unspoken behavior, realistic large
catalogs, missing-node reach, and downstream recall.

## Quotes and practical fidelity — Tom's September 11 calibration

Tom said: “if the quote keeps the sentiment better its a good enough situation.”
Evaluate what a future reader can recover from the whole memory. Exact voice
and sentiment retained in a quote deserve credit even when the prose is less
expressive. There is no need to polish every field to the same literary or
interpretive precision. Source refs are selective visibility when the episode
is part of the meaning, not a 100% coverage objective.

My operational distinction for the next review is:

- **Adequately preserved:** the whole node, including a useful quote, carries
  the intended meaning. A flatter paraphrase alone is not a material failure.
- **Preserved with ambiguity:** the right information remains available but
  another field conflicts with it. Credit retention and record the conflict's
  practical consequence; do not count it as total information loss.
- **Materially wrong or missing:** a priority/fact disappears, an unrelated
  story is overwritten, or wording would lead a reader to act on an unmade
  commitment or false state. Inspect and report that consequence explicitly.

For example, the saved aesthetic quote preserves the original sentiment and
speaker text despite a content attribution error; it is not wholly lost.
Dropping the prototype's first-build priority is a different defect. A D3
proposal called an agreed stack in situation preserves the proposal elsewhere
but could mislead implementation. These examples calibrate review of existing
outputs; they are not new examples to insert into the prompt.

The suggestion that we may be nearing Sonnet 4.6's limits is a reasonable
hypothesis raised by Tom, not a measured capability ceiling. The current model
choice remains 4.6. No model switch or extra rules follow automatically.

## Prior comparison plan — superseded by the completed comparison and V3.3 handoff

The plan below records the earlier reasoning for broadening the evaluation.
The nine-run current-production comparison has since completed. Tom now asks
the successor to author revisions and evaluate broadly using challenges and
example probes; follow the current handoff for the next action. The six-call
diagnostic remains held, and the historical wide gold/LongMem matrix remains
future work rather than a completed result.

The six-call worklist diagnostic is prepared but unrun. It would be a Sonnet
review of saved input/writes, with tools disabled. It is neither a fresh
encoding run nor a post-hoc interview asking why the encoder made its choice.
It cannot demonstrate overall improvement. Hold it while restoring the release
comparison; use it only if its result would change a candidate or release
decision. Tom's latest message directs recall before further probes or tests.

Recommended next work:

1. Use frozen V3.2 as the current contender, not a declared winner. Carry the
   capability map above into the existing quality rubric; do not author V3.3
   merely to fix wording in the familiar design conversation.
2. Prepare the bounded historical gold and ten-item LongMem comparison against
   production, with the original branch candidate as a useful archived
   reference. Resolve and pin the production reference before claiming it is
   today's deployed prompt. Reuse old measurements only where settings and
   substrate actually match; state mismatches instead of silently blending.
3. Review the concrete scope/cost before a large launch. Keep three repetitions
   and sequential state inside a repetition. Freeze comparable read state and
   inspect delivered prompts so a read from a later fixed node is not scored
   as an encoding win. This is the existing broader eval plan, not a mandate
   to rerun all five exploratory arms.
4. Put gains and regressions in one per-item, per-dimension report, including
   recall/answer performance and the practical quote-aware reading above.
   Follow with untouched material to check transfer. If source has been
   inspected for authoring, label it development data.
5. Bring Tom an overall promotion recommendation and its remaining limitations.
   A remaining mistake warrants another intervention when it materially changes
   that decision, not merely because it can be found.

This is a restored strategy, not authorization to run the large cell or deploy.
The new six-call diagnostic's automatic approval rejection is separate from
the already-approved and completed nine-encode V3.2 sanity. Do not describe
the nine-encode sanity as still pending.

## Branch and promotion boundaries

Worktree: `/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242`;
branch `claude/sweet-lichterman-ba9854`, recorded HEAD `9a1727f`.
Never edit the shared root checkout. No merge, activation or daemon restart
without Tom: merging is deployment.

Keep the shared closure/header/nudge, residue/edge-id catalog rendering,
runner continuation, production round-text trace storage, edge-repair
mechanism and confidence/label contract gates visible. Their presence in an
eval candidate does not imply production approval. Promote reviewed diffs
against the then-current default; never overwrite it blindly with an old
full-file candidate.

Conversation anchors: `287985a5` (this thread's original objective),
`6fc1ba6f` (Tom's earlier warning about spiraling into extra fixes), and
`805bab73` (current preservation/quote/release direction). Other-stream
measurements and Astra's authored compression are attributed above.
