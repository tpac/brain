# V3.2 sanity: less text, mixed claim fidelity, no demonstrated repair gain

The nine approved Sonnet 4.6 encodes completed. V3.2 is not a clear improvement
over V3.1 on this development conversation. It preserves much of the useful
design material and sometimes labels proposals more carefully, but agreement
and speaker errors survive. One repeat loses the explicit prototype priority
while retaining its technical details. The new repair example did not produce
a post-write repair call in any of the nine windows.

The strongest new diagnostic evidence is upstream of storage: in repeats 2
and 3, the first reply's `changes` list assigns the assistant's aesthetic
sentence to Tom. The writes and closing checks inherit that attribution.
The review is consistent with its own initial summary, but that summary is
already wrong. More caution in the ending did not prevent this in these runs.

Nothing was changed in the frozen prompt after inspecting outputs. Nothing
is merged, deployed or activated.

## Comparison and execution

One existing 15-pair creative-design conversation, three sequential windows
per repetition, three repetitions. Nine new V3.2 encodes are compared with
nine saved V3.1 encodes. Frozen parent, settings, tools, seed substrate and
initial user request without the gist matched before launch. The baseline
was recorded earlier; this is not a fresh randomized comparison and cannot
exclude unrecorded provider/environment drift.

Model: `claude-sonnet-4-6`; frozen medium effort, 12,288 output-token cap and
five-round setting. All nine new windows finished with DONE, two or three
replies, no API error, no reported output truncation and no partial tool
failure. Every window created nodes. Two first windows read before writing;
the other seven wrote directly. Every write was followed by the close.

The initial launch was rejected by automatic approval review before execution.
Tom then explicitly approved this frozen payload and nine-run transmission
to Anthropic; that approved launch completed.

| Descriptive measure | V3.1 repeats 1 / 2 / 3 | V3.2 repeats 1 / 2 / 3 |
|---|---|---|
| Final semantic nodes | 17 / 18 / 17 | 16 / 18 / 18 |
| Stored text words, including quotes | 3,309 / 3,760 / 3,257 | 3,068 / 3,513 / 3,328 |
| Nodes whose existing text fields changed | 2 / 2 / 4 | 1 / 2 / 3 |
| Stored outgoing semantic relations | 28 / 28 / 28 | 17 / 25 / 27 |

Both arms end with 52 nodes across repeats. Stored text is 10,326 → 9,909
words (4.0% less). Generated output is 41,847 → 38,930 tokens (7.0% less).
These are size observations, not quality scores or actual billed cost.
Relation counts use the existing inventory's outgoing view; the known
physical-edge orientation limitation still applies. No recall/S2/answerer
score is implied.

## What improved or held

- Repeat 3 explicitly keeps D3 as the proposed library in the prototype's
  content and situation and in the revised graph content. It also keeps the
  menubar widget and contextual notifications unconfirmed. This is useful
  narrower wording, though nearby field mappings remain described as agreed.
- Repeat 2 revises the personal-mirror node's reasoning to include the later
  explicit “Good principle” response. Evidence can update reasoning while
  otherwise appropriate content remains intact.
- All three repeats retain the main design areas: living/organic graph,
  attention and temperature, gaps, journal visibility, time-lapse, session
  impact, health map, personal reflection, themes, sound, peripheral presence
  and aesthetic intent. This does not mean every detail or decision survived.
- Concrete details remain useful: the four theme names, journal color mapping,
  graph field mappings, cluster metrics, sound-event ideas and personal
  edit/delete controls are carried. No blanket reduction to vague ideas occurred.

## Failures with useful evidence

**Agreement still changes between fields.** Repeat 2 prototype `da056c35`
calls D3 “my proposed implementation” in content and “the agreed prototype
stack” in situation. Reasoning converts Tom's move to aesthetic priorities
into treating the stack as settled. The source only explicitly selects graph
visualization plus temperature colors as the first prototype scope.

Repeat 2 gap node `b27415f8` is an even cleaner contradiction: content says
“Agreed approach,” while reasoning says the sub-proposal was not explicitly
affirmed and is not jointly settled. Repeat 3 journal node `857c1ca0` likewise
keeps “Proposed” content beneath an “agreed rendering approach” situation.
That situation survives all three windows. These are actual stored words,
not judgments inferred from the node's type.

**Ownership errors begin in the initial evidence summary and spread.** The
assistant says “The aesthetic IS the product for something like this.”
Repeats 2 and 3 assign it to Tom in `changes` before calling tools. Repeat 3
then stores it in a new aesthetic node (`0e872ceb`) and revisions to the
profile (`2409bfd7`) and prior aesthetic node (`0871b943`), with edges giving
the misattribution additional reach. Its closing check explicitly confirms
that the sentence is Tom's and reports no overstatements.

In repeat 1, the four theme names are preserved correctly but reasoning
says they are the assistant's; the user supplied all four. Correct raw quotes
elsewhere do not automatically correct the interpreted fields.

**A concrete new priority disappears during consolidation.** Repeat 1 first
lists Tom's graph/temperature-first choice. It decides to fold the prototype
into graph node `492707b8`, appending D3 and field mappings but omitting that
this is the selected first build. The final semantic nodes do not retain
the priority explicitly. The source and encoding trace still contain it;
the semantic write does not. V3.1 retained the priority in all three repeats;
V3.2 retains it in repeats 2 and 3.

**Degree and scope still shift.** The source assistant proposes “optional
and quiet by default” audio. V3.2 repeats 1 and 3 turn that into an off-by-default
requirement in situation; repeat 1 calls it non-negotiable. The earlier V3.1
repeat 2 had this same strengthening, so this is a recurring failure with
two current instances, not an entirely new class. Repeat 1 also connects
theming to the personal-inference privacy rule: its edge retains a conditional
personal-data clause, but applying a `governed_by` relation to ordinary theming
is an unsupported extension absent such a feature in the source.

**Successful tool results still end semantic work.** No new window made a
second write after inspection. Zero repair calls alone would be fine if the
claims were correct; the contradictory fields above show that defects were
available to catch, including in the same window that created them. This
does not prove that repair is impossible or that a fixed two-write cadence
would solve it. The observed check did not produce a repair here.

## Fields, content value and journals

Title, content, situation and reasoning are present on all 52 nodes in both
arms. Questions occur on 45 → 41 nodes, explicit event-time writes on 45 → 51,
thought on 1 → 1, and explicit confidence writes on 1 → 5. No custom metadata
dimensions are authored. Default neutral emotion and default confidence are
not counted as deliberate field use.

Actual field-change events: V3.1 content 9, situation 4, reasoning 2;
V3.2 content 5, reasoning 4, situation 0. Neither arm changes an existing title
or question in this slice. Changes include enrichment, not only correction;
more revised fields would not by itself establish better memory.

The one V3.2 thought is a 54-word speculation about the unlocated journal
design and when to ask/search for it. It supplies workflow continuity, not
a new substantive reading of the design conversation. Naming thought in the
gist did not yield a demonstrated quality improvement. The 133-word open
journal-spec node in repeat 2 is a separate retrieval handle for a missing
dependency; that can be useful, but it is not newly recovered design knowledge.

Several fields add distinct future use: questions locate the prototype or
session impact view, and situations identify implementation work. Their
main defect is excessive certainty: “no infrastructure needed,” “agreed,”
“specified,” or “non-negotiable” can exceed the source. Extra explanation
does not make such a trigger safe. The 230-word profile in repeat 3 gains
theming and ambient context but also repeats the mistaken aesthetic attribution.
The 219-word ambient-vision node is a stronger example: it retains the user's
vision and the assistant's concrete proposals with their different status.

Two audio-to-ambient relations in repeat 2 largely repeat the same information
under different relation names. The reduction from 84 to 69 stored outgoing
relations therefore cannot simply be read as lost quality, nor as better
selectivity. Retrieval effect remains unmeasured.

Automated exact/whitespace quote checks flagged one nonmatching quote in each
arm. V3.1 concatenated two source spans without an omission marker. V3.2
rewrote the start of the user's journal question inside `their_raw_quote`.
Neither count captures the more consequential ownership errors in content
and reasoning, which required reading those fields against the dialogue.

Journals remain mixed: unresolved design questions are often appropriate,
but repeat 1 emits a `resolved` entry whose text says “not resolved; re-open,”
and repeat 3 resolves the journal thread because a “residue run has concluded”
while carrying the same doubt forward. These do not establish that the missing
spec was found. No literal infinite-session belief is inferred from this.

## Interpretation and next diagnostic

This sample supports keeping V3.2 as an experiment, not promoting it as an
established improvement. It changes vocabulary and node types—more plans and
designs—but those labels do not ensure faithful claims in each field.

The next useful diagnostic is to locate where an evidence summary first
loses speaker, modality or scope, and whether a fresh comparison against the
actual source catches the same saved errors. The current evidence points
to the construction of `changes`, followed by a closing review that accepts
its premises. That is a hypothesis to test, not a verified cause or an
authorization for another model run. No V3.3 text was written here.

All 52 new final nodes and their fields/edges were read, all actual field
revisions inspected, and quote exceptions checked. The baseline uses the
previous full review plus focused source checks here. This is an author
adjudication, not a blind judge. The familiar creative source exercises
agreement and cumulative revision; it does not test the new negative-trial
teaching or demonstrate transfer. Later untouched material remains necessary.

## Reproducible evidence

- `eval/results/s1e_v32_semantic_sanity_2026-09-10/`: completion, frozen-input
  manifest, matching-input preflight, source and all nine exact request/result
  chains, persisted before/after nodes and continuity.
- `comparison_inventory.json`: old/new sequence counts, windows, all text-field
  changes. `quality_inventory.json`: per-node values, word counts, fields,
  quote checks and relations. Each repetition has `quality_packet.md`.
- `ADJUDICATION.md`: compact per-repeat claim ledger with source locations.
- Read-only reproduction: `./dev python3
  eval/fixtures/s1e_guide_v3_2_2026-09-10/analyze_saved.py`.

The old source's `prior_gold` incorrectly calls D3 chosen. The manual rubric
already distinguished it as proposed; no frozen source or prior score was
rewritten. The known eval dispatcher bypass of Scribe attribution and physical
edge-direction limitation remain. No lock/privilege or production-fidelity
improvement is claimed.
