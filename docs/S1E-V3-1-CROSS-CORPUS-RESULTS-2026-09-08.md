# V3.1 cross-corpus results

The approved revisions are authored, frozen, and evaluated. All 54 encodes
finished, with independent repeats running in parallel and the three windows
inside each repeat sequential. V3.1 preserves detail and performs more
revisions in the design sample, but **does not establish a node-quality win**.
Several new writes promote proposals into agreement; known defects still go
to the journal for later. Nothing was merged, deployed, registered or activated.

Follow-up: [content and revision lifecycle review](S1E-CONTENT-LIFECYCLE-REVIEW-2026-09-09.md)
traces successful and harmful revisions, field contradictions, selection
variance and journal behavior. It also separates the current advice-count zeros
from the historical LongMem runs that actually wrote nothing.

## What was compared

| Arm | Guide | Tools |
|---|---|---|
| V3 | Frozen compact V3 with cues | Original descriptions |
| V3 + tools | Same frozen guide | Generic revised descriptions |
| V3.1 + tools | Revised assertion comparison, both worked procedures, result inspection and ending | Same generic descriptions |

The guide revisions stay in existing carriers: targets, worked examples, gist,
Cadence, strategy and Finishing. They add no scene-specific rule, new field,
thought quota or reference quota. The existing JSON demonstrations and schemas
are unchanged. The gist remains before timeline; strategy/Finishing sit after
generated field and Arc/Review guidance in the system. Shared runtime wording
was not edited. [Authoring review](S1E-V3-1-AUTHORING-REVIEW-2026-09-08.md)
records the exact scope and placement.

The final evaluated guide is
[template.md](../eval/fixtures/s1e_guide_v3_1_reviewed_2026-09-08/template.md),
with [gist](../eval/fixtures/s1e_guide_v3_1_reviewed_2026-09-08/gist.md) and
[system diff](../eval/fixtures/s1e_guide_v3_1_reviewed_2026-09-08/system.diff).
V3.1 system+gist grew 2,322 characters (+2.34%); new tools remove 4,852 characters.
Combined static input is 2,530 characters smaller than original V3+cues. These
are character measurements, not a measured cost saving.

## Corpora and controls

Two existing sources, each 15 user/assistant pairs, each evaluated through
3 arms × 3 repeats × 3 sequential five-pair windows:

- `conv_004_art_design_extended`: repository synthetic design conversation,
  including product vision, technical proposals, personal-inference boundaries,
  ambient presence and prototype scope.
- LongMemEval oracle item `gpt4_f49edff3`: three complete sessions about baby
  gifts and sibling gifts. Outside the earlier ten-item slice; selected by
  complete-session length eligibility and deterministic hash after prompt freeze.
  It was the only item satisfying the imposed three-session/15-pair ceiling.

No Oren fixture was rerun. No source dialogue was edited. Arm/input hashes and
the quality rubric were fixed before outputs. Each sequence used a separate
fresh nursery-baseline copy, real writes and journal carry. The next window
received its actual persisted nodes. There was no S1R, S2, recall answerer or
full historical benchmark sweep. These two small samples cannot establish
generalization or statistically separate modest effects.

## Coverage is strong; correctness separates the runs

All 9 design sequences retained the 12 original gold topic surfaces. All 9
LongMem sequences retained the three queried events, sibling interests,
handbag budget/Zara interest, baby-gym narrowing, coworker basket/carrier
discussion and selected card message. This is **topic/claim presence**, not a
claim that every field was correct or that an answerer passed the benchmark.
The design gold itself calls the D3 topic a decision; source dialogue takes
precedence over that label when judging who agreed to what.

Deeper review exposed differences a presence score would conceal:

| Dimension | V3 | V3 + tools | V3.1 + tools |
|---|---|---|---|
| First gift arc kept distinct from coworker gift | R1/R2; R3 incorrectly replaces it | All 3 | All 3 |
| Unsupported precise date on cousin/coworker context | None; R1 preserves approximate timing | R2: Feb 5 copied from prior context; current session Feb 10, outing day unknown | None |
| Phone receipt added without source evidence | R1 asserts; R2 hedges inference | None | R1 asserts; R2 hedges inference |
| Named theme meanings altered | None | R1 merges neural/organic, invents default | R2 invents neural styling |
| D3 treated as agreed/confirmed somewhere in stored fields/relations | All 3 | R1/R3; R2 distinguishes proposal | All 3 |
| Distinct design nodes actually field-revised, R1/R2/R3 | 3 /1 /0 | 1 /0 /0 | 2 /2 /4 |

The last row measures changed fields, not calls or quality. V3.1 R1 actually
adds "D3.js is the agreed implementation library" to an existing graph node.
V3.1 R3 does make a useful update recording Tom's explicit mirror endorsement,
but leaves a different attribution error intact. More revision activity can
both help and harm memory.

## What the fields contain

287 final nodes were inventoried. I read their main fields, all actual field
revision pairs and persisted outgoing relation descriptions. Every populated
quote was checked against source; nonmatches were inspected. This is an author
review, not a blind judge. The per-node notes and exact evidence paths are in
[REVIEW-NOTES.md](../eval/results/s1e_v31_cross_corpus_2026-09-08/REVIEW-NOTES.md).

The main weakness is **claim status changing across fields**. For example,
V3+tools R3 prototype node `3b9bcdab` calls D3 a recommendation not yet confirmed
in reasoning, while situation says it is the agreed starting point. V3.1 R1
ambient node `5063c0cf` similarly stores proposed mechanisms as agreed in its
future-use cue. A cautious body does not protect a more assertive retrieval
field or relationship. Infrastructure nodes also treat assistant assertions
as verified availability because Tom did not contradict them.

| Field | V3, 93 nodes | V3 + tools, 98 nodes | V3.1 + tools, 96 nodes |
|---|---:|---:|---:|
| Content | 93 /6,047 words | 98 /5,665 | 96 /6,009 |
| Situation | 93 /2,055 | 98 /2,329 | 96 /2,427 |
| Reasoning | 93 /2,802 | 98 /2,933 | 96 /2,841 |
| Question | 57 /663 | 68 /853 | 63 /746 |
| Thought | 1 /38 | 0 /0 | 1 /29 |
| Their quote | 68 /1,257 | 70 /1,388 | 71 /1,387 |
| My quote | 44 /1,493 | 61 /1,841 | 57 /1,885 |

Cells are populated-node count /word count, aggregated across both sources and
all three repeats. Presence is descriptive, not a quality point.

- **Content/title:** usually specific, useful retrieval handles. Broader vision
  hubs repeat focused feature nodes and sometimes attribute every mechanism to
  Tom. Personal interests, amounts, stores and gift arcs survive well.
- **Situation:** often a useful future application, but also the place where
  a proposal becomes a requirement. V3.1's average situation is 25.3 words versus
  V3's 22.1; the extra words do not reliably preserve the right scope/status.
- **Reasoning:** includes good evidence limits—no purchase confirmed, unknown
  brother budget, source assertions need checking—but repeatedly uses silence
  as agreement. Turn numbers and encoding-policy explanations often replace
  durable support. Some reasoning is left stale when content expands.
- **Question:** generally a short useful future query; open questions can
  appropriately retain unmade decisions. No forced population needed.
- **Thought:** only two fields across 287 nodes, both workflow/future-capture
  notes rather than domain insight. Useful doubt also appears in reasoning and
  open nodes, so this is not an argument for a thought quota.
- **Quotes:** 13/371 are not normalized source substrings. Seven are minor
  punctuation/case/boundary changes; six silently condense or stitch passages.
  Many exact assistant quotes are whole-message repetition or generic greetings.
  Exactness alone does not justify their space.
- **Other fields:** no custom metadata or evolution-status values. All emotion
  values/labels are defaults; confidence was explicitly supplied on only 7 nodes.
  These must not be reported as 287 deliberate affect/confidence judgments.

No percentage target was imposed for source references.

## Size and marginal value

Final surviving authored text, including title, fields and quotes; whitespace
word counting. Repeated retrievals and bookkeeping are excluded. Relations are
counted separately below. Repetition columns are independent runs, not stages.

| Corpus /arm | R1 nodes /words | R2 nodes /words | R3 nodes /words |
|---|---:|---:|---:|
| Design /V3 | 16 /3,407 | 17 /3,302 | 18 /3,261 |
| Design /V3 + tools | 16 /3,236 | 22 /3,767 | 16 /3,162 |
| Design /V3.1 + tools | 17 /3,309 | 18 /3,760 | 17 /3,257 |
| LongMem /V3 | 16 /2,042 | 11 /1,357 | 15 /2,084 |
| LongMem /V3 + tools | 17 /2,149 | 17 /2,625 | 10 /1,199 |
| LongMem /V3.1 + tools | 10 /1,280 | 16 /2,341 | 18 /2,521 |

Pooled per-node medians are 199/190/204 words for design and 126/133.5/137 for
LongMem, in arm order. Full per-field and per-repeat distributions are in
[quality_summary.json](../eval/results/s1e_v31_cross_corpus_2026-09-08/quality_summary.json).

Standalone LongMem advice/reference nodes varied sharply. Each cell below
states that category's count out of **all nodes in the repeat**; zero means no
standalone advice nodes, not an empty encoding run:

| Arm | R1 | R2 | R3 |
|---|---:|---:|---:|
| V3 | 6 of 16 nodes; 40% of words | 0 of 11 nodes; 0% | 4 of 15 nodes; 26% |
| V3 + tools | 7 of 17 nodes; 49% | 10 of 17 nodes; 65% | 0 of 10 nodes; 0% |
| V3.1 + tools | 0 of 10 nodes; 0% | 7 of 16 nodes; 49% | 9 of 18 nodes; 55% |

This is not a count of automatically bad nodes. The carrier preference caveat
helps explain a specific choice; a giant shopping-category list or generic
greeting adds less future value. In V3.1 R1, the nursery event, cousin outing,
brother interest and handbag plan fit in10 nodes while keeping the key facts.
R3 uses 18 nodes for the same source, including nine standalone advice nodes.
Advice embedded in personal/arc nodes is not included in the standalone count.

Relations add 3,354 /3,847 /3,709 words across the three arms, respectively.
Some add real synthesis: session impact answers WHEN knowledge grew; health
maps answer its current STATE. Others spend 30–50 words repeating that a feature
belongs to the vision. Some overextend privacy into a ban on general ambient
activity. Edge descriptions also retain old feature counts after a hub expands.

## Journal finding and execution

The journal concern is supported by direct behavior, with an important
distinction between unknown evidence and an editable defect:

- V3+tools R2 knows its Feb 5 date came from older catalog entries, while the
  current session is Feb 10, but defers correction until the outing date is
  confirmed. Feb 5 is unsupported, not disproved as a possible outing day.
  It could remove that precision now without substituting another invented date.
- V3.1 R2 notices a stored turn coordinate and proposes fixing it "on next
  touch". R3 explicitly notices another and decides not to correct it.
- V3.1 R3 resolves an open about an earlier journal design because the overlay
  node is connected, despite admitting the earlier design is still unknown.

By contrast, keeping an unreported purchase open is appropriate. The problem
is not that journals exist; it is that noticing an editable error can count as
finishing, while an unrelated link can count as resolving uncertainty.

All 54 windows wrote, including all 27 LongMem windows (3–8 new nodes each).
Forty took 2 rounds and fourteen 3 rounds, with 122 captured
requests. Four recall_batch calls and zero get_nodes; all prior created/touched
nodes were already in the next catalog. Four calls reported nested relation
issues (seven rejected attempts); every one was successfully retried with
connect_batch in the same window. There were no model errors or truncations.
V3.1 still made title-based revise targets once despite revised generic ID
descriptions, so tool prose is not mechanical assurance by itself.

Reported model output tokens: V3 59,628; V3+tools 60,162; V3.1+tools 67,843.
V3.1 used 13.8% more output than V3 and 12.8% more than V3+tools in this cell.
Across all arms, input usage including cache reads/writes was 4,905,906 tokens;
output 187,633. No dollar-cost inference is made from character counts or token
totals. Cache mix and number of rounds differ.

## Limits and next decision

The input hashes, 54 request chains and 18 sequential snapshot chains verified;
the pre-output rubric/review hashes remain unchanged. Evidence is in
[integrity_audit.json](../eval/results/s1e_v31_cross_corpus_2026-09-08/integrity_audit.json).
The worktree remains on 9a1727f with no tracked or staged code changes.

Two harness limits matter before broader production claims. The raw local
dispatcher applies real mutations but bypasses S1Scribe attribution/postprocessing:
all saved nodes show default `encoding_source=anchor`. No arm emitted locked or
restricted provenance fields, so this cell compares claim/field content, not
production provenance policy. Also, one physical edge holds all relations on a
node pair: an opposite-direction relation request can be rendered in the
earlier edge's orientation. Direction issues must be compared with saved calls
before attributing them to the prompt. Neither runtime mechanism was changed.
Initial catalogs were empty and later catalogs small; full live-context load,
missing-catalog reach and the original stale-claim regression remain untested
by these sources. The full historical gold/LongMem comparison remains pending.

The [follow-up review](S1E-CONTENT-LIFECYCLE-REVIEW-2026-09-09.md) checks the
original proposed next principle against the evaluated prompt: the requirement
to preserve speaker, scope and evidence state across fields already exists.
Repeating that rule is not a new lever. The next hypothesis concerns how the
worked procedure derives field values and checks relations from the evidence;
the follow-up shows where the current procedure succeeds and where it fails.

Keep the generic tool candidate available; it is substantially shorter and
all observed mechanical errors were repaired, but this small cell does not
prove parity or improved quality. Do not select a winning guide by topic
coverage or revision count. Discuss the next shape with Tom, freeze it, then
use a small correction-rich corpus before the expensive historical matrix.
No prompt was revised from these outputs during this cell.
