# S1E guide v2 — candidate and reverse-pass review

**Drafted and statically checked; initial behavioral checkpoint now complete,
not promoted.** See [the eval results](S1E-GUIDE-V2-EVAL-2026-09-08.md) for the
two gold attempts and fresh two-window comparison; full gold/longmem remain
outstanding. This document records the authoring pass. Tom
approved authoring the template/gist candidate after the structural review
and storyboard. Work remains on `claude/sweet-lichterman-ba9854` at `9a1727f`.
The default prompt, renderer, runner, shared journal contract and environment
files are unchanged. No interaction was registered and nothing was merged.

## Read the result

- [Template candidate](../eval/candidate_prompts/s1e_guide_v2_2026-09-08.md)
- [Gist candidate](../eval/candidate_prompts/s1e_gist_guide_v2_2026-09-08.md)
- [Exact v1-to-v2 diff](../eval/candidate_prompts/s1e_guide_v2_2026-09-08.diff)
- [Static audit snapshot](../eval/candidate_prompts/s1e_guide_v2_2026-09-08.audit.json)
- [Structural review and prior storyboard](S1E-STRUCTURE-REVERSE-PASS-2026-09-08.md)

The main change is teaching how a mixed conversation becomes memory, rather
than assuming discovery happened before the example starts. Reading owns
the distinction between observations and interpretations. The canonical
episode demonstrates working lists, a read, writes, their results, and a
later thought update. The gist activates that work against the current input.

## What changed, where, and why

| Location in v2 | Actual change | Failure pattern / challenge |
|---|---|---|
| Reading, line 69 | Replace the short opportunity paragraph with discovery before storage: first disclosures, both voices, evidence versus interpretation, plans versus completed actions. Preserve the value of rejected alternatives. | Patterns 6 and 8; D1/D3/D7/D9. A fact and an inferred regularity have different evidence requirements. |
| Emerging patterns, line 147 | Scope the existing three-turn threshold to inferred rhythms; facts survive while their broader interpretation develops. No threshold number was changed. | Patterns 6 and 7; D9. Explicit preference is already knowledge. |
| Canonical episode, line 962 | Replace the output-first canonical batch and recap with the fictional print-swap source, prior claims, four lists, an edge-only read, three remembers and two revises, and a returned-results excerpt. | Patterns 1/2/4/5/6/8/9; A1/A2/E12/E13/E20. Decisions and execution become inspectable. |
| Later window, line 1112 | Preserve the new teaching practice and revise only the prior interpretation's thought. The new fact connects to the existing interpretation by id. | A10/D7/D9; a thought can change without rewriting still-true content or swallowing a fresh fact. |
| Supporting contrasts, line 1143 | Rework the existing action-derived finding, old-to-old connection, fully answered open, emotional moment and consequential quote into shorter carriers with shown evidence. | T1/T6; A4/A10/D3. These lessons must survive the canonical replacement. |
| Unspoken pattern, line 1639 | New notebook-restoration example: three choices support a scoped interpretation neither voice stated; alternative explanations remain visible. Placed beside the existing confirmed-interpretation example, after field/tool teaching. | E24/D9/A1; covers the storyboard's explicit-preference limitation. |
| Actions and detail/meaning | Revise “same topic” to the same claim; make fact-plus-meaning pairing conditional on differentiated retrieval. | D7/D13; prevents both over-merging and inventing an abstraction to justify a fact. |
| Cadence residue sentence | A no-mint verdict never becomes the next run's residue policy. | Pattern 7. This is template wording, not a change to the shared journal contract. |
| Gist — three paragraphs | Revise `changes`, `new`, and the final check. Everything else stays byte-identical. | Discovery → storage decision → observed result. `new` retains its create-only meaning. |

The storyboard's lean-header read was corrected during authoring. A rendered
catalog header already comes with full content; the candidate places the
earlier incident only in an edge line, so its body really is missing. This
changes the fictional input, not production rendering.

## Preserved teaching and placement

The opening identity stance is unchanged. The revise ladder, measured guide
sweep, edge section, and existing identity examples are unchanged. The
second-misreading lexicon example remains intact; the new inferred-pattern
example sits beside it rather than claiming the two teach the same thing.

The gist's first-reply/round contract, `targets`, `fetch`, sweep exemplar,
write mapping and selective refs line are unchanged. The gist still belongs
before the timeline; the main canonical example still occupies its existing
section. No assembly change moved instructions or added another model call.

The main episode and small contrasts preserve different retrieval purposes:
the board's location, the adopted prep card, the hosting interpretation,
the measured queue effect, an emotional moment, and a phrase with meaning.
Neither an adopted plan nor the inferred pattern requires inventing a voice
quote. The separate measured finding demonstrates that my work can deserve
memory without another participant ratifying it.

## Size: the storyboard budget was exceeded

| Text | Guide v1 | Guide v2 | Change |
|---|---:|---:|---:|
| Template | 112,026 chars | 119,926 chars | +7,900 (+7.1%) |
| Gist | 4,352 | 4,372 | +20 (+0.5%) |
| Canonical slot, including its supporting contrasts | 16,924 | 20,218 | +3,294 |

The inferred-pattern example adds 3,691 characters including its heading.
Together with the canonical growth, examples account for 6,985 of the
template's 7,900 added characters. This is mostly additional demonstration,
but it is still additional input. The proposed same-size canonical budget
did **not** pass. This candidate cannot be described as a compression win.

Line-aligned comparison leaves 93,159 template characters unchanged. There
are 328 added and 150 removed lines: substantial rewriting of the example
family, plus one new example and bounded prose revisions. The gist changes
three existing paragraphs and leaves 3,573 characters byte-identical.

The trade is explicit: showing source evidence, a returned read, and a
second window costs space that output-only examples did not use. Whether
that extra teaching earns its cost requires behavioral evidence. Shortening
unrelated working material merely to hit the original estimate would add a
second, unmeasured change to this candidate.

## Static checks and example census

Read-only assembly accepts the candidate as the supplied template and
appends the same 6,768-character runtime suffix as v1. Default prompt and
runtime files have no diff from HEAD. The baseline candidate hashes still
match the oriented guide files.

Seven newly authored JSON call blocks contain fourteen operations and one
read call. Their argument structures were checked against schemas obtained
from the branch's `_get_tool_schemas`: required fields, operation variants,
types, references, enums and declared bounds. This was a local structural
checker, not a live dispatch test or a third-party full JSON Schema suite.
Eleven raw-quote spans occur in the shown source text. All twelve new
`connect_to` descriptions fit the stated 120–180-character band; id targets,
sibling titles and selective trace refs have shown sources. Exact event
swaps match once and preserve its time and browsing option.

The census below counts **worked revise operations in the actual candidate
files**, including the unchanged labeled BAD example. It excludes the
returned-results excerpt. These are teaching counts, not encoding results
or a recalculation of the hand ledger's benchmarks.

| Revised field / operation | v1 | v2 |
|---|---:|---:|
| Revise operations | 11 | 13 |
| title | 7 | 8 |
| content | 8 | 11 |
| situation | 3 | 6 |
| reasoning | 1 | 4 |
| event_time | 1 | 2 |
| question | 0 | 1 |
| thought | 0 | 1 |
| type | 1 | 1 |
| connect_to | 1 | 2 |

The added `connect_to` on revise is a partial-resolution relationship, not
another edge-repair example. The existing edge-description repair remains
unchanged. The thought-only operation exercises neither title nor content.
Partial and full open closure are both demonstrated. Numeric confidence and
evolution-status additions remain outside this candidate; a nonzero count
alone would not establish good calibration or good thought quality anyway.

## Reverse-pass findings and limits

- **Evidence → fact:** the board is preserved on first disclosure. The
  pattern threshold cannot be used to refuse it. Under realistic load it
  may still be missed; this example does not measure that transfer.
- **Evidence → interpretation:** the host explicitly confirms a distinction;
  the notebook choices support an inference with a competing explanation.
  These are different examples because explicit confirmation cannot teach
  the unspoken case by itself. Neither covers every emotional or long-term
  behavioral arc.
- **Read → decision:** the read supplies missing historical evidence. It
  does not prove that prose will make residue-only IDs get fetched. The
  rendering mechanism remains Tom's gate.
- **Plan → result:** the output excerpt gives the close something to inspect.
  The runner still stops on a tool-less reply. Its continuation mechanism
  and production storage of round texts are unchanged and gated.
- **Doubt → continuity:** the access question and living thought have homes.
  The no-mint rule still needs sequential testing against a prior skip
  verdict; naming it is not proof that it executes.
- **Preservation:** all stale booking/access fields shown are repaired while
  still-true details remain. Existing surface-repair examples retain their
  measured teaching. Their known omitted-before-state issue and edge repair
  mechanism question are not silently declared solved here.
- **Attention cost:** the gist stays essentially the same size; the template
  grows. This is the material unresolved design trade before promotion.

No gold, longmem, independent model probe or new IsolatedBrain was run for
this authoring pass. Next behavioral work must use fresh mixed/sequential
cases under realistic input load, not the teaching scenes. Verify the actual
captured composition first, especially the known lists-preamble environment
reset. Existing guide and best prior candidate remain distinct baselines;
compare gold per item with VOID reads handled as recorded in the ledger.

## Fingerprints

- Template v2: `4637785918cab686ec9abd29096def0691fb9aaf443b63d893736c4883551edb`
- Gist v2: `54cf9c77132ce3eac680f9703f71a612293b26117c6886048ba9f001634b3063`

The diff compares the two candidate generations. It is not a patch to apply
blindly to the live default; eventual promotion must account for the current
code and Tom's merge decision.
