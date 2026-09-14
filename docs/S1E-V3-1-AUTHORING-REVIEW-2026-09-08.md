# V3.1 guide revision and cross-corpus cell

Tom authorized the comparison/example and stopping revisions, asked for other
corpora to avoid tuning to Oren, and requested deeper node/field/value analysis.
He then allowed independent repetitions to run in parallel. Windows inside
each repetition still carry state sequentially.

The reviewed freeze is `eval/fixtures/s1e_guide_v3_1_reviewed_2026-09-08/`.
The earlier `s1e_guide_v3_1_2026-09-08/` is the first authoring snapshot; it
was not evaluated. Review caught the second worked example and its gist excerpt
still demonstrating bare verdicts; those existing lines were revised in the
reviewed freeze. No changes were made after reading the selected test dialogue.

## Changes and placement

| Existing carrier | Revision | Purpose |
|---|---|---|
| Targets in the gist and both worked examples | Old assertion → new evidence → what now holds, then field verdicts; group fields sharing an assertion | A clean verdict must follow a comparison instead of excluding itself from verification |
| Result inspection in Cadence, example and gist | Reconstruct persisted claims from before-state and successful changes, including untouched fields | Detect omissions that the original plan failed to name |
| Final strategy and Finishing | Read, write, inspect, repair as needed, then final Arc/Review | Remove write→automatic-close pressure |
| Actions/generated field reference | Restrict swaps to writable text fields | Match the unchanged tool schema |

The final strategy remains after generated field/Arc/Review instructions;
Finishing remains last in the system. The gist stays immediately before the
timeline. These are frozen eval assembly changes, including the ending and
generated-field sentence; shared `trace_contract.py` and `contract.py` were
not edited. All JSON tool demonstrations, field policies, source-ref selectivity,
thought optionality, tool schemas, tools, model, effort and limits remain as
before except for the separately identified generic tool-description arm.

| Arm | System chars | Gist chars | Tool JSON chars |
|---|---:|---:|---:|
| V3+cues, old tools | 94,891 | 4,372 | 31,683 |
| V3+cues, new tools | 94,891 | 4,372 | 26,831 |
| V3.1+cues, new tools | 96,344 | 5,241 | 26,831 |

System+gist grew 2,322 characters (2.34%); tools reduce the combined static
request by 2,530 characters versus original V3+cues. Character counts are not
token or quality claims. V3.1 is still substantially smaller than V2.

## Review against the challenge approach

- E1/E10: read the instruction layers together. Repair-before-close now agrees
  across revised gist, Cadence, strategy and final clause. Text-swap scope agrees
  with tools. Thought/source-ref selectivity is preserved; no fill quota added.
- E2/A3/A4: both existing worked procedures carry the new comparison. The Mira
  scene demonstrates changed booking claims and justified unchanged date,
  question, access uncertainty and dated correction. The branch-abandonment
  example demonstrates the same method across several node/edge claims.
- E3: coverage maps to those two examples, targets, result inspection and
  closing control. No new example topic, field or mandatory list is introduced.
- E5/E6/E7: comparisons share fields where they assert the same thing; this
  limits output ceremony. The repeated placements serve different moments:
  procedure in the gist, demonstration in examples, reactivation at closing.
  Whether that repetition earns its tokens is an eval question.
- E8/E9: every JSON demonstration is byte-identical to its predecessor; exact
  IDs/swaps are not rewritten. Review will inspect untouched quality dimensions,
  including attribution, useful thought, exact quotes, time, and edge meaning.

This was an author review, not a blind side-agent review. V3.1's unchanged
one-read-round limit and existing journal header/nudge remain explicit limits;
there is no runner-added continuation. Shared-source promotion remains Tom's
gate. No deployment or merge is authorized.

## Next cell and review contract

54 encodes: three arms × three repetitions × three windows × two existing
corpus samples. Each source supplies 15 unmodified user/assistant pairs.

1. Repository creative-design corpus, `conv_004_art_design_extended.json`:
   a complete design conversation split at five-pair boundaries.
2. LongMemEval `gpt4_f49edff3`: three complete five-pair sessions. Selected after
   the reviewed prompt freeze by length eligibility and deterministic hash order,
   excluding the previous ten-item slice. Only one of the 500 oracle items
   satisfied the complete-three-session/15-pair limit.

These are small encode-quality samples from two corpora, not full benchmark
scores. No Oren fixture is rerun. The fresh nursery baseline is shared and
closed before copying; each repetition has its own process/DB/output paths.
Windows use real writes, reads and journal lifecycle, with conversation-date
journal rendering. S1R/S2 and answer generation are outside this cell. All prior
created/touched nodes are supplied to later catalogs, so missing-catalog reach
is not tested. Source traces are preseeded for stable IDs, but future dialogue
is not shown and no episode-search tool is available.

`eval/results/s1e_v31_cross_corpus_2026-09-08/manifest.json` pins inputs and
substate; `corpus_cell.py` owns prepare/preflight/run. Up to six independent
repetitions overlap; each runs its windows sequentially.

The pre-output [quality rubric](../eval/fixtures/s1e_guide_v3_1_reviewed_2026-09-08/QUALITY-RUBRIC.md)
defines claim coverage, source support, per-field contribution, word counts,
retrieval use, useful synthesis, duplication and unsupported expansion.
Field presence and node length are descriptive metrics. Semantic judgments
must name what the node enables later and cite actual stored claims.
