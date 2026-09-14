# S1E five-arm freeze — ready for Tom's review

All five arms are authored and sealed with SHA-256 hashes. Local semantic and
mechanical review is complete. **No new model evaluation has run.** Tom's joint
review remains the gate before the historical comparison. Nothing was committed,
merged, deployed or registered; tracked runtime code is unchanged at `9a1727f`.

## The set and its size

Counts are characters, not tokens. Full system includes generated field
reference, Arc/Review, shared Finishing and, where present, the ending strategy.
All arms also use the same 4,372-character v2 gist, tools and input corpus.
Conversation and catalog size is additional and varies by capture.

| Arm — link opens the complete assembled system | Template | Ending strategy | Full system |
|---|---:|---:|---:|
| [V2 reference](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/eval/fixtures/s1e_guide_freeze_2026-09-08/frozen/v2_frozen.system.md) | 119,926 | — | 126,694 |
| [V2 revised](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/eval/fixtures/s1e_guide_freeze_2026-09-08/frozen/v2_revised.system.md) | 120,974 | — | 127,742 |
| [V3](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/eval/fixtures/s1e_guide_freeze_2026-09-08/frozen/v3.system.md) | 85,343 | — | 92,111 |
| [V2 revised + cues](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/eval/fixtures/s1e_guide_freeze_2026-09-08/frozen/v2_revised_titles.system.md) | 123,815 | 737 | 131,321 |
| [V3 + cues](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/eval/fixtures/s1e_guide_freeze_2026-09-08/frozen/v3_titles.system.md) | 87,385 | 737 | 94,891 |

V3's template is **29.5% shorter than revised v2**; the complete system is
27.9% shorter. Revised v2 grows only 1,048 characters (0.9%) over original v2.
V3 adds 1,454 characters to Astra's 83,889-character Shapes proposal. Its final
85.3K size is slightly above the rough 84K target.

The cue packages add 3,579 / 2,780 system characters respectively, including
the strategy and separator. They contain 38 / 27 one-line pointers immediately
under existing authored headings. Removing them restores the exact parent.
The ending strategy follows generated references and Review, immediately before
unchanged Finishing. It is an instruction for responding to results, not a
message injected after each tool response. API tools are a separate request field.

## What is new versus revised

The base versions revise existing teaching: Reading, field consistency, the
Mira later-window example, planned-work language, temporal examples and revise
examples. The central principle is to preserve observations, develop useful
interpretations and keep their support visible as evidence changes.

Only the cue variants add a navigation layer: local pointers plus the
737-character ending strategy. The four working lists remain the planning
artifact. There is no new scratchpad requirement, mandatory thought field or
source-ref coverage target. The source flag remains selective.

Review repaired several examples that undermined that principle:

- A planned card could read as an existing artifact; its title, content,
  reasoning and edges now all describe agreement.
- The later window now narrows an interpretation while preserving Mira's new
  framing routine and earlier hosting facts. Uncertainty does not become rejection.
- A PT estimate no longer becomes clearance, and approximate dates no longer
  imply exact intervals.
- The sweep repairs its stale dependency sentence, preserves the actual quote,
  grounds the successor decision, and changes the moot merge question's type.
- Moving the surfacer into the daemon now repairs its title too. The sandbox
  lexicon retains two observations without generalizing to every destructive verb.

The revised-v2 comparison therefore tests a revision package. V3 tests the
compressed structure with matched intended teaching. The cue comparison tests
local pointers and ending strategy **together**, not their separate effects.

## Review evidence

The offline checker passed before sealing and again against the sealed inputs.
It checks deterministic authoring, frozen parents/gist, tool schema identity,
actual runtime assembly and ending placement; parses seven JSON call blocks,
six sweep operations and three revision-ladder operations; uses the real field
and swap validators; and simulates the central revisions and preservation cases.
Seven negative checks reject substitution, suffix/closure drift, bad swaps,
unknown sibling targets, changed input hashes and paths outside the worktree.
Network connections were disabled and no database instance was created.

Both revised bases carry the same twelve revise operations. Field counts:
title 9, content 10, situation 6, reasoning 4, question 1, event_time 2, type 2,
thought 1, connect_to 2. A thought-only update and a situation-only addition
exercise revision without rewriting title/content. `source_refs` is absent
from revisions deliberately: omission preserves existing source flags.

This is field/structural verification and semantic review, not complete JSON
Schema validation, real dispatch or evidence of model performance.

Detailed [section reverse pass and challenge coverage](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/eval/fixtures/s1e_guide_freeze_2026-09-08/semantic_review.md),
[v2 changes](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/eval/fixtures/s1e_guide_freeze_2026-09-08/v2_revised.diff),
[v3 changes from Astra Shapes](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/eval/fixtures/s1e_guide_freeze_2026-09-08/v3.diff),
and [machine review](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/eval/fixtures/s1e_guide_freeze_2026-09-08/frozen/offline_review.json)
are saved beside the set.

## The remaining discussion before eval

The unchanged gist still describes read → write → close rigidly, while the
templates permit another repair write and a legitimate zero-work skip. Shared
Finishing also describes the expected sequence. This tension is held constant
across all five arms; the ending strategy does not eliminate it. My recommendation
is to keep the comparison fixed and record skip/repair behavior explicitly.
Changing that shared contract would need a separately labeled set and Tom's gate.

Missing catalog IDs remain a rendering problem, and all original promotion and
shared-runtime gates remain Tom's. This freeze does not wire corpus runners;
they must consume the assembled systems and include `arm_sha256` in their cache
identity. A template-only override would put the strategy in the wrong place.

Start our review with Reading and the later-window example, then inspect the
local cues and final strategy in the assembled V3 cue arm. After we are happy
with those choices, verify actual outgoing requests and run the historical
gold/reach/longmem matrix from the
[comparison plan](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/docs/S1E-NEXT-COMPARISON-2026-09-08.md),
followed by broader independent corpora. No new scores are available yet.

## Freeze identity and continuation

Manifest: `eval/fixtures/s1e_guide_freeze_2026-09-08/frozen/manifest.json`

SHA-256: `1ce89a51672dcbdacad5554d8a8c00c61a2097a21b94bd72b2230ddada407829`

From the authorized worktree:

```sh
./dev python3 eval/fixtures/s1e_guide_freeze_2026-09-08/review_and_freeze.py
```

This verifies the set without a model call. The generator refuses `--write`
after sealing. A further text revision belongs to a new set; keep this reference
and the historical ledgers intact.
