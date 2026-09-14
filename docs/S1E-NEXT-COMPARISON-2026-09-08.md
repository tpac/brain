# S1E next comparison — revised guidance, compression and navigation

## Status and Tom's direction

Tom proposed comparing v2 enhanced with the current findings against v3, plus
an enhanced-titles version of each. He suggested light section-local pointers
and an overall strategy reminder after the tools, then the corpus from the
other stream, followed by broader normal corpora to check overfitting.

All five candidates are now authored, locally reviewed and frozen for Tom's
joint review. No new model runs have started. The results, exact sizes,
assembled-system links and known limits are in
`docs/S1E-ARM-FREEZE-REVIEW-2026-09-08.md`. The frozen manifest SHA-256 is
`1ce89a51672dcbdacad5554d8a8c00c61a2097a21b94bd72b2230ddada407829`.
Tom's freeze-before-eval directive is the active gate; the corpus plan below
is still future work. Runner integration must preserve the frozen assembly
and use the complete arm identity before any model cell is launched.

The previous 12-encode diagnostic is complete, with its hand ledger in
`docs/S1E-GUIDE-COMPRESSION-EVAL-2026-09-08.md` and brain finding `68dd8ff8`.
It used one synthetic sequence, not twelve independent conversations.

## Candidate matrix

| Arm | Base and changes | Comparison it enables |
|---|---|---|
| V2 frozen | Existing 119,926-character v2 template, unchanged | Reference for whether the new guidance helps |
| V2 revised | Full v2 structure, with the current evidence-state lessons woven into existing sections/examples | Revised versus frozen isolates the guidance revision |
| V3 | Astra's compact Shapes structure with the same substantive lessons | V3 versus V2 revised tests compression with matched intended teaching |
| V2 revised + cues | V2 revised, plus section cues and final strategy reminder | Tests the navigation package on the long template |
| V3 + cues | V3, plus the same section cues and final strategy reminder | Tests the navigation package on the compact template |

Freeze and fingerprint all files before model runs. Preserve the unchanged v2
gist and its pre-timeline position across these five arms. Hold tools, model,
effort, read/write limits, lists preamble, shared contract and corpus constant.
The cue variants differ from their direct parent only in headings/local cue
lines and the strategy reminder. Report the added characters; unrelated cuts
to compensate would introduce a second difference. V3's base target remains
roughly 84K characters. No source-ref percentage target.

The cue comparison measures **section cues plus the ending reminder together**.
It cannot establish which component helped. If the package improves behavior,
a later headers-only/footer-only ablation can answer that smaller question.
Do not silently expand the initial cell into all combinations.

## What changes in the base teaching

One principle: retain the observation, keep a useful interpretation at its
actual level of support, and advance a claim only with evidence of the change.
Revise existing Reading, later-window thought, planned-work and result-inspection
examples rather than append another independent instruction bank.

- A person's uncertainty neither confirms nor disproves my interpretation.
  My own useful synthesis need not await their ratification, and must not be
  attributed to them. A later fact can narrow it without erasing earlier facts.
- A promise, work in progress and an accomplished result are different claims.
  An unrelated completion or a passed date does not supply missing evidence.
- Titles, situations, reasonings, dates and relationship descriptions carry
  claims as well as content. Preserve speaker identity and evidence state
  across those surfaces; a qualified content paragraph cannot cancel a false
  title or edge.

This addresses the ledger's strings-versus-claims, node-versus-field,
voice-seesaw and residue-as-authority failure patterns. The diagnostic shows
the errors; it does not prove a particular prompt passage caused them.
Use different scenes from the test fixtures in any revised demonstration.

## Section-local cues — proposed shape

Each cue tells the reader what decision to make while in that section.
Keep it short; it should activate the existing teaching rather than add a
second checklist. Apply cues to authored teaching headings, never to headings
inside quoted conversation, example tool JSON or output-format fences.

| Section | Example cue |
|---|---|
| What I Receive | Locate the evidence, prior claims and missing bodies. |
| Reading the conversation | Notice details and changes before judging their meaning. |
| Nodes | Preserve one useful claim with its basis and scope. |
| Edges | Read each relationship as a claim that can become stale. |
| Temporal anchoring | Separate when it happened from when it was planned or reported. |
| Actions | Choose the memory change, then the tool that expresses it. |
| Cadence and worked examples | Follow evidence through decisions, writes and returned results. |
| Identity-bearing examples | Keep the person's pattern and the evidence that bounds it. |
| Closure | Carry forward what the actual results leave unresolved. |

Subsection cues should name that subsection's specific discriminator, with
no mandatory additional output or thought-field quota. Preserve the existing
four working lists as the inspectable planning artifact.

## Ending reminder and actual placement

Interpret Tom's “end the session ... post tools” as the end of the encoder's
instruction assembly, with strategy for performing the encode and responding
to tool results. The API passes tools, system and messages separately; the
visual order in an exported prompt is not evidence of one concatenated string.

Current local code assembles the system as:

```text
template → generated field reference → Arc → Review → Finishing
```

For cue arms, the proposed system is:

```text
template with local cues → generated field reference → Arc → Review
→ brief task strategy → unchanged Finishing
```

The strategy would reconnect the whole job: read the conversation and prior
claims; account for details and implications in the existing lists; fetch what
is needed to decide; apply the memory changes; use returned results to judge
what succeeded and what remains. It would preserve unsettled evidence and
keep a plan or no-mint verdict from becoming a fact in continuity.

This is a stable instruction about working after tool results, not a new
message injected after every tool call. It adds no forced continuation or
automatic write after a failed operation. The system's shared Finishing
contract remains the last block. The user payload remains:

```text
preamble → continuity → catalog → unchanged v2 gist → timeline
```

Implement placement through an evaluation-only adapter and verify the actual
composed request before use. Appending prose to the candidate template alone
would put it before the generated reference, which does not test this design.
Fingerprint the footer and adapter as well as template/gist/schema inputs;
ensure frozen-corpus cache addressing distinguishes the changed assembly.
No edits to shared `trace_contract.py`, live `encoding_prompt.py` or S2 behavior
are implied. A production mechanism for this placement remains a separate
promotion decision.

## Yesterday's corpus — verified from the runners and manifests

Root: `/Users/tpac/AgentsContext/s1e-field-coverage-gold/ab_2026-09-01_03/`.

Gold runner: `tools/run_guide.sh`. It uses these frozen captures under
`payloads_patched_guide/`, the corresponding gold specs in the review worktree,
and the existing hand-adjudication rules:

| Case | Capture | Gold spec | Repetitions |
|---|---|---|---:|
| d827d22f | 2026-08-31/s1e-61608651-16/000-prompt.md | s1e_fieldcov_d827d22f.json | 3 |
| 86af52d1 | 2026-08-31/s1e-67015b88-17/000-prompt.md | s1e_fieldcov_86af52d1.json | 3 |
| a85d5fb5 | 2026-09-01/s1e-831149ce-5/000-prompt.md | s1e_fieldcov_a85d5fb5.json | 3 |
| bb5b1ef4 | 2026-08-31/s1e-5076cdc2-17/000-prompt.md | s1e_fieldcov_bb5b1ef4.json | 3 |
| run44 | 2026-08-17/s1e-17d9ae94-44/000-prompt.md | s1e_run44_staleness.json | 2 |
| d034485c reach probe | 2026-09-07/s1e-01a0791a-23/000-prompt.md | no surface gold; inspect fetch/use | 3 |

That is fourteen scored-case runs plus three reach probes per arm. Historical
scores stay in `ops9/ADJUDICATION.md`; do not overwrite or rederive them. Compare
per item and surface, and show each scoreable denominator. A read of a target
whose available database state postdates the capture remains VOID, not a miss
or success. The reading arm must not appear weaker simply because it has fewer
scoreable runs. Freeze the new cell's read substrate once and record its origin;
today's snapshot is not identical to yesterday's merely because both are
IsolatedBrain copies. A repaired historical read fixture would be a separately
labeled test and require an appropriate rerun of comparison references.

Longmem runner: `tools/run_longmem_guide.sh`. The three existing corpus manifests
confirm the same ten question IDs, oracle file `longmemeval_oracle.json`, S2
cadence 4, lived encoding, and nursery seed-pack digest `914bd00f3a8b`:

```text
54026fce,fca762bc,2311e44b,bc149d6b,71017276,
gpt4_b0863698,cc5ded98,59524333,09ba9854_abs,edced276_abs
```

Existing frozen brains under `/Users/tpac/AgentsContext/eval-corpus/`:

- `e16d35`: historical production; saved sweep `lm9_prod_sweep2`.
- `eaaeb9`: historical branch candidate; saved sweep `lm9_candidate_sweep2`.
- `f4897d`: guide v1.2; saved sweep `lm_guide_sweep`.

For each new prompt, replay the **same source conversations** into a new frozen
corpus, then run the same variance-3 sweep with preflight. Running only another
sweep over an old frozen brain does not test the new encoder prompt. Preserve
the old corpus/sweep labels. Verify seed, surface, S2, retrieval, answerer/judge
and variant settings before claiming a matched comparison with historical
results. Pass `BRAIN_S1E_LISTS_PREAMBLE=1` after `./dev`, which resets inherited
settings; inspect captures rather than trusting launcher comments.

## Execution order and reading the results

1. Author paired guidance changes, then derive cue variants mechanically from
   their direct parents. Reverse-pass the examples with `docs/S1E-CHECKLIST.md`:
   uncertainty, first disclosures, speaker identity, all-surface maintenance,
   local cues and actual results inspection. Freeze hashes and candidate map.
2. Capture each composed request and check placement, schemas, gist, settings
   and isolation. The existing synthetic sequence is a regression check only;
   do not keep rewriting the candidate against its answer key.
3. Run all five arms on the matched historical gold/reach cell and the same
   ten-item longmem slice. Interleave arm order where practical; run only one
   IsolatedBrain at a time. Keep the before-state and every emitted operation.
4. Put results beside historical production, branch candidate and guide in one
   per-item report. Separate actual encoding coverage, retained detail,
   unsupported claims, revise surfaces, recall/answer outcomes, valid
   abstention and VOID reads. Working-list compliance is a process observation,
   not the score. No single blended score that hides a create/revise tradeoff.
5. Freeze the selected version and reference before wider validation. Inventory
   and pin the existing normal LongMemEval/real-chat batches and fresh real S0
   segments, with overlap against the ten-item slice recorded. Define expected
   knowledge from original conversations independently of this encoder's output.
   Keep that wider set out of the authoring loop; a change after seeing its
   failures becomes a new development version and needs fresh held-out coverage.

Historical timing is about 50 minutes per 17-run gold/probe cell and 52 minutes
per longmem build plus about 8 minutes per sweep. Five arms therefore imply
roughly nine hours of sequential model work before hand review and wider suites.
These are planning estimates from the handoff, not timings of the new arms.

## Continuity and boundaries

Continue authoring here; the latest local session report was about 126K of
258.4K tokens with three compactions. That is a snapshot, not a promise of
remaining capacity. Candidate files, hashes, corpus manifests and hand ledgers
must carry execution state so a fresh task can continue without reconstructing
the discussion. Do not create a new task or automation until Tom asks.

Authorized worktree only:
`/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242`, branch
`claude/sweet-lichterman-ba9854`, tracked HEAD `9a1727f`. Never edit the shared
root checkout. Existing Sonnet workflow authorization persists (`c4cbd7da`).
No merge or deployment; all original shared-contract, catalog rendering,
runner-continuation and promotion gates remain Tom's.
