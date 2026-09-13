# Resumed and frozen — 2026-09-08

Tom said “you can continue.” Authoring and local semantic/mechanical review
are now complete; all five arms were sealed and their hashes reverified.
Joint review with Tom is pending. No model evaluation has run in this phase.
Read `docs/S1E-ARM-FREEZE-REVIEW-2026-09-08.md` for current state and the exact
manifest. The draft hashes and pending-helper notes below are historical,
superseded by `frozen/manifest.json`; do not resume those old steps.

Tracked HEAD remains `9a1727f`; tracked runtime files and original candidate
parents are unchanged. The parent paths mentioned by tools during review
were reads, not edits. New candidates/helpers/reports are untracked.

## Historical pause checkpoint

Tom needs to disconnect and will prompt to resume in about an hour. Do not
continue authoring, run evals, schedule a wakeup or launch a new task meanwhile.
No model evaluation or long-running command is active from this work.

## Task and boundaries

Author and freeze five arms, then review before any expensive evaluation:
unchanged v2 reference; revised full v2; compact v3 based on Astra Shapes;
and section-cue variants of each revised base with a shared ending strategy.
Tom's latest authorization is authoring/freezing/review only, not launching
the next model cell. Existing API approval persists but does not override
this pause or the requested review before eval.

Worktree `/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242`,
branch `claude/sweet-lichterman-ba9854`, HEAD `9a1727f`. Never edit the root
checkout. Nothing committed, merged, deployed, or registered. All tracked
runtime files are unchanged. New artifacts are untracked.

## Already written, NOT sealed

In `eval/candidate_prompts/`:

| Draft | Characters | SHA-256 |
|---|---:|---|
| s1e_guide_v2_revised_2026-09-08.md | 120978 | 3eb2340066a984dd23e8a29c43a4426eff3ad6d36c29cd39017fc0c51519fc1d |
| s1e_guide_v3_2026-09-08.md | 85347 | 5ed84ac8e203bc1986b94064b91ce34ec10e264656906746b0d02569d633f6ed |
| s1e_guide_v2_revised_enhanced_titles_2026-09-08.md | 123819 | ed531435ad041d21e9e8835d8a36109f54497d858c37e4826468e4da796ceb91 |
| s1e_guide_v3_enhanced_titles_2026-09-08.md | 87389 | fb393bed04fc787565cdfcd14b218cf2ba4ab888b5d55bfb18f426eaef304878 |
| s1e_guide_navigation_strategy_2026-09-08.md | 737 | 1c4931a8d76b195017d356d703af01f039297e620d1061bee951553975a89b8c |

The original v2, Astra Shapes and the v2 gist remain byte-unchanged.
The cue variants add 38/27 short italic cue lines immediately below authored
headings, outside example fences. Removing these should restore each parent
exactly. The strategy is separate, intended after generated references and
Arc/Review, immediately before the unchanged shared Finishing contract.

Changes matched across the revised bases:

- Reading: preserve speaker, evidence, scope; uncertainty is not rejection;
  a promise, progress and completion have different evidential support.
- Fields: title/reasoning/situation/edges must not overstate content.
- Mira's later-window example now uses framing prints for her own wall:
  preserve the new routine, update only the thought, keep the hosting
  interpretation useful and tentative without requiring her endorsement.
  This replaces the prior teaching-class example, avoiding the diagnostic's
  classroom surface language.
- Mira's arrival plan title/content/edges now explicitly describe agreement;
  memory-write success does not mean the card/checks were completed.
- Inherited temporal defects repaired: reported PT estimate is not clearance;
  season/month midpoint conventions do not establish precise intervals.
- Inherited sweep defects repaired: show the stale situation before its
  verdict; preserve actual “my branch” quote; ground date/unmerged status/
  successor decision in the fictional input; also swap the stale sentence
  “Auth is the dependency the gateway builds on.”

## Files in this directory

- `author_arms.py`: deterministic generator with frozen parent hashes and
  unique-anchor assertions. `--write` was run successfully twice during
  drafting. Without arguments it checks existing output bytes. Refuses
  writing after a frozen manifest exists.
- `later_window.md`: source of the shared new example.
- `authoring_audit.json` and four `.diff` files: current authoring outputs.
- `frozen_arms.py`: NEW, not yet executed/reviewed. Pure render/loader helper;
  inserts strategy before Finishing, validates a sealed manifest and hashes,
  returns system/gist/tools/settings plus an arm identity. Cache rule requires
  this identity, not just template hash. Does not integrate a corpus runner.
- `review_and_freeze.py`: NEW, not yet executed/reviewed. Offline checks,
  disables socket connections, uses actual runtime field/schema renderers
  without a Brain/database, checks examples and negative cases, optionally
  creates `frozen/` with `--seal`. No LLM calls. Needs review and execution.
- `frozen/` DOES NOT EXIST yet. Do not say the set has been frozen.

Neither the repo Python nor bundled runtimes have jsonschema/json5/ajv.
No packages were installed. The checker intentionally uses the repo's field
and swap validators, JSON parsing, and a restricted AST-to-literal parser for
the sweep's unquoted-key teaching syntax. It must describe this as contract
field/structural verification, NOT complete JSON Schema or real dispatch.

## Resume from here

1. Read this note, `docs/S1E-NEXT-COMPARISON-2026-09-08.md`, and the new helper
   code. Recheck branch/status and parent hashes. No new subagent authorization.
2. Run `python3 eval/fixtures/s1e_guide_freeze_2026-09-08/author_arms.py`.
3. Review and run `./dev python3 eval/fixtures/s1e_guide_freeze_2026-09-08/review_and_freeze.py`
   WITHOUT `--seal` first. Fix authoring or guard defects before sealing;
   regenerate via the author script if candidate text changes.
4. Continue semantic reverse review of the exact new diffs, cues, example
   inputs/outputs and assembled suffix placement. Check all matched changes,
   preservation, actual quote spans, open state and bounded interpretations.
   The v3 base is about85K rather than the rough84K target; explain actual
   cost instead of silently trimming unrelated text from cue variants.
5. Record shared inherited limits: unchanged gist still has rigid read/write/
   close phrasing and known read-scope seams; these are held constant, not
   solved by authoring. Do not change S2-shared runtime gates.
6. After review fixes pass, use `--seal` once, then verify the sealed set and
   hash guards. Write a concise review report with the five sizes, matched
   changes, caught/fixed errors, remaining limits and links to diffs. Update
   the experiment plan's status. No expensive eval; let Tom review the set.

Potential checker details to inspect before trusting it: the central JSON
call indices, sweep AST parser, before-state swap simulation, same-batch
title checks, socket-blocked runtime imports, late-added negative-check
report entry after sealing, and refusal to overwrite a partially made
freeze directory. The new helpers have not been tested yet.

Prior results remain in `docs/S1E-GUIDE-COMPRESSION-EVAL-2026-09-08.md`
(brain `68dd8ff8`); next-comparison direction is brain `9481d16b`.
