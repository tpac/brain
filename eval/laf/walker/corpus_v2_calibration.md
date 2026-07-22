# Corpus-v2 calibration gate — package for Tom (2026-07-21)

Protocol: brain node `25cea181`. 40 turns judged (4 known cases + 36
stratified by v0 stratum × strong tier), 2 Fable judge agents, structured
output, order-based key join. Verdicts: `corpus_v2_calibration_verdicts.jsonl`.

## Gate results

| requirement | result |
|---|---|
| ≥85% agreement with hand-checked obvious cases | **~92%** (24/26 firm cases; 1 of the 2 disagreements was MY error — see ex.co/T15 note) |
| dedup task → valid | ✓ valid/cue |
| eval-prod audit → valid | ✓ valid/cue |
| error-logs → echo_mislabel | ✓ echo_mislabel |
| lets-do-it → valid/session (Moment-gold) | ✗ **echo_mislabel — and the judge appears RIGHT** (below) |

Headline distribution (40 turns): 24 valid (22 cue / 1 window / 1 session,
pre-anaphora-fix), 14 echo_mislabel, 2 ambiguous. **NEITHER-stratum sample:
9/14 echo_mislabel** — the mislabel concentration the protocol predicted.

## Item 1 for Tom — the lets-do-it verdict (re-adjudication)

The judge called `lets do it` → S1E-reconciliation-milestone an
**echo_mislabel**. I pulled the ENTIRE session's op_texts (c8b4c3fa/0, seq
0–20): it is the walker superseded-semantics investigation end to end
("INVESTIGATION TASK — the §20 moment-walker…", interrupt grouping, keep-B,
"You think its worth doing?"). **No turn in the session is about S1E
reconciliation.** The gold is 14.5d old, from a different thread.

Your own criterion (449fb9a7): "'let's do it' shouldnt surface any freaking
S1E milestone **unless the msgs before were about that**" — they weren't.
The label appears minted by response-echo ('commit… merged… tests green'
resembles the milestone's content). The earlier hand-read ("legit gold — the
'it' refers to the session's topic", 878bce03 case 4) doesn't survive the
full-session check. Proposed known-correct verdict: **echo_mislabel**, with
the style_note the judge wrote (greenlight → surface the just-proposed plan
referent + Tom-greenlight patterns). The Moment-gold CLASS is still real —
T25 ("sure, need me for anything?" → 0.8d-old sibling-session decision) came
back exactly valid/session with a clean running-field bridge.

## Item 2 for Tom — same-session golds policy

T24: "so youre running only 2 sets? i want something quick" → gold is the
launch plan encoded THE SAME DAY, restating what's already in the
conversation window. Post-anaphora-rule judge: echo_mislabel ("not a
restatement of the in-window plan" belongs in recall). This is the
same-session-echo failure mode (52417fe4) applied as policy: **a node
encoding the plan currently under discussion is not a helpful recall.**
Confirm this is the intended policy for the full run (it demotes some v0
CUE-SUFF golds whose only virtue is same-day redundancy).

## Item 3 for Tom — spend (the big one)

Calibration + recheck burned **~293k subagent tokens for 46 turns**
(~6.4k/turn). Full run at 2152 turns projects **~13M subagent tokens with
Fable judges** — 5× the protocol's 1.5–2.5M estimate. The protocol's 1M
checkpoint would trip ~8% in. Options:

- **A. Fable full run** — best judgment, ~13M tokens (~45 min wall at ~10
  concurrent agents).
- **B. Sonnet full run** — same tokens, ~5× cheaper, rubric is
  well-specified discrimination; risk: subtler helpfulness calls.
- **C. Hybrid (my recommendation)** — Sonnet full run, then Fable re-judge
  of every echo_mislabel + ambiguous verdict plus a 10% valid sample (the
  echo class is the destructive-if-wrong call). Est. ~13M Sonnet + ~3M Fable.

## Fixes applied during the gate

- **Anaphora rule** added to the judge prompt after a cue-ward stratum drift
  (22/24 valids judged cue). Recheck on the 6 drift cases: 2 flipped to
  window, 1 principled echo flip, 3 stayed cue with defensible reasoning.
- **Order-based key join** — judges typo'd 2/40 keys (single UUID segment);
  the collector joins by presentation order and records `key_as_returned`.
- Judge bundles withhold v0_stratum and F0/M_h ranks (only the static-mix
  HIT/MISS line shows) — semantic verdicts stay independent of the
  mechanical stratifier they replace.

## Ready to fire on approval

87 batch files rendered (`corpus_v2_batches/batch_*.md`), runner workflow
(`corpus_v2_run_workflow.js`, per-batch checkpoint files written by the
agents themselves), collector (`corpus_v2_collect.py`, resumable-by-key
merge to `corpus_v2_verdicts.jsonl`). Gap/bridge quality from the gate is
high (lane-specific, style-recall taxonomy emerging: greenlight-patterns,
cleanup-rituals, handoff-craft, eval-cost discipline).
