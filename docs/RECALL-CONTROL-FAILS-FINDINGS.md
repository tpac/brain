> **⚠ NUMBERS UPDATED 2026-06-08 → `docs/RECALL-STATE.md` is canonical.** This doc's headline (18/30,
> 47%, "compositional gap") predates gold-tightening + code-review. **Validated figures: 15/30 fail,
> 57% top-5 / 83% top-25; mechanism = z-average scoring burial + query dilution** (not "compositional").
> The killed-approaches and the corpus methodology here remain correct.

# Recall Control-Fails Investigation — Findings (2026-06-08, autonomous session)

> **Bottom line:** Recall is **largely good enough**. Through the real pipeline (recall top-25 +
> spread activation) the brain surfaces **~83% of essential cluster nodes**; the scary "47% @ top-5"
> was a *pre-pipeline measurement artifact*. **No new retrieval lane is justified** — the lexical lane
> and the trace-chain were both no-wins. The only concrete lever for the small residual gap is a
> **bug fix** in the existing cluster-completion spread variant (flagged, chip + node `8c4193be`).

## How we got here (the arc)
1. **Lexical lane (arm A) — no-win.** Built a keyword-scoring lane to fight "entity burial." Measured:
   dense recall already ranks the answer-node at rank 1 (mean 1.1); arm A was net-neutral and *hurt*
   the one query it targeted (E1b 1→3). Dropped the 2nd-FTS-table idea — porter already matches `ex.co`
   (the tokenizer premise `8bd36e83` was wrong). Nodes: `3c315383`, `6252e425` (the build-gate lesson).
2. **The easy corpus was validation theater.** `recall_corpus_v2` had gold *harvested from recall's own
   output* → Control passed by construction. (Tom caught this.)
3. **Built a real corpus** (`control_corpus.json`): 30 questions GROUNDED in real S0 conversation moments
   (`control_moments_find.py`), across modes (trigger/topic/heavy/remote/episode), gold defined
   **independently** of recall (Sonnet-judged essential/helpful from an exhaustive recall+fts+graph pool),
   **time-scoped** for episodic questions.
4. **On the real corpus, Control fails** 18/30 at top-5 (only 47% of essential nodes) — the gap is
   **compositional** (multi-node answers), NOT lexical burial. Node `67e14230`.
5. **But the misses are reachable.** Reach diagnosis (`control_reach_diagnose.py`): of 50 missed-essential
   nodes, 60% are 1–2 graph-hops from what Control found, 36% keyword-reachable, **only 4% (2 nodes)
   truly unreachable**. Node `e2f3d1e5`.
6. **Post-spread closes it** (`control_postspread_eval.py`): top-25 = 83%, spread-from-top5 = 82%. The
   pipeline already assembles ~83% of the cluster. Node `acc43cb9`.

## Results — essential-gold coverage across the pipeline
| stage | coverage |
|---|---|
| recall top-5 | 44/94 (47%) — pre-pipeline, misleading |
| recall top-25 (surfacer's input) | **78/94 (83%)** |
| top-5 + spread (baseline) | **77/94 (82%)** |
| top-5 + spread (cluster) | errored → 47% (the bug) |

Remote single-fact questions: 0/6 fail (recall finds isolated nodes fine). Fails concentrate in
topic/heavy/episode = whole-cluster answers.

## The one real lever — a bug, not a new mechanism
`spread_activation_cluster` (`servers/scales/s1/surface_contract.py:1192`) **throws on every call**
(`expected 5, got 4`): `edges` built as 4-tuples (line ~1267) but unpacked as 5-tuples (lines
~1297/1301) — a half-finished 4→5-tuple "family" refactor. Gated behind `BRAIN_RECALL_VARIANT=cluster`
(not default), so production (baseline) is unaffected, but the cluster-completion variant — purpose-built
for exactly this compositional gap — is dead. **Chip spawned** (`task_ea49796d`); node `8c4193be`.
Fixing it + re-running `control_postspread_eval.py` is the test for whether the residual ~17% closes.

## What's NOT worth doing (measured no-wins — don't re-propose)
- A new **lexical lane** (arm A): net-neutral-to-negative. The lexical-v2 substrate is committed
  flag-gated-dormant (`BRAIN_RECALL_ARM` unset = byte-identical); recommend revert or leave dormant.
- The **trace-chain** episodic lane: echo-confounded (`bc960665`), 43% of dialogue un-embedded; its
  #11 "win" was replay-inflated. Dormant on `main` (`BRAIN_TRACE_CHAIN`).
- Groups **B/C** (gate/inhibition operators): would polish a rank-1 non-problem; the gap isn't where
  they'd help.

## Next steps (in priority order)
1. **Fix the cluster-spread bug** (chip `task_ea49796d`), re-run `control_postspread_eval.py` — does the
   residual 17% close? If yes, the arc is fully done at "recall is good enough."
2. (Optional) a true end-to-end run with the **Haiku surfacer** (not the top-5 proxy) for a clean number.
3. Decide: **revert** the dormant lexical-v2 substrate, or keep flag-gated.

## Files (all in `eval/oracle_audit/`)
`control_corpus.json` (the real corpus, gold filled) · `control_moments_find.py` (moment finder) ·
`control_gold_judge.py` (Sonnet essential/helpful) · `control_reach_diagnose.py` · `control_postspread_eval.py` ·
results: `control_gold_result.json`, `control_postspread_result.json`. Superseded: `recall_corpus_v2.json`
(harvested-gold, validation theater — kept for the lesson).

## Key nodes
`acc43cb9` (conclusion) · `67e14230` (control-fails) · `e2f3d1e5` (reach diagnosis) · `8c4193be` (cluster bug) ·
`3c315383` (tokenizer/arm-A) · `fb2fc5b4` (kill-verdict lesson) · `6252e425` (build-gate lesson) ·
`bc960665` (trace echo) · `2b1c7751` (trace coverage gap).
