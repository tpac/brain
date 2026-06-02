# S2 Consolidation × Absorb — Session Handoff

**Branch:** `claude/zen-shamir-caf8a3` (worktree `zen-shamir-caf8a3`)
**Status:** infra + fixes done and tested (uncommitted); the consolidation **prompt
rewrite is a candidate, NOT registered/activated** — production still runs v6.
**Next session:** decoder pre-classification (the merge-recall lever), ship decision,
deferred items below.

---

## 1. What this was — and how it evolved

Started as: *"S2 adjustments to source_ref + wire the `absorb` op into consolidation."*
It evolved, through evidence, into:
1. A set of **shipped fixes** to the consolidation/absorb substrate.
2. A reusable **eval methodology + instruments** for S-scale prompt changes (this is
   the durable win).
3. A sharp **diagnosis of the brain's "accretion regime"** — it under-merges (S2) and
   under-revises (S1) — which reframed the whole task.

The original source_ref goal was answered structurally: `absorb` unions source_refs by
default, so wiring absorb *is* the source_ref-preservation fix for consolidation. The
S2-encoder-*authoring* source_refs piece (node `717c80cc`) was deferred (never reached).

---

## 2. What SHIPPED (done, tested, uncommitted on this branch)

| Change | File(s) | Note |
|---|---|---|
| **SKIP-detector recognizes a successful absorb** — a newly-archived cluster member ⇒ "handled" (a merge), not a SKIP rejection. Snapshots archived members pre/post the encoder run. | `servers/scales/s2/consolidation.py` | + fixed 3 **stale** tests that predated `absorb` joining `VALID_BATCH_OPS` (they asserted absorb was invalid / "closed five"). New detector tests in `tests/test_s2_consolidation.py`. |
| **co_accessed-on-remember REMOVED** — recall-only now. The `auto_connect` block created co_accessed by temporal write-adjacency (pre-Phase-5 noise) AND fixed pair physical-direction by creation accident. | `servers/brain_remember.py` | `auto_connect` param kept as a **deprecated no-op** (filtered by `_CONTROL_FIELDS`); recall-time Hebbian path (`recall_write_queue`) unchanged. |
| **`absorb` MCP description corrected** — "losslessly merges" → **"content-DESTRUCTIVE: survivor keeps its content; absorbed content lost unless you write a `content` override."** | `servers/brain_mcp.py` | Cross-caller fix (Anchor/S1/S2). Sonnet's own self-diagnosis confirmed "lossless" drove reflexive over-merging. |
| **Absorb preservation gate** — reusable `snapshot_pre → absorb → audit` auditor across all transfer dimensions; CI test proves a rich-fixture absorb is lossless. | `eval/absorb_preservation_probe.py`, `tests/test_absorb_preservation.py` | |
| **Tests green** | — | 28+ across consolidation/absorb/invalid-op/examples. |

**Temperature note:** a `temperature=0` experiment was added then **fully reverted** at
Tom's instruction ("run without playing with temperature"). `runner.py`,
`consolidation_encoder.py`, `consolidation_contract.py` are back to model default. No
temperature code remains. (Reasons it was dropped: temp 0 didn't make the encoder
deterministic anyway; newer models — e.g. `claude-opus-4-8` — *deprecate* `temperature`
and 400 on it.)

---

## 3. What's NOT shipped (candidate / eval-only)

- **The consolidation prompt rewrite** — `eval/candidate_prompts/s2_consolidation_absorb.md`.
  NOT registered as an interaction, NOT activated. Production `s2_consolidation_enrichment`
  is still v6. (The eval-gate discipline: register DORMANT → eval → activate → `./dev sync-prompts`.)
- **The S2 dimensions-eval platform** (`consolidation_quality_contract.py`, the
  contract eval) — new infra, eval-only, not wired to production.

---

## 4. The process we developed (the durable methodology)

**S-scale prompt-change loop:** `prompt change → examples (the key lever) → feed to
probes ↔ feed to dimensions eval → corpus/A-B`. Examples beat rules for steering Sonnet
(brain `f3086e73`, `928f5694`).

**Instruments built (all reusable for future S-scale prompts):**

1. **Ground-truth corpus** — `eval/corpus/clusters.json` (20 real production clusters)
   + `eval/corpus/labels.json` (Anchor's hand-judged action per cluster: absorb / split
   / keep / skip + rationale + confidence). The **oracle**. Built via
   `eval/build_consolidation_corpus.py --dump`.
2. **Mechanical corpus scorer** — `eval/score_consolidation_corpus.py`. **No LLM grader**:
   classifies the prompt's emitted ops and compares to the labels. Reports merge-recall /
   over-merge / under-merge / exact-action / absorb-lossless, per arm, K samples. This is
   the authoritative instrument.
3. **Dimensions eval** — `eval/agent_introspect/consolidation_contract_eval.py` +
   `servers/scales/s2/consolidation_quality_contract.py` (10 dims C1–C10, 3 hard-gate:
   C4 content, C5 provenance, C8 locked). Independent **Opus** scorer (`OPUS_MODEL` in
   `_common.py`). **Mechanism-blind** (scores the decision, not the op vocabulary —
   `SCOPE` controls n_a). Also hosts `validate_example_authoring()` (the authoring gate).
4. **Authoring gate** — `validate_example_authoring` + `--validate` mode +
   `tests/test_consolidation_examples.py`. Mechanically enforces: no `...`/truncation,
   every `absorb` has `content` (and the title-rewrite vs title-keep patterns both
   appear), every caller capability exemplified. Anti-pattern (BAD) blocks marked `❌`
   are skipped.
5. **Variance analysis** — `--samples K`. Necessary because **n=1 is unreliable even at
   temp 0** (proven: the verdict flipped PASS→FAIL across identical runs; the encoder is
   non-deterministic).
6. **Probes** — `eval/s2_absorb_prompt_probe.py` (behavioral A/B + Sonnet critique +
   `--restraint` self-diagnosis + `--drilldown` per-cluster decisions). The shared
   capture-only arm runner is `run_capture_variant` in `eval/s2_consolidation_eval.py`.

**Methodology lessons learned (hard-won):**
- Don't eval on a one-sided cluster set. The cold-start decoder surfaces only
  `likely_keep` clusters → an eval there only measures *over*-merge and rewards
  restraint. Use the trace-mined ground-truth corpus (both axes).
- Examples must **not re-text the current session / real codebase** (contamination /
  teaching-to-the-test). Use synthetic ids + generic domains. The authoring examples were
  de-session-ified (inventory, onboarding, report-rendering, checkout-latency, Postgres
  — none echo this codebase).
- A noisy LLM grader on a one-sided set is worse than a mechanical scorer on ground truth.

---

## 5. Key findings (the intellectual core)

1. **Accretion regime.** The brain under-modifies at both scales. S2 consolidation
   merge:keep collapsed **~100×** over a month (cold-start ~2.6:1 → recent ~0.03:1; 499
   `similar_to` : 16 merges). S1 Scribe revises at ~0.13:1 and absorbs 0 (sibling
   stream's finding). Same disease, two scales.
2. **The real failure is UNDER-merging, not over-merging.** Ground-truth corpus:
   production v6 misses **8/12** genuine duplicates; ~0 over-merge. The entire
   mid-session "over-absorption" fight was an **artifact** of the one-sided cold-start set
   + examples contaminated with live ids. *(Tom's push for a ground-truth corpus caught this.)*
3. **MCP op descriptions steer behavior cross-caller.** `absorb` "lossless" *under*-warned
   loss → over-merge. `revise` *over*-warns loss → accretion. **Shared fix philosophy:**
   *state what's lost AND the how-to-avoid; never scare the caller off the modify/merge
   path.* (`revise` fix is the sibling's — see §7.)
4. **`absorb` is content-destructive.** It keeps the survivor's content; the absorbed
   node's content is orphaned unless you write a `content` override. Fix = **mandatory
   title+content rewrite** in the prompt → losslessness **68% → 95%**.
5. **Examples fixed MECHANICS (losslessness); recall needs the two levers below.**
   The rewrite fixed losslessness but did NOT move merge-recall. **We asked the encoder
   directly why it under-merged the corpus duplicates** (`eval/why_underconsolidate_probe.py`)
   — its self-diagnosis:
   - **Lever A — the `decoder_pre_class: likely_keep` signal is the #1 pull.** Every
     under-merged cluster had it. The encoder: *"it reverses the burden of proof —
     instead of 'why keep these separate?' I shift to 'why override the decoder's keep
     recommendation?'"* Fixable in the **prompt** (tell it to treat `likely_keep` as a
     null/weak signal) AND/OR the **decoder** (stop mislabeling duplicates as
     `likely_keep` / surface them as `likely_consolidate`).
   - **Lever B — type-difference still defaults to keep.** The "run the claim test"
     line is too soft. It over-keeps `bug`+`fact` (one incident), `plan`+`fact` (one
     project), `mechanism`+`architecture` (one system), and misses `fact`+`event`
     supersession (should EVOLVE). Fix: **concrete cross-type-duplicate pairings** in the
     prompt + an explicit **supersession→EVOLVE** rule.
   So recall IS partly promptable after all — these three additions weren't tried yet.
6. **Edge-direction model gap (deferred, real).** v22 stores one physical direction per
   edge pair; co_accessed (recall-time, dominant) materializes that direction by accident,
   and later semantic edges (`depends_on`, `corrects`, …) inherit it. Real fix = per-
   relation direction — a recall read-path + connection-shape contract change. Deferred.

---

## 6. Latest eval state (corpus K=3, refined prompt, **default temp**)

`eval/reports/corpus_score_k3_v2.json`:

| | UNDER | correct | OVER | exact-action | absorb-lossless | merges |
|---|---|---|---|---|---|---|
| baseline v6 | 6 | 8 | 3 | 8/17 | 7/7 | 7 |
| candidate | 7 | 7 | 3 | 7/17 | **20/21** | **21** |

- **Losslessness solved** (20/21=95%), candidate merges **3× more** and stays lossless.
- **Merge-recall not improved** — still keeps clear dupes: clusters `5` (bug+fix same
  incident), `6` (two identical principles), `10` (same test arc), `11` (stale→activated
  supersession), `12` (`/watch` dup).
- **Over-merge (3)** = clusters `16/17/18`, which are **borderline labels** (two DAL
  phases; fix+restart-followup; bug+design-direction) — partly a ground-truth-label
  issue, not clear error.
- Caveats: default-temp noise at n=20 (clusters `0/16/17` flip 33–67% across samples);
  losslessness signal robust, under/over counts noisy. Not directly comparable to the
  earlier temp-0 candidate run.

Dimensions eval (`dims_v3.json`, K=2 default temp): candidate near-perfect on all 10
dims (C4/C5 = 1.0); the "GATE: FAIL" was a 0.01 C10 noise blip (gate tie-at-ceiling logic
since fixed).

---

## 7. The consolidation prompt candidate — coverage

`eval/candidate_prompts/s2_consolidation_absorb.md`. Examples cover these **merge
patterns**: full synthesis · multi-node cumulative (+`drop_fields`) · **title-stable
append** (keep title, append facts/dates/numbers verbatim — *don't smooth*) · partition
4→2 survivors · evolve (+`prune_edges`+`disconnect`) · **contradiction → `corrects` edge,
keep both** · keep · skip · **locked** (locked is always survivor). Capabilities
exemplified: absorb(title+content), KV revise, `drop_fields`, `prune_edges`, `connect`
(similar_to / corrects / depends_on), `revise`, `disconnect`. Plus a **Good/Bad
contrastive pair** (Postgres). The **claim test** is the core decision gate ("name each
node's claim; shared noun ≠ shared claim; can the operator recover BOTH claims after the
merge?").

---

## 8. Open threads / next steps (ranked by leverage)

1. **Merge-recall — two levers, both confirmed by the encoder's own self-diagnosis**
   (`eval/why_underconsolidate_probe.py`, §5.5):
   - **A. The `likely_keep` pre_class bias.** Either (prompt) add "treat `decoder_pre_class:
     likely_keep` as a null/weak signal — run the claim test regardless," and/or (decoder)
     fix `ConsolidationDecoder._pre_classify` so obvious duplicates (two identical
     principles, a stale supersession, a `/watch` dup) aren't labeled `likely_keep`.
   - **B. Cross-type-duplicate guidance.** Add concrete pairings to the prompt — `bug`+`fact`
     = one incident; `plan`+`fact` = one project; `mechanism`+`architecture` = one system —
     plus an explicit **supersession→EVOLVE** rule (newer absorbs older, not KEEP).
   - Then re-run the corpus scorer (K=3) — the duplicates `5,6,10,11,12` are the regression
     targets. This is promptable + structural; do the prompt side first (cheap), measure,
     then the decoder if needed.
2. **Ship decision on the prompt.** Losslessness win is real (95%) and shippable; recall
   is not yet fixed. Either ship for the losslessness gain now (register DORMANT → eval →
   activate → `./dev sync-prompts`) or hold until the decoder lifts recall. Recommend
   pairing with the decoder fix.
3. **Corpus label-review pass (do this before trusting per-cluster numbers).** A
   production-faithful interview (`eval/interview_encoder_probe.py`, runs the EXACT
   encoder setup then interviews it in-thread) surfaced that:
   - **Variance is enormous** — clusters `11`/`12`, scored *under-merge* in the corpus
     K=3, were **absorbed** in the faithful runs. Same prompt+render, default temp,
     opposite calls. Per-cluster counts are noisy; the regime finding rests on the
     *trace data* (§5.1), not these.
   - **Label `11` is likely WRONG.** `fact`("v24 dormant") + `event`("v24 activated") is
     **temporal supersession, not redundancy** — keep both moments + a `supersedes` edge
     (matches the CONTRADICTION principle), NOT absorb-evolve. The model argued this
     correctly. Review `11`, `12`, `16`, `17`, `18` (the borderline/supersession tail).
   - **Introspection is post-hoc** — the encoder rationalizes whatever it just did; trust
     the *decision distribution*, not its stated "why".
   - **Free prompt-rule refinement:** add *"temporal supersession ≠ knowledge redundancy
     → keep both moments + a `supersedes` edge"* (sharpens EVOLVE vs CONTRADICTION).
   - The **solid** under-merge signal is the *unambiguous* dups (`6` two identical
     principles, `5` bug+fix one incident, `10` same arc) — anchor recall work on those.
4. **Brain-health (task #4):** production consolidation has stopped merging — once recall
   improves, re-run consolidation to clear the accumulated duplicate backlog.
5. **Edge-direction model gap** (deferred §5.6) — its own focused session; recall
   read-path + connection-shape contract change, benchmark-gated.
6. **source_ref S2-encoder authoring** (`717c80cc`) — the deferred piece never reached.

---

## 9. Cross-stream coordination (in flight)

- Sibling stream **`17d21ad4`** owns the **`revise` MCP description** fix (reframe the
  WHEN-TO fork: `same-concept→revise; supersedes→encode-new+correction-edge`, NOT soften
  the loss-warning) AND the **self-channel** code. I won't touch `brain_mcp.py revise`.
- **drop/new-ids bug** handed to `17d21ad4` (msg `22f3c77b68c4`): session_id is the
  self-channel delivery key but it's **volatile** — every resume spawns a new sid;
  presence showed 3 "me" streams; `drain_inbox` only drains the current sid, so directed
  messages to a pre-resume sid orphan. Proposed split: they own `signal.py`/`boot`, I take
  the repro + a regression test. **Awaiting their reply.** (Not editing self-channel code —
  shared-git-index hazard, brain node on that.)
- Shared philosophy (both absorb + revise): state loss + give the how-to, never scare off
  modify/merge.

---

## 10. File inventory

**Modified:** `servers/scales/s2/consolidation.py`, `servers/brain_remember.py`,
`servers/brain_mcp.py`, `eval/s2_consolidation_eval.py` (added `run_capture_variant`),
`eval/agent_introspect/_common.py` (`OPUS_MODEL` + `call_sonnet` temperature param — the
temp param is harmless/inert), `tests/test_invalid_op_suppression.py`,
`tests/test_s2_consolidation.py`.

**Created:** `eval/candidate_prompts/s2_consolidation_absorb.md`,
`eval/absorb_preservation_probe.py`, `eval/s2_absorb_prompt_probe.py`,
`eval/build_consolidation_corpus.py`, `eval/score_consolidation_corpus.py`,
`eval/why_underconsolidate_probe.py` (retrospective: asks why it under-merged §5.5),
`eval/interview_encoder_probe.py` (production-faithful: clones the exact encoder setup,
lets it decide, interviews it in-thread §8.3),
`eval/corpus/{clusters,labels}.json`,
`servers/scales/s2/consolidation_quality_contract.py`,
`eval/agent_introspect/consolidation_contract_eval.py`,
`tests/test_absorb_preservation.py`, `tests/test_consolidation_examples.py`.
Reports under `eval/reports/` (`corpus_score_*`, `dims_*`, `consol_dims_*`, `absorb_prompt.json`).

**Run the instruments:**
```
./dev python3 eval/agent_introspect/consolidation_contract_eval.py --validate          # authoring gate (free)
./dev python3 eval/score_consolidation_corpus.py --samples 3                            # ground-truth corpus (encoder only, no grader)
./dev python3 eval/agent_introspect/consolidation_contract_eval.py --clusters 6 --samples 2   # dimensions eval (Opus grader)
./dev python3 eval/build_consolidation_corpus.py --dump --n 20                          # rebuild corpus clusters
./dev python3 -m pytest tests/test_consolidation_examples.py tests/test_absorb_preservation.py tests/test_s2_consolidation.py -q
```
