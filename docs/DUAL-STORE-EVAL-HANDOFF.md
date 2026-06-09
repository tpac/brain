> **⚠ SUPERSEDED 2026-06-08 → `docs/RECALL-STATE.md` is canonical.** This handoff's plan — build the
> lexical lane (Step 1 "SETTLED") + the A/B operator arms — was **eval-KILLED**: the lexical lane is a
> measured no-win (`3c315383`) and the tokenizer premise was wrong (porter already matches `ex.co`). The
> carrier/z-average **DO-FIRST** it names IS still the #1 lever. Kept for history; read RECALL-STATE.md.

# Dual-Store Recall — Master Execution Handoff (the eval)

> **Read this first. Self-contained — you should be able to execute the eval from this doc alone.**
> Status 2026-06-07: trace-chain lane shipped flag-gated-dormant on `main`; eval found the merge is the
> problem; bio+IR research done; architecture refined to **SETTLED (fix lexical) + A/B (operator) +
> inhibition**. Nothing is in production. Discipline tags: **[VERIFIED]** measured/cited /
> **[HYPOTHESIS]** / **[PRECONDITION]**.

---

## 0. How we got here (the journey) + research links

1. **The burial.** Query **#11** *"what did we do on the last session we worked on ex.co?"* returns **0
   EX.CO nodes** to the surfacer. **[VERIFIED]** the embedding is healthy — raw `_primary` cosine ranks
   the best EX.CO node at **rank 3**; the *scoring pipeline* buries it (the z-weighted multi-vector
   average + title-boost). Diagnosis: `docs/RECALL-BURIAL-HANDOFF.md`.
2. **Dual-store design.** Two stores — semantic nodes + episodic S0 traces — bridged. The rescue is the
   **trace-chain**: `query → top-T s0 dialogue traces → trace's stored vector → nodes` (a 2nd cosine
   retrieval that de-dilutes the buried query). Design: `docs/RECALL-DUAL-STORE-DESIGN.md`.
3. **Built it, flag-gated.** `BRAIN_TRACE_CHAIN` (default OFF) on `main @ 7afb827` — the trace-chain lane
   + a reserved-tail merge in `servers/brain_recall.py`.
4. **Eval caught the merge bug.** Deterministic top-25 A/B (`trace_chain_top25_diff.py`): **#11 WINS**
   (gold node `174fd960` absent→present) but **#2/#12 go net-negative** — the **blind reserved-tail
   EVICTS EX.CO node `8359cf1d`** ("CTV kit") from ranks 21-25 to insert rescues. **[VERIFIED]** Smoking gun.
5. **Lane audit (measured).** Recall is effectively **MONO-LANE**: across 12 queries × 25 slots, dense
   fills **100%** (`embedding_only` 83% + `both` 16% + `embedding+keyword` 2%); **`fts5_only` = 0,
   `keyword_only_fallback` = 0**. The two lexical lanes contribute **zero unique nodes** — they're a
   strict subset of dense. **[VERIFIED]** This explains why ENTITY queries (#2/#12) bury: *their natural
   lane (lexical) is dead.* The recall `keyword` lane also still scores on the **v25-scrubbed `keywords`
   field** (dead weight + old/new asymmetry).
6. **The meshing question** → foundational research (2 deep-research runs, ~10M tokens):
   - **BIOLOGY [VERIFIED, 18 claims]:** memory systems **GATE / INHIBIT**, not sum — *bounded
     gate-then-facilitate, transient, never permanent deletion* (Wimber 2015, Bekinschtein 2018, Currò
     2023, Shin & Jadhav 2024, Rolls 2013). Divisive-WTA **refuted**. Run `wf_52645f55`.
   - **IR/CS [VERIFIED, 23 claims]:** dense fails on rare entities (EntityQuestions DPR 49.7 vs BM25 72.0
     Acc@20). **The bug is the WEIGHT, not the additive operator** — SPAR (the canonical rescue) is
     *additive* but its lexical lane is *strong + tuned*; tuned convex **beats RRF** (Bruch 2023).
     Bounded-multiplicative gate is sound-but-**unproven-superior**. Run `wf_8e7adfee`.
   - Synthesis + the design correction: `docs/research/recall-gating-inhibition.md`.
   - Broader cognitive dossier (from a sibling stream): `docs/research/memory-biases-and-recall.md`.
7. **Two priors corrected by verification** (the discipline working): divisive-normalization-WTA
   (refuted) and multiplicative-gate-as-the-answer (IR says tuned-additive is the proven fix; the gate
   must A/B-win).

**Research run outputs live in `/tmp/.../tasks/{wf_52645f55,wf_8e7adfee}.output` (EPHEMERAL — the verified
findings are persisted in `docs/research/recall-gating-inhibition.md`).**

---

## 1. Current state (what's on main)

- **`main @ 7afb827`**, `BRAIN_TRACE_CHAIN` default OFF → **production dormant** (daemon serves the code
  but never runs the lane; recall identical to pre-`7afb827`). Promotion = flip the default after the eval.
- **Code:** `servers/brain_recall.py` (`_trace_chain_candidates` ~line 1037; STEP 4.6; reserved-tail
  merge at the `scored_results[:limit]` cut; STEP 6.9 floor-bypass). `servers/brain_constants.py`
  (`TRACE_CHAIN_RESERVE=5`, `_T=5`, `_N=25`).
- **Probes** (`eval/oracle_audit/`, isolated): `trace_chain_wired_check.py` (flag off/on guarantees),
  `trace_chain_top25_diff.py` (entered/dropped per query), `lane_contribution_probe.py` (discovery mix),
  `dual_store_merge_probe.py` (offline sweep), `dual_store_validation_probe.py` (census/separability).
- **Test:** `tests/test_trace_chain_lane.py` (flag-gate contract).
- **Corpus:** `eval/oracle_audit/meshed_top10.json` — EX.CO = ranks **2, 11, 12**; controls = **1, 3-10**.
  Gold for #11 = `174fd960`. KNOWN_EXCO set is INCOMPLETE (eyeball titles for EX.CO too).

---

## 2. THE EVAL — execution-ready (Phase A: deterministic top-25, NO Haiku)

**Setup (run from THIS repo root / `main`, where the code lives — the probes' `ROOT=/Users/tpac/brain`
is correct now):** `IsolatedBrain` (copies live DBs, never touches the daemon), 12-corpus, flag OFF vs ON,
via `./dev`. Toggle the lane with `os.environ['BRAIN_TRACE_CHAIN']='1'`.

### Step 0 — Re-confirm the baseline (already measured; re-run on `7afb827`)
```
./dev python3 eval/oracle_audit/trace_chain_wired_check.py     # flag-off no-op; #11 gold rescued; controls held
./dev python3 eval/oracle_audit/lane_contribution_probe.py     # confirms mono-lane: fts5_only=0, keyword_only=0
./dev python3 eval/oracle_audit/trace_chain_top25_diff.py      # #11 WIN; #2/#12 evict 8359cf1d (the bug)
```
**[VERIFIED expectations]** mono-lane; #11 gold absent→present; #2/#12 evict an EX.CO node.

### Step 1 — BUILD the SETTLED fix: resurrect the lexical lane (the proven rare-term rescue)
This is the high-confidence, both-fields-agree, build-now piece. Three sub-fixes (all already diagnosed):
1. **Tokenizer** — add a 2nd FTS5 table with `tokenize="unicode61 tokenchars '.-_/'"` (no porter), so
   `ex.co` survives as one token (`8bd36e83`).
2. **IDF-at-term-level with df-flooring** — replace the flat-0.20 passthrough with real `bm25()`/IDF
   weighting (rare term → high weight) + a BM25-style **IDF floor/smoothing** for ~5k-corpus df=1–2
   instability (`1f301cc5`; `939a5f18` signal÷prevalence). **[VERIFIED concern — IDF unstable at small N]**
3. **Real lane weight** — give lexical comparable, *tuned* strength (NOT fixed 0.10). Also **drop the
   recall keyword lane's read of the dead `keywords` field**, and consolidate the two overlapping lexical
   lanes (keyword + fts5) into ONE.
**Pass:** entity queries #2/#12 surface EX.CO via the lexical lane (which is their natural lane), without
moving the 9 controls. Expect this alone to help #2/#12 *more cleanly than the trace-chain* (entity ≠ episodic).

### Step 2 — A/B THE OPERATOR (do NOT assume; the eval decides)
On the resurrected lexical lane + dense carrier, compare:
- **Arm A — tuned-additive convex:** `score = w_d·dense_norm + w_l·lexical_norm`, `w_l` tuned strong. *(IR-proven baseline — SPAR/Bruch.)*
- **Arm B — bounded floored gate:** `score = dense × (1 + α·IDF·match)`, **floor 1.0, positive-evidence-only** (a present rare term lifts; absent → ×1.0, never zeroes). *(Bio-motivated; must BEAT Arm A to justify.)*
Both require **lane normalization first** (commensurable scales — fixes Fault 1). Measure each on the
top-25 diff + control stability.

### Step 3 — Fix the trace-chain merge (for #11, the episodic query)
The trace-chain stays (it WINS #11) but the **blind reserved-tail must be replaced** — it evicts good
EX.CO nodes (`8359cf1d`). Replace with either the operator from Step 2 OR a **value-aware insert** (only
displace a baseline node if the rescue out-normalizes it) + **transient mild RIF** (down-weight
near-duplicates, floored ≥0.5, reversible — NOT delete). **[bio-VERIFIED: RIF is small/transient]**

### Phase A pass criteria (the gate to Phase B)
- #11: gold/EX.CO rescued into top-25 (trace-chain) — **without** evicting EX.CO on #2/#12.
- #2/#12: EX.CO surfaced via the resurrected lexical lane.
- **Controls: top-(25−K) unchanged** (selection-level safety is Phase B).
- Report multi-dimensional (per CLAUDE.md): rescue · eviction · control-displacement · per-query, not just aggregate.

### Phase B (LATER — Haiku selection + outcome; NOT Phase A)
- **Tier-2 selection gate** — `eval/frame_replay.py`. **NOT turnkey: swap its hardcoded 5-query `CORPUS`
  for the `meshed_top10` queries first.** Then `capture tc_off` / `BRAIN_TRACE_CHAIN=1 capture tc_on` /
  `compare`. Gate: zero control *selection* movement. Mechanism verified: it runs `recall→run_surface→Haiku`,
  reads picks from `/tmp/brain-*-surface-selected.json`.
- **Tier-3 outcome** — `eval/longmem/answerer.py` fed `frame_replay`'s captured `additional_context`
  (off vs on) for #11. ~30 lines of glue. (The "oracle measure" is a SPEC doc, not a runner.)

---

## 3. The architecture (refined: SETTLED vs A/B + inhibition)

`score(n) = carrier(n) [combine] lexical/episodic signals, then inhibition, then defender`

- **Carrier = the HEALTHY dense signal. [PRECONDITION]** Raw `_primary` cosine ranks the buried EX.CO
  node at 3; the z-weighted average is what buried it. A buried carrier can't be gated back → resolving
  the z-average (`RECALL-BURIAL-HANDOFF.md` ▶DO-FIRST) is a precondition. This is why the burial workstream
  and the meshing converge.
- **SETTLED (build, operator-agnostic, proven):** fix the lexical lane (tokenizer + IDF-floor + real
  weight) + normalize lanes onto a commensurable scale.
- **OPERATOR = A/B (open):** tuned-additive convex (Arm A) vs bounded floored gate (Arm B). Eval decides.
- **INHIBITION (bio-only):** transient mild RIF dedup at selection — replaces blind eviction. Floored, reversible, per-query.
- **Episodic-as-gate: [HYPOTHESIS]** — biology verified PFC→hippocampus gating, NOT trace→node. Hold.
- **The intentional divergence from biology:** inhibition is *per-query structural*, not *temporally
  transient* (a node down-weighted on query A is full-strength on query B) — engineer determinism where
  biology leaves it to chance.

---

## 4. Open questions / holds
- **Carrier / z-average DO-FIRST** — fix before gating (precondition). `RECALL-BURIAL-HANDOFF.md`.
- **#11's rescue is partly eval-echo** — two driving traces are from a prior session that asked+answered
  #11 (`cfb74766`); the real April episode isn't embedded. For a *clean* number: backfill April s0
  dialogue OR exclude eval-replay sessions. Mechanism real, magnitude confounded.
- **Episodic-as-gate** — unverified analogy. Hold.
- **co_access trace↔node bridge** (Tom's idea, deferred) — write `node_source_refs` via the judge-selected
  Hebbian step so the structural chain self-densifies. Secondary. Note: its substrate
  (`_hebbian_strengthen`) is the path whose test is currently RED.
- **Backfill** the ~3,500 un-embedded historical s0 dialogue traces (embed worker is 30-day-windowed;
  effective episodic depth ~2 weeks; raw `trace_events` retained from 2026-04-05, never pruned).
- **⚠ `main` moved without me earlier** (`fc77bca` v25 from another stream) despite "single session on
  main" — worth confirming nothing else commits to main.
- **Pre-existing RED test** `test_hebbian_surface_selected_to_co_accessed_edges` (proven not-mine) — fix
  before building co_access on it.

---

## 5. Links · nodes · commands · invariant

**Docs:** `RECALL-DUAL-STORE-DESIGN.md` (architecture) · `recall-gating-inhibition.md` (bio+IR verdict) ·
`memory-biases-and-recall.md` (dossier) · `RECALL-BURIAL-HANDOFF.md` (burial diagnosis + z-average DO-FIRST).
**Research runs:** `wf_52645f55` (bio), `wf_8e7adfee` (IR) — outputs ephemeral in `/tmp`; verdicts persisted.
**Key nodes:** `0dc705a1` dual-store design · `87a04434` gating/inhibition verdict · `05b40294` eval-phase
pointer · `b8b8370b` episodic-lane validated · `8bd36e83` tokenizer bug · `1f301cc5` fts5 anti-IDF ·
`939a5f18` signal÷prevalence · `174fd960` #11 gold · `8359cf1d` evicted EX.CO node · `94f6e01a` meta-trap.
**Commands:** `./dev python3 eval/oracle_audit/<probe>.py` · `./dev pytest tests/test_trace_chain_lane.py -q`.
**Production-safety invariant:** `BRAIN_TRACE_CHAIN` unset/≠'1' ⇒ lane never runs ⇒ recall byte-identical to
pre-`7afb827`. Promotion is a deliberate default-flip after Phase A + Phase B pass.
