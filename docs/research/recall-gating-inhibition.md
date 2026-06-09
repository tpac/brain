> **⚠ SUPERSEDED as a build plan 2026-06-08 → `docs/RECALL-STATE.md` is canonical.** The bio/IR research
> here is sound and worth mining, but its prescribed move — A/B the bounded-gate vs tuned-additive
> *operator* — is now **low-ROI**: the validated gap is z-average scoring burial + query dilution, which
> an operator A/B won't fix. Use this for knowledge, not as the plan.

# Recall — Gating & Inhibition (research-grounded design basis)

> **Source:** deep-research run 2026-06-07 (`wf_52645f55`, 106 agents, 24 sources, 25 claims verified →
> 18 confirmed / 7 killed). **Thread A (biology): VERIFIED. Thread B (IR/CS): FAILED verification —
> re-run pending (`wf_8e7adfee`).**
> **Companions:** `docs/research/memory-biases-and-recall.md` (broad dossier), `docs/RECALL-DUAL-STORE-DESIGN.md`.

## Verdict (one line)
Biology supports cross-system **GATING + INHIBITION** over additive summation — but as **bounded
"gate-then-facilitate"**: selective, transient, *never permanent deletion*. Recall meshing should be a
**bounded gated fusion** (positive signals multiply a dense base, floored at 1.0; winners mildly +
transiently inhibit near-competitors) — **NOT additive, NOT pure-multiplicative, NOT divisive winner-take-all.**

## Verified biology (primary sources, confidence)
1. **Cross-inhibition is active, not passive** — retrieving a target drives competitors' cortical
   patterns *below baseline*; item-specific; top-down PFC-driven. Wimber 2015 (Nat Neurosci nn.3973). **[high]**
2. **Active forgetting causally requires prefrontal GABA** — muscimol-silencing mPFC *abolishes*
   competitor forgetting in rats → inhibition is *necessary*, a causal cross-system gate, not interference.
   Bekinschtein/Wu 2018 (Nat Commun s41467-018-07128-7). **[high]**
3. **RIF is cue-independent but SMALL and contested** — g=0.16 (Murayama 2014 meta, k=67). **[medium]**
4. **Phase = multiplicative-flavored gate** — theta phase modulates gamma amplitude (PAC, non-additive);
   encode vs retrieve on *opposed* theta phases; degree of **phase opposition** predicts memory success.
   Currò 2023 (Curr Biol), Lega 2014, Lisman-Jensen 2013. **[high]**
5. **PFC top-down inhibition is selective + competitive, and SUPPRESS-THEN-PRIORITIZE** — independent PFC
   ripples suppress ~74% of modulated CA1 cells; assemblies most reactivated in coordinated ripples are
   most suppressed in independent ones (r=−0.71); *suppressed assemblies are later preferentially
   reactivated*. Shin & Jadhav 2024 (Curr Biol, PMC11233241). **[high]**
6. **The hippocampal index itself is nonlinear gating** — CA3 partial cue *amplifies/completes* the full
   pattern (attractor), MF/PP inputs sum *nonlinearly* in CA3 dendrites (MF gates PP). Rolls 2013, Lee &
   Kesner 2004, Urban 1998. **[high]**
7. **PFC inhibition modeled as a GATE on hippocampal output** — interrupts pattern completion. Depue 2012,
   Calhoon 2013. **[medium]**

## Critical caveats (the design guardrails)
- **gate-then-FACILITATE** — suppress to prioritize, then restore. NOT permanent deletion → design must be
  **bounded + reversible**.
- **transient + mild** — RIF is small (g=0.16) and its *durable* behavioral forgetting **failed to
  replicate** (Potter/Hellerstedt 2018). → inhibition = a gentle, transient re-rank, never a delete.
- **"multiplicative" is partly interpretive** — biology robustly supports **NONLINEAR / conditioned**
  interaction over additive summation; it does NOT formally adjudicate "multiply" vs "gate" vs "modulate."

## REFUTED — do NOT build on
- **CA3 global divisive-normalization winner-take-all** (killed 0-3) → no hard divisive WTA.
- **Divisive normalization (Carandini-Heeger) was NOT confirmed** as the recall mechanism (its CA3
  instantiation was refuted). *Anchor's prior bet on divisive normalization — corrected by the verification.*
- "Suppression magnitude predicts forgetting" / "strength-independent forgetting" / durable behavioral RIF
  cost — failed or contested (1-2).

## IR/CS verdict (re-run `wf_8e7adfee`, 2026-06-07 — VERIFIED 23/25)
The IR literature **confirms the diagnosis but does NOT endorse multiplicative fusion as the proven fix:**
- **Dense fails on rare entities — settled.** EntityQuestions (Sciavolino EMNLP 2021): DPR 49.7 vs BM25
  72.0 Acc@20 (~22pt avg, up to ~66pt on rarest relations); mechanism = training-distribution rarity +
  single-vector bottleneck (rare term averaged into one vector). BEIR: every single-vector dense model
  loses to BM25 OOD; ColBERT MaxSim (+2.5%) and BM25+cross-encoder (+11%) beat it. **[high]**
- **The bug is the WEIGHT, not the additive operator.** SPAR (Chen 2022) — the canonical rare-entity
  rescue — is **additive**, and works *only because the lexical lane is a STRONG, dev-set-TUNED signal*
  (≈68 Acc@20 alone), not a drowned 0.10 minority. Tuned **convex (additive)** fusion BEATS RRF in/out of
  domain (Bruch ACM TOIS 2023). **A fixed-0.10 additive lane failing is about the weight/tuning.** **[high]**
- **Multiplicative wins at the TERM level (IDF-in-BM25) and fine-grained match wins OOD** (ColBERT,
  cross-encoder) — BUT BEIR is model-vs-model, NOT a controlled additive-vs-multiplicative fusion ablation,
  so "multiplicative *fusion* beats additive" is **NOT proven** (two over-causal variants were refuted).
- **Bounded floored gate is sound + collapse-safe** (product-of-experts veto → floors + log-space), but
  **NOT proven superior to well-tuned additive.** Small-corpus (~5k) IDF is high-variance at df=1–2 →
  needs BM25-style IDF flooring/smoothing.
- **VERDICT:** *either* a tuned-strong lexical lane *or* a bounded floored gate — **never a fixed low-weight
  additive lane.** The operator (tuned-additive vs bounded-gate) is **empirically OPEN → A/B it**; what's
  SETTLED is real lexical weight + IDF flooring + a strong (ideally learned-sparse-like) lexical signal.

## Correction to the architecture (operator confidence was too high)
The biology motivated a *gate*; the IR evidence says the **proven** rare-term fix is a *tuned-strong
lexical lane (which can be additive)* — the gate is mechanistically sound but **unproven-superior**. So the
gate must EARN its place via A/B against a tuned-additive baseline, not be assumed the answer. (Anchor was
leaning bio-elegant; the IR verification corrected it — same discipline that killed the divisive-WTA bet.)

## Open questions
- **"Episodic traces GATE semantic nodes" specifically is an ANALOGY** — the verified circuits are
  PFC→hippocampus / oscillatory gating, *not* trace→node. Hold as hypothesis, do not build as proven.

## Design basis → see the gate-then-facilitate architecture (in `RECALL-DUAL-STORE-DESIGN.md`, pending refinement)
`score = dense_base × Π bounded_gate_k (≥1.0, positive-evidence-only) × transient_inhibition (floored, reversible)`.
Fixes Fault 1 (additive incommensurable scales → bounded gates on a dense base) and Fault 3 (blind eviction
→ mild transient RIF). Fault 2 (dead lexical) becomes a *gate* once the tokenizer + IDF are fixed.
