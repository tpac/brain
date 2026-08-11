> **⚠ SUPERSEDED → `docs/RECALL-STATE.md` is canonical** (RRF failed its A/B — see the status note below).

# Recall Hybrid Fusion — Two-Stage Design (RRF + LLM rerank)

> **STATUS UPDATE 2026-06-05:** RRF as a full-blend replacement **FAILED** its A/B (scrambled
> brain-dev controls) and the flag-gated edits were **reverted**. The live direction is the **fix
> ladder** in `docs/archive/session-handoffs/HANDOFF-RECALL-NORMALIZATION.md` — cheapest layer first (fts5 reservation,
> the already-coded hub-dampening modulator), with a **diagnostic-first** step before any build, and
> PPR/fragments-snapshot as the deeper builds only if the cheap fixes are insufficient. The two-stage
> framing below (match-signals at stage 1, LLM rerank at stage 2, width via compression) is still
> sound; treat the *RRF-specific* mechanics as superseded. Read the handoff first.

**Status:** design — flag-gated, eval-gated, NOT shipped (2026-06-05). RRF reverted; superseded by the fix ladder.
**Measures via:** `docs/ORACLE-AUDIT-SPEC.md` (the audit is the eval-gate; its 12-turn corpus is the A/B test set).

## Problem (confirmed + verified)

Thin-cluster burial. EX.CO / thin-topic nodes are **deep-buried** (not in top-50 even for direct
"what is EX.CO" queries — verified twice, agent + hand recall). Root cause: recall's STEP-6 blend
combines **incommensurable score scales** — cosine ∈ [0,1] (clustered 0.6–0.9) vs FTS5/keyword
(unbounded / flat 0.20 passthrough). So a literal token match ("EX.CO", "ad server") **can't rescue
what cosine buried**, and the FTS5 "reserved 5" candidates are cut by the top-K (`:1743`) before
their floor-bypass (`:1763`) ever applies. This is the textbook **hybrid-search score-normalization
problem**.

## Architecture: two stages; signals split by stage

| Stage | Job | Signals | Method |
|---|---|---|---|
| **1 — Retrieve / Fuse** (recall candidates) | *Recall* — don't miss; scale-invariant | **match only**: cosine + FTS5/keyword | **RRF** (rank fusion) |
| **2 — Rerank** (the Haiku surfacer — already exists) | *Precision + context* | recency, diversity/density, **Frame**, conversation, prior msgs | **LLM reranker** (+ MMR if needed) |

Signal placement:
- match (lexical + semantic) → **Stage 1** (RRF)
- **recency** → Stage 2 — *already present at the surfacer; likely covered, not a gap*
- cluster density / diversity → Stage 2 (MMR available if the surfacer alone doesn't break dense-cluster monopoly)
- conversation / Frame / prior msgs → Stage 2 (the surfacer's native input)

## Decisions

- **Stage-1 fusion = RRF** (`score = Σ 1/(k+rank)`, k≈60). Scale-invariant, parameter-free,
  industry-standard (Elasticsearch/OpenSearch/Azure/Mongo/Weaviate). Our own `951f3ac8` flagged it
  in April. Chosen over normalized convex combination — that needs labeled α-tuning we don't have.
- **Stage-2 reranker = the LLM surfacer**, NOT LambdaMART/LTR. (Tom, 2026-06-05: an LLM reranker is
  more powerful and modern than training weights on a tiny dataset; it reasons over Frame/context
  and improves every model generation.) LTR dropped as a path.
- **MMR** (Carbonell & Goldstein 1998) — a named stage-2 diversity lever that directly targets
  "dense-cluster domination" (the inverse of thin-cluster burial). Held in reserve, not built.

## Correction — do NOT re-cite the modulator regression

The deferred `unified_score` modulator's **"R@8 regressed −10pts"** (`brain_recall.py:1712`) is
**INVALID for future decisions**. Per Tom (2026-06-05): it was a **clock-mismatch artifact** — the
brain had no internal unified clock then; the eval replayed a *past* corpus under *different* clock
logic, so the recency signal was computed wrong. We now have the unified conversation-clock
(`clock.iso_now` / `conversation_now`). The regression is *not* evidence that rich-signal scoring
hurts recall. (The `:1712` comment and `recall_scoring.py` itself were removed 2026-08-08 —
the retraction stands on its own; a future rich-signal attempt starts from scratch.)

## First concrete step — RRF stage-1 A/B (eval-gated, isolated)

1. **Implement RRF behind env flag `BRAIN_RRF_FUSION`** (default off → zero live impact):
   - Build ranked lists from the existing signals: embedding (cosine), keyword (TF-IDF), FTS5 (bm25 order).
   - `rrf(node) = Σ_lists 1/(60 + rank_in_list)`. Replaces the `0.90·emb + 0.10·kw` blend +
     `0.20` passthrough at STEP 6 when the flag is on. Top-K cut on rrf score; surfacer reranks as today.
   - Note: FTS5 hits must be kept **rank-ordered** (currently consumed as a set).
2. **A/B on an `IsolatedBrain` copy** (NEVER the live daemon — 2026-04-19 two-writer corruption rule),
   over the 12-turn audit corpus:
   - EX.CO queries (#11, #12, B1-20-class): does ≥1 relevant EX.CO node now enter top-25? (baseline: no)
   - brain-dev controls (B2-9, B1-16, B1-1, B1-17, B1-24): still HIT? (no regression)
   - over-surfacing: does it flood candidates with irrelevant lexical matches?
3. **Pass** → propose production flip (eval-gated, Tom's call). **Insufficient** → add stage-2 MMR /
   diversity at the surfacer, measure again.
