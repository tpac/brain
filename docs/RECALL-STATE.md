# RECALL — Canonical State (2026-06-08)

> **THE single source of truth for where recall stands.** Supersedes the scattered/contradictory
> handoffs (see §6). If another recall doc disagrees with this one, this one wins.
> Built from a validated control-fails eval + a thorough harvest of all 13 recall docs.

## 1. Validated truth (what's real)

- **Control recall fails 15/30** on a validated, S0-grounded corpus (`eval/oracle_audit/control_corpus.json`).
- **Essential-node coverage: 57% in top-5, 83% in top-25** (the surfacer's input window). Single-fact
  ("remote") queries: **perfect, 0/6 fail**. Failures concentrate in episodic + multi-node answers.
- **Numbers are trustworthy:** gold was tightened (94→58 essential, dropped cluster-padding), two
  buried-answer gold errors hand-fixed (TO3, EP5), and the eval survived a code-review — one real bug
  (time-scope post-filter) found + fixed, **headline unchanged**. Nodes: `795a2d96`, `9212bce4`.
- **Prior inflated framing corrected:** the early "18/30 / 47% / compositional gap" came from
  over-marked gold; the validated figure is 15/30 · 57%/83%.

## 2. The mechanism (why the 17% misses) — node `18a60483`

Per-miss diagnosis of the 10 essential nodes never reaching top-25:

| mechanism | share | what it means |
|---|---|---|
| **z-average SCORING burial** | ~40% (4/10) | raw `_primary` cosine *already ranks them top-12*; the pipeline's z-weighted multi-vector averaging + title-boost buries them out of top-25. The embedding has the answer; the scoring discards it. |
| **query DILUTION** | ~40% (4/10) | a single rare term (`ex.co`, `similar_to`, `research`) ranks the node **#1–2**, but the long multi-word query drowns it. |
| **no vector** | 1/10 | node never embedded (recent) — encoding-coverage gap. |

Backstops already in the pipeline: **graph/spread reaches ~60% of misses 1–2 hops** from surfaced
seeds; **keyword/fts admits 7/10**. "Episodic" was the *symptom* (long vague queries dilute + their
answers are scoring-buried), not a distinct mechanism.

## 3. The levers, prioritized (DO measure-first)

**Diagnostic-first (30 min, not a build):** rerun the control-failures through `eval/longmem/analyzer.py`
(11-bucket: `ranker_buried` vs `encoder_filtered` vs index-miss) to confirm attribution before building.
The fix ladder (from `HANDOFF-RECALL-NORMALIZATION`, principle = **relevance ÷ prevalence**): apply at
the layer the failure localizes to, cheapest first.

1. **#1 — z-average / carrier scoring fix** (the DO-FIRST; `RECALL-BURIAL-HANDOFF`). Recovers the ~40%
   scoring-buried. **Cheap start:** the **hub-dampening modulator is already coded + deferred** (≈
   `brain_recall.py:1712`, `signal÷prevalence` at the degree layer); its old −10pt regression was a
   **clock-mismatch artifact** — re-evaluate with the unified clock before anything new.
2. **fts5 reservation-before-cut** — reserve lexical-matched candidates *before* the `[:limit]` cut
   (today's floor-bypass runs *after* the cut, can't rescue dead nodes; finding `703a9402`). Cheap;
   admits the dilution/keyword misses (fts already finds 7/10).
3. **Graph / cluster expansion** (the backstop already reaches ~60% of misses 1–2 hops away). Three
   underused signals to wire in:
   - **(a) aspect-aware spread** (insight `d26ad9fb`) — steer expansion by prompt/arc aspect proximity.
   - **(b) consume `co_anchored` edges** (shipped `07ab3f1`, live in dispatch but NOT used by recall).
   - **(c) community-completion** — communities are used in the **Frame** (ambient prior) but **never in
     recall retrieval/ranking** (`brain_recall.py` = zero community usage; `community_member` is
     *deliberately excluded* from generic spread — it over-connects). A *targeted, query-gated*
     completion — when a candidate is a member of a thick/active community, pull the community **node**
     (its synthesized summary) + relevant co-members, MMR-bounded — **directly attacks the multi-node /
     whole-cluster miss** (recall gets 1–3 of N essential; the rest are co-members). = CA3
     pattern-completion applied to S2 communities. Node `f-community-completion` / see brain.
4. **Episodic-recency promote-GATE** (insight `1939273d`) + **power-law decay tail** (dossier) — episodic
   traces from window X *promote* nodes in an already-good list (recency/decay), NOT a retrieval lane.
   Power-law decay keeps old-but-important nodes reachable instead of bottoming at zero.
5. **Query-decomposition for dilution** — isolate the rare entity, retrieve it separately. ⚠ NOT
   query-intent *classification* — that was empirically rejected (`RECALL-DUAL-STORE-DESIGN` §4, Exp B:
   match-strength doesn't separate episodic from control).
6. **Disconfirmation pass** (dossier's flagged #1 gap) — a 2nd retrieval seeded from correction-aspect
   edges + a valence-stripped query, to surface what the prompt/Frame prior did *not* pre-activate.

**Standing direction** (insight `acd991ae`): mine underused layers — aspects, Frame, S0 traces, S2
communities, open KV fields (`situation`/`question`/`reasoning`) — before building new machinery.

## 4. Eval tooling (reuse — don't rebuild)

- **Corpus:** `eval/oracle_audit/control_corpus.json` (30 Qs, real moments, dual-tier gold).
- **Scorers/probes** (`eval/oracle_audit/`): `control_score.py` (headline, recall-only),
  `control_miss_mechanism.py` (per-miss process), `control_misses.py`, `control_reach_diagnose.py`,
  `control_corpus_review.py` (gold-as-titles review), `control_postspread_eval.py`.
- **Platform:** `eval/longmem/` Frozen Corpus (encode-once, sweep-many) + `analyzer.py` 11-bucket
  attribution + S2-reached probe. `IsolatedBrain` for daemon-safe runs.
- **Corpus blind-spot to fix in v2** (`6759bb48`): gold compiled from the recall pool can't contain
  nodes recall fully buries → seed gold from the question's moment-anchor too.

## 5. Killed / dormant (do NOT re-propose without new evidence)

- **Lexical keyword lane** ("arm A", porter-bm25 additive): net-neutral-to-negative; `3c315383`. The
  flag-gated `BRAIN_LEXICAL_V2`/`BRAIN_RECALL_ARM` substrate is **uncommitted on disk** — revert or leave
  dormant (Tom's call). *(Note: fts as an admission backstop ≠ a ranking lane — keep the former.)*
- **Trace-chain as a primary RETRIEVAL lane**: echo-confounded (query→trace cosine returns the prior
  *asking* of the question), ~43% of dialogue un-embedded; `bc960665`, `2b1c7751`. *(Episodic traces as a
  promote-GATE — §3.4 — is the live form.)*
- **RRF full-blend fusion**: failed A/B, scrambled brain-dev controls.
- **Gate/inhibition operator A/B** (groups B/C): low-ROI — an operator won't fix scoring-burial/dilution.
- **Query-intent classification**: rejected (Exp B).

## 6. Doc map (after consolidation)

- **Canonical:** this file (`RECALL-STATE.md`).
- **Vindicated — keep, still actionable:** `RECALL-BURIAL-HANDOFF.md` (the z-average #1 lever + probe
  suite), `HANDOFF-RECALL-NORMALIZATION.md` (the relevance÷prevalence fix ladder), `RECALL-OVERVIEW.md`
  (pipeline reference), `research/memory-biases-and-recall.md` (the opportunity dossier),
  `EPISODIC-REFERENCES.md` (co_anchored substrate), `RECALL-TEMPORAL-ANCHOR-SPEC.md` (episodic temporal anchor).
- **SUPERSEDED — banner → here:** `DUAL-STORE-EVAL-HANDOFF.md` (builds the killed lexical A/B/C),
  `RECALL-HYBRID-FUSION-DESIGN.md` (RRF failed), `research/recall-gating-inhibition.md` (the gate A/B is
  low-ROI; the bio/IR knowledge stays sound). `RECALL-CONTROL-FAILS-FINDINGS.md` carries the early
  numbers — read this doc for the validated set. `RECALL-FUNNEL-PLAN.md` design intent valid, line-refs stale.

## 7. Next session — prioritized plan

1. **Diagnostic:** run control-failures through the 11-bucket `analyzer.py` → confirm % `ranker_buried`
   (z-average) vs `encoder`/index. Measure before building.
2. **Lever #1:** re-evaluate the already-coded hub-dampening modulator with the unified clock + the
   z-average/carrier fix (`RECALL-BURIAL-HANDOFF` DO-FIRST). Eval-gate on `control_score.py`.
3. **fts5 reservation-before-cut** (cheap; admits the dilution misses).
4. **Re-run `control_postspread_eval.py`** — `f3520f28` shipped the cluster-spread fix (commit `33661fc`);
   measure whether cluster-completion now closes the graph-recoverable misses.
5. Then, by appetite: aspect-aware spread + `co_anchored` consumption + **community-completion** (targets
   the multi-node gap directly); episodic-recency gate + power-law decay; disconfirmation pass.

**Cross-stream (this session, attributed):** `f3520f28` shipped the cluster-spread fix +
`LINEAGE_FAMILIES→aspects` + noise guard. `7c3447b8` root-caused recall timeouts = S1Surface latency
(Haiku 55–75% of recall, 18.7k-token prompt, 98% tool-fire, httpx pool expiry); keepalive shipped pending merge.
