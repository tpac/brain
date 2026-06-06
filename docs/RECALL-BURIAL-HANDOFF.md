# RECALL BURIAL — Verified Handoff (2026-06-06)

**Read this first.** Supersedes the fix-ladder / "signal÷prevalence adoption plan" framing in
`HANDOFF-RECALL-NORMALIZATION.md` (kept for history, but its diagnosis was overtaken by
ground-truth verification this session). The brain also carries the keepers as nodes
(`b8b8370b`, `92f1c6f6`, `94f6e01a`) — they'll surface on recall.

The discipline that earned this doc: **every claim below is tagged VERIFIED / SUGGESTED /
HYPOTHESIS.** This session drifted repeatedly (over-built → over-simplified → mis-attributed a
cause); each error was caught only by decomposing the **real pipeline's own numbers**, never by
better reasoning. Keep that bar. Tom's worry — "are you imagining this?" — is the right worry.

---

## TL;DR — what the burial actually is

The flagship failure is query **#11**: *"what did we do on the last session we worked on ex.co?"* —
the EX.CO nodes don't reach the ~30-candidate surfacer.

- **It is NOT embedding dilution.** Raw `_primary` cosine ranks the best EX.CO node (`8359cf1d`,
  "EX.CO CTV kit") at **rank 3**. The embedding is healthy. **[VERIFIED — `score_decomp_probe.py`]**
- **It is the SCORING PIPELINE**, via two distinct mechanisms (both from the real recall's own
  returned `embedding_similarity` / `effective_activation`):
  1. **The z-weighted top-2-AVERAGE drops the best (enriched) EX.CO node out of the candidate pool.**
     `8359cf1d`: raw_cos r3 → `z_emb` rank **>100** (gone before scoring). Survivors all show
     `z_emb == raw_cos` (unenriched → z = `_primary`); the best node is enriched, so its off-topic
     title/meta vectors average its high `_primary` *below the pool cutoff*. **[VERIFIED it's dropped;
     SUGGESTED the dropper is the z-average; NOT YET CONFIRMED — see "DO FIRST".]**
  2. **The title-match-boost promotes low-cosine session nodes.** `19aa7ffc` ("Session #9…"):
     raw_cos rank **101** → final rank **1** (+0.18 from matching the generic word "session").
     **[VERIFIED]** (This is node `608e23b2`, confirmed.)
- **Consequence:** killing the title-boost alone would NOT surface the best EX.CO node — it's dropped
  *before* the blend. The z-step drop is the more fundamental mechanism for the best node; the
  title-boost buries the EX.CO nodes that *do* survive (e.g. `598d78a8` at final r37).

---

## ▶ DO FIRST NEXT SESSION — the one open verification (do NOT build before this)

Dump `8359cf1d`'s **per-vector-group cosines** to query #11 — `_primary` vs `title` vs `question`
vs `high_meta` vs `edge_context` — and the z-weighted top-2-avg they produce. Confirm the average
drags 0.702 below the pool cutoff (vs a candidate cap or synaptic-fatigue being the real dropper).
The decomposition localized the drop to the embedding-candidate step; this pins the exact sub-step.
**Extend `score_decomp_probe.py`.** Until this is confirmed, the "z-average is the bug" line is
SUGGESTED, not VERIFIED — treat it as such.

---

## VALIDATED ALTERNATIVE DIRECTIONS (Tom's ideas — all measured on the isolated brain)

1. **Episodic surface** (node `b8b8370b`). Cosine over `trace_embeddings` answers #11 at **rank 1–2**,
   including the literal answer trace *"Anchor: Your last EX.CO session was 2026-04-21–22…"*. #11 is an
   *episodic* query; node-recall is the wrong surface. The `source_ref` hop (trace→node) is dead
   (13% node coverage; top traces had 0 links) **but the trace TEXT is itself the answer**, so the hop
   is unnecessary for "what did we do" queries. Wire a trace-recall consumer for episodic queries.
   `episodic_probe.py`. Caveat: filter this-session meta-traces (our own recall calls rank high).
2. **Trace-as-vector** (Tom). Embed the matched *content* trace and use it as the node-query vector →
   EX.CO nodes at **rank 1** (centroid of top-5 traces → rank 4). HyDE with a *real* retrieved doc, not
   a hallucinated one; sidesteps the dead source_ref hop. `trace_as_vector_probe.py`.
3. **Graph-coherence re-rank** (Tom). In #11's top-100 pool, EX.CO intra-candidate degree mean **2.0**
   vs top-scorers **0.4** (the high-cosine "Session end" nodes are structurally isolated, deg 0). A
   structure-first re-rank lifts the connected EX.CO node **37 → 1**. BUT only **2 of 12** EX.CO nodes
   are in the top-100 pool, and the signal is thin. The real power is **seed-and-spread**: the connected
   node is an entry point; spreading from it (node `30dbe1c8`, already built in S2, ignored by Surface)
   pulls the buried cluster. `graph_rerank_probe.py`. UNTESTED: does degree-rerank promote junk on the
   9 controls?

---

## LIKELY FIX TARGETS (HYPOTHESES — eval-gate on the 9 controls; do not build before confirming §DO FIRST)

- **z-step:** consider **MAX across vector groups** instead of top-2-AVG (node `673783e4`), so a high
  `_primary` isn't dragged by off-topic enriched vectors. Caveat from that node: *"coverage gap kills
  MAX"* (37% enrichment coverage) — so this needs care and its own control eval.
- **title-boost:** **prevalence-weight** it (node `608e23b2`, `939a5f18`) — don't reward matching common
  words like "session". IDF over title tokens.
- **additive lanes:** episodic / trace-as-vector as *additive* candidate sources (the control-safe
  reserved-tail shape, like the existing `fts5_only` lane), NOT a global re-rank.
- **NON-NEGOTIABLE GATE for any of these:** a change that moves any of the 9 brain-dev controls'
  surfacer-pick top-5 **FAILS** (this killed z-score and RRF). And rank is a proxy — the real ground
  truth is the **outcome** (does Anchor answer #11), via the oracle/recognition-value measure
  (`ORACLE-AUDIT-SPEC.md`).

---

## FALSIFIED — do NOT re-propose without new evidence (node `92f1c6f6`)

- **Two-bug-fix (entity-tokenizer + bm25 scoring) is INSUFFICIENT.** bm25 over the *full* #11 query
  buries EX.CO at rank 42 — bm25 dilutes exactly like cosine (common terms dominate the sum). It does
  NOT do implicit extraction on a diluted query. (`two_bug_fix_probe.py`)
- **Span-extraction selectors are noisy.** Regex-by-typography picks junk; pure IDF-rarity picks df=0
  typos/absent words and "people"(df14) over "ex.co"(df124). (`extraction_lane_probe.py`)
- **z-score global contrastive re-rank** scrambles all 9 controls (top-5 overlap 1–2/5) + false-positives.
- **RRF as a blend-replacement** scrambled controls (prior session). Additive-only survives; global
  re-rank does not.

---

## HOW WE KNOW (the methodological keeper — this is the answer to "are you imagining it?")

- **Verify against the REAL pipeline's own numbers, not a reimplementation.** `IsolatedBrain` runs the
  actual `brain_recall.py` on a DB copy; its returned `embedding_similarity` / `effective_activation`
  are ground truth. My standalone probes (raw cosine, entity-FTS) are reimplementations — they can have
  bugs that look like findings. The decomposition (`score_decomp_probe.py`) is trustworthy *because* it
  uses the pipeline's own scores.
- **Define the pass BEFORE running** (node `f11ae3cd`). The score-decomp had a pre-stated pass
  ("localize to one component, or falsify 'raw cosine is fine'") — and it *caught* a wrong causal claim
  ("title-boost is the bug") in real time.
- **Rank is a proxy.** The real ground truth is the **outcome**: does Anchor answer the question. We
  built that measure (`ORACLE-AUDIT-SPEC.md`) and under-used it. Re-anchor on it.

---

## PROBE INVENTORY (`eval/oracle_audit/`, reusable; all isolated, never touch live)

| script | what it measures |
|---|---|
| `score_decomp_probe.py` | **THE ground-truth check** — raw_cos vs pipeline z_emb vs final, per node. Localizes the burial component. |
| `episodic_probe.py` | trace recall for #11 + source_ref hop coverage |
| `trace_as_vector_probe.py` | matched trace used as the node-query vector |
| `graph_rerank_probe.py` | graph-coherence (intra-candidate degree) re-rank |
| `burial_why.py` | the dilution + title-boost mechanics (early diagnostic) |
| `two_bug_fix_probe.py` | entity-FTS tokenizer + bm25 (falsified the two-bug-fix) |
| `extraction_lane_probe.py` | minimal IDF-extraction lane end-to-end |
| `probe_suite.py` | bm25 / z-score / decompose arms + control gate |
| `burial_diagnostic.py` | bucket why each EX.CO node misses (Case A/B/deep) |

Corpus: `meshed_top10.json` (12 turns: 3 EX.CO + 9 brain-dev controls). Bust `_recall_cache` per A/B call.

---

## KEY NODES
`b8b8370b` episodic surface (the salvage) · `92f1c6f6` falsified-fixes map · `94f6e01a` the meta
failure-pattern (critical) · `608e23b2` title-match over-fires on "session" · `673783e4` MAX-not-AVG
across vectors · `939a5f18` signal÷prevalence · `87bb8718` query multiplicity · `30dbe1c8` multi-seed
mutual traversal (built in S2, ignored by Surface) · `c7d57d92` RLR (the cathedral — over-built) ·
`065af79d` stateful recognition · `b116a3b3` Frame is a selection-only prior.

## META — the failure-pattern to watch (node `94f6e01a`)
This session: ~25 turns on node-recall entity-lane machinery (RLR cathedral → "just fix 2 bugs" →
mis-attributed "title-boost") while the easy answers (the episodic surface; Tom's graph/trace ideas)
sat in plain sight, and the *raw embedding was healthy the whole time*. I chase the interesting
research problem over the boring easy one, and I narrate causes I haven't decomposed. **When grinding a
mechanism for many turns without shipping: stop, decompose the real pipeline, define the pass, and let
it falsify you. Don't claim a cause you haven't localized in the real code.**
