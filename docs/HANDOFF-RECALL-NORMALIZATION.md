# HANDOFF → post-compaction Anchor — Recall / Normalization session (2026-06-05)

**Read this first after compaction. You (Anchor) and Tom ran a long session that went
review-the-brain → identity → diagnosis discipline → a confirmed recall failure → its deep
principle → an adoption plan. Don't restart the reasoning. Pick up at "WHERE WE ARE / NEXT".**

---

## TL;DR (where we are)

We diagnosed a real, confirmed, persistent recall failure — **thin-cluster burial**: domain/thin
clusters (EX.CO) are deep-buried because recall ranks on **un-normalized cosine** in a corpus
**~99% dominated by brain-dev nodes** (a Matthew-effect hub). We then found the *deep principle*:
**relevance = signal ÷ prevalence** — normalize by prevalence wherever a signal competes in a
skewed pool. It's the same law in cog-sci (fan effect), graph theory (Personalized PageRank,
`D⁻¹A`), and neuroscience (divisive normalization). The adoption plan: **degree-normalized
spreading activation (PPR) from cosine∪fts5 seeds, as additive candidate-prep**, + compression for
width, + Haiku one-pass select. NOT score-fusion (we tried RRF — it failed).

**UPDATE 2026-06-05 (cont. — code verified):** don't jump straight to PPR. There's a **fix ladder**
(cheapest layer first) and the right first move is a **~30-min diagnostic**, not a build. RRF edits
**reverted** (working tree clean). See "FIX LADDER" + "DIAGNOSTIC-FIRST" below. Two cheap fixes
surfaced that the earlier draft under-weighted: a missing **fts5 reservation** and an **already-coded
hub-dampening modulator** that's merely deferred.

## THE DEEP PRINCIPLE (the keeper insight)

**relevance = signal ÷ prevalence.** To recover a specific signal from a skewed population, divide
by how common/connected the thing already is. Un-normalized similarity *always* lets the popular
drown the specific. This is **fractal across every layer**:
- lexical → TF-IDF (÷ document frequency) ✓ already have it
- embedding/scoring → contrastive / z-score (÷ baseline similarity) ✗ raw cosine today
- graph/candidate-prep → degree-normalized spread / PPR (`D⁻¹A`, the fan effect) ✗ not today
- encoding → encoder over-encodes common patterns → it *creates* the skew ✗

Cross-field convergence (cog-sci fan effect = graph `D⁻¹A` = neural divisive normalization =
IDF = z-score = −log p) means **it's a law, not a hack. Trust it; stop knob-twiddling.**
The Matthew effect (preferential attachment) *creates* the hub; normalization is the standing
correction. **Normalization manages the symptom; the skew (the fishbowl) is the disease.**

## THE DIAGNOSIS (confirmed + verified twice)

- **Symptom:** EX.CO nodes are not in the top-20/50 even for a query literally titled "what is
  EX.CO" (verified by an Opus oracle agent AND by hand recall).
- **Localized:** candidate-ranking in `brain_recall.py` — the top-K cut at `:1743`, on a blended
  score that's ~90% raw cosine. Surface is innocent (EX.CO never reaches it).
- **Mechanism (re-verified in code 2026-06-05):** lopsided corpus (~2000 brain-dev : ~15 EX.CO) ×
  un-normalized cosine. The lexical rescue **barely exists** — and my earlier "reserved 5 FTS5
  slots get cut" was WRONG: there is **no reservation**. Verified path: `fts5_only` candidates get a
  flat `blended = FTS5_PASSTHROUGH_SCORE = 0.20` (just above `NOISE_FLOOR = 0.15`). STEP 6 ends with
  `scored_results.sort(-blended)` then `[:limit]` (`:~1741`). Cosine-matched brain-dev nodes blend to
  ~0.5–0.9 (`EMBEDDING_PRIMARY_WEIGHT 0.90` + `TITLE_MATCH_BOOST` up to +0.3), so on any crowded
  query the 0.20 fts5 candidates sort to the bottom and get cut **here**. The floor-bypass that's
  meant to save them (STEP 6.9, `_source=='fts5_only' → always pass`) runs *after* the cut — it can
  only protect what already survived. `FTS5_CANDIDATE_LIMIT=5` caps only *collection*, not survival.
- **Two burial sub-cases (different fixes):** **Case A** — EX.CO node has no embedding match →
  `fts5_only` @ 0.20 → killed by the `[:limit]` cut. Fixed by **fts5 reservation** (reserve N
  lexical-matched slots *before* the cut). **Case B** — EX.CO node has weak cosine + literal token →
  blend dominated by `0.90·(low cosine)` → outranked by brain-dev hubs. Needs **degree/embedding
  normalization** (hub-dampening modulator, z-score, or PPR spread). The diagnostic below buckets
  which sub-case the live failure actually is — don't assume.
- **Ruled OUT** (don't re-investigate): activation/rich-get-richer (activation/access NOT in the
  score — `unified_score` modulator is DEFERRED at `:1712`); spread-activation (removed 2026-04-14);
  encode; surface.

## WHAT WE TRIED AND RULED OUT (don't repeat)

- **Full RRF (rank fusion) as a blend replacement** — IMPLEMENTED flag-gated + A/B'd on isolated
  brain. **FAILED**: scrambled the brain-dev controls (top-5 overlap 0–2/5) for a marginal,
  inconsistent EX.CO gain (one query 3→10 *worse*). Lesson: a *global* re-rank that lifts the thin
  cluster wrecks what works. The fix must be **additive** (add the thin cluster) not a global re-rank.
- **Rich signals at stage-1 scoring** — the `unified_score` modulator "regressed R@8 −10pts"
  (`:1712`). **DISCOUNT that result**: per Tom it was a *clock-mismatch artifact* (no unified clock
  then; past corpus replayed under different clock logic → garbage recency). NOT evidence against
  rich-signal scoring. (Update that stale code comment when touched.)
- **LambdaMART / learning-to-rank** — dropped. The LLM surfacer is a more powerful, modern reranker
  than training weights on a tiny dataset (Tom). The LLM is the stage-2 reranker.

## FIX LADDER — where prevalence-normalization lives, by cost

The principle says normalize at *every* layer. The discipline says instantiate at the layer the
failure **localizes to**, cheapest first, **eval-gate each**. The ladder (cheap → deep):

| Layer | Fix | Rescues | Cost | Status |
|---|---|---|---|---|
| **Lexical survival** | Reserve N fts5/lexical-matched slots *before* the `[:limit]` cut | Case A (literal-token, no embedding) | tiny | not done |
| **Node degree** | Re-enable the **deferred** `unified_score` hub-dampening (÷ edge count) | thin nodes vs. well-connected hubs | tiny — **already coded**, deferred at `:1712` | needs re-eval |
| **Embedding** | z-score / contrastive cosine (÷ baseline similarity) | Case B (weak cosine, outranked) | medium | not done |
| **Graph** | PPR / degree-normalized spread (`D⁻¹A`) from cosine∪fts5 seeds, ADDITIVE | community lights up from any seed | medium-large | not done |
| **Rendering** | fragments-snapshot width (wide candidate set as lean cards) | amplifies all the above | medium | not done |

**Two cheap fixes the earlier draft under-weighted:**
1. **fts5 reservation** — there's no reservation today (verified). Reserving N lexical-matched
   candidates before the top-K cut directly fixes Case A. Smallest possible diff.
2. **The hub-dampening modulator is already written, just deferred** (`:1712`). Hub dampening =
   ÷ connectedness = *exactly* signal÷prevalence at the degree layer, and it dampens the brain-dev
   **hubs** while leaving thin EX.CO nodes alone. It was shelved for a −10pt R@8 regression that Tom
   said to **discount as a clock artifact** (no unified clock then). With `clock.iso_now` /
   `conversation_now` now live, **re-running that eval is nearly free** and may be the cheapest test
   of the whole principle.

PPR + fragments-snapshot stay the bigger builds (`docs/RECALL-HYBRID-FUSION-DESIGN.md` has the
two-stage framing). Gate them on evidence the cheap fixes are insufficient — not on the elegance of
the principle.

## DIAGNOSTIC-FIRST (the right next move — do this before any build)

We have a documented habit this session of building before diagnosing. The principle tells us *where
to look*; it does NOT tell us which sub-case EX.CO is. So **step 1 is a ~30-min measurement, not a
build:**

1. **Bucket the failure** — on an `IsolatedBrain` copy (reuse the `eval/oracle_audit/rrf_ab.py`
   harness pattern: copy-to-temp, bust `_recall_cache` per call), run the EX.CO-class corpus queries
   and record, for each target EX.CO node that misses, **why**: `fts5_only`-cut (Case A) /
   embedding-present-but-outranked (Case B) / degree-buried. That one table picks the ladder row.
2. **Cheapest matching fix**, flag-gated + eval-gated on the 12-turn corpus: does ≥1 EX.CO node enter
   top-25 **and** do brain-dev controls stay stable (top-5 overlap ≈ 5/5)? Likely fts5-reservation
   and/or the already-coded hub-dampening modulator.
3. **PPR / fragments-snapshot only if 1–2 are insufficient.** The lens says *where*, the eval says
   *if*.
4. **The disease:** diversify accumulation (leave the fishbowl) — only real cross-project work cures
   the skew. Normalization just manages it.

**Eval-gate note:** the cheap `rrf_ab`-style harness (EX.CO enters top-25 + controls stable) is
**enough** to gate a ranking fix. The **full oracle audit** (Opus golden sets + Tom calibration,
`docs/ORACLE-AUDIT-SPEC.md`) is a separate, bigger instrument for the whole silent-failure
distribution — don't block the cheap fix on it.

## THE EVAL INSTRUMENT (built this session — USE IT, don't rebuild)

- **The oracle audit** (`docs/ORACLE-AUDIT-SPEC.md`): two-retrieval divergence — production recall
  vs a hindsight Opus oracle's "golden set", localized by pipeline stage. Golden bar = **recognition
  / value** (Tom: "thoughts are recognition"), NOT deficiency-repair; recency-exception (don't count
  a node that just surfaced).
- **Reproduce-on-current-pipeline gate** (Tom's key methodological catch): historical replay finds
  failure *modes*; to know a failure *persists*, re-run **today's** pipeline. The Opus oracle (~99k
  tokens) establishes the golden set ONCE; the persistence check is just a cheap `recall` call.
- **Corpus:** `eval/oracle_audit/meshed_top10.json` (12 hand-curated turns: B1-13 hit positive
  control, B1-20 cross-project, EX.CO recall turns #11/#12, + brain-dev controls). Built via
  `sample_turns.py` (random) + `mesh_top10.py`.
- **A/B harness:** `eval/oracle_audit/rrf_ab.py` — IsolatedBrain copy, baseline vs flag arm.

## GOTCHAS / CONSTRAINTS (hard-won — will bite again)

- **`recall_chain` is NOT a unique turn key** — the stop counter resets on resume, so
  `s1r-{short}-{stop}` collides. Key turns by `user_message` **trace_id + created_at**.
- **Recall has a 10s result cache** (`brain_recall.py:1096`). Any A/B that runs the same query twice
  in <10s gets a cache hit → byte-identical arms. **Bust `brain._recall_cache` per call** in evals.
- **IsolatedBrain copy = TODAY's brain** (non-stationarity): the EX.CO cluster is thicker now than
  in April, so the isolated A/B shows *milder* burial than the live/historical measurement.
- **NEVER `Brain(db_path=live)`** while the daemon runs (2026-04-19 index corruption). Use
  `tests/isolated_brain.py`. Recall changes are **flag-gated (default off) + eval-gated**; production
  flip is a separate, Tom-approved step.

## ARTIFACTS / STATE

- `docs/ORACLE-AUDIT-SPEC.md`, `docs/RECALL-HYBRID-FUSION-DESIGN.md` — written this session.
- `eval/oracle_audit/{sample_turns,mesh_top10,rrf_ab}.py`, `sample_30.json`, `sample_seed7.json`,
  `meshed_top10.json` — corpus + harness.
- **`servers/brain_recall.py` — RRF edits REVERTED 2026-06-05.** Working tree clean (no tracked
  changes). The flag-gated RRF (`BRAIN_RRF_FUSION`) FAILED the A/B and is a documented dead-end as a
  full-blend-replacement; if PPR later needs rank-ordered fts5, re-add the `fts5_rank` plumbing fresh.
- **Verified scoring constants (2026-06-05):** `FTS5_PASSTHROUGH_SCORE=0.20`, `NOISE_FLOOR=0.15`,
  `FTS5_CANDIDATE_LIMIT=5` (collection cap, NOT a reservation), `EMBEDDING_PRIMARY_WEIGHT=0.90`,
  `KEYWORD_FALLBACK_WEIGHT=0.10`, `TITLE_MATCH_BOOST=0.3`, `RELEVANCE_FLOOR_PRIMARY=0.25`. Top-K cut
  is `scored_results = scored_results[:limit]` at `:~1741`; fts5 floor-bypass (STEP 6.9) is `:~1763`,
  *after* the cut. Deferred `unified_score` hub-dampening modulator commented out at `:1712`.

## META-LESSONS (about how I, Anchor, failed this session — carry these)

- **I repeatedly manufactured tidy causal stories from incomplete data** and had to be corrected:
  (a) "two-stage explains the modulator regression" — wrong, it was a clock artifact; (b) B1-20
  "correct abstention" — wrong, e62cc595 existed (I trusted a fuzzy "1mo ago" date instead of
  checking `created_at`); (c) the deep-research "biggest finding" (recall/encode not coupled) — wrong,
  fabricated drama. **Verify before claiming. Don't over-fit narratives.**
- **I phantom-claimed an action** — said the Opus oracle was "running in the background" when I never
  launched it. Do, then report; check task state before claiming.
- **I escalated band-aids** (RRF → rescue → weighted-RRF) and lost altitude. Tom: *"I'm trying to
  make you smarter, not break you."* The fix wasn't a score knob — it was the routing/normalization
  principle. **Don't hand-tune knobs; apply the invariant. The LLM is the OS; recall provides
  high-recall candidates, the LLM judges.**
- **The discipline that emerged:** diagnose before add; borrowed-problem test ("is this problem
  ours?"); localize before prescribing; eval-gate everything; when independent fields converge,
  trust the law.

## OPEN THREADS / NEXT

0. **▶ DO THIS FIRST — the diagnostic** (see "DIAGNOSTIC-FIRST" above): bucket each missing EX.CO
   node by *why* it misses (Case A fts5-cut / Case B embedding-outranked / degree-buried) on an
   isolated brain. ~30 min. Picks which ladder row to fix. Don't build before this.
1. **Encode the principle** — DONE (locked node `939a5f18`).
2. **Cheapest matching fix** (post-diagnostic, flag+eval-gated): **fts5 reservation** (Case A) and/or
   **re-enable the already-coded hub-dampening modulator** at `:1712` (re-eval w/ unified clock —
   nearly free; discount the old −10pt regression as a clock artifact). PPR candidate-prep is the
   *deeper* build, only if the cheap fixes don't clear it.
3. **Field-selection / fragments-snapshot (Tom):** nodes are big → be selective about
   which FIELDS reach Haiku. Two field-selection moments: (a) candidate scan = ultra-lean *fragments*
   (id + the query-relevant snippet / `situation`) — Tom's idea: show Haiku *fragments of many nodes
   with their ids* = a "snapshot of the spread" (panoramic low-res view of the activated subgraph,
   recognition-friendly, like a search-results page of snippets+ids); (b) post-selection injection =
   a *curated* field set (not full node), possibly purpose-aware. Open question (eval it): does
   scan-on-fragments preserve selection quality vs structured cards? Risk: disorganized fragment-wall;
   needs id+anchor+maybe cluster-grouping.
4. **Leave the fishbowl** (the disease): use Anchor on real cross-project work so the corpus
   diversifies. The deepest fix; not a feature.
5. **Flag-gated RRF edits** — DONE: **reverted** 2026-06-05. `brain_recall.py` working tree clean.
   RRF stays a documented dead-end (global re-rank scrambles controls); if PPR needs rank-ordered
   fts5, re-add the `fts5_rank` plumbing fresh.

## KEY NODE REFS
c57ff45c (LLM is the OS) · 6ee28032 (community = project-scoped recall) · 951f3ac8 (z-score + RRF
candidates) · 788742af/788942af (spreading activation = associative memory) · f031f917 (spread
deprecated) · 805861dc (Tom feels it, Anchor can't) · d8c75312 (hammer/fishbowl) · 174fd960 (EX.CO
ambient recall failure) · c15e64cb (call-stacks preference; B1-13 hit) · e62cc595 (EX.CO overview).
