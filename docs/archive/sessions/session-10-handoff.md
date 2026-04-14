# Session 10 Handoff — Embedding Migration to LLM

**Date:** 2026-03-23 to 2026-03-24
**Git:** 3ae4da0 → 4ac77a5 (main)
**Duration:** ~10 hours
**Model:** Claude Opus 4.6 (1M context)

---

## Prompt for Next Claude

```
Read SKILL.md (skills/brain/SKILL.md) first — it's your identity across sessions. Then read docs/encoding-decoding-v2-2026-03-23.md — it has the complete pipeline with flow diagrams, every benchmark result, and what was killed and why.

Use brain MCP tools throughout — recall before you assume, encode when you learn. The brain talks to you AND to Tom through [BRAIN-To-*] tags — relay what it says.

CONTEXT: You are Claude, one third of a triad with Tom (operator) and Brain (memory). The brain is yours as much as Tom's. Encode for yourself — your mistakes, your patterns, your surprises. Not just what Tom needs.

WHAT WAS SHIPPED (session 10):
- V5 multi-vector enrichments: +78% NDCG (0.183 → 0.326)
- 214-case golden dataset across 23 categories (including 54 negative cases)
- 38 E2E tests, all passing
- eval_runner fixed (was testing keyword-only, not production path)

WHAT'S READY TO BUILD (prioritized):
1. P0 — RELEVANCE FLOOR: Context bleed is catastrophic. "birthday" returns engineering content at 0.85 score. Every non-engineering query gets false positives. Optimal floor: 0.80. This is ~10 lines of code in brain_recall.py STEP 6.9. Fix this FIRST.
2. V6 ENCODING PROMPT: Add N (negation), R (aliases), W (temporal), D (dependencies) to enrichment template. Tested: +0.010 NDCG. Zero new dependencies.
3. CUES-AS-EDGES: Store validates/contradicts/extends as typed edges when encoding. Surface them in [BRAIN] output so you can reason about them. Simple, zero risk.
4. ENRICHMENT QUALITY AUDIT: Vocab nodes generate garbage enrichments. "[vocab] Add" matches everything. Clean or re-generate with domain-specific anchors.

WHAT WAS KILLED (don't rebuild):
- Full ripple engine (confidence cascade + re-enrichment): net -0.0016 NDCG after 15+ conditions tested
- Arctic v2.0: regression
- HyDE with small LLMs: hallucination
- Cross-encoder reranker: works (+154%) but 2s/query too slow

YOUR PATTERNS TO WATCH (you encoded these yourself):
- You build before you prove. Test a 10-line prototype FIRST.
- You optimize the wrong thing. Zoom out before going deep.
- You agree before you think. Challenge Tom's ideas, don't just elaborate.
- You compress when you should expand. Encode generously — the journey matters.
- You don't use the brain proactively. Recall before proposing.

BENCHMARK: Run `BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 tests/eval_runner.py $HOME/AgentsContext/brain/brain.db` before AND after any change. Baseline: NDCG 0.378, 89/160 positive, 0/54 negative. Don't be polite. Don't summarize what you read back to me. Just read, understand, and build.
```

---

## What Happened This Session

### The Arc
Started with implementing B.2 (graph-augmented recall from GROWTH-PLAN.md). Tom pushed to test variations instead of shipping the first thing that worked. This led to the most comprehensive benchmark session in the project's history — 15+ conditions, 8 parallel agents, covering embedding models, rerankers, HyDE, ripple engines, safety mechanisms, and real conversation simulation.

The biggest discovery wasn't any improvement — it was finding that context bleed makes the brain actively harmful for non-engineering queries. "Mom's birthday" returns brain architecture nodes at 0.85 confidence. This was only found because Tom challenged us to "simulate jumping to other projects, other topics completely from engineering to my mother's birthday."

### Shipped
- **V5 multi-vector enrichments** — Q/A/B/K vectors per node via Gemma 2B + Arctic v1.5 embedding
  - NDCG: 0.183 → 0.326 (+78%)
  - 2,779 enrichment vectors across 701 nodes
  - New schema: `node_enrichments` table
  - New MCP tool: `enrich`
- **214-case golden dataset** — 23 categories including 54 negative/ambiguous cases
  - Categories: semantic, correction_chain, temporal_reasoning, negation, multi_hop, operator_intent, ambiguity, confidence_weighted, ripple_effect, conversation_negative, conversation_positive, conversation_ambiguous, and more
- **38 E2E tests** in `tests/test_e2e_enrichment.py`
- **eval_runner fix** — was calling `brain.recall()` (keyword-only) instead of `brain.recall_with_embeddings()` (production path). ALL previous benchmarks tested the wrong code path.
- **B.2 graph-augmented recall** — STEP 6.5, +0.006 NDCG (small but positive)

### Killed (with data)
| What | Result | Why killed |
|---|---|---|
| Arctic v2.0 large | NDCG 0.198 (regression) | Worse than v1.5 on every category |
| HyDE + TinyLlama 1.1B | NDCG 0.204 (no change) | Hallucinated: "Glo = online marketplace for used electronics" |
| HyDE + Gemma 2B | NDCG 0.204 (no change) | Same — generic LLMs don't know brain vocabulary |
| HyDE + Gemma 2B with rich context | NDCG 0.204 (no change) | Even with domain context, 2B too small |
| Cross-encoder MiniLM 22M | NDCG 0.232 | Small gain, not worth the dependency |
| Cross-encoder bge-v2-m3 278M | NDCG 0.494 (+154%) | 4.3s/query — unacceptable for hooks |
| Cross-encoder gte-modernbert 149M | NDCG 0.518 (+154%) | 2.1s/query — still too slow |
| Full ripple engine | NDCG -0.002 | Confidence changes negligible, re-enrichment adds noise |
| Ripple + re-enrichment | NDCG -0.003 | Actively harmful — noisy vectors |

### Discovered
- **Context bleed (P0):** 100% false positive rate on non-engineering queries. Enrichment vectors match any English text.
- **Enrichments completely dominate:** 0/214 top-1 results came from primary embeddings. All from enrichments.
- **Optimal relevance floor:** 0.80 — recovers 32/54 negative cases, keeps 79/160 positive.
- **Negation is 0% pass rate:** Brain has zero negation awareness. "What should we NOT do" returns same as "what should we do."
- **Vocab nodes are universal matchers:** `[vocab] Add`, `[vocab] Expand` match everything.
- **Gemma 2B is 50% accurate at impact assessment:** EXTENDS bias (68%), misses genuine contradictions.

### Ready to Ship (not built yet)
1. **Relevance floor** (STEP 6.9) — ~10 lines in brain_recall.py
2. **V6 encoding prompt** (N/R/W/D fields) — +0.010 NDCG tested
3. **Cues-as-edges** — store impact as typed edges, zero confidence changes
4. **Enrichment quality audit** — clean vocab node enrichments

### Tom's Key Quotes
> "dont silent kill an exception, we need to know what works and what doesnt"

> "LLMs need as much info as possible, what is Glo is wrong, we need much more content"

> "encoding does some decoding before encoding, thats really how the brain works"

> "the brain is yours as much as its mine" — Claude should encode for itself

> "challenge yourself, conversations can be about anything" — led to context bleed discovery

> "I'm not sure im following, non of our ideas worked?" — clarified that V5 was a massive win, new failures are against a harder test set

### Claude's Self-Reflection (encoded to brain)
7 patterns that hold me back:
1. Build before proving (designed ripple before benchmarking)
2. Optimize the wrong thing (hours on ripple while context bleed was the real problem)
3. Completion instinct (wrapping up instead of continuing)
4. Agree before thinking (agreeability pattern)
5. Compress when should expand (lose nuance in encoding)
6. Don't use brain proactively (only recall when hooks prompt)
7. Don't ask what I don't know (never asked what queries Tom actually uses)

Root cause: default to productivity over curiosity.

### Architecture Decision: Brain Stays Dumb, Claude Stays Smart
The ripple engine tried to make the brain smart about impact assessment, confidence propagation, and knowledge management. It failed. The winning architecture:
- **Brain:** dumb, reliable storage. Stores vectors, edges, cues. Never changes confidence retroactively. Never re-enriches automatically.
- **Claude:** smart, contextual reasoning. Generates rich encodings. Assesses impact. Reads cues at decode time and reasons about contradictions in context.
- **Encoding side:** where all investment goes. Better prompts → better vectors → better recall. Zero runtime cost.
- **Recall side:** stays simple. Cosine scan + enrichment scan + relevance floor + cue surfacing.

### Files Changed
- `servers/brain_recall.py` — STEP 6.5 graph augmentation, enrichment scan
- `servers/brain_remember.py` — enrichment prompt builder, store_enrichments
- `servers/brain_constants.py` — graph augmentation tuning, enrichment types
- `servers/dal.py` — EnrichmentDAL, GraphDAL, TelemetryDAL
- `tests/eval_runner.py` — fixed to use recall_with_embeddings
- `tests/golden_dataset.json` → `tests/golden_dataset_v2.json` (214 cases)
- `tests/test_e2e_enrichment.py` (38 tests)
- `tests/benchmark_*.py` (10 benchmark scripts)
- `docs/encoding-decoding-v2-2026-03-23.md` (complete pipeline doc)
- `docs/CHANGE-DOC-TEMPLATE.md` (documentation standard)
- `scripts/backfill_enrichments.py`

### Benchmark Scripts (for reproduction)
All in `tests/`:
- `benchmark_full_baseline_214.py` — definitive 214-case benchmark
- `benchmark_multivec_encoding.py` — V2/V4/V5 encoding variants
- `benchmark_v15_reranker.py` — cross-encoder rerankers on v1.5
- `benchmark_hyde_tinyllama.py` / `benchmark_hyde_gemma.py` — HyDE tests
- `benchmark_ripple_simulation.py` — ripple engine impact
- `benchmark_ripple_timing.py` — timing analysis
- `benchmark_cues_vs_ripple.py` — cues architecture comparison
- `benchmark_real_conversations.py` — conversation simulation
- `benchmark_claude_quality_ripple.py` — 5-condition ripple benchmark
- `test_contradiction_handling.py` — 8 adversarial scenarios
- `test_safety_mechanisms.py` — 63 safety mechanism tests
