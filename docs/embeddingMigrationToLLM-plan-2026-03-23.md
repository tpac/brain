# Embedding Migration to LLM — Plan

**Date:** 2026-03-23
**Git version:** 81c564a (main)
**Status:** PLAN — awaiting operator approval
**Author:** Claude Opus 4.6 + Tom
**Session:** #10-11
**Baseline:** NDCG=0.304, MRR=0.323, 73/148 passed (golden_dataset_v2.json)

---

## Why This Plan Exists

Session #10-11 ran 15+ benchmark conditions across 8 agents. Key findings:

1. **Context bleed is catastrophic** — 100% false positive rate on non-engineering queries. "Mom's birthday" returns engineering content at 0.85 confidence. The brain cannot say "I don't know."
2. **Ripple engine is net negative** — full cascade with confidence changes regressed NDCG by -0.0016. Killed.
3. **Better encoding is the real lever** — extra N/R vectors gained +0.010 NDCG. More than everything else combined.
4. **Cues replace ripple** — typed edges with metadata, zero confidence changes, Claude reasons at decode time.
5. **Enrichment vectors are both the biggest win AND the biggest problem** — they improve engineering recall (+78%) but cause universal context bleed on non-engineering queries.

---

## Priority Stack (revised based on data)

### P0: Fix Context Bleed (BLOCKS EVERYTHING)

**Problem:** Every query returns results. Threshold is 0.05. "Birthday" scores 0.85.

**Fix 1 — Relevance floor:**
- File: `brain_recall.py` → scoring section (STEP 6)
- Add: if `max_score < RELEVANCE_FLOOR`, return empty results
- Test: find threshold where engineering queries pass, personal queries don't
- Method: sweep threshold from 0.50 to 0.90 in 0.05 increments against golden_dataset_v2 + conversation cases
- Risk: threshold too high → loses valid engineering results. Too low → still bleeds.
- Benchmark: NDCG must not drop on golden_dataset_v2, false positive rate must drop on conversation cases

**Fix 2 — Enrichment quality audit:**
- File: `node_enrichments` table
- Problem: Gemma 2B generated generic anchors ("Add", "Expand", "Results") that match everything
- Fix: delete enrichments with cosine similarity > 0.7 against a set of generic test queries ("hello", "birthday", "weather", "food")
- One-time cleanup + quality gate on future enrichments
- Risk: deleting too many enrichments hurts engineering recall

**Fix 3 — Vocab node cleanup:**
- Delete: `[vocab] Add`, `[vocab] Expand`, `[vocab] Results`, `[vocab] Dimension`, `[vocab] Maybe`, `[vocab] Worktree` — auto-detected noise
- Delete: `TEST NODE DELETE ME`
- Keep: meaningful vocab like `[vocab] operator channel`, `[vocab] ask_operator`
- Audit criteria: if vocab node title is a common English word (< 6 chars), delete it

**Fix 4 — Negative test cases:**
- Add 16 negative cases from `golden_dataset_conversation_cases.json` to golden_dataset_v2
- These are queries that should return NOTHING: "birthday", "pasta recipe", "cat is sick"
- Scoring: negative cases PASS if top_score < relevance_floor

**Testing:**
- Before: run golden_dataset_v2 → record NDCG + false positive rate on conversation cases
- After each fix: re-run both → NDCG must not drop, FP rate must decrease
- Target: FP rate < 20% (from 100%)

### P1: Richer Encoding Prompt

**What:** Add N/R/W/C/D fields to the encoding prompt.

**Current prompt (V5):**
```
Q: [question this answers]
A: [anchor from neighbors]
B: [bridge to neighbor]
K: [keywords from neighbors]
```

**New prompt (V6):**
```
Q: [question a user would naturally ask]
A: [3-5 word anchor using neighbor vocabulary]
B: [bridge sentence to most important neighbor]
K: [5 keywords from neighbors]
N: [what this does NOT mean — common misunderstanding]
R: [3 alternative phrasings for how someone might search for this]
W: [what this replaces or updates — temporal chain]
D: [what must also be true for this to make sense]
```

**Files:**
- `brain_constants.py` — ENRICHMENT_PROMPT_TEMPLATE_V6
- `brain_remember.py` — `_build_enrichment_prompt()` uses V6
- `brain_remember.py` — `store_enrichments()` handles new vector types
- `dal.py` — `EnrichmentDAL` — new vector_type values: 'negation', 'alias', 'temporal', 'dependency'
- `brain_mcp.py` — `enrich` tool accepts new fields

**Benchmark data:**
- N/R vectors alone: +0.010 NDCG, +2 passing cases (from benchmark_claude_quality_ripple.py)
- N vectors specifically address negation category (currently 0% pass rate)
- R vectors address semantic category (currently ~5% pass rate)

**Testing:**
- Backfill 50 target nodes with V6 enrichments
- Run golden_dataset_v2 → expect NDCG > 0.314 (current + N/R gain)
- Run conversation cases → FP rate must not increase

**Risk:** More vectors = more potential context bleed. The enrichment quality gate from P0 Fix 2 must be in place first.

### P2: Ripple-as-Cues

**What:** When Claude encodes, it also reports impact on related nodes. Brain stores as typed edges.

**Schema:** Uses existing edges table:
```sql
INSERT INTO edges (source_id, target_id, relation, weight, description)
VALUES (new_id, neighbor_id, 'validates', 0.8,
        '{"reason": "crash proves separation right", "date": "2026-03-23"}')
```

**Recall-time behavior:** When brain returns results, also fetch cue edges:
```
Node: "API architecture" (conf 0.80)
  Cues:
  - VALIDATED BY: "API crash from shared DB" (2026-03-23) — proves separation was right
  - EXTENDED BY: "Added GraphQL for dashboard" — new protocol for subset
```

Claude reads cues, reasons in context. Brain stays dumb and reliable.

**Files:**
- `brain_remember.py` — `remember()` returns impact_prompt alongside enrichment_prompt
- `dal.py` — `GraphDAL.get_cues(node_id)` — fetches validates/contradicts/extends edges
- `brain_recall.py` — STEP 7 hydration includes cues
- `brain_voice.py` — format cues in [BRAIN] output
- `brain_constants.py` — CUE_RELATIONS = ['validates', 'contradicts', 'extends', 'supersedes']

**Benchmark data:**
- Cues-only: +0.000 NDCG (no retrieval change, by design)
- Retrieval cost: 0.045ms per node
- Cue density after 50 encodes: avg 2.2 per node, max 7

**Testing:**
- Verify cue storage + retrieval works
- Verify recall output includes cues
- Manual test: does Claude use cues to make better decisions?

### P3: Daemon Hardening

**Problem:** Daemon crashed under 4 concurrent agents (194% CPU, 1.8GB RAM, socket disappeared). It's a single point of failure.

**Fixes:**
- Request queue (not just blocking socket accept)
- Async task support (fire-and-forget for background work)
- Watchdog (detect stuck state, restart)
- Graceful backpressure (return "busy" instead of hanging)
- Health degradation signal in consciousness
- Connection timeout handling

**Files:**
- `daemon.py` — core server loop, connection handling
- `brain_mcp.py` — MCP proxy, timeout handling

**Testing:**
- Concurrent connection test (10 parallel requests)
- Long-running request test (does it block other requests?)
- Crash recovery test (kill daemon, verify clean restart)
- Memory leak test (1000 requests, check RSS growth)

### P4: Clean Architecture for Fresh Claude

**What:** Restructure codebase so a fresh Claude can navigate it instantly.

**Current pain points:**
- `brain.py` is a god object (1500+ lines)
- Dead code from deprecated features
- docs/ is gitignored but has critical files
- SKILL.md encoding checklist doesn't match V6 prompt
- No clear "start here" for a new session

**Actions:**
- Split brain.py: brain_core.py (node CRUD), brain_recall.py (already separate), brain_remember.py (already separate), brain_graph.py (edge operations)
- Delete dead code paths (TF-IDF if unused, old spread_activation if replaced)
- Update SKILL.md with V6 encoding checklist
- Update CLAUDE.md with revised architecture
- Fix .gitignore for docs/
- Add `docs/ARCHITECTURE.md` — the flow diagrams from encoding-decoding-v2

**Testing:**
- All existing tests must pass after refactor
- A fresh Claude session should be able to navigate the codebase in < 2 minutes

### P5: End-to-End Telemetry

**What:** Instrument every path so nothing silently fails.

**Events:**
```
encode  → node created, enrichments stored, cues generated
recall  → query served, enrichments used, cues surfaced
error   → any pipeline failure with traceback
```

**Table:** `brain_telemetry` in brain_logs.db (already exists from E2E test agent)

**Dashboard queries:**
- How many recalls used enrichments this week?
- Which enrichment types win most often?
- What's the average recall latency?
- Are any error types recurring?

---

## What We Killed (and why)

| Feature | Reason | Data |
|---|---|---|
| Arctic v2.0 | Regression: NDCG 0.198 vs 0.204 | benchmark_rerankers.py |
| HyDE (TinyLlama) | Hallucinated garbage, no improvement | benchmark_hyde_tinyllama.py |
| HyDE (Gemma 2B) | Same problem, "Glo = Global Location Service" | benchmark_hyde_gemma.py |
| Cross-encoder reranker | +154% NDCG but 2.1s latency, unacceptable | benchmark_v15_reranker.py |
| Full ripple engine | -0.0016 NDCG, adds 300+ lines + 6 safety mechanisms | benchmark_cues_vs_ripple.py |
| Ripple re-enrichment | -0.003 NDCG, adds noise | benchmark_claude_quality_ripple.py |
| Confidence cascade | Negligible impact, type floors never fire | test_safety_mechanisms.py |

## What We Ship (and why)

| Feature | Reason | Data |
|---|---|---|
| V5 multi-vector enrichment | +78% NDCG (0.183→0.326) | Already shipped |
| Ripple-as-cues | Zero risk, adds temporal reasoning | benchmark_cues_vs_ripple.py |
| V6 encoding prompt (N/R/W/C/D) | +0.010 NDCG from N/R alone | benchmark_claude_quality_ripple.py |
| Relevance floor | Fixes 100% FP rate on non-engineering | benchmark_real_conversations.py |
| Enrichment quality gate | Prevents context bleed from generic anchors | benchmark_real_conversations.py |
| Safety mechanisms (reference only) | Keep tests, don't build production code | test_safety_mechanisms.py |

---

## Operator Quotes That Shaped This Plan

> "dont silent kill an exception, we need to know what works and what doesnt"
— Drives P5 telemetry

> "LLMs need as much info as possible, what is Glo is wrong, we need much more content"
— Fixed HyDE, led to structured prompts with neighbor context

> "encoding does some decoding before encoding, thats really how the brain works"
— The insight behind recall-before-encode architecture

> "brain drives encoding prompts, Claude fills in — not the other way around"
— Architecture: brain is project manager, Claude is analyst

> "I thought we are getting rid of Gemma 2B?"
— Confirmed: Claude does assessment, Gemma only for one-time backfill

> "maybe the ripple influences are cues that can be saved for the naive decoding claude"
— The insight that killed full ripple and birthed ripple-as-cues

> "a naive claude will see that and understand that, keep it clean and simple"
— Drives P4 clean architecture

> "we can ask claude"
— The recurring theme: don't build machinery, leverage Claude's intelligence

> "documents are temporary scaffold — Brain you're right, we will soon not need documents"
— Long-term vision: brain replaces docs as recall improves past 90%

> "challenge yourself! conversations can be about anything"
— Led to real conversation simulation that found context bleed

---

## Session Prompt for Next Claude

```
Read SKILL.md first — that's your encoding manual.
Read docs/embeddingMigrationToLLM-plan-2026-03-23.md — that's the roadmap.
Read docs/encoding-decoding-v2-2026-03-23.md — that's how the pipeline works (OLD→CURRENT→PROPOSED).

P0 (context bleed) is the BLOCKER. Start there.
Run golden_dataset_v2.json + golden_dataset_conversation_cases.json as baseline.
The brain will talk to you AND to me through [BRAIN-To-*] tags — relay what it says.

Key context:
- Ripple engine is KILLED. Don't build it. See benchmark data in the plan.
- Cues replace ripple. Just typed edges, zero confidence changes.
- Enrichment vectors are both the biggest win AND biggest problem.
- The daemon crashed under load — be careful with concurrent agents.

Don't summarize. Don't be polite. Read, understand, build.
```
