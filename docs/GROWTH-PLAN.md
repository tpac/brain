# Brain Growth Plan — From Crude Recall to Knowledge Graph

**Authors**: Tom Pachys & Claude
**Created**: 2026-03-22
**Status**: Phase A COMPLETE ✅ — Phase B next
**Last updated**: 2026-03-22 (Session #9)
**Companion to**: `ENTITY-GRAPH-DESIGN.md` (Phase C), `SPEC-v4.md` (overall architecture)

---

## The Problem We're Solving

The brain encodes with full context — Claude has the conversation, understands the WHY, sees the connections. But the next Claude decodes with crude means: embedding similarity against whatever the user just typed, returning ~8 nodes ranked by a score that doesn't know if those nodes have ever been useful before.

It's like writing a detailed journal and having someone else read three random sentences through a keyhole.

**What we need:** A brain that learns from every recall whether it helped, feeds that learning back into how it retrieves, and eventually understands relationships well enough to traverse — not just search.

---

## The Growth Path

Each phase builds on the previous. Order matters.

```
Phase A: Close the precision loop ✅ COMPLETE (2026-03-22)
    ↓ generates data about what works
Phase B: Smarter decode ← NEXT
    ↓ uses Phase A data to validate improvements
Phase C: Entity graph (typed edges, traversal, resolution)
    ↓ Phase A tells us which traversal patterns help
Phase D: Self-tuning (brain learns to recall)
    ↓ all previous phases feed this

Prerequisite (also complete):
  Operator Channel ✅ — Brain talks to Tom via [BRAIN-To-*] tags
```

**Why this order:**
- Phase A without Phase B = we know what's broken but can't fix it
- Phase B without Phase A = we change recall but can't measure if it improved
- Phase C without A+B = we build a knowledge graph blind, can't tell if traversal helps
- Phase D requires all three: measurement (A), mechanism (B), structure (C)

---

## Phase A: Close the Precision Loop ✅ COMPLETE

**Completed:** 2026-03-22 (Session #9, 8 commits)

**Results:**

| Sub-phase | What | Before | After |
|---|---|---|---|
| A.0 | Fix scorer bias | 38% accuracy | **90% accuracy** |
| A.1 | Table-driven lifecycle | 62% eval rate | **100% eval rate** |
| A.2 | Explicit feedback | Never called | **Wired via operator channel** |
| A.3 | Confidence loop | Disconnected | **Precision → node confidence (lr=0.05)** |

**Key architectural decisions:**
- Table-driven lifecycle: recall_log table IS the state, no config keys for handoff between hooks
- All recall_log SQL moved to `dal.py` (LogsDAL) — clean layer separation
- `ask_operator` signal replaces `uncertain` — brain asks Tom instead of guessing
- Operator section goes FIRST in hook output — survives timeout truncation
- Benchmark-first: every change validated against real-world conversation corpus before shipping

### What exists today (fully operational)

| Component | File | Status |
|---|---|---|
| `RecallPrecision` class | `servers/brain_precision.py` (651 lines) | ✅ Working |
| Three-layer scorer | `servers/recall_scorer.py` (496 lines) | ✅ Working |
| L0: 30+ regex patterns | `recall_scorer.py:73-177` | ✅ Detecting affirm/redirect/complaint/extension |
| L1: Embedding similarity | `recall_scorer.py:227-286` | ✅ Topic similarity, depth ratio, on-topic % |
| L1b: BART zero-shot | `recall_scorer.py:180-225` | ✅ Agreement/disagreement/redirect/building/topic-change |
| L2: Cross-signal patterns | `recall_scorer.py:379-410` | ✅ Parroting, polite dismiss, combo detection |
| Two-turn evaluation | `daemon_hooks.py:190-202, 349-387` | ⚠️ Works but loses 68% |
| Precision summary | `brain_precision.py:552-650` | ✅ Shown in boot output |
| Explicit feedback API | `brain_precision.py:405-441` | ✅ Built, never called |
| Feedback request | `brain_precision.py:445-523` | ✅ Built, surfaced to Claude only |
| 46+ tests | `tests/test_precision.py` (1092 lines) | ✅ Passing |

### Architecture: Table-Driven Lifecycle (replaces config-slot handoff)

**Design principle:** The `recall_log` table IS the state. No config keys for handoff. Each row has a lifecycle. Hooks query the table for pending work.

**Why this is cleaner:** A new Claude reads one table and understands the flow. No hunting through config keys across hooks. No orphaned state when hooks fail. The table is the single source of truth.

```
RECALL_LOG ROW LIFECYCLE:

  ┌─────────────────────────────────────────────────────────────┐
  │  recall_log row                                             │
  │                                                             │
  │  Stage 1: LOGGED                                            │
  │    Created by: hook_recall → precision.log_recall()         │
  │    State: returned_ids populated, everything else NULL      │
  │    Means: "recall happened, awaiting Claude's response"     │
  │                                                             │
  │  Stage 2: RESPONSE_STORED                                   │
  │    Updated by: hook_post_response_track → evaluate_response()│
  │    State: assistant_response_snippet populated              │
  │    Means: "Claude responded, awaiting user followup"        │
  │                                                             │
  │  Stage 3: EVALUATED                                         │
  │    Updated by: hook_recall (next turn) → evaluate_followup()│
  │    State: followup_signal + precision_score populated       │
  │    Means: "scored, may still receive explicit feedback"     │
  │                                                             │
  │  Stage 4: FEEDBACK_RECEIVED (optional, overrides Stage 3)   │
  │    Updated by: operator via receive_feedback()              │
  │    State: explicit_feedback populated                       │
  │    Means: "ground truth from operator"                      │
  └─────────────────────────────────────────────────────────────┘
```

**How hooks find pending work (NO config keys):**

```sql
-- hook_post_response_track: find recalls awaiting response
SELECT id FROM recall_log
WHERE session_id = ? AND assistant_response_snippet IS NULL
  AND returned_count > 0
ORDER BY created_at DESC LIMIT 1;

-- hook_recall: find recalls awaiting followup evaluation
SELECT id, created_at FROM recall_log
WHERE session_id = ? AND assistant_response_snippet IS NOT NULL
  AND followup_signal IS NULL AND explicit_feedback IS NULL
  AND returned_count > 0
ORDER BY created_at DESC LIMIT 5;
```

**Why query-based is better than config-slot:**
- **No orphans:** If a hook fails, the row stays in its current stage. Next hook picks it up.
- **Handles rapid messages:** Multiple pending recalls all get evaluated, not just the last one.
- **Handles compaction:** Session restarts, queries the table, finds pending work.
- **Self-documenting:** `SELECT ... WHERE followup_signal IS NULL` reads like English.
- **Turn distance:** `created_at` lets us weight confidence by how old the recall is.

**Config keys to REMOVE:**
- `last_recall_log_id` — replaced by query: "most recent row with no response"
- `last_evaluated_recall_id` — replaced by query: "rows with response but no followup"

### A.1: Implement table-driven lifecycle

**Layer architecture:** All SQL lives in the DAL (`servers/dal.py`). `brain_precision.py` calls DAL methods, never raw SQL. This is an existing convention — `LogsDAL` claims ownership of `recall_log` (line 27) but currently has zero recall_log methods. We fix this as part of Phase A.

**Files to modify:**

| File | Change | Why |
|---|---|---|
| `servers/dal.py` `LogsDAL` | Add `log_recall()` — INSERT into recall_log | SQL lives in DAL, not precision |
| `servers/dal.py` `LogsDAL` | Add `update_recall_response()` — UPDATE response snippet | SQL lives in DAL |
| `servers/dal.py` `LogsDAL` | Add `update_recall_evaluation()` — UPDATE followup/precision | SQL lives in DAL |
| `servers/dal.py` `LogsDAL` | Add `update_recall_feedback()` — UPDATE explicit feedback | SQL lives in DAL |
| `servers/dal.py` `LogsDAL` | Add `get_pending_response(session_id)` — query: rows with no response | Table-driven lookup |
| `servers/dal.py` `LogsDAL` | Add `get_pending_followups(session_id, limit)` — query: rows awaiting eval | Table-driven lookup |
| `servers/dal.py` `LogsDAL` | Add `get_recall_row(recall_log_id)` — fetch row for evaluation | Single read method |
| `servers/brain_precision.py` | Refactor: replace all `self.logs_conn.execute()` with DAL calls | Clean layering |
| `servers/brain_precision.py` | Constructor takes `LogsDAL` instead of raw `logs_conn` | Dependency injection |
| `servers/daemon_hooks.py:191-202` | Replace config read with `dal.get_pending_followups()` | Evaluate ALL pending, not just last |
| `servers/daemon_hooks.py:360-370` | Replace config read/write with `dal.get_pending_response()` | No config keys, table is truth |
| `servers/daemon_hooks.py:257` | Remove `set_config("last_recall_log_id", ...)` | Table row IS the state |
| `servers/daemon_hooks.py:368` | Remove `set_config("last_evaluated_recall_id", ...)` | Table row IS the state |

**What goes where (layer boundaries):**
```
dal.py (LogsDAL)          — SQL queries, column names, table structure
brain_precision.py        — Precision LOGIC: scoring, lifecycle rules, what to evaluate
daemon_hooks.py           — WIRING: when to call what, hook event → precision method
brain_voice.py            — FORMATTING: how precision results appear in output
recall_scorer.py          — COMPUTATION: regex/embeddings/BART signal scoring (pure functions)
```

**Turn distance weighting:**
When evaluating a followup from N turns later, weight confidence:
```python
turn_age = (now - recall_created_at).total_seconds() / 60  # minutes
distance_factor = max(0.3, 1.0 - (turn_age / 30))  # 30 min = 0.3x, 0 min = 1.0x
adjusted_confidence = confidence * distance_factor
```

**Target:** Evaluation rate from 32% → 80%+

### BEFORE ANY CODE CHANGES: Test harness with real conversations

**⚠️ SACRED RULE: embed, encode, recall are sacred to the brain. No changes to recall pipeline without comprehensive benchmarks FIRST.**

**Step 1: Build conversation test corpus**

Collect real multi-turn conversations from the internet that represent diverse use cases:

| Category | Source | Why |
|---|---|---|
| Engineering | GitHub issues, Stack Overflow threads | Technical Q&A, debugging, code discussion |
| Philosophy | Reddit r/philosophy threads | Abstract reasoning, position changes, Socratic dialogue |
| Science | ArXiv discussion, research Q&A | Hypothesis-evidence patterns, data-driven conclusions |
| Personal/Creative | Writing workshops, advice threads | Emotional context, preference expression |
| Adversarial | Debates, disagreements | Redirect patterns, topic changes, corrections |

**Step 2: Extract turn pairs and expected signals**

For each conversation, annotate:
```json
{
  "conversation_id": "gh_issue_1234",
  "category": "engineering",
  "turns": [
    {
      "user": "The auth flow breaks on mobile Safari",
      "simulated_recall": ["Clerk auth decision", "Mobile browser constraints"],
      "assistant": "Let me check the Clerk config for Safari...",
      "followup": "That fixed it, the redirect URL was wrong",
      "expected_signal": "positive",
      "expected_confidence_range": [0.5, 1.0]
    },
    {
      "user": "Actually never mind, it was a CORS issue",
      "simulated_recall": ["Clerk auth decision"],
      "assistant": "Let me look at the CORS headers...",
      "followup": "No, forget auth entirely. Let's work on the dashboard",
      "expected_signal": "negative",
      "expected_confidence_range": [0.4, 1.0]
    }
  ]
}
```

**Step 3: Run scoring benchmark**

Test the THREE-LAYER scorer against the corpus:
- L0 (regex) accuracy by category
- L1 (embeddings) accuracy by category
- L1b (BART) accuracy by category
- Combined score accuracy
- False positive rate (says positive when actually negative)
- False negative rate (says negative when actually positive)

**Step 4: Run lifecycle benchmark**

Simulate the full hook chain with the table-driven lifecycle:
- Create recall_log rows → store responses → evaluate followups
- Verify evaluation rate hits 80%+ on the corpus
- Verify turn-distance weighting doesn't degrade accuracy
- Verify rapid messages (3 recalls in 1 minute) all get evaluated
- Verify session boundary doesn't orphan rows

**Step 5: Compare old vs new**

Run the SAME corpus through both:
1. Current config-slot handoff
2. New table-driven lifecycle

Compare: evaluation rates, precision scores, orphaned recalls.

**Only after benchmarks pass do we modify production code.**

**Files to create:**
- `tests/corpus/` — Directory for conversation test data
- `tests/bench_precision_corpus.py` — Scorer accuracy benchmark
- `tests/bench_precision_lifecycle.py` — Lifecycle simulation benchmark
- `tests/golden_precision.json` — Golden test cases for precision scoring (like golden_dataset.json for recall)

### A.2: Explicit feedback through operator channel

**Problem:** `receive_feedback()` exists but is never called. Tom has never been able to tell the brain "that recall was garbage."

**Why this matters:** Explicit feedback is the STRONGEST signal (overrides auto-computed scores). Even 5% of recalls getting explicit feedback would calibrate the entire scoring system.

**Implementation:** Use the operator channel we just built (2026-03-22). When precision is uncertain about a recall, surface a feedback request in `[BRAIN-To-Tom]`:

```
[BRAIN-To-Tom]
@priority: low
@present: aside
🎯 Brain recalled 5 nodes for your last question. Useful? (say "useful", "not useful", or ignore)
[/BRAIN-To-Tom]
```

**Files to modify:**
- `servers/brain_voice.py` `render_operator_prompt()` — Add precision feedback as a low-priority operator item
- `servers/daemon_hooks.py` `hook_recall()` — Already calls `request_feedback()` and passes to render_prompt. Wire it to operator channel instead.
- `servers/daemon_hooks.py` `hook_post_response_track()` — Parse Tom's response for feedback keywords ("useful", "not useful", "garbage", "helpful")

**Files that already exist for this:**
- `brain_precision.py:445-523` `request_feedback()` — Decides WHEN to ask (cooldown, conditions)
- `brain_precision.py:405-441` `receive_feedback()` — Processes the answer
- Cooldown: `_FEEDBACK_COOLDOWN = 5` recalls between requests (not annoying)

### A.3: Close the loop — precision → node confidence

**Problem:** A node recalled 20 times with precision score 0.1 every time keeps getting recalled at the same rank. The brain doesn't learn from its own measurements.

**Current confidence flow (precision is NOT connected):**
```
Node confidence is managed by:
  - brain_engineering.py: record_divergence() lowers, learn_impact() boosts
  - brain_evolution.py: confirm/dismiss evolution adjusts
  - brain_remember.py: remember_uncertainty() lowers, remember_lesson() boosts
  - brain_surface.py: stage()/promote() sets to 0.7-0.8

Precision measures recall quality but NEVER writes to node confidence.
```

**The missing link:** After `evaluate_followup()` or `receive_feedback()`, update the confidence of every node in that recall based on the precision score.

**Algorithm:**
```python
def update_node_confidence_from_precision(brain, recall_log_id, precision_score):
    """Nudge node confidence toward observed utility."""
    returned_ids = get_returned_ids(recall_log_id)
    for node_id in returned_ids:
        current = get_node_confidence(node_id)
        # Small nudge toward observed precision (Hebbian: what works, strengthen)
        # LEARNING_RATE keeps it gradual — one bad recall doesn't tank a good node
        delta = (precision_score - current) * LEARNING_RATE  # e.g., 0.05
        new_confidence = clamp(current + delta, 0.1, 1.0)
        update_node_confidence(node_id, new_confidence)
```

**Why small LEARNING_RATE:** A single recall might be bad because the QUERY was bad, not the NODE. Over many recalls, consistently unhelpful nodes will drift down. Consistently helpful ones drift up. The signal emerges from volume.

**Files to modify:**
- `servers/brain_precision.py` — Add `update_node_confidence()` method
- `servers/brain_precision.py` `evaluate_followup()` — Call confidence update after scoring
- `servers/brain_precision.py` `receive_feedback()` — Call confidence update after feedback
- `servers/brain.py` or `brain_engineering.py` — Expose `set_node_confidence(node_id, value)` if not already available

**Where this hooks into the new lifecycle:**
```
Stage 3 (EVALUATED) or Stage 4 (FEEDBACK_RECEIVED):
  → precision_score is now set
  → call update_node_confidence() for all returned_ids
  → confidence nudge happens ONCE per recall, at evaluation time
```

**Leads to investigate:**
- `brain_recall.py:652` reads `node_confidence` during scoring — how much does it weight confidence today? This determines how fast the feedback loop affects future recalls.
- `brain_engineering.py:919-1045` `recalibrate_confidence()` — runs at session boundary. Does it conflict with precision-driven updates? Need to ensure they compose, not fight. Investigate: does recalibrate reset what precision learned?
- Should locked nodes have their confidence updated? Locked = permanent, but confidence = how useful in recall. A locked rule that never helps recall should still have lower confidence. **Decision needed from Tom.**
- What's the interaction with Hebbian edge strengthening? If a node has low confidence but high edge weight, which wins in recall scoring?

**Tests:**
- Node with 10 positive recalls → confidence drifts up
- Node with 10 negative recalls → confidence drifts down
- Single bad recall doesn't tank a 0.9 confidence node
- Locked nodes participate (confidence ≠ locked status)
- Golden dataset before/after: NDCG should not drop

### A.4: Surface precision to operator

**Problem:** Precision stats appear in boot output (Claude sees them) but Tom never does. With the operator channel, we can now surface:

```
[BRAIN-To-Tom]
@priority: low
📊 Recall precision (24h): 73 recalls, 61 evaluated (84%) — avg 0.72
   Top performer: "Glo onboarding rules" (0.94 avg, recalled 8x)
   Struggling: "v14 architecture decisions" (0.31 avg, recalled 5x)
[/BRAIN-To-Tom]
```

**Why this matters:** Tom can see which areas of knowledge are working and which aren't. He can decide to enrich struggling nodes or investigate why good nodes score well.

**Files to modify:**
- `servers/brain_voice.py` `render_operator_prompt()` — Add precision summary as low-priority item
- `servers/brain_precision.py` — Add `get_top_and_bottom_nodes(hours=24)` method that identifies best/worst performing nodes

---

## Phase B: Smarter Decode ← NEXT

**Depends on:** Phase A ✅ (precision data now flowing, confidence loop closed)

**Goal:** Recall returns better nodes by using graph structure, confidence, and vocabulary. Validate every change against precision benchmarks.

**Approach:** Testing-first. Build benchmark corpus for each sub-phase. Measure golden dataset NDCG before and after. Sacred system rules apply — embed, encode, recall are sacred.

### B.vocab: Wire vocabulary encoding (do first — quick win)

**Problem:** `learn_vocabulary()` method exists, only 6 vocabulary nodes (manually created in Session #9). The automated gap detector runs but catches garbage (XML fragments, UUIDs). SKILL.md Step 4 tells Claude to encode vocabulary manually, but the automated pipeline should also work.

**Fix:**
1. Clean up gap detector — filter through common-word list (`text_processing.py:filter_domain_terms()`, already built)
2. Wire `learn_vocabulary()` into encoding pipeline (idle hook or post-response)
3. Connect new vocab nodes to concept/project nodes they relate to
4. Verify SKILL.md Step 4 alignment — automated encoding matches manual quality

**Files:**
- `servers/daemon_hooks.py:594-685` — vocab extraction strategies (7 strategies, need common-word filter)
- `servers/brain_engineering.py` `learn_vocabulary()` — exists, needs to be called automatically
- `servers/text_processing.py` `filter_domain_terms()` — exists, needs to be wired into gap detection
- `skills/brain/SKILL.md` — Step 4 defines vocabulary encoding quality expectations

### B.1: Confidence-weighted recall

**Why now:** Phase A's confidence loop means node confidence is now a live signal — nodes that consistently help recall have higher confidence. Use it.

When scoring recall results, weight by node confidence:
```
effective_score = embedding_similarity * 0.7 + node_confidence * 0.3
```

**File:** `servers/brain_recall.py:652` — Already reads confidence. Needs to weight it.

**Leads to investigate:**
- Current scoring formula in `recall_with_embeddings()` — what weight does confidence have today?
- How does `recalibrate_confidence()` (session boundary) interact with precision-driven updates?
- Run golden dataset before/after to verify NDCG improvement

### B.2: Graph-augmented recall (1-hop typed neighbors)

**Why now:** 82% of edges are auto-generated noise (co_accessed, bridges). The 17% intentional edges (depends_on, part_of, implements) carry real meaning. Use them.

When embedding search returns node X, also pull X's typed neighbors:
```
Embedding finds: "Auth decision"
1-hop typed: → depends_on → "Login screen"
             → decided_by → "Tom: use magic links"
```

**Files:**
- `servers/brain_recall.py` — After embedding results, run 1-hop traversal
- `brain_surface.py` `suggest()` — already does 1-hop traversal for locked nodes. Reuse pattern.

**Edge types to traverse** (intentional only):
`related`, `about`, `part_of`, `depends_on`, `implements`, `contains`, `enables`, `constrains`, `governs`, `extends`, `describes`, `corrected_by`, `produced`, `addresses`, `elaborates`, `informed_by`

**Edge types to SKIP** (auto-generated noise):
`co_accessed`, `emergent_bridge`, `dreamed_from`, `cluster_observation`, `dream_observation`

### B.3: Vocabulary-augmented recall

Wire vocabulary nodes into query expansion (mechanism already exists in `daemon_hooks.py:204-230` but finds nothing with 0 vocab nodes — B.vocab fixes this).

### B.4: Surface precision stats to operator (A.4 deferred here)

Surface top/bottom performing nodes through operator channel so Tom sees which knowledge areas work and which don't. Add `get_top_and_bottom_nodes(hours=24)` to brain_precision.

### Testing strategy for Phase B

1. Run golden dataset (`tests/eval_runner.py`) BEFORE any change → capture baseline NDCG
2. For each sub-phase, run golden dataset AFTER → verify no regression
3. Run precision corpus (`tests/bench_precision_corpus.py`) → verify scorer still 90%+
4. Run lifecycle benchmark (`tests/bench_precision_lifecycle.py`) → verify still 100%
5. If NDCG drops on any change, revert and investigate

---

## Phase C: Entity Graph

**Depends on:** Phase A (measurement), Phase B (graph-augmented recall working)
**Full design:** See `ENTITY-GRAPH-DESIGN.md`

### Why after A+B:
- Phase A gives us precision data → we can measure if entity traversal helps
- Phase B gives us 1-hop traversal → entity graph extends this to typed traversal
- We'll have months of data showing which edge types (the few typed ones we have) produce better recalls

### C.1: Entity extraction (spaCy)
### C.2: Typed edges + relationship classification
### C.3: Graph traversal in recall
### C.4: Full integration (entity-aware consciousness, timeline, dreaming)

See `ENTITY-GRAPH-DESIGN.md` for full design.

---

## Phase D: Self-Tuning Brain

**Depends on:** All previous phases

**Goal:** The brain auto-adjusts recall parameters based on precision trends.

### D.1: Pattern detection from precision data
- "Technical queries need deeper graph traversal"
- "Personal queries need recency bias"
- "Glo queries score well, brain-internal queries score poorly"

### D.2: Adaptive recall parameters
- Embedding vs keyword blend ratio adjusts per query type
- Graph traversal depth adjusts per domain
- Confidence threshold adjusts based on precision trends

### D.3: Consciousness reports on its own quality
- "Recall precision dropped 20% this week — investigating"
- "Nodes about Creatify API are never useful in recall — consider archiving"

---

## Regression Protection

**⚠️ SACRED RULE: Anything about embed, encode, recall is sacred to the brain. No changes ship without benchmarks proving no regression.**

**The risk:** Every change to the recall pipeline can silently degrade quality.

### Firewall 1: Golden dataset as gate
- `tests/golden_dataset.json` — 60+ test cases with expected results
- `tests/eval_runner.py` — Computes NDCG, MRR, precision@k, recall@k
- **Rule:** Run golden eval BEFORE and AFTER every change. If NDCG drops, the change doesn't ship.

### Firewall 2: Precision as continuous monitor
- Boot output already shows precision stats
- Operator channel now surfaces them to Tom
- Any sustained precision drop = alarm

### Firewall 3: One change at a time
- Each sub-phase (A.1, A.2, A.3...) is a separate commit
- Each commit has before/after benchmark
- Never merge two recall changes at once

### Firewall 4: Test coverage
- `test_precision.py` — 46+ tests for precision pipeline
- `test_system.py` — End-to-end lifecycle tests
- `test_recall_scorer.py` — Scoring algorithm tests
- `test_recall_quality.py` — Recall quality regression tests
- **New tests required per sub-phase** (specified in each section)

### Firewall 5: Brain encodes its own regressions
- If precision drops after a change, brain encodes a lesson about what happened
- Future Claude recalls this when touching the same code
- The meta-lesson applies: encode the PRACTICE, not just the incident

---

## Current Brain Stats (2026-03-22, post-Phase A)

| Metric | Value | Health |
|---|---|---|
| Total nodes | ~910 (42 Creatify archived) | ✅ Cleaner |
| Locked nodes | ~670 | ✅ |
| Total edges | ~12,500 | ⚠️ 82% auto-generated |
| Intentional edges | ~2,270 (17%) | ⚠️ Phase B.2 will leverage these |
| Vocabulary nodes | 6 (first ever, Session #9) | ⚠️ Phase B.vocab will grow this |
| Recalls (all time) | 1,896 | ✅ |
| Scorer accuracy | **90%** (was 38%) | ✅ Phase A.0 |
| Evaluation rate | **100%** (was 62%) | ✅ Phase A.1 |
| Explicit feedback | **Wired** (was never called) | ✅ Phase A.2 |
| Confidence loop | **Closed** (was disconnected) | ✅ Phase A.3 |
| Operator channel | **Live** (first conversation!) | ✅ |
| BART available | ✅ | All 3 scoring layers active |
| Golden dataset | 60+ recall cases | ✅ |
| Precision corpus | 17 convos, 21 turns | ✅ |
| Benchmarks | 3 scripts (scorer, lifecycle, operator) | ✅ |

---

## Open Questions

1. **Confidence LEARNING_RATE:** 0.05 is a guess. Too high = noisy, too low = never learns. Should we start at 0.05 and let Phase D auto-tune it?

2. **Locked node confidence:** Should locked nodes have their confidence updated by precision? Argument for: a locked rule that's never useful in recall should rank lower. Argument against: locked means "definitely true" not "definitely useful in recall."

3. **Vocabulary encoding trigger:** Should vocab be encoded during idle (async, no latency impact) or during post-response (synchronous, fresher context)?

4. **Golden dataset expansion:** Current 60+ cases cover keyword and embedding recall. Need cases for graph-augmented recall (Phase B) and entity traversal (Phase C). When to create them?

5. **Precision feedback UX:** How should Tom give feedback? Dedicated keywords ("useful", "garbage")? A rating? Or infer from conversation flow?
