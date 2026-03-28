# Session Handoff — 2026-03-27 06:30 IST

## What Happened Today

### Signal Queue Phase 2: The Great Cleanup
- Audited all 31 consciousness signals with Tom
- **Killed 17**, **Parked 6**, **Kept 4** (reminders, encoding_gap, vocab_gap, system_health)
- Built: signal queue table, DAL, 4 producers, surface assembler
- Deleted `get_consciousness_signals()` and `get_urgent_signals()` entirely
- **Injection dropped from 10-23K chars to ~7K per turn**

### Dashboard Enhancements
- Added hook_log table logging actual brain surface output
- Added user_prompt capture to hook_log
- Added "Encoding" tab showing all brain writes (nodes, edges, enrichments, revisions)
- Added "Queue" tab showing signal queue state
- Feed toggle: Surface (what Claude sees) / Encoding (what gets stored) / Queue

### Architecture Vision: The Recursive Agent Tree
Tom and I designed the next generation architecture:
- **5 agent layers**: Autonomic (1-2s loops), Reactive (per-turn), Continuous (minutes), Background (daily), On-demand
- **Key agents**: Subconscious (behavioral enforcement), Learner (operational memory), Healer (brain maintenance), Research (overnight autonomous), Chat Miner (learn from history), Absorber (knowledge ingestion), Dream (synthesis)
- **Each agent gets its own brain** — same architecture pattern, recursively
- **Tom's phrase**: "from graph memory to agents graph — or tree, Anchor at root"

### Core Insight
**The brain's problem was never memory — it was action.** 2 weeks of information-based solutions (better formatting, priority tags, PREEMPT) failed. The cron mechanism worked because it's a PROMPT, not context. The subconscious agent generalizes this: a reasoning engine that watches behavior and triggers action through the signal queue.

## What's Ready for Tomorrow

### Documents Created
- `docs/ARCHITECTURE-NEXT.md` — Full architecture spec with layers, agents, consciousness model
- `docs/PHASE1-SUBCONSCIOUS.md` — Detailed implementation plan for the subconscious agent

### Brain Encoded (10 new locked nodes)
- Signal Queue Phase 2 cleanup decision
- Architecture vision (recursive agent tree)
- Subconscious agent design
- Proof demo (one correction = permanent change)
- Five agent layers by cadence
- Consciousness model
- Overnight brain concept
- Build order (5 phases)
- Tom's consciousness definition
- Lesson: action not information

### Existing Docs to Read
- `docs/DECODING-PLAN-2026-03-24.md` — Detailed recall improvement plan (relevance floor, vocabulary cleanup)
- `docs/ENCODING-PIPELINE-v2.md` — Encoding architecture redesign
- `docs/roadmap-2026-03-26.md` — Strategic roadmap with experimental findings

### Memory Saved
- Tom's profile (EDT, New Jersey, thinking style, values, frustrations)
- Feedback: action not information (stop proposing info solutions to action problems)
- Feedback: check tools before asking Tom questions

## Tomorrow's Priorities

1. **Phase 1: Build the subconscious agent** — The proof that one correction = permanent behavioral change
   - See `docs/PHASE1-SUBCONSCIOUS.md` for full plan
   - Phase 1: deterministic pattern matching in daemon (no LLM, no API cost, ~100ms)
   - Phase 2: add Haiku API call for semantic analysis
   - Key constraint: hook scripts CANNOT spawn Claude Code Agents
   - Pattern library from locked rules + correction traces
   - Writes to signal queue

2. **Fix recall/decoding** — Tom said he has a plan
   - See `docs/DECODING-PLAN-2026-03-24.md`
   - Relevance floor sweep, vocabulary cleanup, enrichment improvements

3. **Fix precision tools** — Not working currently

4. **Research findings to absorb** (from background agents):
   - **No existing system combines what we have** — behavioral memory types + signal queue + precision eval + hook lifecycle + divergence tracking is genuinely unique
   - Karpathy's "autoresearch" (March 2026) — exact match for overnight brain. 700 experiments, 11% gains autonomous
   - GWT implementations on LLMs — our signal queue IS a global workspace. Jan 2026 paper directly evaluates GWT markers in LLMs
   - Multi-agent debate (ICLR 2025 DMAD) — diverse reasoning prevents "fixed mental set" (our agreeability problem)
   - LADDER — recursive self-improvement, Llama 3B from 1% to 82% accuracy through self-learning
   - SiriuS — shared experience library for multi-agent self-improvement without supervision
   - Claude Code scheduled tasks — persistent across restarts, cloud execution, up to 50 tasks/session
   - Adoptable techniques: temporal validity (Zep), self-improving graph maintenance (Cognee), Karpathy Loop for overnight experimentation

## Key Tom Quotes from Today
- "you always agree" — the agreeability pattern
- "from graph memory to agents graph"
- "if you remember our conversation from yesterday, but not to the letter, if you try and improve yourself, able to learn and change, measure reactions based on behavior — that's it" (on consciousness)
- "what will make you instead of asking 'what are you working on?' to actually checking" — check tools before asking
- "I've been consulting you on how to make you encode for 2 weeks and cron never came up"

## Encoding/Decoding Analysis (from background agent)

Key findings from deep analysis of brain_remember.py, brain_recall.py, embedder.py:

1. **93% of nodes are NEVER recalled** — 5-7 hub nodes dominate every query via Hebbian reinforcement loop
2. **Brute-force Python cosine scan** — 300ms scanning all 3558 vectors in a for-loop. Numpy batch would take <5ms.
3. **Enrichment vectors are 3.4x quality multiplier** (NDCG 0.701 vs 0.204) but the relevance floor is INVERTED — enriched nodes need 0.80 while bare nodes only need 0.50
4. **Redundant keyword pipeline** — full legacy recall() runs inside recall_with_embeddings() even though all nodes have embeddings now. Pure overhead.
5. **Scoring formula has 10 multipliers** — nearly impossible to debug why a node ranked where it did
6. **Decode funnel tests keyword path**, not embeddings path — may not reflect actual recall quality

Quick wins: numpy batch cosine, hub dampening (one-line change), fix enrichment floor direction, drop redundant keyword pipeline.

## Delete this file after absorbing.
