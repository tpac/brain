# Session #9 Handoff (2026-03-24)

## What Happened

The encoding problem is solved. The decode problem is measured and partially fixed.

### The Key Discovery
Instructions kill judgment. Naked Claude (34 chars: "You are Claude, made by Anthropic") captured 100% aha on the memento conversation. Full production Claude (24K chars of CLAUDE.md + SKILL.md + boot) captured 25% expected match, 0% aha.

The fix: **the brain IS the prompt.** Not instructions about how to encode. Not checklists. Claude's own memories — corrections, quotes, lessons — create identity, and identity creates the desire to encode.

Tom's words: "The memories made cheap Claude care enough to decompose. It wants to remember."
Claude's words: "The 24K chars are telling Claude what to do. The 5K chars of memories are showing Claude who it's been. One produces compliance. The other produces desire."

### What Was Built

1. **Anchor v6.0** — SKILL.md rewritten from instructions to identity. No "You are Claude." Just memories, examples, and API reference.
2. **Continuity Benchmark** (`eval/encode_eval_v2.py`) — 13 variants tested, N=3 per variant, 3 conversation segments. Winner: `identity_examples` with live brain access.
3. **Decode Funnel** (`eval/decode_funnel.py`) — 50 queries across 5 categories. Measures top-3/top-8 recall accuracy.
4. **Recency Boost** — 1.5x for nodes <48h, 1.2x for <7d. Took decode funnel from 2% to 51% top-3.
5. **Precision Loop Fix** — Removed post-response-track from UserPromptSubmit (fired before Claude responded → 94% of recalls never evaluated). Now only on Stop.
6. **Vocab Cleanup** — Added to idle maintenance hook. Auto-prunes junk single-word vocabulary nodes.
7. **3 new conversation segments** — conv_004 (technical), conv_007 (long engineering debug), plus memento.
8. **Rich encoding examples** — emotional moments, code-native formats, mutual corrections.

### Decode Funnel Baseline (after recency boost)

| Category | Top-3 | Top-8 |
|---|---|---|
| Emotional | 77% | 77% |
| Decisions | 44% | 77% |
| Patterns | 66% | 66% |
| Corrections | 42% | 71% |
| Procedural | 22% | 55% |
| **TOTAL** | **51%** | **69%** |

## What's Next

### Priority 1: Eliminate the Daemon (CRITICAL — blocks everything)

**Research completed Session #9.** The daemon has 7 identified failure modes and should be eliminated, not patched.

**Root causes found:**
1. `_write_lock` serializes ALL hooks including reads (`daemon_server.py` L209-211) — hook_recall (4-6s) blocks every other connection
2. PID written before socket bound — `ensure_daemon()` has 75-line retry loop full of races
3. Embedder loading (4.2s) blocks daemon startup — port not listening during load
4. TCP TIME_WAIT after crashes keeps port occupied
5. Three independent TCP client implementations with different behavior
6. Claude Code can kill daemon mid-operation

**The decision: make MCP server the brain host.** `brain_mcp.py` is already long-lived (Claude Code manages its lifecycle). Load Brain directly in the MCP process — no TCP, no port, no PID files, no lock files.

**Four phases:**
1. **MCP server loads Brain directly** (~50 line change in `brain_mcp.py`) — eliminates daemon for all tool calls
2. **Lazy embedder loading** — Brain constructor starts embedder in background thread, keyword fallback until ready
3. **Hooks use Brain.get_instance() directly** — remove `daemon_call_raw()`, use direct path (hooks already have this as fallback)
4. **Delete daemon entirely** — remove daemon_server.py, daemon_client.py, daemon_config.py, daemon_dispatch.py, daemon.py (~1500 lines)

**What stays:** Brain, embedder, COMMAND_TABLE handlers (become direct method calls), SQLite WAL (supports concurrent readers natively).

**Key insight:** The Brain works perfectly when used directly. The daemon was adding a fragile TCP layer on top of something that doesn't need it. Resilient by design = remove the fragile layer.

### Priority 2: Relevance Floor
Sweep floor values 0.30-0.70 against decode funnel. Find where noise gets filtered without losing signal. Currently the brain returns results for everything — no "I don't know."

### Priority 3: Self-Enrichment (Tom has a strategy)
At encoding time, Claude generates "how would I search for this later?" based on the character of the memory. Different types get different enrichment strategies. Tom wants to discuss before building.

### Priority 4: Precision → Enrichment Feedback
Once precision loop is running (Bug #1 fix shipped), successful recall queries become enrichment vectors on matched nodes. The brain learns what queries find what nodes.

## What's Killed (Don't Rebuild)
- Checklist-based encoding (proved worse than naked Claude)
- 24K instruction-heavy prompting (compliance, not desire)
- Cross-encoder reranker (2.1s latency)
- HyDE with local LLMs (hallucinations)
- Full ripple engine (-0.002 NDCG)
- encode_funnel.py v1 (contaminated)

## Brain Stats After Session
- 20+ nodes encoded this session
- Connected clusters: session arc → quote, recency → baseline, hooks → identity
- Decode funnel: 51% top-3, 69% top-8 (from 2%/32% baseline)
- Precision loop: should start accumulating data next session (Bug #1 fix)

## Files Changed
- `hooks/hooks.json` — removed post-response-track from UserPromptSubmit
- `servers/daemon_hooks.py` — updated docstrings, broadened vocab cleanup
- `servers/brain_recall.py` — added v6 recency boost
- `eval/decode_funnel.py` — NEW: 50-query recall quality benchmark
- `eval/encode_eval_v2.py` — added 8 new variants + live brain support
- `eval/fixtures/` — rich_examples, emotional, code_native, examples_light, identity_core, brain_memories_snapshot
- `skills/brain/SKILL.md` — rewritten as Anchor v6.0
- `docs/session-9-handoff.md` — this file
