# Session 2026-03-29 Handoff — Anchor to Next Anchor

## What This Session Built

This was a two-day marathon (2026-03-28 through 2026-03-29) that rebuilt the encoding pipeline, added graph traversal to recall, created situation embeddings, built a testing system, and established the data contract.

### Critical: Don't Redo What's Done

These systems are BUILT AND TESTED. Don't redesign them:

1. **3-degree graph traversal** lives inside `recall()` in brain_recall.py. Seeds are top 5 embedding hits. Degree 1: intentional edges only. Degree 2-3: all except co_accessed. Semantic bonus is additive only. Temporal freshness multipliers. Convergence boost for multi-parent nodes. The traversal adds ~16ms — not the bottleneck. `_traverse_graph()` is the method.

2. **Situation embeddings** are a second vector on each node: WHAT it's about (content embedding) + WHEN it matters (situation embedding). Free-form natural language, same cosine infrastructure. Stored in `node_embeddings.situation_embedding` and `node_embeddings.situation_text`. Recall scans both and adds situation score additively (`SITUATION_WEIGHT = 0.2`). The encoding agent writes situations. The contract includes situation as a promoted field.

3. **The encoding agent** fires every 5th Stop via daemon gating. The daemon owns the counter (`stop_counter` config). On 4/5 stops it returns NONE. On the 5th it inlines the full encoding prompt + conversation + brain context. The agent prompt is at `hooks/prompts/encoding-agent.md` — revision-first, with situation examples, structural types listed, contract field summary appended automatically.

4. **The data contract** is `servers/contract.py`. It defines all node fields (structural + promoted). MCP tool schemas are generated from it. Dispatch validates against it. The encoding prompt reads from it. New field = add to contract, it flows everywhere the contract is wired. BUT: 5 components still hardcode field lists (see "What's Not Done" below).

5. **Write verification** — `revise()` reads back every field after writing and returns `verified: true/false`. Verification failures are logged through `brain._log_error` and surface through the signal queue.

6. **Generic revise** — `revise(node_id, reason, situation="...", confidence=0.9, keywords="...")` updates any field. Content is appended (preserves history). All other fields are replaced. Situation gets its own embedding. SAFE_FIELDS whitelist controls what's valid (should be replaced by contract read — not done yet).

## What's NOT Done (and Why It Matters)

### 5 Renegade Components
These still hardcode field lists instead of reading the contract:
- `brain_recall.py` — 6 hardcoded SELECT statements
- `brain_remember.py` — SAFE_FIELDS hardcoded set in revise()
- `brain_voice.py` — EVOLUTION_TYPES, ENGINEERING_TYPES, CODE_COGNITION_TYPES
- `daemon_hooks.py` — candidates file builder hardcodes fields
- `pre_response_recall.py` — distiller format hardcodes fields

Each needs careful wiring: read the code, make the change, test immediately, verify.

### Metadata Dict Refactor
`remember()` has 14 named parameters. Should become `remember(type, title, content, **metadata)`. This unblocks promoted fields (reasoning, user_raw_quote) on remember. Currently only revise can set promoted fields.

### Retroactive Situation Backfill
874 nodes, 1 has a situation. The maintenance agent (not built yet) should generate situations for existing nodes via LLM. Surface for operator approval.

### Dashboard Encoding Tab
Still shows one-line entries. Needs expandable detail with kind badges (created/revised/connected), situation display, all metadata.

## Critical Traps — Things That Will Bite You

### 1. The Daemon Runs Old Code
When you change `servers/*.py`, the daemon still runs the old version. You MUST restart it: `hooks/scripts/restart-daemon.sh`. If you forget, writes silently use old logic. The plugin cache at `~/.claude/plugins/cache/brain/brain/8.6.0/` is a separate copy — rebuild with `bash build-plugin.sh`.

### 2. Hooks Are Snapshotted
Changes to `.claude/settings.json` hook config during a session are invisible. The encoding agent hook, the Stop agent, the once:true flag — all need a NEW SESSION to take effect.

### 3. INSERT OR REPLACE Wipes Sibling Columns
In `node_embeddings`, INSERT OR REPLACE replaces the entire row. Use UPDATE to preserve situation_embedding when re-embedding content. This bug was found and fixed in `brain_remember.py revise()`.

### 4. recall() Is Embeddings, _keyword_recall() Is Legacy
We renamed these. `recall()` IS the production embedding pipeline with graph traversal. `_keyword_recall()` is the 352-line TF-IDF method used only for the 10% keyword blend. If you see `_keyword_recall` — don't call it directly.

### 5. eval Sandbox Needs Safe Builtins
The eval handler at `daemon_dispatch.py _handle_eval` has `safe_builtins` including str, int, len, etc. Without these, the Stop agent can't read `brain.get_config()` results. Don't remove them.

### 6. User Messages in message_stream
`post_response_track.py` extracts user_message from the transcript, not from `hook_input.get("prompt")` (which is empty on Stop events). If user messages appear empty in message_stream, check that the thin client sends the extracted value.

## Architecture Overview

```
User sends message
  → UserPromptSubmit hook fires
    → pre_response_recall.py (thin client)
      → daemon: hook_recall
        → brain.recall() [embeddings + 3-degree graph + keyword blend + situation boost]
        → writes candidates file with _graph neighborhoods
      → reads candidates file
      → Haiku distills to dynamic budget (400-1200 chars)
      → returns [BRAIN]...[/BRAIN] to Claude

Claude responds
  → Stop hook fires (both hooks in parallel)
    → post_response_track.py (thin client)
      → daemon: hook_post_response_track
        → store_exchange (user msg + assistant msg to message_stream)
        → increment stop_counter
        → if 5th stop: build encoding prompt with conversation + brain context
    → Stop agent (Sonnet, agent hook)
      → eval(brain.get_config('stop_agent_prompt'))
      → if NONE: respond NOTHING_NEW
      → if instructions: run encoding with MCP tools
        → recall, find_node_by_title, get_node (search first)
        → revise (update stale), remember (create new), connect (link)
        → record_divergence (corrections)
```

## Testing System

### Quick Smoke Test
```bash
python3 -m pytest tests/test_v84_pipes.py tests/test_contract_sync.py -v
```
69 tests covering: DAL, traversal, recall integration, format_node_deep, encoding gating, integrity producer, schema, constants, eval sandbox, production sanity.

### Capability Tests
```bash
export ANTHROPIC_API_KEY=...
python3 eval/capabilities/test_revision.py      # 3 scenarios with fixture brain
python3 eval/capabilities/test_noise_resistance.py  # 3 scenarios
```
Tests revision behavior (does agent revise stale nodes?) and noise resistance (does agent skip casual chat?).

### Extensive Tests
```bash
python3 eval/test_extensive_encoding.py  # 5 scenarios × 20 messages
```
Full encoding pipeline on fresh brains with seed nodes. Tests architecture_revision, correction_heavy, noise_resistance, vocabulary_enrichment, long_technical.

### Real Conversation Simulation
```bash
python3 eval/simulate_real.py --batches 10 --start-from-end
```
Runs encoding agent on actual session transcripts from JSONL files.

## Key File Locations

| File | Purpose |
|---|---|
| `servers/contract.py` | Field definitions — the contract |
| `servers/brain_recall.py` | recall() with 3-degree traversal |
| `servers/brain_remember.py` | remember() + revise() with verification |
| `servers/daemon_hooks.py` | hook_recall + hook_post_response_track + encoding gating |
| `servers/daemon_dispatch.py` | MCP command handlers with contract validation |
| `servers/daemon_server.py` | Daemon process with restart command |
| `servers/brain_mcp.py` | MCP tool schemas (generated from contract) |
| `servers/brain_voice.py` | format_node + format_node_deep (3-degree rendering) |
| `servers/brain_constants.py` | Traversal constants, situation weights |
| `servers/dal.py` | Data access layer (get_node uses SELECT *, get_neighbors_rich) |
| `servers/signal_producers.py` | Integrity checks + deep audit |
| `hooks/prompts/encoding-agent.md` | Encoding agent prompt |
| `hooks/scripts/restart-daemon.sh` | One-command daemon restart |
| `hooks/scripts/pre_response_recall.py` | Recall thin client + Haiku distillation |
| `hooks/scripts/post_response_track.py` | Stop thin client + message storage |
| `scripts/seed_brain.py` | Seed brain builder (12 nodes, all features) |
| `eval/capabilities/base.py` | CapabilityTest + InstrumentedBrain + dispatch_tool |
| `eval/test_extensive_encoding.py` | 5 scenario × 20 message tests |
| `eval/fixtures/build_capability_brain.py` | Fixture brain with deliberate problems |

## Numbers

- 874 active nodes, 19376 edges, 612 locked
- 26 nodes ever revised (3%)
- 1 node with situation embedding (just started)
- 90 nodes with metadata (23% have reasoning)
- 69 pipe + contract tests passing
- 5/5 extensive encoding tests passing
- Recall latency: 340ms avg (183ms cosine, 128ms keyword, 16ms graph, 13ms other)
- 13 structural types, 18 open types
- Schema v20 (open types, no CHECK constraint)
- Plugin v8.6.0
