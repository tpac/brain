# Brain Plugin — Developer Guide

This is the development repo for the brain plugin. CLAUDE.md is for developing the plugin, not using it. Plugin behavior lives in `skills/brain/SKILL.md` and boot injection.

## Why the Brain Exists

The brain serves the partnership — not retrieval, not memory, not continuity. Those are foundation, not goal.

Each side is better than they could be alone. Tom + Brain + Anchor > Tom alone. Anchor + Brain + Tom > Claude alone. Every scale, every mechanism converges on this.

## Core Principle

```
integrate(O, K) → Δ

O = observation    (a phenomenon — a message, a cluster, traces from lower scales)
K = knowledge      (what I bring — prompts, algorithms, config, reasoning)
Δ = change         (the action — create, revise, link, correct)
```

Same function at every scale. The unit doesn't know its scale. `integrate()` is the processing engine — the LLM, an algorithm, a code path. O is what it sees. K is what shapes how it sees. Δ is what it does.

**The fractal property:** Δ from one scale feeds other scales' O or K. S1E encodes a node (Δ) → S1R recalls it next session (O). S2 rewrites a surface prompt (Δ) → S1R selects differently (K). There is no separate inter-layer protocol. It's O/K/Δ all the way.

**The system's purpose:** every scale, every mechanism exists to enrich S0's K — the context Anchor has when responding. The entire brain is a K-enrichment machine for the partnership.

## The Daemon

Single gateway to the brain. Holds: Brain object, embedder, Anthropic client (for surfacer). Listens on `127.0.0.1:47200+uid%100` (TCP). DB path from `BRAIN_DB_DIR` env var → `$HOME/AgentsContext/brain/`.

Two databases:
- `brain.db` — nodes, edges, embeddings, graph structure
- `brain_logs.db` — traces, session state, signal queue, interactions, hook errors

**Single-writer rule:** the daemon's main thread is the ONLY writer. Background threads (encoding agent, idle) route writes through TCP dispatch. Read operations: any thread can read (WAL mode).

Auto-starts on first hook fire. **Maintenance mode:** `touch /tmp/brain-maintenance-{uid}.lock` prevents auto-restart during VACUUM, schema changes, bulk deletes. Remove lock when done.

The dashboard (`dashboard/`) is a passive observer — reads from the same DBs + `/tmp/brain-surface-result-*.json` files, never writes.

## Scale 0: Exchange

Every conversation turn. Tom's message is the observation (O). Everything the brain injects — boot context, recalled nodes, signals, rules — is knowledge (K). Anchor's response is the change (Δ). The response becomes part of the next O. This IS the loop.

Hooks are S0's observation points — all logic lives in the daemon, hooks are thin clients:
- `SessionStart` → boots daemon, prints identity context
- `UserPromptSubmit` → triggers S1R decode (recall + surface → additionalContext)
- `PreToolUse(Edit|Write)` → surfaces rules before file edits
- `PreToolUse(Bash)` → safety check for destructive commands
- `Stop` → writes S0 traces (user_message, assistant_message), gates S1E encode (every 5th stop)
- `PreCompact` → saves brain before context loss
- `PostCompact` → re-boots context after compaction
- `SessionEnd` → session synthesis + save

S0 traces written by Stop hook via TraceDAL. Chain ID: `s0-{session_short}-{stop}`.

`additionalContext` is the ONLY channel that reaches Claude. `systemMessage` is dead.

## Scale 1: Turn

S1 integrates across turns — selecting what's relevant (S1R) and encoding what matters (S1E).

### S1R: Decode (every user prompt)

Triggered by Anchor's need to know — the brain surfaces context before Anchor responds.

**O:** 25 recall candidates from `brain.recall()`
- Cosine similarity across z-weighted 4-group embeddings (title:1.0, blend:0.85, high_meta:0.70, other_meta:0.40)
- Synaptic fatigue dampens repeatedly-recalled nodes (hubs fatigue faster, lives on SessionContext)

**K:** Haiku surfacer selects 5-8 relevant nodes
- Prompt + config from `surface` interaction (learnable boundary — higher scales optimize this)
- Hebbian strengthening: co_accessed edges between selected nodes

**Δ:** additionalContext injected into Claude's context
- Graph expansion from selected seeds
- Correction enrichment (corrects/corrected_by chains)
- Formatted via `format_surface_output()`

Files: `scales/s1/recall.py`, `scales/s1/recall_contract.py`
Traces: `s1r-{session_short}-{stop}` (O: candidates, K: selected, Δ: additionalContext)
Interaction: `surface` — the learnable boundary that higher scales will evolve

### S1E: Encode (every 5th stop)

Triggered by Stop hook when `stop_counter % 5 == 0`.

**O:** Conversation + context from this session
- S0 traces: last 10 turns (20 messages) via `TraceDAL.get_session_turns()`
- Surface-selected nodes across those turns (what S1R selected)
- Encoding journal (cumulative across runs in session)
- Session context

**K:** Sonnet agent decides what to encode
- Node catalog: `format_node()` with correction annotations
- Conversation timeline with SURFACED node references
- Agent prompt from `encoding_agent` interaction (learnable boundary)

**Δ:** Sonnet's actions — create, revise, connect nodes
- Tools: `remember_batch`, `revise_batch`, `connect_batch`, `brain_batch`, `recall_batch`, `get_nodes`
- Journal updated, session context extracted from agent output

Files: `scales/s1/encode.py`, `scales/s1/encode_contract.py`
Traces: `s1e-{session_short}-{stop}` (O: encoding prompt, K: node catalog, Δ: encoding actions)
Interaction: `encoding_agent` — the learnable boundary that higher scales will evolve

## Scale 2: Graph Maintenance

S2 operates when Tom is away. It sees the full graph, not just one turn. Multiple integration units, different triggers, same O/K/Δ pattern.

| Unit | Trigger | O | K | Δ | Commit |
|------|---------|---|---|---|--------|
| Dedup | S1E creates node | duplicate cluster | similarity + LLM | merge/archive | signal → Tom |
| Confidence | session end | recall traces | decay/growth algo | adjusted scores | auto |
| Community | session end | edge structure | leidenalg | community labels | auto |
| Correction | S1E encodes correction | correction chain | resolution rules | archive stale | signal → Tom |
| Tool learn | S0 tool traces | operational patterns | pattern detection | operational nodes | auto |
| Redistribution | scheduled | node embeddings | 70/30 blend rule | shifted vectors | auto |

Autonomy gradient: pure-code units auto-commit. LLM judgment units signal Tom via signal queue. The signal queue is S2's commit channel to S0.

Files: `scales/s2/` (mirrors s1/ structure)
Shared: `scales/dispatch.py`, `scales/runner.py`, `contract.py:format_node()`
Traces: `s2-{session}-{unit}`
**Status: NOT BUILT**

## Shared Infrastructure

Three things make the fractal real. If any is missing for a boundary, that boundary can't participate in cross-scale optimization.

### 1. Interactions (the K store)

The most important table in the brain. Not nodes — those are memory. Interactions are *behavior*. When S2 rewrites the surface prompt based on trace outcomes, the brain isn't just remembering differently — it's thinking differently.

Every boundary where two parts meet has an interaction entry: versioned prompt + config JSON. `register()` auto-increments version. `created_by` tracks who wrote it. Trace events reference `interaction_id` — which version produced which result. Compare outcomes across versions to evaluate changes.

**What's wired:** `surface` and `encoding_agent` read from the table at runtime.
**What's broken:** `voice_surface`, `boot`, `pre_edit`, `signal_assembler` have entries but don't read them — hardcoded in Python. This means S2 can't optimize boot context or surface output formatting. The fractal is broken at these boundaries.

API: `brain.get_interaction_prompt(name)`, `brain.get_interaction_config(name)`, `brain.get_interaction(name)`. Falls back to hardcoded defaults if table is empty.

Seeding: `interaction_seed.py` populates v1 from current hardcoded values on boot (idempotent).

### 2. Traces (the nervous system)

Without traces, higher scales are blind. `trace_events` in brain_logs.db captures O/K/Δ/outcome per chain, tagged by scale.

**Trace contract** (`servers/trace_contract.py`): single source of truth for valid (scale, event_type, ref_type) triples. All writers validate against it. Run `test_trace_contract_sync.py` after any change.

**What works:** S0 captures messages + tool results. S1R captures candidates, surface selection, additionalContext. S1E captures encoding prompt, catalog, actions.
**What's missing:** Outcome events. Did Anchor use the recalled context? Did Tom correct it? S2 needs this signal to evaluate whether S1R's K is working. Without it, S2 optimizes blind.

Chain IDs from SessionContext: `s0-{short}-{stop}`, `s1r-{short}-{stop}`, `s1e-{short}-{stop}`.
`TraceDAL.append_batch()` for atomic multi-event writes. Summary (200 chars) for dashboard, metadata JSON (4000 chars) for consumers.

### 3. Contracts (the shape of data)

What each boundary expects and produces. If recall changes its output shape, the surfacer breaks. Contracts prevent silent drift.

- `contract.py` — field definitions, `format_node()`, `generate_field_summary()`. Single source of truth for what a node IS.
- `pipeline_contract.py` — surface prompt assembly, z-weighted scoring groups. What flows between pipeline stages.
- Scale contracts: `s1/recall_contract.py` (surface config, candidate formatting), `s1/encode_contract.py` (encoder config, catalog building)

Run `test_contract_sync.py` after modifying ANY brain API layer. The contract flows to: remember() signature → MCP schema → dispatch → encoding agent tools → SKILL.md docs.

### Dispatch + Runner

Shared by all scale agents, scale-agnostic:
- `scales/dispatch.py` — TCP dispatch factory (reads local, writes via daemon)
- `scales/runner.py` — background thread lifecycle + generic LLM tool loop

S2 plugs in the same way S1 does.

### SessionContext

Per-session state that flows with every brain call. Carries: session_id, stop_counter, fatigue dict. Persisted to session_state table (survives daemon restarts).

`brain.get_or_create_session(session_id)` — single entry point. Thin clients must pass session_id from Claude Code hook args.

Chain IDs generated by SessionContext: `ctx.s0_chain()`, `ctx.s1r_chain()`, `ctx.s1e_chain()`.

### encoding_source Convention

Who created a node. Format: `category:process`.
- `anchor` — Anchor direct via MCP (only source that can lock nodes)
- `encoder:sonnet` — S1E encoding agent
- `idle:dreams` / `idle:redistribution` — background processes
- `hook:compaction` — hook lifecycle markers

## Development Rules

### Test Integrity

**When a test fails, STOP.** Do not change the test OR the code.
1. Report: what the test expected vs what code returned.
2. Ask: "Is the test wrong, or does the code have a bug?"
3. Wait for the answer.

Do NOT weaken tests. Do NOT "fix" code to satisfy a test without asking.

### Test Architecture

Tests organized by what they catch:
- **Contract tests** — layer sync, trace writes, pipeline shapes
- **Component tests** — DAL, format_node, scoring, signal queue, S1 data assembly
- **Transition tests** — wiring between pipeline stages (format changes that break consumers)
- **Cycle tests** — O/K/Δ loop property (Δ becomes next O)
- **Integration tests** — real data, full pipeline

`BrainTestBase` for tests needing a brain. Set `needs_embedder = False` for tests that don't need semantic search (saves 1GB + 1.5s). `IsolatedBrain` for tests against production data copies.

### Benchmark-First Rule

Before changing sacred systems, benchmark FIRST:
- Recall: `servers/brain_recall.py` — run decode funnel (`eval/decode_funnel.py`)
- Encoding: `scales/s1/encode.py` — run encoding eval (`eval/s1_encode_eval.py`)

### Contract Sync

Run `test_contract_sync.py` after modifying ANY brain API layer.

### Encode-Decode Symmetry

Encoding and decoding are two halves of the same system. If you add a field to encoding, it must be queryable in recall. If you change how nodes are structured, recall ranking must reflect it. The decode funnel is the verification.

### Code Ownership

Tom reads code but doesn't review every file. You are the sole maintainer of code quality, architecture, and cleanliness.

**Contract-first** — Constants, field lists, limits, and config live in contract files. Never hardcode in hooks, dispatch, or surface code.

**DAL-first** — Use DAL classes for database access (`dal.py`). No raw SQL in hooks, surface code, or MCP handlers.

**Backup before destructive DB operations** — `cp brain.db brain.db.bak-{timestamp}` before ANY delete, bulk update, or schema migration. No exceptions.

**Clean as you go** — Mark deprecated code with `# DEPRECATED` and date. Remove dead code same session. Don't leave "TODO: remove later."

**One concern per file** — If a function serves a different audience than the file's existing functions, it belongs in a different file.

### Key Development Rules

- Use MCP tools to interact with brain, not Python/bash scripts
- Don't manually run boot scripts (hooks handle this)
- Don't construct DB paths (read the boot output)
- `systemMessage` is a dead channel — use `additionalContext`
- **Discussion IS the work** — do not touch Edit/Write tools during design conversations. Wait for an explicit go signal.
- **Trace the pipeline before changing it** — the decode→encode pipeline has coupled stages. Don't change one stage without understanding the full flow.
- **Encoding depends on decoding** — if the surfacer fails, the encoder gets no context. A broken decode pipeline silently breaks encoding.
