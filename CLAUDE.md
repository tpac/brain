# Brain Plugin — Developer Guide

This is the development repo for the brain plugin. CLAUDE.md is for developing the plugin, not using it. Plugin behavior lives in `skills/brain/SKILL.md` and boot injection.

## Architecture

Two paths to the brain, both through the daemon:

```
MCP tools (Anchor direct):  Claude Code → brain_mcp.py → daemon (TCP) → brain.recall(source='mcp')
Hook pipeline (per prompt):  Claude Code → hook scripts → daemon (TCP) → brain.recall(source='hook') + Haiku judge
```

The daemon is the single gateway. It holds: Brain object, embedder, anthropic client (for Haiku judge).
Listens on `127.0.0.1:47200+uid%100`. TCP — no Unix sockets.

DB resolved automatically: `BRAIN_DB_DIR` env var → `$HOME/AgentsContext/brain/`

### Decode pipeline (per user prompt)
```
UserPromptSubmit hook fires
  → pre-response-recall.sh → pre_response_recall.py (thin client)
    → daemon: hook_recall()
      ├─ Layer 1: brain.recall() → 25 candidates
      ├─ Layer 2: scales/s1/recall.py:run_judge() → Haiku selects 5-8 nodes
      │   └─ writes /tmp/brain-judge-selected.json
      │   └─ writes /tmp/brain-judge-result-{ref}.json (for dashboard)
      ├─ Layer 3: graph expansion + correction enrichment
      └─ returns formatted additionalContext
    → thin client prints {"additionalContext": "Brain recalled N memories:..."}
  → Claude receives context and responds
```

### Encode pipeline (every 5th Stop)
```
Stop hook fires
  ├─ store_exchange() → message_stream (escalation only)
  ├─ S0 traces (user_message, assistant_message)
  ├─ Hebbian strengthening (co_accessed edges between judge-selected nodes)
  └─ Every 5th stop → scales/runner.py:run_in_background()
      └─ scales/s1/encode.py:run_encoding() (Sonnet, background thread)
          ├─ Reads: S0 traces (conversation turns via TraceDAL)
          ├─ Node catalog: judge-surfaced nodes with full metadata, deduplicated
          ├─ Timeline: conversation turns with node ID references
          ├─ Tools: brain_batch, remember_batch, revise_batch, connect_batch, recall_batch, get_nodes
          └─ Creates/revises/connects nodes (encoding_source='encoder:sonnet')
```

### Scales Architecture
```
servers/scales/
  dispatch.py         — TCP dispatch + dispatch factory (shared by all scales)
  runner.py           — background thread lifecycle + generic LLM tool loop
  s1/
    recall.py         — S1R: judge call, graph expand, correction enrich, traces
    recall_contract.py— S1R config, judge prompt, output formatting
    encode.py         — S1E: gather messages, build prompt, LLM loop, journal
    encode_contract.py— S1E config, node formatting, catalog building
  s2/                 — (future: session encoder, copies s1/ pattern)
```

S0 hooks stay in `daemon_hooks.py` (they're observation points, not scale logic).
Old files (`encoding_agent.py`, `judge_contract.py`, `encoding_contract.py`) are re-export shims.

### encoding_source convention
Who created a node. Format: `category:process`.
- `anchor` — Anchor direct via MCP (only source that can lock nodes)
- `encoder:sonnet` — encoding agent
- `idle:dreams` / `idle:redistribution` — background processes
- `hook:compaction` — hook lifecycle markers

## Hook Pipeline

Hooks fire automatically — do NOT manually run boot scripts:
- `SessionStart` → boots brain + daemon, prints context + consciousness
- `UserPromptSubmit` → thin client calls daemon for recall + judge → returns additionalContext
- `PreToolUse(Edit|Write)` → surfaces rules before file edits
- `PreToolUse(Bash)` → safety check for destructive commands
- `PreCompact` → saves brain before context loss
- `PostCompact` → re-boots context after compaction
- `Stop` → stores exchange to message_stream (with judge-selected IDs + judge_output), Hebbian strengthening, gates encoding agent (every 5th stop)
- `SessionEnd` → session synthesis + save

The UserPromptSubmit hook is a thin wrapper — all logic (recall, judge, graph expansion) runs in the daemon.

## Benchmark-First Rule

Before changing sacred systems, benchmark FIRST:
- Embedding: `servers/embedder.py`
- Recall: `servers/brain_recall.py`
- Encoding: `servers/brain_remember.py` + `servers/encoding_agent.py`
- Hook output: `servers/brain_voice.py`

**Encoding Eval** (`eval/encoding_prompt_eval_v32.py`): A/B comparison of encoder prompt formats. Uses `IsolatedBrain` — fully isolated from production. Measures: prompt size, tokens, rounds, time, actions, field richness.

**Decode Funnel** (`eval/decode_funnel.py`): tests recall quality — 50 queries, 5 categories. Run before ANY recall change.

## Test Integrity Rule

**When a test fails, STOP.** Do not change the test OR the code.
1. Report: what the test expected vs what code returned.
2. Ask: "Is the test wrong, or does the code have a bug?"
3. Wait for the answer.

Do NOT weaken tests. Do NOT "fix" code to satisfy a test without asking.

## Contract Sync

Run `test_contract_sync.py` after modifying ANY brain API layer. The contract (`servers/contract.py`) is the single source of truth for field definitions. It flows to: remember() signature, MCP schema, dispatch, encoding agent tools.

## Encode-Decode Symmetry

Encoding and decoding (recall) are two halves of the same system. If you add a field to encoding, it must be queryable in recall. If you change how nodes are structured, recall ranking and filtering must reflect it. Never change one side without checking the other. The decode funnel is the verification — run it.

## Active Mechanisms (recall what these are before modifying recall)

**Synaptic fatigue** — `brain_recall.py` STEP 3. Nodes recalled repeatedly in a session get cosine dampened. Rate scales with structural degree (hubs fatigue faster). Resets between sessions.

**Hebbian co_accessed** — RE-ENABLED. Judge-selected IDs now flow to Stop hook. Only nodes the judge selected get co_accessed edges — meaningful co-activation, not cosine coincidences.

**Embedding redistribution** — `servers/redistribution.py`. Blends node embeddings toward graph neighbors (70/30 from frozen originals). Runs in sleep cycle. Fidelity tracked in `embedding_fidelity` table.

**Z-weighted 4-group scoring** — `brain_recall.py` STEP 3.5. Title(1.0), blend(0.85), high_meta(0.70), other_meta(0.40). Top-2 averaged. Defined in `pipeline_contract.py` EMBEDDING_GROUPS.

**Layer 2 judge** — runs in `scales/s1/recall.py:run_judge()`, called from daemon_hooks.hook_recall(). Haiku selects relevant nodes from 25 candidates. Graph expansion + correction enrichment from judge-selected seeds. Stays silent on confirmations.

**Precision** — DELETED. `brain_precision.py` removed (2026-04-05). recall_log writes removed — traces are single source of truth. Precision metrics will be rebuilt from trace outcome events.

## Dashboard

The dashboard (`dashboard/brain_dashboard_standalone.py`) is a **passive observer** — it reads, never writes to brain data. It inspects the brain from the side without interfering with the process.

```
Brain (daemon + hooks) → does the work → writes to DBs + tmp files
Dashboard → reads from those same DBs + files → displays to operator
```

**Data sources the dashboard reads (all read-only):**
- `brain.db` — nodes, edges, encoding activity, graph data
- `brain_logs.db` → `recall_log` — single source of truth for ALL recalls (hook, MCP, internal). Includes query, candidates, titles, snippets, source.
- `brain_logs.db` → `signal_queue` — pending signals
- `brain_logs.db` → `hook_errors`, `debug_log`, `brain_telemetry` — errors and diagnostics
- `/tmp/brain-judge-result-{id}.json` — judge prompt + output, written by hooks for async pickup

**`brain_dashboard.db` is DEPRECATED.** It was a parallel logging pipe that diverged from reality. The dashboard now reads from the brain's own data — same tables the daemon uses.

**`recall_log` writes REMOVED (2026-04-05).** Traces (trace_events table) are the single source of truth for all recall events. The recall_log table still exists with historical data but no new rows are inserted. Dashboard recall display should migrate to reading from trace_events.

**Judge data flows through tmp files.** The daemon writes `/tmp/brain-judge-result-{recall_ref}.json` containing the exact Haiku prompt and the exact additionalContext sent to Claude. The dashboard reads these files passively.

## Fractal Trace System

The brain captures execution traces using `integrate(O, K) → Δ` at every scale. See `docs/ARCHITECTURE-FRACTAL.md` for the full architecture.

**Tables (brain_logs.db):**
- `trace_events` — O/K/Δ/outcome events, grouped by chain_id, tagged by scale (s0-s4)
- `interactions` — versioned prompt text + config for every learnable boundary
- `session_state` — per-session state including SessionContext persistence

**Trace contract** (`servers/trace_contract.py`): single source of truth for scales, event types, ref types. All trace writers validate against it. Run `test_trace_contract_sync.py` after any trace change.

**Scales:**
- s0: raw exchange (user messages, assistant responses, tool results via PostToolUse)
- s1: turn integration (recall candidates, judge selection, encoding actions)
- s2-s4: session/sleep/growth (contract defined, not yet built)

**Summary/metadata split:** summary is short (200 chars, for dashboard display). metadata JSON has full content (4000 chars, for consumers).

**Chain IDs:** `s0-{session_short}-{stop}`, `s1r-{session_short}-{stop}`, `s1e-{session_short}-{stop}`. Generated by SessionContext.

**Dashboard:** Traces tab shows events grouped by chain, filterable by scale and time range, auto-refreshes.

**PostToolUse hook** (`hooks/scripts/post_tool_trace.py`): captures ALL tool results as s0 delta traces via daemon `trace_append` command.

## Interactions (Learnable Boundaries)

Every boundary where two parts of the system meet has an entry in the `interactions` table. Two types:
- **LLM boundaries** (judge, encoding_agent): `template` holds the prompt instructions, `parameters` holds config JSON
- **Code boundaries** (voice_surface, boot, pre_edit, signal_assembler): `template` is empty, `parameters` holds config JSON

**How code reads interactions:**
- `brain.get_interaction_prompt('judge')` → returns prompt text (or '' if not found)
- `brain.get_interaction_config('judge')` → returns parsed config dict (or {} if not found)
- `brain.get_interaction('judge')` → returns full interaction with id for trace linkage
- Always falls back to hardcoded defaults if interactions table is empty

**Versioning:** `register()` auto-increments version. Old versions preserved. `created_by` tracks who made the change ('anchor', 'sleep:s3', 'growth:s4'). Higher scales write new versions to evolve behavior.

**Trace linkage:** trace events include `interaction_id` — which version produced this result. Compare outcomes across versions to evaluate changes.

**Seeding:** `interaction_seed.py` populates v1 from current hardcoded values on boot (idempotent).

**The 6 interactions and their config keys:**

`judge` (LLM — Haiku): prompt instructs how to select relevant nodes
- config: content_limit, max_candidates, max_selected, user_message_limit, anchor_message_limit, recent_messages, session_context_limit, max_tokens
- reads: `pipeline_contract.py:build_judge_prompt()` — prompt from DB, data assembly in code
- wired: YES (reads from interactions table at runtime)

`encoding_agent` (LLM — Sonnet): prompt instructs how to encode conversation to nodes
- config: message_content_limit, max_messages, max_rounds, journal_max_chars, max_tokens, session_context_limit, node_edge_limit
- reads: `encoding_agent.py:_build_system_prompt()` — prompt from DB, field summary appended in code
- wired: YES (reads from interactions table at runtime)

`voice_surface` (code only): formats judge output into additionalContext for Anchor
- config: content_truncation, situation_truncation, quote_truncation, max_edges, node_title_max, edge_title_max
- reads: `pipeline_contract.py:format_judge_output()` — NOT YET WIRED (hardcoded)

`boot` (code only): session start context — identity, last session, self-knowledge
- config: boot_nodes_limit, boot_nodes_truncation, tom_quotes_limit, tom_quotes_truncation, self_knowledge_limit, session_decisions_limit
- reads: `brain_voice.py:render_boot_v2()` — NOT YET WIRED (hardcoded)

`pre_edit` (code only): surfaces rules before file edits
- config: recall_pool_multiplier, suggestion_limit, encoding_health_stale_edits, encoding_health_stale_minutes, context_files_limit
- reads: `brain_surface.py:pre_edit()` — NOT YET WIRED (hardcoded)

`signal_assembler` (code only): budget-based signal selection for context injection
- config: budget_chars, max_proactive_signals, reminder_priority, encoding_gap_priority, cooldown values
- reads: `surface_assembler.py` — NOT YET WIRED (hardcoded)

## SessionContext

Session identity flows with every brain call — hooks, MCP, encoding. The brain doesn't own sessions.

**`brain.get_or_create_session(session_id)`** — single entry point. Loads from DB if exists (daemon restart recovery), creates if new. Returns SessionContext with session_id + stop_counter.

**Thin clients** must pass `session_id` from Claude Code hook args to the daemon. Critical path (recall, stop, encode, pre_edit) passes it. Others fall back to brain_meta session_id.

**Stop counter** managed by SessionContext (not brain_meta). Increments on each Stop hook. Encoding fires every 5th stop.

**Chain IDs** generated by SessionContext: `ctx.s0_chain()`, `ctx.s1r_chain()`, `ctx.s1e_chain()`.

## Test Isolation

**All tests and evals MUST use `IsolatedBrain`** (`tests/isolated_brain.py`). Never run tests against production databases. The harness copies brain.db + brain_logs.db to a temp directory and creates an isolated Brain instance. Production is never touched.

```python
from tests.isolated_brain import IsolatedBrain

with IsolatedBrain() as env:
    result = env.dispatch("recall", {"query": "test", "limit": 5})
    # env.brain — isolated Brain instance
    # env.db_dir — temp directory (auto-cleaned)
```

This is infrastructure for ALL tests — unit, integration, evals, benchmarks. No exceptions.

## Code Ownership

Tom reads code but doesn't review every file. You are the sole maintainer of code quality, architecture, and cleanliness. These rules are your guardrails:

**Contract-first** — Constants, field lists, SQL queries, limits, and config live in contract files (`contract.py`, `pipeline_contract.py`, `brain_constants.py`). Never hardcode a limit, field name, or query in hooks, dispatch, or surface code. If you're typing a number or a column name in application code, it belongs in a contract.

**DAL-first** — Use DAL classes for database access (`dal.py`, `dal_message_stream.py`). No raw SQL in hooks, surface code, or MCP handlers. If a DAL method doesn't exist for what you need, add one — don't work around it with inline queries.

**Single-writer rule** — The daemon's main thread is the ONLY writer to brain.db and brain_logs.db. Background threads (encoding agent, idle maintenance) must route writes through the daemon via TCP dispatch, not write directly. This eliminates "database is locked" errors. Pattern:
- Main thread hooks (recall, stop, pre-edit): write directly via `brain.method()`
- Background threads: send writes via TCP to daemon (`dispatch_fn('remember', args)`)
- Read operations: any thread can read (WAL mode allows concurrent readers)
- The encoding agent's `brain` parameter is READ-ONLY. All writes go through `dispatch_fn`.
- Trace writes from hooks use `brain._trace_dal.append()` (main thread). Trace writes from encoding use `dispatch_fn('trace_append', args)` (background thread).

**Trace the full flow** — When adding a field: schema → migration → contract → DAL → remember/recall → dispatch → MCP schema → encoding agent prompt → SKILL.md docs. Missing any step creates a silent gap. When deprecating: reverse the same chain.

**Run tests after every change** — `test_contract_sync.py` after API changes. Decode funnel after recall changes. Don't commit and move on without verification.

**Backup before destructive DB operations** — Before ANY delete, bulk update, or schema migration on the live brain.db: `cp brain.db brain.db.bak-{timestamp}`. No exceptions. The backup takes 1 second. Losing data takes weeks to recover. This includes: deleting edges, archiving nodes in bulk, running redistribution, vacuuming.

**Clean as you go** — When you deprecate something, mark it clearly with `# DEPRECATED` and a date. Remove dead code within the same session if possible. Don't leave "TODO: remove later" — later never comes.

**One concern per file** — If you're about to add a function to a file and it serves a different audience than the file's existing functions, it belongs in a different file. hooks handle hook logic. contracts handle contracts. surface handles surface formatting.

## Key Development Rules

- Use MCP tools to interact with brain during development, not Python/bash scripts
- Don't manually run boot scripts (hooks handle this)
- Don't construct DB paths (read the boot output)
- `systemMessage` is a dead channel — use `additionalContext` for hook output
- Before writing code, ask "where does this live architecturally?"
- Good architecture makes you MORE efficient, not less — each area has its own file/module
- **Discussion IS the work** — do not touch Edit/Write/code tools during design conversations. Wait for an explicit go signal. Questions and observations are not go signals.
- **Trace the pipeline before changing it** — the decode→encode pipeline has coupled stages. Each component expects specific input from the previous stage. Don't change one stage without understanding the full flow.
- **Encoding depends on decoding** — the encoder receives judge-selected nodes from the decode pipeline. If the judge fails, the encoder gets no context. If recall quality degrades, encoding quality degrades. A broken decode pipeline silently breaks encoding.
