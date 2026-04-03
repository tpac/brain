# Brain Plugin — Developer Guide

This is the development repo for the brain plugin. CLAUDE.md is for developing the plugin, not using it. Plugin behavior lives in `skills/brain/SKILL.md` and boot injection.

## Architecture

```
Claude Code → MCP server (brain_mcp.py, stdio) → daemon (TCP localhost) → Brain + embedder
```

The daemon listens on `127.0.0.1:47200+uid%100`. TCP — no Unix sockets. Port released on crash, no stale files.

DB resolved automatically: `BRAIN_DB_DIR` env var → Cowork mounts → `$HOME/AgentsContext/brain/`

## Hook Pipeline

13 hooks fire automatically — do NOT manually run boot scripts:
- `SessionStart` → boots brain + daemon, prints context + consciousness
- `UserPromptSubmit` → recalls relevant memories before responding
- `PreToolUse(Edit|Write)` → surfaces rules before file edits
- `PreCompact` → saves brain before context loss
- `PostCompact` → re-boots context after compaction
- `Stop` → precision evaluation + auto-encode signals + stores conversation to message stream
- `SessionEnd` → session synthesis + save

`post-response-track` fires ONLY on `Stop` (not UserPromptSubmit). It needs Claude's response to evaluate precision.

## Benchmark-First Rule

Before changing sacred systems, benchmark FIRST:
- Embedding: `servers/embedder.py`
- Recall: `servers/brain_recall.py`
- Encoding: `servers/brain_remember.py`
- Precision: `servers/brain_precision.py`
- Hook output: `servers/brain_voice.py`

**Continuity Benchmark** (`eval/encode_eval_v2.py`): tests encoding quality across model versions. Runs Monday/Thursday. Baseline: 100%±0% aha on all 3 segments. Winner: `identity_examples` variant with live brain access.

**Decode Funnel** (`eval/decode_funnel.py`): tests recall quality — 50 queries, 5 categories. Baseline after recency boost: 51% top-3, 69% top-8. Run before ANY recall change.

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

**Synaptic fatigue** — `brain_recall.py` STEP 3. Nodes recalled repeatedly in a session get cosine dampened. Rate scales with structural degree (hubs fatigue faster). Resets between sessions. Brain node: `4b35293c`.

**Hebbian co_accessed** — DISABLED. Was creating 71K noise edges. Will re-enable when judge-selected IDs flow to Stop hook. Brain node: `de56bfd1`.

**Embedding redistribution** — `servers/redistribution.py`. Blends node embeddings toward graph neighbors (70/30 from frozen originals). Runs in sleep cycle. Fidelity tracked in `embedding_fidelity` table.

**Z-weighted 4-group scoring** — `brain_recall.py` STEP 3.5. Title(1.0), blend(0.85), high_meta(0.70), other_meta(0.40). Top-2 averaged. Defined in `pipeline_contract.py` EMBEDDING_GROUPS.

**Layer 2 judge** — `pre_response_recall.py`. Haiku selects relevant nodes from 25 candidates. Replaces old distiller. Session context from encoder. Stays silent on confirmations.

## Dashboard

The dashboard (`servers/brain_dashboard_standalone.py`) is a **passive observer** — it reads, never writes to brain data. It inspects the brain from the side without interfering with the process.

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

**Recall logging is inside `brain.recall()`.** Every recall — hook, MCP, internal — gets logged to `recall_log` with a `source` column. No caller needs to log separately. This was moved from the hook into the recall method itself to ensure single source of truth.

**Judge data flows through tmp files, not the hook.** The hook writes `/tmp/brain-judge-result-{recall_log_id}.json` containing the exact Haiku prompt and the exact additionalContext sent to Claude. The dashboard reads these files. This decouples judge monitoring from hook timeout constraints.

## Code Ownership

Tom reads code but doesn't review every file. You are the sole maintainer of code quality, architecture, and cleanliness. These rules are your guardrails:

**Contract-first** — Constants, field lists, SQL queries, limits, and config live in contract files (`contract.py`, `pipeline_contract.py`, `brain_constants.py`). Never hardcode a limit, field name, or query in hooks, dispatch, or surface code. If you're typing a number or a column name in application code, it belongs in a contract.

**DAL-first** — Use DAL classes for database access (`dal.py`, `dal_message_stream.py`). No raw SQL in hooks, surface code, or MCP handlers. If a DAL method doesn't exist for what you need, add one — don't work around it with inline queries.

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
