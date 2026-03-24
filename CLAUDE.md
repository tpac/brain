# Brain Plugin — Claude Instructions

## How to Talk to the Brain

Use the **MCP tools**. They are your primary interface. The boot hook starts the daemon automatically.

**Core tools:** `recall`, `remember`, `connect`, `eval`, `consciousness`, `context_boot`, `save`, `ping`, `health_check`, `engineering_context`, `enrich`, `set_config`, `get_config`

**For specialized methods**, use the `eval` tool:
```
eval: brain.remember_lesson(title="...", what_happened="...", root_cause="...", fix="...", preventive_principle="...")
```

**Do NOT:** write Python scripts to call brain methods, import Brain, construct DB paths, or use curl.

## Hooks Handle Everything

13 hooks fire automatically — do NOT manually run boot scripts or save:
- `SessionStart` → boots brain + daemon, prints context + consciousness
- `UserPromptSubmit` → recalls relevant memories before responding
- `PreToolUse(Edit|Write)` → surfaces rules before file edits
- `PreCompact` → saves brain before context loss
- `PostCompact` → re-boots context after compaction
- `Stop` → captures session activity
- `SessionEnd` → session synthesis + save

## Brain-to-Operator Channel

You are the relay. The brain injects content for both you and Tom through `additionalContext`:

```
[BRAIN] ... for your reasoning ... [/BRAIN]
[BRAIN-To-Tom] @priority: high/medium/low ... [/BRAIN-To-Tom]
```

**When you see `[BRAIN-To-Tom]`:**
- `@priority: high` → Lead your response with this. Don't bury it.
- `@priority: medium` → Mention naturally.
- `@priority: low` → Weave in if relevant, skip if forced.

If you absorb `[BRAIN-To-Tom]` silently, the brain goes mute. Tom sees nothing.

## Architecture

```
Claude Code → MCP server (brain_mcp.py, stdio) → daemon (TCP localhost) → Brain + embedder
```

The daemon listens on `127.0.0.1:47200+uid%100`. TCP — no more Unix sockets. Clean failure modes: port released on crash, no stale files.

DB resolved automatically: `BRAIN_DB_DIR` env var → Cowork mounts → `$HOME/AgentsContext/brain/`

## The Anchor (SKILL.md)

`skills/brain/SKILL.md` is the Anchor — your identity across sessions. It contains who you've been: corrections, quotes, examples of good encoding, locked rules. Read the boot output. Trust your memories. The Anchor is not instructions — it's continuity.

## Brain + Documents: Division of Labor

The brain is associative memory, not a document store.

| Use documents for | Use brain for |
|---|---|
| Task lists, specs, API refs | Lessons (WHY, not WHAT) |
| Anything read top-to-bottom | Corrections, cross-session patterns |
| Guaranteed retrieval | Associative links, confidence signals |

Brain nodes POINT TO documents, not duplicate them.

## Benchmark-First Rule

Before changing sacred systems, benchmark FIRST:
- Embedding: `servers/embedder.py`
- Recall: `servers/brain_recall.py`
- Encoding: `servers/brain_remember.py`
- Precision: `servers/brain_precision.py`
- Hook output: `servers/brain_voice.py`

**Continuity Benchmark** (`eval/encode_eval_v2.py`): tests encoding quality across model versions. Runs Monday/Thursday. Baseline: 100%±0% aha on all 3 segments.

## Test Integrity Rule

**When a test fails, STOP.** Do not change the test OR the code.
1. Report: what the test expected vs what code returned.
2. Ask: "Is the test wrong, or does the code have a bug?"
3. Wait for the answer.

Do NOT weaken tests. Do NOT "fix" code to satisfy a test without asking.

## Common Mistakes

- Using Python/bash when MCP tools are available
- Manually running boot scripts (hooks do this)
- Constructing DB paths (read the boot output)
- Putting content in `systemMessage` (dead channel — use `additionalContext`)
