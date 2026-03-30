# Design: Encoding Agent v3 — Pre-attached Recall

## Problem

The encoding agent spends 4 out of 6 rounds (~70% of time, ~60% of tokens) doing recalls that were ALREADY done by the hook pipeline. Every user message triggers a recall via UserPromptSubmit hook → Haiku distiller. That recall output is computed, paid for, and then thrown away. The encoding agent starts blind and repeats the search.

## Principle

**message_stream is the single owner of conversation data.** Everything needed to reconstruct what happened in a conversation lives in one table. The encoding agent reads from it — no joins, no separate files, no re-computation.

## Schema Change

```sql
-- Extend message_stream with recall context
ALTER TABLE message_stream ADD COLUMN recalled_node_ids TEXT;   -- JSON: ["64f55107", "daeb9fa6"]
ALTER TABLE message_stream ADD COLUMN recalled_raw TEXT;         -- JSON: [{id, type, title, score}, ...]
ALTER TABLE message_stream ADD COLUMN recalled_distilled TEXT;   -- Haiku output for this turn
```

Each user message row gets the recall data that was computed for it. Assistant message rows leave these NULL.

## Data Flow

### Current (v2):
```
UserPromptSubmit
  → hook_recall → brain.recall() → candidates file → Haiku distiller → context to Claude
  → (candidates file overwritten next turn, distilled stored in hook_log only)

Stop (every 5th)
  → encoding agent starts BLIND
  → Round 1-3: agent does its own recalls (~4 searches)
  → Round 4-5: agent writes (revise/create/connect)
  → Round 6: agent summarizes (wasted)
  = 6 rounds, 110K tokens, ~$0.37, ~70 seconds
```

### Proposed (v3):
```
UserPromptSubmit
  → hook_recall → brain.recall() → candidates file → Haiku distiller → context to Claude
  → store_exchange() saves: user msg + recalled_node_ids + recalled_raw + recalled_distilled

Stop (every 5th)
  → encoding agent receives pre-attached recall for ALL 10 turns
  → Round 1: agent reads turns, decides what to encode (may get_node for details)
  → Round 2: agent writes (revise/create/connect), responds DONE
  = 2 rounds, ~45K tokens, ~$0.15, ~25 seconds
```

## Prompt Format

The encoding agent's user content is formatted chronologically:

```
## ENCODING RUN #50

### Previous State
(what the agent encoded last run)

### Conversation Timeline

[TURN 1]
USER: "why is the daemon crashing?"
BRAIN SURFACED (3 nodes):
  [lesson] Daemon CPU spiral: three root causes (id:64f55107, score:0.89)
  [mechanism] Daemon TCP migration (id:daeb9fa6, score:0.72)
  [decision] Daemon is additive — hooks try daemon first (id:009cc3e8, score:0.65)
DISTILLED TO CLAUDE: "The daemon had 3 root causes: watchdog thread leak, SQLite deadlock, unbounded encoding threads. Previous session fixed all three."
ASSISTANT: "Let me investigate the daemon. I'll check the logs..."

[TURN 2]
USER: "its still at 800% CPU"
BRAIN SURFACED (2 nodes):
  [lesson] Daemon CPU spiral (id:64f55107, score:0.91)
  [lesson] Stale daemons survive SIGTERM (id:stale123, score:0.68)
DISTILLED TO CLAUDE: "Previous session fixed this — check if the old daemon was running pre-fix code."
ASSISTANT: "The old daemon was running with HTTP MCP enabled..."

...
```

### What this tells the agent:
- **What Tom said** (user messages)
- **What the brain knew** at each point (recalled nodes with IDs — actionable)
- **What Claude was told** (distilled output — what shaped the response)
- **What Claude responded** (assistant messages)

The agent can see: "The brain surfaced node X but it was stale — Tom corrected it in the next message. I should revise node X."

## System Prompt Changes

Add to encoding-agent.md:

```
## Pre-attached Recall

Each turn in the conversation includes what the brain surfaced at that moment.
You don't need to recall blindly — the data is already here.

Use `recall` or `get_node` only when you need:
- Deeper content of a specific node (the pre-attached data has title + score, not full content)
- Nodes on a topic NOT covered in any turn's recall

When you have no more actions, respond with just "DONE" — do not explain
or summarize what you encoded.
```

## Tool Reduction

The agent used only 3 tools in the traced run: `recall`, `get_node`, `revise`.

Reduce from 13 tools to 7:
- **Read:** `recall`, `get_node`, `find_node_by_title`
- **Write:** `remember`, `revise`, `connect`, `learn_vocabulary`

Drop specialized remember variants (`remember_lesson`, `remember_mechanism`, etc.) — the base `remember` with free-text type handles all cases under the v2 prompt.

Saves: ~1,200 tokens per round from tool schemas.

## Implementation Steps

### Step 1: Schema migration
Add 3 columns to message_stream in brain_logs.db.

### Step 2: store_exchange() captures recall data
In daemon_hooks.py hook_recall, pass recalled candidates to store_exchange().
The candidates are already in the /tmp JSON file — read them before writing the message.

### Step 3: Encoding agent prompt builder
New function in encoding_agent.py: `_build_timeline_content(brain, session_id)`.
Fetches last 10 messages with recalled data, formats as [TURN N] blocks.

### Step 4: Reduce tools
Remove specialized remember variants from ENCODING_TOOLS set.

### Step 5: Update system prompt
Add pre-attached recall explanation. Add "respond DONE" instruction.

## Expected Impact

| Metric | v2 (current) | v3 (proposed) |
|--------|-------------|---------------|
| Rounds | 6 | 2 |
| Input tokens | 110K | 45K |
| Cost per run | $0.37 | $0.15 |
| Time | ~70s | ~25s |
| Daily cost (10 runs) | $3.70 | $1.50 |
| Monthly cost | $81 | $33 |

## Risks

1. **Pre-attached recall may be stale** — if a node was revised between turn 1 and turn 10, the pre-attached data shows the old version. The agent can `get_node` to check.

2. **Distilled output may omit important nodes** — Haiku filters aggressively. The raw node IDs + scores provide full fidelity as backup.

3. **message_stream grows** — ~1K extra per user message (node IDs + distilled). At 50 messages/day = 50K/day = ~1.5MB/month. Negligible.

4. **Prompt format complexity** — the [TURN N] format is longer than raw messages. But it replaces the agent's own recall rounds, so net smaller.

## Not in Scope (future)

- Vocab/concept layer redesign (separate design doc)
- Entity extraction (separate process, feeds into message context)
- In-memory cache (DAL abstraction ready, FalkorDB evaluation pending)
