# Brain Plugin — Claude Instructions

## How to Talk to the Brain

Use the **MCP tools** (`mcp__plugin_brain_brain__*`). They are your primary interface. The boot hook starts the daemon automatically — you don't need to import Python modules, construct paths, or run bash scripts.

**Core MCP tools:**
- `recall` — semantic search by meaning (query, limit)
- `remember` — store a node (type, title, content, keywords, locked, confidence, project)
- `connect` — create edge between nodes (source_id, target_id, relation, weight)
- `consciousness` — get all consciousness signals
- `context_boot` — full brain boot (stats, locked rules, signals)
- `eval` — escape hatch for any brain method not exposed as a tool
- `set_config` / `get_config` — brain configuration
- `health_check`, `save`, `ping`, `engineering_context`

**For specialized methods** (remember_lesson, remember_impact, record_divergence, learn_vocabulary, etc.), use the `eval` tool:
```
eval: brain.remember_lesson(title="...", what_happened="...", root_cause="...", fix="...", preventive_principle="...")
```

**Do NOT:**
- Write Python scripts in Bash to call brain methods
- Import Brain or use sys.path
- Construct DB paths or use BRAIN_DB_DIR/BRAIN_SERVER_DIR directly
- Use curl commands — there is no HTTP server

## Hooks Handle Everything

**Do NOT manually run boot scripts, resolve DB paths, or import the Brain module.**

The plugin has 13 hooks that fire automatically:
- `SessionStart` → `boot-brain.sh` — boots brain + daemon, prints context + consciousness signals
- `UserPromptSubmit` → `pre-response-recall.sh` — recalls relevant memories before responding
- `Notification(user_message)` → `pre-response-recall.sh` + `post-response-track.sh` — recall + vocab gap detection
- `Notification(idle_prompt)` → `idle-maintenance.sh` — consolidation, healing, reflection
- `PreToolUse(Edit|Write)` → `pre-edit-suggest.sh` — surfaces rules before file edits
- `PreCompact` → `pre-compact-save.sh` — saves brain before context loss
- `PostCompact` → `post-compact-reboot.sh` — re-boots context after compaction
- `Stop` → `post-response-track.sh` — captures session activity
- `SessionEnd` → `session-end.sh` — session synthesis + save

If a hook fails, check the boot output for error messages. The brain logs errors to `brain_logs.db` and surfaces them as consciousness signals.

## Brain-to-Operator Channel (2026-03-22)

### Why this exists

The brain has consciousness — dreams, tensions, reminders, health alerts — but the human operator never saw any of it. Claude Code hooks can inject text into YOUR context via `additionalContext`, but there is NO working channel that renders text directly to the human in the terminal. We tested `systemMessage`, stderr, `stopReason`, and `reason` — all invisible. (See brain node "Failed hook-to-operator experiments" for details.)

**The solution:** You are the relay. Both channels — yours and the operator's — are multiplexed through the single working pipe (`additionalContext`), tagged so you know which is which.

### What you'll see in your context

```
[BRAIN]
(recall results, graph activity, consciousness signals — for YOUR reasoning)
[/BRAIN]

[BRAIN-To-Tom]
@priority: high
🔔 REMINDER DUE: Ship the feature (due 2026-03-22T09:00)

@priority: medium
⚡ Tension: React hooks rule vs Hook Violation pattern

@priority: low
💭 Dream: Frontend polling ↔ Growing signal density — 12 hops apart
[/BRAIN-To-Tom]
```

### What you MUST do

When you see `[BRAIN-To-{name}]` tags, **relay that content to the operator in your response**:

- **`@priority: high`** → Lead your response with this. Reminders, errors, health alerts. Don't bury it. These are things the human set or needs to know NOW.
- **`@priority: medium`** → Mention naturally in your response. Tensions, stale reasoning, hypotheses to validate. These affect reasoning quality.
- **`@priority: low`** → Weave in if relevant to what you're discussing. Dreams, novelty, fading knowledge. Skip if it would feel forced.

Rules:
- Do NOT summarize or filter high-priority items — relay faithfully
- You CAN contextualize and make it conversational (don't just dump raw text)
- If no `[BRAIN-To-*]` section exists, there's nothing to relay — don't mention the channel
- `[DEBUG]` sections within operator channel: present as-is, formatted for readability

### What breaks if you ignore this

If you absorb `[BRAIN-To-*]` content silently and never mention it, the brain goes mute. The human sees nothing. Reminders are missed. Tensions go unnoticed. This section is the most fragile link in the chain — it depends entirely on you following through.

### Where the plumbing lives (don't modify without understanding)

| Component | File | What it does |
|---|---|---|
| `wrap_for_hook()` | `servers/brain_voice.py` | Single wrapping point — merges for_claude + for_operator into tagged output |
| `render_operator_prompt()` | `servers/brain_voice.py` | Curates what the operator sees — reminders, tensions, dreams with @priority |
| `_operator_boot_summary()` | `servers/brain_voice.py` | Boot-time operator content with consciousness highlights |
| `hook_recall()` | `servers/daemon_hooks.py` | Calls wrap_for_hook, puts merged result in additionalContext |
| `hook_post_compact_reboot()` | `servers/daemon_hooks.py` | Same pattern for post-compaction |
| `format_boot_context()` | `servers/brain_surface.py` | Returns merged string (backward compat wrapper) |

**Do NOT** put anything in `systemMessage` — it's a dead channel that doesn't render. All operator content goes through `wrap_for_hook()` into `additionalContext`.

## Architecture (for reference, not for direct use)

Brain is a Python module (`servers/brain.py`) behind a persistent daemon (`servers/daemon.py`) that keeps the embedder loaded in memory. The MCP server (`servers/brain_mcp.py`) is a thin stdio proxy to the daemon via Unix socket. Claude Code starts the MCP server automatically from `.mcp.json`.

```
Claude Code → MCP server (brain_mcp.py, stdio) → daemon (Unix socket) → Brain + embedder
```

The boot hook resolves the DB automatically:
1. `BRAIN_DB_DIR` env var (explicit override)
2. `/sessions/*/mnt/AgentsContext/brain/` (Cowork mounts)
3. `$HOME/AgentsContext/brain/` (local, typically a symlink to Google Drive)

## Brain + Documents: Division of Labor

**The brain is associative memory, not a document store.** Don't encode formal plans, task lists, or specs as brain nodes. Those belong in markdown files in the repo.

| Use documents for | Use brain for |
|---|---|
| Task lists, refactoring targets | Lessons learned (WHY, not WHAT) |
| Architecture specs, API refs | Corrections and self-correction traces |
| Anything needing guaranteed retrieval | Vocabulary, cross-session patterns |
| Anything 1-10 pages read top-to-bottom | Associative links, confidence signals |

**Brain nodes should POINT TO documents**, not duplicate them. Example: a brain node says "refactoring targets are in REFACTORING.md — top priority is mark_recall_used because it blocks the precision feedback loop." The node holds the *why* and the *pointer*. The document holds the *what*.

**Active documents:**
- `REFACTORING.md` — current cleanup targets with priorities and status
- `docs/HOOKS-ARCHITECTURE.md` — ⚠️ DEPRECATED (2026-03-22). Hook output format changed — see "Brain-to-Operator Channel" below. Keep for historical reference only.

## CRITICAL: SKILL.md Is Your Operating Manual (2026-03-22)

**`skills/brain/SKILL.md` is the brain's encoding playbook. Without it, the brain cannot grow.**

At session start, verify the skill is loaded. If you don't see it in your available skills, the brain is running blind — encoding quality will be poor, vocabulary won't be captured, connections won't be made.

The skill defines:
- The **4-Step Encoding Checklist** (uncertainty → facts → connections → vocabulary) — follow it after every significant exchange
- **Encoding quality scoring** — aim for 8+/10 on every node
- **Known failure modes** — compression instinct, agreeability, opinion hallucination
- **API quick reference** — every brain method and when to use it

**If you skip the checklist, the brain degrades. Every unencoded decision is lost forever.**

## What To Do Each Session

1. **Verify SKILL.md is loaded.** This is non-negotiable. The skill defines how to encode.
2. **Read the boot output.** It contains session context, locked rules, consciousness signals (fading knowledge, tensions, vocabulary gaps, errors, mental model drift).
3. **If starting from a compaction summary**, encode the delta into the brain (Step 2a in SKILL.md). This is mandatory — compacted knowledge is lost forever if not encoded.
4. **Use the 4-step checklist throughout** — uncertainty, facts, connections, vocabulary. Don't batch encoding at the end.
5. **Let hooks do their job** — don't manually call suggest before edits (the PreToolUse hook does it), don't manually save (hooks save at compaction and session end).
6. **Check REFACTORING.md** if doing cleanup work. Pick one target. Don't boil the ocean.
7. **One refactor per session.** Commit before compaction. Update REFACTORING.md when done.

## Benchmark-First Rule for Sacred Systems (2026-03-22)

**Before changing any sacred system, build the test harness FIRST. Benchmark with real-world cases. Only ship after benchmarks prove no regression.**

Sacred systems (NEVER modify without benchmarks):
- **Embedding pipeline** — `servers/embedder.py`
- **Recall pipeline** — `servers/brain_recall.py`, `recall_scorer.py`
- **Encoding pipeline** — `servers/brain_remember.py`, `brain_engineering.py`
- **Precision pipeline** — `servers/brain_precision.py`
- **Hook output format** — `servers/brain_voice.py` `wrap_for_hook()`

The process:
1. Build test corpus from real conversations (engineering, philosophy, science, personal, adversarial)
2. Run current system against corpus → capture **baseline metrics**
3. Implement candidate changes (option A, option B)
4. Run ALL options against the **same corpus** → compare
5. Pick winner based on data. Only then modify production code.

Eval framework: `tests/golden_dataset.json` (60+ cases), `tests/eval_runner.py` (NDCG/MRR/precision@k), `tests/run_tests.py --golden`

See `docs/GROWTH-PLAN.md` for the full testing strategy.

## Test Integrity Rule

**When a test fails, STOP. Do not change the test OR the code to make it pass.** This is a hard rule.

If a test fails:
1. **Stop.** Do not change anything.
2. **Report to the user:** what the test expected vs what the code returned.
3. **Ask:** "Is the test expectation wrong, or does the code have a bug?"
4. **Wait for the answer** before proceeding.

This applies in BOTH directions:
- **Do NOT weaken the test** — changing `assertEqual(0.7)` to `assertGreater(0)`, removing assertions, adding `try/except` around assertions.
- **Do NOT "fix" the code to satisfy a test** — the test might be wrong, or the "fix" might break something else. Only the user can decide which side is correct.

Exception: tests you JUST wrote in the same session that fail on first run — you may fix those since they have no history. But if an EXISTING test that was previously passing starts failing, that's a regression signal. Stop and report.

If you must adjust a test or code after user approval, add a comment explaining why:
```python
# ADJUSTED: weight is 0.3 not 0.7 because Hebbian learning applies on connect()
# Confirmed with Tom 2026-03-21 — this is expected behavior
self.assertAlmostEqual(edge[2], 0.3, places=1)
```

## Common Mistakes

- Using Python/bash to call brain methods when MCP tools are available
- Using `curl` commands — there is no HTTP server
- Manually running `boot-brain.sh` — the SessionStart hook does this
- Constructing DB paths by guessing — read the boot output
- Importing Brain or using sys.path — use MCP tools instead

## Key v5 Features

- **12 consciousness signals** — fading knowledge, tensions, hypotheses, vocabulary gaps, silent errors, mental model drift, uncertain areas, dream insights, reminders, host changes, novelty, recent encodings
- **Engineering memory types** — purpose, mechanism, impact, constraint, convention, lesson, vocabulary, mental_model
- **Confidence-weighted recall** — node confidence (0.1-1.0) affects recall ranking
- **Vocabulary system** — context-dependent term mappings, gap detection, auto-connection
- **Error logging** — `_log_error()` writes to brain_logs.db, surfaced via consciousness
- **Self-correction traces** — divergence recording, pattern extraction, impact cross-reference
- **Session synthesis** — auto-generated session summaries with reflection prompts
