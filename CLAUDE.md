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
- `docs/HOOKS-ARCHITECTURE.md` — hook system design (v5.3+)

## What To Do Each Session

1. **Read the boot output.** It contains session context, locked rules, consciousness signals (fading knowledge, tensions, vocabulary gaps, errors, mental model drift).
2. **If starting from a compaction summary**, encode the delta into the brain (Step 2a in SKILL.md). This is mandatory — compacted knowledge is lost forever if not encoded.
3. **Check REFACTORING.md** if doing cleanup work. Pick one target. Don't boil the ocean.
4. **Use the brain throughout** — remember decisions, recall context, connect related nodes.
5. **Let hooks do their job** — don't manually call suggest before edits (the PreToolUse hook does it), don't manually save (hooks save at compaction and session end).
6. **One refactor per session.** Commit before compaction. Update REFACTORING.md when done.

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
