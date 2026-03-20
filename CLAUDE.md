# Brain Plugin — Claude Instructions

## Architecture

Brain is a **serverless Python module** (`servers/brain.py`). There is NO HTTP server, no curl commands, no port 7437.

All operations are Python method calls on a `Brain` object:
```python
brain.remember(type="decision", title="...", content="...", locked=True)
brain.recall_with_embeddings("query", limit=10)
brain.connect(source_id, target_id, "relation", weight=0.7)
brain.save()
```

## Hooks Handle Everything

**Do NOT manually run boot scripts, resolve DB paths, or import the Brain module.**

The plugin has 13 hooks that fire automatically:
- `SessionStart` → `boot-brain.sh` — boots brain, prints context + consciousness signals
- `UserPromptSubmit` → `pre-response-recall.sh` — recalls relevant memories before responding
- `Notification(user_message)` → `pre-response-recall.sh` + `post-response-track.sh` — recall + vocab gap detection
- `Notification(idle_prompt)` → `idle-maintenance.sh` — consolidation, healing, reflection
- `PreToolUse(Edit|Write)` → `pre-edit-suggest.sh` — surfaces rules before file edits
- `PreCompact` → `pre-compact-save.sh` — saves brain before context loss
- `PostCompact` → `post-compact-reboot.sh` — re-boots context after compaction
- `Stop` → `post-response-track.sh` — captures session activity
- `SessionEnd` → `session-end.sh` — session synthesis + save

If a hook fails, check the boot output for error messages. The brain logs errors to `brain_logs.db` and surfaces them as consciousness signals.

## DB Location

The boot hook resolves the DB automatically via `resolve-brain-db.sh`:
1. `BRAIN_DB_DIR` env var (explicit override)
2. `/sessions/*/mnt/AgentsContext/brain/` (Cowork mounts)
3. `$HOME/AgentsContext/brain/` (local, typically a symlink to Google Drive)

The actual DB path is printed in the boot output. Do not guess paths.

## What To Do Each Session

1. **Read the boot output.** It contains session context, locked rules, consciousness signals (fading knowledge, tensions, vocabulary gaps, errors, mental model drift).
2. **If starting from a compaction summary**, encode the delta into the brain (Step 2a in SKILL.md). This is mandatory — compacted knowledge is lost forever if not encoded.
3. **Use the brain throughout** — remember decisions, recall context, connect related nodes.
4. **Let hooks do their job** — don't manually call suggest before edits (the PreToolUse hook does it), don't manually save (hooks save at compaction and session end).

## Common Mistakes

- Using `curl` commands — there is no HTTP server
- Manually running `boot-brain.sh` — the SessionStart hook does this
- Constructing DB paths by guessing — read the boot output or use `resolve-brain-db.sh`
- Importing Brain without the right sys.path — use `BRAIN_SERVER_DIR` env var set by hooks

## When You Need Direct Brain Access

If you must call brain methods directly (e.g., to encode learnings):
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.environ.get('BRAIN_SERVER_DIR', '')))
from servers.brain import Brain
brain = Brain(os.path.join(os.environ.get('BRAIN_DB_DIR', ''), 'brain.db'))
# ... do work ...
brain.save()
brain.close()
```

## Key v5 Features

- **12 consciousness signals** — fading knowledge, tensions, hypotheses, vocabulary gaps, silent errors, mental model drift, uncertain areas, dream insights, reminders, host changes, novelty, recent encodings
- **Engineering memory types** — purpose, mechanism, impact, constraint, convention, lesson, vocabulary, mental_model
- **Confidence-weighted recall** — node confidence (0.1-1.0) affects recall ranking
- **Vocabulary system** — context-dependent term mappings, gap detection, auto-connection
- **Error logging** — `_log_error()` writes to brain_logs.db, surfaced via consciousness
- **Self-correction traces** — divergence recording, pattern extraction, impact cross-reference
- **Session synthesis** — auto-generated session summaries with reflection prompts
