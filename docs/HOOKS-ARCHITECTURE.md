# Brain Hooks Architecture — Complete Reference

**Version:** v5.3.1 (2026-03-20) — Daemon consolidation + graph change tracking
**Status:** Authoritative — READ THIS BEFORE TOUCHING ANY HOOK

This document describes every hook in the brain plugin system: what event triggers it, what it does, what brain methods it calls, what output format it uses, whether its stdout is visible to Claude, what tables it reads/writes, and why it exists. It also covers the shared infrastructure (hook_common.py, resolve-brain-db.sh, the daemon, the consciousness layer) and the output visibility model.

---

## Table of Contents

1. [Output Visibility Model](#1-output-visibility-model)
2. [Execution Architecture](#2-execution-architecture)
3. [Shared Infrastructure](#3-shared-infrastructure)
4. [Hook Reference (all 14 hooks)](#4-hook-reference)
5. [Database Tables](#5-database-tables)
6. [Daemon Command Reference](#6-daemon-command-reference)
7. [Consciousness & Awareness Layer](#7-consciousness--awareness-layer)
8. [Error Handling Architecture](#8-error-handling-architecture)
9. [Pending Message Pattern](#9-pending-message-pattern)
10. [Common Pitfalls & Failure Modes](#10-common-pitfalls--failure-modes)

---

## 1. Output Visibility Model

This is the most important thing to understand. **Not all hook stdout reaches Claude.** The Claude Code harness only injects stdout into Claude's context for specific event types. If you write output in the wrong hook, it vanishes silently.

### Events where stdout IS injected into Claude's context:

| Event | Hook | Output Format | How it reaches Claude |
|-------|------|---------------|----------------------|
| `SessionStart` | boot-brain.sh | Free text (printed lines) | Injected as session start context |
| `UserPromptSubmit` | pre-response-recall.sh | `{"additionalContext":"..."}` JSON | `additionalContext` field injected before Claude responds |
| ~~`UserPromptSubmit`~~ | ~~post-response-track.sh~~ | ~~moved to Stop only (Session #9)~~ | Fired before Claude responded → 94% of precision evals empty |
| `PostCompact` | post-compact-reboot.sh | Free text (printed lines) | Injected as post-compaction context |
| `PreToolUse(Edit\|Write)` | pre-edit-suggest.sh | `{"decision":"approve","reason":"..."}` JSON | `reason` field shown as tool-use metadata |
| `PreToolUse(Bash)` | pre-bash-safety.sh | `{"decision":"approve\|block","reason":"..."}` JSON | `reason` shown; `block` prevents execution |

### Events where stdout is INVISIBLE to Claude:

| Event | Hook | Why invisible | How output reaches Claude instead |
|-------|------|---------------|-----------------------------------|
| `Notification(idle_prompt)` | idle-maintenance.sh | Notification stdout is not injected | Stores as `pending_hook_messages` in brain_meta → drained by next UserPromptSubmit recall |
| `Stop` | post-response-track.sh | Stop stdout is not injected | Stores encoding checkpoint as `pending_hook_messages` |
| `StopFailure` | stop-failure-log.sh | StopFailure stdout not injected | Logs to miss_log table; surfaced via consciousness signals |
| `SessionEnd` | session-end.sh | SessionEnd stdout not injected | Session synthesis stored in session_syntheses table; surfaced at next boot |
| `PreCompact` | pre-compact-save.sh | PreCompact only reads decision JSON | Prints `{"decision":"approve"}` (must not block compaction) |
| `ConfigChange` | config-change-host.sh | ConfigChange stdout not injected | Stores as `pending_hook_messages` |
| `PostToolUse(Bash)` | post-bash-host-check.sh | PostToolUse stdout not injected | Stores as `pending_hook_messages` |
| `WorktreeCreate` | worktree-context.sh | WorktreeCreate stdout not injected | Sets config keys; output printed but may not reach Claude |
| `WorktreeRemove` | worktree-cleanup.sh | WorktreeRemove stdout not injected | Clears config keys only |

### Key insight: the pending message pattern

Background hooks that generate output Claude should see use `store_pending_message()` to queue messages. The next `UserPromptSubmit` hook (`pre_response_recall.py`) calls `drain_pending_messages()` and includes them in its `additionalContext` output. This is the bridge between invisible hooks and Claude's context.

---

## 1b. Registration Requirement — `.claude/settings.json`

**Hook scripts do nothing unless registered in `.claude/settings.json`.** Claude Code only runs hooks that are declared in the `hooks` section of `settings.json`. The scripts in `hooks/scripts/` are just files on disk — they are not discovered automatically.

All 14 hooks are registered in the project-level `.claude/settings.json` with their event types, matchers, and timeouts. If you add a new hook script, you must also add its entry to `settings.json` or it will never fire.

**Do not modify `.claude/settings.json` without operator approval.** This file controls what code runs on every Claude Code event. Changes here affect security (pre-bash-safety), data integrity (pre-compact-save), and session continuity (post-compact-reboot).

---

## 2. Execution Architecture

### The thin bash shim pattern

Every hook follows the same structure. The bash script (`.sh`) is a thin shim that:
1. Sources `resolve-brain-db.sh` to set env vars
2. Captures stdin (hook input from Claude Code) into `$HOOK_INPUT`
3. Execs into the Python script

```bash
#!/bin/bash
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && exit 0
export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/script_name.py"
```

**Why this pattern exists:** Previously, hooks had embedded `python3 -c '...'` blocks inside bash. Apostrophes in Python strings would break the bash quoting, causing silent failures. Extracting to separate `.py` files eliminates this class of bug permanently.

### Centralized daemon architecture (v5.3+)

All hook logic lives in `servers/daemon_hooks.py`. Each hook .py file is a **thin client** (~15-30 lines) that:

1. Applies local pre-screen guards (e.g., regex for destructive commands, short message skip)
2. Sends a single `hook_*` command to the daemon
3. Prints the result

```
if daemon_available():
    resp = daemon_call_raw("hook_recall", args)     # Fast (~50-200ms)
    print(json.dumps(resp["result"]["json"]))
else:
    from servers.daemon_hooks import hook_recall     # Direct fallback (~2-3s)
    result = hook_recall(brain, args, [])
    print(json.dumps(result["json"]))
```

**Why centralized:** Previously each hook had separate `_run_daemon()` and `_run_direct()` paths that duplicated logic and diverged over time (e.g., direct recall had priming/aspirations/tensions that daemon path lacked). Now `daemon_hooks.py` is the single source of truth. Both daemon and direct fallback execute the same function.

**Graph change tracking:** The daemon maintains an in-memory `graph_changes` list. Every mutation (remember, connect, dream, heal, consolidate, worktree) appends a description. `hook_recall` drains this list and surfaces it as "GRAPH ACTIVITY" in the recall output, giving Claude visibility into what changed between prompts.

The daemon (`servers/daemon.py`) keeps the Brain instance and embedding model loaded in memory. Boot starts the daemon as a background process. Subsequent hooks connect via Unix socket at `/tmp/brain-daemon-{uid}.sock` for sub-100ms responses. If the daemon isn't running (crashed, never started), hooks fall back to importing the hook function directly from `daemon_hooks.py` with a fresh Brain instance.

### Environment variables set by resolve-brain-db.sh

| Variable | Purpose | Resolution chain |
|----------|---------|-----------------|
| `BRAIN_DB_DIR` | Path to directory containing brain.db | 1. Existing env var, 2. `/sessions/*/mnt/AgentsContext/brain/`, 3. `$HOME/AgentsContext/brain/` |
| `BRAIN_SERVER_DIR` | Path to `servers/` directory | `$PLUGIN_ROOT/servers` |
| `PLUGIN_ROOT` | Path to brain plugin root | `$CLAUDE_PLUGIN_ROOT` or auto-detected from script location |

### Timeout budget

Each hook has a timeout in hooks.json. If the hook exceeds it, Claude Code kills the process. The hook output is lost, and Claude proceeds without it. This is why graceful degradation matters — every hook must either complete within budget or fail silently.

| Hook | Timeout | Notes |
|------|---------|-------|
| boot-brain.sh | 15s | Longest — loads embedder, computes consciousness |
| pre-response-recall.sh | 5s | Must be fast — fires every prompt |
| post-response-track.sh | 5s | Stop only (Session #9: removed from UserPromptSubmit) |
| pre-edit-suggest.sh | 8s | Can be slow if many suggestions |
| pre-bash-safety.sh | 8s | Fast regex pre-screen; only slow if destructive |
| idle-maintenance.sh | 30s | Long — dream, consolidate, heal, tune |
| pre-compact-save.sh | 10s | Synthesis + save |
| post-compact-reboot.sh | 10s | Full context re-injection |
| session-end.sh | 10s | Synthesis + consolidate + save |
| stop-failure-log.sh | 5s | Quick log entry |
| config-change-host.sh | 5s | Host scan |
| post-bash-host-check.sh | 5s | Host scan after env-changing commands |
| worktree-context.sh | 5s | Git branch detection + config |
| worktree-cleanup.sh | 5s | Config clear |

---

## 3. Shared Infrastructure

### hook_common.py — the shared module

Every `.py` hook script imports from `hook_common.py`. It provides:

#### Path setup (module-level)
- `server_dir` — from `$BRAIN_SERVER_DIR`
- `db_dir` — from `$BRAIN_DB_DIR`
- `db_path` — `{db_dir}/brain.db`
- Adds parent of `server_dir` to `sys.path` so `from servers.brain import Brain` works

#### Functions

| Function | Purpose | Returns | Used by |
|----------|---------|---------|---------|
| `get_hook_input()` | Parse `$HOOK_INPUT` env var as JSON | `dict` (empty dict on failure) | All hooks that receive stdin |
| `get_brain()` | Import Brain class and instantiate with db_path | `Brain` instance or `None` | All hooks (direct path) |
| `close_brain(brain)` | Safely close brain connection | None | All hooks (direct path) |
| `daemon_available()` | Check if daemon Unix socket exists | `bool` | All hooks (path selection) |
| `daemon_call(cmd, args, timeout)` | Send command to daemon, return result | `dict` (result only, empty on failure) | Hooks that need simple daemon calls |
| `daemon_call_raw(cmd, args, timeout)` | Send command to daemon, return full response including `ok` field | `dict` (full response with `ok` field) | Hooks that need to check success/failure |
| `store_pending_message(brain_or_daemon, message)` | Queue a message for next UserPromptSubmit to surface | None | Background/invisible hooks |
| `drain_pending_messages(brain_or_daemon)` | Read and clear all pending messages | `list[str]` | pre_response_recall.py, post_compact_reboot.py |
| `log_hook_error(source, error, context, level)` | Log error to brain_logs.db AND stderr | None | All hooks (error handling) |
| `get_unsurfaced_hook_errors(limit)` | Read unsurfaced errors from brain_logs.db | `list[dict]` | Consciousness layer |
| `mark_hook_errors_surfaced(error_ids)` | Mark errors as surfaced so they don't repeat | None | Consciousness layer |

#### Error logging architecture

`log_hook_error()` writes to two places:
1. **stderr** — always, immediately (never silent)
2. **brain_logs.db → hook_errors table** — persistent storage for consciousness layer to pick up

The hook_errors table schema:
```sql
CREATE TABLE IF NOT EXISTS hook_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    hook_name TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'error',
    error TEXT NOT NULL,
    context TEXT DEFAULT '',
    traceback TEXT DEFAULT '',
    surfaced INTEGER DEFAULT 0
)
```

This table is in `brain_logs.db` (separate from `brain.db`) because hook errors can occur when Brain itself fails to import. Using direct SQLite ensures errors are logged even during catastrophic failures.

### resolve-brain-db.sh — DB path resolution

Sources by every bash shim. Resolution chain:
1. `$BRAIN_DB_DIR` already set and contains brain.db → use it (fast path for non-boot hooks)
2. `/sessions/*/mnt/AgentsContext/brain/brain.db` → Cowork mounted volumes
3. `$HOME/AgentsContext/brain/brain.db` → local (typically symlink to Google Drive)
4. If none found → `$BRAIN_DB_DIR` is empty, shim exits 0

### The daemon (servers/daemon.py)

A Unix socket server that keeps Brain + embedder loaded in memory. Started by boot hook as a background process.

- **Socket:** `/tmp/brain-daemon-{uid}.sock`
- **PID file:** `/tmp/brain-daemon-{uid}.pid`
- **Protocol:** JSON over newline-delimited Unix socket. Request: `{"cmd":"...", "args":{...}}\n`. Response: `{"ok":true, "result":{...}}\n` or `{"ok":false, "error":"..."}\n`
- **Lifecycle:** Started at boot, stays running across hooks within a session, cleaned up at session end (`shutdown` command) or via PID file in test teardown

---

## 4. Hook Reference

### 4.1 boot-brain.sh → boot_brain.py

**Event:** `SessionStart`
**Matcher:** (none — fires for all sessions)
**Timeout:** 15s
**Output visible:** YES — stdout injected as session start context
**Output format:** Free text (multi-line brain state report)

**Purpose:** Initialize the brain for this session. Print full context so Claude understands the brain's state, history, rules, consciousness signals, and developmental guidance. This is the MOST IMPORTANT hook — it defines Claude's starting context.

**What it does (in order):**
1. Starts the daemon as a background process (`ensure_daemon()`)
2. Imports Brain, opens DB
3. Calls `brain.reset_session_activity()` — clears session counters
4. Calls `brain.validate_config()` — checks for config issues
5. Resolves user/project from env or stored config
6. Calls `brain.context_boot(user, project, task)` — gets locked nodes, total counts, last session note
7. Calls `brain.get_engineering_context(project)` — purposes, constraints, conventions, file inventory, vocabulary, impacts
8. Calls `brain.get_correction_patterns(limit=5)` — recurring divergence patterns
9. Calls `brain.get_last_synthesis()` — last session's decisions, corrections, open questions
10. Calls `brain.health_check(auto_fix=True)` — integrity checks, auto-repair
11. Calls `brain.list_staged()` / `brain.auto_promote_staged()` — staged learnings management
12. Calls `brain.get_suggest_metrics()` — pre-edit suggestion stats
13. Calls `brain.procedure_trigger("session_start")` — triggered procedures
14. Calls `brain.get_consciousness_signals()` — ALL 25+ consciousness signals (the full scan)
15. Calls `brain.assess_developmental_stage()` — brain maturity assessment
16. Calls `brain.scan_host_environment()` — detect environment changes
17. Calls `brain.get_surfaceable_dreams()` — cross-cluster bridge insights
18. Calls `brain.auto_generate_self_reflection()` — periodic self-assessment
19. `brain.save()` — persist any changes from the above
20. Prints structured output: version, session number, last session, engineering context, correction patterns, vocabulary, constraints, health alerts, locked rules, staged learnings, consciousness signals (25+ types), developmental stage guidance, brain stats, embedder status, suggest metrics

**Brain methods called:** `reset_session_activity`, `validate_config`, `get_config`, `context_boot`, `get_engineering_context`, `get_correction_patterns`, `get_last_synthesis`, `health_check`, `list_staged`, `auto_promote_staged`, `get_suggest_metrics`, `procedure_trigger`, `get_consciousness_signals`, `assess_developmental_stage`, `scan_host_environment`, `get_surfaceable_dreams`, `auto_generate_self_reflection`, `get_debug_status`, `log_debug`, `save`, `close`

**Tables read:** nodes, edges, brain_meta, session_syntheses, correction_traces, node_metadata, node_embeddings, projects, project_maps, session_activity, staged_learnings, bridge_proposals, brain_logs.db→hook_errors, brain_logs.db→error_log
**Tables written:** brain_meta (session config), session_activity (reset), nodes (staged promotion)

**Consciousness signals surfaced (25+ types):**
- `reminders` — due reminders
- `evolutions` — active tensions, hypotheses, patterns, aspirations
- `fluid_personal` — personal nodes that may have changed
- `fading` — important nodes untouched for 14+ days
- `stale_context_count` — old context nodes needing cleanup
- `failure_modes` — active failure mode nodes
- `performance` — recent performance observations
- `capabilities` — capability nodes
- `interactions` — interaction observation nodes
- `meta_learning` — meta-learning method nodes
- `novelty` — new concept nodes from current session
- `miss_trends` — queries that keep failing in recall
- `encoding_gap` — long session with no encoding
- `encoding_depth` — shallow encoding warning
- `encoding_bias` — type imbalance in encoding
- `session_health` — multi-dimensional session health assessment
- `density_shift` — knowledge concentration imbalance
- `emotional_trajectory` — emotion intensity trending up/down
- `rule_contradictions` — recent nodes that may conflict with locked rules
- `stale_reasoning` — detailed rationale nodes that may be outdated
- `uncharted_code` — frequently accessed files without purpose/mechanism nodes
- `stale_file_inventory` — file inventory changes since last scan
- `vocabulary_gap` — unmapped operator terms
- `recurring_divergence` — repeated correction patterns
- `validated_approaches` — recently validated nodes
- `uncertain_areas` — unresolved uncertainty nodes
- `mental_model_drift` — mental models that may need revision
- `silent_errors` — brain errors from last 24h
- `hook_errors` — structural hook failures from brain_logs.db

---

### 4.2 pre-response-recall.sh → pre_response_recall.py

**Event:** `UserPromptSubmit`
**Matcher:** (none — fires for every user message)
**Timeout:** 5s
**Output visible:** YES — `additionalContext` injected before Claude responds
**Output format:** `{"additionalContext":"..."}` or `{"decision":"approve"}`

**Purpose:** Surface relevant brain memories before Claude responds. This is the brain's recall pipeline — the primary mechanism by which past knowledge influences current responses. Also serves as the awareness heartbeat, checking for urgent signals on every prompt.

**What it does:**
1. Parses hook input for `prompt` field
2. **Short-circuit:** Messages < 5 chars, starting with `/` or `!` → `{"decision":"approve"}` (skip recall)
3. Stores last user message in brain config (for operator voice capture)
4. **Vocabulary expansion:** Checks user message against vocabulary mappings to enrich the query (e.g., "working copy" → add "worktree")
5. **Recall:** Sends enriched query to `recall` (daemon) or `recall_with_embeddings` (direct) with limit=8
6. **Segment boundary detection** (direct path only): Checks if query represents a context shift
7. **Pending messages:** Drains `pending_hook_messages` from background hooks
8. **Urgent signals:** Calls `get_urgent_signals()` — lightweight consciousness check (hook errors, brain errors, overdue reminders)
9. **Early exit:** If no results AND no pending messages AND no urgent signals → `{"decision":"approve"}`
10. **Priming check** (direct path): Checks if query touches an active primed topic (tensions, hypotheses, open questions)
11. **Format output:** Assembles context string with sections:
    - `BRAIN AWARENESS:` — urgent signals (hook errors, brain errors, reminders)
    - `BRAIN RECALL:` — matched nodes with type, title, content, score
    - `ACTIVE EVOLUTION:` — evolution nodes (tensions, hypotheses) in results
    - Segment boundary note (if context shift detected)
    - Priming note (if query matches active concern)
    - `ASPIRATION COMPASS:` — relevant aspiration nodes
    - `HYPOTHESIS TO VALIDATE:` — hypothesis needing confirmation
    - `BRAIN AGENDA:` — active tensions
    - Instinct check nudge (if correction pattern matches)
    - `QUEUED MESSAGES:` — drained pending messages from background hooks
12. Outputs `{"additionalContext": assembled_context}`

**Brain methods called (daemon path):** `set_config`, `vocab_check`, `recall`, `urgent_signals`, `instinct_check`
**Brain methods called (direct path):** `set_config`, `resolve_vocabulary`, `recall_with_embeddings` (fallback: `recall`), `check_segment_boundary`, `add_to_segment`, `get_urgent_signals`, `get_active_primes`, `check_priming`, `get_relevant_aspirations`, `check_hypothesis_relevance`, `get_active_evolutions`, `get_instinct_check`, `save`, `close`

**Tables read:** nodes, node_embeddings, brain_meta, correction_traces, node_metadata, session_syntheses, brain_logs.db→hook_errors, brain_logs.db→error_log
**Tables written:** brain_meta (last_user_message, segment tracking), recall_log (via recall)

---

### 4.3 post-response-track.sh → post_response_track.py

**Event:** `Stop` only (Session #9: removed from UserPromptSubmit — fired before Claude responded, breaking precision loop)
**Matcher:** (none)
**Timeout:** 5s
**Output visible:** YES on UserPromptSubmit (printed text), NO on Stop (stored as pending)
**Output format:** Free text (encoding checkpoints)

**Purpose:** Two functions:
1. **Vocabulary gap detection** — scans user message for unmapped terms (quoted phrases, "the X" patterns, hyphenated terms, action targets) and stores gaps in brain config
2. **Encoding heartbeat** — rotating 5-focus checkpoint system that nudges Claude to encode knowledge during the session

**The 5-focus encoding checkpoint cycle:**
1. UNCERTAINTY — what don't you understand? Encode brain.remember_uncertainty()
2. CONNECTIONS — what connections discovered? Use brain.connect()
3. DECISIONS + LESSONS — what was decided/learned? brain.remember(type='decision')
4. BLAST RADIUS — what could break? brain.remember_impact()
5. PATTERNS — what patterns observed? brain.remember_convention()

**What it does:**
1. Determines event type (UserPromptSubmit has `prompt`, Stop has `last_assistant_message`)
2. Records message via `brain.record_message()` / daemon `record_message`
3. Gets encoding heartbeat — checks if encoding is overdue based on message count and time
4. If nudge triggered: gets current checkpoint focus, rotates index, delivers checkpoint
5. **Delivery depends on event:** UserPromptSubmit → print (visible), Stop → store_pending_message (for next session)
6. **Vocab gap detection** (direct path, UserPromptSubmit only):
   - Extracts quoted terms, "the X" patterns, action targets ("fix the auth module"), hyphenated terms
   - Filters against known vocabulary nodes and existing node titles
   - Stores new gaps in `vocabulary_gaps` config (JSON array, capped at 20)

**Brain methods called:** `record_message`, `get_encoding_heartbeat`, `get_config`, `set_config`, `save`, `close`
**Tables read:** nodes (vocabulary + title lookup), brain_meta (checkpoint_index, vocabulary_gaps)
**Tables written:** brain_meta (checkpoint_index rotation, vocabulary_gaps), session_activity (message count)

---

### 4.4 pre-edit-suggest.sh → pre_edit_suggest.py

**Event:** `PreToolUse`
**Matcher:** `Edit|Write` (only fires when Claude uses Edit or Write tools)
**Timeout:** 8s
**Output visible:** YES — `reason` field shown as tool-use metadata
**Output format:** `{"decision":"approve","reason":"..."}` or `{"decision":"approve"}`

**Purpose:** Before Claude edits a file, surface all brain knowledge relevant to that file: engineering constraints, conventions, purpose nodes, mechanism nodes, impact maps, code cognition nodes, procedures, and context files. Also checks encoding health and warns if Claude hasn't been remembering.

**What it does:**
1. Parses tool_input for file_path
2. **Skip non-source files:** .log, .map, .lock, .json (except package.json)
3. Calls `brain.pre_edit(file=filename, tool_name=tool_name)` — gets suggestions, procedures, context files
4. Gets change impacts via `brain.get_change_impact(filename)`
5. Checks encoding health via `_format_encoding_warning()`
6. **If no suggestions and no impacts and no encoding warning:** `{"decision":"approve"}`
7. **Otherwise:** Formats suggestions into categorized sections:
   - `CHANGE IMPACT WARNING:` — impact map nodes
   - `ENGINEERING MEMORY:` — purpose, mechanism, impact, constraint, convention, lesson, vocabulary nodes
   - `CODE KNOWLEDGE:` — fn_reasoning, param_influence, code_concept, arch_constraint, causal_chain, bug_lesson, comment_anchor nodes
   - `OTHER RULES & DECISIONS:` — remaining rule, decision nodes
   - `TRIGGERED PROCEDURES:` — matched procedures
   - `CONTEXT FILES:` — context files to read before editing
   - Encoding warning (if applicable)
   - Locked rule communication prompts

**Brain methods called (daemon):** `pre_edit`
**Brain methods called (direct):** `pre_edit`, `get_change_impact`, `log_debug`, `save`, `close`

**Tables read:** nodes (by file match, type, locked status), node_metadata, project_maps, brain_meta (encoding stats)
**Tables written:** brain_meta (suggest_log), suggest_log (in logs DB)

---

### 4.5 pre-bash-safety.sh → pre_bash_safety.py

**Event:** `PreToolUse`
**Matcher:** `Bash` (only fires when Claude uses Bash tool)
**Timeout:** 8s
**Output visible:** YES — `reason` shown; `block` decision prevents execution
**Output format:** `{"decision":"approve|block","reason":"..."}` or `{"decision":"approve"}`

**Purpose:** Catch destructive commands BEFORE execution. This is the safety layer — the only structural mechanism that can prevent damage. PostToolUse fires too late.

**What it does:**
1. Parses tool_input for `command` field
2. **Fast regex pre-screen** against 11 destructive patterns:
   - `rm -rf`, `rm --force`
   - `git worktree remove`
   - `git reset --hard`
   - `git clean -fd`
   - `git checkout -- `
   - `git push --force`
   - `DROP TABLE`, `DELETE FROM`, `TRUNCATE`
   - `rmdir`
   - `xargs rm`
3. **Non-destructive commands → instant `{"decision":"approve"}`** (no Brain needed, ~0ms)
4. **Destructive commands → initialize Brain, call `brain.safety_check(command)`:**
   - Matches command against critical nodes and safety rules
   - Returns `critical_matches` (should block) and `warnings` (should warn)
5. **Decision logic:**
   - Critical matches → `{"decision":"block","reason":"BRAIN SAFETY: ..."}` — command BLOCKED
   - Warning matches → `{"decision":"approve","reason":"BRAIN WARNING: ..."}` — command allowed with warning
   - Destructive, no matches → `{"decision":"approve","reason":"Destructive command. Proceed carefully."}`
   - Brain unavailable → `{"decision":"approve","reason":"Brain unavailable for safety check."}`

**Brain methods called:** `safety_check`
**Tables read:** nodes (critical=1 nodes, safety-related nodes), node_embeddings
**Tables written:** None

---

### 4.6 idle-maintenance.sh → idle_maintenance.py

**Event:** `Notification`
**Matcher:** `idle_prompt` (fires when Claude is idle)
**Timeout:** 30s
**Output visible:** NO — Notification stdout is invisible to Claude
**Output format:** Free text stored as pending message

**Purpose:** Background maintenance: dreaming (cross-cluster bridge discovery), consolidation (strengthen frequent memories), self-healing (merge duplicates, auto-lock, cleanup), auto-tuning (parameter adjustment), reflection, summary/embedding backfill, quote pruning, session health check. This is the brain's "overnight processing."

**What it does (9 operations):**
1. `brain.dream()` — generates cross-cluster bridge insights
2. `brain.consolidate()` — boosts frequently-accessed nodes, discovers evolution patterns
3. `brain.auto_heal()` — merge duplicates, auto-lock popular nodes, clean orphans, normalize edges
4. `brain.auto_tune()` — adjust scoring parameters based on recall patterns
5. `brain.prompt_reflection()` — generate reflection prompts about session
6. `brain.auto_generate_self_reflection()` — periodic self-assessment
7. `brain.backfill_summaries(batch_size=50)` — generate summaries for nodes lacking them
8. `brain.backfill_embeddings(batch_size=20)` — generate embeddings for nodes lacking them
9. `brain.prune_irrelevant_quotes(batch_size=30)` — remove low-quality auto-captured quotes

After all operations, stores combined output as pending message via `store_pending_message()`.

**Brain methods called:** `dream`, `consolidate`, `get_active_evolutions`, `auto_heal`, `auto_tune`, `prompt_reflection`, `auto_generate_self_reflection`, `backfill_summaries`, `backfill_embeddings`, `prune_irrelevant_quotes`, `assess_session_health`, `save`, `close`

**Tables read/written:** Nearly all tables — this is the most comprehensive maintenance operation

---

### 4.7 pre-compact-save.sh → pre_compact_save.py

**Event:** `PreCompact`
**Matcher:** (none)
**Timeout:** 10s
**Output visible:** NO (only decision JSON read by harness)
**Output format:** `{"decision":"approve"}` — MUST always approve, never block compaction

**Purpose:** Save brain state before context compaction destroys conversation history. Synthesizes the session so decisions/corrections/open questions are captured. Writes a compaction boundary marker node.

**What it does:**
1. Calls `brain.synthesize_session()` — captures session arc (decisions, corrections, teaching arcs, open questions)
2. Creates a `context` type node as compaction boundary marker (title: "Compaction boundary at {timestamp}")
3. `brain.save()` — flush everything

**Critical constraint:** This hook MUST output `{"decision":"approve"}`. If it blocks compaction, Claude Code cannot compact its context and will eventually hit token limits.

**Brain methods called:** `synthesize_session`, `remember` (boundary marker), `save`, `close`
**Tables written:** session_syntheses (synthesis), nodes (boundary marker), brain_meta

---

### 4.8 post-compact-reboot.sh → post_compact_reboot.py

**Event:** `PostCompact`
**Matcher:** (none)
**Timeout:** 10s
**Output visible:** YES — stdout injected as post-compaction context
**Output format:** Free text (multi-line context re-injection)

**Purpose:** Re-inject brain context after compaction. This is the SAFETY NET. When Claude's context is compacted, conversation history is lost. This hook re-boots the essential context: locked rules, consciousness signals, recalled context related to recent work, pending messages, and transcript path for full rehydration.

**What it does:**
1. Checks if pre-compact synthesis ran; if not, runs it now
2. Calls `brain.context_boot()` — gets locked rules
3. Gets consciousness signals (reminders, evolutions)
4. Recalls context related to recent work (using last synthesis + recent node titles as query)
5. Drains pending messages from background hooks
6. Outputs: LOCKED RULES, REMINDERS, EVOLUTIONS, OPEN QUESTIONS, RECALLED CONTEXT, QUEUED MESSAGES, transcript path hint

**Brain methods called (daemon):** `context_boot`, `consciousness`, `last_synthesis`, `get_config`, `recall`
**Brain methods called (direct):** `get_config`, `context_boot`, `get_consciousness_signals`, `assess_developmental_stage`, `recall_with_embeddings`, `synthesize_session`, `save`, `close`

**Tables read:** nodes, session_syntheses, brain_meta, node_embeddings
**Tables written:** session_syntheses (if synthesis wasn't run pre-compact), brain_meta (pending messages cleared)

---

### 4.9 session-end.sh → session_end.py

**Event:** `SessionEnd`
**Matcher:** (none)
**Timeout:** 10s
**Output visible:** NO
**Output format:** stderr diagnostic only

**Purpose:** Clean shutdown: synthesize session, consolidate memories, save and close. The daemon (if running) is shut down.

**What it does:**
1. Synthesize session (decisions, corrections, teaching arcs, open questions)
2. Consolidate (strengthen frequent memories, detect patterns)
3. Save (flush WAL)
4. Shutdown daemon (if running)
5. Close brain

**Brain methods called (daemon):** `synthesize_session`, `consolidate`, `save`, `shutdown`
**Brain methods called (direct):** `synthesize_session`, `consolidate`, `save`, `close`

---

### 4.10 stop-failure-log.sh → stop_failure_log.py

**Event:** `StopFailure`
**Matcher:** (none)
**Timeout:** 5s
**Output visible:** NO

**Purpose:** Log API failures (Claude stop errors) to the brain's miss_log for pattern detection. If Claude is frequently hitting errors, the brain can surface this as a trend.

**What it does:**
1. Parses error type and details from hook input
2. Calls `brain.log_miss()` with signal="api_failure"
3. Save

**Brain methods called:** `log_miss`, `save`, `close`
**Tables written:** miss_log

---

### 4.11 config-change-host.sh → config_change_host.py

**Event:** `ConfigChange`
**Matcher:** (none)
**Timeout:** 5s
**Output visible:** NO — stores as pending message

**Purpose:** Detect host environment changes when Claude Code configuration changes. Scans environment (Python version, Node version, OS, etc.) and compares against stored baseline. If changes found, stores as pending message for next recall.

**What it does:**
1. Calls `brain.scan_host_environment()` — compares current env against stored baseline
2. If changes detected: formats change list, stores as pending message

**Brain methods called:** `scan_host_environment`, `save`, `close`
**Tables read/written:** brain_meta (host_environment baseline)

---

### 4.12 post-bash-host-check.sh → post_bash_host_check.py

**Event:** `PostToolUse`
**Matcher:** `Bash`
**Timeout:** 5s
**Output visible:** NO — stores as pending message

**Purpose:** After bash commands that could change the environment (pip install, brew install, npm install, conda, pyenv, nvm, etc.), scan for environment changes. Only fires for environment-changing commands — skips all other bash commands.

**What it does:**
1. **Fast pre-screen:** Regex check against 12 environment-changing patterns (pip/brew/apt/npm/cargo/gem/pyenv/nvm/conda install/uninstall/activate)
2. Non-matching commands → exit immediately (no Brain init)
3. Matching commands → `brain.scan_host_environment()`, store changes as pending message

**Brain methods called:** `scan_host_environment`, `save`, `close`

---

### 4.13 worktree-context.sh → worktree_context.py

**Event:** `WorktreeCreate`
**Matcher:** (none)
**Timeout:** 5s
**Output visible:** Partially (prints to stdout, but WorktreeCreate visibility depends on harness)

**Purpose:** Track git worktree/branch context in brain config. When a worktree is created, detect the branch and store worktree name, branch, and CWD in brain config.

**What it does:**
1. Gets worktree name and CWD from hook input
2. Runs `git rev-parse --abbrev-ref HEAD` to detect branch
3. Sets config: `current_worktree`, `current_branch`, `current_cwd`
4. Calls `brain.scan_host_environment()` to capture new context

**Brain methods called:** `set_config`, `scan_host_environment`, `save`, `close`
**Tables written:** brain_meta (worktree/branch/cwd config)

---

### 4.14 worktree-cleanup.sh → worktree_cleanup.py

**Event:** `WorktreeRemove`
**Matcher:** (none)
**Timeout:** 5s
**Output visible:** NO

**Purpose:** Clear worktree context from brain config when a worktree is removed.

**What it does:**
1. Clears config: `current_worktree`, `current_branch`, `current_cwd` (set to empty string)
2. Save

**Brain methods called:** `set_config`, `save`, `close`
**Tables written:** brain_meta

---

## 5. Database Tables

The brain uses two SQLite databases: `brain.db` (main) and `brain_logs.db` (logging/diagnostics).

### brain.db — Main Database

#### Core tables

| Table | Purpose | Key columns | Used by hooks |
|-------|---------|-------------|---------------|
| `nodes` | All knowledge nodes — decisions, rules, lessons, mental models, etc. | id, type, title, content, keywords, locked, critical, archived, activation, emotion, confidence, scope, project, access_count, last_accessed, evolution_status, created_at, updated_at | ALL hooks |
| `edges` | Connections between nodes | source_id, target_id, edge_type, weight, context, created_at | boot, recall, consolidation |
| `brain_meta` | Key-value config store | key, value, updated_at | ALL hooks (config) |
| `node_embeddings` | Vector embeddings for semantic search | node_id, embedding, model, created_at | recall, suggest, safety, consciousness |
| `node_metadata` | Rich metadata: reasoning, alternatives, user quotes, validation | node_id, reasoning, alternatives_considered, user_raw_quote, context_at_time, correction_of, last_validated, validation_count | boot, recall, consciousness |
| `session_syntheses` | Session summary records | session_id, decisions_made, corrections_received, teaching_arcs, open_questions, reflection_prompts, duration_minutes, created_at | boot, pre-compact, post-compact, session-end |
| `correction_traces` | Divergence records: assumed vs reality | session_id, original_node_id, claude_assumed, reality, underlying_pattern, severity, created_at | consciousness, instinct check |
| `session_activity` | Per-session counters | session_id, message_count, remember_count, recall_count, boot_time | track, encoding heartbeat |
| `projects` | Project-level metadata | id, name, system_purpose, architecture, key_decisions, status | boot (engineering context) |
| `project_maps` | Project file inventory and maps | project, map_type, title, content, file_path, purpose | boot, suggest |
| `summaries` | Node summary/embedding support | node_id, summary, keywords | recall |
| `node_vectors` | TF-IDF vectors for keyword search | node_id, term, weight | recall (TF-IDF path) |
| `doc_freq` | Document frequency for TF-IDF | term, count | recall (TF-IDF path) |
| `reasoning_chains` | Linked reasoning chains | id, decision_node_id, project, chain_type, created_at | recall (reasoning chain lookup) |
| `reasoning_steps` | Steps within reasoning chains | id, chain_id, step_type, content, evidence_node_id | recall |
| `bridge_proposals` | Cross-cluster bridge candidates (from dreaming) | id, source_id, target_id, rationale, status, matures_at | dreaming, consolidation |
| `prune_archive` | Archived/pruned items for audit trail | id, item_type, original_data, pruned_at, reason | healing, pruning |
| `staged_learnings` | Pending learnings awaiting confirmation | id, node_id, status, confidence, times_revisited | boot (staged display) |
| `emotion_calibration` | Emotion label calibration data | label, valence, arousal, count | scoring |
| `version_history` | Schema version tracking | version, migrated_at | schema migrations |

### brain_logs.db — Logging Database

| Table | Purpose | Key columns | Used by hooks |
|-------|---------|-------------|---------------|
| `access_log` | Node access tracking | session_id, node_id, event_type, created_at | recall (access recording) |
| `recall_log` | Recall query + result logging | session_id, query, intent, result_count, timings, created_at | recall (diagnostics) |
| `miss_log` | Failed recall / miss tracking | session_id, signal, query, expected_node_id, context | stop-failure-log, consciousness (miss trends) |
| `debug_log` | Debug event logging | session_id, event_type, source, metadata, created_at | suggest (when debug on) |
| `dream_log` | Dream generation records | session_id, dream_type, nodes_involved, result | idle-maintenance |
| `tuning_log` | Auto-tune parameter changes | param, old_value, new_value, reason | idle-maintenance |
| `eval_snapshots` | Eval/benchmark snapshots | snapshot_name, data, created_at | eval system |
| `suggest_log` | Pre-edit suggestion records | session_id, file_target, suggestions_served, procedures_served | suggest hook |
| `curiosity_log` | Curiosity/exploration logging | session_id, topic, question, source | exploration |
| `health_log` | Health check results | session_id, check_type, issues_found, actions_taken | health checks |
| `staged_learnings` | (also in logs DB) staging records | | |
| `hook_errors` | Hook-level structural failures | hook_name, level, error, context, traceback, surfaced, created_at | ALL hooks (error logging), consciousness (surfacing) |

---

## 6. Daemon Command Reference

The daemon supports 40+ commands. Each maps to a Brain method. Key commands used by hooks:

| Command | Hook that calls it | Brain method | Purpose |
|---------|-------------------|-------------|---------|
| `context_boot` | post-compact-reboot | `context_boot()` | Get locked rules, counts, session note |
| `recall` | pre-response-recall, post-compact-reboot | `recall_with_embeddings()` | Semantic + keyword recall |
| `remember` | pre-compact-save | `remember()` | Store a new node |
| `record_message` | post-response-track | `record_message()` | Increment message counter |
| `heartbeat` | post-response-track | `get_encoding_heartbeat()` | Check if encoding is overdue |
| `vocab_check` | pre-response-recall | vocabulary expansion check | Expand query with vocab mappings |
| `pre_edit` | pre-edit-suggest | `pre_edit()` | Get file-relevant suggestions |
| `safety_check` | pre-bash-safety | `safety_check()` | Check command against destructive patterns |
| `consciousness` | post-compact-reboot | `get_consciousness_signals()` | Full consciousness scan |
| `urgent_signals` | pre-response-recall | `get_urgent_signals()` | Lightweight awareness check |
| `instinct_check` | pre-response-recall | `get_instinct_check()` | Check for instinct nudges |
| `dream` | idle-maintenance | `dream()` | Generate cross-cluster bridges |
| `consolidate` | idle-maintenance, session-end | `consolidate()` | Strengthen frequent memories |
| `auto_heal` | idle-maintenance | `auto_heal()` | Merge duplicates, cleanup |
| `auto_tune` | idle-maintenance | `auto_tune()` | Adjust scoring parameters |
| `synthesize_session` | pre-compact, session-end | `synthesize_session()` | Capture session arc |
| `save` | many hooks | `brain.save()` | Flush WAL |
| `shutdown` | session-end | (daemon exits) | Clean daemon shutdown |
| `get_config` | many hooks | `get_config()` | Read brain_meta |
| `set_config` | many hooks | `set_config()` | Write brain_meta |
| `get_active_evolutions` | idle-maintenance | `get_active_evolutions()` | Get evolution nodes |
| `prompt_reflection` | idle-maintenance | `prompt_reflection()` | Generate reflection prompts |
| `self_reflection` | idle-maintenance | `auto_generate_self_reflection()` | Self-assessment |
| `backfill_summaries` | idle-maintenance | `backfill_summaries()` | Generate missing summaries |
| `backfill_embeddings` | idle-maintenance | `backfill_embeddings()` | Generate missing embeddings |
| `prune_irrelevant_quotes` | idle-maintenance | `prune_irrelevant_quotes()` | Remove low-quality quotes |
| `last_synthesis` | post-compact-reboot | `get_last_synthesis()` | Get last session synthesis |
| `scan_host` | boot | `scan_host_environment()` | Detect env changes |

---

## 7. Consciousness & Awareness Layer

The consciousness layer is implemented in `servers/brain_consciousness.py` as a mixin on the Brain class. It has two tiers:

### Full scan: `get_consciousness_signals()`
- Called ONLY at boot and post-compact reboot
- Expensive (~500ms): queries dozens of tables, runs embedding similarity checks
- Returns 25+ signal categories (see boot hook section above)
- Used to generate the full "BRAIN CONSCIOUSNESS" section of boot output

### Lightweight heartbeat: `get_urgent_signals()`
- Called on EVERY `UserPromptSubmit` (via recall hook)
- Fast (<50ms): only checks 3 things:
  1. Unsurfaced hook errors in brain_logs.db
  2. Silent brain errors from last 2 hours
  3. Overdue reminders
- Returns list of text lines; empty = nothing urgent
- Surfaced as "BRAIN AWARENESS:" section at top of recall output

### Why two tiers?
The full scan is too expensive to run on every prompt. But urgent signals (broken hooks, errors, due reminders) need immediate visibility. The two-tier design means:
- Boot: you get the full picture (25+ signal types)
- Every prompt: you get urgent alerts only (3 checks)
- If a hook breaks mid-session, you'll know on the next prompt, not at the next boot

---

## 8. Error Handling Architecture

### The error visibility problem
Before the current architecture, errors were invisible:
- `except: pass` swallowed exceptions silently
- stderr went to Claude Code logs but not Claude's context
- No persistent record of failures

### Current architecture: three layers

1. **hook_common.log_hook_error()** — logs to stderr AND brain_logs.db → hook_errors table
   - Called by hook_common functions (get_brain, get_hook_input, close_brain) on failure
   - Available to all hook scripts for explicit error logging
   - Uses direct SQLite (not Brain) — works even when Brain import fails

2. **brain._log_error()** — logs to brain_logs.db → error_log table
   - Called by Brain methods internally when operations fail
   - These are "silent brain errors" — things that failed inside brain methods

3. **Consciousness layer surfaces both:**
   - `get_consciousness_signals()` reads both hook_errors (brain_logs.db) and error_log (brain_logs.db) → surfaced at boot
   - `get_urgent_signals()` reads both → surfaced on every prompt via recall hook

### Remaining gaps (known, not yet fixed):
- 52 `except: pass` locations across hook scripts that swallow errors without logging
- daemon-client.sh still has embedded Python (utility script, not used by extracted hooks)
- `2>/dev/null` in resolve-brain-db.sh suppresses some bash errors

---

## 9. Pending Message Pattern

Background hooks whose stdout is invisible to Claude use a message queue:

### Write side: `store_pending_message(brain_or_daemon, message)`
- Reads `pending_hook_messages` from brain_meta (JSON array)
- Appends new message
- Caps at 5 messages (oldest dropped)
- Writes back to brain_meta

### Read side: `drain_pending_messages(brain_or_daemon)`
- Reads `pending_hook_messages` from brain_meta
- Clears the queue (sets to `[]`)
- Returns list of message strings

### Hooks that WRITE pending messages:
- `idle_maintenance.py` — maintenance results
- `config_change_host.py` — environment changes
- `post_bash_host_check.py` — environment changes after bash
- `post_response_track.py` — encoding checkpoints (on Stop event only)

### Hooks that READ (drain) pending messages:
- `pre_response_recall.py` — includes in `additionalContext` output
- `post_compact_reboot.py` — includes in reboot output

---

## 10. Common Pitfalls & Failure Modes

### Don't add output to invisible hooks
If you add `print()` to session-end.py, idle-maintenance.py, or config-change-host.py, the output vanishes. Use `store_pending_message()` instead.

### Don't block compaction
pre-compact-save.sh MUST output `{"decision":"approve"}`. Any other decision or missing output will block compaction and eventually crash Claude Code.

### Don't forget the dual path
Every hook needs both daemon and direct paths. If you add a feature to the daemon path, add it to the direct path too. Users without a running daemon will miss it.

### Timeout awareness
The 5s timeout on pre-response-recall means you can't do expensive work on every prompt. The consciousness heartbeat (`get_urgent_signals()`) was designed to be <50ms for this reason.

### Graceful degradation
Every hook must exit 0 on failure. A crashing hook means Claude proceeds without brain context. The pattern is:
```python
try:
    if daemon_available():
        _run_daemon()
    else:
        _run_direct()
except Exception:
    print(APPROVE)  # or just sys.exit(0)
```

### The quoting problem (solved)
Embedded `python3 -c '...'` in bash breaks on apostrophes in Python strings. The thin shim pattern (`exec python3 script.py`) eliminates this permanently. NEVER go back to embedded Python in bash.

### Database locking
Multiple hooks can fire concurrently. SQLite handles this via WAL mode, but timeout on connection should be reasonable (3-5s). The daemon serializes access through a single Brain instance.

---

## File Inventory

### Hook scripts (hooks/scripts/)

| File | Type | Lines | Purpose |
|------|------|-------|---------|
| resolve-brain-db.sh | Shared bash | 50 | DB path resolution |
| hook_common.py | Shared Python | 244 | All shared functions |
| boot-brain.sh | Bash shim | 53 | SessionStart shim + daemon start |
| boot_brain.py | Python | 751 | Full boot output |
| pre-response-recall.sh | Bash shim | 6 | UserPromptSubmit shim |
| pre_response_recall.py | Python | 330 | Recall + awareness heartbeat |
| post-response-track.sh | Bash shim | 6 | Stop only shim (Session #9: removed from UserPromptSubmit) |
| post_response_track.py | Python | 234 | Vocab gaps + encoding checkpoints |
| pre-edit-suggest.sh | Bash shim | 6 | PreToolUse(Edit\|Write) shim |
| pre_edit_suggest.py | Python | 267 | File-relevant suggestions |
| pre-bash-safety.sh | Bash shim | 6 | PreToolUse(Bash) shim |
| pre_bash_safety.py | Python | 99 | Destructive command interception |
| idle-maintenance.sh | Bash shim | 6 | Notification(idle) shim |
| idle_maintenance.py | Python | 343 | Dream/consolidate/heal/tune |
| pre-compact-save.sh | Bash shim | 6 | PreCompact shim |
| pre_compact_save.py | Python | 68 | Session synthesis + save |
| post-compact-reboot.sh | Bash shim | 6 | PostCompact shim |
| post_compact_reboot.py | Python | 303 | Context re-injection |
| session-end.sh | Bash shim | 6 | SessionEnd shim |
| session_end.py | Python | 51 | Synthesis + shutdown |
| stop-failure-log.sh | Bash shim | 6 | StopFailure shim |
| stop_failure_log.py | Python | 29 | API failure logging |
| config-change-host.sh | Bash shim | 6 | ConfigChange shim |
| config_change_host.py | Python | 40 | Host env change detection |
| post-bash-host-check.sh | Bash shim | 6 | PostToolUse(Bash) shim |
| post_bash_host_check.py | Python | 51 | Post-bash env change detection |
| worktree-context.sh | Bash shim | 6 | WorktreeCreate shim |
| worktree_context.py | Python | 46 | Git context tracking |
| worktree-cleanup.sh | Bash shim | 6 | WorktreeRemove shim |
| worktree_cleanup.py | Python | 20 | Clear worktree config |

### Brain server files (servers/)

| File | Purpose |
|------|---------|
| brain.py | Main Brain class — assembles all mixins, owns `__init__`, `_combined_score()`, `save()`, `close()` |
| brain_recall.py | `recall()`, `recall_with_embeddings()`, vocabulary expansion, scoring |
| brain_remember.py | `remember()`, `remember_rich()`, keyword extraction, critical flag |
| brain_surface.py | `context_boot()`, `pre_edit()`, `safety_check()` |
| brain_consciousness.py | `get_consciousness_signals()`, `get_urgent_signals()`, `assess_developmental_stage()`, priming, instinct check |
| brain_engineering.py | `get_engineering_context()`, file inventory, change detection |
| brain_evolution.py | `auto_heal()`, `auto_tune()`, `consolidate()`, evolution management |
| brain_dreams.py | `dream()`, cross-cluster bridge discovery |
| brain_connections.py | `connect()`, spreading activation |
| brain_vocabulary.py | `learn_vocabulary()`, `resolve_vocabulary()`, admission guards |
| brain_absorb.py | `absorb_document()`, bulk import |
| brain_constants.py | All scoring constants, decay rates, thresholds |
| schema.py | Schema v16 definition, migrations |
| daemon.py | Unix socket server, 40+ command handlers |
| embedder.py | Embedding model loading, `embed()`, `cosine_similarity()` |
