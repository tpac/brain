# Brain Plugin — Hook Architecture Reference

> **This is the single source of truth.** If it's not in this doc, it's not real.
> Last updated: v5.3.0 (2026-03-20)

## How Hook Output Reaches Claude

**Only 3 event types inject stdout into Claude's context:**

| Event | Stdout → Claude? | JSON fields | Can block? |
|-------|------------------|-------------|-----------|
| **SessionStart** | ✅ YES | `additionalContext` | Yes |
| **UserPromptSubmit** | ✅ YES | `additionalContext` | Yes |
| **PostCompact** | ✅ YES | `additionalContext` | No |

**All other events: stdout is invisible to Claude** (logged in verbose mode only).

For **PreToolUse**: Claude gets feedback via `{"decision":"approve/block","reason":"..."}` JSON — the `reason` field is fed back as tool-use feedback, NOT as general context.

For **Stop**: Claude gets feedback only via JSON `decision` field. Plain stdout is invisible.

---

## Registered Hooks (15 total)

### 1. SessionStart → `boot-brain.sh` (15s)
- **Purpose:** Boot brain, print context + consciousness signals
- **Output:** Plain stdout → ✅ injected into Claude's context
- **What Claude sees:** Locked rules, consciousness signals, dev stage, session context
- **Status:** ✅ WORKING

### 2. UserPromptSubmit → `pre-response-recall.sh` (5s)
- **Purpose:** Recall relevant memories before Claude responds
- **Output:** `{"additionalContext": "BRAIN RECALL..."}` → ✅ injected
- **What Claude sees:** Recalled nodes, evolution tracking, instinct checks, aspirations
- **Status:** ✅ WORKING (fixed v5.3 — was using `reason` which is metadata, not context)

### 3. UserPromptSubmit → `post-response-track.sh` (3s)
- **Purpose:** Vocab gap detection + encoding checkpoint injection
- **Output:** Plain stdout → ✅ injected (UserPromptSubmit)
- **What Claude sees:** Encoding checkpoint prompts with rotating 5-focus cycle + session stats
- **Status:** ⚠️ PARTIAL — works on UserPromptSubmit, but also registered on Stop where stdout is INVISIBLE

### 4. PreToolUse(Edit|Write) → `pre-edit-suggest.sh` (8s)
- **Purpose:** Surface relevant brain rules before file edits
- **Output:** `{"decision":"approve","reason":"..."}` → ✅ reason fed back as tool feedback
- **What Claude sees:** Relevant rules, conventions, encoding warnings before editing
- **Status:** ✅ WORKING

### 5. PreToolUse(Bash) → `pre-bash-safety.sh` (8s)
- **Purpose:** Block/warn on destructive bash commands
- **Output:** `{"decision":"block/approve","reason":"..."}` → ✅ reason fed back
- **What Claude sees:** Safety warnings, critical node matches, block reasons
- **Status:** ✅ WORKING

### 6. Notification(idle_prompt) → `idle-maintenance.sh` (30s)
- **Purpose:** Dream, consolidate, self-reflect during idle
- **Output:** Plain stdout
- **What Claude sees:** ❌ NOTHING — Notification stdout is NOT injected
- **Status:** ❌ OUTPUT IS DEAD — maintenance runs but results are invisible

### 7. PreCompact → `pre-compact-save.sh` (10s)
- **Purpose:** Synthesize session + save before compaction
- **Output:** `{"decision":"approve"}` + stderr logging
- **What Claude sees:** Nothing (no context to inject, just saving)
- **Status:** ✅ WORKING (correctly — no output needed)

### 8. PostCompact → `post-compact-reboot.sh` (10s)
- **Purpose:** Re-inject brain context after compaction
- **Output:** Plain stdout → ✅ injected into Claude's context
- **What Claude sees:** Locked rules, consciousness signals, recalled context (v5.3+)
- **Status:** ✅ WORKING (improved v5.3 — now includes recall)

### 9. Stop → `post-response-track.sh` (5s)
- **Purpose:** Encoding checkpoint on Stop events
- **Output:** Plain stdout
- **What Claude sees:** ❌ NOTHING — Stop stdout is NOT injected
- **Status:** ❌ OUTPUT IS DEAD on this event — checkpoints only work via UserPromptSubmit (#3)

### 10. StopFailure → `stop-failure-log.sh` (5s)
- **Purpose:** Log API failures to brain for pattern detection
- **Output:** None (logging only)
- **What Claude sees:** Nothing (correct — logging only)
- **Status:** ✅ WORKING

### 11. SessionEnd → `session-end.sh` (10s)
- **Purpose:** Session synthesis + consolidation + clean shutdown
- **Output:** stderr logging only
- **What Claude sees:** Nothing (correct — session is ending)
- **Status:** ✅ WORKING

### 12. ConfigChange → `config-change-host.sh` (5s)
- **Purpose:** Detect host environment changes
- **Output:** Plain stdout
- **What Claude sees:** ❌ NOTHING — ConfigChange stdout is NOT injected
- **Status:** ❌ OUTPUT IS DEAD — changes detected but never surfaced

### 13. PostToolUse(Bash) → `post-bash-host-check.sh` (5s)
- **Purpose:** Detect env changes after pip install, brew, etc.
- **Output:** Plain stdout
- **What Claude sees:** ❌ NOTHING — PostToolUse stdout is NOT injected
- **Status:** ❌ OUTPUT IS DEAD — changes detected but never surfaced

### 14. WorktreeCreate → `worktree-context.sh` (5s)
- **Purpose:** Track git branch/worktree info
- **Output:** Plain stdout (git context info)
- **What Claude sees:** ❓ UNCLEAR — WorktreeCreate stdout is structural (path), not context
- **Status:** ⚠️ VERIFY — may need to store in brain config and surface via recall instead

### 15. WorktreeRemove → `worktree-cleanup.sh` (5s)
- **Purpose:** Clear worktree config from brain
- **Output:** None
- **What Claude sees:** Nothing (correct — cleanup only)
- **Status:** ✅ WORKING

---

## Summary

| Status | Count | Hooks |
|--------|-------|-------|
| ✅ Working | 8 | boot, recall, pre-edit, pre-bash, pre-compact, post-compact, stop-failure, session-end |
| ⚠️ Partial | 2 | post-response-track (Stop path dead), worktree-context (verify) |
| ❌ Dead output | 4 | idle-maintenance, config-change, post-bash-host, Stop path of track |

**4 hooks produce output that Claude never sees.**

---

## Fix Plan

### Dead outputs that need fixing:
1. **Stop → post-response-track.sh** — Encoding checkpoints on Stop are invisible. Options: (a) remove Stop registration, (b) use `additionalContext` JSON, (c) accept only UserPromptSubmit works
2. **idle-maintenance.sh** — Dream/consolidation results invisible. Options: (a) store results in brain config, surface via next recall, (b) accept as background-only
3. **config-change-host.sh** — Host changes invisible. Fix: store as brain node, surface via consciousness signals on next boot/recall
4. **post-bash-host-check.sh** — Same as config-change. Fix: store as brain node

### Format issues:
- All hooks outputting to context-injecting events should use `{"additionalContext":"..."}` for clean injection
- PreToolUse hooks correctly use `{"decision":"...","reason":"..."}` for feedback

---

## Event Lifecycle (typical session)

```
SessionStart
  └→ boot-brain.sh ── prints context, rules, signals

User sends message
  └→ pre-response-recall.sh ── recalls relevant memories (additionalContext)
  └→ post-response-track.sh ── vocab gaps + encoding checkpoint (stdout)

Claude uses Edit/Write tool
  └→ pre-edit-suggest.sh ── surfaces rules (decision+reason)

Claude uses Bash tool
  └→ pre-bash-safety.sh ── safety check (decision+reason)
  └→ post-bash-host-check.sh ── env change check (⚠️ output dead)

Claude finishes responding
  └→ post-response-track.sh ── encoding checkpoint (❌ output dead on Stop)

Context fills up
  └→ pre-compact-save.sh ── synthesize + save
  └→ post-compact-reboot.sh ── re-inject context + recall

Session ends
  └→ session-end.sh ── final synthesis + consolidation + shutdown
```
