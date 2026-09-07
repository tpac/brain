# Brain Plugin — Hook Architecture Reference

> **This is the single source of truth.** If it's not in this doc, it's not real.

## How Hook Output Reaches the Model

Hook stdout is read by two hosts — Claude Code and Codex (ChatGPT's Codex mode) —
and both parse it against strict per-event JSON schemas. `hook_common.emit_hook_output`
is the single writer of what the brain says: scripts hand it the daemon's
`{decision, reason}` and never print a decision themselves. Its one sibling,
`hook_common.emit_updated_input`, writes the one tool input the brain rewrites
(PreToolUse `permissionDecision: allow` + `updatedInput`, the caller-identity
stamp on the brain's own MCP tools).

| Event | Model-visible channel | Block | Nothing to say |
|-------|-----------------------|-------|----------------|
| **SessionStart** | plain stdout or `hookSpecificOutput.additionalContext` | — | exit 0, no output |
| **UserPromptSubmit** | `hookSpecificOutput.additionalContext` | never (a daemon block here is downgraded to context and logged) | exit 0, no output |
| **PreToolUse** | `hookSpecificOutput.additionalContext` (delivered with the tool result — the model sees it as the tool completes, not before) | never (downgraded to context and logged) | exit 0, no output |
| **PostToolUse** | `hookSpecificOutput.additionalContext` on both hosts; plain stdout is debug-only | — | exit 0, no output |
| **Stop** | none — plain stdout is invisible in Claude Code and invalid in Codex | `{"decision":"block","reason"}` → the host continues the turn with `reason` as the next prompt (this is how self-messages are delivered) | exit 0, no output |
| **All other events** | none — stdout is debug-only | — | — |

**The brain informs; it never gates.** Silence is "no opinion": the host applies
its own permission mode and allow/deny rules. Never print `{"decision":"approve"}`:
Codex treats it as invalid output and marks the hook run FAILED, and on Claude Code
it meant `permissionDecision: allow` — skip the permission prompt — which a memory
plugin has no business deciding for its user.

Two manifests, one set of scripts: `hooks.json` (Claude Code) and
`hooks.codex.json` (Codex — the shared events only, `additionalContextLimit: 0`
on the two injecting hooks, SessionEnd within Codex's 3 s cap, plus the
Codex-only identity stamp). `tests/test_hooks_manifest_sync.py` keeps them in
step.

---

## Registered Hooks

### 1. SessionStart → `boot-brain.sh` (15s)
- **Purpose:** Boot brain, print context + consciousness signals
- **Output:** Plain stdout → ✅ injected into Claude's context
- **What Claude sees:** Locked rules, consciousness signals, dev stage, session context
- **Status:** ✅ WORKING

### 2. UserPromptSubmit → `pre-response-recall.sh` (5s)
- **Purpose:** Recall relevant memories before Claude responds
- **Output:** `hookSpecificOutput.additionalContext` → ✅ injected
- **What Claude sees:** Recalled nodes, evolution tracking, instinct checks, aspirations
- **Status:** ✅ WORKING

### 4. PreToolUse(Edit|Write) → `pre-edit-suggest.sh` (8s)
- **Purpose:** Surface relevant brain rules before file edits
- **Output:** `hookSpecificOutput.additionalContext` when there are rules to surface, else nothing
- **What Claude sees:** Relevant rules, conventions, encoding warnings before editing
- **Status:** ✅ WORKING

### 5. PreToolUse(Bash) → `pre-bash-safety.sh` (8s)
- **Purpose:** Warn about destructive bash commands with the brain's context (never blocks)
- **Output:** `additionalContext` with the brain's safety context (critical brain-tracked resources, matching warnings); nothing when the command is clean. Never blocks.
- **What Claude sees:** Safety warnings and critical node matches, alongside the command's result
- **Status:** ✅ WORKING

### 6. PreToolUse(mcp__brain__*) → `stamp-caller-session.sh` (5s, Codex only)
- **Purpose:** Attribute brain MCP calls on a host that gives the proxy no session identity. Codex hands stdio MCP servers no thread id, so the hook signs the `session_id` it receives (HMAC-SHA256, secret at `~/.config/brain/hook-secret`) and rewrites the tool input with `_caller_session` + `_caller_sig`; the proxy accepts the pair only when it verifies and strips the signature before dispatch. No daemon call.
- **Output:** `hookSpecificOutput.permissionDecision: allow` + `updatedInput` (the brain permitting its own tools — the emitter refuses any other tool name); nothing for a payload without `session_id` or a dict `tool_input` (logged)
- **What Claude sees:** Nothing — the rewrite happens before the call; `post_tool_trace.py` strips the pair before recording the input, so no trace ever carries a replayable signature
- **Not on Claude Code:** the proxy reads `CLAUDE_CODE_SESSION_ID` there (decision fa0f5f5a); an unattributed call is noted once per proxy process in `hook_errors`
- **Status:** ✅ BUILT — live under Codex still to verify (E5)

### 7. Stop → `post-response-track.sh` (5s)
- **Purpose:** Record the turn (S0 traces) and deliver pending self-messages
- **Output:** `{"decision":"block","reason":"..."}` only when a self-message must be delivered; otherwise nothing
- **What Claude sees:** The block reason, as the prompt of the continued turn
- **Status:** ✅ WORKING

### 8. StopFailure → `stop-failure-log.sh` (5s)
- **Purpose:** Log API failures to brain for pattern detection
- **Output:** None (logging only)
- **What Claude sees:** Nothing (correct — logging only)
- **Status:** ✅ WORKING

### 9. SessionEnd → `session-end.sh` (10s)
- **Purpose:** Session synthesis + consolidation + clean shutdown
- **Output:** stderr logging only
- **What Claude sees:** Nothing (correct — session is ending)
- **Status:** ✅ WORKING

### 10. ConfigChange → `config-change-host.sh` (5s)
- **Purpose:** Detect host environment changes
- **Output:** Plain stdout
- **What Claude sees:** ❌ NOTHING — ConfigChange stdout is NOT injected
- **Status:** ❌ OUTPUT IS DEAD — changes detected but never surfaced

### 11. PostToolUse(Bash) → `post-bash-host-check.sh` (5s)
- **Purpose:** Detect env changes after pip install, brew, etc.
- **Output:** Plain stdout
- **What Claude sees:** ❌ NOTHING — PostToolUse stdout is NOT injected
- **Status:** ❌ OUTPUT IS DEAD — changes detected but never surfaced

### 12. WorktreeCreate → `worktree-context.sh` (5s)
- **Purpose:** Track git branch/worktree info
- **Output:** Plain stdout (git context info)
- **What Claude sees:** ❓ UNCLEAR — WorktreeCreate stdout is structural (path), not context
- **Status:** ⚠️ VERIFY — may need to store in brain config and surface via recall instead

### 13. WorktreeRemove → `worktree-cleanup.sh` (5s)
- **Purpose:** Clear worktree config from brain
- **Output:** None
- **What Claude sees:** Nothing (correct — cleanup only)
- **Status:** ✅ WORKING

---

## Summary

| Status | Count | Hooks |
|--------|-------|-------|
| ✅ Working | 7 | boot, recall, pre-edit, pre-bash, post-response-track, stop-failure, session-end |
| ⚠️ Partial | 2 | worktree-context (verify), stamp-caller-session (built, Codex E5 pending) |
| ❌ Dead output | 2 | config-change, post-bash-host |

**2 hooks produce output that Claude never sees.**

---

## Fix Plan

### Dead outputs that need fixing:
1. **config-change-host.sh** — Host changes invisible. Fix: store as brain node, surface via consciousness signals on next boot/recall
2. **post-bash-host-check.sh** — Same as config-change. Fix: store as brain node

### Format:
- All model-visible text goes through `emit_hook_output` as `additionalContext`; blocks use the event's block form. No script prints a decision directly.

---

## Event Lifecycle (typical session)

```
SessionStart
  └→ boot-brain.sh ── prints context, rules, signals

User sends message
  └→ pre-response-recall.sh ── recalls relevant memories (additionalContext)

Claude uses Edit/Write tool
  └→ pre-edit-suggest.sh ── surfaces rules (additionalContext)

Claude uses Bash tool
  └→ pre-bash-safety.sh ── safety check (additionalContext warning / deny)
  └→ post-bash-host-check.sh ── env change check (⚠️ output dead)

Claude finishes responding
  └→ post-response-track.sh ── records the turn; blocks only to deliver a self-message

Context fills up
  └→ (no brain hooks — compaction is invisible to the brain;
      the next UserPromptSubmit fires hook_recall and surfaces Frame)

Session ends
  └→ session-end.sh ── final synthesis + consolidation + shutdown
```
