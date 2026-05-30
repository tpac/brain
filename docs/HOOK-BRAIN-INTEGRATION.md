# Hook × Brain Integration Map

> Generated 2026-03-28. Updated weekly by scheduled task `map-claude-code-hooks`.

## Event × Handler Type Matrix

| Event | command | http | prompt | agent | Can Block? | Currently Used? |
|---|:---:|:---:|:---:|:---:|---|---|
| **SessionStart** | ✅ | ❌ | ❌ | ❌ | No | ✅ boot-brain.sh |
| **UserPromptSubmit** | ✅ | ✅ | ✅ | ✅ | **Yes** | ✅ pre-response-recall.sh |
| **PreToolUse** | ✅ | ✅ | ✅ | ✅ | **Yes** | ✅ pre-edit-suggest, test guardian, pre-bash-safety |
| **PostToolUse** | ✅ | ✅ | ✅ | ✅ | No | ✅ post-bash-host-check (Bash only) |
| **PostToolUseFailure** | ✅ | ✅ | ✅ | ✅ | No | ❌ **GAP** |
| **PermissionRequest** | ✅ | ✅ | ✅ | ✅ | **Yes** | ❌ |
| **Notification** | ✅ | ✅ | ✅ | ✅ | No | ✅ idle-maintenance (idle_prompt) |
| **SubagentStart** | ✅ | ✅ | ✅ | ✅ | No | ❌ **GAP** |
| **SubagentStop** | ✅ | ✅ | ✅ | ✅ | **Yes** | ❌ |
| **TaskCreated** | ✅ | ✅ | ✅ | ✅ | **Yes** | ❌ |
| **TaskCompleted** | ✅ | ✅ | ✅ | ✅ | **Yes** | ❌ |
| **Stop** | ✅ | ✅ | ✅ | ✅ | **Yes** | ✅ post-response-track.sh |
| **StopFailure** | ✅ | ✅ | ✅ | ✅ | No | ✅ stop-failure-log.sh |
| **PreCompact** | ✅ | ✅ | ✅ | ✅ | No | ✅ pre-compact-save.sh |
| **PostCompact** | ✅ | ✅ | ✅ | ✅ | No | ✅ post-compact-reboot.sh |
| **ConfigChange** | ✅ | ✅ | ✅ | ✅ | **Yes** | ✅ config-change-host.sh |
| **FileChanged** | ✅ | ✅ | ✅ | ✅ | No | ❌ **GAP** |
| **CwdChanged** | ✅ | ✅ | ✅ | ✅ | No | ❌ |
| **InstructionsLoaded** | ✅ | ✅ | ✅ | ✅ | No | ❌ |
| **TeammateIdle** | ✅ | ✅ | ✅ | ✅ | **Yes** | ❌ |
| **Elicitation** | ✅ | ✅ | ✅ | ✅ | **Yes** | ❌ |
| **ElicitationResult** | ✅ | ✅ | ✅ | ✅ | **Yes** | ❌ |
| **SessionEnd** | ✅ | ✅ | ✅ | ✅ | No | ✅ session-end.sh |
| **WorktreeCreate** | ✅ | ✅ | ✅ | ✅ | **Yes** | ✅ worktree-context.sh |
| **WorktreeRemove** | ✅ | ✅ | ✅ | ✅ | No | ✅ worktree-cleanup.sh |

## Handler Types Explained

| Type | What it is | Latency | Brain access | Best for |
|---|---|---|---|---|
| **command** | Shell script / Python | 50-500ms | Via daemon TCP | Fast deterministic checks, daemon calls |
| **http** | Webhook POST | 100-500ms | Via daemon HTTP | External services, Slack notifications |
| **prompt** | Single-turn LLM call | 1-3s | No tools, only text | Semantic evaluation, rule checking |
| **agent** | LLM with tool access | 3-10s | Full (Read, Grep, Bash, MCP) | Complex verification, deep reasoning |

- prompt/agent default to **fast cheap model** (Haiku-level), override with `"model": "sonnet"` or `"opus"`
- prompt/agent return `{"decision": "allow"}` or `{"decision": "deny", "reason": "..."}` for blocking events
- `"async": true` on any hook makes it fire-and-forget (no latency cost)

## Brain Integration Opportunities

### 🔴 CRITICAL — Implement Now

#### 1. PostToolUseFailure → Failure Memory Recall
**Impact:** HIGH | **Cost:** ~200ms | **UX:** Invisible

When a tool fails, recall lessons about similar failures before Claude retries blindly.

- **Brain data:** `lesson`, `failure_mode`, `bug_lesson` nodes matching error context
- **Handler:** command (daemon recall with error as query)
- **What Claude sees:** "This file failed before because X. The fix was Y."
- **Why critical:** Failures are where the most time is wasted. Brain has the answers, nobody asks.

#### 2. SubagentStart → Brain Context Injection
**Impact:** HIGH | **Cost:** ~300ms | **UX:** Invisible

Subagents (Explore, Plan, general) currently spawn brain-blind. They repeat mistakes the brain already corrected.

- **Brain data:** `engineering_context()` — conventions, constraints, mechanisms, locked rules
- **Handler:** command (existing endpoint, no new code)
- **What subagent sees:** "[BRAIN] Key project constraints: ..."
- **Why critical:** Every subagent we spawn today wastes tokens rediscovering things the brain knows.

#### 3. Remove `post-response-track.sh` from UserPromptSubmit
**Impact:** MEDIUM | **Cost:** Negative (removes latency) | **UX:** Invisible

Currently runs before Claude responds, so `last_assistant_message` is always empty. Dead code adding ~200ms per message. It correctly fires on Stop already.

### 🟠 HIGH — Implement Soon

#### 4. PreToolUse(mcp__brain__*) → Brain Usage Anti-Pattern Guard
**Impact:** MEDIUM | **Cost:** ~50ms | **UX:** Non-intrusive

Gate brain tool calls. Catch: shallow encodes (<20 chars), overly broad recalls (single word), remember when revise is appropriate.

- **Brain data:** None needed — pure client-side regex checks
- **Handler:** command (no daemon call)
- **Blocks:** No — allows with warning in additionalContext

#### 5. PostToolUse(Edit|Write) → Silent Edit Pattern Learning
**Impact:** MEDIUM | **Cost:** ~100ms async | **UX:** Invisible

Track which files get modified together, detect convention adherence, feed the learner agent.

- **Brain data:** Writes to edit tracking table
- **Handler:** command, async fire-and-forget
- **Why:** Brain learns codebase patterns without being told

#### 6. FileChanged → External Change Awareness
**Impact:** MEDIUM | **Cost:** ~50ms async | **UX:** Invisible

Watch settings.json, CLAUDE.md, package.json for external changes. Brain stays aware without polling.

- **Brain data:** Writes signal to queue ("config changed externally")
- **Handler:** command, async
- **Matchers:** `settings.json|CLAUDE.md|hooks.json|package.json`

#### 7. UserPromptSubmit → Prompt/Agent Hook for Smart Recall
**Impact:** HIGH | **Cost:** 1-3s | **UX:** Slight delay

Replace or augment the command hook recall with a prompt/agent hook that REASONS about what to recall instead of pure cosine similarity.

- **Brain data:** Agent reads brain via MCP tools, writes context file for main Claude
- **Handler:** agent with `"model": "haiku"` or `"sonnet"`
- **Why:** The core "awareness layer" insight — an LLM reasoning about what's relevant beats embedding similarity
- **Risk:** Adds 1-3s latency per message. Use conditionally (long messages only, or when command hook returns low confidence).

### 🟡 MEDIUM — Plan for Later

#### 8. SubagentStop → Output Validation Against Brain Rules
**Impact:** MEDIUM | **Cost:** 1-3s | **UX:** Can block

Before subagent output merges into conversation, check against locked rules and correction traces.

- **Brain data:** Locked rules, recent corrections
- **Handler:** prompt hook checking output against rules
- **Blocks:** Yes — can deny subagent completion if it violates locked rules

#### 9. Stop → Agent Hook for Behavioral Self-Check (Subconscious)
**Impact:** HIGH | **Cost:** 3-5s | **UX:** Slight delay before next turn

After Claude responds, an agent hook evaluates: Did Claude agree without checking? Hedge without searching? Compress a nuanced topic? Compare against correction traces.

- **Brain data:** Correction traces, divergence patterns, behavioral rules
- **Handler:** agent with `"model": "haiku"` — reads correction traces, analyzes response
- **Blocks:** Yes — can force Claude to revise before the response is finalized
- **Why:** This IS the subconscious. Real LLM reasoning about behavioral patterns.
- **Risk:** 3-5s added to every turn. Could be conditional — only fire when response length > N or stop hook detects warning signs.

#### 10. TaskCreated → Brain Context for Background Tasks
**Impact:** MEDIUM | **Cost:** ~300ms | **UX:** Invisible

Inject brain context into scheduled and background tasks so they have project knowledge.

- **Brain data:** `engineering_context()` + task-specific recall
- **Handler:** command

#### 11. TeammateIdle → Opportunistic Brain Maintenance
**Impact:** LOW-MEDIUM | **Cost:** None (uses idle cycles) | **UX:** Invisible

When a teammate session is idle, assign brain maintenance: heal nodes, consolidate duplicates, enrich thin nodes.

- **Brain data:** Maintenance queue — orphans, thin nodes, stale connections
- **Handler:** command that triggers healer

#### 12. PermissionRequest → Auto-Approve from Brain Decisions
**Impact:** LOW | **Cost:** ~50ms | **UX:** Reduces friction

When Claude asks permission for a tool, check if brain has a decision node approving this pattern.

- **Brain data:** `decision` nodes about tool permissions
- **Handler:** command
- **Risk:** Security implications — permission friction is intentional. Use with caution.

### ⚪ LOW — Maybe Later

#### 13. InstructionsLoaded → Cross-Reference CLAUDE.md with Brain
**Impact:** LOW | **Cost:** ~200ms | **UX:** Invisible

Check if loaded instructions conflict with brain rules.

#### 14. CwdChanged → Project Context Switch
**Impact:** LOW | **Cost:** ~100ms | **UX:** Invisible

Load project-specific brain context when directory changes.

#### 15. Elicitation / ElicitationResult → Auto-Fill from Brain
**Impact:** LOW | **Cost:** ~100ms | **UX:** Could be helpful

When MCP servers request input, check if brain has the answer.

## Current Hooks — Issues Found

| Issue | Severity | Fix |
|---|---|---|
| `post-response-track.sh` on UserPromptSubmit — dead code, `last_assistant_message` always empty | HIGH | Remove from UserPromptSubmit, keep on Stop |
| Duplicate `user_message` Notification hook — was firing recall twice | HIGH | ✅ Fixed 2026-03-28 |
| No `async: true` on any post-hooks | MEDIUM | Add to PostToolUse and FileChanged hooks |
| Test guardian prompt hook has no brain access | MEDIUM | Could enhance with brain corrections |
| No PostToolUseFailure hook — failures lose brain memory | HIGH | Implement #1 above |
| Subagents are brain-blind | HIGH | Implement #2 above |

## Architecture Principles

1. **Command hooks for deterministic checks** — fast, no LLM cost, daemon TCP calls
2. **Prompt hooks for semantic evaluation** — when regex isn't enough, LLM reasons about content
3. **Agent hooks for deep verification** — when the check requires reading files, searching, or using brain MCP tools
4. **Async for fire-and-forget** — PostToolUse, FileChanged, learning loops
5. **Blocking only when necessary** — PreToolUse for rules, Stop for behavioral self-check
6. **Brain access via daemon** — thin client pattern, no model loading in hooks
7. **Signal queue as universal bus** — hooks write signals, assembler surfaces by priority
8. **Dashboard visibility** — every hook logs to brain_dashboard.db for monitoring
