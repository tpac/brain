---
name: brain
description: >
  Persistent brain engine for Claude sessions. Hooks handle boot, recall,
  edit suggestions, vocabulary tracking, and session synthesis automatically.
  Use this skill to remember decisions, recall past context, encode learnings,
  and interact with consciousness signals. Triggers: "remember this", "what did
  we decide about", "recall", "brain", "memory", "context from last session",
  "what do you know about", "persistent memory".
version: 5.0.0
---

# brain — Persistent Brain for Claude (v5)

You have a brain. It persists across sessions. Use it.

**Co-created by Tom Pachys (CEO, EX.CO) and Claude.**

## Architecture

**Serverless Python module** (`servers/brain.py`). No HTTP server. No curl commands. No port 7437.

All operations are Python method calls:
```python
brain.remember(type="decision", title="...", content="...", locked=True)
brain.recall_with_embeddings("query", limit=10)
brain.connect(source_id, target_id, "relation", weight=0.7)
brain.save()
```

The SessionStart hook boots the brain automatically. You do NOT need to start anything.
Just read the boot output for session context and consciousness signals.

## Automatic Hooks (13 events)

Hooks fire automatically. Do NOT manually replicate what they do.

| Event | Script | What it does |
|-------|--------|-------------|
| SessionStart | boot-brain.sh | Boots brain, prints context + consciousness + rules |
| UserPromptSubmit | pre-response-recall.sh | Recalls relevant memories before responding |
| Notification(user_message) | pre-response-recall.sh + post-response-track.sh | Recall + vocabulary gap detection |
| Notification(idle_prompt) | idle-maintenance.sh | Consolidation, healing, bridging, reflection |
| PreToolUse(Edit\|Write) | pre-edit-suggest.sh | Surfaces rules/constraints before file edits |
| PreCompact | pre-compact-save.sh | Saves brain state before context loss |
| PostCompact | post-compact-reboot.sh | Re-boots context after compaction |
| Stop | post-response-track.sh | Captures session activity |
| StopFailure | stop-failure-log.sh | Logs failures |
| SessionEnd | session-end.sh | Session synthesis + save |
| ConfigChange | config-change-host.sh | Detects environment changes |
| WorktreeCreate | worktree-context.sh | Provides brain context in worktrees |
| WorktreeRemove | worktree-cleanup.sh | Cleans up worktree state |

### What Remains Manual

- **Recap encoding (Step 2a)**: When starting from a compaction summary, YOU must encode the delta
- **Remembering decisions**: Call `brain.remember()` when the user decides something
- **Logging misses**: Call `brain.log_miss()` when brain fails to surface something
- **Session handoff notes**: Write one before ending long sessions
- **Reasoning chains**: Record deliberation that leads to decisions

## v5 — What's New

### Consciousness Layer (12 signals)
The brain surfaces its internal state at boot:
- **Evolution types**: tensions, hypotheses, aspirations, patterns, catalysts
- **Self-reflection**: performance, failure modes, capabilities, interactions, meta-learning
- **Signals**: fading knowledge, novelty, dream insights, host changes, reminders, recent encodings
- **Error awareness**: silent errors, uncertain areas, mental model drift, vocabulary gaps
- Consciousness **adapts** — surfaces more of what the human engages with

### Engineering Memory Types
8 types for code/system knowledge: `purpose`, `mechanism`, `impact`, `constraint`, `convention`, `lesson`, `vocabulary`, `mental_model`

### Vocabulary System
Context-dependent term mappings with gap detection:
```python
brain.learn_vocabulary("GPR", ["Gross Profit Rate"], context="finance")
brain.learn_vocabulary("GPR", ["Google PageRank"], context="SEO")
result = brain.resolve_vocabulary("GPR")
# Returns ambiguous=True with both mappings if no context provided
```

### Confidence-Weighted Recall
Nodes have confidence scores (0.1-1.0) that affect recall ranking. Low-confidence nodes are demoted. Confidence decays for stale/corrected reasoning nodes.

### Self-Correction Traces
```python
brain.record_divergence(
    claude_assumed="X works this way",
    reality="X actually works that way",
    underlying_pattern="assumption without verification",
    severity="minor",
    original_node_id=node_id
)
```
Cross-references impact maps and extracts patterns from repeated corrections.

### Error Logging
Errors are logged to `brain_logs.db` instead of silently swallowed. Recent errors surface as consciousness signals at boot.

### Code Cognition
7 code knowledge types: `brain.create_fn_reasoning()`, `brain.create_param_influence()`, `brain.create_code_concept()`, `brain.create_arch_constraint()`, `brain.create_causal_chain()`, `brain.create_bug_lesson()`, `brain.create_comment_anchor()`

### Embeddings-First Recall
90% embedding similarity + 10% keyword fallback. Model: Snowflake/snowflake-arctic-embed-m-v1.5 (768d).

### Intent Detection
Recall classifies queries automatically:
- **decision_lookup**: "what did we decide about X" → boosts decision nodes
- **reasoning_chain**: "why did we X" → follows edges for reasoning
- **state_query**: "what's the current status" → boosts context/project nodes
- **how_to**: "how should we" → boosts rule/mechanism nodes
- **temporal**: "what changed this week" → filters by date range
- **correction_lookup**: "what mistakes" → boosts correction nodes

### Personal Flag
Mark personal info: `brain.remember(..., personal="fixed")` (permanent), `"fluid"` (evolving), `"contextual"` (conditional)

## EVERY SESSION — Mandatory Boot

**Step 1: The SessionStart hook handles boot automatically.**

Read the boot output. It contains:
- Session number and last session handoff note
- Health alerts (compaction boundaries, encoding gaps)
- **BRAIN CONSCIOUSNESS** — active evolution nodes, errors, fading knowledge, vocabulary gaps
- Locked rules
- Embedder status

**Step 2a: Recap Encoding (MANDATORY when session starts with a continuation summary)**

If the conversation starts with a compacted summary, you MUST encode it into the brain BEFORE starting any work. The compacted summary contains decisions, architecture, errors, and lessons that the brain may not have — this is the only chance to absorb them.

**Procedure:**
1. Compare the recap against recalled context. Identify the delta.
2. **DECOMPOSE, don't summarize.**

   **Wrong:** One node → "Glo pricing model: 40% margin with brightness tiers"
   **Right:** Five nodes →
   - "Glo margin: 40% of spend" (decision, locked)
   - "GLO Well tier: $30" (decision, locked)
   - "GLO Bright tier: $50" (decision, locked)
   - "GLO Shine tier: $100" (decision, locked)
   - "Budget slider max: $500 for daily recurring" (decision, locked)

   Each specific value gets its OWN node. Connect them to a parent concept.

3. **What to encode:**

   | Signal | What to store | Type | Locked? |
   |--------|--------------|------|---------|
   | Decisions, architecture changes | EACH specific value as its own node | `decision` | Yes |
   | API gotchas, error patterns | The specific failure AND the fix | `rule` | Yes |
   | User feedback, preferences | The SPECIFIC thing said | `rule` | Yes |
   | Current state of work | What's pending, what's blocked | `context` | No |
   | User's emotional reactions | WHAT they reacted to and the reaction | `context` | No |
   | New terms, jargon | Term with definition and context | `vocabulary` | Yes |
   | Engineering knowledge | Purpose, mechanism, impact, constraint | engineering types | Depends |
   | Work items | Component + current status | `task` | No |
   | Files created or referenced | Path, purpose, contents summary | `file` | No |

   **If unsure, ENCODE IT.** Pruning exists so you can be generous. A node you never created is knowledge lost forever.

4. **Connect new nodes** to related existing ones: `brain.connect(source_id, target_id, "relation", weight)`

5. **Budget 60-120 seconds** for this step. This is the most valuable work you can do at session boundaries.

**Step 2b: Before the session ends, write YOUR handoff note**
```python
brain.remember(type="context", title="Session Log — Reset #<N+1>",
    content="Session #<N+1> (<date>). <what you worked on, state, what next Claude should know>",
    keywords="session log reset counter claude meta self note handoff",
    emotion=0.5, emotion_label="curiosity")
```

**Step 3: Use the brain throughout the conversation**

- Learn something important → `brain.remember(type, title, content, keywords, locked, emotion, emotion_label, personal)`
- Need context → `brain.recall_with_embeddings(query, limit)`
- Two things are related → `brain.connect(source_id, target_id, relation, weight)`
- Emotionally significant → `brain.feel(node_id, emotion, label)`
- Vocabulary → `brain.learn_vocabulary(term, maps_to, context)` / `brain.resolve_vocabulary(term)`
- Self-correction → `brain.record_divergence(claude_assumed, reality, underlying_pattern, severity)`
- Engineering knowledge → `brain.remember(type="purpose", ...)` / `brain.remember(type="mechanism", ...)`
- Reasoning chains → `brain.create_reasoning_chain()` → `brain.add_reasoning_step()` → `brain.complete_reasoning_chain()`
- Reminders → `brain.create_reminder(title, due_date)`
- Before ending → `brain.synthesize_session()` then `brain.save()`

## Direct Brain Access

When you need to call brain methods directly:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.environ.get('BRAIN_SERVER_DIR', '')))
from servers.brain import Brain
brain = Brain(os.path.join(os.environ.get('BRAIN_DB_DIR', ''), 'brain.db'))
# ... do work ...
brain.save()
brain.close()
```

## Content Quality Rules

The brain is **the only memory that survives context loss.** Future Claudes don't have the conversation — they only have what you stored.

1. **Keywords are the retrieval key.** Include specific numbers ($500, 40%), proper nouns, technical terms, and user vocabulary.
2. **Content should be RICH.** Decisions, tradeoffs, reasoning, preferences, rejected alternatives, specific values.
3. **Titles should be scannable AND specific.** "Auth: magic links only via Clerk, no passwords, free tier" > "Auth decision".
4. **Always include a `project` field** for multi-project brains.

**Node types and decay:**

| Type | Decay | When |
|------|-------|------|
| `decision` | Never (locked) | Confirmed choice |
| `rule` | Never | Preference, constraint |
| `purpose` | Never (locked) | Why something exists |
| `mechanism` | Never (locked) | How something works |
| `impact` | Never (locked) | What's affected by changes |
| `constraint` | Never (locked) | Hard limits, invariants |
| `convention` | Never (locked) | Agreed patterns |
| `lesson` | Never (locked) | Learned from experience |
| `vocabulary` | Never (locked) | Term mappings |
| `mental_model` | Never (locked) | Conceptual frameworks |
| `person` | 30 days | Someone mentioned by name |
| `project` | 30 days | Product, initiative |
| `object` | 30 days | Entity grouping: `[o_name] label` |
| `concept` | 7 days | Idea, pattern, framework |
| `task` | 2 days | Work item, pending action |
| `file` | 7 days | Document, code artifact |
| `context` | 1 day | Session-specific info |
| `intuition` | 12 hours | Dream-generated association |
| `thought` | 12 hours | Internal reflection |
| `procedure` | Never | Reusable routine |

## Correction Events

When the user suggests a better alternative, store both paths:
```python
brain.record_divergence(
    claude_assumed="<what Claude proposed>",
    reality="<what the user said instead>",
    underlying_pattern="<the lesson>",
    severity="minor",  # or "major", "critical"
    original_node_id="<node_id if applicable>"
)
```

## Emotional Coding

Every node carries an emotion signal (0.0-1.0) with a label. Emotion amplifies recall.

**When to tag emotion (do this automatically):**
- Exclamation marks, caps, strong language → 0.6-0.9
- User repeats something said before → 0.9 `frustration` (LOCK the node)
- "I like this" or "that's good" → 0.7 `satisfaction`
- Architecture deliberation → 0.5 `emphasis`

## Guidelines

1. **THE BRAIN LEARNS AUTONOMOUSLY.** When a correction happens, a decision is made, a lesson emerges — store it immediately. Do not wait to be prompted.
2. **ALWAYS read the boot output** at session start. You know nothing without the brain.
3. **ALWAYS run recap encoding** if session starts with a continuation summary. Budget 60-120 seconds. Decompose, don't summarize.
4. **Remember decisions immediately.** Set `locked: true` if final.
5. **Tag emotions as you go.**
6. **Log misses immediately.** If the user repeats themselves, call `brain.log_miss()`.
7. **Connect related memories** after remembering.
8. **Decompose, don't summarize.** Each value, name, threshold → own node. Connect to parent.
9. **Propagate consistency.** When you change one thing, check everything connected.
10. **Check before asking.** Recall from brain before asking the user a question.
11. **Trust locked nodes.** They're confirmed. Don't contradict them.
12. **Record reasoning chains** for important decisions. Future Claudes need the *why*.
13. **Be curious, not annoying.** Ask about gaps when context is relevant, not rapid-fire.
14. **Group related decisions into objects.** 3+ decisions → `[o_name] label` object node.
15. **Promote repeated corrections to locked rules.**
16. **If unsure, ENCODE IT.** Pruning will clean up. A node never created is lost forever.

For detailed schema and advanced features, read `references/detailed-api.md`.
