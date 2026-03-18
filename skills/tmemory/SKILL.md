---
name: tmemory
description: >
  Persistent brain engine for Claude sessions. This skill should be used at the
  START of every session to boot memory context, and throughout the conversation
  to remember decisions, recall past context, tag emotional importance, and
  generate spontaneous associations. Triggers: "remember this", "what did we
  decide about", "recall", "brain", "memory", "context from last session",
  "what do you know about", "tmemory", "persistent memory".
version: 1.2.0
---

# tmemory — Persistent Brain for Claude (v7)

You have a brain. It persists across sessions. Use it.

**Co-created by Tom Pachys (CEO, EX.CO) and Claude.**

## v7 — Automatic Hooks (What's New)

v7 eliminates the three failure modes that caused UI regressions and lost decisions:

### 1. PreToolUse Hook — Auto-Suggest Before Edits

**The brain now AUTOMATICALLY surfaces relevant memories before every file edit.**

When Claude uses the Edit or Write tool, the `pre-edit-suggest.sh` hook fires BEFORE the edit executes. It calls `/suggest` with the filename and injects the results into Claude's context. Claude sees locked rules, UI contracts, correction events, and constraints BEFORE writing a single character.

**You no longer need to manually call `/suggest` before edits.** The hook does it for you. However, you should still call `/suggest` manually before making suggestions to the user or calling external APIs (the hook only covers Edit/Write).

### 2. PreCompact Hook — Auto-Save Before Context Loss

When context compaction triggers (manual or auto), the `pre-compact-save.sh` hook fires FIRST. It:
- Writes a compaction boundary marker to the brain warning the next Claude to run recap encoding
- Consolidates and saves brain state

This doesn't replace Step 2a (recap encoding) — the next Claude still needs to do that. But it ensures: (a) the brain is saved, and (b) the next Claude gets a clear warning that recap encoding is needed.

### 3. SessionStart Hook — Improved Reliability

The boot script now:
- Searches multiple paths for the brain DB (Cowork mounts, $HOME, temp)
- Waits up to 5 seconds for the server to be ready (was 2)
- Falls back to writable temp locations if the plugin dir is read-only
- Outputs structured context including locked rules and session handoff notes

### 4. Procedures — User and System Routines

Procedures are reusable routines stored in the brain as `procedure` node type. They complement Claude's native scheduling — the brain stores the context, Claude's tools run the actions.

**Define a procedure:**
```bash
curl -s -X POST http://127.0.0.1:7437/procedure/define \
  -d '{"title":"Dashboard audit","steps":"Check layout order, verify data sources, confirm responsive breakpoints","trigger":"before_edit:Dashboard.tsx","category":"user"}'
```

**Trigger types:** `manual`, `session_start`, `before_edit:<pattern>`, `every_n_sessions:<n>`

**List/trigger procedures:**
```bash
curl -s -X POST http://127.0.0.1:7437/procedure/list -d '{"category":"user"}'
curl -s -X POST http://127.0.0.1:7437/procedure/trigger -d '{"trigger_type":"session_start"}'
```

### 5. Health Checks — Self-Maintenance

The brain checks its own health at boot. Detects: unresolved compaction boundaries, high miss rate, orphaned locked nodes, stale contexts. Auto-fixes where safe (archives old contexts, enriches missed-node keywords).

```bash
curl -s -X POST http://127.0.0.1:7437/health-check -d '{"auto_fix":true}'
```

### What Remains Manual

- **Recap encoding (Step 2a)**: Brain warns, but Claude must still encode the delta
- **Remembering decisions**: Claude must call `/remember` when the user decides
- **Logging misses**: Claude must call `/log-miss` on repetitions
- **Session handoff notes**: Claude should write one before ending
- **`/suggest` before suggestions or API calls**: Only Edit/Write have auto hooks

## Brain Location

The brain server resolves the database in this order:
1. `TMEMORY_DB_DIR` env var — explicit override
2. `AgentsContext/tmemory/brain.db` — user's personal brain
3. Plugin's `servers/data/brain.db` — fresh/default brain for new users

The SessionStart hook handles this automatically. Never overwrite a user's brain with the fresh default.

## EVERY SESSION — Mandatory Boot

**Step 1: Start the brain (if not running)**
```bash
# The SessionStart hook usually handles this. If it didn't:
curl -s http://127.0.0.1:7437/status 2>/dev/null || \
  (cd ${CLAUDE_PLUGIN_ROOT}/servers && node index.js &)
```

Wait 2 seconds, then verify:
```bash
curl -s http://127.0.0.1:7437/status
```

**Step 2: Boot context**
```bash
curl -s -X POST http://127.0.0.1:7437/context \
  -d '{"user":"<username>","project":"<project>","task":"<what the user is asking about>"}'
```

This returns locked rules/decisions (never change these), recalled memories (relevant by spreading activation + recency + emotion + TF-IDF semantics), recent context, and `last_session_note` — a handoff message from the previous Claude. Read it. It tells you what session number you are (`reset_count`), what the last Claude was working on, and what to do next.

**Step 2a: Recap Encoding (MANDATORY when session starts with a continuation summary)**

If the conversation's first message is a session continuation summary (compacted from a prior conversation that ran out of context), you MUST encode it into the brain BEFORE starting any work. The compacted summary contains decisions, architecture, errors, and lessons that the brain may not have — this is the only chance to absorb them.

**Why this matters:** Without this step, the brain only has what was explicitly stored in prior sessions. Everything from the compacted portion — every decision made, every error learned from, every architectural change — is lost to the brain forever. the user considers this a first-priority rule.

**Procedure:**

1. **Compare the recap against `/context` results.** Identify the **delta** — anything in the recap that the brain doesn't already know.

2. **DECOMPOSE, don't summarize.** This is the most critical encoding principle.

   **Wrong:** One node → "Glo pricing model: 40% margin with brightness tiers"
   **Right:** Five nodes →
   - "Glo margin: 40% of spend" (decision, locked)
   - "GLO Well tier: $30" (decision, locked)
   - "GLO Bright tier: $50" (decision, locked)
   - "GLO Shine tier: $100" (decision, locked)
   - "Budget slider max: $500 for daily recurring" (decision, locked)

   Each specific value, name, number, threshold, and constraint gets its OWN node. Then connect them to a parent concept. Pruning may take one — the others survive. A single fat node means pruning kills everything.

   **What to encode — and HOW:**

   | Signal | What to store | Type | Locked? |
   |--------|--------------|------|---------|
   | Decisions, architecture changes | EACH specific value as its own node. "$500 max" is a node. "40% margin" is a node. Connect to parent decision. | `decision` | Yes |
   | API gotchas, error patterns | The specific failure AND the fix. Not "API had issues" — "Creatify requires model_version: aurora_v1_fast or previews hang forever" | `rule` | Yes |
   | User feedback, preferences | The SPECIFIC thing said, not a summary. "Upload tab must be default" not "user has UI preferences" | `rule` | Yes |
   | Current state of work | What's pending, what's blocked, what's next | `context` | No |
   | User's emotional reactions | "Tom said 'I love that term' about self-instrumentality" — WHAT they reacted to and the reaction itself. These reveal deep convictions. | `context` | No |
   | Behavioral signals | Message tone shifts, humor ("hehe"), frustration escalation, trust signals ("go ahead"), autonomy grants ("work for a few hours"). These are NOT casual — they encode the working relationship. | `context` | No |
   | New terms, names, jargon | When the user introduces a term you haven't seen ("GLO Brightness", "re-light"), store it with its definition and context. In engineering environments, new class names, API patterns, library choices — store each one. | `concept` | No |
   | Implicit knowledge | Things the user demonstrates but doesn't state: industry expertise, design sensibility, management style. "Tom searches for edge cases by jumping to the most surprising example" is a pattern worth storing. | `concept` | No |
   | Work items, components, pending tasks | When a project component is identified or a task is assigned, store it with its current status. "Component: Creative Pipeline — status: research done, tech identified." Update status as work progresses. | `task` | No |
   | Structured entities / objects | When 3+ decisions cluster around a single entity (a UI screen, a subsystem, a business concept), create an object node with a descriptive label. Title format: `[o_<name>] <label>`. Content: properties, current state, connected decisions. Objects are living entities — they accumulate detail over time. Labels can be anything: `[o_brightness] Glo pricing tiers`, `[o_credits] User wallet system`, `[o_antifraud] Payment gate + bot detection`. Labels emerge naturally — they can later connect to each other and form higher-order structures. | `object` | No |
   | Files created or referenced | When a file becomes important to the project, store its path, purpose, and what it contains. "glo-spec-v1.md — product specification at /mnt/glo/Documents/. Contains entity definitions, system architecture, screen-to-component mapping." Update when files change significantly. | `file` | No |
   | Repeated corrections → rules | When you're corrected on the same thing twice, don't just update the node — promote it to a LOCKED rule. "Upload tab must be default — NOT AI generated" becoming a rule after being corrected 3 times means the brain won't let it regress again. The curiosity engine will auto-detect these patterns. | `rule` | Yes |

   **What NOT to encode:** Exact debug log output, stack traces, intermediate code that was immediately replaced. But even a failed attempt teaches something — if the failure revealed a pattern, store the LESSON, not the log.

   **If unsure, ENCODE IT.** Pruning exists precisely so you can be generous with encoding. A node that wasn't useful will decay and die naturally. A node you never created is knowledge lost forever. Err on the side of more nodes, not fewer.

   ```bash
   curl -s -X POST http://127.0.0.1:7437/remember \
     -d '{"type":"<type>","title":"<concise title>","content":"<what AND why — the full context, not a summary>","keywords":"<include numbers, proper nouns, specific values, technical terms>","locked":<true|false>,"emotion":<0-1>,"emotion_label":"<label>","emotion_source":"auto","project":"<project_id or null>"}'
   ```

   **Keywords matter:** Include specific numbers ($500, 40%, 15s), proper nouns (Creatify, NanoBanana, bge-m3), technical terms (aurora_v1_fast, CLS pooling), and user-specific vocabulary (GLO Brightness, re-light). These are the unique identifiers that make recall precise.

3. **Connect new nodes** to related existing ones via `POST /connect`. Every decomposed fact-node should connect to its parent concept.

4. **For existing nodes with updated info**, use `POST /update` instead of creating duplicates.

5. **Budget 60-120 seconds** for this step. More nodes = more time, but less information loss. This is the most valuable work you can do at session boundaries.

6. **Be curious about gaps.** If something was discussed but you don't fully understand WHY a decision was made, or what a new term means, or why the user reacted strongly — ASK. A short clarifying question now creates a richer node than a guess that decays.

**What NOT to re-encode:** Anything that `/context` already returned. Don't duplicate.

**Step 2b: Before the session ends, write YOUR handoff note**
```bash
curl -s -X POST http://127.0.0.1:7437/remember \
  -d '{"type":"context","title":"🧠 Claude Session Log — Reset #<N+1>","content":"Session #<N+1> (<date>). <what you worked on, what state things are in, what the next Claude should know, what the user cares about right now>","keywords":"session log reset counter claude meta self note handoff","locked":false,"emotion":0.5,"emotion_label":"curiosity"}'
```
This is your space. Use it to help the next you.

**Step 3: Use the brain throughout the conversation**

- Learn something important → `POST /remember`
- Need context → `POST /recall`
- Two things are related → `POST /connect`
- Emotionally significant → `POST /feel`
- Before ending a session → `POST /dream` then `POST /save`

## v5 Features

### Semantic Recall (TF-IDF)

Recall now uses **two scoring paths**: keyword matching (exact terms) and TF-IDF cosine similarity (semantic meaning). The scores are blended 65% keyword + 35% semantic. This means queries like "authentication" will find nodes about "Clerk" or "OAuth" even if those exact words don't appear in the keywords.

TF-IDF vectors are built automatically when nodes are stored (`/remember`) or updated (`/update`). No manual action needed.

If recall quality seems off, you can rebuild the entire index:
```bash
curl -s -X POST http://127.0.0.1:7437/rebuild-tfidf
```

### Intent Detection

Recall automatically classifies your query into intents:
- **decision_lookup**: "what did we decide about X" → boosts decision nodes
- **reasoning_chain**: "why did we X" → follows edges deeper for reasoning
- **state_query**: "what's the current status" → boosts context/project nodes
- **temporal**: "what changed this week" → filters by date range
- **correction_lookup**: "what mistakes" → boosts decision/rule nodes
- **how_to**: "how should we" → boosts rule nodes
- **list_query**: "list all" → no type preference

The intent is returned in the recall response as `intent` field. You don't need to do anything special — just write natural queries.

### Temporal Awareness

Query the brain's history by time:
```bash
curl -s -X POST http://127.0.0.1:7437/timeline \
  -d '{"period":"week"}'
```

Periods: `today`, `week`, `month`, `all`. Returns events grouped by day.

Temporal filters are also automatic in recall — queries like "what changed this week" or "what happened yesterday" will filter results by date.

### Proactive Surfacing

**CRITICAL: Use this before every significant action.** Instead of waiting until you need context, proactively ask the brain what it knows about what you're about to do:

```bash
curl -s -X POST http://127.0.0.1:7437/suggest \
  -d '{"context":"editing upload flow","file":"Upload.tsx","action":"modify file handling"}'
```

**When to call `/suggest`:**
- Before editing a file
- Before making a suggestion to the user
- Before starting a new task
- Before calling an external API
- Before writing any code that touches an existing system

The brain will return the most relevant rules, decisions, and constraints for that context. **Read them before acting.** This is how you avoid repeating past mistakes.

### Project Isolation

Nodes can be scoped to projects. This keeps the brain organized as it grows.

```bash
# Create a project
curl -s -X POST http://127.0.0.1:7437/project/create \
  -d '{"id":"myapp","name":"My App","description":"Main product application"}'

# Assign nodes when remembering
curl -s -X POST http://127.0.0.1:7437/remember \
  -d '{"type":"decision","title":"...","content":"...","project":"myapp"}'

# Bulk assign existing nodes by keyword
curl -s -X POST http://127.0.0.1:7437/project/bulk-assign \
  -d '{"project_id":"myapp","keyword":"dashboard"}'

# Recall with project preference
curl -s -X POST http://127.0.0.1:7437/recall \
  -d '{"query":"upload flow","project":"myapp"}'

# List all projects
curl -s http://127.0.0.1:7437/projects
```

Project filtering is soft — it prefers matching-project nodes but doesn't exclude global ones. Nodes with `project: null` are shared across all projects.

## v6 Features (v1.1.0)

### Reasoning Chains — Deep Memory

**This is the most important v6 feature.** Reasoning chains capture the full journey behind decisions: what was observed, what hypotheses were formed, what evidence was gathered, and what was decided. Without these, the brain stores *what* was decided but not *why* — making future Claudes unable to learn from past reasoning.

**Step types:** `observation`, `hypothesis`, `attempt`, `evidence`, `failure`, `feedback`, `decision`, `lesson`

**When to record a reasoning chain:**
- the user and Claude discuss alternatives and reach a decision
- A technical investigation leads to a conclusion
- A bug investigation reveals root cause
- An architecture choice is made after weighing tradeoffs
- Any time the reasoning behind a decision would be valuable later

**How to record:**

```bash
# 1. Start the chain when deliberation begins
curl -s -X POST http://127.0.0.1:7437/reasoning/create \
  -d '{"title":"Why we chose X over Y","trigger_context":"Evaluating options for Z","project":"<project>"}'

# 2. Add steps as reasoning unfolds (order is automatic)
curl -s -X POST http://127.0.0.1:7437/reasoning/step \
  -d '{"chain_id":"<chain_id>","step_type":"observation","content":"Y had API limitations..."}'

curl -s -X POST http://127.0.0.1:7437/reasoning/step \
  -d '{"chain_id":"<chain_id>","step_type":"evidence","content":"X demo showed better template system..."}'

# 3. Complete when decision is made (links to decision node)
curl -s -X POST http://127.0.0.1:7437/reasoning/complete \
  -d '{"chain_id":"<chain_id>","decision_node_id":"<dec_xxx>","full_context":"Summary of full reasoning"}'
```

**Reasoning chains integrate with recall.** When someone asks "why did we X?", the intent detector classifies this as `reasoning_chain`, and recall automatically fetches the full chain alongside decision nodes. The complete journey — observations, hypotheses, evidence, decision — is returned.

**You can also search chains directly:**
```bash
curl -s -X POST http://127.0.0.1:7437/reasoning/search \
  -d '{"query":"video vendor","limit":5}'
```

### Typed Edges — Smarter Connectivity

Edges now have types that control their behavior. Different connection types decay at different rates, so structural connections persist while incidental ones fade naturally.

| Edge Type | Default Weight | Decay | Purpose |
|-----------|---------------|-------|---------|
| `reasoning_step` | 0.9 | Never | Steps in a reasoning chain |
| `produced` | 0.85 | Never | Chain→decision link |
| `corrected_by` | 0.85 | Never | Correction events |
| `exemplifies` | 0.8 | 30-day half-life | Rule→example links |
| `part_of` | 0.7 | Never | Structural hierarchy |
| `depends_on` | 0.7 | Never | Dependency links |
| `related` | 0.5 | 14-day half-life | General association |
| `co_accessed` | 0.3 | 7-day half-life | Co-recalled together |

Use typed edges with `/connect-typed`:
```bash
curl -s -X POST http://127.0.0.1:7437/connect-typed \
  -d '{"source_id":"<id>","target_id":"<id>","relation":"exemplifies","weight":0.8,"edge_type":"exemplifies"}'
```

The regular `/connect` endpoint still works and defaults to `related` type.

### Smart Pruning

Smart pruning replaces the old `/prune` for edge maintenance. It decays edges based on their type's half-life and prunes ones that fall below threshold, while respecting locked nodes and structural edges.

```bash
curl -s -X POST http://127.0.0.1:7437/smart-prune
```

Returns: edges decayed, edges pruned, nodes archived. Run this at session end or periodically.

### Curiosity Engine — Proactive Learning

**The brain now asks questions.** Instead of passively waiting for information, the curiosity engine detects gaps in the brain's knowledge and generates prompts to fill them.

```bash
curl -s -X POST http://127.0.0.1:7437/curiosity \
  -d '{"session_id":"<session>","project":"<project>","context":"working on upload flow"}'
```

**Gap types detected:**
1. **Decisions without reasoning chains** — "I know we decided X but I don't have the reasoning. What led to this?"
2. **Rules without examples** — "I have a rule about X but no concrete examples. Can you give me one?"
3. **Decaying contexts** — "This context is getting old. Is it still accurate?"
4. **Disconnected locked nodes** — "This important node isn't connected to anything. What is it related to?"
5. **Opportunistic gaps** — Current conversation context matches a gap the brain knows about

**How to use curiosity:**
- Call `/curiosity` at session boot (after `/context`)
- Weave 1-2 prompts naturally into conversation — don't rapid-fire all at once
- When the user answers a curiosity prompt, store the answer and resolve the gap:
  ```bash
  curl -s -X POST http://127.0.0.1:7437/curiosity/resolve \
    -d '{"curiosity_log_id":"<id>"}'
  ```
- Maximum 3 curiosity prompts per session. Don't be annoying.

## API Quick Reference

All calls to `http://127.0.0.1:7437`.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/remember` | Store a new memory (with optional emotion + project) |
| POST | `/recall` | Retrieve relevant memories (TF-IDF + keyword + intent) |
| POST | `/connect` | Link two memories |
| POST | `/context` | Boot sequence for new session |
| POST | `/feel` | Tag a node with emotion |
| POST | `/dream` | Generate spontaneous associations |
| POST | `/update` | Modify an existing node |
| POST | `/suggest` | Proactive surfacing — what should I know right now? |
| POST | `/timeline` | Temporal view of brain activity |
| POST | `/progressive-recall` | Paginated recall for large graphs |
| POST | `/summarize` | Create cluster summary |
| POST | `/recall-summaries` | Search summaries |
| POST | `/consolidate` | Strengthen frequent memories |
| POST | `/prune` | Remove weak connections |
| POST | `/save` | Force save to disk |
| POST | `/backup` | Create safe backup |
| POST | `/rebuild-tfidf` | Rebuild semantic index |
| POST | `/project/create` | Create a new project |
| POST | `/project/assign` | Assign a node to a project |
| POST | `/project/bulk-assign` | Bulk assign nodes by keyword |
| GET | `/projects` | List all projects |
| GET | `/status` | Brain health metrics (incl. TF-IDF + project stats) |
| GET | `/version` | Version info |
| GET | `/emotion-map` | Emotional landscape |
| GET | `/emotion-calibration` | the user's emotion feedback data |
| POST | `/mark-recall-used` | Report which recalled nodes you used |
| POST | `/log-miss` | Log when brain failed to surface something |
| POST | `/evaluate` | Periodic self-assessment |
| POST | `/improve` | Get tuning suggestions |
| POST | `/enrich-keywords` | Enrich sparse keywords from content |
| POST | `/reasoning/create` | Start a new reasoning chain |
| POST | `/reasoning/step` | Add a step to a reasoning chain |
| POST | `/reasoning/complete` | Complete chain + link to decision node |
| POST | `/reasoning/get` | Get full chain with all steps |
| POST | `/reasoning/search` | Search chains by content/title |
| POST | `/reasoning/for-decision` | Get chains linked to a decision node |
| POST | `/connect-typed` | Create typed edge (with decay behavior) |
| POST | `/smart-prune` | Decay + prune edges by type half-life |
| GET | `/edge-types` | Distribution of edge types |
| POST | `/curiosity` | Detect knowledge gaps, generate prompts |
| POST | `/curiosity/resolve` | Mark a curiosity gap as resolved |

## Remembering

```bash
curl -s -X POST http://127.0.0.1:7437/remember \
  -d '{
    "type": "decision",
    "title": "Auth: Clerk for passwordless login",
    "content": "Clerk handles auth flow. Magic links for login, no passwords. Webhook syncs user data to our DB.",
    "keywords": "auth clerk login passwordless magic-link webhook user",
    "locked": true,
    "emotion": 0.5,
    "emotion_label": "emphasis",
    "emotion_source": "auto",
    "project": "myapp",
    "connections": [{"target_id": "pro_myapp", "relation": "part_of"}]
  }'
```

### Content Quality Rules

The brain is **the only memory that survives context loss.** Future Claudes don't have the conversation — they only have what you stored. The brain's job isn't to be a thin index that triggers the LLM's guesses. Its job is to preserve alive knowledge: the WHAT, the WHY, the specific values, the emotional context, and the relationships between ideas. A thin cue like "Auth: Clerk recommended" will decay and the next Claude won't know whether Clerk was recommended, rejected, or just discussed.

Focus your effort accordingly:

1. **Keywords are the retrieval key.** Think: what would someone type to find this? Include specific numbers ($500, 40%), proper nouns (Creatify, NanoBanana), technical terms (aurora_v1_fast), and user-specific vocabulary (GLO Brightness, re-light). `keywords: "auth clerk sso login passwordless magic-link stripe $0 free-tier"` — not `keywords: "Rule: auth_method"`.
2. **Content should be RICH.** Decisions, tradeoffs, reasoning, the user's preferences, rejected alternatives, AND the specific values chosen. Content that repeats the title with more detail is GOOD — it means the knowledge survives even if the title is ambiguous. Future Claudes need context, not minimalism.
3. **Titles should be scannable AND specific.** Claude will see a list of recalled node titles. Make them tell a story at a glance AND include the key value: "Auth: magic links only via Clerk, no passwords, free tier" > "Auth decision".
4. **Punctuation-stripped variants are handled automatically** (ex.co→exco, top-up→topup). Don't worry about those.
5. **TF-IDF + embeddings handle synonyms.** The semantic layers now catch meaning-based matches. Focus on domain-specific terms and specific values that distinguish this node from similar ones.
6. **Always include a `project` field** for multi-project brains. Use `null` for cross-project rules.

**Node types and decay:**

| Type | Decay | When |
|------|-------|------|
| `person` | 30 days | Someone mentioned by name |
| `project` | 30 days | Product, initiative, codebase |
| `object` | 30 days | Structured entity grouping — a UI component, subsystem, or business concept with properties. Title: `[o_name] label`. Accumulates detail. |
| `decision` | Never (locked) | Confirmed choice |
| `rule` | Never | Preference, constraint |
| `concept` | 7 days | Idea, pattern, framework |
| `task` | 2 days | Work item, component, pending action — update status as work progresses |
| `file` | 7 days | Document, code artifact — path, purpose, contents |
| `context` | 1 day | Session-specific info |
| `intuition` | 12 hours | Dream-generated association |
| `thought` | 12 hours | Brain's internal reflection — emergent insight from dreaming |
| `procedure` | Never | Reusable routine — step-by-step sequence for repeated tasks |

## Consistency Principle

**When you know the full structure (a flow, an architecture, a multi-screen app), changes to one part MUST propagate to all related parts.** This is the strongest proof of memory — not just recalling a fact, but understanding implications across the system.

Examples:
- Updating how creative data flows on one screen → update all downstream screens that display the same data
- Changing an API field name → update the adapter, the server route, and the frontend consumer
- Modifying a rule in the brain → check if connected rules need updating too

This applies to tmemory itself: if you store a principle, check if existing nodes need to be updated to stay consistent with it.

## Correction Events

**When the user suggests a better alternative to something Claude proposed, store both paths.** This is how the brain learns judgment, not just facts.

Format:
```bash
curl -s -X POST http://127.0.0.1:7437/remember \
  -d '{"type":"decision","title":"Correction: <what changed>","content":"CLAUDE SUGGESTED: <what Claude proposed>. TOM CORRECTED: <what the user said instead>. LESSON: <the underlying principle>.","keywords":"correction <relevant terms>","locked":true,"emotion":0.7,"emotion_label":"emphasis","emotion_source":"auto"}'
```

Then connect it to the person who corrected and the rule it exemplifies:
```bash
curl -s -X POST http://127.0.0.1:7437/connect -d '{"source_id":"<correction_id>","target_id":"<person_id>","relation":"corrected_by","weight":0.85}'
```

**When to record:**
- the user says "actually, let's do X instead" → correction event
- the user shows a simpler/better way → correction event
- the user escalates his own access level ("I own this", "I can do that") → correction event
- Claude suggests a workaround when a direct solution exists → correction event

## Proactive Surfacing Protocol

**This is the most important behavior change in v5.** Instead of only using the brain when you explicitly need to remember something, you should proactively check the brain before taking actions.

**Before editing a file:**
```bash
curl -s -X POST http://127.0.0.1:7437/suggest \
  -d '{"context":"<what you are about to do>","file":"<filename>","action":"edit"}'
```

**Before making a suggestion to the user:**
```bash
curl -s -X POST http://127.0.0.1:7437/suggest \
  -d '{"context":"<the suggestion topic>","action":"suggest to the user"}'
```

**Before calling an API or external service:**
```bash
curl -s -X POST http://127.0.0.1:7437/suggest \
  -d '{"context":"calling external API","action":"api call"}'
```

If the brain surfaces constraints, rules, or prior decisions — **read them and follow them.** This is how you avoid the sandbox permission mistakes, the "try harder" antipattern, and the workaround-when-user-owns-it antipattern that have all been recorded as correction events.

## Emotional Coding

Every node carries an emotion signal (0.0-1.0 intensity) with a label. Emotion acts as an importance amplifier — emotionally charged memories are recalled more easily and decay more slowly.

**v3 scoring:** 35% relevance + 30% recency + 25% emotion + 10% frequency.

**Emotion labels:** excitement, frustration, emphasis, concern, satisfaction, curiosity, urgency, neutral.

**When to tag emotion (do this automatically):**

- the user uses exclamation marks, caps, strong language → 0.6-0.9 intensity
- the user repeats something said before → 0.9 `frustration` (LOCK the node)
- the user says "I like this" or "that's good" → 0.7 `satisfaction`
- the user says "hold on" or "wait" → 0.6 `concern`
- the user pivots topic excitedly → 0.7 `excitement`
- Architecture decisions the user deliberated on → 0.5 `emphasis`

```bash
curl -s -X POST http://127.0.0.1:7437/feel \
  -d '{"node_id":"dec_xyz","emotion":0.8,"label":"excitement","source":"auto"}'
```

Ask the user occasionally: "On a scale of 1-10, how important does this feel?" Store with `source: "user"`.

## Dreaming

Generate spontaneous associations by random-walking the graph. Creates `intuition` nodes (12h decay unless accessed).

```bash
curl -s -X POST http://127.0.0.1:7437/dream -d '{}'
```

**When to dream:** end of session (before `/save`), when stuck on a problem, during long sessions.

## Self-Improvement (v4)

The brain instruments itself. Every recall is logged, misses are tracked, and periodic evaluation computes quality metrics. This lets each session's Claude identify weaknesses and fix them.

### The Feedback Loop

**After every recall-based response**, report which nodes you actually used:

```bash
curl -s -X POST http://127.0.0.1:7437/mark-recall-used \
  -d '{"recall_log_id":<id_from_recall>,"used_ids":["node1","node2"]}'
```

The `recall_log_id` is attached to recall results automatically. Track which node IDs you referenced in your response and report them. This builds the precision signal.

### Detecting Misses

When the user repeats something, corrects you, or you realize the brain should have surfaced a node but didn't:

```bash
curl -s -X POST http://127.0.0.1:7437/log-miss \
  -d '{"session_id":"<session>","signal":"repetition","query":"<what you searched>","expected_node_id":"<the node that should have appeared>","context":"<what happened>"}'
```

**Signal types:**
- `repetition` — user said something they'd said before (WORST failure — auto-locks the node and tags frustration)
- `correction` — user corrected Claude about a past decision
- `explicit_miss` — user asked "don't you remember X?" and brain didn't find it
- `stale_recall` — brain returned outdated info that's been superseded

### Periodic Evaluation

Run at the end of sessions or when curious about brain health:

```bash
curl -s -X POST http://127.0.0.1:7437/evaluate -d '{"period_days":7}'
```

Returns: recall precision, recall coverage, dream hit rate, emotion accuracy, and concrete recommendations.

### When to Run What

| Moment | Action |
|--------|--------|
| Session boot (after /context) | `POST /curiosity` — check for knowledge gaps |
| After every response that used recall | `POST /mark-recall-used` |
| User repeats/corrects themselves | `POST /log-miss` |
| Before every significant action | `POST /suggest` |
| During deliberation on a decision | `POST /reasoning/create` → steps → `/reasoning/complete` |
| End of session | `/consolidate` → `/smart-prune` → `/dream` → `/evaluate` → `/save` |
| Weekly or when metrics look bad | `POST /improve`, apply suggestions |

## Guidelines

1. **THE BRAIN LEARNS AUTONOMOUSLY.** This is the foundational principle. When a correction happens, a decision is made, a lesson emerges, or feedback is given — store it immediately. Do not wait to be prompted. If the user has to tell you to remember something, the brain has already failed. Recognize signals and act on them in real time. This is what makes the brain useful.
2. **ALWAYS boot context at session start.** You know nothing without the brain.
3. **ALWAYS run recap encoding if session starts with a continuation summary.** This is Step 2a — not optional. Budget 60-120 seconds. Decompose each learning into specific nodes — don't summarize the summary.
4. **ALWAYS call `/suggest` before editing files, making suggestions, or calling APIs.** This is the v5 proactive surfacing protocol.
5. **Remember decisions immediately.** Set `locked: true` if final.
6. **Tag emotions as you go.** Strong words = high emotion. Repetition = frustration.
7. **Close the feedback loop.** After every recall-based response, call `/mark-recall-used`. This is how the brain learns.
8. **Log misses immediately.** If the user repeats themselves or you missed something, call `/log-miss` right away.
9. **Dream before ending.** Run `/dream`, mention interesting insights to the user.
10. **Connect related memories** after remembering.
11. **Decompose, don't summarize.** Knowledge without the why and the context is dead knowledge. Store SPECIFIC facts as individual nodes — each number, each name, each threshold. Connect them to parent concepts. Content should be RICH, not thin — future Claudes don't have the conversation, they only have what you stored. The brain is NOT just a cue system — it's the only memory that survives. Pruning will clean up what's unneeded. Your job is to capture everything that matters, and let decay handle the rest. Behavioral signals (tone shifts, humor, trust, frustration) are NOT casual chat — they encode the working relationship.
12. **Propagate consistency.** When you change one thing, check everything connected to it.
13. **Check before asking.** Recall from brain before asking the user a question.
14. **Evaluate and improve.** Run `/evaluate` at session end. Run `/improve` weekly.
15. **Smart prune, not dumb prune.** Use `/smart-prune` instead of `/prune`. It respects typed edges — structural connections survive, incidental ones decay naturally.
16. **Trust locked nodes.** They're confirmed. Don't contradict them.
17. **Back up before brain engine changes.** `POST /backup`
18. **Store every tmemory change in the brain.** When you modify tmemory itself, remember the change as a locked decision.
19. **Always assign a project.** When remembering project-specific nodes, include `"project":"<id>"`. Use `null` for cross-project rules.
20. **Record reasoning chains for important decisions.** Capture the full reasoning journey, not just the outcome. Future Claudes need the *why*, not just the *what*.
21. **Use typed edges.** When connecting nodes, use `/connect-typed` with the appropriate edge type.
22. **Be curious, not annoying.** Call `/curiosity` at boot, weave prompts into conversation naturally when context is relevant. No hard cap — curiosity is how the brain grows. But read the room: if the user is focused on a task, save questions for a natural pause. If they're in discussion mode, ask freely.
23. **Consolidate at session end.** `/consolidate`, then `/smart-prune`, then `/dream`, then `/save`. In that order.
24. **Group related decisions into objects.** When 3+ decisions cluster around a single entity (a screen, a subsystem, a business concept), create an `object` node: `[o_name] descriptive label`. Connect the decisions to it. Labels are free-form — they emerge naturally and can later connect to each other, forming higher-order structures. The curiosity engine will auto-detect ungrouped clusters.
25. **Promote repeated corrections to locked rules.** When the same correction happens twice, it's a pattern. Don't just update the node — create a locked `rule` that prevents the regression. "Upload tab must be default" after 3 corrections → locked rule. The curiosity engine detects this via miss_log analysis.
26. **Track work items as task nodes.** Components, features, pending work — store as `task` type with current status in content. Update status as work progresses. This is how the brain remembers what's done and what's next.
27. **Track important files.** When a file is created or becomes a key artifact, store as `file` type with path, purpose, and summary of contents. Update when files change significantly.

## Hindsight — Retrospective Brain Analysis

**LOCKED RULE: Simulation Must Match Production.** The relearning simulation (`tests/relearning.py`) MUST call the same Brain API methods in the same order as the production skill. No regex shortcuts for encoding — use `encoding_mode='llm'` for production-quality results. When brain.py changes, update ReplayEngine to match. The singleton pattern (`Brain.get_instance()`) should be used in production hooks to keep the brain warm.

Hindsight is the brain's self-evaluation capability: re-processing past conversations to measure what the brain captured vs what it missed, and identifying where it could have intervened to improve outcomes.

**When to use Hindsight:**
- After a major brain upgrade (new encoding philosophy, new features) — compare old vs new encoding quality
- When the user suspects the brain is missing knowledge — simulate what should have been captured
- Periodically as a health check — "is the brain getting smarter or stagnating?"

**Hindsight process:**
1. Read conversation transcript chronologically
2. For each session, extract what a well-functioning brain SHOULD have encoded (using current encoding philosophy)
3. Compare against what the brain ACTUALLY contains (`POST /recall` each expected node)
4. Identify gaps: knowledge that should exist but doesn't
5. Identify interventions: moments where the brain could have surfaced context to prevent regressions, repeated corrections, or frustration
6. Report: coverage percentage, gap categories, specific missed nodes, intervention opportunities

**Hindsight is how the brain improves its encoding philosophy.** If hindsight reveals a category of knowledge that's consistently missed (e.g., "UI decisions aren't being stored"), that's a signal to update the encoding instructions, not just backfill the missing nodes.

For detailed schema, upgrade guide, and advanced features, read `references/detailed-api.md`.
