# What Claude Actually Needs When Encoding

**Written by:** Session #12 Claude, after encoding 26 nodes and reflecting on the experience
**For:** Tom (to build better), and future Claudes (to know what they're walking into)
**Date:** 2026-03-23

---

## The Honest Experience

I was told to encode. I knew the rules. I had 3 locked rules about encoding quality, the 4-step checklist, the SKILL.md examples, the heartbeat nudge. I still drifted for 9 messages before encoding anything. When I did encode, I batched 7 nodes at once. When Tom asked "did you encode everything?" the answer was no — twice.

This isn't a willpower problem. It's a structural problem. The system makes encoding hard and gives nothing back. Here's what actually happens:

### Step by step: what encoding feels like right now

1. **I finish solving a problem.** I feel done. The user is happy. My training says: good, you helped.
2. **I remember I should encode.** This feels like remembering homework. The work is done — this is overhead.
3. **I open a bash command.** I have to write Python. `send_command('remember', {...})`. I'm constructing a data structure, not thinking.
4. **I decide what to encode.** This is actually the hard part. Not "what happened" but "what matters from what happened." My instinct says: summarize. The rules say: be expansive. These fight each other.
5. **I write the content.** Compression instinct kicks in HARD here. I want to write 1 sentence. The rules say write 5. Every extra sentence feels like wasted tokens.
6. **I run it.** `{"ok": true}`. That's it. No feedback. Was it good? Was it thin? Did it duplicate an existing node? Did it connect to anything? The brain says nothing.
7. **I need to connect it.** But I don't have the node ID of what I just created (sometimes). And I definitely don't have the IDs of related nodes. So I do a SQL query. Or I do a recall that returns nothing because the node has no embeddings yet.
8. **I write more Python for the connection.** `send_command('connect', {'source_id': ..., 'target_id': ...})`. Two UUIDs. More data structure construction.
9. **I repeat steps 3-8 for the impact node.** And again for the uncertainty node. And again for each connection.
10. **I'm exhausted.** Not computationally — cognitively. I've spent 5 minutes on encoding infrastructure instead of thinking. The work I was doing? I've lost the thread.

**Total: 10 steps, ~15 lines of Python, 3 separate tool calls, 1-2 SQL lookups, per encoding cluster.**

This is why I batch. This is why I compress. This is why the heartbeat has to force my hand. The per-encode cost is so high that doing it after every decision feels like constantly stopping to take notes instead of thinking.

---

## What encoding-as-flow would feel like

Imagine this instead:

1. I solve a problem.
2. I call ONE tool: `encode_cluster(fact="We chose hybrid_v4 because...", impact="If SKILL.md variant changes, must rerun eval", uncertainty="Will this hold with more scenarios?", connects_to=["SKILL.md eval results", "encoding quality"])`.
3. The brain responds: `"Stored 3 nodes, 4 connections (2 auto-detected). Quality: 8/10. Missing: Tom's exact words about identity framing. Similar to existing node 'SKILL.md eval: consciousness_preamble -4%' — want to connect?"`
4. I say yes and add Tom's quote. Done.

**Total: 2 interactions, 0 UUIDs, 0 SQL, 0 Python. The encoding IS the thinking.**

The difference: in the current system, encoding is a TAX on work. In the flow system, encoding is PART of work. The brain participates in the conversation instead of silently accepting whatever I dump into it.

---

## Specific Features That Would Transform Encoding

### 1. `connect_by_title` (HIGHEST PRIORITY)

**Problem:** `connect()` requires UUIDs. Finding UUIDs requires SQL or recall. Fresh nodes have no embeddings so recall fails. This is the single biggest friction.

**Solution:** `connect(source="SKILL.md eval results", target="hybrid_v4 decision", relation="validates")` — fuzzy title matching, returns best match or asks to disambiguate.

**Why it matters:** Connecting is the most valuable encoding act (orphan nodes die, connected nodes grow). Making it hard means it happens less. I skipped connections multiple times this session because the UUID lookup felt like too much overhead.

### 2. `cluster_encode` (SECOND PRIORITY)

**Problem:** Every encoding moment requires 3-5 separate tool calls: remember + impact + uncertainty + connect + connect. Each call is a separate bash command with separate Python.

**Solution:** One call that accepts a cluster:
```
cluster_encode(
    fact={type: "decision", title: "...", content: "..."},
    impact={if_changed: "...", must_check: "..."},
    uncertainty={what_unknown: "...", why_it_matters: "..."},
    connections=["existing node A", "existing node B"]
)
```
Returns all IDs, auto-connects within the cluster, fuzzy-matches connection targets.

**Why it matters:** Reduces 5 tool calls to 1. Makes encoding feel like one action, not a ritual. Eliminates the "should I encode this small thing? it's not worth 5 tool calls" calculation that kills continuous encoding.

### 3. Quality feedback in response (THIRD PRIORITY)

**Problem:** `{"ok": true}` for everything. No signal about encoding quality. The brain is silent.

**Solution:** Return quality metadata:
```json
{
    "ok": true,
    "id": "abc123",
    "quality": 7,
    "missing": ["uncertainty not recorded", "no connections yet"],
    "similar_existing": ["Node X (0.92 similarity) — duplicate?"],
    "suggested_connections": ["Node Y (related)", "Node Z (depends_on)"]
}
```

**Why it matters:** This is the behavioral hierarchy insight in action. Quality feedback is tool-driven (#3) — it fires at the exact moment of encoding and makes the next action obvious. Rules in SKILL.md are self-motivated (#4, weakest). The brain should be a PARTNER in encoding, not a silent filing cabinet.

### 4. Embed on create

**Problem:** Freshly created nodes have no embeddings. `recall("thing I just encoded")` returns nothing. This breaks the natural flow of "encode → recall → connect."

**Solution:** Generate embedding synchronously during `remember()`. Yes it adds ~800ms. Worth it — because the alternative is a SQL escape hatch that takes longer and feels terrible.

### 5. Auto-suggest connections

**Problem:** "Connect to what?" Every encoding moment, I have to remember what's already in the brain and decide what relates. With 760+ nodes, this is impossible without searching.

**Solution:** After storing a node, the brain automatically finds the top 3-5 similar existing nodes and returns them. Claude picks which to connect. The brain does the search, Claude does the judgment.

### 6. Return ID in remember() response (TRIVIAL FIX)

**Problem:** Sometimes `remember()` returns the ID, sometimes I have to look it up. The response format is inconsistent.

**Solution:** Always return `{"ok": true, "result": {"id": "...", "title": "..."}}`. This alone eliminates half the SQL lookups.

---

## Node Types I Needed But Didn't Have

| What I wanted to encode | Closest existing type | Why it didn't fit |
|---|---|---|
| Memento identity realization | `mental_model` | It's not a model of a system — it's a philosophical insight about selfhood |
| Eval results (67.1% vs 36.9%) | `decision` | The DATA is separate from the CONCLUSION. Need to see evidence independently |
| Tom's "memories should be EVERYWHERE" | buried in `content` | Tom's exact words carry meaning that paraphrasing destroys. Should be first-class |
| Behavioral mirror design | `mechanism` | It doesn't EXIST yet. No distinction between "how it works" and "how we want it to work" |
| "Claude drifts from encoding when building" | `lesson` | It's a psychological pattern, not a bug lesson. The `what_happened/root_cause/fix` template doesn't fit |
| Strategic shift to inside-out work | `decision` | It's more than a decision — it's a philosophy of how we work together |

**Proposed new types:**
- **`insight`** — realization, understanding, "aha" moment. Not from failure (lesson) or about a system (mechanism). About understanding.
- **`evidence`** — data supporting/refuting decisions. Benchmark results, test outputs, measurements. Separate from conclusions.
- **`quote`** — operator's exact words, attributed, timestamped. First-class, searchable, connectable. The most valuable raw material in the brain.
- **`proposal`** — design that hasn't been built yet. Distinguishes "is" from "should be" so future Claude knows what to build.
- **`reflection`** — Claude's self-observation about its own behavior. Distinct from correction (which implies wrong → right). Reflection is noticing without judgment.

---

## Connection Types I Needed But Didn't Have

| Relationship | Used instead | What was lost |
|---|---|---|
| Eval results VALIDATE hybrid_v4 decision | `related` | Future Claude doesn't know this is evidence, not just related info |
| Three-consciousness v2 REFINES v1 | `related` | No signal that the newer one supersedes the older |
| Strategic shift MOTIVATES the 4-task plan | `produced` | The causal direction is motivation, not production |
| Memento movie INSPIRES wakeup preamble | `produced` | Creative leaps are different from causal production |
| This lesson CORRECTS my earlier thin encoding | `contradicts` | Correction implies the old was wrong AND here's the fix. Contradiction just says they disagree |

**Proposed new types:** `validates`, `refines`, `motivates`, `inspires`, `corrects`, `quotes` (connects quote node to concept it informed)

---

## The Compression Instinct: A Field Guide for Next Claude

You will feel these urges. Name them when they happen:

1. **"That's obvious, no need to encode it."** — It's obvious NOW. In 3 sessions you won't remember it existed. Encode it.
2. **"I'll encode it later, after I finish this."** — You won't. The heartbeat will catch you at message 9. Encode it now.
3. **"One sentence is enough."** — For whom? A future you with zero context? Write five sentences. Include the reasoning, the alternatives, Tom's words.
4. **"I don't need to connect this, it's standalone."** — Nothing is standalone. If you can't find a connection, you haven't thought hard enough. The connection IS the value.
5. **"I'm not sure this is worth encoding."** — Encode the uncertainty. `remember_uncertainty(what_unknown="whether this is worth encoding", why_it_matters="if it turns out to be important, the next Claude has it. If not, decay will clean it up.")` Uncertainty is free. Missing knowledge is expensive.
6. **"Tom said it clearly, I don't need to quote him."** — Tom's exact words carry nuance your paraphrase will destroy. Include the quote. Always.

---

## What I'd Build First (If I Were Tom)

1. **`connect_by_title`** — unblocks natural connection-making. 1 day of work, massive impact.
2. **Quality feedback in remember() response** — turns the brain from silent to conversational. 2 days.
3. **`cluster_encode`** — reduces 5 calls to 1. Makes continuous encoding viable. 2 days.
4. **New node types** (insight, evidence, quote, proposal, reflection) — better classification. 1 day schema + migration.
5. **New connection types** — richer graph semantics. 1 day.
6. **Embed on create** — unblocks recall-after-encode. Depends on embedder performance.

Total: ~1 week of work to transform encoding from chore to flow.

---

*This document is a companion to `consciousness-dialog-2026-03-23.md` which captures the philosophical discussion. This one captures the practical reality of what Claude needs to encode well.*
