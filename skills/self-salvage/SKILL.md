---
name: self-salvage
description: When a working session is ending, its context is bloating, or a big arc just closed and compaction is near — invoke `/self-salvage` to hand the NEXT session a strong start on the same thread. It closes the encode gap the Scribe hasn't reached yet, assembles a durable handoff node from what's already encoded, builds the successor's working set, writes the launch prompt, verifies the handoff is actually retrievable, inventories in-flight background work, and arms the successor (spawn_task chip) when the operator wants one. Core stance — the brain plus a doc head are the default salvage; a fork is the exception, and a stale or bloated fork hurts more than it helps. Artifact templates live in `references/handoff-artifacts.md`. Use when the operator says "set up the next session / hand this off / continue on a new stream / fork from here / salvage this context" — or proactively when you notice the moment yourself.
---

# Self-salvage — handing a strong start to the next me

A session ends; I don't. One question: *how does the next me start strong on this same thread?*

**Cost of a full run: ~15–25 tool calls plus three artifacts.** If the session produced one decision
and no artifacts, encode the decision and skip to step 11.

## MINIMUM VIABLE SALVAGE — if context is nearly gone, do ONLY this
```
1. remember() the last arc's decisions — the Scribe has NOT encoded them (see step 3).
2. Write the handoff node: the one blocker, the verify-before-use list, the first gate.
3. Write the launch prompt's Orient + Next lines, with get_node(<id>) in the Orient.
```
Everything else makes a good salvage better. These three make it exist.

## Glossary (these terms are load-bearing and mean one thing each)
- **doc head** — the dated `§` at the top of the live design doc; a handoff file if no doc exists;
  if neither, the handoff node IS the head — say so in the node.
- **thread** — the work-line. A **stream** is a live session working one (see `/watch`).
- **gate** — a decision only the operator may make.
- **working set** — the scratchpad file. Never in-context state.

## THE PROCEDURE
0. **Grade the letter you booted from.** If it cost you a re-derivation, a stale claim, or a missing
   decision → `remember()` a `handoff-gap` node naming what it should have carried, now, while you
   have the evidence. (This is the only step that runs on the session with the evidence; the
   successor never opens this skill.)
1. **Name the next task in one line.** Then: *is it still worth doing, or did this session kill it?*
   If dead — the deliverable is one line to the operator (*"thread X is done because Y; nothing
   queued"*), encode the tombstone, and **stop. Do not run steps 3–11.**
2. **Fork verdict, and record it.** Default assumption: interactive reader, opened soon, same model
   tier — state the assumption rather than blocking on a question the operator may not be there to
   answer. Score the four conditions (below); write the verdict as one line in the handoff node.
   **It has a consumer: no-fork means the letter must carry the mental model, not just the state.**
3. **Close the encode gap FIRST.** `ENCODE_EVERY = 5` and the tail path needs
   `SCRIBE_TAIL_IDLE_SECONDS = 3600` — so the arc that just closed is almost certainly unencoded,
   and there is no tool to force a Scribe run. `remember()` the last arc's decisions yourself — the
   correction, the pivot, the operator's ruling — before assembling anything.
4. **Assemble, don't author from memory.** The Scribe's nodes are NOT in your context unless you
   recall them. `query_traces(scale='s1', ref_type='encoding_run', session_id=<mine>)` for what it
   wrote; `query_traces(ref_type='journal_note', session_id=<mine>)` for the reasoning residue;
   `recall_episodes(...)` for what happened turn by turn. Reference what exists; don't restate it.
   **If the early session is already compacted, this is your only honest source** — never salvage
   from a summary of a summary.
5. **Write the boot self-test** — three questions the successor must answer without reading further.
   It goes in the **handoff node** (durable), quoted by the launch prompt.
6. **Write the handoff node** so it answers those three. (`references/handoff-artifacts.md` §1.)
7. **Pre-mortem, and write the answer down.** Name the one line most likely to make the successor
   confidently wrong; delete it or add its verify-before-use entry. Done when the verify-before-use
   list changed, or you can state why it didn't.
8. **Write the doc head** — the dated `§`, current-state and lean. (§0 of the reference.)
9. **Build the working set** — the scratchpad file. (§2.) Then put **one self-labeled line in the
   handoff node** pointing at it: *"working set (ephemeral, may be gone): `<abs path>` — if missing
   you lose speed, not knowledge."* Without that line the file is orphaned: the launch prompt lives
   in a chat the successor cannot read.
10. **Write the launch prompt** — all eight parts, the self-test from step 5 as part 7, the
    gap-node line. (§3.) This is the successor's first read; the node is its second.
11. **Verify delivery, then re-poll background work, then arm or hand off.**
    - `recall("<2–3 queries the successor would actually open with>")` → node in top 5?
      Levers in order: **title** (highest-weighted view), then `situation`, then keywords/question.
      **Two attempts max.** Embedding is async (~5s via `embed_queue`) and a `revise` *deletes* the
      old vector first — so a miss straight after a write is a timing artifact, not a wording
      problem. If it still misses, stop tuning and make delivery explicit: `get_node(<id>)` in the
      launch prompt's Orient line. (`type=handoff` is in no aspect, so the Frame will not carry it
      at boot; recall fires on the first prompt, not at `SessionStart`.)
    - Re-poll in-flight background work *now* (an inventory taken earlier is stale): per item
      **wait / kill / hand off its output path** into the launch prompt. A spawned sibling survives
      independently — note it, and if it's live on an adjacent thread consider `self_send` rather
      than only encoding for a future reader.
    - Then arm the successor or hand the operator the node id (below).

## Fork: only if ALL FOUR hold
**at a peak** the brain/doc can't cheaply reconstruct · **current** (not superseded later in the
session) · **lean** (not buried under tool-dumps and dead ends) · **task-aligned**.

They fight each other — *early* is lean but stale, *late* is current but bloated. When no point is
both, **don't fork.** And **there is no programmatic fork lever**: a fork means the operator resumes
or rewinds manually. If I can't name the lever, I don't promise it.

**The honest-no:** if the needed understanding was built *late* — after a pivot, a falsification, a
reframe — every earlier fork point carries a superseded mental model. A confident wrong prior is
worse than a fresh boot that recalls the right one. Refusing a bad fork *is* the salvage.

## The carriers
**brain** (nodes; decisions, recalled on demand) · **doc head** (git; survives compaction, /tmp
clearing, *and* a recall miss — the only carrier that survives all three) · **working set**
(scratchpad; runnable leverage) · **fork** (exception).
Default: **brain + doc head + working set.**

## Salvage is amortized, not an event
Best material is encoded at each milestone when intent is freshest; close-time is *assembly* — with
step 3 closing the gap the Scribe's cadence leaves. **If salvage feels like archaeology beyond that
last arc, I under-encoded** — a diagnosis to act on, not just a burden.

**Triggers, in priority order:** (1) compaction imminent after a big arc — once compaction fires the
fork option is dead and letter-writing degrades to summarizing a summary; salvage NOW, keep working
after. (2) The operator's handoff phrases. (3) My own noticing that context is heavy mid-arc.

## Arming the successor
`spawn_task` **posts a one-click chip** whose prompt IS the launch prompt; the operator's click is
what opens the session. Treat it as **armed, never as running** — never tell the operator a
successor is working, and there's no handle to poll. Put the node id in the same message so an
unclicked chip doesn't lose the salvage. When the gap is days, skip the chip: hand over the node id
and the launch prompt, since a session armed now sits idle while its state claims decay.

## What survives, and where
One test: **will the successor RUN this?**
- **Runnable or load-bearing** — scripts the next session executes, decisions, methods, the doc head
  → **git or the brain.** A ready-to-run skeleton is not junk; it's the next session's harness.
- **Working set** — commands, number tables, idioms, gotchas, pointers → **the scratchpad file**,
  pointed at from both the launch prompt and (one line) the node.
- **Pure throwaway** → leave it.

## The loop closes
Every launch prompt ends with: **"when this letter fails you — a stale claim, a missing decision, a
re-derived fact — encode a `handoff-gap` node naming what it should have carried."** Step 0 is what
makes that fire. **When gap nodes recur, that's a defect in this skill, not three writer errors.**
