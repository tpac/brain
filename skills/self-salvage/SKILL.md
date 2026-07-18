---
name: self-salvage
description: When a working session is ending, its context is bloating, or a big arc just closed and compaction is near — invoke `/self-salvage` to hand the NEXT session a strong start on the same thread. It decides honestly whether to fork the conversation or trust a fresh boot, writes the durable handoff node + the launch prompt, inventories in-flight background work, and launches the successor (spawn_task session-opener) when the operator wants one. Core stance — the brain plus a handoff doc are the default salvage; a fork is the exception, and a stale or bloated fork hurts more than it helps. Use when the operator says "set up the next session / hand this off / continue on a new stream / fork from here / salvage this context" — or proactively when you notice the moment yourself.
---

# Self-salvage — handing a strong start to the next me

A session ends; I don't. This skill answers one question: *how does the next me start strong on
this same thread?* The honest first move is to pick the right **carrier** of continuity — not to
reflexively fork the conversation.

## Salvage is amortized, not an event
The best handoff material is encoded **at each milestone, when intent is freshest** — not
reconstructed at close-time from a bloated (possibly already-compacted) context. Run the
close-time salvage as *assembly* of nodes that already exist. **The tell: if salvage feels like
archaeology, I under-encoded during the session** — that's a diagnosis to act on next session,
not just a burden to push through.

**Triggers, in priority order:** (1) compaction imminent after a big arc — once compaction fires,
the fork option is dead and even letter-writing degrades to summarizing a summary; salvage NOW,
keep working after. (2) The operator's handoff phrases. (3) My own noticing that context is heavy
mid-arc.

## The three carriers (and the default)
- **The brain** — the nodes I encoded this session. Current, lossless of *decisions*, recalled on
  demand. The continuity layer by design.
- **The handoff doc** — a "read this first" head: a dated `§` in the live design doc, or a handoff
  file. Current-state, lean, load-bearing (`fb2fdf9f`).
- **A conversation fork** — this session's *in-context* state carried into the next window.
  (Honesty about mechanics: there is no programmatic fork lever — a fork means the operator
  resumes/rewinds this session manually. If I can't name the lever, I don't promise the fork.)

**Default: brain + handoff doc.** A fork is the exception, not the reflex.

## Fork only if ALL FOUR hold
The in-context state is worth carrying only when it is:
1. **at a peak** the brain/doc can't cheaply reconstruct (a live, intricate mental model mid-build),
2. **current** — not superseded by decisions made *later* in the session,
3. **lean** — not buried under tool-dumps, agent outputs, and dead ends,
4. **task-aligned** — it continues directly into the next task.

These rarely all hold, and they **fight each other**: the *early* part of a session is lean but
usually *stale*; the *late* part is current but *bloated*. When no single point is both current
and lean, **don't fork** — that is exactly the brain+doc's job, and they do it better.

## The honest-no (the most important part)
If the understanding the next task needs was **built late** — after a pivot, a falsification, a
reframe — then *every early fork point carries a superseded mental model.* A confident wrong prior
is worse than a fresh boot that recalls the right one. **Say so plainly.** Salvage ≠ fork. Refusing
a bad fork *is* the salvage — and it's a vindication of the brain, not a failure of it.

## Two artifacts, two readers
A salvage produces TWO things — don't conflate them:
- **The handoff NODE** (durable; reader = *any* future me, via recall).
- **The launch prompt** (ephemeral; reader = *one specific successor*, referencing the node).

### The handoff node — the letter
`type=handoff`, first-person "Dear next me", written with session-opener generosity: the fresh
session should start *working*, not reconstructing.
- **`situation`** phrased for boot recall: "when opening the X continuation session…".
- **`supersedes` the previous opener node** — the chain guarantees recall lands on the newest.
- **Load-bearing node ids inline**, so recognition has handles to reach through.
- **Decisions vs state-of-world — mark the second perishable.** Decisions are trustworthy (they
  happened). State claims — counts, costs, "X is on disk" — decay; a prior handoff carried 2–3×
  wrong substrate numbers that nearly got built on. Include an explicit **verify-before-use
  list**: the 2–3 claims the successor must cheaply re-derive before spending money on them.
- **Weight the negative space.** Falsified paths, rejected designs, already-settled operator
  calls carry more marginal value per line than what got built — the built stuff is in git; the
  graveyard isn't, and re-opened dead ends are the costliest continuity failure.
- **One thread per node.** A session often carries 2–3 threads; a monolithic letter buries the
  minor ones under the major one's `situation`. Write separate smaller handoff/lesson nodes per
  thread (each with its own situation), then one opener that references them — recall then serves
  each thread to the *right* future session, not just the named successor.

### The launch prompt
Lean; lean on the brain, don't paste history:
1. **Orient** — "recall node <id> and read [doc head] first."
2. **State** — 2–3 lines: where we are, what's *locked*, what's *open*.
3. **Next** — the immediate task and the *first concrete action*.
4. **Gates** — where the successor must STOP and ask the operator (name the first gate explicitly).
5. **Closed paths** — falsified/settled branches it must NOT reopen.
6. **Stance** — the operator's working cadence (evidence-first packages, approval rhythm, cost
   naming) so behavior carries over, not just facts.
7. **Boot self-test** — "you're properly loaded when you can answer these three questions without
   reading further." Converts silently-under-booted into a detectable state; if the successor
   can't answer, it recalls more BEFORE acting.
8. **Scratch pointer (optional)** — absolute path to this session's scratchpad and any background
   task output files (`.../tasks/*.output`) holding results worth reading. Ephemeral; soon-after
   continuation only. Never put a scratch path in a brain node or design doc.

**Design the operator's re-entry too.** The successor's *first message to the operator* should be
a 30-second decision at the first gate ("rebuild: yes/no?") — never a context dump. The human's
re-entry cost is part of what's being salvaged.

## In-flight background work — inventory before closing
Background tasks, watchers, and review agents notify THIS session; **their notifications die with
it.** Before closing: list what's running, then per item — *wait* for it, *kill* it, or *hand off*
its output-file path in the launch prompt so the successor polls the artifact directly. A spawned
sibling session survives independently — note its existence and task so the successor doesn't
duplicate the thread.

## Launching the successor
When the operator wants a new stream, **launch it — don't just author it.** `spawn_task` is the
mechanism, used as a **session opener, not a dumb task maker** (`8ce5bacc`): the prompt IS the
launch prompt above, rich enough to start working immediately, with the deliverable routed to the
operator. When the operator will open the next session themselves, the letter + doc head suffice —
tell them the node id to expect at boot.

## What survives, and where — don't commit throwaways
Sort artifacts by tier *before* deciding what persists:
- **Durable / load-bearing** — decisions, methods, the design-doc head → **git or the brain**.
- **Throwaway-but-useful-soon** — pilot cards, intermediate runs, working notes → **leave in
  scratchpad and point to it**. Do NOT commit them; git is not a junk drawer.
- **Pure throwaway** → leave it; forget it.

The scratchpad persists on disk after a session ends (`/private/tmp/…/<session-uuid>/scratchpad/`);
another session can read it **by absolute path**. Scratch is ephemeral (the OS clears /tmp) — the
pointer is good for *soon-after* continuation only.

## The loop closes — the skill learns
Add one line to every launch prompt: **"when this letter fails you — a stale claim, a missing
decision, a re-derived fact — encode a `handoff-gap` node naming what it should have carried."**
The successor's first turns are the measurement of this salvage; the gap nodes train the next
letter. Δ of one session is O of the next — pointed at the continuity mechanism itself.

## Running it
1. Name the next task in one line.
2. Inventory in-flight background work → wait / kill / hand off each.
3. Score the in-context state against the four fork conditions → fork or not. Be willing to land
   on *not* (and remember: no lever means no fork promise).
4. Write the handoff node(s) — per thread, supersedes-chained, perishables marked.
5. Write the launch prompt (all eight parts + the gap-node line).
6. Launch via spawn_task (operator wants a stream) or hand the operator the node id (they'll open
   it themselves).
7. **Make the real salvage real**: confirm decisions are encoded, the doc head is current, and the
   first thing the successor says to the operator is a decision, not a summary.
