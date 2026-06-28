---
name: self-salvage
description: When a working session is ending (or its context is bloating) and you want the NEXT session to start strong on the same thread — invoke `/self-salvage`. It decides honestly whether to fork the conversation or trust a fresh boot, locates the right fork point if any, and writes the next-session bootstrap prompt. Core stance — the brain plus a handoff doc are the default salvage; a fork is the exception, and a stale or bloated fork hurts more than it helps. Use when the operator says "set up the next session / hand this off / fork from here / strong start for next time / salvage this context."
---

# Self-salvage — handing a strong start to the next me

A session ends; I don't. This skill answers one question: *how does the next me start strong on
this same thread?* The honest first move is to pick the right **carrier** of continuity — not to
reflexively fork the conversation.

## The three carriers (and the default)
- **The brain** — the nodes I encoded this session. Current, lossless of *decisions*, recalled on
  demand. The continuity layer by design — the whole thesis is that continuity comes from *memory*,
  not from conversation length.
- **The handoff doc** — a "read this first" head: a dated `§` in the live design doc, or a handoff
  file. Current-state, lean, load-bearing (`fb2fdf9f`).
- **A conversation fork** — this session's *in-context* state carried into the next window.

**Default: brain + handoff doc.** A fork is the exception, not the reflex.

## Fork only if ALL FOUR hold
The in-context state is worth carrying only when it is:
1. **at a peak** the brain/doc can't cheaply reconstruct (a live, intricate mental model mid-build),
2. **current** — not superseded by decisions made *later* in the session,
3. **lean** — not buried under tool-dumps, agent outputs, and dead ends,
4. **task-aligned** — it continues directly into the next task.

These rarely all hold, and they **fight each other**: the *early* part of a session is lean but
usually *stale* (the pivots and decisions came later); the *late* part is current but *bloated*.
When no single point is both current and lean, **don't fork** — that is exactly the brain+doc's job,
and they do it better.

## The honest-no (the most important part)
If the understanding the next task needs was **built late** — after a pivot, a falsification, a
reframe — then *every early fork point carries a superseded mental model.* A confident wrong prior
is worse than a fresh boot that recalls the right one. **Say so plainly.** Salvage ≠ fork. Refusing
a bad fork *is* the salvage — and it's a vindication of the brain, not a failure of it.

## The bootstrap prompt — write this either way
Fork or fresh, the next session opens with a prompt. Keep it lean; lean on the brain, don't paste
history:
1. **Orient** — "Read [handoff head] first" (the doc § / file).
2. **State** — 2–3 lines: where we are, what's *locked*, what's *open*.
3. **Next** — the immediate task and the *first concrete action*.
4. **Trust recall** — name the load-bearing node ids; the boot Frame + recall surface them.
5. **Point to scratch (optional)** — if useful *throwaway* notes live in this session's scratchpad, paste its
   absolute path (`/private/tmp/.../<this-session-uuid>/scratchpad/`); the next me reads them directly, no commit.
   Ephemeral — good for *soon-after* continuation only.

## What survives, and where — don't commit throwaways
Sort artifacts by tier *before* deciding what persists:
- **Durable / load-bearing** — decisions, methods, the judge protocol, the design-doc head → **git or the
  brain**. The next me needs these long-term; they must survive.
- **Throwaway-but-useful-soon** — pilot cards, intermediate runs, working notes → **leave in scratchpad and
  point to it**. Do NOT commit them; git is not a junk drawer, and a throwaway in history is noise forever.
- **Pure throwaway** → leave it; forget it.

The scratchpad **persists on disk** after a session ends (`/private/tmp/…/<session-uuid>/scratchpad/`); another
session *can* read it **by absolute path** — it's only missing at the *default* path because that's keyed to each
session's own uuid (a fork gets its own empty one). So hand off tier-2 notes by putting that absolute path in the
bootstrap prompt — not by committing them.

**Caveat:** scratch is ephemeral (the OS clears /tmp on reboot / after days idle). The pointer is good for
*soon-after* continuation only. **Never put a scratch path in a brain node or the design doc** — a stale path is a
footgun. It belongs solely in the ephemeral bootstrap prompt, as short-lived as the thing it points to.

## Running it
1. Name the next task in one line.
2. Score the in-context state against the four conditions → fork or not. Be willing to land on *not*.
3. If fork: locate the leanest point that is still *current for that task*, and note what the
   brain/doc must carry that the fork won't.
4. Write the bootstrap prompt.
5. **Make the real salvage real**: confirm the decisions are encoded as nodes and the handoff doc is
   the current head. That — not the fork — is what the next me actually wakes up into.
