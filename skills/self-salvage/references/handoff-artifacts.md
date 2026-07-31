# Handoff artifact templates

Loaded on demand by `/self-salvage`. The procedure lives in `SKILL.md`; these are the four artifact
specs. Each is split **MUST** (write these or the artifact fails) vs **REFERENCE** (include if cheap).

---

## §0 — The doc head (git; the only carrier that survives compaction, /tmp clearing AND a recall miss)

A dated `§` at the top of the live design doc. If no design doc exists, a handoff file. If neither,
the handoff node is the head — say so in the node so nobody hunts for a file.

**MUST**
- `## <n> — <one-line claim> (<date>) ◀ ACTIVE ARC`
- **Read first:** the handoff node id and the one measurement/decision id it rests on.
- The central finding in 3–5 lines, with numbers.
- **What's locked** and **what's open** — separately.
- **Do not reopen** — the closed paths, one line each.

**REFERENCE** — the next 2–4 builds; commit hashes; the first gate.

Why it exists: recall can miss, `/tmp` gets cleared, and context compacts. This is the copy that
can't disappear. **Spend proportionate care here — it outlives every other artifact.**

---

## §1 — The handoff node (durable; reader = any future me, via recall)

`type=handoff`, first-person "Dear next me". **Budget: ≤800 words.**

**MUST**
- **First line is the BLOCKER**, not the arc narrative: the open gate, who owns it, what unblocks
  it. If nothing is blocked, the first line is the first concrete action. The central *fact* comes
  second. (A letter that opens on a fact makes the successor read 400 words before learning the
  thread is waiting on a human.)
- **The boot self-test** — three questions the successor must answer without reading further.
- **`situation` written in the words the successor's first prompt will actually use** — not a
  description of the session. Descriptive writer-voice phrasing ("when opening the X continuation
  session") measured 14% top-25 entry against trigger-register phrasing's 54% on this brain. Name
  the actions and the utterances that should trigger it.
- **Verify-before-use list** — the 2–3 state claims the successor must cheaply re-derive, **each
  with one clause saying how it was checked**. Unchecked claims don't get the label. A prior handoff
  shipped substrate numbers 2–3× wrong that were nearly built on; another shipped a wrong claim
  *labeled "verified real"* because verification had checked a counter, not the substrate.
- **The fork verdict, one line**, and what follows from it (no-fork ⇒ this letter must carry the
  mental model, not just the state).
- **Negative space, ≤6 lines** — this session's closures plus the one most-tempting older path.
  Everything else graduates to durable nodes *before* you write the letter. ("Graduated" = closed
  more than two sessions ago and not reopened since.)
- **One self-labeled working-set line**: *"working set (ephemeral, may be gone): `<abs path>` — if
  missing you lose speed, not knowledge."* This is the exception to the no-scratch-paths-in-durables
  rule, and it exists because the launch prompt lives in a chat the successor cannot read.

**REFERENCE**
- Load-bearing node ids inline, so recognition has handles.
- **Thread identity**: prefix the title `[thread:<slug>]`; find the prior opener carrying that slug
  (`filter_nodes(field='type', include=['handoff'])`) and `connect(source=<new>, target=<old>,
  relation='supersedes')`. No prior opener → you're opening the thread; say so. **Supersedes is
  hygiene, not delivery** — stale openers are known to stay live, so never rely on newest-wins.
- **Adjacent threads and their owners**, so the successor doesn't collide or duplicate.
- **Cost per queued item** ("≈20 min compute, no LLM spend").
- **Written-at stamp**, plus: *run `git log --since=<written-at>` before trusting the queue order* —
  siblings commit while you sleep. (Unconditional; don't try to predict an open-by date.)
- **Open gates are perishable**: *when the operator rules, encode the ruling immediately* — else the
  next session re-asks.
- **One thread per node.** A session carrying 2–3 threads gets separate smaller handoff nodes, each
  with its own `situation`, plus one opener referencing them — so recall serves each thread to the
  right future session.

---

## §2 — The working set (the scratchpad file)

Not a junk drawer: what's expensive to re-derive but wrong for the brain or git.

**MUST** — lead with these two, in this order:
1. **The one decision awaiting a human**, with the command to read its evidence and the framing for
   the ask.
2. **Copy-paste commands** — exact invocations, env vars, resource caps, paths.
3. **Every measured number in one table.** A node holds *meaning*; forty values make it
   unrecallable-by-meaning and unreadable. Nowhere means re-running expensive probes to remind
   yourself. **This artifact class has no other home** — it generalises past any one project.

**REFERENCE** — in any order: harness idioms (imports, contracts, helper functions that took real
time to get right) · gotchas too small for nodes · the next build's skeleton, **marked UNTESTED if
you didn't run it** · pointers (snapshots, caches, commits, node ids).

**Open the file with: "pure accelerant — nothing here is the only copy,"** and say where the durable
copies are. Path shape: `/private/tmp/.../<session-uuid>/scratchpad/` — it survives session end, but
the OS clears `/tmp`, so the successor must degrade to brain+git losing speed, not knowledge.

---

## §3 — The launch prompt (ephemeral; reader = one specific successor)

Lean. Lean on the brain; don't paste history. **This is the successor's first read.**

**MUST**
1. **Orient** — `recall <node id>` *and* `get_node(<id>)`, plus the doc head to read first. Include
   the explicit `get_node` — `type=handoff` is in no aspect, so the Frame won't carry it at boot.
2. **Next** — the immediate task and the *first concrete action*. Open on the **first gate**, not on
   state.
3. **Gates** — where to STOP and ask; name the first one explicitly. **Before asking the operator
   anything in the gate list, recall the gate — a prior stream may already have ruled.** *For an
   autonomous reader, replace gates with stop-conditions plus an explicit list of what it may decide
   alone.*
4. **Closed paths** — what it must NOT reopen.
5. **The boot self-test**, quoted from the node.
6. **The gap-node line**: *"when this letter fails you — a stale claim, a missing decision, a
   re-derived fact — encode a `handoff-gap` node naming what it should have carried."*

**REFERENCE** — 2–3 lines of state (where we are / locked / open) · the operator's stance and cadence
so behavior carries over, not just facts · the working-set path with its degradation note.

**Design the operator's re-entry.** The successor's first message should be a 30-second decision at
the first gate — never a context dump. The human's re-entry cost is part of what's being salvaged.
