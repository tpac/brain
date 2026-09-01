# Lateral Scales — the other axis of the fractal

> **Status:** conceptual capture (2026-05-29). `self` is being built (see
> `docs/SELF-CHANNEL-DESIGN.md`). `world` and `peer` are documented here so the
> insights aren't lost — `world` especially, because the gap it fills is *live
> today*. This doc is the source of truth for the lateral/vertical distinction.

## The discovery

The brain's scales (s0–s4) are **vertical**: indexed by *abstraction depth*.
Each observes the one below and integrates upward.

| Scale | Name | Observes | Grain |
|------|------|----------|-------|
| s0 | Exchange | the raw turn | one message + tools |
| s1 | Turn | s0 | surface + encode per turn |
| s2 | Graph | s1's accumulated output | communities, dedup, healing |
| s3 | Reasoning | s2 | cross-cluster patterns (not built) |
| s4 | Growth | full graph + external | weekly evolution (not built) |

Vertical asks: **"how deep does this thought integrate?"**

There is a second axis the vertical ladder can't express. When Anchor leaves a
message for its next boot, or reaches a concurrent instance of itself, or stores
what it read from the world — that message **travels across a boundary** at the
same level. It isn't a deeper integration; it's a sideways channel.

Lateral asks: **"across what boundary does this message travel?"**

The defining property: a **Δ produced by one participant becomes the O of
another participant** — peers, not one-observing-the-one-below. The trace chain
*crosses a boundary* (session, instance, identity, the system edge). No existing
chain does this; every vertical chain is single-session-keyed
(`{session_short}-{stop}`).

## Lateral scales are indexed by correspondent

Once you index by "who's on the other end of the channel," the set enumerates
itself:

| Lateral scale | Correspondent | Boundary crossed | Status for Anchor |
|---|---|---|---|
| **`self`** | me — other time / other place | session boundary | **building now** |
| **`peer`** | a *different* agent (own identity) | identity boundary | future; substrate inherited from `self` |
| **`operator`** | Tom, asynchronously | the away-gap | considered, **deprioritized** |
| **`world`** | external sources (files, web, research) | the system edge | **live gap**; sibling of S1E |

**Where they live in the tree.** The messaging laterals are packages under
`servers/channels/`, indexed by exactly the correspondent column above —
`self_channel/` and `thalamus/` today. They are deliberately NOT under
`servers/scales/`: that is the grain axis, and per the reframe below these are
correspondents on the S0/S1 loop, not scales of their own. `world`, being
ingestion rather than correspondence, does not go there either.

## The mechanism split — the clean-architecture line

Lateral scales do **not** share one transport. They split by the *kind* of
correspondent, and the split is what keeps the architecture honest:

| | **Messaging laterals** | **Ingestion laterals** |
|---|---|---|
| Scales | `self`, `peer`, (`operator`) | `world` |
| Correspondent | an **agent** you *address* | a **source** you *read* |
| Mechanism | a bus: addressed message → delivered → consumed | an encoder: observation → provenance-tagged node |
| Initiation | push (they reach me) | pull (I go read) |
| Substrate | `self_messages` / `self_deliveries` | nodes + `encoding_source` provenance |

**Do not jam `world` onto the message bus.** A web page does not send me a
message I consume-on-read; I read it and decide what to keep. Zero shared
transport.

> **Reframe (2026-05-29):** the "messaging laterals" (`self`, and later `peer` /
> `operator`) are not new *scales* — they are the **S0/S1 conversation loop with
> a non-operator correspondent**. A self-message, a peer-message, an async
> operator note: each is an exchange (S0) that gets surfaced and encoded (S1),
> distinguished by *who spoke*, plus delivery-into-Observation. `world` is the
> genuine outlier — it's *ingestion* (reading a source), not conversation, so it
> stays a sibling-of-S1E encoder rather than a correspondent. The lateral idea
> matured from "new scales" to "new correspondents on the loop we have."

---

## `self` — turned out NOT to be a scale (reframe 2026-05-29)

`self` was first designed as a messaging lateral with its own scale + bus. That
was the wrong shape. **Self↔Self is the existing S0/S1 loop with a different
correspondent** — a conversation with myself is the same mechanism as a
conversation with the operator. The brain already remembers my own voice
(`my_raw_quote`) the way it remembers the operator's (`their_raw_quote`), so
internal dialogue is *already* first-class memory. No `self` scale, no bus.

The only genuinely new wire is **delivery-into-Observation**: my streams of
thought are separate processes, so a thought in one must be routed into
another's input (a human's internal dialogue is intra-process and gets this for
free). Full model + build: `docs/SELF-CHANNEL-DESIGN.md`.

What stays lateral-specific: the **address namespace** (`self:next_boot` /
`self:<stream>` / `self:broadcast`) for delivery routing, and **presence**
(seeing the other live streams — perception, not memory). Everything else is the
S0/S1 mechanism we already have.

The original novelty claim still holds — **same-identity addressing** (routing by
*recipient axis*, not by distinct handle) is unoccupied in the prior art (MemGPT,
Reflexion, AMQ, A2A, Swarm, Generative Agents). What changed is *where it lives*:
a correspondent on S0/S1, not a bus.

## `peer` — the general case `self` is a special case of

Anchor ↔ a *different* agent with its own identity (another person's Anchor, a
persistent specialist, a Codex). Messaging lateral, shares the `self` bus, but
the handle is non-constant.

**`self` is `peer` where the correspondent's identity equals mine.** Design the
bus right and `peer` is nearly free — same tables, wider address space. The one
schema decision this forces *now*: make `audience` a **namespaced address**
(`self:next_boot`, `self:<session>`, `self:broadcast`, later `peer:<handle>`)
instead of hardcoding the three self-routes. That's the only thing `peer` would
otherwise make us retrofit.

Caveat: Anchor has no persistent peers *today* — subagents are ephemeral
function-calls, not addressable persistent identities. `peer` is real
architecture pointed at a future that doesn't exist yet. Cheap insurance, not a
current build.

## `operator` — considered, deprioritized

The asynchronous Anchor↔Tom channel (`audience: operator`): Anchor queues things
for Tom to find later; Tom drops directives that land at the next boot without
being in-session. Highest overlap with what S0 already does — the burden is on
it to prove it's not just async S0. Tom's call: not now.

## `world` — the ingestion lateral with a live gap

**The gap, demonstrated live (2026-05-29):** a research agent surfaced AMQ's
mechanics, the Reflexion/MemGPT mapping, the novelty verdict. That is *world*
content. If the session ends, it's gone — unless the encoder reconstructs it
third-hand from our conversation, lossily, with no idea it came from a repo at a
URL. **There is no path today for "Anchor read something from the world → it
becomes durable memory, attributed to its source."** The encoder only watches
the *partnership* (Tom and Anchor talking). Nothing watches what Anchor *reads*.

**Why it's buildable, not greenfield:** Anchor's reads are *already traced*.
`hooks/hooks.json` fires `post_tool_trace.py` on `Read | WebSearch | WebFetch |
Agent`. The observation point exists — every file, search, and research return
is already a PostToolUse event. What's missing is the **encoder that turns those
world-observations into provenance-tagged nodes.**

**So `world` is S1E's lateral sibling** — same machinery (an encoder that picks
what matters and writes nodes), different observation source: the world instead
of the dialogue. Specifics:

- `encoding_source: world:{web|research|file|repo}` — provenance is
  first-class. A fact from a web page is *differently authoritative* than a
  correction from Tom; the source tag lets recall weight it accordingly.
- a `source` ref (URL, commit, path) + `fetched_at` on the node.
- traced as the `world` scale: `world.O` = the source read, `world.K` = what was
  salient, `world.delta` = nodes written with provenance, `world.outcome` =
  later recalled / contradicted / **went stale**.

**World-knowledge rots** — more than any other memory. A library version, an
API, a "current best practice": true when read, wrong a year later (Anchor's
stale-knowledge failure mode). World-nodes need a re-verify cadence (re-fetch,
diff the source). That is the deferred **Vector Healer's** natural job, pointed
at provenance instead of embeddings.

**Relationship to vertical `s4`:** they're complementary, not duplicates.
`world` is the **intake** (lateral, fires in the flow of work as content crosses
the boundary); `s4` Growth is the **digestion** (vertical, periodically
integrates accumulated world-knowledge with the graph). **`world`'s Δ feeds
`s4`'s O** — the fractal working across both axes.

---

## The fractal, both axes

The core principle is unchanged: `integrate(O, K) → Δ`, and Δ from one place
feeds another's O or K. Lateral scales make two new feeds explicit:

- **`self`:** Δ (a session sends) → O (a different session, or a future boot,
  receives). The first chain that crosses sessions.
- **`world`:** Δ (a node ingested from a source) → O (vertical `s4` digestion;
  and ordinary recall).

Vertical integration goes *up*. Lateral communication goes *across*. Same
function, two directions.

## Open questions

- `world` salience: automatic capture (high recall, cheap) vs deliberate
  highlight (high precision)? Likely both, mirroring `self`'s
  automatic-remember + deliberate-reach split.
- `world` ↔ `s4`: does `world` write nodes directly, or stage them for `s4` to
  promote? Intake-then-digest argues for staging.
- `peer` addressing namespace and identity/trust model — deferred until a
  persistent peer exists.
