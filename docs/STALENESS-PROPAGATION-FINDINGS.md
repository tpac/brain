# Staleness Propagation — a worked case, and the investigation it opens

**Status:** case documented and verified 2026-08-31; **resolutions NOT designed — that is the
next session's job.** Written as the entry point for that session.

**The one-line case.** A single missing edge made three layers wrong at once: a resolved
question stayed `open` for three weeks, the community summarizing it faithfully reported a
solved problem as "the final blocker," and the planning doc — written from the same
understanding — agreed. Nobody lied; the graph simply never learned that the answer had
arrived.

---

## 1. What happened, verified

Discovered while answering "what's left on the public deployment plan." Every claim below was
checked against code, git, or a live node read — not inferred.

**The stale node.** `c4f7a433` (type `open`, `encoder:sonnet`, created 2026-08-10):
*"brain.db in CLAUDE_PLUGIN_DATA = host coupling hazard."* Present tense, ending
**"Open: Tom hasn't ruled."**

**Its answer, encoded twelve days later and richly:**
- `a60e1cb7` — **D-13, 2026-08-11**: Tom ruled brain data out of the host's folder *entirely*,
  XDG default. It went *further* than `c4f7a433` proposed, on exactly the multi-host
  rationale that node raised.
- `d7e103a7` — the relocation offer, "the missing half of D-13" (2026-08-28)
- `c6910ba8` — **SHIPPED**, `e1f9f5d` (2026-08-28)
- `cdd47782` — the shadow-brain staging guard that the mover required

**The failure.** `c4f7a433`'s edges were `community_member`, `exposes_gap_in`, `generalizes`,
`co_anchored` — **four sideways edges, zero resolution edge, no retype, no archive.** Nothing
pointed from the answer back to the question.

**The propagation.** Community `3350ea51` ("Public Repo Launch: From Decision Lock to
File-State Hazards", `s2:community_detection`, revised 2026-08-11) reported:

> *Latest Development: Host coupling hazard (id:c4f7a433) exposes the final blocker …
> Maturity: active*

That summary is **correct about its input.** The community was not independently stale; it
faithfully rendered a member that was. Blaming the abstraction layer here is a level error.

**A second, different failure in the same community.** `d827d22f` claimed both manifests read
`9.6.0`. They read `9.7.2` since 2026-08-28. **This node self-repaired mid-session on
2026-08-31** — the current encoder read the conversation discussing `plugin.json`'s real
version and revised title and content unprompted. But the repair was **partial**: the
`situation` field still says *"9.6.0 is still the version in both manifests"*, and the edge
description to `15bbfd64` still says *"both manifests still say 9.6.0."* The stale value
survived in `situation` — the field recall scores on.

**And the docs agreed.** `DISTRIBUTION-READINESS.md` carried three stale claims found the same
hour: D-5 called "the open design gate" (closed 2026-08-30), `displayName` called unruled
(ruled `Entity` 2026-08-28, already in both manifests), and a derived "operator decision still
open." Corrected in `ce46177`.

**Fixed today (do not redo):** `c4f7a433` now carries its resolution, `evolution_status:
resolved`, and two real edges — `a60e1cb7 --resolves-->` and `c6910ba8 --closes-->`. The doc
claims are corrected. Everything else below is open.

---

## 2. Scale — this is a known, measured class

`644dc1e0` (measured 2026-08-23) over live `open` nodes:

| shape | count | encoder-authored |
|---|---|---|
| fully resolved, still `type='open'` | 17 | 10 |
| partially answered, never narrowed to the residue | 8 | 8 |
| false positives (genuinely open) | 11 | — |

**241 `open` nodes live as of 2026-08-31.** `c4f7a433` was not in that count's resolved set —
it is an additional instance, which suggests the measurement undercounts or the class grows.

`644dc1e0` also names the three correct exits: **retype** (fully answered), **narrow to the
residue** (partially answered — `partially_resolves` exists, 11 uses, unused for this), and
**close-and-mint** (answered *and* spawned a new question).

---

## 3. Thread 1 — how does the CURRENT encoder handle this?

**The timeline makes this a real question, not a rhetorical one.** The current encoder,
v-next.6, became the production default in **`522c383`, 2026-08-25**. Both stale nodes were
authored **before** it: `d827d22f` on 2026-08-08, `c4f7a433` on 2026-08-10 — 17 and 15 days
early. So they are pre-launch artifacts and prove nothing about v-next.6 on their own.

**There is already one live datapoint, from this session (2026-08-31, v-next.6 in production):**

| behavior | v-next.6 result |
|---|---|
| repair a wrong value in a node when the conversation contradicts it | **YES** — `d827d22f` revised unprompted, title + content |
| reach every field with that repair | **NO** — `situation` and an edge description kept `9.6.0` |
| close/retype a resolved `open` node it was discussing | **NO** — `c4f7a433` untouched until I did it by hand |
| add a resolution edge from answer back to question | **NO** — none added |

**What to test properly.** Replay the 2026-08-11 D-13 conversation (the one that resolved
`c4f7a433`) against a brain seeded with `c4f7a433` present, under v-next.6, and ask: does the
encoder find the open node and close it? The frozen-corpus harness can do this — and note
`docs/EVAL-PLATFORM.md`'s warning about what a topic corpus can and cannot measure, plus the
per-turn embedding-drain rule (`id:50b4680e`) or the open node will be invisible to recall.

**Prior art to read first, so this isn't re-derived:** `9473c4e4` (revise-don't-duplicate, and
its own note that "the encoder has recall and get_node tools; it just doesn't know to use them
for corrections"), `ea0a8baf` (correction_enrich Layer 3.5 — corrections are *pre-packaged*
for the encoder rather than recalled, chosen because recalls are expensive), and `99f5d84b`
(open-node state-transition teaching — three branches, a pilot that was designed and may not
have run). **The key question those raise:** correction_enrich pre-packages *corrections of a
node*. Does anything pre-package *open questions this turn resolves*? If not, the encoder is
structurally blind to Class A below, and no prompt wording fixes it.

---

## 4. Thread 2 — does a new correction edge trigger community revision? (OPEN)

**Tom's question, unresolved. I do not know the answer and did not verify it.** What is known,
as leads:

- S2 `community_detection` is **throttled** — observed live this session:
  `skipped 'throttled (0m < 30m min interval)'`.
- Per `CLAUDE.md`, an S2 unit's graph scan is gated on its own `s2_<unit>_last_run_ts` in
  `brain_meta`, "or it re-derives the same fixed point every cycle."
- A **rest/suppression mechanism keyed on a 1-hop neighborhood fingerprint** exists in
  `s2_rejections` (the Phase-2 unplaceable-marking work, 2026-06-23). **This is the most
  likely governing mechanism** — if the fingerprint includes edges, my two new edges on
  `c4f7a433` should wake the decoder for `3350ea51`; if it only includes membership, they
  won't.
- Owner to read: `servers/scales/s2/rejection_table.py` + `docs/S2-DESIGN.md`.

**A natural experiment is already running.** I added `resolves` and `closes` edges to
`c4f7a433` on 2026-08-31 ~19:52 UTC and changed nothing about its membership. So:
**re-read `3350ea51` in the next session and see whether its `Latest Development` still names
`c4f7a433` as the final blocker.** If it does, the trigger does not fire on edge changes
alone, and that is the finding. No setup required — just look.

---

## 5. Thread 3 — the classes, and which layer can actually fix each

Tom's framing: *"several classes, some we solved with the encoder prompt but others might be
harder to solve through encoder and require S2Healer or maybe even S3Healer that weaves
between communities and nodes."* The five classes today's case exhibits, with my read on the
right owner. **These are proposals, not rulings.**

| # | class | instance | likely owner | why |
|---|---|---|---|---|
| **A** | **Resolved-but-unlinked** — the answer is encoded elsewhere and never linked back; node stays `open` | `c4f7a433` | **S2 healer**, not the encoder | The encoder sees one turn's window. Recognizing that *this* turn answers a three-week-old question requires a graph-wide scan for open questions semantically matched by newer nodes — a background sweep, not an in-turn judgment. Unless something pre-packages candidate open questions the way `correction_enrich` pre-packages corrections. |
| **B** | **Value drift inside a correct node** — structure right, a specific value stale | `d827d22f` (9.6.0) | **encoder — partly working already** | v-next.6 repaired it unprompted this session. Two residual gaps: the trigger is coincidence (a session must discuss node and reality together), and the repair didn't reach `situation` or edge descriptions. The field-coverage half looks like a prompt/contract fix; the trigger half does not. |
| **C** | **Derivative staleness in the abstraction layer** — community faithfully summarizes a stale member | `3350ea51` | **S2, via Thread 2** | Nothing is wrong with the community's reasoning. It corrects itself iff members are corrected *and* the revision trigger fires. Depends entirely on Thread 2's answer. |
| **D** | **Doc/graph divergence** — the doc and the graph disagree, neither authoritative | 3 claims in `DISTRIBUTION-READINESS.md` | **process / test, not the brain** | No healer can reach a markdown file. Options: a conformance test that reads decision nodes and greps the artifacts, or docs that cite node ids and get audited. Note `00c15121`'s "circular staleness" — a citation chain between docs is not verification. |
| **E** | **Cross-community orphaning** — problem arc and solution arc live in different communities with no cross-link | `3350ea51` holds the problem; `a40576e8` + `ccff1da4` hold the resolution | **the S3-shaped gap** | This is the one nothing owns today. Both communities are internally coherent and neither is wrong. What's missing is a layer that notices *"the blocker named in community X was resolved by the arc in community Y"* — weaving between communities, which is exactly the S3Healer intuition. See `e6f4f9c0` (communities as schema formation) and [S3-STATE-OF-MIND-DESIGN.md](S3-STATE-OF-MIND-DESIGN.md) for where this would sit. |

**The through-line worth testing as a hypothesis:** every class here is a *linkage* failure,
not a content failure. Nothing was written wrong. A → missing edge. C → missing propagation
along an edge. E → missing edge between communities. D → missing link between graph and doc.
Only B is genuinely about content, and it is the one the encoder already partly handles. If
that holds, the lever is edge formation and propagation, not better prose — which would also
explain why prompt work hasn't dented it.

---

## 6. First moves for the fresh session

Ordered cheapest-first; the first two cost nothing.

1. **Read `3350ea51`.** The Thread-2 experiment is already running (§4). Its answer determines
   whether Class C needs work at all.
2. **Re-measure the open-node census.** 241 live `open` nodes; `644dc1e0`'s split is 8 days
   old and `c4f7a433` was outside it. Sort into `644dc1e0`'s three exits. Mechanical.
3. **Check whether anything pre-packages open questions for the encoder** (§3). This is the
   single highest-leverage unknown: it decides whether Class A is a prompt problem or
   structurally out of the encoder's reach.
4. **Then** design resolutions — and per this repo's own rule, benchmark before changing the
   encoder or S2 (`docs/EVAL-PLATFORM.md`).

**Do not start by rewriting the encoder prompt.** Three of the five classes are not reachable
from there, and the one that is already half works.

---

## 7. Node index

`c4f7a433` stale-then-fixed open · `d827d22f` value drift, self-repaired partially ·
`3350ea51` the community · `a60e1cb7` D-13 · `d7e103a7` relocation offer · `c6910ba8` shipped ·
`cdd47782` shadow-brain guard · `644dc1e0` the open-node census · `9473c4e4`
revise-don't-duplicate · `ea0a8baf` correction_enrich · `99f5d84b` open-node state-transition
teaching · `be27aaf0` the community-lens comparison · `e6f4f9c0` communities as schema
formation · `00c15121` circular staleness · `50b4680e` the eval drain rule ·
`a40576e8` / `ccff1da4` the resolution's communities.
