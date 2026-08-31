# Staleness Propagation — a worked case, and the investigation it opens

**Status:** case documented and verified 2026-08-31. **Class B's field-coverage half is now
diagnosed and has a drafted (unpromoted) prompt fix — §6a.** Classes A, C, D, E and Class B's
*trigger* half remain undesigned. §3's Thread-1 question is answered; §4's Thread-2 experiment
is still sitting there, free to read.

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

**→ The second row is answered in §6a** (and the first row's version question is settled there
from the run's own trace, not from dates). The remaining rows are Class A, still open.

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
| **B** | **Value drift inside a correct node** — structure right, a specific value stale | `d827d22f` (9.6.0) | **encoder — partly working already** | v-next.6 repaired it unprompted this session. Two residual gaps: the trigger is coincidence (a session must discuss node and reality together), and the repair didn't reach `situation` or edge descriptions. The field-coverage half looks like a prompt/contract fix; the trigger half does not. **§6a confirms both halves of that split and diagnoses the field-coverage one.** |
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

## 6a. Thread 1, Class B answered — why the repair stopped at two fields

**Investigated 2026-08-31 (second session).** §3 left the field-coverage half of Class B as
"looks like a prompt/contract fix." It is one, and the mechanism is now specific.

**The three answers up front:**

1. **The prompt teaches multi-field revision in prose — once — and demonstrates it in 2 of its
   11 revise examples.** It is not silent. It is out-argued by its own examples.
2. **The partial repair is explained by the prompt**, not by the op shape, not by
   `content_edits`, and not by what the encoder could see. All three alternatives were tested
   and killed.
3. **Both stale surfaces were prompt-reachable** — including the edge description, through an
   affordance the prompt never names. The *trigger* half of Class B remains out of reach, as §5
   said.

### 6a.1 The forensic record

The repair is trace `703e70a5`, inside encode run `80f21f0d` (session `61608651`, stop 16,
2026-08-31 19:46 UTC). The run's own metadata reads `interaction_fingerprint: 3817564d21c4`,
`interaction_source: default`, `interaction_version: 0` — **v-next.6, production default,
pointer-less.** The version question in §3 is settled at the point of use, not inferred from
dates.

The op, verbatim from the batch input:

```
{op: "revise", node_id: "d827d22f",
 reason: "plugin.json version field is now 9.7.2, not 9.6.0; name field still says brain
          (that claim stands); title carries a stale version number that embeds and ranks
          against live queries",
 title: "…name still 'brain', version stale (9.7.2, not yet 0.9.0)",
 content_edits: [{old: "- `D-10` (v0.9.0): both manifests still say `9.6.0`.", new: "…9.7.2…"}]}
```

Two fields. And the `reason` string is the finding: *"title carries a stale version number that
embeds and ranks against live queries"* is a near-paraphrase of the prompt's own line 815 —
*"A node whose title carries the old value while its content carries the new value embeds both
into recall and ranks against itself."*

**The encoder did not miss the rule. It read the rule, quoted its rationale back, and applied
the half the rationale is about.** In the same batch it authored a node whose content diagnoses
the problem in its own words: *"The node's title carries a stale value."* A four-surface
staleness was framed to itself as a title problem.

### 6a.2 What the prompt teaches vs. what it demonstrates

Prose, lines 810–817 — explicit and correct:

> **I revise EVERY field the new information contradicts** — not just the headline. […] I update
> **title**, **content**, **situation**, and **reasoning** in one revise call. […] Half-revised
> nodes are the worst kind.

Then the examples. All 11 revise ops in the 110,452-char prompt:

| # | node | fields revised | |
|---|---|---|---|
| 1 | `4a9f21c7` | content_edits | |
| 2 | `d0e4b856` | situation | *additive — fills a missing field* |
| 3 | `97b1f24e` | title, content_edits, situation, reasoning, event_time | **multi-field** |
| 4 | `2b8ef0c1` | title | |
| 5 | `7c1a4d93` | type, title | |
| 6 | `e91a6d05` | content_edits | *labeled **Bad** (hub-only)* |
| 7 | `e91a6d05` | content_edits | sweep |
| 8 | `7d21c4aa` | title, content_edits | sweep |
| 9 | `b8e05f92` | title, content_edits, situation | **multi-field**, sweep |
| 10 | `c37d10be` | title, content_edits | sweep |
| 11 | `a45c88f1` | title | sweep |

**`question`: 0 of 11. Edge description: 0 of 11.** `revise_edge` appears **zero times** in the
whole prompt. The sweep — the example a value-drift repair is most likely to imitate — is five
ops, four of them title+content only.

The asymmetry is not mention-frequency, it is **rationale attachment**. The prompt states a
*named failure mode* for partial revision three separate times, and all three name the same
field:

- L814 — "A node whose **title** carries the old value…"
- L1219 — "a stale **title** embeds and ranks against the new content"
- L1252 — "The half-maintained alternative — **content updated, title left stale** — is the
  failure mode the brain has historically suffered from."

`situation` gets a rationale exactly once (L1378–1383, on `b8e05f92`) and it is the **dead-referent**
case — a situation *pointing* at a thing that no longer exists ("before merge" → "any rebuild").
The case that failed is a situation *repeating* a falsified value. The prompt never models it.

### 6a.3 The prompt's depiction of the catalog omits both surfaces that went stale

This is the mechanism underneath the mechanism. Every catalog excerpt in the prompt — the
before-state the encoder is shown reading — renders **title + content and nothing else**:

- **Zero** catalog excerpts render a `situation:` line. All 25 `situation:` occurrences in the
  prompt sit inside op payloads — what the encoder *writes*, never what it *reads*.
- The single edge line shown inside a catalog excerpt (L1289) carries **no description**:
  `[decision id:a45c88f1] "Rollout order: …" implements this`.

Production renders both. `build_node_catalog` uses "full rich" render; `edge_style: 'oneline'`
(the description-stripping mode) is set **only** in `surface_contract.py:838`, the recall-injection
path — not the encoder's. So the encoder's real input carries `— {description}` on every edge
line, and `situation` on every entry at both catalog tiers (`AGED_NODE_CONFIG` keeps situation;
it drops edges via `edge_limit: 0`).

**The prompt's model of its own input is lossier than the real input, on exactly the two surfaces
that stayed stale.** The encoder is never shown what a stale `situation` or a stale edge
description looks like, so it has no template for recognizing one.

### 6a.4 The alternatives, tested and killed

| candidate cause | verdict |
|---|---|
| **Encoder couldn't see the stale `situation`** | **Killed.** `situation` renders at both catalog tiers. It also patched `content` with a verbatim `old` string, which requires the full body in view. |
| **Encoder couldn't see the stale edge description** | **Killed.** `d827d22f` was pulled into the window by a hand read at turn 16, so it is a current-window id — not aged either way (`catalog_view`: an id at/after the cutoff, or with no stop at all, never ages). Full tier → edges render with descriptions. |
| **Batch op shape forbade it** | **Killed.** `situation` is a legal field on a `revise` op and the prompt states the semantics (L844: "present REPLACES, absent PRESERVES"). |
| **`content_edits` semantics** | **Not a blocker — but a real cost gradient.** Content has a cheap patch form; `title`/`situation`/`reasoning` have none and must be rewritten whole (L829–836). The prompt names this asymmetry and compensates for it *on title only*. |

### 6a.5 The completeness self-check is node-scoped, so a partial sweep passes it

The prompt's guard against exactly this (L934–939) requires a `sweep:` line every close, and
says *"writing it is how I check: a window that contains a state change next to `sweep: none` is
a contradiction I resolve."*

The encoder wrote one:

> `sweep: d827d22f title + version patched (9.6.0 → 9.7.2) · 41648eb1 marked resolved…`

It is honest, it names the fields it touched, and **it passes** — because the check only fires on
`sweep: none`. It is binary (swept / didn't sweep) where the failure is graded (swept two of
four surfaces). The one mechanism designed to catch this cannot see it.

### 6a.6 The general law — the failure migrates to whichever field the examples omit

`4b23bc51` (encoder prompt v15.11, ~3 months ago) recorded the **inverse** of this bug:

> "TITLE-SILENT on revise: Anatomy says 'content is replaced on revise' but title is unmentioned.
> The revise_batch example shows content/situation/reasoning only — never title. **Encoder follows
> example literally**; stale titles bias recall ranking toward old values."

Then titles were added to every revise example, and given a rationale, three times. The stale
value now survives in `situation`. **Partial-revision failure does not track what the prose says;
it tracks which field the worked examples omit.** That is a law with a prediction: fix `situation`
the same way and the residue moves to `question` (0 of 11) or edge descriptions (0 of 11) — both
of which carry their own embeddings and neither of which any example touches.

### 6a.7 Reachability — sharper than §5's default

Class B splits cleanly, and the completeness half lands on the **prompt** side:

- **`situation` staleness → prompt.** Visible, legal to write, weakly taught, never demonstrated
  in the failing shape.
- **Edge-description staleness → prompt, via an unnamed affordance.** `revise_edge` is **not** in
  the encoder's toolset — `ENCODING_TOOLS` (`encode.py:1633`) is exactly six tools:
  `remember_batch, revise_batch, brain_batch, connect_batch, recall_batch, get_nodes`.
  But `brain_batch`'s `connect` reaches it: `GraphDAL.add_relation` is a **field-preserving
  upsert** ("Active row exists → field-preserving UPDATE"), so re-connecting the same
  (source, target, relation) with a new `description` overwrites the stale text. The prompt
  frames `connect` purely as creation — *"wire edges between two **existing** catalog nodes"* —
  where "existing" qualifies the endpoints, not the edge. Nothing tells the encoder an edge
  description can rot or that it holds the pen.
- **The trigger → NOT prompt.** This repair happened because the conversation coincidentally
  discussed the node and reality in the same window. Nothing pre-packages "catalog claims this
  window may falsify" the way `correction_enrich` pre-packages corrections. That is Tier 1 of
  `d26d414f` — the entity-collision claims-check banner, designed 2026-08-24 and never built —
  and it stays an assembly/healer job.

So §5's "do not start by rewriting the encoder prompt" holds for Classes A, C, D, E. **For Class
B's completeness half specifically, the prompt is the right layer** — and no healer is needed for
it.

### 6a.8 Draft change — NOT registered, NOT promoted

Per CLAUDE.md the eval gate is a process rule: this lands as an override and is promoted only
after an eval passes. **Nothing below has been applied to the production default, registered as a
version, or deployed as an override.** Tom's ruling is required first. Shape follows the
v-next.6 precedent (`eeef0191`, `ebcb01e1`): surgical revisions to existing examples, no new
sections, no new assets.

**(a) Give the rationale sentence every embedded surface** — L814–817. The encoder quoted this
sentence back; it is the highest-leverage line in the document.

> …I update **title**, **content**, **situation**, **question**, and **reasoning** in one revise
> call. A stale value survives in every surface I leave it in — and title, situation and question
> each carry their OWN embedding, so a node whose title is fixed while its situation still asserts
> the old value ranks against itself on the field recall scores hardest. Before I close a revise I
> re-read the node's own catalog entry and ask which surfaces still SAY the old thing: title,
> content, situation, question, and the descriptions on its edge lines. Half-revised nodes…

**(b) Put a stale `situation` and a stale edge description in a catalog excerpt** — the
`97b1f24e` before-state at L1198–1200, so the reader sees what one looks like:

```
[fact] "Priya's yoga practice — twice a week" (id:97b1f24e)
    Priya practices yoga twice a week, started 2023-08-11. She says it
    helps her feel grounded and centered.
    Situation: When planning Priya's week — two evenings are yoga.
    [routine id:5c2d90ab] "Priya's weekly rhythm" this constrains — the two
    yoga evenings are the fixed points the rest of the week fits around
```

The existing revise op already replaces `situation`, so it now visibly *de-stales* one instead of
merely enriching it. Add one `connect` op beside it carrying the corrected description, with the
comment that makes the affordance explicit: *`connect` on a pair that already has this relation
UPDATES it — the description is mine to correct, and a stale edge `why` is a claim like any
other.*

**(c) One edge-description op in the sweep**, plus a bullet in "Why each move earns its place":
the audit node's edge to the rollout decision still describes a live branch. This is the only
place the sweep discipline currently stops at the node boundary.

**(d) Make the `sweep:` self-check field-scoped** — L934–939. Change the form to
`sweep: {event} → {node_id}:{fields}` and extend the check: *a node listed with only
`title,content` when its situation, question, or an edge line also carried the value is the same
contradiction as `sweep: none` against a state change.* Without this, (a)–(c) have no guard.

**Before any of it ships:** benchmark per §6.4. The run-44 replay (`eval/encoder_prompt_ab.py`)
is the existing instrument for the sibling-reach half; this change needs a *field-coverage*
metric that does not exist yet — scoring which surfaces of a known-stale node a revise reaches.
That instrument is the first build step, not the prompt edit.

---

## 7. Node index

`c4f7a433` stale-then-fixed open · `d827d22f` value drift, self-repaired partially ·
`3350ea51` the community · `a60e1cb7` D-13 · `d7e103a7` relocation offer · `c6910ba8` shipped ·
`cdd47782` shadow-brain guard · `644dc1e0` the open-node census · `9473c4e4`
revise-don't-duplicate · `ea0a8baf` correction_enrich · `99f5d84b` open-node state-transition
teaching · `be27aaf0` the community-lens comparison · `e6f4f9c0` communities as schema
formation · `00c15121` circular staleness · `50b4680e` the eval drain rule ·
`a40576e8` / `ccff1da4` the resolution's communities.

Added by §6a: `4b23bc51` the v15.11 title-silent gap (the inverse bug) · `d26d414f` the
three-tier design whose Tier 1 owns the trigger half · `fe0a4c9d` v-next.6's four surgical
revisions · `4f201dd1` the v-next.6 ship record · `eeef0191` / `ebcb01e1` Tom's
revise-existing-examples shape. Traces: `703e70a5` the revise, `80f21f0d` the run.

Encoded by §6a: `450650d5` the 11-op census and the rationale-attachment diagnosis ·
`08913f27` the migration law (failure tracks the omitted example field) · `7c75df5c` the
catalog-depiction gap · `17ec149b` the node-scoped `sweep:` check · `ff95dde0`
`connect`-as-upsert reaches edge descriptions.

**`d827d22f` is now fully repaired** — `situation` and the `15bbfd64` edge description both
corrected 2026-08-31, using the `connect`-upsert path §6a.7 describes. That repair also
**verified the upsert empirically**: one `gaps_in` edge to `15bbfd64` before and after, new text,
no duplicate row. The evidence of the original failure is preserved in trace `703e70a5`'s deltas
and quoted verbatim throughout §6a.
