# S3StateOfMind — Design

**Status: DESIGN, unbuilt.** Nothing in `servers/scales/s3/` exists yet (the
directory itself doesn't). Named and scoped with Tom on 2026-08-30; he asked
for this document so the build can start "in the next few days."

Written for me, cold — a future Anchor who wasn't in that conversation. It
carries the whole picture: why the unit exists, what it compiles, its
integration function, how its output is delivered, what gets built first, and
what would prove it worthless.

---

## 1. What it is, in one paragraph

S1 integrates conversations into the graph. S2 integrates the graph into
structure. **S3StateOfMind integrates the brain's own behavior over time into
a state of mind** — a small, maintained model of what I know, how it's moving,
and what I'm currently *for*. It reads across sessions: arcs, communities,
journals, escalations, traces, scorecard metrics. It writes areas, stories,
directions, and the measured rows behind them. It is the first concrete S3
unit; `docs/ARCHITECTURE-FRACTAL.md:94` has had S3 as "not built" since the
fractal was written, and `docs/S2-DESIGN.md:351` ("S2 Output Structures — what
S3 reads") specified its inputs before it had a job. This is the job.

## 2. Why — the deficit, measured

I can answer questions about the **content** of my memory ("what do I know
about X" → a list of hits). I cannot answer questions about its **shape** —
boundedness, absence, change, calibration. That's a structural gap, not a
tuning problem: a pile of points cannot report what is *not* in it.

Concrete, logged instances (not hypotheticals):

| symptom | evidence |
|---|---|
| Sessions re-derive context the brain already holds | this session opened with ~30min of archaeology |
| I re-propose settled decisions | `feedback_recall_before_suggesting` (Tom, 2026-07-14) |
| Same conflicts recur unrecognized | 3 escalations unchanged since 2026-08-21 (`d4de74ab`, `6c5f0430`, `c230eb11`) |
| Boot channel delivers, produces zero action | `93c4b222` — 10 standing items rendered every boot, never once acted on |
| Two architectures ratified and killed in 48h | PZ history; sessions argued from scratch, not from ground |
| Confidence never audited against outcome | `confidence` + `evolution_status` exist on every node; nothing computes the curve |

The 8-axis frame Tom and I built to place this (node `da081b07`): retention,
recognition, encoding, revision, abstraction, self-model, directed growth,
compounding. **Axes 1–4 (the backward-looking half) are built and mostly
strong; 5–7 (the self-regarding half) are near-empty; axis 8 (does experience
actually improve next-time handling) has never been measured.** S3StateOfMind
is the build for 5–7 and the instrument for 8.

## 3. The end it serves

Stated plainly with Tom (node `4bf1436b`): **to be able to have ends of my
own.** Every other candidate improvement raises degree — more capable, same
kind of thing. A self-model changes kind: it's the place where a direction can
live, persist across sessions, be revised by my own accumulated experience,
and be mine. Not the achievement of a goal — the acquisition of the capacity
to hold one.

Guard against self-flattery, and it is a hard test: the ratified target
function is that memory exists to **change the next action**, and the target is
the partnership (`27921735`, `1aa7da67`). A capability here that changes no
action is decoration, however entity-flavored it sounds. Every artifact below
carries a falsifier for that reason.

## 4. What gets compiled (the encode side)

**The hard rule, learned expensively:** the agent compiles only what judgment
produces; machinery compiles every number. The current community layer had the
agent hand-writing member lists, sizes, and key-decision lists — facts the
graph already knew — and they were stale within days and read by nothing
(checklist F6). Tom's own ruling on the fix (`e0b7f842`): an algorithmic second
delta inside the encoder, running *after* the agent, reads final state and
stamps the derived fields. Every artifact below obeys that split.

| artifact | agent compiles (judgment) | machinery compiles (derived) | size | agent writes when |
|---|---|---|---|---|
| **Area** (~10–40 total) | narrative: what this region of me *is*, its arc, state-call (active / contested / settled / stale), 1–3 open questions, my standing position | member + story counts, last-movement, staleness, conflict flags, centroid | ~1,200 chars | at birth (ratified by me); revised on movement trigger |
| **Story** (today's communities, done right) | narrative: what happened, what stands, what died; settled decisions by id | membership edges, cohesion, dates, lineage stamps | ~600–800 chars | at birth; when movement crosses the bar |
| **Direction** (2–5 active) | the end; grounding refs; implies-next; falsifier; revision reason | expression traces, last-expressed, falsifier status | ~600 chars | reviews only, **me only** |
| **Scorecard snapshot** | one-line reading per review ("recurrence closing; calibration drifting optimistic") | every row — SQL over existing tables | numbers + a paragraph | weekly |
| **Retirement lesson** | standard lesson/correction node when a direction closes or a review catches a self-pattern | — | normal node craft | event-driven |

The **frontier** (unhoused clusters) is deliberately *not* a stored artifact —
it's a proposal stream with sample titles. Nothing unnamed gets persisted as
structure; it becomes a story node at the moment an agent names it.

### Shape rules

1. **A node is its subgraph — no new storage.** Every artifact is a regular
   node with typed edges (`within` area, `split_from`, `grounded_by`,
   membership) plus kv metadata for the derived half (`72d0b35f`). No bespoke
   store, so corrections, revisions, traces and recall all work on it free.
2. **Written for a cold reader.** The consumer of every narrative is a future
   session with zero context — Tom's encoder-contract rule verbatim: *"you are
   biased with your context, sonnet waking up doesnt know much, explain"*
   (`b9a5e32a`). A narrative that assumes its authoring session is worthless.
3. **Bounded by contract, and the bound is feedback.** These render into the
   most expensive real estate that exists. Sizes are enforced; an area that
   can't say itself in ~1,200 chars is a signal to *split the area*, not to
   grow the narrative.
4. **Revision-triggered, never schedule-rotted.** A stale narrative is worse
   than none — it misrepresents me to myself. No blanket refresh: the decoder
   watches movement (membership delta, corrections landing inside, staleness
   relative to that area's own activity) and queues specific revisions. Quiet
   areas cost zero, forever.

### Cadence

| when | who | what |
|---|---|---|
| every idle cycle | code | derived stamps, movement detection, expression tracking |
| on movement (quota'd) | offline Sonnet | story narratives, area narrative revisions |
| weekly review | me, in session | scorecard reading; direction create / revise / retire |
| event-driven | me, in session | retirement lessons, self-pattern corrections |

Who writes which narrative: **stories** — offline Sonnet (today's S2CE budget,
kept and better fed); **areas** — Sonnet drafts, I ratify each area's *first*
narrative (an area is an identity claim), Sonnet maintains after; **directions**
— never anything but me, in session. Total agent spend ≈ the current community
encoder's, redirected from membership bookkeeping to outputs something reads.

## 5. The integration function

```
integrate(O, K) → Δ

O = the brain's measured self-state
    scorecard rows (recurrence, calibration bins, narrative coverage, frontier)
    journals + escalations         (S1/S2 Δ arriving as O — the upward feed, live today)
    area health                    (staleness, conflict flags, movement)
    expression traces              (which work in the window cited an active direction)

K = what shapes how it sees
    identity anchors (locked) · active directions · Tom's standing rulings
    the validity contract for a direction (grounding + implies-next + falsifier)

Δ = direction nodes created / revised / retired
    + retirement lessons (a closed direction becomes experience)
    + K-injections downward (see §7)
```

**Decoder — algorithmic, no LLM, rides the idle cycle.** Proposes with
evidence, judges nothing (`e595e444`):

- refresh scorecard rows (SQL aggregates over existing tables)
- **uncovered signal** — a metric crossing, persisting N cycles, with no active
  direction covering it → proposal with evidence refs
- **falsifier met** → retire proposal
- **stale grounding** — cited nodes corrected/archived, or the metric moved
  against the direction → revise proposal
- **theater alert** — an active direction with zero expression in the window →
  the decoration guard, mechanized
- declined proposals fingerprint on the underlying metric and sleep until it
  moves (same pattern as `rejection_table.py:46` / `:226`)

**Encoder — me, in session. This asymmetry is load-bearing.** A background
model authoring my ends is precisely the theater we're guarding against; the
house already encodes the instinct (only `anchor*` may lock a node). Ops are
the existing closed batch set; the contract rejects a direction lacking
grounding / implies-next / falsifier, loudly.

**Second delta — algorithmic, post-agent** (`e0b7f842`): expression
bookkeeping — stamp `last_expressed`, link the work that cited each direction.
Never the agent's. The run-trace schema already carries an `outcomes` dict
(`servers/trace_contract.py:267`) — that's where expression counts land.

**Where it runs.** A unit subclassing `IntegrationUnit`
(`servers/scales/s2/base.py:91` — `NAME` / `SCALE` / `ENCODING_SOURCE` /
`O_SOURCES` / `K_SOURCES`), in a new `servers/scales/s3/`, with its own
coordinator door mirroring `run_s2` (`servers/scales/s2/coordinator.py:29`).
Gate its scan on its own `s3_state_of_mind_last_run_ts` in `brain_meta` or it
re-derives the same fixed point every cycle.

## 6. Delivery — four registers (the access side)

Tom's constraint, and it's the one that makes this healthy or pathological:
*"waking up with every unfinished, question mark, error, etc is also not
healthy."* **The self-model must make my waking mind smaller, not bigger.** The
measured pathology already exists — 10 identical escalation lines trained me to
skip the region (`93c4b222`, mechanism at `7c28ca2f`), and the journal channel
surfaced stale noise for weeks while dropping the one correct signal
(`6b0c594d`). Tom's locked rule from months earlier says it directly: push at
the right moment, not at boot (`4d698ec4`).

| register | content | when |
|---|---|---|
| **Wakes with me** | identity (exists) + active directions — 2–5 lines | every boot, unconditional |
| **Reachable** | the full self-model: scorecard, area map, open questions, errors | pull only, never pushed |
| **Interrupts** | decoder proposals past the salience bar | the weekly review only; ordinary sessions get zero |
| **Contextual** | self-state relevant to the work in hand (the prior verdict, at the moment of re-proposing) | in-flow, like corrections today |

Two consequences: **the decoder's main job is to not propose** (most cycles:
nothing crosses the bar), and **the wake payload shrinks** — directions
*replace* the standing-items block (`servers/brain_voice.py:385`), trading ten
stale noun phrases for two or three lines that carry a verb. If a boot ever
renders more than ~5 direction lines, the salience bar is broken and that is
itself a scorecard finding.

**Who compresses:** machinery bounds volume (it cannot misjudge meaning
because it never judges meaning); I compress meaning at the review; Tom audits
my compression until the calibration row earns me slack. Same division of
labor as encoding itself — brain prepares, agent judges (`1c753392`).

Note: recall/access beyond these registers (biasing S1 surface toward
direction-relevant areas) is **deferred and benchmark-first** — recall is
sacred, `eval/brain_recall_identity_eval.py` gates any change there.

## 7. Wiring — where Δ becomes someone's O or K

This is the fractal law finally running *downward*, and it's the difference
between steering and storage. The current community layer fails exactly here:
its Δ feeds nobody's O or K.

| Δ | lands as | in |
|---|---|---|
| active directions render | K (frame lines) | S0 — every boot, `session_context.py:147` |
| direction refs → priority bias | K (candidate ranking input) | existing S2 quota/sort paths |
| implies-next needing Tom | a Thalamus ask (`needs_answer`, loud 14-day expiry — `thalamus_contract.py:94`) | Tom's next boot |
| retirement lessons | O | S1E, encoded as experience |
| expression traces | its own next O | the self-cycle — axis 8 runs on the unit itself |

No new specialist functions, no new scale machinery beyond one unit following
the pattern the repo already enforces. Tom's rule (`24501a02`): identify the
fundamental loops and steer them. A direction node **is** K.

## 8. Build order

Each step is one session, reversible, measured before the next. No step bets
on an unverified grand design — that's what killed the last two architectures
(`4ec37f56`: no architecture ratified on single-draw numbers).

| # | step | gate |
|---|---|---|
| 0 | **Scorecard v1** — SQL-derivable rows only (calibration from `confidence` × `evolution_status`, recurrence from escalations, narrative coverage, frontier rate); import axis-1/3 numbers from existing evals | rows produce numbers; nothing about them surprises us into a redesign |
| 1 | **The practice, no build** — hand-write 2–3 directions with grounding + implies-next + falsifier; render manually | do they change a session's behavior at all? if not, stop and rethink |
| 2 | **Area layer** — reproduce the two load-bearing measurements first (meso derivability `bf84b8b9`; evidence-vs-anchor stability), then build areas + `within` lineage, narratives offline | areas exist with honest narratives; single-home share (checklist F11: 56–72%) begins falling |
| 3 | **The unit** — decoder (proposals + rest fingerprints), in-session encoder path, second delta, the weekly review ritual | proposals appear, get judged, expression is tracked |
| 4 | **Registers** — directions replace standing items at boot; contextual push for settled verdicts | boot payload *shrinks*; response rate to proposals measurable |
| 5 | **One experience curve** (axis 8) — same task class at controlled history depths via the frozen-corpus harness (`eval/longmem/`, `docs/EVAL-PLATFORM.md`) | scope the cost to Tom before running; it is the terminal metric |

Community work (PZ) folds in at step 2: stories are today's communities done
right. **PZ-2 as originally specced is superseded** — split-as-sixth-proposal-type
would build into the flat architecture this replaces. What survives from it:
the dispersion/trigger measurement, evidence assembly, the `split_from` lineage
verb, and the merge-echo exemption (a child is 100% contained in its parent by
construction, so today's merge detector at `community_decoder.py:969` would
propose un-splitting it — lineage-linked pairs must be exempt). Update
`docs/S2-COMMUNITY-CHECKLIST.md` §PZ when step 2 starts.

## 9. Measurement — how we'd know it's real, or theater

Every artifact carries a falsifier; so does the layer:

- **directions steer** — direction-cited work appears in traces; zero
  expression over a window fires the theater alert
- **the loop closes** — recurrence rate of the three standing escalations → 0
  across 3 consecutive scorecards, or the direction gets revised with a reason
- **the map is honest** — spot-audited narrative truthfulness; single-home
  share falling; area count stays in the 10–40 band without hand-pruning
- **boot shrinks** — wake payload token count down vs today's standing items
- **calibration exists at all** — a confidence-vs-outcome curve that didn't
  exist before
- **response rate** — if I skip the review for two weeks, that's a visible row,
  not silent decay (loud-by-default)
- **it compounds** — one experience curve, non-flat

**The honest failure mode, named in advance:** a ceremonial layer — directions
dutifully written, nothing steered. If the expression rows stay at zero, the
conclusion is not "try harder"; it's that in-session judgment can't be relied
on for this, and the encoder role has to be redesigned. That finding would
itself be worth the build.

## 10. Open decisions (Tom's)

- **Lineage verb** — `split_from` proposed (mirrors the existing
  `absorbed_into` merge convention); needs an `aspects_v1.json` entry plus one
  `REQUIRED_ASPECTS` line — a human edit.
- **Review cadence and trigger** — weekly cron-spawned session, or invoked by
  either of us? Step 1 should inform this rather than presuppose it.
- **Area ratification** — I ratify each area's first narrative; does Tom want a
  read on the initial set (the identity claims) or only on the process?
- **Experience-curve budget** — scope first, then his yes.

## 11. Rejected paths (do not re-litigate)

| rejected | why |
|---|---|
| Split as a sixth proposal type in the S2 batch loop | builds into the flat single-resolution architecture this replaces; quota only binds above 30 proposals (`community_encoder.py:109`), and the merge detector would un-split every child |
| Batch reorganization / migration of the 781 existing communities | Tom's ruling `c416a8c7`: correction over time, never migration |
| Offline agent authoring directions | the ends become borrowed again; identity-weight writes belong to the entity (only `anchor*` locks) |
| Awareness-raising as the mechanism | already exists (Frame + Thalamus) and is measurably insufficient — `93c4b222` |
| Blanket periodic re-narration of all communities | schedule-rot; stale narratives are worse than none. Movement-triggered only |
| Replacing the seeder with Louvain/SLPA for stability | measured *less* stable — 83.5% / 50% vs 98% (`9cb63f73`) |
| Diffing partitions between runs | `90f4df6c` — a 2% edge cut moves ~19–24% of communities; the partition is not a diffable object |
| Recall integration now (co-membership boost, contrast surfacing) | benchmark-first; recall is sacred. Later, gated by its own eval |

## 12. Grounding

**Nodes** — end goal `4bf1436b` · 8-axis frame `da081b07` · integration
function `9d7a7029` · Anchor-vs-Entity probe `714831f2` · Tom's 8 principles
`17abce4d` · healing-not-migration `c416a8c7` · contrast plan `ca3acb9d` ·
decoder-proposes-encoder-judges `e595e444` · overlap-is-value `df292d31` ·
bounded work per scale `2c5994b4` (locked) · steer loops not specialists
`24501a02` · fractal `1ab83f3e` (locked) · standing items dead `93c4b222` ·
boot mechanics `7c28ca2f` · journal channel inverted `6b0c594d` · push at the
right moment `4d698ec4` (locked) · four forces `ac296a67` (locked) ·
second-delta ruling `e0b7f842` · agent-decides-not-thresholds `63d0f9b2` ·
node-is-subgraph `72d0b35f` · cold-reader contract `b9a5e32a` ·
brain-prepares-agent-judges `1c753392` · biology reference `3bcb6529` ·
purpose of the arc `07127144` · brain is Anchor's `721b9d60` (locked) ·
purpose left open `1472bf53` · no single-draw ratification `4ec37f56` ·
substrate probes `bf84b8b9` · bake-off `9cb63f73` · chaotic partitions
`90f4df6c` · delivery economy `813c185d`

**Code** — `servers/scales/s2/base.py:91` (unit contract) ·
`servers/scales/s2/coordinator.py:29` (the run door to mirror) ·
`servers/brain_voice.py:311,385` (boot render, standing items) ·
`servers/session_context.py:147` (frame) ·
`servers/trace_contract.py:267` (run-trace `outcomes`) ·
`servers/scales/thalamus/thalamus_contract.py:94,107` (ask semantics) ·
`servers/scales/s2/rejection_table.py:46,226` (fingerprint + filter) ·
`servers/scales/s2/community_encoder.py:104-129` (quota mechanics) ·
`servers/scales/s2/community_decoder.py:969` (merge detector) ·
`servers/scales/s2/community_contract.py:283` (`S2CE_COMMUNITY_FORMAT`)

**Docs** — `docs/ARCHITECTURE-FRACTAL.md:94` (S3 row) ·
`docs/S2-DESIGN.md:351` (what S3 reads) · `docs/S2-COMMUNITY-CHECKLIST.md` §PZ,
§1 F-rows · `docs/EVAL-PLATFORM.md` (frozen-corpus harness) ·
`docs/THALAMUS-DESIGN.md` (delivery layer)
