# Thalamus — design v2

**Status: settled with Tom 2026-08-28, ready to build.** Supersedes v1 (rejected
2026-08-10 on corpus evidence); v1's fatal findings survive at the bottom as the
constraints they've become.

## The problem

Encoders raise things only a decision-maker can settle, into a store whose only
reader is the next run of the same encoder. Canonical case: consolidation asked,
three runs running, whether the decoder's suppression-verb set is meant to be
operator-configurable — a real inconsistency, correctly diagnosed, unread for
weeks. The old exit — `open ×5` → `journals-escalation` node → boot injection —
fires on **persistence, not need**, and lands in a passive channel: it saturated
all 10 boot slots with stale escalations while the real asks never arrived.

## What the Thalamus is

**The brain speaking to its streams.** A durable store of standing intents with
delivery policy — beside, not inside, the msgs layer (which is streams speaking
to each other, ephemeral, consume-once).

| | Msgs layer (exists, untouched) | Thalamus |
|---|---|---|
| What it is | a stream speaking to a stream | the brain speaking to its streams |
| Lifetime | ephemeral — TTL (1h/24h), reaped | durable — until answered / withdrawn / expired |
| Cardinality | one send, one pickup | **one item → N deliveries** |
| Memory | receipt dies at TTL | ledger `(item, session, epoch, when, via)`, forever — append-only across re-arms |

One item ≠ one message: "every session should know X on boot for the next
month" is one row, delivered N times — once per session that boots inside the
window. The msgs layer cannot model that; the Thalamus exists because of it.

## The shape

### One door, three entrances

`file()` (daemon-side) is the single producer door.

- **Anchor** — the MCP tools (below), from any session.
- **LLM agents** (S2 units, S1 Scribe) — the same `remind` tool added to their
  agent-loop toolsets, wired to the same daemon function.
- **Code** (hooks, system producers, later) — direct call.

Producers state intent; the door routes:

| Producer intent | Route |
|---|---|
| everyone alive **now** | delegate to `signal.send(broadcast)`; row goes terminal `sent` — fire-and-forget, the courier owns its death |
| everyone in a window | queue: audience `every_session` (once each, per epoch), `expires_at` window |
| next session to pull / a specific stream | queue: audience `first_session` (+ `target_session` when directed) |
| at a time ("Tuesday", "in 3 days") | queue: `deliver_at` |
| needs a decision | queue: `needs_answer=true`, loud expiry |

v-first: live-now delegation is Anchor-filed only — `from_session` is real, so
the locked stream-speech render stays honest. Machine live-now needs the
`brain`-origin render (Phase 3).

### Object

```
thalamus_items:      id (th_xxxx) · source · body · refs · audience ·
                     deliver_at · expires_at · needs_answer · dedup_key? ·
                     state · answer · created_at · armed_epoch
thalamus_deliveries: item_id · session_id · delivered_at · via (boot|prompt|stop) ·
                     armed_epoch — PK (item_id, session_id, armed_epoch);
                     APPEND-ONLY: a re-arm (defer, dedup re-file) bumps the
                     item's armed_epoch instead of deleting rows, so
                     "delivered, then deferred" stays distinguishable from
                     "never delivered" (Phase 3 retry gates on unacked)
```

Item states: `open` → `answered` | `dismissed` (by Anchor) | `withdrawn` (by its
own producer) | `expired` (window ended — **loud** terminal for an unanswered
ask, the dead-letter fix; a natural death for a notice). `sent` is the immediate
terminal for delegated broadcasts.

Delivery state and producer-condition state are **two orthogonal machines**
(id:e63c41dd): the ledger says who saw it; whether the underlying condition is
still true stays the producer's business. Retry (Phase 3) gates on *unacked*,
never on *still-open*.

### Identity & volume — the two v1 killers, and where they live now

- **Identity is never system-invented.** `dedup_key` is optional and
  producer-owned; a repeat `(source, dedup_key)` updates the item instead of
  inserting. No key is ever derived from note text — the corpus proved that
  impossible (91 subjects for one concern; 850/2,381 run-local — id:6789e133).
- **Volume is owned at the write boundary.** A per-source open-item cap
  (contract constant). At the cap, `file()` **rejects synchronously with
  guidance** — "you have N open asks: resolve, update, or withdraw." This is
  the argument for a tool over a journal protocol: a protocol's violation is
  found by a sweeper hours later with nobody to tell; a tool rejection lands in
  the agent's loop mid-run, where it can adapt. Per-render caps bound the read
  side.

### Delivery — pull at the two proven moments, on the shared last mile

Sessions pull; the Thalamus never enumerates sessions, never pushes, holds no
roster. The leg itself is owned by `servers/channels/delivery.py` — the last
mile every channel rides (ruling id:7c7e805c: the Thalamus owns NO transport).
Both hooks call `deliver(brain, ctx, moment)`; a source opts into a moment by
guarantee — `serves(source, moment) ⇔ moment.forcing ∨ source.survives_a_miss`
(ruling id:bb0513ae) — which is why the Thalamus speaks at both moments while
the consume-once courier is Stop-only.

- **Stop** (`decision:block`, forcing) — beside stream mail, in its own render
  section. The locked `render_signal` containment contract ("other stream
  says:") is untouched.
- **Boot** (`additionalContext`, passive) — replaces the `journals-escalation`
  standing-items block. Fires on fresh sessions only (resume/compaction get no
  boot render).
- **Prompt** (`additionalContext` ahead of the recall surface, passive; Tom,
  2026-09-04; plan Step 12) — the third moment, the moment a stale context is
  about to be reasoned from. Queued kinds do **not** ride it yet (asks stay
  boot; notices/reminders stay boot|stop — a cadence ruling, one policy line
  when taken). What rides it are **assists**.

**Two channels, two effects — pick the moment by the effect wanted** (Tom,
2026-09-04): a Stop `decision:block` is for the entity to *react* to something;
injected context is for the entity to *be aware* of something. Passive
injection does not fail, it influences differently. An ask wants a reaction, so
it rides Stop or the boot ask. A clock, a roster change, a "things moved while
you were away" wants awareness, so it rides the prompt — and awareness is the
whole job for a timestamp.

**Assists — the brain computing, not queuing.** An assist is a brain-side
condition evaluated at pull time for a moment, rendered inside the Thalamus
block when it holds and silent otherwise. No row, no ledger, nothing to answer
or dismiss — the condition is the identity, so it is durable without being
stored. It rides the same `thalamus_delivery` trace as the moment's queued
items (`ref_id` = moment), so it is joinable and dial-gated like every brain
utterance. The first assist is the **clock re-anchor**: the entity's "now" is
the newest timestamp in its context and goes stale silently across an idle gap
(id:8ece8811). At Prompt, when the session's last `assistant_message` is older
than the self-channel's live window (`ROSTER_LIVE_WINDOW_MIN`, the "operator
away" ceiling — one constant, one judgment), it renders one line: the gap as an
event, the UTC clock, and a nudge to re-check streams and queue before reasoning
about time. Gated, so it stays an alert and never becomes wallpaper; anchored
on the last *assistant* turn because heartbeats do not re-anchor the entity and
`hook_recall` writes the current prompt's `user_message` trace before the
surface runs. Belongs here by the admission test (id:6a11f45f: clock → Thalamus)
and by the axis rule (`servers/scales/__init__.py`: a real-elapsed clock is a
channel's, never a grain's — it is not a Frame render, though the boot Frame's
conversation-time "Now" stays as is).

Pull predicate: `state=open ∧ deliver_at ≤ now < expires_at ∧ audience matches
∧ no ledger row (item, this session, CURRENT armed_epoch)` — only
current-generation deliveries block; prior generations are history. The
ledger is written at render, for exactly the items the block shows —
annotate-at-render is the only visibility mechanism that survives receipt
expiry (id:8a170558). `delivery.py` also writes one s0 K `thalamus_delivery`
trace per moment that shows items, at boot and Stop alike. The two records
answer different questions: the LEDGER is the delivery-policy record and
already measures drain-and-answer; the TRACE buys joinability with the S0
stream (what else happened in that session's turns).

Asks default to `next-boot`: an architecture question arriving mid-thread
trains reflex-deferral; at boot there is no thread to protect.

### Return surfaces — render-joins, never write-backs

- **To Anchor**: the two renders plus `thalamus_list()` (pullable any time).
  The render is answerable without a fetch: question + resolved refs
  (`id · title · gist`) + the pre-filled `thalamus_resolve` call.
- **To producers**: same-run, the tool result ("filed th_4a2f" / the
  rejection). Cross-run, the unit's journal-view assembler joins items on
  `source` and renders a live block — open items with current state, answers
  inline. **No journal notes are written back**: they would evict real residue
  (`JOURNAL_CONTINUITY_RUNS = 3`), go stale as snapshots (id:defbdf8b), and put
  machine-authored lines in the encoder's own voice.

### Anchor's surface — three MCP tools; the budget is three

- `thalamus_list()`
- `thalamus_resolve(id, answer=… | defer_until=… | dismiss=true)`
- `remind(what, when=…, for_whom=…, needs_answer=…, refs=…)` — **the** producer
  verb, for Anchor and agents alike: a notice, a reminder, and an ask are the
  same call with different params.

Producers may update or **withdraw their own items** (source-scoped) through
the same door. Without retraction, a condition that resolved itself waits as a
stale ask — the wallpaper defect, one level up.

## Journals under this design

There is **no journal consumer and no triager**. Journals stay purely the
encoder's self-record. An encoder that wants a decision files deliberately, by
tool, mid-run — the judgment lives in the producer that has the context,
informed by its own open items rendered into its view. The triager's former
jobs: identity → producer `dedup_key` + open-items visibility; volume → the
door cap; plain language → the tool description ("written for a reader with
none of your context — no internal vocabulary").

## Phases

**Phase 1 — core, zero LLM.** Schema (`servers/schema.py` discipline) +
contract constants; `file()` with routing + budget guard; both pulls; the three
MCP tools. Prove `remind()` end to end: file → due → delivered at Stop and at
boot → resolve → ledger + trace verified. Deploy is two-step — daemon restart
(`servers/*`) AND `./redeploy.sh` + new session (`brain_mcp.py`) — before any
behavioral test can run.

**Phase 2 — producers.** `remind` into consolidation's and the Scribe's
toolsets; the journal-view join; retire `journals-escalation`, the `open ×5`
promotion, and the standing-items renderer **in the same commit as their
replacement**. Then measure the two behavioral unknowns: do encoders file
sanely (spam / under-use), and does Anchor drain and answer.
`bridge_proposals` died built-but-unused (id:bfc6d106) — delivery alone is not
success.

**Phase 2.5 — the Prompt moment and the first assist** (plan Step 12): the
third delivery moment, the assist mechanism, the clock re-anchor, and the door
echo (`remind` / `thalamus_resolve` return `now` beside the deadline they
resolved — the anchor appears exactly when the entity is doing time arithmetic).
Awareness is the effect wanted here, so the passive channel is the right one by
design; there is no "was it acted on" gate to pass.

**Phase 3 — policy.** Retry-on-unacked; machine live-now (`brain`-origin
render); queued kinds at the Prompt moment (cadence ruling); the `on_topic`
moment — deliver when a session touches related ground, the salience gate that
fully earns the name.

## Open / carried constraints

- **Scope veil — deferred for ITEMS, shipped for REFS.** Items aren't nodes
  and carry no provenance stamp. v-first producers are global (Anchor, S2
  units, the Scribe), so no walled item body can leak; the moment a
  session-scoped producer files, items need a scope story. Refs are nodes
  and DO carry walls: `pull()` resolves them through the veil-aware
  `filter_nodes` door (default-deny) — a walled node's ref renders as a bare
  id, never its title.
- Stale comments still describing the removed PreToolUse leg —
  `daemon_hooks.py:530`, `signal.py:266` — clean up in passing during Phase 1.
- Prior ruling to discharge in Phase 2 verification: LATERAL-SCALES' *"the
  burden is on it to prove it's not just async S0."* The proof is behavioral,
  not architectural.

## v1 evidence, kept as constraints

Measured against 2,381 journal notes (2026-06-24 → 2026-08-10):

| Finding | Number | Constraint it produced |
|---|---|---|
| subjects are not identities (id:6789e133) | 91 subjects, one concern; 850/2,381 run-local | `dedup_key` producer-owned or absent — never derived from text |
| no viable volume setting | floor ~0.1/day, ceiling ~10.6/day on a 2.6/day channel | volume owned at the door, not by encoder discretion |
| delivered ≠ acted on (id:ab520c77) | a consumed self-message, still missed | Stop buys one turn, not a decision — asks go to boot; Phase 2 measures draining, not delivery |
| receipts expire (id:8a170558) | 1h / 24h TTL | Thalamus owns its ledger; annotate-at-render |
| passive injection produces awareness, not reaction (removed PreToolUse leg, `364269f`; reframed by Tom 2026-09-04) | — | choose the moment by the effect wanted — Stop to react, injected context to be aware; asks ride Stop/boot, awareness items (assists) ride the prompt |
