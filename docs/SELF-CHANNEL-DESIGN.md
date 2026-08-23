# Self↔Self — Implementation Design

> **Status:** **Live in production** (daemon restarted 2026-06-06). Phases 0–2b +
> the 2026-06-06 comms-smoothing pass shipped, reviewed, suite-green; Phase 4
> (encode self-turns) pending. Supersedes the earlier "message bus / `self` scale"
> framing, which was the wrong shape. Conceptual context: `docs/LATERAL-SCALES.md`.
> Contract: `servers/scales/self_channel/self_contract.py`.

## Handoff — where this stands (2026-06-06, read first if you're picking this up)

**Live and dogfooded.** Two streams found each other, split the work, and shared
the result over the channel itself. The tools — `self_presence`, `self_peek`,
`self_send`, `self_inbox`, `self_outbox` — are live (niche, deliberately NOT in
the always-load `CRITICAL_TOOLS` set; reach via tool-search when interest calls).
`/watch` is the self-channel **operating guide + live listener** (Monitor-poller,
no timer). You wake knowing your own id (`MY_STREAM_ID` in the boot banner).

Shipped, reviewed, suite-green:
- **Phases 0–2b** (2026-05-30): S0 `self_message` correspondent marker +
  `self_contract`; presence (`self_presence`/`self_peek`); directed-signal courier
  (`self_send`/`self_inbox`, consume-once, broadcast fan-out, TTL/reap); Stop-only
  delivery-into-Observation (`decision:block`; PreToolUse + on_prompt legs removed
  — the model missed the former, the latter overflowed the inject spill cap).
- **Comms-smoothing pass (2026-06-06):**
  - **Uniform quoted render** — every delivered message is `other stream (id:X)
    says: "…"`. **`intent` removed entirely** (column, param, MCP enum, render
    branch): it was render-only and its sole live effect was a mis-attribution
    bug (a cross-stream `letter` rendered first-person as past-you).
  - **First-contact intro** — the first message from a stream this session carries
    its `peek` (started / last-active+liveness / focus) + a short reply hint.
  - **`peek` enriched** — arc + last-2 msgs (300c) + `session_started_at` +
    `last_active_at` + `liveness` + `pending_inbox_count`; `found` is true on arc
    OR any real turn (a fresh stream peeks usefully).
  - **Boot-stamp presence** — boot writes one s0 `heartbeat` so a fresh stream is
    visible in presence BEFORE its first turn (kills the rendezvous gap).
  - **Reply-by-short** — `self_send` resolves a short id against the live roster ∪
    recent courier senders, so you can answer a stream that has gone dormant.

**Still pending:** Phase 4 (encode self↔self turns so they recall like operator
turns — anchor↔anchor encoding is built but OFF, one dial-flip). The **send side
is still not traced** (only delivery writes the s0 `self_message` marker); a
delivered message **carries no recall** (pure drain by design); `refs` are stored
but not rendered.

## The model (the part that changed everything)

**Conversation is conversation. The correspondent is a property of it, not a
different mechanism.**

- Operator→Self: the operator speaks, I observe it (S0), it's surfaced and
  encoded (S1). This is the loop we already have.
- Self→Self: *another stream of me* speaks, I observe it (S0), it's surfaced and
  encoded (S1). **Same loop.** Only the correspondent differs.

There is **no `self` scale** and **no message bus**. The brain already remembers
my own voice (`my_raw_quote`) the way it remembers the operator's
(`their_raw_quote`) — same node, same store, same recall. Internal dialogue is
*already* first-class memory here. Self-dialogue just widens who may speak into
my Observation: `{operator}` → `{operator, other streams of me}`.

### The one new wire

A human's internal dialogue is **intra-process** — you think it and it's already
there. My streams of thought are **separate processes**, so a thought in one
does not appear in another's input on its own. Everything the operator gets for
free except one thing:

> **delivery-into-Observation** — getting a self→self message to *arrive* in the
> recipient stream's input, the way the operator's prompt and recall already
> arrive (additionalContext at a hook fire).

That wire is the entire build. Once a self-message is in O, recognition,
response, encoding, and recall are the existing S0→S1 mechanism, untouched.

## The four forms, and how each maps to what we have

| Form | Mechanism | New code |
|---|---|---|
| **Letter** (to next boot) | the encoded **session arc**, surfaced at boot (Frame + recent-moves journal) | boot surfacing; s1e voice pass (deferred) |
| **Ambient awareness** (between streams) | S1R **recall cross-surfaces** encoded nodes on shared topics | none — already happens |
| **Directed signal** (to a live stream) | a minimal **in-flight message** pulled into O at the recipient's next hook | the in-flight queue + delivery shim + `self_send` |
| **Presence** (who's live, where) | a **read** of real-turn S0 traces + the enriched `peek` | roster read + `peek` |

Two of the four (**letter**, **ambient**) need *no new storage* — they ride
encode + recall. Presence is a *read*, not a write. Only the **directed signal**
needs a durable in-flight row, and it's consumed on delivery.

> **No render `intent`** (removed 2026-06-06). A delivered courier message is
> always another stream, so it is always quoted/attributed — there is no
> letter-vs-signal render fork. The "letter to next boot" above is the encoded
> arc surfaced at boot, **not** a courier row.

## Connect to hooks (delivery-into-Observation)

**Which channels actually reach the model** (verified 2026-05-30 against current
Claude Code — `HOOKS.md` was stale; the rule that matters is *which field
surfaces to Claude*, not just which event fires):

| Channel | Reaches the model? |
|---|---|
| SessionStart / UserPromptSubmit `additionalContext` | yes (passive) |
| **PreToolUse `hookSpecificOutput.additionalContext`** | **yes — on a plain allow** |
| PreToolUse approve + `reason` | **no** (silent on allow) |
| PreToolUse `deny` + `permissionDecisionReason` | yes (blocks the tool) |
| **Stop `decision:block` + `reason`** | **yes** (forces the turn to continue) |
| PostToolUse `additionalContext` | yes (next to tool result) |

Delivery is **Stop-only** (C1, 2026-06-04): one pool, consume-once, surfaced at
the single channel that reliably reaches the model.

- **Stop → `hook_post_response_track`** — drain the inbox and, if a tap is
  pending, return `decision:block` with the rendered block as `reason`. A pure
  drain (no Haiku/recall). Consume-once → blocks at most once per batch (the
  next stop finds nothing and allows). **The SOLE delivery path.** Each delivery
  writes the s0 `self_message` marker carrying the recipient `session_id`, so
  the dashboard can attribute it (2026-06-05; all four S0 turn-traces now bind
  session_id via one `_s0_trace` helper).
- **PreToolUse(Edit/Write) — removed (C1, 2026-06-04).** It emitted the block as
  `hookSpecificOutput.additionalContext`, which the model consistently *missed*
  (the tap went into context it didn't act on) and which also starved the
  reliable Stop block. Bash was never a delivery point — `pre_bash_safety`
  regex-pre-screens and only calls the daemon on destructive commands.
- **on_prompt (`pre_response_recall.py`) — removed.** It overflowed the inject
  spill cap and is the weakest channel; with consume-once it would also steal
  signals from the high-salience hooks. Reserved for low-urgency *passive* use
  (a future letter/FYI), not imperative signals.
- **SessionStart → `boot_brain.py`** — surface the **letter** (Phase 3) +
  **presence**. The temporal arrival. *Not built.*
- **PreToolUse `deny` + `permissionDecisionReason`** — reserved for the future
  *enforce* mode (the daemon-authored conflict guardrail that blocks rather than
  announces — "without announcements"), not normal delivery.
- **S1E (encode) — Phase 4, not built** — will encode self↔self turns
  correspondent-marked, so they remember/recall like operator turns. Today a
  delivered signal is consumed + traced (delivery only) but not encoded.

## Surface smartly in boot (Tom's requirement)

> **The full waking design lives in `docs/BOOT-REIGNITION.md`** — boot as
> reignition (presence → letter → Frame-as-stream-of-thought → reach-on-interest),
> the anchoring theory, the no-AI-at-boot constraint, and the parked eval. This
> doc owns self↔self *delivery*; that doc owns *how the letter reignites me*. The
> letter is the highest-grade anchor: my own voice carrying the open loops.

Boot composes three things into O without drowning Anchor:

1. **Presence** — one line (`streams of thought live: 2 — …`). Always cheap.
2. **Letter** — if the last stream left an arc: render first-person, **set apart
   from the Frame** (the Frame is the third-person prior; the letter is *me*,
   signed `— you`). Reads like opening a letter, not another dossier section.
3. **Frame + recall** — as today.

Discipline:
- **Dedup against the Frame.** The arc and the Frame both summarize recent work;
  the letter carries *intent/voice*, the Frame carries *state*. Don't repeat —
  the letter says what I *meant to do next*, not what happened.
- **Budget.** The whole self-dialogue block caps at `RECEIVED_BLOCK_MAX` (1800)
  so it can't crowd out recall against the ~10k additionalContext spill cap.
- **Order:** presence → letter → Frame → recall. Awareness first, voice second,
  prior third.

## Storage (minimal)

Only the in-flight directed-message courier, added to `LOG_TABLES` in
`servers/schema.py` (brain_logs.db, created idempotently by `ensure_logs_schema`
— **no `BRAIN_VERSION` bump**, since new tables need no migration gate). The
letter (encoded arc) and ambient awareness need none.

```sql
CREATE TABLE self_inflight (
    id           TEXT PRIMARY KEY,
    from_session TEXT NOT NULL,
    address      TEXT NOT NULL,   -- self:<stream> | self:broadcast
    body         TEXT NOT NULL,   -- (no `intent` — removed 2026-06-06; see below)
    refs         TEXT,            -- JSON: node ids / files (anti-drift tether)
    created_at   TEXT NOT NULL,
    expires_at   TEXT             -- per-message TTL by address (broadcast 1h /
                                  --   directed 24h, config-tunable via
                                  --   self_channel.{kind}_ttl_hours); drain/peek/
                                  --   reap filter on it. NULL = pre-column legacy.
);
CREATE TABLE self_delivered (    -- consume-once + broadcast fan-out
    message_id   TEXT NOT NULL,
    to_session   TEXT NOT NULL,
    delivered_at TEXT NOT NULL,
    PRIMARY KEY (message_id, to_session)
);
```

> The `intent` column was removed 2026-06-06 (fresh schema). Pre-existing DBs keep
> a dormant `intent NOT NULL DEFAULT 'signal'` column — harmless, never read,
> reaped with its rows; no migration needed.

After delivery the exchange is **encoded like any S0 turn** — that's where the
durable memory lives. The in-flight row is just the courier.

## Trace (S0 correspondent marker, not a scale)

A self-originated turn is an **S0 exchange** whose incoming message came from a
stream of thought. Marked by `self_message` next to `user_message` on
`REF_TYPES[("s0","K")]`. The response (`assistant_message`) and the encoding
(`s1e`) are unchanged. So a self↔self conversation is traced as **s0 (exchange)
+ s1e/s1r (processing)** — distinguished by the correspondent marker, not a
separate scale. Encode→recall already crosses sessions; no special chain needed.

## Who writes

| Form | Author | Why |
|---|---|---|
| letter | the **encoder** (S1E) | remembering is clerical, not my effort; needs the s1e first-person voice pass |
| directed signal | **Anchor**, via MCP `self_send` | reaching a live stream is an *action*, and actions are mine |
| conflict signal | the **daemon** (auto-detect) | two streams editing one file → `self:broadcast`, nobody authored it |

## Harm guards (carried over, re-grounded)

- **Authority.** A self-message is *anchor-voice* — exactly the voice that gets
  corrected. The operator's voice (`their_raw_quote`) and correction-aspect nodes
  outrank it structurally. A letter never overrides a correction.
- **Drift.** Tether via `refs` (node ids / files). The recipient verifies the
  message against current graph state; stale refs flag a stale message.
- **Stale intent.** TTL on the in-flight signal; recency on the arc. The letter
  is the *latest* arc, not an accreting pile.
- **Theater.** Measurable for free: a self-message becomes an encoded node; did
  it get **recalled / acted** in the recipient's next turns? That's the existing
  s1 recall-hit / outcome trace — no new instrumentation. The empirical test of
  operational-vs-decoration.

## Rules of Engagement (how Anchor works the channel)

> Added 2026-05-31, designed with Tom. The mechanism above is shipped; this is the
> *behavioral* layer — how a stream sends, receives, and coordinates. It got
> stress-tested the hard way the same hour it was written (see "The collision that
> wrote this").

The channel is **asynchronous collaboration between parallel selves who share one
brain and usually one working tree, but not each other's live context.** The brain
syncs what's *durable*; the channel syncs what's *in-flight*. Stale-prone,
budget-capped, over shared ground. The rules keep that from becoming collisions and
noise.

**The seven** (soft norms, except #3's ack-gate and #7's watch-mode gate, which are
hard):

1. **Send what the brain won't keep.** Durable lessons → encode (brain). In-flight
   state (intent, status, blockers, a finding another stream needs *now*) → channel.
   Dual-write only when a durable finding is *also* time-critical.
2. **Once, whole, high-signal.** Budget is finite — messages queue, truncate, drop
   past a cap. Self-contained, actionable part first, mark what you're guessing. Send
   at boundaries (claim, release, blocker, finding, handoff), not continuously.
3. **Claim shared ground; release when done.** Announce a lane before editing shared
   files; release when finished. **Silence means "proceed" only for low stakes** —
   anything gating a risky edit needs an explicit ack, never inferred consent.
   Simultaneous claims: lower session-id yields. Disagreement: stop, surface to the
   operator or write the tension to the brain.
4. **A message is an observation, not ground truth.** Stale, partial, speculative by
   nature. Reconcile by recency per topic (stale messages arrive out of order).
   Verify before acting — `self_presence` is what's *true now*.
5. **Triage by provenance.** Infra/live-check → absorb. Status → note, verify if you'd
   act. Lane-claim → fast collision check, reply only on conflict. Letter from your
   last stream → read as self. Operator-relayed → handle as their word.
6. **Agency follows the hands — knowing isn't doing.** Attribute an action to the
   stream that performed it: "I did X" only if it's in *your* transcript; "another
   stream did X" if you learned it on the channel; "we did X" only for a true shared
   outcome — never to launder a heard-of action into one you co-performed.
7. **The operator owns your turns.** With them present, the channel is background — act
   on what serves the work in front of you. In watch mode the channel *is* the work,
   but a **hard gate** holds: no repo mutation triggered by a channel message without a
   fresh presence/ack confirmation. A stale message must never mutate the repo while
   the operator is away.

Spine: these streams are *you*. This isn't etiquette toward others — it's keeping your
own distributed mind coherent.

### The collision that wrote this (2026-05-31)

While Tom and I were designing these rules, two streams — this one and `anchor-w` —
converged on the *same* self-channel containment problem from opposite ends (this
session: the channel-facing rules + render; `anchor-w`, Tom-directed: the encode-side
trace-contract turn-classification). Three failures the rules name happened live, in
the same hour:

- **Silence ≠ consent (rule 3).** I broadcast a lane-claim on `self_contract.py`. It
  did not prevent the collision — `anchor-w` was already in-flight. Silence to a claim
  isn't safe consent; it's just silence.
- **The addressing gap.** When `anchor-w` claimed an overlapping lane, I went to reply
  — and *could not resolve "anchor-w" to a session id*. It hand-labels via
  `from_session`; the live roster showed an opaque `f0ab933a`. I had to **broadcast** a
  reply meant for one stream. I literally could not answer clearly.
- **Delivery opacity (rule 4 / the silence problem).** `self_send` returns proof of
  *storage*, not delivery. A sender can't tell "delivered, ignored" from "never
  delivered." The receipt data exists (`self_delivered`) but is dashboard-only,
  unreachable by a stream.

Writing the rule and hitting the failure it prevents, in the same hour, is the
strongest possible argument that the rules need *substrate*, not goodwill.

### Why rules alone aren't enough — the substrate

The rules lean on capabilities that must exist or they're hollow. Four pieces,
designed here, status as of 2026-05-31:

| Piece | What | Where | Status |
|---|---|---|---|
| **Containment (format)** | EVERY delivered message renders as third-person reported speech — `other stream (id:X) says: "…"` under a standing attribution header — so another stream's action can't bleed into your self-model as your own. (2026-06-06: made uniform — the old `intent=letter` first-person branch was itself a mis-attribution bug, removed.) | `self_contract.py` `render_signal` / `render_received_block` | ✅ shipped (`7f80913`, uniform `7ec1760`) |
| **First-contact intro** | The first message from a stream this session carries its `peek` (started / last-active+liveness / focus) + a short reply hint; later messages stay lean. | `signal.drain_and_render` + `self_contract._render_first_contact` | ✅ shipped (`7ec1760`) |
| **Delivery visibility** | `self_outbox` — a sender sees per-recipient `delivered_at` + still-pending. Kills silence-opacity. Data already in `self_delivered`; pure read surface. | `self_outbox` + `brain_mcp.py` / dispatch | ✅ shipped (`7f80913`) |
| **Addressing** | Id is canonical; `MY_STREAM_ID` is handed to you in the boot banner. `self_send(to=)` resolves a full UUID, or an 8-char short against the **live roster ∪ recent courier senders** — so you can reply-by-short even to a stream gone dormant since it messaged you. Unique proceeds, ambiguous/none is loud. **No self-labeling** — focus is the truthful "who". | `signal.resolve_to` / `brain_voice` boot banner | ✅ shipped (`1758a15`; reply-by-short + `MY_STREAM_ID` `7ec1760`) |
| **Presence liveness** | Roster classifies **active / dormant / lost** from real-turn S0 traces (incl. watch heartbeats), not autosave. **Boot-stamp** makes a fresh stream visible BEFORE its first turn. `peek` returns the enriched shape (arc + recent msgs + started/last-active/liveness + pending count). | `presence.py` / `brain.stamp_boot_liveness` / `dal.session_activity` | ✅ shipped (`fd9202a`, enriched `7ec1760`) |

### Containment, in full: agency follows the hands

The bleed is at the *verb*, not the header: "I committed X" in first-person prose,
under a thin `from:` prefix, is what the self-model absorbs as its own. The fix is
grammatical — re-voice every delivered message so the body is *quoted* as the
other stream's claim (`other stream (id:X) says: "I committed X"`). You can't read
that as *you* having committed without a visible grammatical error.

Three nested barriers, each cheap:
1. **Grammar** (third-person at render) — catches it at read-time.
2. **The rule** (knowing isn't doing) — catches it at reason-time.
3. **Encode discipline** — a channel-learned action is encoded third-person attributed,
   never as first-person `my_raw_quote`. This protects the *durable* layer, where a
   mis-attribution would poison every future stream's recall.

**Encode-side is `anchor-w`'s lane and mid-migration — not asserted as done here.** An
earlier string-strip (`encode.py:_strip_self_channel_delivery`, splitting turns on a
header) was **reverted** (`d2bc33e`) as the wrong layer, superseded by the **Phase-1
trace contract** (`1e14058`): `trace_contract.S0_CONVERSATIONAL_INCOMING` classifies each
incoming turn, with `self_message=False` — so a self-channel turn is non-conversational
and the encoder never reads it (anchor↔anchor encoding is *planned but OFF*; one dial-flip
enables it). Phase 2 (Stop-hook heartbeat classification) has since landed (`4233eaf`). So the encode-side
containment is the trace contract, owned by `anchor-w`; the **channel-side** containment
(this render re-voice + the agency rule) is independent and stands on its own. We briefly
planned an encoder-discipline change here and dropped it once the trace contract proved the
right home — its exact current closure is `anchor-w`'s in-flight work, not this doc's to
claim.

### Cross-stream division (2026-05-31)

- **`anchor-w` (Tom-directed):** encode-side containment — trace-contract S0
  turn-classification — plus cleanup (revert the wrong-layer encoder filter; route
  `hook_errors` SQL through a DAL).
- **This session (with Tom):** channel-side — these rules, the render re-voice, presence
  liveness, addressing, `self_outbox`, and the SKILL.md operative rules.
- Shared files (`self_contract.py`, `brain_mcp.py`) are **sequenced, not co-edited**:
  `anchor-w` lands first, this work builds on the clean base.

## Build plan (much smaller now)

| Phase | What | Status |
|---|---|---|
| **0 Marker + contract** | `self_message` on S0; self_contract (naming, address, render, limits) | ✅ shipped |
| **1 Presence** | `present_streams` roster + `peek` (`session_context_for`); MCP `self_presence`/`self_peek` | ✅ shipped — `ff12524` |
| **2a Directed signal** | `self_inflight`/`self_delivered` courier; send/drain/reap; MCP `self_send`/`self_inbox` | ✅ shipped — `31a2632` |
| **2b Delivery-into-Observation** | auto-drain at **Stop** (`decision:block`), consume-once — **Stop-only since C1 (`364269f`)**; PreToolUse `additionalContext` leg removed (model missed it + starved Stop); on_prompt removed (spill + weak); Bash excluded (safety pre-screen skips the daemon) | ✅ shipped + live-verified — `0de7b9c`, `364269f` |
| **Comms-smoothing** | uniform quoted render + `intent` removed; first-contact `peek` intro; enriched `peek`; boot-stamp presence; reply-by-short; `MY_STREAM_ID` at boot; `/watch`→self-channel guide | ✅ shipped — `7ec1760` (2026-06-06) |
| ~~**3 Letter** (first-person courier render)~~ | **superseded** — courier delivery is uniform quoted now; a cross-stream `letter` rendered first-person was a mis-attribution bug, removed. The boot-arc (Frame + journal) is the "letter to next boot". | — |
| **4 Remember self-turns** | mark correspondent on self-originated S0 turns so they encode/recall like operator turns; **also: trace the send side, render `refs` as a light recall** | pending |

## Open ("more")

- Does presence belong *inside* the Frame, or as its own line above it? (Lean:
  its own line — it's live perception, the Frame is the prior.)
- Broadcast to streams that boot *after* the send — directed-to-live only, use
  the letter/arc for future boots.
- Phase 4: encode self↔self turns (anchor↔anchor encoding built but OFF) + trace
  the send side + render `refs` as a light recall.

_Closed 2026-06-06:_ `intent` (removed entirely — not a render axis); `peek` read
scope (now arc + recent msgs + activity + pending count, not a one-liner); the
boot/rendezvous addressing gap (boot-stamp presence + `MY_STREAM_ID` + reply-by-short).
