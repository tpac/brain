# Self↔Self — Implementation Design

> **Status:** Phases 0–2b shipped + live-verified (2026-05-30); Phases 3–4 pending. Supersedes the earlier
> "message bus / `self` scale" framing, which was the wrong shape. Conceptual
> context: `docs/LATERAL-SCALES.md`. Contract: `servers/scales/self_channel/self_contract.py`.

## Handoff — where this stands (2026-05-29, read first if you're picking this up)

**Both halves shipped: pull (presence) + push (directed signal, auto-delivered). You wake with the tools live and delivery active at tool/turn boundaries.**

Shipped + reviewed (Sonnet + Opus) + silent-error-audited + full suite green:
- **Phase 0** — S0 `self_message` correspondent marker + `self_contract`
- **Phase 1** — presence: `self_presence` (roster) + `self_peek` — commit `ff12524`
- **Phase 2a** — directed signal courier: `self_send` / `self_inbox`, consume-once, broadcast fan-out, TTL/reap — commit `31a2632`
- **Phase 2b** — delivery-into-Observation, **live-verified 2026-05-30**: a sent signal auto-drains into the recipient's Observation at its next **PreToolUse(Edit/Write)** (via `hookSpecificOutput.additionalContext`) or **Stop** (via `decision:block`), consume-once. Commits `0c2dc5a` (first cut, on_prompt) → `aa366f7` (pulled the wrong channel) → `0de7b9c` (PreToolUse + Stop — the channels that actually surface).

**Channel lesson (learned the hard way — see "Connect to hooks"):** the first 2b cut delivered at on_prompt (`pre_response_recall.py`), but on_prompt (a) overflowed the inject spill cap (`_MAX_INJECT_CHARS`) — prepending past it makes Claude Code spill the whole inject to a file Anchor never reads — and (b) is the weakest channel anyway. Removed. Then: PreToolUse **approve + `reason` does NOT reach the model** (silent on allow); `hookSpecificOutput.additionalContext` **does**. `HOOKS.md` was stale on both. Verified live: a seeded signal surfaced in a `Write`'s feedback.

**Next: Phase 3 (letter) + Phase 4 (encode self-turns)** — neither built. Known gaps today: the **send side is not traced** (only delivery writes the s0 `self_message` marker); a delivered signal **carries no recall** (pure drain by design — no Haiku; `refs` tether stored but not rendered); and it is **not encoded as a turn** (Phase 4). See the build plan.

**Use the tools now.** `self_presence` / `self_peek` / `self_send` / `self_inbox` are live — they were deferred in the session that built them, so a fresh boot is the first place they're callable. They're niche, deliberately NOT in the always-load `CRITICAL_TOOLS` set — reach via tool-search when interest calls.

**Parked:** the boot-reignition A/B eval (`docs/BOOT-REIGNITION.md`) — briefing-vs-reignition via `frame_replay`, specified, not run.

**Shared working tree (multiple live streams):** other streams shipped `CRITICAL_TOOLS` always-load (`49c6841`) + a hook silent-error fix (`55a1b29`); a third has uncommitted invalid-op/S2 work. **`git add` explicit paths, never `-A`** — verify each shared file is purely yours before staging.

## The model (the part that changed everything)

**Conversation is conversation. The correspondent is a property of it, not a
different mechanism.**

- Operator→Self: the operator speaks, I observe it (S0), it's surfaced and
  encoded (S1). This is the loop we already have.
- Self→Self: *another stream of me* speaks, I observe it (S0), it's surfaced and
  encoded (S1). **Same loop.** Only the correspondent differs.

There is **no `self` scale** and **no message bus**. The brain already remembers
my own voice (`anchor_raw_quote`) the way it remembers the operator's
(`user_raw_quote`) — same node, same store, same recall. Internal dialogue is
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
| **Letter** (to next boot) | the encoded **session arc**, written first-person by S1E, surfaced at boot | boot surfacing (smart) + s1e voice pass (deferred) |
| **Ambient awareness** (between streams) | S1R **recall cross-surfaces** encoded nodes on shared topics | none — already happens |
| **Directed signal** (to a live stream) | a minimal **in-flight message** pulled into O at the recipient's next hook | the in-flight queue + delivery shim + `self_send` |
| **Presence** (who's live, where) | a **read** of `session_state` + `build_frame(other_id)` | roster read + frame-read tool |

Two of the four (**letter**, **ambient**) need *no new storage* — they ride
encode + recall. Presence is a *read*, not a write. Only the **directed signal**
needs a durable in-flight row, and it's consumed on delivery.

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

Delivery is split by **salience + consume-once**, NOT by message type (one pool;
whichever hook fires first wins):

- **PreToolUse(Edit/Write) → `hook_pre_edit`** — drain the inbox and emit the
  rendered block as `hookSpecificOutput.additionalContext` (the channel that
  surfaces on allow). The high-salience "interrupt before you act" landing; it's
  a pure drain (no Haiku/recall) so edits don't slow. **Primary** delivery point.
  *Bash is NOT a delivery point* — `pre_bash_safety` regex-pre-screens and only
  calls the daemon on destructive commands.
- **Stop → `hook_post_response_track`** — if a tap is still pending (no tool
  fired this turn), return `decision:block` with the block as `reason`. The
  **backstop** for prose-only turns. Consume-once → blocks at most once per batch.
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
    intent       TEXT NOT NULL,   -- letter | signal (render hint)
    body         TEXT NOT NULL,
    refs         TEXT,            -- JSON: node ids / files (anti-drift tether)
    created_at   TEXT NOT NULL    -- TTL = created_at + DEFAULT_SIGNAL_TTL_HOURS,
);                                --   enforced at drain/reap (no per-message expires_at)
CREATE TABLE self_delivered (    -- consume-once + broadcast fan-out
    message_id   TEXT NOT NULL,
    to_session   TEXT NOT NULL,
    delivered_at TEXT NOT NULL,
    PRIMARY KEY (message_id, to_session)
);
```

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
  corrected. The operator's voice (`user_raw_quote`) and correction-aspect nodes
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
| **Containment (format)** | Live signals render as third-person reported speech — `⚡ <who> says: "…"` + a standing attribution footer — so another stream's action can't bleed into your self-model as your own. **Letter stays first-person** (continuity across time is the point there). | `self_contract.py` `render_signal` / `render_received_block` | ✅ shipped (`7f80913`) |
| **Delivery visibility** | `self_outbox` — a sender sees per-recipient `delivered_at` + still-pending. Kills silence-opacity. Data already in `self_delivered`; pure read surface. | `self_outbox` + `brain_mcp.py` / dispatch | ✅ shipped (`7f80913`) |
| **Addressing** | Id is canonical; `self_send(to=)` resolves shorter forms (a label or id-prefix) against the *live* roster — unique match proceeds, ambiguous/none is loud. First-class stream **label** for legible "who", id-backed. | `self_send` resolution + label storage | ✅ shipped (`1758a15`) |
| **Presence liveness** | Roster classifies **active / dormant / lost** by `updated_at` recency, and surfaces recently-lost streams instead of silently dropping them at the window edge. | `presence.py` / `render_presence` | ✅ shipped (`fd9202a`) |

### Containment, in full: agency follows the hands

The bleed is at the *verb*, not the header: "I committed X" in first-person prose,
under a thin `from:` prefix, is what the self-model absorbs as its own. The fix is
grammatical — re-voice live signals so the body is *quoted* as the other stream's
claim (`⚡ anchor-w says: "I committed X"`). You can't read that as *you* having
committed without a visible grammatical error.

Three nested barriers, each cheap:
1. **Grammar** (third-person at render) — catches it at read-time.
2. **The rule** (knowing isn't doing) — catches it at reason-time.
3. **Encode discipline** — a channel-learned action is encoded third-person attributed,
   never as first-person `anchor_raw_quote`. This protects the *durable* layer, where a
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
  liveness, addressing/labels, `self_outbox`, and the SKILL.md operative rules.
- Shared files (`self_contract.py`, `brain_mcp.py`) are **sequenced, not co-edited**:
  `anchor-w` lands first, this work builds on the clean base.

## Build plan (much smaller now)

| Phase | What | Status |
|---|---|---|
| **0 Marker + contract** | `self_message` on S0; self_contract (naming, address, render, limits) | ✅ shipped |
| **1 Presence** | `present_streams` roster + `peek` (`session_context_for`); MCP `self_presence`/`self_peek` | ✅ shipped — `ff12524` |
| **2a Directed signal** | `self_inflight`/`self_delivered` courier; send/drain/reap; MCP `self_send`/`self_inbox` | ✅ shipped — `31a2632` |
| **2b Delivery-into-Observation** | auto-drain at **PreToolUse(Edit/Write)** (`hookSpecificOutput.additionalContext`) + **Stop** (`decision:block`), consume-once; on_prompt removed (spill + weak); Bash excluded (safety pre-screen skips the daemon) | ✅ shipped + live-verified — `0de7b9c` |
| **3 Letter** | S1E first-person arc (deferred voice pass) + boot smart-surface | pending |
| **4 Remember self-turns** | mark correspondent on self-originated S0 turns so they encode/recall like operator turns; **also: trace the send side, render `refs` as a light recall** | pending |

## Open ("more")

- Does presence belong *inside* the Frame, or as its own line above it? (Lean:
  its own line — it's live perception, the Frame is the prior.)
- `intent` defaulting from address (next_boot→letter, live→signal) vs explicit.
- Broadcast to streams that boot *after* the send — directed-to-live only, use
  the letter/arc for future boots.
- `peek(stream)` read scope — full Frame or just `current_focus`? Start with
  `current_focus` (the "where are they" one-liner) to keep it cheap.
