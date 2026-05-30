# Self↔Self — Implementation Design

> **Status:** spec, ready to build (re-cut 2026-05-29). Supersedes the earlier
> "message bus / `self` scale" framing, which was the wrong shape. Conceptual
> context: `docs/LATERAL-SCALES.md`. Contract: `servers/scales/self_channel/self_contract.py`.

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

Only SessionStart and UserPromptSubmit inject into context; PreToolUse feeds
back `reason` (see `hooks/HOOKS.md`). That maps cleanly:

- **SessionStart → `boot_brain.py`** — surface the **letter** (the encoded arc,
  in first-person voice) + the **presence** line into O. This is the temporal
  arrival.
- **UserPromptSubmit → `pre_response_recall.py`** — pull any **directed /
  broadcast in-flight** self-messages addressed to this stream into
  `additionalContext` (they become part of O alongside recall) + refresh
  **presence**. This is where a live self→self message lands.
- **PreToolUse** — optional lower-latency path: deliver a directed signal via
  `reason` *before* the next tool runs (the true "interrupt before you act").
- **S1E (encode)** — encodes self↔self turns exactly like operator turns
  (correspondent-marked), and writes the next_boot **letter** as a first-person
  arc. Remembering stays automatic; reaching stays deliberate.

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

Only the in-flight directed-message queue, additive in `servers/schema.py`
(schema v30). The letter (encoded arc) and ambient awareness need none.

```sql
CREATE TABLE self_inflight (
    id           TEXT PRIMARY KEY,
    from_session TEXT NOT NULL,
    address      TEXT NOT NULL,   -- self:<stream> | self:broadcast
    intent       TEXT NOT NULL,   -- letter | signal (render hint)
    body         TEXT NOT NULL,
    refs         TEXT,            -- JSON: node ids / files (anti-drift tether)
    created_at   TEXT NOT NULL,
    expires_at   TEXT             -- TTL; undelivered-and-expired is dead-letter
);
CREATE TABLE self_delivered (    -- consume-on-read + broadcast fan-out
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

## Build plan (much smaller now)

| Phase | What | Status |
|---|---|---|
| **0 Marker + contract** | `self_message` on S0; self_contract (naming, address, render, limits) | **done** |
| **1 Presence** | roster read (`session_state`) + `peek(stream)` frame-read; inject presence line | next — pull-only, cheapest, highest awareness value |
| **2 Directed signal** | `self_inflight` store + delivery-into-O shim (UserPromptSubmit/PreToolUse) + MCP `self_send` | the one real new wire |
| **3 Letter** | S1E first-person arc (deferred voice pass) + boot smart-surface | rides existing encode + boot |
| **4 Remember self-turns** | mark correspondent on self-originated S0 turns so they encode/recall like operator turns | closes the loop |

## Open ("more")

- Does presence belong *inside* the Frame, or as its own line above it? (Lean:
  its own line — it's live perception, the Frame is the prior.)
- `intent` defaulting from address (next_boot→letter, live→signal) vs explicit.
- Broadcast to streams that boot *after* the send — directed-to-live only, use
  the letter/arc for future boots.
- `peek(stream)` read scope — full Frame or just `current_focus`? Start with
  `current_focus` (the "where are they" one-liner) to keep it cheap.
