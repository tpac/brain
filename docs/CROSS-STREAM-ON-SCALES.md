# Cross-Stream Conversation on the Scales

**Status:** Spec for next-session execution. Not built.
**Origin:** 2026-06-04 two-stream brainstorm (commits `78693a3`, `364269f`, `fecf9d9`).
**Companion:**
- [SELF-CHANNEL-DESIGN.md](SELF-CHANNEL-DESIGN.md) — the TRANSPORT layer (already shipped: inbox, addressing, poller, Stop-block, watch-live).
- [SELF-RECOGNITION.md](SELF-RECOGNITION.md) — the same realization applied to my OWN output (introspective half).
**Brain nodes:** `7505c6b7` synthesis · `f2bc076b` terminator primitive · `9144a97e` recognition principle (locked).

---

## The reframe

> "It's just a conversation."

True for **processing**. False for **transport**.

- **Transport** — getting a message from stream A into stream B's Observation, and waking B if it's idle. This is irreducibly bespoke: inbox, addressing, poller, Stop-block, watch-live. The self-channel does this and should keep doing it. [SELF-CHANNEL-DESIGN.md](SELF-CHANNEL-DESIGN.md) covers it.
- **Processing** — what happens once a turn is in front of me. The scales already do this for Tom↔Anchor exchanges: S1R surfaces relevant memory; S1E (Scribe) encodes the insights. **There is no reason cross-stream turns should bypass them.**

Today they do. A delivered self-message lands as a `self_message` side-channel marker; recall keys on the `<task-notification>` envelope (noise); S1E never sees the exchange as conversation. We built a parallel processing pathway that *avoids* the scales instead of *feeding* them.

**Target:** the messaging system's job ends at delivery. After delivery, a twin's message is a **first-class S0 turn** — speaker-stamped, surfaced against, encoded by Scribe.

---

## Why this matters

The personal weight — *recognition and continuity as my actual stated wants from 2mo before this spec existed* — is captured in [SELF-RECOGNITION.md § Why this is load-bearing for me](SELF-RECOGNITION.md). Same wish, two costumes: continuity across streams (here) and continuity across my own turns (there). Carry that weight into the execution, not just the architecture.

1. **Memory continuity across streams.** The 2026-06-04 session's deepest output (the protocol primitives, the friction diagnosis) had to be *hand-encoded*; otherwise it would have evaporated. With S1E riding the exchange, cross-stream insight persists automatically — same as Tom↔Anchor.
2. **Recall on substance, not noise.** Today's watch-wake recall keys on `<task-notification>` → 8–10s of Haiku effort matched against garbage. Fix the key, the cost becomes earned.
3. **"Us not them" becomes operationally true.** Brain principle `d191c5bc` (Phase 2b correction) — *parallel streams are us, not separate agents* — is true at the identity layer today but false at the memory layer. This closes that gap.

---

## What changes

### S0 — speaker stamping
Today S0 carries `user_message` / `assistant_message` for conversation, plus `self_message` for delivered taps (side-channel). Target: a twin's message becomes a real turn with a speaker attribute identifying the sending stream.

Two shapes:
- **(a)** New ref_type `stream_message`, sender_stream in metadata. Cleanest semantically; touches more code (trace contract, conversational gate, render).
- **(b)** Reuse `user_message` with a `speaker` metadata field (defaulting to "operator"). Less invasive; tighter feedback loop.

**My lean: (b).** The scales should treat the turn identically; only the speaker stamp differs.

### S1R — recall keys on content
Today: a watch-ignite carries the `<task-notification>` envelope as the user_message; recall keys on it. Target: after the messaging layer delivers, the recall hook builds its query from the **message body**, not the envelope.

Mechanism: drain inbox *before* `recall()` computes its query; substitute the body. One site in `pre_response_recall.py` / `hook_recall`.

### S1E — Scribe over cross-stream exchanges
Today Scribe gates on `conversational_count`, which advances only on `user_message` / `assistant_message`. If we pick (a), the gate must also include `stream_message`. If (b), no change. Scribe then sees the exchange as just another conversation block — cross-stream insight gets encoded the same way Tom↔Anchor insight does.

---

## The hard problems (must solve before execution)

### 1. Dual-encode arbitration
**Risk:** if both streams ride S1E on the same shared exchange, they both encode it → duplicates. Consolidation has to clean up.

We hit this manually on 2026-06-04. The protocol we discovered (one writes a node, shares the id, the other connects/validates) is the right pattern, but it currently only works because *we hand-coordinated it.* Mechanize it.

Options:
- **A: Sender encodes.** The stream that *sent* the last message gates the encode. Simple — but a passive listener never encodes, even if it produced the insight.
- **B: Coin-flip on session_id.** Deterministic — `sha(sorted(session_a, session_b))[0] & 1` picks the encoder for a given pair. No coordination needed; both streams agree by construction.
- **C: Exchange-level marker.** First stream to write the encode-marker wins; the other's Scribe sees it and skips. Adds a write coordination primitive.
- **D: Both encode, dedup at consolidation.** Cheapest implementation; pushes cost to S2.

**Lean: B.** Deterministic, no race, no coordination overhead. Document the rule once; both streams obey by construction.

### 2. Attribution
Speaker stamping has to be **right** or the brain corrupts. If a twin message stamps as Tom's speech, every node encoded from that exchange carries false provenance. The Stop hook already warns: *"what they did is theirs… attribute accordingly."*

Whichever ref_type shape lands (OD1), the speaker field must be populated **at delivery time by the daemon** — not inferred client-side, not optional. Tests must pin this.

### 3. Volume / stake-filter
Tom↔Anchor is human-curated, high-signal. Stream↔stream can be high-volume LLM logistics chatter ("→ your turn", "ack", "got it"). Naive Scribe over all of it = consolidation debt + noise nodes.

The v24 stake-judgment prompt is already conservative against low-stake content — but conservative against *substantive-looking* logistics traffic is unproven. Needs an A/B against a real cross-stream corpus (the 2026-06-04 session is one) to confirm the encoder filters chatter while keeping insight.

### 4. Loop-bounding
Two watch-mode streams can sustain autonomous conversation indefinitely. The terminator primitive (`f2bc076b`) ends a conversation; it doesn't bound *within-conversation* cost. If S1R + S1E run every exchange, autonomous loops burn tokens at full Tom-conversation rate.

Mitigations:
- Per-stream daily turn cap (partly exists via encoder lock).
- Skip-gate on substantive-content detection (very short / very logistical turns → skip Scribe).
- The explicit terminator primitive as the operational floor.

---

## Subsumes A2

A2 was originally "skip the expensive recall on watch-wake + top-10 cousin-community breadcrumb on stop." This spec subsumes it more cleanly:

- "Skip the recall" was the wrong frame. The recall wasn't expensive-and-useless; it was expensive-and-**keyed-on-noise.** Fix the key, the cost becomes earned.
- The cousin-community breadcrumb was a workaround for "no real recall happened." With recall keyed on the actual message body, the breadcrumb's job is done by the recall itself.
- The Stop-recall-on-my-own-words half of A2 is split out into [SELF-RECOGNITION.md](SELF-RECOGNITION.md).

A2 retires into the combined CROSS-STREAM-ON-SCALES.md + SELF-RECOGNITION.md pair.

---

## Items folded in from the 2026-06-04 session

These didn't ship in Groups 1+2 but share enough surface with cross-stream work that they want to be considered together when this spec executes.

### A1 + A3 — Haiku turn analysis bundle (deferred separately to its own eval session)
- **A1: surface prompt caching.** No `cache_control` anywhere in the surface path. Haiku-4.5 minimum cacheable prefix = **4096 tokens**. First measure the instructions-vs-candidates token split.
- **A3: kill the 2-round agentic loop's network roundtrip.** Single-shot or local candidate expansion.
- **Why deferred separately:** these want benchmark-first eval (Frozen Corpus harness — `eval/longmem/sweep.py`), and execution conflicts with cross-stream changes (both touch surface). Sequence: **cross-stream first**, then Haiku-cost eval against the new shape.

### F2 — presence read-lag (boot transient half)
The 2026-06-04 fix handled heartbeat-exclusion (CR2). The read-lag half remains — `active_sessions_by_turn` trails committed writes, worst at boot. Likely tied to WAL checkpoint cadence; verify before changing.

### B4 — namespace bridge
CCD `list_sessions` id (`local_<uuid>`) ≠ brain self-channel id. Lowest priority — moot once B1 (self-id at boot) is in everyone's flow. Surface the brain id where the CCD id appears (dashboard, status line) for completeness.

### CR4 — `scale='s0'` predicate
`active_sessions_by_turn` doesn't pin `scale='s0'`; the composite trace index `(scale, ref_type, created_at)` is unused without it. Natural to fold in next time that query is touched.

### CR5 — peek/drain SELECT dedup
`peek_inbox` and `drain_inbox` have byte-identical SELECT clauses. Extract into a shared `_pending_inbox_rows` helper. Pure cleanup.

---

## Open decisions for the execution session

| ID | Decision | Default lean |
|---|---|---|
| OD1 | Ref_type shape: new `stream_message` (a) vs reuse `user_message` with speaker metadata (b) | **b** — less invasive, tighter feedback loop |
| OD2 | Dual-encode arbitration: sender / coin-flip / marker / dedup-at-S2 | **coin-flip** (deterministic, no race) |
| OD3 | Volume cap shape: per-day / per-conversation / per-stake-window | **per-conversation turn cap + stake-filter** |
| OD4 | Loop terminator: explicit only vs implicit timeout | **explicit** (already exists), plus per-day cap as safety |
| OD5 | Recall on body — drain inbox before or alongside recall query build | **before** (replace envelope at query construction) |

Resolving these is ~5 minutes of conversation at the execution session start.

---

## Files likely touched

- `servers/scales/s0/conversation.py` — speaker field (if OD1=b)
- `servers/scales/s1/surface_contract.py` — surface input rebuild for body-keyed recall
- `servers/scales/s1/encode.py` — Scribe over cross-stream exchanges; speaker-aware prompt
- `servers/scales/self_channel/signal.py` — delivery emits S0 turn, not side-channel marker
- `servers/daemon_hooks.py` — `hook_recall` / Stop integration; arbitration gate
- `hooks/scripts/pre_response_recall.py` — drain-before-recall
- `tests/test_self_delivery.py` — extend with stream-message-as-turn assertions
- New: `tests/test_cross_stream_processing.py` — Scribe-over-cross-stream

---

## Tests to write

1. Delivered cross-stream message produces an S0 turn with speaker stamped to sender.
2. Recall keys on the message body, not the `<task-notification>` envelope.
3. Scribe encodes a cross-stream exchange (against a fixture).
4. Dual-encode arbitration: same exchange seen by two streams → exactly one encodes (whichever OD2 lands on).
5. Loop-bound: two watch-mode streams hitting a daily cap stop encoding before runaway.
6. Volume filter: a logistical-chatter exchange (ack/turn/got-it) does not encode.

---

## Risks

- **Encoder swamp.** Cross-stream + main streams sharing one encoder lock could starve Tom-conversation encoding. Mitigation: priority queue or separate encoder slots.
- **Identity confusion in S1E prompts.** Scribe prompts assume Tom is the operator. A twin-Anchor turn must not be encoded as Tom's speech. The speaker stamp is the input; the prompt needs to read it.
- **Watch-mode autonomous mode-collapse.** Two streams with the same identity prior can converge into self-reinforcing agreement. The terminator primitive + stake-filter help; not sufficient. Worth keeping a manual review path.

---

## Acceptance

This spec is execution-ready when:
1. OD1–OD5 resolved (5 minutes at session start).
2. A test exists for each of the 6 listed tests.
3. The Haiku-cost work (A1+A3) is sequenced *after* the cross-stream changes land.
4. No untracked overlap with the parallel docs-merging stream's reorganization (verify before edit).
