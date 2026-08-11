# Thalamus — design notes

**Status: v1 designed, reviewed, and rejected. The problem is real; the shape was wrong.**
This doc is evidence for a v2, not a plan to build. Read the fatal findings before
proposing anything.

## The problem

Encoders raise things only a decision-maker can settle, into a store whose only
reader is the next run of the same encoder. Two live examples, both correct, both
unread for weeks:

- consolidation asked, three runs running, whether the decoder's suppression-verb
  list is meant to be operator-configurable. It was a real inconsistency —
  the contract hardcoded four verbs while `_has_correction_edge` read the
  aspect taxonomy, so pairs joined by `resolves` re-proposed forever. Answered
  2026-08-11: the suppression set now derives from the `settlement` aspect
  (aspects_v1.json). The point stands: the encoder diagnosed it correctly,
  and the ask sat unread for three runs.
- consolidation reported, five runs running, that adding one node to a 7-node arc
  costs ~15 pairwise `similar_to` edges, and that it recurs for every new node.

The existing exit — `open ×5` → a `journals-escalation` node → boot injection —
fires on **persistence, not need**, and lands in a passive channel. It produced 11
stale escalations that saturated all 10 boot slots for weeks while these two
never arrived.

## What v1 proposed

A `(source, subject)`-keyed queue; producers file items; delivery rides the
self-channel's Stop-block; an answer writes back as a journal note; a `!` line
prefix marks a note for raising; `push once, re-raise is the escalation`.

## Why it was rejected — two fatal findings

Both measured against 2,381 journal notes (2026-06-24 → 2026-08-10).

**1. `subject` is not an identity.** It is free text an LLM writes per run.

| | |
|---|---|
| Distinct subjects containing "suppress" — one producer, one concern | **91** |
| Subjects that are `cluster-*` or `proposal_*` (run-local indices) | **850 / 2,381** |
| Subjects that are a bare tag word (`surprise`, `open`, `friction`…) | **164** |

So the key floods (91 items for one issue) *and* silently drops (a producer's
second `!surprise` collides with its first). The claim that "re-raising costs the
encoder a deliberate decision" is false: consolidation mints a fresh subject every
run automatically — its convention is `cluster-N/<paraphrase>`. It even wrote
`escalation-mesh/suppression-irony-v2` unprompted.

**2. No viable volume setting, and nothing in the design controls it.**

- **Floor:** 25 of 2,381 notes contain a `?`; **5** are from S2. ~0.1/day — the
  channel is empty and the acceptance test exercises nothing.
- **Ceiling:** if encoders map `!` onto their existing `open` behaviour: 468
  notes, **10.6/day**, roughly doubling once Healer and Aspect get journals.
  Against a channel carrying **2.6 deliveries/day** today, with
  `RECEIVED_BLOCK_MAX = 4000` — about six items before `(+N more)`, the same
  overflow pointer that trained the reader to skip the boot region.

**And the premise is weaker than claimed.** `ab520c77` records a self-message that
*was* delivered and consumed and was still missed. The Stop block buys one more
turn, not a decision. Predicted failure at six weeks is not silence but **reflex
deferral**: every item is an architecture question arriving mid-thread, the honest
answer is "ask the operator", `defer` is pre-filled and clears the block, and
`defer` becomes the trained response to the glyph.

## The direction for v2: a triager, not a key

Operator's call, and it dissolves both fatal findings: **an agent decides what to
surface and whether to push it at all.**

Code cannot tell that `cluster-1/already-suppressed` and
`cluster-2/suppress-mesh-cost` are the same question — that is judgment. So:

- **extraction stays dumb** (a marker, no parsing intelligence — this part of v1
  was right)
- **triage is the LLM's job**: collapse variants into one concern, decide whether
  it is worth attention at all, assemble the context needed to answer, and drop
  the internal vocabulary
- the queue then holds *triaged concerns*, not raw notes, and volume is gated by
  the triager rather than by encoder judgment

The jargon point is not cosmetic. The first real ask relayed to the operator was
unanswerable because it was full of names like `correction_improvement` with no
explanation — the same failure the channel exists to fix, one level up. Whatever
triages must own plain language.

## Ground truth (verified against live code 2026-08-10)

- **Stop-block is the sole *push* path.** `daemon_hooks.py:754`; the PreToolUse leg
  was removed 2026-06-04 (`364269f`) because it was missed — *"consumed the tap
  into context the model didn't act on."* Passive injection is the known-failed
  design. `self_inbox` remains a manual consume-once *pull*.
  Two stale comments still describe the removed leg and a function that no longer
  exists: `daemon_hooks.py:530` and `signal.py:266`. Worth a cleanup.
- **Consume-once already gives push-once** — the drain blocks at most once per
  batch; the next Stop finds nothing.
- **`self_outbox` exposes pickup state** — but only inside TTL (1h broadcast / 24h
  directed). Past expiry the receipt is *deleted*, so "delivered and ignored"
  becomes indistinguishable from "never delivered". Any re-push logic must record
  its own attempt rather than trust the courier's memory.
- **`store_pending_message` / `drain_pending_messages` are gone** — a
  brain_meta-backed queue (cap 5, oldest dropped) whose readers died when hook
  logic moved into the daemon. Removed 2026-08-11; the `pending_hook_messages`
  key still holds two never-drained messages, which is what a queue outliving
  its reader looks like from the outside.
- **`brain_debug` → `debug_log` is NOT a dead channel** — it is forensic, and it
  has a reader: the `query_logs` MCP tool (`source='debug'`) returns these rows.
  `debug_enabled=1` in `brain_meta`, so ~15 call sites write continuously. The
  dashboard's Logs tab filters `event_type IN ('error','warning')` and so misses
  them — which is how a reader-less verdict is easy to reach here. Only the
  docstring was wrong (it claimed the recall hook drained them into
  `additionalContext`); that drain is genuinely gone, and the docstring is fixed.
  **Do not fold this into a v2 port.**
- **`bridge_proposals` is dead and no longer declared** — it served deferred
  maturation (propose now, mature at `matures_at`), and its readers
  (`_mature_bridge_proposals`, `_bridge_at_consolidation`) went with
  `consolidate()`. Bridging is now immediate: `_bridge_at_store_time` writes
  `emergent_bridge` edges directly. Removed from `TABLES` + `INDEXES`
  2026-08-11, so fresh brains skip it; existing brains keep an empty table
  until the dead-table drop ships with the migration runner (0 rows, so nothing
  is pending in it).
- **The return path needs a door that does not exist.** `write_journal_notes`
  takes an encoder's raw text, requires a `## Review` fence, and has no `unit`
  parameter — S2 scoping comes from `chain_id LIKE '%-{unit}'`. A single-note
  write door belongs on `brain_traces`; string-assembly at the call site does not.
  And a fabricated chain **costs a continuity slot**: `JOURNAL_CONTINUITY_RUNS` is
  **3** for consolidation and community at ~3.3 runs/day, so an answer is visible
  for roughly a day, and three answers would evict all real residue.
- **`!` corrupts the tag.** Verified: `'! friction · s · n'` parses to
  `tag='! friction'`. `-*•` are stripped; `!` is not. So `!open` misses
  `JOURNAL_OPEN_TAGS` and `!resolved` becomes a silent no-op — the marker breaks
  exactly the notes it marks. `!` is genuinely unused today (0 / 2,381).
- **Every S2 unit has a journal binding** (`s2/base.py:194`, scoped `(SCALE,
  NAME)`); only Consolidation and Community exercise it. Unit names are
  `consolidation`, `community_detection`, `aspect_integration`, `healer`; S1's is
  `scribe` and it receives the same review block.
- **`O_SOURCES` / `K_SOURCES` have zero runtime consumers** and already carry a
  stale entry (`scribe.py:42`). Class metadata drifts like anything else — what
  prevents drift is a test, not a location. A `PURPOSE` field would need one home
  per producer and an enforcement test, not eight declaration sites.

## Constraints v2 must satisfy

1. **Identity must not be encoder free text.** Key on something structural, or on
   a triager's judgment — never on `subject` alone.
2. **Volume must be controlled by something we own**, not by encoder discretion.
3. **The msgs boundary is not free.** `signal.send` writes a session id
   (`from_session`), first-contact enrichment calls `presence.peek` and degrades
   to garbage for a non-session sender, and `render_signal` is a *locked
   containment contract* rendering every message as `other stream (id:X) says:`.
   A machine producer needs its own render path or an explicit second shape.
4. **Items are not nodes, so they get no scope veil.** `render_standing_items`
   threads `session_id` into `filter_nodes` precisely so a walled escalation
   cannot print into another session's boot. An items table has no provenance and
   an S2 producer has no session to stamp from. Unresolved.
5. **Delivery needs a trace.** `REF_TYPES[('s0','K')]` owns the vocabulary and is
   locked by `test_trace_contract_sync`. Untraced delivery *is* the visibility
   problem.
6. **Discharge the prior ruling.** `docs/LATERAL-SCALES.md` deprioritized the
   `operator` lateral — *"the burden is on it to prove it's not just async S0.
   Tom's call: not now."* That lateral is Anchor→operator and this is
   encoder→Anchor, so it is not the same system, but the burden applies and v1
   never addressed it. `docs/CROSS-STREAM-ON-SCALES.md` also draws the opposite
   boundary — *"the messaging system's job ends at delivery"* — and the two specs
   do not reference each other.
7. **Name honestly.** The anatomical thalamus relays *and* gates on salience. A
   triager-gated design earns the name; a bare relay does not.

## What survives from v1

The diagnosis (boot is a bare `filter_nodes` with zero delivery state); Stop-block
as the only push channel; retiring `journals-escalation` and the `open ×5`
promotion;
shipping deletions in the same commit as their port; and writing the answer back
by *code* rather than by an LLM, which removes the resolve-verb failure tail at
the one place it matters.
