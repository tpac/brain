# Idleness Clock — Architecture Plan

**Scope.** Who owns the definition of "this session has gone quiet", and which clock measures it.
Boundary traced: `servers/brain.py` (`scribe_due`, `run_maintenance_if_due`), `servers/brain_traces.py`
(`present_streams`), `servers/dal_logs.py` (`active_sessions_by_turn`, `_attended_sql`,
`conversational_turns_since`), `servers/activity_state.py`, `servers/clock.py`,
`servers/trace_contract.py`, `servers/scales/s1/encode_contract.py`, and the presence consumers
(`servers/channels/self_channel/presence.py`, `signal.py`).

**The finding, in one sentence.** `present_streams` publishes the *reachability* clock under a neutral
name (`updated_at`), and `scribe_due` reads it as the *quiet* clock — so the gate divides a count over
one row set (`turns` = non-envelope `user_message`) by a clock over a different row set (`last_turn` =
a MAX that includes `heartbeat`).

**What the review deliberately does NOT recommend.** `active_sessions_by_turn` is cohesive — do not
split it. `scribe_due` is not a god-function — do not decompose it. The `encode_contract` constants are
one coherent ripeness policy — leave them. No per-session idleness service, no ref-type-set parameter:
both were evaluated and rejected (see Step 2, *Alternatives rejected*).

**Coverage caveats.** All mechanical claims were read from the cited files. The *magnitude* of the
live bug is NOT measured — that is Step 0, and it gates everything after it. Five angle agents
(placement, unification, cohesion, coupling, altitude) plus a brain history pass; the agents disagreed
on one point (which replacement signal) and that disagreement is resolved in Step 2 with the argument
that settled it.

## Dependency summary

```
Step 0 (measure)  ──gates──>  Step 2 (the fix)  ──>  Step 3 (clock helper)
Step 1 (name the set)  — independent, may land any time, do NOT bundle with Step 2
Step 4 (rename key)    — independent, cosmetic-but-clarifying
```

**Steps 1 and 4 LANDED** (`PRESENCE_LIVE_REF_TYPES` in `trace_contract.py`, both DAL sites importing
it; projection key renamed, `presence.py` keeps `updated_at` as its own PUBLIC output key so the MCP
surface is unchanged). Steps 2 and 3 are stood down by the Step 0 result — Step 3 (`clock.py`) was
handed to the stream owning `daemon_server.py`, whose own census of wall-clock interval comparisons
justifies it independently of this plan.

Step 1 and Step 4 are independent of everything and of each other. Step 2 must not start before
Step 0 returns a positive result. Step 3 edits a line Step 2 also edits — sequence it after.

---

## STEP 0 RESULT (2026-09-14) — MEASURED LATENT. Steps 2 and 3 are NOT authorized on a live-bug basis.

21-day snapshot, 4554 s0 rows. 17 sessions had any tail after their last real turn (most delays
0.0-0.2h); 2 passed the 1h threshold with an apparent backlog; **both collapse** — `8bec823e` had
`turns_since_last_encode` = 1 and a 45.3h "delay" caused by ONE `<task-notification>` row, not
heartbeats; `10bb2f82` had 2. The tail needs > 2. **Blocked and never encoded: 0.**

Why it stays latent: the wake envelope that keeps a row out of `conversational_turns_since` is the
same thing that makes a session heartbeat-driven. A session gaining countable turns is one where the
operator is present, and operator presence advances *both* clocks. The row sets diverge mainly for
machine-driven sessions, which have `turns` ≈ 0 and are skipped at `if turns <= 0: continue` before
the clock is read. A pure `/watch` stream does not fail to encode because of the clock — it has
nothing to encode.

Also corrected: "the tail can never fire" (stated by two review agents) is wrong even structurally.
The tail is DELAYED by (last heartbeat − last real turn); once heartbeats stop it becomes eligible 1h
later, and 1h is inside the 5h candidacy window, so it still fires.

**Per this step's own decision rule: STOP.** Steps 2 and 3 are not justified by a live defect. The
row-set mismatch remains a real latent trap and Steps 1 and 4 stand on their own (naming + a
misleading key name), but nothing here is urgent. Reopen Step 2 only if a future change makes
machine-driven sessions accumulate countable turns — that is the premise this result rests on.

Reproduce: the measurement script pattern is in this step's *What to measure*; it needs only a
`snapshot_to` copy of `brain_logs.db` and read-only SQL, no Brain instance.

---

## Step 0 — Measure whether the tail is actually blocked in production

**Problem.** The whole plan rests on a structural argument. On 2026-09-14 a structurally identical
argument about the daemon idle timeout ("pings reset `last_activity`, so it can never fire") was
measured FALSE — `daemon.log` showed 90 firings, 87 of them at genuine 4h idleness (brain node
`9a3b6bed`). Heartbeats are wakeup-driven, not on a fixed timer, so the bug may only bite
frequently-woken streams. Do not write code before this returns.

**What to measure.** Per session over the last 14 days, from `trace_events` (scale `s0`):
1. The longest window in which every consecutive gap between `live_types` rows
   (`OPERATOR_DIALOGUE_REF_TYPES + ('heartbeat',)`) stayed under `SCRIBE_TAIL_IDLE_SECONDS` (3600)
   **while** the gap between consecutive *conversational* rows exceeded it. That window is the
   interval during which the tail should have been eligible and was not.
2. Cross-reference with `turns_since_last_encode > SCRIBE_TAIL_MIN_TURNS` (i.e. the session actually
   had a drainable backlog) and with the absence of an `encoding_run` trace for that session.
3. Count sessions where `stamp_boot_liveness`'s per-resume heartbeat (`servers/brain_voice.py:363`)
   alone pushed a matured tail past its threshold — this is expected to be far more common than the
   `/watch` wakeup path, because it fires on *every* boot and resume.

**How to run it.** `IsolatedBrain` from `tests/isolated_brain.py` (safe snapshot; never open a second
`Brain` writer against the live DB, never `cp` a live WAL database), then SQL over the copy via
`./dev python3`. Reading through `query_traces` is not sufficient — the MCP surface renders short
session ids and cannot do the per-session gap arithmetic.

**Decision rule.** If no session shows a sustained sub-hourly `live_types` chain covering a real
backlog, STOP and report: the finding is true but does not bite, and the remaining value is Step 1
and Step 4 (naming/hygiene) only. If sessions do show it, proceed to Step 2 and record the count.

**Cross-check instrument.** `daemon.log` is the cheap second source and has already out-performed
code-reading once today: the `9a3b6bed` correction came from a `grep -c` over it. Use it to corroborate
the trace-derived answer (encode dispatches per session), not to replace it — the per-session gap
arithmetic needs `trace_events`.

**Verification.** N/A — this step *is* the verification.
**Blast radius.** None (read-only).
**Depends on.** None.
**Respects.** Brain node `9a3b6bed` (measurement beats code-reading for behavioral claims); the repo
rule against a second live writer.

---

## Step 1 — Name the presence-liveness ref-type set in the contract  ✅ LANDED

**Problem.** `live_types = OPERATOR_DIALOGUE_REF_TYPES + ('heartbeat',)` is composed **inline, twice**,
inside SQL-building code (`servers/dal_logs.py:1418` and `:1493`). The parts route through the
contract; the *combination* — the locked 2026-06-04 "a watch listener is the most reachable stream"
decision — does not. Because the set has no name, `scribe_due` inherited it invisibly: there is no
symbol to grep and no import to notice at the consumer. This is the *mechanism* of the main finding,
not a cosmetic duplicate.

**Target state.** A named constant in `servers/trace_contract.py` beside `OPERATOR_DIALOGUE_REF_TYPES`
(~line 315), e.g. `PRESENCE_LIVE_REF_TYPES`, carrying the B2 rationale in its comment. Both DAL sites
import it. Zero behavior change.

**Files & call sites.** `servers/trace_contract.py` (+1 constant); `servers/dal_logs.py:1418`, `:1493`.
**Verification.** `tests/test_trace_contract_sync.py`, `tests/test_self_presence.py`. Behavior is
unchanged, so both must stay green without edits.
**Blast radius.** Two call sites, one new constant. Diff under 15 lines.
**Depends on.** None — independent. Do NOT bundle it with Step 2; it is hygiene and should be
separately revertable.
**Respects.** The locked B2 decision (brain node `083b745d`) — this *names* that decision, it does not
change it. Heartbeats continue to count toward presence liveness.

---

## Step 2 — Give `scribe_due` a conversational-recency clock, and bound the tail

**Problem.** `servers/brain.py:1154-1158` derives `idle` from `present_streams()['updated_at']`, which
is `last_turn` — a MAX over `live_types`, heartbeats included. Both firing clauses read it:
`five_plus` (`idle < SCRIBE_ACTIVE_WINDOW_SECONDS`) and `tail` (`idle > SCRIBE_TAIL_IDLE_SECONDS`).
A heartbeat therefore (a) resets the tail clock, and (b) for 10 minutes makes a quiet session look
actively conversing. `stamp_boot_liveness` (`servers/brain.py:1188`, called from
`servers/brain_voice.py:363` on **every** boot/resume) writes one, so resuming a session holding 3
unencoded turns pushes its matured tail out another hour, per resume. A `/watch` listener writes one
per wake re-arm (`servers/daemon_hooks.py:631`), and for as long as it keeps waking, `tail` is
unsatisfiable — a watch session holding 3-4 unencoded turns (below `ENCODE_EVERY`) has no other drain
path.

**Target state.** `scribe_due` measures idle from the **last non-envelope conversational turn, either
side** — the row set `{turns that count} ∪ {their replies}`. `now=` stays wall-clock (unchanged
contract; a sibling stream depends on this — see *Respects*). Add an explicit upper bound on the tail.

Two acceptable plumbings; **prefer (e′)**:

- **(e′) session-keyed, preferred.** New `dal_logs.last_attended_turn_at(session_id)` reusing the
  existing `_attended_sql` predicate; a thin `brain_traces.py` wrapper (the trace-read guardrail
  `tests/test_traces_layer_guardrail.py:31` sets `EXCLUDE = {'brain_traces.py', 'dal_logs.py'}`, so a
  new trace read has exactly one legal home); `scribe_due` calls it per candidate. Purely additive —
  touches **zero** existing tests. Keyed by session id, so it does not depend on the candidate source
  and survives the parked SCAN/DRAIN redesign without re-plumbing.
- **(P) projection carry.** Carry the already-computed `conv_recency` (`servers/dal_logs.py:1449-1452`)
  through `present_streams`' projection (`servers/brain_traces.py:1053`) and read it in `scribe_due`.
  One file cheaper, but bets that `present_streams` remains the candidate source through that
  redesign, and edits `tests/test_self_presence.py:321-326` (which documents the 4-key boundary as
  intentional) plus the `tests/test_daemon_hooks.py` stub dicts.

**THE REGRESSION THIS STEP MUST CARRY.** Today `idle` is *accidentally* bounded: candidacy requires
`last_turn` within `SCRIBE_CANDIDATE_WINDOW_MIN` (5h) and `idle` is measured from that **same**
`last_turn`, so it can never exceed ~5h. Switching the clock removes that bound — the conversational
subquery has no time bound, so a stream heartbeating for three days carries a three-day-old clock and
would tail-drain a three-day-old conversation stamped "today". Bound the tail explicitly; reusing
`SCRIBE_CANDIDATE_WINDOW_MIN * 60` preserves today's envelope exactly. **A fix without this bound
trades one bug for a worse one.**

**The downstream consumer of the accidental bound.** `servers/daemon_server.py:1367` prunes
`_scribe_attempts` with `horizon = SCRIBE_CANDIDATE_WINDOW_MIN * 60` — the same constant the candidacy
gate passes to `present_streams` (`servers/brain.py:1127`) — and justifies it in a comment: "those
sessions have aged out of consideration anyway". `:1341-1342` then keys `_scribe_failures` to the
surviving `_scribe_attempts`, so pruning an attempt also drops that session's failure COUNT, which is
what `scribe_repeated_failure` escalates on at `SCRIBE_MAX_FAILED_RETRIES` (3).

That comment is true today only because candidacy and idle are measured from the same timestamp. Once
the tail has its own bound, "aged out of consideration" and "aged out of the candidacy window" are two
different statements, and the prune's justification no longer follows from the constant it uses.

**Give the bound its own name** (e.g. `SCRIBE_TAIL_MAX_IDLE_SECONDS` in `encode_contract.py`) and have
`daemon_server.py:1367` derive its horizon from *that*, not from `SCRIBE_CANDIDATE_WINDOW_MIN`. Set it
to `SCRIBE_CANDIDATE_WINDOW_MIN * 60` initially to preserve today's envelope exactly; the point is that
the coupling becomes explicit and survives either constant being retuned. The sibling stream owning
`daemon_server.py` has explicitly left this line to this step rather than touching it in their plan.

*Calibration, so the next session does not over-state this — the prune does NOT silence
`scribe_repeated_failure`, for two independent reasons:*

1. `SCRIBE_RETRY_COOLDOWN_SECONDS` is 120 (`encode_contract.py:110`), so a still-due failing session
   re-fires about every two minutes and reaches `SCRIBE_MAX_FAILED_RETRIES` (3) in roughly six — three
   orders of magnitude inside the 5h prune horizon.
2. `servers/daemon_server.py:1375` re-stamps `self._scribe_attempts[sid] = now` on **every** attempt,
   so the prune age is measured from the LAST attempt, not the first. An actively failing session
   refreshes its entry every cooldown and never ages toward the horizon at all.

The count is only lost for a session selected fewer than 3 times in 5h — which means it was being
starved of *selection*, a different problem with a different fix. Fix the naming because the coupling
is incidental and uncommented-as-such, **not** because the alarm is currently broken. (Both this claim
and its rebuttal were checked against the code by two streams independently; do not re-derive a
scarier version from the coupling finding alone.)

**Why this row set and not a stricter one.** `conv_recency`/`last_attended_turn_at` is correct not
because it is heartbeat-free but because its row set is provably the event set that can move `turns`,
plus the replies that close those turns: `conversational_turns_since` (`servers/dal_logs.py:1526`)
counts non-envelope `user_message` rows, and for a `user_message` row "attended" and "envelope-free"
are the *same* predicate (the row self-tests against its own latest preceding `user_message`).
Therefore the clock cannot be stale while the counter is moving.

**Alternatives rejected** (do not re-litigate):
- *A `user_message`-only MAX* (e.g. returning `MAX(created_at)` alongside the count from
  `conversational_turns_since`): looks tidier, but is the strict clock in disguise — it measures idle
  from the *prompt*, so a turn lasting longer than `SCRIBE_ACTIVE_WINDOW_SECONDS` flips `five_plus`
  off at the exact moment the exchange completes.
- *Two clocks, one per clause*: the clauses partition a single axis (`[0,600)` active / limbo /
  `(3600,∞)` abandoned) with the limbo band deliberate. Two clocks let a session land outside both
  bands on different axes and starve permanently; one clock makes that impossible by construction.
  The strict/loose disagreement is bounded by one turn's duration against thresholds of 600s/3600s —
  it buys minutes.
- *A per-session idleness abstraction mirroring `ActivityState`*: two users, two definitions.
  `ActivityState` explicitly disclaims the per-session grain (`servers/activity_state.py:8-10`).
- *Making the ref-type set a query parameter*: mis-models the axis. The gate does not want a different
  ref-type *set*; it wants the attendedness *predicate*, which is not a set.

**Files & call sites.** `servers/dal_logs.py` (+1 method), `servers/brain_traces.py` (+1 wrapper),
`servers/brain.py:1154-1176` (clock read + tail bound), `servers/scales/s1/encode_contract.py` (+1
named bound), `servers/daemon_server.py:1367` (derive the prune horizon from the new name). Under (P) instead:
`servers/brain_traces.py:1053`, `servers/brain.py:1156`, plus the two test files named above.

**Verification.** The existing idle-tail tests live in `tests/test_daemon_hooks.py:497-580`
(`test_idle_tail_fires`, `_guard_below_min_turns`, `_not_yet_idle_enough`,
`_encodes_a_dangling_question`) and stub `present_streams` with hand-authored `updated_at` — they
assert the *intended* semantics and have never exercised the real source, which is why this defect has
been green since it shipped. Add **one integration test through the real DAL**: a session with an old
conversational turn plus a recent heartbeat, asserting the tail fires. Add a second pinning the new
upper bound (a very old conversational turn does NOT tail-drain). Tier: `tests/test_daemon_hooks.py`,
`tests/test_self_presence.py`, `tests/test_trace_contract_sync.py`,
`tests/test_traces_layer_guardrail.py`, plus `tests/test_maintenance_gate.py` as the sibling gate.

**Blast radius.** Under (e′): additive, no existing test edits. Under (P): two test files move. Either
way the `five_plus` clause changes behavior for machine-woken streams — they stop qualifying as
"actively conversing" and take the tail path instead, which is the intended correction.

**Depends on.** Step 0 (gate). Step 1 is not a prerequisite but makes the diff more readable.
**Respects.** Locked B2 (`083b745d`) — presence keeps its heartbeat-inclusive clock; this stops a
second consumer inheriting it. Sibling stream `ace832e1` owns `daemon_server.py`'s poll machinery and
has confirmed `scribe_due(now=)` must stay wall-clock (its commit `e03e26b` passes wall-clock `now`
alongside monotonic local deltas) — this step does not change that param's basis, only the stored
timestamp it is subtracted from. Compatible with the approved-but-unimplemented "activity triggers the
scan, idle gates the drain" design (`docs/HISTORIC-DIGESTION.md`, brain node `164df998`): that design
changes the *candidate source* and explicitly keeps idle gating the drain, making this clock the sole
remaining drain gate — so this is a prerequisite for it, not a competitor.

---

## Step 3 — Add the missing ISO→elapsed helper to `clock.py`, with a required failure policy

**Problem.** `servers/clock.py` owns the ISO format contract (the `Z` vs `+00:00` hazard, the
tz-naive→UTC rule, `resolve_offset`'s fail-loud stance) but has only string→string helpers
(`iso_now`, `iso_cutoff`, `iso_after`) and **no** string→elapsed-seconds companion. So the parse is
re-implemented per consumer, and each one invented a different meaning for failure:

| site | fallback | what it means |
|---|---|---|
| `servers/brain.py:1158` | `0.0` | "just active" — arms `five_plus`, permanently disarms `tail` |
| `servers/channels/self_channel/presence.py:24` | `1e9` | "lost" |
| `servers/brain_assembly.py:604` | log + `0` | mixed |
| `servers/scales/s1/surface_contract.py:127` | `None` | absent |

Two consumers of the *same field* pick opposite defaults, neither logged. `brain.py:1158` resolves
"unknown" to "maximally active", silently — contrary to the repo rule "log failures; do not silently
drop failed processing".

**Target state.** `age_seconds(iso, *, now, on_error)` in `servers/clock.py`, with `on_error`
**required — no default**, so every caller must decide what an unparseable timestamp means rather than
inherit someone else's coercion. Converge the four sites. In `scribe_due` the correct outcome for an
unknown timestamp is to skip the session and log, never a value. After Step 2 the case is unreachable
when `turns > 0` (a counted `user_message` is always attended, so it always sets the clock) — state
that invariant in a comment rather than leaving it to an exception handler.

**Also author the three-valued clock rule in the module Contract.** `time.monotonic()` does NOT advance
while the process is suspended (macOS sleep) — documented at `servers/brain_constants.py:361-365` for
the SDK timeout, and nowhere in `clock.py`. So "which clock for an elapsed interval" is three-valued,
not two:

| | Question | Clock |
|---|---|---|
| A | semantic / persisted time | wall (`scribe_due`, the maintenance `brain_meta` stamp, backup mtime seed) |
| B | elapsed real time that MUST count suspend | wall (this is *why* the 4h idle timeout fired on lid-open) |
| C | elapsed process time, immune to NTP steps | monotonic (tick gates, cooldowns, rate limits, stall checks) |

`clock.py`'s Contract today says only "Exempt: telemetry/perf timers (use `time.monotonic` explicitly)"
— an escape hatch from the semantic-time rule, not a rule of its own, so every author re-derives the
answer per site. `age_seconds` is category A/B **by construction** (it parses a persisted ISO string
against wall-clock now); say so in its docstring, or someone will reach for it for a category-C tick
gate and silently get suspend-sensitive behavior.

**Do NOT widen `tests/test_clock_contract_sync.py`'s `PROTECTED_DIRS`** (currently `servers/scales`) to
cover `servers/`. A test in that file asserts it equal to `test_time_window_contract.CTX_PROTECTED_DIRS`;
widening it silently widens the *semantic-time* rule too. The interval-clock rule needs its own
ratcheting guardrail file (the sibling stream owning `daemon_server.py` is writing one in the
`test_raw_sql_guardrail` mold) — two rules, two files.

**Files & call sites.** `servers/clock.py` (+1 function); `servers/brain.py:1154-1158`,
`servers/channels/self_channel/presence.py:19-29`, `servers/brain_assembly.py:604`,
`servers/scales/s1/surface_contract.py:127`.
**Verification.** `tests/test_clock_contract_sync.py`, `tests/test_self_presence.py`,
`tests/test_daemon_hooks.py`, plus whatever covers `surface_contract`. Each converted site keeps its
current behavior except `brain.py`, whose new behavior is Step 2's.
**Blast radius.** Four call sites, one new helper. Behavior-preserving except where explicitly
changed. Note `servers/brain.py:1121` and `:1930` are a verbatim duplicate boot-grace expression —
fold into a `Brain._in_boot_grace(now)` predicate in the same pass if it stays small.
**Depends on.** Step 2 (edits the same lines in `brain.py`; sequencing avoids touching them twice).
**Respects.** Sibling stream `ace832e1` raised the same question from the daemon side and deferred the
shape to this plan — send them the resolved signature rather than letting a second shape appear.

---

## Step 4 — Rename the projection key to `live_recency`  ✅ LANDED

**Landed in two passes, and the first one was wrong — recorded because the error is instructive.**
Pass one renamed `updated_at` → `last_turn`, the DAL's own alias. That traded one misleading name for
a subtler one: `trace_contract.py` declares `"heartbeat": False — a wakeup re-arm is never a turn`,
while the value being named `last_turn` *counts heartbeats*. The rename made two layers consistently
wrong instead of inconsistently wrong. Pass two renamed the value at BOTH layers to `live_recency`,
which pairs with the `conv_recency` the same query already computes: two recencies over two row sets,
one proving reachability and one proving work. `presence.py` keeps `updated_at` as its own PUBLIC
output key, so the `self_presence` MCP surface is unchanged.

### Original step (historical)

**Problem.** `servers/brain_traces.py:1054` projects the key as `updated_at` — named after the
`session_state` column its own docstring explicitly says it is *not* ("Liveness is sourced from
real-turn S0 traces … NOT `session_state.updated_at`"). The DAL's own name for the value is
`last_turn`. The misleading name is part of why a second consumer read it as a generic "when was this
session last active" and got the reachability clock.

**Target state.** The projection key says what it is. `updated_at` → `last_turn`.

**Files & call sites.** `servers/brain_traces.py:1054`; `servers/channels/self_channel/presence.py:77`,
`:82`; any test asserting the key name (`tests/test_self_presence.py`).
**Verification.** `tests/test_self_presence.py`, `tests/test_self_signal.py`, `tests/test_daemon_hooks.py`.
Grep for the string form `'updated_at'` as well as attribute access — the dict-key form hides in stubs.
**Blast radius.** Small but wide-ish: a dict key with three production readers. Pure rename.
**Depends on.** None — but land it AFTER Step 2 so the rename does not collide with the clock change.
**Respects.** Nothing contested; presence behavior is unchanged.

---

## Also noted, deliberately NOT in this plan

- `scribe_due`'s starvation alarm (`servers/brain.py:1138-1146`) is the one side effect in a read-only
  decision function. It should move to the backlog reporter that `docs/HISTORIC-DIGESTION.md` §10
  already scopes — that design owns it.
- `SCRIBE_CANDIDATE_WINDOW_MIN` doing three jobs (candidacy filter, staleness bound, failure-
  bookkeeping horizon) is a known, recorded item owned by the same design.
- `servers/brain_traces.py:12-17`'s Sections docstring omits Presence and refers to `live_sessions()`,
  which no longer exists anywhere in `servers/` or `tests/`. Docstring rot; fix opportunistically.
