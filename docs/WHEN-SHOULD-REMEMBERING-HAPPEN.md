# When Should Remembering Happen — handoff + open items

**Written:** 2026-05-31, end of the turn-classification session.
**Status:** the tactical work shipped (all on `main`, tested, tree clean, unpushed); the strategic question is parked for a fresh dive. This doc is the cold-start record — it assumes zero context.

Tom's framing: *"a real dive will be with the title 'When should remembering happen' but that's a bigger change I don't want to do now."* This is that dive's brief, plus every other leftover from the session.

---

## 0. The headline question (the dive)

**When is a turn a "real exchange worth remembering," and where should that decision live?**

The session fixed the *symptom* (`/watch` wakeups were firing the S1 Scribe on churn) but surfaced that the *signal* we used is too coarse. The dive is to design the right answer, not patch the current one.

### What we built (the current mechanism)

A turn is classified at the Stop hook:
- `hook_recall` (daemon side, `servers/daemon_hooks.py`) stamps `ctx.last_recall_stop = ctx.stop_counter` whenever it runs.
- `post_response_common` classifies `is_conversational = (ctx.last_recall_stop == ctx.stop_counter)`.
- **Two counters, two jobs** (`servers/session_context.py`): `stop_counter` = per-stop SEQUENCE (advances every stop → unique chain IDs); `conversational_count` = integration CADENCE (advances only on conversational turns). The Scribe gates on `conversational_count % 5`.
- Conversational turn → `user_message` + `assistant_message` S0 traces. Heartbeat → a single `heartbeat` ref_type (observability), no user/assistant trace, no cadence tick.
- The contract is `servers/trace_contract.py` → the **S0 TURN CLASSIFICATION** block + `CONVERSATIONAL_REF_TYPES` (the encoder/`get_session_turns` whitelist; flip `self_message` into it to enable anchor↔anchor encoding later).

### The flaw the dive must resolve (finding #1)

The signal is *"did `hook_recall` run this stop?"* But `hooks/scripts/pre_response_recall.py:27` skips recall **client-side for ANY prompt that is <5 chars, or starts with `/` or `!`** — not just `/watch`. So these REAL operator turns are misclassified as heartbeats and **dropped from encoding** (no `user_message`/`assistant_message`, no cadence tick, no Hebbian):
- short prompts: `Go`, `ok`, `yes`, `no`, `.`
- any real slash-command (`/code-review`, …) or bang.

The harm: Anchor's *substantive response* to a short operator directive ("Go" → implement + explain) is never recorded in S0 → invisible to the encoder via `get_session_turns` → never remembered. And a session of mostly-short turns never accumulates cadence → the Scribe never fires.

### Why it can't be cheaply patched

The daemon **cannot distinguish a scheduled `/watch` wakeup from a short/slash operator prompt**:
- Both skip recall (same `pre_response_recall` predicate), so "recall ran" is identical for both.
- The Stop hook receives the *expanded transcript text* (the `/watch` skill body, or the typed text), **not** the raw `/brain:watch` — so content-matching the raw command isn't available there.
- The harness delivers a `ScheduleWakeup` re-invocation as a normal `UserPromptSubmit`, indistinguishable from typed input at the daemon.

So a real fix needs a **finer signal** that separates "scheduled wakeup" from "operator acted." Candidate directions for the dive (unexplored):
1. A wakeup-explicit marker — have `/watch` / `ScheduleWakeup` tag its turns so the daemon knows "this was a heartbeat," instead of inferring from recall-skip.
2. Re-home the cadence decision fully in the brain — the Scribe reads the S0 record and decides "have N real exchanges happened" rather than the hook gating a counter (Tom's "filtering is the brain's job" principle, taken further).
3. Reconsider the recall-skip predicate itself — `<5 chars` is the part that catches `Go`/`ok`; maybe short prose should still count as conversational even when recall is skipped.

The deeper frame Tom named: **filtering noise is the brain's job, not the hook's.** S0 should observe faithfully + classify descriptively; S1+ should decide what's worth integrating. The current design splits this partway (hook tags, brain paces) — the dive is whether the "what's a turn" judgment belongs even lower-touch.

---

## 1. What shipped this session (context for the dive)

All on `main`, tested, **unpushed** (origin at `495f0df`), interleaved with `31e8c3ff`'s self-channel commits. My stack:

| Commit | What |
|---|---|
| `fce3570` | dashboard Boot view: fingerprint short-circuit stops the every-5s re-render/collapse. |
| `ec86aca` | retire `brain_dashboard.db` — removed the dead errors-panel source (the `CANTOPEN` noise loop); redirected the MCP health-monitor DAEMON_DOWN write to `brain_logs.db.hook_errors`. |
| `1e14058` | **Phase 1** — S0 turn-classification *contract* in `trace_contract.py` (heartbeat ref_type, `S0_CONVERSATIONAL_INCOMING` dial, `CONVERSATIONAL_REF_TYPES` derived; `get_session_turns` reads it). Zero behavior change. |
| `d2bc33e` | revert `ef5ff1f` (an encoder-side self-channel filter — wrong layer; superseded by the contract). |
| `dcc4bf5` | route the DAEMON_DOWN `hook_errors` write through `LogsDAL.log_hook_error` (no raw SQL in the MCP layer). |
| `4233eaf` | **Phase 2** — Stop-hook turn classification (heartbeats stop ticking the Scribe). |
| `f1abe15` | **Phase-2 revise** — split overloaded `stop_counter` → sequence (`stop_counter`) + cadence (`conversational_count`); fixed a chain-ID collision wart. |
| `770f443` | review fixes — contract-comment drift, fail-safe gate default, `hook_errors` DDL dedup via `schema.py`. |

Normal user↔anchor sessions are **byte-identical** under all of this; only watch sessions change.

---

## 2. Open — mine (from the xhigh code review)

- **#1 — the headline above.** Deferred to this dive. Not hacked.
- **#9** — `last_turn_conversational` is a third field for one fact (derivable from `is_conversational`). Left as-is: it's reasonable given the eval caller (`eval/s1s_ab_wiring_check.py` calls `post_response_common` directly), and `770f443` made its gate default fail-safe (`False`). Revisit if the gate is refactored.
- **#12** — no test ties the real `pre_response_recall` skip predicate to the classifier; the heartbeat test fabricates the stale-`last_recall_stop` state. Add a true end-to-end test when #1 is properly fixed.
- **#15** — the conversational vs heartbeat S0-trace branches duplicate append boilerplate. Minor.
- **#6 (NOT a bug)** — the MCP health-monitor's own `brain_logs.db` connection: the finder flagged "no busy_timeout," but `sqlite3.connect(timeout=3)` *is* the busy-timeout, and WAL/pragmas don't matter for one tiny write. No action.

## 3. Open — `31e8c3ff`'s self-channel work (relayed to it; theirs to action)

- **#3** — `servers/scales/self_channel/signal.py` `resolve_to()` calls `present_streams` **without `exclude_session`** → a sender passing its own 8-char short self-resolves → message self-addressed, filtered at drain, silently undeliverable. (Likely still live; the real one.)
- **#7 / #13** — `presence.py` `_age_min` uses bare `datetime.fromisoformat`; a blank/non-ISO `updated_at` → bare except → `1e9` → a LIVE stream silently classified `lost`. Route the parse through `servers/clock.py`.
- **#10** — `self_contract.address_from_target` is now dead code (`resolve_to` replaced its caller).
- **#11 / #14** — label N+1 lookups / `self_label_*` namespace: **likely moot** — `31e8c3ff` dropped `from_label` (ids-only now).

(Full ranked JSON of all 15 findings was produced in the session transcript.)

## 4. Open — operational / decisions (Tom's)

- **Daemon restart** — Phase 2 + the revise + the `brain_dashboard.db` retirement are all daemon code; live only on the next daemon restart. The running daemon is on pre-change code.
- **Push** — the whole stack (`fce3570`→`770f443`, interleaved with `31e8c3ff`'s) is unpushed; origin at `495f0df`. Left to the integrating stream / Tom.
- **`brain_dashboard.db`** — the file is now orphaned (no readers/writers after `ec86aca`). Awaiting Tom's explicit ok to delete it.

## 5. Open — the proactive architecture work (Step 3, parked)

Tom asked: *"are there more misplaced functions/definitions/decisions? how would you plan for the right architecture?"* The plan (parked):
1. **Codify the placement laws** (from CLAUDE.md prose → an explicit list): observation layers never judge value; schemas/definitions/limits live in `schema.py`/contracts, never inline in writers; no raw SQL outside `dal/`; one name, one concept.
2. **Make each mechanical law a scanning test** (extend the family of `test_time_window_contract`, `test_prompt_sync`, `test_dispatch_contract_sync`) — e.g. `no-raw-SQL-outside-dal`, `no-CREATE-TABLE-outside-schema`. These become the architectural eyes Tom doesn't have (he doesn't read code) and catch existing + future misplacements mechanically.
3. **Structured read-sweep** for the non-mechanical laws (filtering-in-observation, wrong-scale) — one pass per subsystem. Costs real tokens; get Tom's go + scope first.
4. **Triage by damage × risk; fix high-damage-low-risk first; lock each with a test.**

Misplacements already identified this session (for the audit's seed list): `stop_counter` overloading (fixed in `f1abe15`); `hook_errors` schema in 3 places (reduced in `770f443`); the recall-skip predicate inline in a hook (the #1 conflation); "heartbeat" naming overlap with `record_message`.
