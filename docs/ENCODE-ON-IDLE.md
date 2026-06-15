# Encode-on-Idle — don't lose sub-5-turn sessions

**Status:** design captured 2026-06-15. Mechanism deferred; this session's work is
the **prerequisite audit** (Part 2: make the encoder flexible on turn count).

## Problem

The S1 Scribe fires every `ENCODE_EVERY` (=5) conversational turns (the Stop-hook
gate in `daemon_hooks.hook_post_response_track`, cadence read live from traces via
`turns_since_last_encode`). The gate only fires **when a turn happens** — so the
*tail* of a session (the turns since the last encode) is never flushed:

- Long session: loses the trailing 1–4 turns since its last every-5 encode.
- **Short session (< 5 turns total): encodes nothing at all.** A quick correction,
  a key decision, a 3-turn debugging insight → gone.

`hook_session_end` does **not** encode (`synthesize_session` was removed 2026-04-13;
it only discards the session context and saves). There is no live PreCompact/
PostCompact brain hook. So today the tail is fully lost, not "lost under threshold."

## Core insight

The tail problem is **unsolvable from the Stop hook alone** — the tail is defined by
the *absence* of a next turn. Any fix needs a trigger that fires **without a turn**:
an idle timer, a lifecycle event, or a boot sweep.

The enabling property: **a flush is idempotent.** `turns_since_last_encode` is
trace-derived and self-resetting — an encode writes its `encoding_prompt` trace, and
the count drops to 0. So redundant flush triggers cost nothing; whichever fires first
encodes the tail, the rest no-op.

## Mechanism (decided): idle flush, sourced from traces

A single mechanism — **encode-on-idle** — covers steady-state *and* post-reboot,
provided one rule is honored:

> The idle poll enumerates candidate sessions **from the trace log, not from the
> in-memory `_session_contexts` cache.**

Why this rule is load-bearing:
- The in-memory cache is **empty after a daemon restart** and only lazily
  repopulates when a session next interacts. An idle session that never sends
  another prompt post-reboot would never be loaded → never flushed → lost.
- The traces survived the reboot. `SELECT DISTINCT session_id FROM trace_events
  WHERE created_at > cutoff` + per-session `turns_since_last_encode` finds exactly
  the sessions with a pending tail. (`TraceDAL.active_sessions_by_turn` already
  enumerates this way.)

Consequences:
- **No separate boot-sweep mechanism.** After restart the idle poll resumes and,
  over the next ticks, picks up orphaned idle sessions and flushes them.
- **No thundering herd.** Every encode already drains through the single
  `_encoding_lock` (serialized). Add a "at most one flush per poll tick" cap and the
  post-reboot catch-up is spread over minutes, one Sonnet call at a time.
- **Window length is a cost/latency knob, not a correctness knob.** Because the poll
  guarantees eventual flush, a generous window can't lose data — it only delays the
  tail-encode. So bias the window *long* to avoid wasting an encode on a session
  you're about to return to.

Wiring: reuse the daemon's existing idle poll (the same loop that calls
`run_maintenance_if_due`). Add an S1 tail-flush step that, per trace-derived idle
session with `turns_since_last_encode > 0`, force-dispatches one `run_encoding`
(bypassing the ≥5 gate) through `_encoding_lock`, capped per tick.

### Open tuning knobs (defer to build)

- **Idle window.** Don't guess — **mine it.** Every `user_message` trace carries a
  per-session timestamp, so the real inter-turn gap distribution is already on disk.
  Set the window near the 95th–99th percentile of normal inter-turn gaps so ordinary
  think/meeting pauses don't trip it. Prior: ~15–30 min.
- **Minimum flush size.** Tom's call (2026-06-15): **1 vs 2 isn't worth gating on** —
  the real challenge is decoupling the encoder from "5" (Part 2). Default: flush any
  tail ≥ 1 and let the encoder decide it's thin and write little/nothing.
- **Trace retention.** The flush must happen before `brain_logs` cleanup prunes the
  tail's `user_message` traces. Verify retention >> realistic daemon-down windows.

## Prerequisite (THIS SESSION): make the encoder flexible on turn count

Encode-on-idle will routinely feed the encoder **fewer than 5 turns** (often 1–3).
Today the encoder is implicitly calibrated to "~5 turns." That is already a latent
fragility independent of this feature — when `_encoding_lock` backs up, a run already
covers 6–8 turns — but the **low** end (1–2 turns) is untested.

Goal: the encoder should be **batch-size-agnostic by principle** — "encode what's
worth encoding from whatever you see; writing nothing is valid" is correct at every
size, and also hardens the existing lock-backup case.

**Audit task (filled in below as we map the code):** find every place the encoder
assumes a fixed/large turn count — prompt language, few-shot framing, token/truncation
budgets, node-count calibration, the message-gathering window — and list what must
change.

Verification before shipping the mechanism: run `s1_encode_eval` / the longmem Frozen
Corpus harness on deliberately small batches (1, 2, 3 turns) vs the 5-turn baseline.
Measure nodes-per-turn, confidence calibration, recall pass-rate of the resulting
nodes, and the noise/false-positive rate. Expected failure mode: **over-encoding thin
material** (inflating one throwaway turn into several low-value nodes).

---

## Encoder flexibility audit — findings (2026-06-15)

### Headline: the encoder is *already* structurally flexible on turn count. The real gap is **terminal-run semantics**, not batch size.

The encode pipeline is `_gather_messages → _build_user_content (+ muster) →
run_llm_loop → post-process` (`servers/scales/s1/encode.py`). Tracing it shows the
"5" is the **cadence**, not a window the encoder is built around:

- **Gather window is 20 messages (~10 turns), not 5.** `_gather_messages` →
  `get_conversation(limit=ENCODING_AGENT['max_messages']=20)`
  ([encode_contract.py:41](../servers/scales/s1/encode_contract.py)). A short
  session simply returns fewer turns — already handled. Consecutive every-5 runs
  therefore **overlap** (~10-turn window, 5-turn cadence); the encoder relies on the
  **journal** to skip already-handled material, not on the window size.
- **An idle-flush of a long session's tail is NOT new behavior.** It's an "early"
  every-5 run with a smaller new-delta (e.g. 3 turns since the last journal entry
  instead of 5). The encoder already handles a variable delta via the journal +
  node catalog. No dedup risk.
- **Node-count calibration is already batch-size-agnostic by principle.**
  [encoding_prompt.py:995](../servers/scales/s1/encoding_prompt.py): "Be expansive…
  if this turn has ten encoding-worthy atoms, call remember_batch with ten." There's
  no "expect ~N nodes per run" anywhere. Good — this is the right instruction at
  every size.

So feeding the encoder 1–3 turns does not break gathering, dedup, or node-count
calibration. Two things *do* need to change:

### CHANGE 1 (substantive) — terminal-run semantics: don't defer when there's no next run

The encoder is explicitly trained to **defer ambiguous/forming material to the next
run**:
- [encoding_prompt.py:987](../servers/scales/s1/encoding_prompt.py): "You run every 5
  messages. This isn't the only chance to encode — ambiguous topics will have more
  context next run."
- The `WATCHING:` journal category
  ([encoding_prompt.py:1291](../servers/scales/s1/encoding_prompt.py)): "threads
  forming across turns that aren't ready to encode yet."

This heuristic is **exactly wrong for an idle/terminal flush.** When a session has
gone idle and we're flushing its tail, there may be **no next run** — material the
encoder defers expecting more context is the material we're trying not to lose. A
terminal flush has different semantics than a mid-session run.

**Fix:** thread a run-context signal into the prompt so a terminal flush suppresses
deferral. Concretely:
- `run_encoding(...)` gains a `final: bool` (or `reason: 'cadence'|'idle_flush'`)
  param ([encode.py:21](../servers/scales/s1/encode.py)).
- The Stop-hook gate passes `final=False` (normal cadence); the idle-flush passes
  `final=True`. Plumbed through `run_in_background` (which already forwards
  `session_id`/`counter` to `run_fn`).
- `_build_user_content` injects one run-context line when `final`: *"Run context:
  this session has gone idle — this is likely the LAST encode of it. Do not defer
  encodable material to a future run; if a thread is worth keeping, encode it now or
  it is lost. WATCHING is for genuinely not-yet-meaningful threads only."*

This is the one change that actually matters for not losing sessions. It's small and
localized (one param + one conditional prompt line).

### CHANGE 2 (trivial) — fix the inaccurate cadence wording

[encoding_prompt.py:987](../servers/scales/s1/encoding_prompt.py) "You run every 5
messages" is doubly imprecise: the cadence is every 5 **turns** (≈10 messages), the
window is 20 messages, and with idle-flush it won't be "every 5" at all. Reword to be
cadence-agnostic: *"You run periodically — usually every few turns, sometimes when a
session goes idle. Each run you see a sliding window of recent turns; the journal
tells you what earlier runs already handled."* (Prompt change → goes through the
register → activate → `./dev sync-prompts` discipline; eval-gated, see below.)

### Non-issues (verified, no change)

- **Minimum flush size (1 vs 2 turns).** Tom's call: not worth gating on. Flush ≥1.
- **`max_rounds`/round structure.** Target 2 rounds; a 1-turn flush finishes in 1–2.
  No change.
- **Starvation alarm.** `scribe_is_starved` (4×ENCODE_EVERY) is a gate-wedge monitor;
  idle-flush *resets* `turns_since_last_encode`, so it can only *reduce* false
  starvation. No conflict.

### Soft / leave alone

- [encoding_prompt.py:391](../servers/scales/s1/encoding_prompt.py) "The five other
  turns in the session…" — illustrative source_refs-sparsity example; gently implies
  multi-turn but isn't a structural assumption. Leave it.

### To verify before shipping (eval, not code-reading)

- **Scouts (muster) over a tiny window.** facts/quote/temporal scouts run over the
  gathered messages; confirm they degrade gracefully (likely fewer candidates, no
  error) on a 1–2 turn input. Low risk, but verify.
- **Thin-material calibration.** Run `s1_encode_eval` / longmem Frozen Corpus on 1/2/3
  turn batches vs the 5-turn baseline (with and without the CHANGE-1 `final` line).
  Watch for **over-encoding thin material** and confidence inflation. This is the
  empirical gate on both the prompt rewrite and the `final` semantics.

### Summary of the change set

| # | Where | Severity | Change |
|---|-------|----------|--------|
| 1 | `encode.py` `run_encoding` + `run_in_background` plumbing + `_build_user_content`; prompt | **substantive** | `final`/`reason` param → run-context line that suppresses defer-to-next-run on a terminal flush |
| 2 | prompt line 987 | trivial | reword "every 5 messages" → cadence-agnostic |
| — | gather / node-count / dedup / rounds | none | already flexible |

---

## How the encoder currently distinguishes new vs already-handled (verified 2026-06-15)

Worth pinning precisely, because it's a common misconception that the encoder is
told "you see N new turns + context for the past M." **It is not.**

- The timeline renders **all ~10 turns identically** as `[TURN 1]…[TURN N]`
  ([encode.py:383-424](../servers/scales/s1/encode.py)). There is **no positional
  new-vs-context demarcation** anywhere in what the encoder sees.
- The **only** marker of "already handled" is the prose **Encoding Journal**
  (`ENCODED:/SKIPPED:/WATCHING:` from prior runs). The encoder infers what's new by
  reasoning against the journal, not from a turn boundary.
- The **Node Catalog** is built **only from surfaced nodes** (Haiku surface
  selections in the window) — so the encoder is **blind to nodes Anchor wrote
  directly via MCP** (`encoding_source='anchor'`) unless they were also surfaced.
  There is no query for anchor-authored nodes in the encode path.

So the new/old boundary today is **soft and self-reported** (journal prose), not
ground-truth.

## Design direction: per-turn provenance ledger (Tom, 2026-06-15)

Tom's proposal: reorganize the timeline so each of the ~10 turns shows, inline, what
already happened to it across **four provenance streams**:

1. **Surfaced** — nodes Haiku/S1R surfaced on that turn.
2. **Encoded by S1S** — nodes a prior Scribe run wrote anchored to that turn.
3. **Encoded by Anchor** — nodes Anchor wrote directly via MCP during that turn.
4. **Endo-surfaced** — nodes the *endo* system (new, soon to launch) surfaces on Stop
   / PreToolUse events.

### My take: this is the right direction — it supersedes the soft-vs-explicit boundary debate

The boundary stops being something we *assert* (a fragile last-encode-timestamp →
turn-index mapping) and becomes **emergent**: turns with entries under "Encoded by
S1S" are handled; the trailing turns with *empty* encode columns are self-evidently
the ones that "didn't get an encoding opportunity yet." Exactly Tom's point. And it
composes with the terminal-flush `final` flag: provenance shows *which* turns are
unencoded, `final` says *this is the last look at them*.

**Why it's more than nicer display — three structural wins:**

1. **Ground truth replaces self-report.** Today "what's already done" is the journal's
   prose (the encoder's own prior account, which can drift). Per-turn provenance is
   reconstructed from **traces + node `source_refs`** — what actually got written, by
   whom, anchored to which turn. Same principle that fixed the Scribe cadence: *traces
   never lie; self-maintained state desyncs.* The join is well-supported — a node's
   `source_refs` already anchor it to its originating turn trace_id(s)
   ([encoding_prompt.py:236](../servers/scales/s1/encoding_prompt.py)), so "encoded
   from turn N" = nodes whose `source_refs` include turn N's trace_id.

2. **Cross-actor coordination — closes the Anchor blind spot.** Stream 3 means the
   Scribe finally sees what Anchor already captured mid-conversation and stops
   re-encoding it — and, more valuably, can target what Anchor *missed*. Distinguished
   cleanly by `encoding_source` (`anchor` vs `encoder:sonnet` vs endo's own tag).

3. **Coverage feedback.** "Surfaced on turn 3 but nothing encoded from turn 3" is a
   visible signal — either a correct skip or an encoding gap the encoder can look
   harder at. The brain starts seeing its own coverage.

**What the journal becomes:** provenance takes over the *factual* half (ENCODED — now
ground-truth). The journal keeps the *judgment* half (SKIPPED rationale, WATCHING
threads) — the parts not derivable from traces. Clean division: traces for "what
happened," journal for "what I decided/noticed."

### Constraints / risks to design against

- **Token cost.** 10 turns × 4 streams could bloat the prompt. Keep the ledger terse:
  id + short title references *into* the catalog (which already renders full content
  once), plus counts — never re-render node bodies per turn.
- **Provenance ≠ mandate.** Showing "node X surfaced on turns 3,4,7" must not nudge
  the encoder toward dense source_refs / over-linking (the anti-pattern at
  [encoding_prompt.py:394](../servers/scales/s1/encoding_prompt.py)). Frame as "this
  is what happened," not "link to these."
- **"Already encoded" ≠ "don't touch."** A turn encoded by a prior run may still need
  a **revise** if later turns reframe it. The ledger should read "already captured —
  revise if new context changes it," not "done."
- **Endo isn't live yet.** Design the ledger **stream-extensible** (add the endo
  column when it ships) — don't block the first three streams on it.

### Net

Per-turn provenance is the structural answer to both this doc's questions at once: it
makes turn-count flexibility a non-issue (the boundary is emergent from "what's been
encoded," not a hard-coded 5) *and* gives the terminal flush a precise picture of the
unencoded tail. It's a bigger change than Changes 1–2 above, but it's the one that
moves the encoder from self-reported continuity to ground-truth awareness of
everything that happened since its last run — across all four actors.

---

## Prompt information architecture — research synthesis (2026-06-15)

Contained web research into how to lay out a long mixed-content prompt (conversation +
per-turn provenance) for an LLM agent. Sources: [Anthropic long-context
tips](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/long-context-tips),
["Lost in the Middle", Liu et al. 2023](https://arxiv.org/abs/2307.03172), Anthropic
prompting best-practices.

### The reconciliation: two kinds of "instruction" → a sandwich

"Instructions first" and "data first, query last" aren't in conflict — they govern
different things:

- **Stable role / rules / format** ("how to encode") → **first** (system prompt). This
  is also our 1h cache prefix, so it's forced here anyway.
- **Actionable task / focus query** ("what to do with *this* data") → **last**, after
  the variable data. Anthropic: *"Place your long documents near the top, above your
  query… Queries at the end can improve response quality by up to 30%… especially with
  complex, multi-document inputs."*

→ **Sandwich:** stable framing on top, variable data in the middle, the actionable task
restated at the bottom. "Lost in the middle" is why: accuracy peaks at the **beginning
and end**, worst in the middle, *even for long-context models*. Never put anything
load-bearing in the dead middle.

### Decisions

| Question | Verdict | Basis |
|---|---|---|
| Instruction placement | Stable rules **before** the data; actionable task **after** it. Nothing load-bearing in the middle. | Anthropic long-context tips; lost-in-the-middle |
| Glossary vs inline | **Keep the glossary.** Catalog = full nodes once, near the **top**; reference by **id** everywhere else (incl. provenance). Never re-inline node bodies. | Anthropic doc+metadata structure; token economy |
| ID-dereference reliability | Reliable on frontier models **iff** the catalog isn't buried — keep it compact and **high**. | positional effects |
| Provenance: interleaved vs parallel | **Interleave, terse, after each turn.** A parallel section forces list-by-index alignment (the cross-reference that degrades mid-context); inline keeps the turn↔provenance binding local. *(reasoned from positional findings, not directly tested — A/B with `s1_encode_eval` if built.)* | reasoning |
| Formatting | **XML tags** for major compartments (`<node_catalog>`, `<timeline>`, `<turn>`). | Anthropic trains Claude on structured prompts |
| Cross-model | Structural guidance **transfers** (lost-in-the-middle is general; ends beat middle everywhere). Only divergence: XML is Claude-specific convention — and our pipeline is all-Claude (Sonnet encoder, Haiku surface), so no hedge needed. | cross-model long-context literature |

### Caching interaction (our constraint)

"Task at the end" does **not** fight prompt caching: everything after the first variable
byte is uncached anyway, so a trailing task block costs nothing. The cached 1h prefix
stays = system + stable preamble at the very top. Only the `final`/run-context line is
dynamic; keep the rest of the closing task in the stable preamble.

---

## Proposed section structure

Top → bottom is the sandwich. Load-bearing content sits on both strong ends (catalog at
top, newest/unencoded turns at the bottom); already-handled older turns tolerate the
dead middle.

| # | Section | Position | Cache | Role |
|---|---------|----------|-------|------|
| 1 | System: role + encoding rules + format + how-to-read | top (strong) | 1h | stable "how to encode" |
| 2 | Preamble: section legend + "read before acting" | top (strong) | 1h | teaches the layout once |
| 3 | `<continuity>`: encoding journal + session context | early | 5m | priors — what prior runs did / what the session is about |
| 4 | `<node_catalog>`: full rich nodes, once | early-mid (must precede refs) | 5m | the glossary everything dereferences by `id:` |
| 5 | `<timeline>`: turns + terse inline provenance | bulk; newest last → strong end | 5m | the data + the emergent boundary |
| 6 | `<task>`: encode-what's-new + identify-first + `final` line | very end (strongest) | mostly 1h; `final` line dynamic | the actionable query |

### Example — a timeline turn (interleaved, terse provenance)

```xml
<turn n="8">
  <user trace="a1b2c3d4">…operator text…</user>
  <assistant trace="e5f6a7b8">…Anchor text…</assistant>
  <provenance>
    surfaced(Haiku):  id:1c0v, id:9ab2
    encoded(S1S):     —
    encoded(Anchor):  id:7f3e
    endo:             id:44d1
  </provenance>
</turn>
```

`encoded(S1S): —` is the **emergent boundary** — it clusters on the recent turns, so
"what hasn't had an encoding pass" is visible at a glance, no computed turn-index.
Node bodies live only in the catalog (§4); the turn carries `id:` refs only — never
re-inline (glossary+reference rule).

### Example — the closing task block (strongest position, last)

```xml
<task>
  Encode what's new since your last journal entry.
  Turns whose provenance shows no encoded(S1S)/encoded(Anchor) are your focus —
  they have not had an encoding pass yet.

  First, list the turn numbers that are unencoded. Then encode them.

  <!-- dynamic, only on an idle/terminal flush: -->
  FINAL FLUSH: this session has gone idle — this is likely the last encode of it.
  Do not defer encodable material to a future run; capture it now or it is lost.
</task>
```

Two research techniques fold in here: the task is **last** (the query-at-end ~30%
effect), and "**list the unencoded turns first, then encode**" is Anthropic's
ground-before-acting / extract-quotes-first move — it forces the model to bind to the
data before writing, turning the emergent boundary into an explicit decision.

### Why this composes with the three threads

- **Turn-count flexibility** is a non-issue: no "5" anywhere — the boundary is whatever
  provenance shows unencoded. A 2-turn flush and a 9-turn one read identically.
- **Terminal-flush gap** → §6's `final` line (the one substantive behavior change).
- **Cross-actor awareness** (Anchor's direct writes, endo) → two more provenance lines
  in §5; stream-extensible, endo slots in when it ships.

### Open questions for eval

1. **Interleaved vs parallel provenance** — the one layout choice reasoned (not cited).
   A/B inline-`<provenance>`-per-turn vs a separate aligned block; measure correct
   turn↔node association and tail-coverage.
2. **Catalog size vs position** — a large catalog pushes early turns toward the dead
   middle. Measure whether large-catalog runs lose recall/linking on mid-window turns;
   consider a size cap or summarizing cold catalog entries.
3. **Ground-before-act** — does "list unencoded turns first, then encode" actually
   improve tail coverage and reduce over-encoding, or just add latency?
4. **`final` line efficacy** — does it measurably reduce tail loss WITHOUT inflating
   node count (the over-encode-thin-material risk)?
5. **Provenance vs journal-only dedup** — does ground-truth provenance reduce
   re-encoding of what Anchor/prior-S1S already wrote, vs today's journal-prose baseline?
6. **Effort tier interaction** — see below; effort is co-gated with batch size.

---

## Effort / thinking level for S1 Scribe

**Current baseline:** the encoder runs `claude-sonnet-4-6` with **thinking OFF** (no
`thinking` param, no `effort`/`output_config`) — direct generation over ≤5 tool-use
rounds ([runner.py:313](../servers/scales/runner.py),
[encode.py:197](../servers/scales/s1/encode.py)). `effort` requires adaptive thinking
(`thinking:{type:"adaptive"}` + `output_config:{effort: low|medium|high|xhigh|max}`).

**The trade-off for THIS task.** Encoding is reconciliation-heavy (atomize; dedup
against journal + catalog + 4-actor provenance; decide links; calibrate confidence) —
so *some* reasoning helps, and the richer provenance-ledger input increases that load.
But it's a **bounded, contract-specified** task, not open-ended research. And it runs
**often** (every 5 turns + idle flushes), so thinking tokens multiply across runs.

**The specific risk: high effort fuels over-encoding.** Sonnet/Opus 4.6 over-explore at
high `effort`; on a thin idle-flush batch that directly produces the
over-encode-thin-material failure mode (one throwaway turn inflated into several
low-value nodes). So effort should **scale with batch size**, not be pinned high.

| Effort | Quality on a real batch | Over-encode risk (thin batch) | Cost/latency | Verdict |
|--------|------------------------|-------------------------------|--------------|---------|
| off (today) | adequate; weaker dedup/linking reasoning | low | cheapest | baseline |
| low | better dedup/reconciliation; little over-think | low | small bump | likely sweet spot for small flushes |
| medium | best reconciliation on big/backed-up batches | moderate | moderate | likely sweet spot for normal/large batches |
| high+ / xhigh / max | diminishing returns; over-exploration | **high** | expensive | avoid for encoding |

**Recommendation (eval-gated):** turn on adaptive thinking at a **modest** tier and
**co-gate it with batch size via the same `reason`/`final` param** we already need —
small idle flush → `low` (or off); normal/large batch → `medium`. Never high+. Interleaved
thinking also helps the multi-round flow (reflect on round-1 writes before round-2).
Gate on `s1_encode_eval` across tiers (off/low/medium/high) measuring nodes-per-turn,
dedup correctness, confidence calibration, noise rate, AND token cost + latency — pick
the tier where quality plateaus before cost climbs.

