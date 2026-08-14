# Digesting Historic Data

Encoding old conversations into a live brain — what we established, and what's still open.

Parked behind moments recall (door 2). This note exists so the thread can be picked up cold.

Brain nodes: `f5808eba` (mutating vs additive), `dde74b04` (scope), `41432d58` (two use cases),
`dfd1cd6b` (proposal), `aa9015ca` (presence-bound retry), `f924b40c` (bi-temporal deferred).

---

## Insights

### 1. It isn't a time-travel problem — it's a mutation problem

Sorting encoder writes by whether order matters:

| Order-free (safe at any age) | Order-critical (unsafe once the brain has moved) |
|---|---|
| `remember` | `revise`, `absorb`, `archive` |
| plain `connect` | correction-family edges: `corrects`, `supersedes`, `reframes`, `resolves`, `fixes` |

The right column means *"this came after and overrides that."* A 2-day-old encode writing
`corrects` on a node a newer session already corrected inverts the chain — and `get_node()`
walks corrections on every canonical pull, so recall would serve the stale correction as
current truth.

So the question is not "how do we order time" but "is the late writer allowed to mutate."

### 2. `connect` is the bridge

The safe op set (`remember` + `connect`) is exactly the set that "attach old knowledge to the
current graph" requires. The blocked set is exactly the ops that would rewrite current
knowledge from a stale vantage. Additive-first isn't a constraint on the bridge — it *is* the
bridge.

### 3. The scope is narrow: constrain one chokepoint, don't teach every encoder

- **Only S1 Scribe consumes a conversation window.** Every S2 unit reads the *graph*, which is
  always as-of-now, so they have no "old vs current" question. (Sole conversation read in S2 is
  `healer_decoder`'s `get_conversation_around` — read-side, not a window encoder.)
- **All encoders write through one function.** `S1Scribe(IntegrationUnit)` writes through the
  same `_make_encoder_dispatch` (`servers/scales/s2/base.py:218`) as consolidation, community,
  healer and aspect.
- **That function already implements this shape of policy.** `archive_guard` is a per-run
  parameter restricting which nodes `archive` may target; violations are dropped and logged.
  A late-encode policy restricts *which ops* rather than *which nodes* — one more parameter.
- **The classification already exists.** "Is this relation order-critical?" is "is it in the
  `correction_improvement` aspect?" — `aspects_v1.json` names them, and code reads the registry
  rather than hardcoding.

Net: the encoder never needs to know it's late. It finds certain ops unavailable, exactly as
consolidation finds out-of-cluster archives dropped today.

### 4. Two use cases, one mechanism, different prerequisites

| | Backlog recovery (hours–2 days) | Historic training (weeks–months) |
|---|---|---|
| Dating error | noise | **fatal to recall** |
| Needs bi-temporal | no | **yes, prerequisite** |
| Needs additive-first | yes | yes |

Six months of knowledge stamped "today" corrupts recall's recency weighting across the whole
corpus. Backlog recovery tolerates the error; historic training does not.

### 5. The timestamp situation

- `created_at` — transaction-time (when the brain wrote it). Always.
- `event_time` / `event_date` kv — **subject** time: when the described event happened
  ("I went to the gym last Tuesday"). Not the conversation's date.
- **Valid-time — when the conversation happened — has no field.** It is recoverable only by
  walking `source_refs` → s0 traces.

This is deliberate. Commit `1a79d66` reverted a conversation-time anchor on `created_at`:
back-dating without a first-class transaction-time column is bi-temporal done halfway, making
`created_at` ambiguous and destroying "when was this actually written."

### 6. The 5-hour candidate window exists for exactly this reason

`SCRIBE_CANDIDATE_WINDOW_MIN = 300` was set in that same commit, with the rationale recorded:

> NOT longer: we stamp now() (transaction-time), so a longer look-back would resurrect
> genuinely-stale conversations dated "today" — unbounded catch-up is bi-temporal work.

Widening the window is not a config tweak. It is the thing bi-temporal gates.

Related: the retry is **presence-bounded, not backlog-bounded** (`aa9015ca`). `scribe_due` only
iterates `present_streams(window_min=300)`, so past 5h idle a preserved backlog is unreachable
regardless of size. Real unattended retry band is 1h–5h idle.

### 7. Interpretation and integration want opposite time anchors

- **Interpretation** — what did this conversation mean? Historically faithful. `conversation_now`
  already resolves relative dates against `session_started_at` (`encode.py:467`).
- **Integration** — where does this attach? Must be the **current** graph, because that's what
  it's writing into.

Time-limiting recall serves the first and breaks the second: you get well-understood nodes wired
only to what the brain knew then — islands. Historic ingestion that doesn't connect to current
knowledge is archival, not memory.

### 8. The encoder does not reach, and the catalog is hook-derived

Measured over 278 completed S1 encode runs (daemon telemetry, spans several prompt versions):

| reads/run | S1 encoder | S2 consolidation encoder |
|---|---|---|
| 0 | 276 | 174 |
| 1 | 2 | 89 |
| 2 | 0 | 4 |

0.7% vs ~35% on identical machinery — so it's design, not capability. The `s1e` v34 prompt
mentions neither `recall_batch` nor `get_nodes`: not a prohibition, a silence.

`_build_catalog` (`encode.py:505`) assembles context from `judge_outputs` — the per-message
recall-hook product — plus session-touched node ids. Volume: catalog p50 41 ids (p90 80,
max 127), prompt body p50 205K chars (p90 380K, max 504K).

**Consequence:** skip the recall hook during a replay and the catalog collapses to near-nothing,
and the encoder will not compensate. "Catalog + timeline is sufficient" (`11cf8246`) was a virtue
in the live path; it becomes a liability in the historic path.

### 9. The seam: keep the decoder, drop the surface encoder

Recall *is* the decoder; S1Surface *is* the encoder (`da2193b6`). The expensive part of replaying
the loop is running it per message — but the encoder doesn't need `additionalContext`, it needs
the catalog. So:

**Run recall once per encode window, not per message, and feed candidates into the catalog
directly.**

- current-brain attachment (the bridge) — free
- real retrieval instead of LLM-guessed queries
- ~1 recall per 5–20 messages instead of 1 per message
- **no change to S1E** — it still receives a populated catalog, the shape it's tuned for

Cost: the catalog today is Haiku's *curated selections*, not raw candidates. Feeding raw
candidates changes its composition and needs work in `build_node_catalog` plus an eval.

Why not the alternative (teach S1E to reach): it asks the encoder to improvise retrieval —
an LLM guessing query strings in place of the tuned embedding + TF-IDF + graph-expansion
pipeline. Strictly worse retrieval, on the highest-risk component in the system.

### 10. Backlog recovery should be event-triggered, not time-windowed

The candidate window is a proxy for the real question. "Was this session active within N hours"
stands in for "does it carry unencoded knowledge worth draining" — and the proxy fails at the
boundary, where any chosen number is just a choice of where to lose data.

More decisively: **no window can fix the failure we found.** The tail band was slept through, and
any window can be slept through. But waking *is* an event — the one a sleeping machine reliably
generates. Triggering on brain activity turns the failure mode into the trigger.

`SCRIBE_CANDIDATE_WINDOW_MIN` currently does three jobs at once:

1. the trigger's candidacy filter
2. the de-facto staleness bound ("too old to bother")
3. the `_scribe_attempts` / `_scribe_failures` bookkeeping horizon
   (`daemon_server.py:1153`, `horizon = SCRIBE_CANDIDATE_WINDOW_MIN * 60`)

Event-triggering separates 1 from 2: the trigger becomes an event, and staleness becomes an
explicit policy chosen on its own merits. (3 should stop being derived from it either way.)

**Refinement — activity triggers the scan, idle still gates the drain.** Draining at first
activity puts a backlog encode in contention with the live session exactly when work resumes;
recall is already 7–11s, and the first prompts of the morning are the worst moment to add a
concurrent Sonnet run. The existing tail clause only fires when a session is quiet, and that
instinct is right. So: wake makes the brain *notice* the backlog; idle still decides when to
drain it. Pacing is then mostly free — `MAX_CONCURRENT_ENCODES`, the 120s per-session cooldown,
and one-session-per-poll selection already exist.

**The unencoded-backlog alert is a companion, not an alternative.** It's computable today —
`present_streams` × `turns_since_last_encode`. Current signals are `scribe_starvation` (20+ turns)
and `scribe_repeated_failure`; nothing reports "N sessions holding M unencoded turns, oldest X
hours." Its real job is being the escape hatch for whatever falls outside the staleness bound:
sessions too old to auto-drain become visible instead of silently dropped.

Measured snapshot (2026-08-14) for sizing: candidate pool 22 streams at 5h, 42 at 36h — but only
**3** carried more than 2 unencoded turns, the tail clause's threshold. Pool size is not the bound;
`turns > 2` is.

---

## Open questions

**Blocking historic training**
1. Does bi-temporal need the full model, or is a `valid_time` column on nodes enough for the
   recall-recency case? What reads must switch anchors?
2. What exactly does consolidation need? A training run followed by an S2 cycle consolidates
   old-dated knowledge on this-week trace evidence — coherent, but ingestion-behavior evidence
   standing in for historical-behavior evidence (`c12c4735`).

**Design, answerable now**
3. Do raw recall candidates serve the encoder as well as Haiku-curated `judge_outputs`? Needs an
   eval before trusting the seam in §9.
4. **Anachronism:** recall against today's graph surfaces nodes that didn't exist during the
   conversation. Right for integration, contaminating for interpretation — can the encoder
   attribute knowledge to a conversation that didn't have it?
5. Drop-and-log, or S2 adjudication? Dropping a blocked correction is lossy, but "this old
   conversation conflicts with current knowledge" is arguably the most valuable output of
   historic ingestion. Start with drop+log and let the log size the next step.
6. Backlog drain order: `scribe_due` returns most-overdue *by turn count*. Chronological by
   window end is what makes a multi-session backlog self-consistent.

**Backlog recovery (shippable without bi-temporal)**
7. What counts as "first brain activity"? Daemon start after boot-grace, or the first
   `hook_recall` following an idle gap? The second is closer to the true wake signal, and
   `last_user_activity` already tracks it.
7a. What is the staleness bound, now that it's an explicit policy rather than a side effect of
   the presence window? Same axis bi-temporal gates — past a couple of days you are back to
   nodes stamped "today."
8. Cadence 5 → 10: halves the runs, but the window is already 10 turns
   (`max_messages: 20`), so overlap goes from 5 turns to zero — every turn seen once, no second
   look. Widening the window to restore overlap hits a tuned ceiling: 30 messages regressed hard
   (8 rounds / 8 recall calls / 291K tokens vs 2 / 0 / 58K). 24 may have headroom; untested.
   Also shifts more short sessions onto the fragile tail path — pair with the candidacy fix.
