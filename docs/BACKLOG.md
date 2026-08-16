# Backlog — the working queue

**The brain is the source of truth. This file is a queue.** One line per open item:
what it is, the brain node that holds the reasoning, and the code symbol to start
from. If this file and the brain disagree, the brain wins. If it and the code
disagree, the code wins.

**Cite code by symbol, never `file:line`** — the line anchors in this doc rotted
silently more than once.

**Last swept: 2026-08-13.** Every item in *Now* and *Decisions* was verified against
live code this date. Items verified done were **struck, not archived** — git and the
brain hold the history, so this file no longer carries a ship log or a completed
table. Two whole priority bands (the old P1/P2 recall arc) were written pre-LAF and
described a retrieve-then-rank pipeline `BRAIN_RECALL_VARIANT=laf_v1` has replaced;
what survived the check is folded into *Now* as LAF lanes.

**The mission:** recall — the moment relevant memories rise into Anchor's awareness
when the operator speaks. Everything here either improves that moment, validates it,
or is hygiene that stops it regressing.

Companion docs: [DISTRIBUTION-READINESS.md](DISTRIBUTION-READINESS.md) (launch — its
own ordered checklist, not duplicated here) · [EVAL-PLATFORM.md](EVAL-PLATFORM.md) ·
[TEMPORAL-ARCHITECTURE.md](TEMPORAL-ARCHITECTURE.md) ·
[IDENTITY-RESEARCH-2026-05-24.md](IDENTITY-RESEARCH-2026-05-24.md) (the borrow list
and the eight research questions live there).

---

## Now — verified open

### 🔴 1. Prompt improvements never reach existing installs
`interaction_seed._register` returns early when the name already exists, so an install
seeds its prompts at **first boot and is frozen there forever**. Fresh installs are
fine; every existing one is permanently stale, for *every* prompt change. Verified
2026-08-13: the early return is live, and the only other `register_interaction` caller
is the MCP handler (a human). Blocks the value of every prompt improvement the moment
a second machine exists — so it gates launch, not just hygiene.
**The fix exists in git and was thrown out with something else.** `dfc74ee`
(2026-08-09, "versioned migration layer + shipped-prompt reconcile") built exactly
this: `SEED_PROMPTS_VERSION` + `reconcile_seeded_prompts`, which advances a prompt
**only** while the install is still running the shipped default — the moment a human
registers or activates anything, that prompt is hands-off for good. Gated so it runs
once per version bump. It shipped with 145 lines of tests
(`tests/test_seed_prompt_reconcile.py`). It was reverted in `58581ff` because it rode
in the **same commit** as the migration-runner layer that three reviewers found dead by
construction. So the work here is re-landing the reconcile half on its own, not
designing it — and it couples to item 3, which is the other half of that same revert.
Same-day correction that still holds: **S2 does not rewrite prompts** (zero
`register_interaction` call sites under `servers/scales/`; the `s2:*` `created_by`
values are caller-supplied strings). CLAUDE.md's present-tense self-modification claim
is aspirational, which reopens interactions-as-files.

### 🔴 2. Fatigue: accumulates, but nothing observable happens
**The four-test "dampening cluster" was closed 2026-08-13 — the tests were stale, not
the code.** They read `brain._fatigue_ctx` / `brain._session_fatigue`, attributes that
do not exist in `servers/`, so they could never observe fatigue no matter how well it
worked; and `test_hub_dampening` drove `brain.recall()` while hub dampening only
touches the keyword channel inside `_keyword_recall`. All four deleted, the
`pytest.ini` deselect block deleted, one honest replacement test added
(`test_fatigue_accumulates_within_a_session`), suite green.

**What the replacement exposed, unexplained, is the real item.** Give recall a session
and fatigue *is* recorded — but:
1. **The count never passes 1.** A second recall of the same query in the same session
   leaves the top node at `fatigue=1`, though `_mark_accessed` increments
   unconditionally when ctx is not None and `get_or_create_session` returns a **cached
   instance by reference**.
2. **The score does not move.** Those two recalls return `effective_activation`
   identical to 16 decimal places, though the path reads as intact: LAF field score →
   `sim *= (1 - fatigue)` → `embedding_scores` → `blended` → `effective_activation`.

**⏸ PARKED 2026-08-13 — operator call: the fatigue rework is a whole session, don't
touch it piecemeal.** Don't debug the outside mechanism: the destination is already
decided, fatigue becomes a **LAF inhibition lane** with a trainable `gain_fatigue`
rather than an outside multiplicative dampener (brain `7e9e36a7`), and the lane fit
subsumes both observations above. The two observations are recorded here and in the
replacement test's docstring so the eventual session starts from evidence instead of
re-measuring. Nothing about the current state is load-bearing enough to rush: the
mechanism records, it just doesn't visibly bite.

### 🔴 3. Migration runner rebuild (attempt 2) — unowned
Attempt 1 was reverted: `MAIN_MIGRATIONS` was dead by construction (`ensure_schema`
stamped `BRAIN_VERSION` at step 7, the runner re-read it at step 8, saw itself current,
ran nothing). Verified 2026-08-13: **`MAIN_MIGRATIONS` no longer appears anywhere in
`servers/`** — this is a build, not a patch. Fix: the runner owns the stamp; the
logs-side integration was correct and is the model. Mechanism + repro: brain
`b5b72b74`. Latent only because the list is empty — it detonates on the first real
migration, which is exactly when strangers have brains nobody can hand-fix.
**What a version bump costs — measured 2026-08-14, the backup is not the problem:**
`_backup_before_migration` copies the live 723 MB `brain.db` in **0.121 s** (APFS
copy-on-write clone), so the boot-stall fear here was wrong. The fleet rule (brain
`56890464`) still binds, but it binds the *migration step*: a closed port reads as
*dead* and the watchdog can `kickstart -k` after ~20 s, so a slow row-by-row step
is the hazard, not the copy. Second real cost: each bump leaves a `.vN.bak` that
nothing prunes.
**Queued behind it, one bump should serve all:** 12 + 15 undeclared dead tables
(inventory: brain `2b49ac02` — "undeclared" is not "drop-safe"; ~12 names still return
code hits) · `bridge_proposals` (undeclared, 0 rows, still present on existing brains)
· 648 stale `keywords` KV rows (purge + `contract.py` skip_keys guard release ship
together). `brain_logs.db` has no migration mechanism at all after the revert — its
half is blocked, not deferred.

### 🔴 4. Generic-edge pollution — 18.1% of the live graph carries no relation signal
7,243 of 39,975 live `edge_relations` are `related` (2,527) / `related_to` (4,716).
Edge weight is dead (static `0.5`); relevance is cosine against
`edge_relations.embedding` — so nearly one in five edges contributes nothing to
activation while the `brain_batch` write door already forbids the verbs. A
recall-quality problem in a data-hygiene costume.
**Leak-vs-legacy: ANSWERED 2026-08-13 — legacy** (brain `5090d78b`). Live counts by
month: Mar 589 · Apr 4,414 · May 1,823 · Jun 405 · **Jul 0 · Aug 0**, total 7,231, no
NULL `created_at`. Last one minted 2026-06-23. Nothing is producing them, so
reclassification is safe — the pool will not refill.
**But the fix reaches only a third.** `RelationReclassifier`
(`servers/scales/s2/archive/reclassify.py` — verified `archive/`, in-repo,
out-of-package) works by reading an edge's description. Only **2,332 of 7,231 (32%)
have one**; the other **4,899 (68%) have nothing to read**.

**Route decided 2026-08-13 (operator): S2Healer, incrementally.** These are *not* noise
edges — `related` / `related_to` sit in the **`generic_relation`** aspect, which is not
the read block list (`noise` is), so they surface in reads and occupy activation while
contributing nothing. That rules out "leave them alone". The Healer is the right home:
it already runs decoder-finds-candidates → encoder-judges-per-node on an idle cycle, so
it absorbs both subsets without new infrastructure — description present → reclassify
the verb; description absent → judge from the two endpoint nodes, which is the only
signal left. Slow and continuous beats a one-shot bulk pass, and bulk passes are what
produced 1,488 of these in the first place. Design precedent for the same Healer slot:
brain `5d1dc397` (stale-status audit — aspect-resolved detection, encoder judgment).
Provenance: 5,188 `(unset)` (pre-provenance era), 1,353 `migration:v22`, 551
`s2:consolidation`, 135 `migration:consolidation_edge_recovery-20260421`. Two
migrations account for 1,488 — this is substantially bulk-operation damage, not the
encoder writing bad verbs one at a time.
**Correction to a claim this file used to repeat:** the ban is not a hard rejection at
the write door. It lives in the `connect_to` **schema description** in `contract.py`
("NEVER `related`/`related_to`/empty") — it instructs the model. `dispatch_write` still
defaults `relation = args.get("relation", "related_to")` in two places, so a caller
that omits the field silently gets one. That default is the suspect if they ever
reappear.

### 5. Healer and AspectIntegration write zero journal notes
Two of four S2 units are mute: across 589 notes / 14 days the split is S1E 317 /
community 172 / consolidation 100, healer and aspect **zero** (brain `78677e17`).
Cause is structural, not neglect — both are single-shot `_call_llm()` agents, so the
`## Review` fence contract has nowhere to live. **Decided approach (brain `4a21305b`,
Option A):** optional `review: [{tag, subject, note}]` as the last field of their
output JSON + a structured ingestion path in `write_journal_notes`. Sharpened since
filing: with `min_count_threshold=1` every aspect classification is a single-example,
permanent, never-revisited decision whose rationale survives only in
`aspects_proposed.json`, overwritten each cycle — the journal is the only possible
audit trail (brain `abfec5e6`).

### 6. `source_refs` render expansion at SURFACE_FORMAT
The recall-side joint-reactivation read shape: when a source-anchored node surfaces,
expand its `source_refs` inline. Designed in
[EPISODIC-REFERENCES.md](EPISODIC-REFERENCES.md) §8. Verified 2026-08-13: **zero
`source_ref` handling anywhere in `surface_contract.py`** — ground-up, not
"almost done". Behind it: `source_summary` parallel-pathway recall scoring (§9.5,
`max(legacy_weighted_sum, source_summary_score)`, backwards-compatible by design) and
S2Healer `source_refs` cleanup (§10.6 — scan for invalid trace_ids, archive orphan
`co_anchored` edges).

### 7. Frame frontier as a scoring lane (was P1.1)
Frame ships as a *prompt prior* — a "Partnership context (your prior)" block that
sways Haiku's choice **among** the candidates. The distinct, still-unbuilt lever
changes **which** candidates make the pool: score on Frame frontier IDs (and 1-hop
neighbors). Verified 2026-08-13: no `frontier_ids` / `frame_match_boost` in `servers/`.
Under LAF this is a lane, not a bolt-on boost. **Acceptance must be built first** —
the two failure queries this item has always cited (`aspect_encoder_pickup`,
`frame_recall_resume`) do **not** exist in `eval/frame_replay.py`; add them and
re-confirm the miss reproduces before building.

### 8. Cue-side temporal lane in LAF
No activation field reads a time expression *in the query* ("2 years ago", "last
month", "before the migration") and turns it into a window over node `event_time`. The
existing recency operator works on the node side (temporal distinctiveness), not the
cue side. Brain `56ffde45`. **Measurable-first:** the residual analysis (brain
`9f053861`) found the corpus contains zero temporal cues, so the field is untestable
until gold-growth adds them — this is "make it measurable", not "build it".

### 9. Healer returns unsolicited fields
The healer asks for specific missing fields; Haiku returns all three. Rejected and
logged loudly at `healer_encoder` (`healer_unsolicited_field`) — verified live. Now
has production evidence: **8 occurrences in a single 20-item corpus build** (brain
`ee7ba843`, which also found healer hyperactive at 184 actions vs consolidation's
zero). Fix is prompt (move the single-field example first) or `tool_choice` schema.

### 10. Frame's `## Recent moves` renders empty forever
`frame.py::_render_recent_moves` reads the `encoding_journal_{sid}` blob via
`brain.get_recent_encoding_journal` — and s1e v29+ never writes it. Verified live: both
symbols present, wired into the Frame render. This is the cut the v29 activation
called for and skipped; it kills an always-empty boot-context section. The blob
*writer* (`encode._save_journal`) is called only on the control arm.
Sibling tiers of the same sweep: the `BRAIN_S1E_LIVED_SEQUENCE` rollback path stays
until the flag itself retires; vestigial `SKIPPED:`/`WATCHING:` journal instructions
sit in the journal-exempt healer prompt, and a stale `'encoding_journal'` label sits in
`scribe.K_SOURCES`.

### 11. Thalamus — test the premise before designing v2
v1 was rejected on two fatal findings; v2's direction is a triager, not a key
([THALAMUS-DESIGN.md](THALAMUS-DESIGN.md)). **The premise — "answer an encoder and it
stops re-asking" — is being tested for free right now:** after v13 + the settlement
aspect, the consolidation journal should stop raising suppression complaints within
2–3 idle cycles (brain `3fa5adb7`; watch via
`query_traces(ref_type='journal_note', scale='s2')`). Validate or kill it before
spending build effort. Standing burden: `bridge_proposals` was built for inter-scale
coordination and **died unused** — [LATERAL-SCALES.md](LATERAL-SCALES.md) requires any
new inter-layer channel to prove it isn't just async S0 (brain `bfc6d106`).

### 12. Small, verified, cheap
- **Dead extras guard** in `trace_contract.build_delta_metadata` — reserved keys are
  keyword-only, so `if k not in metadata` can never fire. Remove it and retire
  `test_extras_do_not_overwrite_reserved_keys` (`tests/test_trace_delta_shape.py`).
  ~10 min.
- **`haiku_id_outside_candidates`** still logged as an *error* — investigation confirmed
  it isn't a bug (Haiku correctly picks IDs from prior turns that resolve to real
  nodes). Rename to `haiku_id_from_prior_context`, downgrade to debug so real errors
  aren't buried. ~20 min.
- **29 pre-existing test failures** from the `session_context` signature drift
  (scout_muster / trace_system / s1_data_assembly still call stale signatures). Pure
  test maintenance.
- **`judge_output` → `surface_output`** across the trace metadata contract. Derived
  field, no data migration. Defer until something else touches `dal.py`.

---

## Decisions needed

These aren't builds — they gate other work. Each needs the operator.

| # | Question | Gates | Lean |
|---|---|---|---|
| **D-5** | Seed pack: what a stranger's Anchor wakes up as | the public publish (5.6) | dedicated session, redo from current understanding |
| **Aspects fork** | Should `part_of`/`contains` veto consolidation the way correction edges do? Three reviewers split 1–2; sitting uncommitted (brain `24529407`) | aspect-ownership thread | restore the union |
| **Thalamus** | Recipient is Anchor not the operator (re-confirm) · accept the boot-renderer gap or overlap the two renderers · is it the next build? (brain `3c4aadad`) | the Thalamus build | recipient = Anchor |
| **Fencing** | Multi-user: fence sessions/traces per operator, or intentionally shared? Operator's own framing — "not sharing is a function not a structure" (brain `030be61c`) | multi-user completeness | soft fencing (display/routing) |
| **Watchdog** | Daemon `memory_watchdog`: enable now or after the next leak? (rss 1.14 GB, +139 MB growth 2026-08-04 unlocated, brain `b92443f3`) | daemon stability | enable now |
| **Aspects** | AspectIntegration auto-merge, or operator-review gate in production? | S2 autonomy | auto-merge for now |
| **Dormant prompts** | Three versions registered DORMANT since May (`s1e` v24-era multi-ref anchoring, `s1_scout_facts` v7, `s1_scout_quote` v4) — activate or retire? Note the methodological floor: **LLM encoders are stochastic at N=1**, so any re-run decision needs N≥3 | prompt hygiene | retire what nobody has needed in three months |
| **S1S rubric** | The per-node/per-run Haiku-judge rubric was designed and never built — still wanted, or drop the design intent? | validation infra | drop |

---

## Parked with a design — the brain holds it, don't redesign

- **`remind()` / prospective memory** — guaranteed time-triggered delivery (recall is
  probabilistic; a reminder must fire). Full design + scope guards: brain `ee7224ed`.
  It has since argued for itself in the wild — one ping needed three mechanisms because
  none is both durable and time-triggered (brain `34119238`).
- **Audience-before-authorship stance for `skills/brain/SKILL.md`** — drafted text
  locked, placement decided, operator call was "remember + backlog, build later". Brain
  `bc01bf96`; the incident that produced it is brain `6d4c012a`. Ship both layers or the
  reminder is decorative — a boot line won't surface mid-assembly (brain `c86ff5e9`).
- **Ranking on edge weight instead of cosine** — policy says never rank on weight;
  live paths still do. Weight is uncalibrated (written once at creation, never learned).
  Worst offender is the surface spread truncation on the S1R hot path
  (latency-sensitive, benchmark-first). The recall-traverse and neighbor-attach sites
  sit inside the path LAF replaces — fix-or-deprecate is decided by the LAF ship, don't
  polish condemned code. Query-less callers need an operator tiebreak (recency vs
  relation-aspect priority). Brain `39f01805`, `5a58ea33`.
- **Encoder-visibility thread** — per-encoder `get_nodes_config` gradient +
  aspect-owned read-exclusion (`get_node` derives `exclude_relations` from the `noise`
  aspect; the DAL drops its hardcoded constant) + curriculum metadata view. Phased,
  impact-analyzed. Memory `project_encoder_visibility_thread`.
- **Two execution-ready specs, never started** —
  [CROSS-STREAM-ON-SCALES.md](CROSS-STREAM-ON-SCALES.md) (make a delivered cross-stream
  message a first-class S0 turn so S1R surfaces against the body, not the
  `<task-notification>` envelope; 5 open decisions resolvable at session start) and
  [SELF-RECOGNITION.md](SELF-RECOGNITION.md) (cheap Stop-recall on my own output,
  stashed for the next prompt; 6 open decisions, stance discipline is the safety lever).
- **Endo-recall: which gate actually binds.** The §18.19 sequence puts a
  lens-independent Opus-judge gold corpus first (~75–85 cues, ~10M tokens, explicit go
  required — brain `aa29ac12`). But the teacher-on-production baseline is
  corpus-independent and already satisfies measure-before-you-change: n=90 real surface
  turns, 51% served_well / 78% partial+, and it localizes the failure to **factual
  queries (~46%)**, the flat-embedding wall (brain `a39e0104`). It also killed a lever —
  cosine-rank re-ranking is dead. Endo has no Haiku layer to compensate for weak
  ranking (brain `7ed82521`), so raw ranking matters more there than on the Haiku path.
- **Dead trace-vocabulary prune** — 15 dead `ref_types` across 6 buckets in
  `REF_TYPES`. No functional cost (a dead whitelist entry merely permits a write nobody
  makes), so this is tidiness. **Must be its own review, never a footnote in an additive
  commit** — six entries are pinned by assertions in `test_trace_system.py` and
  `recall_quality_signal` has a live dashboard reader, so the prune means deleting
  contract-pinning test assertions. `scout_input`/`scout_findings` **stay** (operator
  ruled: never-written ≠ write-only — brain `57d30c1d`).
- **LAF eval substrate rebuild** — `eval/laf/walker/cross_check.py` refuses to run
  (artifacts stamped `None` vs expected pipeline versions); `verify_substrate.py` T5
  fails on corpus growth; T4's real defect is a stale premise (under `laf_v1` the
  reported `embedding_similarity` *is* the LAF field score, not a raw cosine — fix T4 to
  compare the raw channel). No embedder drift — that warning is spurious. Deferred until
  LAF experimentation resumes.
- **S2 Healer temporal enrichment** — dangling-anchor resolution, implicit sequence
  edges, date propagation, cross-session temporal consolidation. Clean architectural
  slot: same idle cycle, same Haiku + `revise()` machinery.
- **Conversation-time backdating** — the helper exists (`clock.conversation_now(at=)`);
  consumers still call `iso_now()`/`iso_cutoff()` bare. Plus a `recall(query, as_of=)`
  parameter for replay. Operator called this "very important for Evals".

---

## Long tail — not verified this sweep

Small items carried forward. **Status unchecked as of 2026-08-13** — verify against
code before picking one up; roughly 40% of items in this band have historically turned
out already done.

- `s2_vector_healer` unit — repair stale vectors that escaped `revise()` invalidation
- Encoder activation visibility into S1R — encoder revises blind to which fields fired
- Consolidate three `render_rich_node` configs (HAIKU_FORMAT / SURFACE_FORMAT /
  S1_NODE_CONFIG) into one family with named modes
- `build_node_catalog()` regex-extracts node IDs from rendered text — track surfaced
  IDs in traces instead
- Encoder uses two tool families where one `brain_batch` would do
- Agentic surface trace observability — `tool_call` / `tool_round` / `surface_variant`
  ref_types; today the agentic loop is opaque
- `PostToolUseFailure` → failure-memory recall (the brain holds the lesson; nobody asks)
- `SubagentStart` → brain context injection (subagents spawn brain-blind and repeat
  corrected mistakes)
- Dashboard Frame view — display any session's Frame as observability
- **Fresh-Claude vs Anchor calibration test** — spawn a fresh session with the brain
  skill loaded, run identical wakeup probes ("Who am I working with? What's open? Where
  were we?"), compare against fresh Claude *without* the brain. The delta is what the
  brain buys at the wakeup moment, and it's the only empirical path to validating
  SKILL.md / boot changes — today the operator is the only sensor. Referenced from
  [RECALL-OVERVIEW.md](RECALL-OVERVIEW.md)
- `eval_runner.py` bypasses the enrichment scoring step production uses, so scoring
  improvements are invisible to that eval
- `brain_dashboard.db` write removal — pending the dashboard's actual deprecation
- Historical `co_accessed` trim; empty-description generic-edge archive sweep (the
  reclassifier can't fix these — no description to read)
- Query-aware KV field promotion in render (temporal query promotes `event_time`, "what
  did X say" promotes `user_raw_quote`)
- Dispatcher enforcement for mandatory metadata fields — generative encoder rules run
  ~0–20% compliance vs 100% for restraint rules; prompt iteration has a visible ceiling
- UTC-internal clock refactor — required only if the daemon ever leaves the operator's
  machine
- `get_node_lineage(node_id)` — encoder read API wish: creation chain + revision chains
  + related traces in one call
- Wider quote-fidelity audit (200 nodes); quote_fidelity substring validation (needs
  conversation context threaded into `remember()`)
- Latency: lighter candidate format; candidate-count A/B; the `surface_haiku` floor is
  **not** a single call — it's a 2-round agentic loop at 8–10s with **no `cache_control`
  anywhere** in the surface path (min cacheable prefix 4096 tokens; caching round-1's
  prefix is the reliable win)
- Identity-architecture gaps (identity-eval scaffolding, partner-minting flow,
  identity-filter query, self-narrative generation, damage resilience) — full table and
  the eight open research questions in
  [IDENTITY-RESEARCH-2026-05-24.md](IDENTITY-RESEARCH-2026-05-24.md)
- Write-path autocommit (Option B) — defence-in-depth only; the root fix shipped, so
  there's no urgency and it's benchmark-gated

---

## Struck this sweep — verified done or superseded, do not re-add

- **Phrase-anchored title boost** → shipped and evolved past the item: `BRAIN_TITLE_BOOST`
  defaults to `idf2`, and under `laf_v1` it's a gain-weighted lane inside the field.
- **Posture / recency-intent detection (as written)** → obsolete. It was specified as a
  conditional gate on `unified_score` in `recall_scoring.py`; **neither the function nor
  the module exists any more.** The live descendant is the cue-side temporal lane (Now #8).
- **Agentic Haiku-first recall, 7-tool `fetch_batch`** → shipped in substance under a
  different name: `scales/s1/fetch_tools.py` (`recall_topical`, `recall_by_time`,
  `recall_verbatim`, `recall_by_aspect`, `expand_node`) driving the 2-round agentic
  surface loop. `fetch_batch.py` was never created and shouldn't be.
- **Wire Frame into S1 Scribe** → **decided against**, not pending. The encoder stays a
  per-session view (its own arc + journal); brain-wide Frame is for Anchor's recall. The
  encoder needs distance, not more context (brain `eaf833c5`).
- **Kill-or-keep spread activation** → resolved in practice: removed from the recall
  path, retained in `surface.py` as post-selection expansion. De-facto matches the lean.
- **Edge selection called twice per recall** → gone. The two surviving `select_edges`
  call sites act on different objects (per-candidate connections in `daemon_hooks`
  vs `arc_node` in `surface_contract`); the same-node double call is not there.
- **`spread_seed_no_vectors` archived-node race** → answered differently and better:
  every producer is gated at source, with a tripwire in `fetch_tools` that logs an error
  *every* time an archived id slips through and drops it. No grace period needed.
- **`connect_to` ID-shape in the encoder prompt** → shipped 2026-08-11 (s1e v34, ids ride
  the `title` slot, hex-prefix resolver). Gate: 20-item A/B, `id_ok` 0→37. The separate
  `{id: ...}` key idea is retired.
- **`keywords` API surface cleanup** and **contract-test line-pin cleanup** → done and
  re-verified in code.
- **The v20-era open questions (Q1–Q7) and the 10 evaluator contract refinements** →
  written against s1e v20 and quality-contract v1; the encoder is now at v34 and the
  contract at v3+. Not carried forward. Ground truth is
  `servers/scales/s1/quality_contract.py`; one of them (canonical-beats-§7.6 in Sonnet's
  attention) has a standing answer in brain `03781b86`.
- **The session-capture log, the "Completed since" table, and the episodic-references
  ✅ status block** → removed. That's history: git holds the commits, the brain holds the
  meaning, and keeping them here made a 764-line file where ~30 lines were live.

---

## How to update this doc

Add to *Now* only what you verified against code that day, with the brain node id.
When you finish something, move it to *Struck* with one line saying what replaced it —
future readers need to know it was answered, not forgotten. When *Struck* grows past a
screen, delete its oldest half; git remembers. Never let this file grow a history
section again.
