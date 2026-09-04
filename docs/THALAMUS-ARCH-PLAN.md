# Thalamus — architecture review plan (2026-08-29)

## §2026-09-03 — Turns & voices SUBSTRATE built (dial-gated, zero exposure) ◀ ACTIVE ARC

Tom's ruling: a Stop-block continuation IS a new turn whose incoming side is
the delivered message — "informative in the form of brain msg or other
stream msg." This section builds the structural half; the ENCODER half
(teaching the prompt the correspondent elements, then flipping) belongs to
the encoder stream's session and is deliberately not touched here.

**Built, all riding existing substrate** (no new tables, no new write paths):
- `deliver()` stamps each source's rendered block into its trace content
  (capped loudly in `_s0_trace`, the one cap site) with the moment as the
  trace's `ref_id`, and returns `(composite, traced_ref_types)` — traced
  sources only;
- the dial (`trace_contract.S0_CONVERSATIONAL_INCOMING`) gains
  `thalamus_delivery: False` — the brain is a first-class, flippable
  correspondent; the comment states what the dial actually governs
  (encoder timeline + embed lockstep + the continuation stamp; NEVER the
  Scribe cadence, hardcoded to `user_message` in
  `conversational_turns_since`, and NEVER presence/episodes — see round 2);
- `SessionContext.last_delivery_stop` (persisted like `last_recall_stop`):
  the Stop hook stamps it when a dial-on, traced correspondent blocks, and
  `post_response_common` classifies that stop as the delivery turn's
  REACTION — a real `assistant_message` — then consumes the stamp;
  dial-off (today) never stamps, so behavior is bit-identical;
- `get_session_turns` AND `get_conversation` rows carry `ref_type` — the
  correspondent axis (operator / stream / brain) for the encoder session
  to render;
- **chain model settled by the new-turn ruling**: the delivery K OPENS the
  successor chain and the reaction completes it there — incoming K +
  assistant delta on one chain, the same shape as an operator turn. This
  supersedes §2026-09-01 delta #5 (the `increment=False` plumbing was
  removed; `post_response_common` increments unconditionally again).

**Pre-merge review round 2 (5 opus finders) falsified the substrate's
central claim and drove a fix pass.** The dial's derived tuple also fed
presence (focus/recency/recent_msgs), `recall_episodes`' default lens, the
LAF trace matrix, and the dual-store trace chain — a flip would have made
roster focus read "delivered 1 notice at stop" and let deliveries outrank
work. **Tom's ruling (Option A): scopes split.** New
`OPERATOR_DIALOGUE_REF_TYPES` (user/assistant, deliberately not
dial-derived) now pins presence, episodes default, LAF, and the trace
chain; only `get_session_turns`, the embed lockstep (SAID_AND_DID), and
the continuation stamp ride the dial. `recall_episodes`' MCP description
states the contract (default = operator dialogue; deliveries opt-in via
ref_type — the tool_result convention) — it lives in brain_mcp.py and
reached sessions with the 5.2 redeploy (2026-09-04). Other round-2 fixes: `ref_type` now flows through
`get_conversation` (it was dropped by the row whitelist — the deliverable
was inert for the actual encoder path); the continuation stamp is gated at
the hook over TRACED sources only and CONSUMED the moment its stop passes
(no stale-match days later); the gate binds to the import-frozen tuple so
both dial halves flip together; `_s0_trace` gained the one `content=` cap
site (three per-site spellings collapsed); delivery traces carry
the moment marker (boot prelude vs Stop turn — the boot-ask/first-prompt
chain collision is the encoder session's render decision, marker
provided; round 3 moved it to `ref_id`); `last_delivery_refs` collapsed
away;
tests patch the frozen tuple instead of mutating the dial dict (the old
restore-to-literal would have rewritten a flipped production default
mid-suite).

**Round-3 review (8 opus finders, high effort, recall-biased) — 10 verified
findings, 9 fixed, 1 ruled to this checklist.** The fixes: the stamp is
armed inside the delivery try via `trace_contract.arms_continuation`
(named predicate over the dial dict) with a wall-clock freshness window
(`DELIVERY_REACTION_WINDOW_MIN` — an ESC'd continuation fires no Stop, so
counter and stamp freeze; the window stops a later wakeup claiming it) and
disarmed on boot/resume (`reset_session_activity` — a continuation cannot
survive a boot); classification is a 2-line read-and-clear (the prior
consume block was provably dead); the content cap is LOUD
(`cap_text_loud` in `_s0_trace`, marker names the dropped count; the
assistant-side pre-cap deleted); the moment rides `ref_id` (indexed —
`query_traces(ref_type=…, ref_id='boot')`), not an unregistered metadata
key; the synthetic 'reacted with actions only' phrase is gone (a summary
fallback becomes turn CONTENT downstream — no fabricated speech in the
dialogue lane); three missed dial-riders pinned (LAF's moment stack
filters rows to operator dialogue, `scribe_due`'s five-plus gate keys on
ref_type, `eval/laf/composition_probe` repointed so the gating eval
measures production's corpus); `ref_type` flows through
`_conversation_by_session` too and is shape-tested at the DAL; Option A is
now guarded (operator ⊆ conversational subset test; the episodes-default
test rewritten — it asserted the superseded dial contract and would have
argued for undoing the split — and extended to prove a delivery stays out
of the default even flipped-on, reachable only by explicit ref_type);
`recall_episodes`' top-level blurb aligned with its parameter text.

FLIP-DAY CHECKLIST (the encoder session's, beyond the prompt):
- **BLOCKER — `encode._lived_turns` grouping** (encode.py:1168): a flipped
  delivery episode matches no branch, so it is dropped AND the following
  reaction `assistant_message` OVERWRITES the previous operator turn's real
  reply — a fabricated exchange; `_window_n_turns` (counts role=='user')
  simultaneously diverges from `_lived_turns`' turn count. Both must learn
  the correspondent rows before any flip. (Deliberately untouched here —
  encoder files belong to the encoder stream.)
- **RULED to checklist (Tom, 2026-09-03): reaction rows in pinned scopes.**
  On flip, reaction `assistant_message` rows enter presence focus/recency,
  the episodes default (as orphan replies), LAF and the dual-store — the
  rows cannot exist until the flip, so the filter design (summary-marker à
  la WAKE_ENVELOPE, or accept) is the flip session's call.
- teach `embed_queue._render_trace_for_embedding` (and
  `trace_contract.render_trace`) speaker branches for self_message /
  thalamus_delivery — today both fall through to a ref_type literal,
  splitting the dialogue embedding neighborhood;
- decide the boot-prelude render from the delivery trace's `ref_id`
  ('boot' vs 'stop');
- `encode.py:387`-area docstring still names the pre-ref_type
  get_conversation shape (encoder file — same boundary);
- then flip the dial row + restart, behind `s1_encode_eval`.

NEXT-ARC THREAD (Tom, 2026-09-03, from a live encoder prompt): **the fourth
correspondent — the HARNESS.** Claude Code's own injections
(system-reminders, worktree notices, mid-turn interjection envelopes)
arrive INSIDE user_message content, so the timeline attributes machine
prose to the operator ("the harness speaks through Tom's mouth").
`is_machine_turn` covers only whole-turn wake envelopes; a system-reminder
glued to a real prompt is fused. Substrate shape when picked up: split
harness blocks out of the user content at the trace boundary (hook_recall)
into their own s0 K row — a fourth incoming ref_type on the dial
(dial-off like the others) — so the encoder can render a system voice
distinct from the operator. Render half = encoder stream; split half =
this arc's successor. Not built here (scope: turns-and-voices substrate).

PROMPT MOMENT + FIRST ASSIST (Tom, 2026-09-04, from the 12h clock incident,
id:8ece8811): **the Thalamus injects at the user prompt** — a third delivery
moment, passive, ahead of the recall surface — and its first rider is a
computed *assist*, not a queued item: the clock re-anchor, rendered only when
the session's last assistant turn is older than the live window. Tom's
ruling: "It's absolutely the right location, but it means the thalamus needs
to also support injecting on user prompt … add it to the Thalamus thread as
the first context assist injection mechanism and alert." Spec: Step 12.
Design prose: THALAMUS-DESIGN.md §Delivery (Prompt, Assists) + Phase 2.5.
Brain: id:3ed231ee (design), the ruling node linked from it.

Exposure: NOTHING reaches the model or the encoder until a dial row flips —
the flip is the encoder session's move, after the prompt is taught and
eval-gated (`s1_encode_eval`; attribution/containment must be measured).

---

## §2026-09-01 — Step 8 SHIPPED as channels/delivery.py; queue resumed at Step 10

**Step 8 shipped** per rulings id:7c7e805c (Thalamus owns NO transport) and
id:bb0513ae (the msgs layer is the last mile; sources opt into moments by
guarantee): `servers/channels/delivery.py` owns the leg both hooks ride —
`Moment` (BOOT passive / STOP forcing), `Source` (courier / thalamus),
`serves(source, moment) ⇔ moment.forcing ∨ source.survives_a_miss`, and
`deliver()` (walk, failure-isolate, trace per source, join, WARN >5000 —
ruling id:1e22a2f0, folding Step 10(a)'s composed-reason ownership in). The
hooks' inlined two-source blocks are gone; `brain_voice` no longer imports
thalamus; `_s0_trace` moved to `brain_traces.py` (the traces door — the
AST guardrail resolves it by call name, so the move is invisible to it).
Moment vocabulary moved to `delivery.py`; `thalamus_contract` derives its
names from it (the reverse import would close the package cycle — delivery
imports the channel packages call-time inside the render adapters).

**Pre-merge review (8 finder angles, all findings verified) drove a fix
pass**; the declared behavior deltas after it:
1. the boot leg writes the s0 K `thalamus_delivery` trace (was 96%
   untraced; asks were 100% invisible to `query_traces`);
2. the composed leg WARNS above 5000 chars — `_log_warning`, not an error;
3. trace summary wording is `delivered N <noun> at <moment>` (was
   `… via Stop block`; no consumer parses it — verified);
4. the boot error key `boot_thalamus_failed` is replaced by per-source
   `<source>_delivery_boot` + leg-level `boot_delivery_failed` (no readers);
5. ~~the Stop delivery trace lands on the chain of the turn it blocked~~
   SUPERSEDED by §2026-09-03: the new-turn ruling reversed this — the
   delivery K deliberately OPENS the successor chain (the delivery turn),
   and the `increment=False` plumbing was removed;
6. boot delivery no longer dies with the Frame: its ctx resolves in its own
   try, so a Frame failure can't silently suppress an ask.
Review ordering fix worth naming: sources ledger/consume inside render, so
`deliver()` keeps the block FIRST and traces after — a trace failure costs
the trace, never an already-recorded delivery. `REF_THALAMUS_DELIVERY` now
lives in `trace_contract.py` (cycle-free, registered in REF_TYPES by
construction); `thalamus_contract` re-exports it. Known residual: the trace
guardrail sees delivery.py's dynamic `ref_type` as an empty triple (vacuous
pass) — the realistic drift class is covered by the import-time guards.

NOTE the trace's honest claim (measured 2026-08-31): the LEDGER already
measures drain-and-answer; the trace buys S0-stream joinability.
THALAMUS-DESIGN.md's delivery section states the final shape. A pre-Step-8
subtraction pass (merge 213759c) removed 4 dead symbols, fixed 3 stale
docstrings, and added two file() door guards: a directed ask rejects
(undeliverable by construction — asks are boot-only and a nameable session
already booted), and expires ≤ when rejects (never becomes due).

**Queue:** Step 10 remainder — (b) only: `compose_block_loud` extracted to
`loud_truncation.py`, both contracts call it ((a)'s join-cap owner landed in
`delivery.py`). Then Step 11 remainder: the `render_boot_v2` comment about
what boot commits (now: ledger rows + the delivery trace), the three
identical boot try/except sections → `_boot_section` helper, and the
sweep-block consolidation (ruling id:23cd4d61: after 7–10).

---

## §2026-08-31 — Steps 7, 9 and 11-partial SHIPPED; queue resumed at Step 8

**Done since the section below:** Steps 7 and 9, each reviewed pre-commit
(the review window is pre-commit — merging to main IS deploying).

- **Step 7 — door polish (b7e364a).** `file()` 139 → 66 lines: validate,
  resolve the grammars, route to `_file_live` / `_file_queued`, both writing
  through one `_insert_item` (one INSERT, one id-minting site, KeyError on a
  missing column rather than a silent NULL). ONE envelope module-wide —
  `{'ok', 'id'}`, with `filed` a compatibility alias **for one release**,
  pinned by a test that it never disagrees. `source` validated against the
  `category:process` grammar via a new `contract.validate_encoding_source`
  — the convention had been declared in contract.py's field spec and
  enforced nowhere. Shape only; the category set stays open so a Phase 2
  producer needs no registration.
- **Two pre-existing defects the review surfaced (cd36029).** The
  `thalamus_items` DDL still defaulted `audience` to the pre-rename `'once'`
  (v3 renamed rows, not the column default) — a value outside the closed set,
  so any insert omitting the column would never deliver and would die
  silently. Latent only because `_insert_item` names every column. Existing
  installs keep the old stored default; SQLite cannot alter one without a
  table rebuild, and the single-writer guard makes it unreachable. And the
  dedup change-gate compared three fields, so a re-file that flipped
  `needs_answer` was read as identical — a note could not be escalated into
  an ask. It now compares (and sets) all six producer-controlled delivery
  attributes: a re-file leaves the row in the state the producer described,
  or it is a no-op.
- **Step 9 — one relative-time grammar (c746023).**
  `clock.resolve_offset(value, *, direction)`; each caller keeps its own
  empty-value convention, `'now'`, and error subject. The two copies had
  already drifted and one was WRONG: an offset-bearing literal kept its
  offset in the trace bound, so it sorted by text rather than by the instant
  it named and silently shifted the window. Review also found that a large
  shorthand raised **OverflowError**, which both doors' `except ValueError`
  missed — it escaped as an opaque failure. Fixed, and then the fix's own
  message had to be recomposed: the shared function raises a bare REASON,
  each caller supplies the SUBJECT, or a valid-but-out-of-range shorthand
  gets told it "is not shorthand".

- **Step 11's ready items — DONE (e0f1ea8).** `_ITEM_COLS` is now the single
  tuple of column names; `_ITEM_SELECT` joins it, `_INSERT_COLS` filters
  `armed_epoch` out of it, `_row_to_item` zips it. The hazard removed: rows
  were mapped by numeric position against a column list that, being a string,
  nothing could validate against — a column added mid-list shifted every
  field after it into the wrong key. Position is now irrelevant, and the
  SELECT cannot diverge from the mapping by construction. The package
  `__init__.py` states what the package IS — no integrate loop; a delivery is
  traced as an s0 K event beside `self_message`, so it rides the S0 loop as
  an incoming correspondent. (Its "why not under `scales/`" half was retired
  when the package moved — that argument now lives one level up, in
  `servers/channels/__init__.py`.) NOTE the plan's earlier framing was wrong — the
  LATERAL-SCALES "prove it's not just async S0" burden belongs to the
  `operator` channel (async Anchor↔Tom, deprioritized), not to this package,
  whose recipient is Anchor.

**Step 8 is now BIGGER and changed subsystems** (operator ruling
id:7c7e805c): Thalamus owns NO transport. It must not deliver by itself,
only leverage messaging — so boot delivery requires giving the MESSAGING
system a boot leg FIRST (a self-channel change), after which Thalamus rides
it and retires its own direct boot pull; the delivery report then comes
along for free. The June failure does not condemn this: boot reaches the
model through `additionalContext` and demonstrably lands — what failed was
`additionalContext` **mid-thread at PreToolUse**, competing with recall.

**Step 10 narrowed** (ruling id:1e22a2f0): the composed Stop reason WARNS
above 5000 chars — no cap yet. Each source already caps itself at 4000, so
4000 would fire on one legitimate full block and ~8000 is both full; 5000 is
where more than one channel is contributing meaningfully. To the errors
table, the channel that has a reader. The `compose_block_loud` extraction is
untouched by that ruling and stands as planned.

**Both packages MOVED out of `scales/`** (operator-approved, ahead of Step 8):
`servers/scales/{self_channel,thalamus}` → `servers/channels/`. Neither runs an
`integrate(O, K) → Δ` loop — `thalamus/__init__.py` opened by saying so;
`self_channel/__init__.py` makes the adjacent claim that there is no `self`
trace scale.

The load-bearing reason, stated correctly (an earlier draft of this paragraph
asserted a causality git disproves): `test_clock_contract_sync`'s zone was
ENUMERATED (`scales/s1`, `scales/s2`) simply because that was its original
scope — the list landed 2026-05-13, sixteen days before `self_channel` and
three months before `thalamus`, and was never revisited when they arrived. So
the two packages were not carved out deliberately; they sat in the guardrail's
BLIND SPOT, and `presence.py`'s live `datetime.now()` went unscanned for three
months. The move closed that hole rather than working around an accommodation.
With them out the zone can be the prefix `servers/scales`, so a future s3/s4 is
covered by this contract automatically — **and by its twin only because
`test_time_window_contract`'s `CTX_PROTECTED_DIRS` was collapsed to the same
prefix alongside it.** Step 8 therefore builds `delivery.py` into
`servers/channels/`, not `scales/`.

**Queue:** Step 8 (fresh session — new subsystem, and the biggest of the
three), then Step 10, then what remains of Step 11 — its boot-renderer
comment WAITS for Step 8 (what boot commits changes when the boot leg moves)
and its sweep-block consolidation waits for 7–10 per ruling id:23cd4d61.
Nothing else in the plan is open.

---

## §2026-08-30 (late) — canonical-pull arc SHIPPED end-to-end; Steps 7/9 now done above

**Read first:** ship milestones id:0cbc1e53 (canonical pull + TTL) and
id:b0940238 (review fix pass); ruling id:d42a49ce.

Steps 1–6 + two review fix rounds live (merge 9766ca7, logs schema v2).
The archived-refs gate closed by WIDENING and the whole widened arc is
now shipped, reviewed, and consolidated — four merges on main:

- **Canonical-pull redirect (2f0b54b):** absorbed ids resolve at the door
  (get_node / get_nodes / filter_nodes-by-id / recall_node) — survivor
  under the requested id, `_redirected_from` marked in every render;
  retired nodes render `⚠ ARCHIVED`; writes refuse with the pointer;
  thalamus `_attach_ref_lines` inherits redirect + titles for free
  (`old ↦ new · Title (absorbed)`). Consumer trace id:6686c3a5.
- **8-angle Opus review → hotfix (9f769b4):** endpoint-dedup regression
  fixed; the TTL reaper's rotation wire REMOVED before it fired (the
  tagged-bak corpus is the recovery scripts' data source).
- **Fix pass (2bed982):** one owner per redirect concern —
  contract.py holds REDIRECTED_FROM_KEY + both markers + the shared
  write-refusal; canonicalize_results owns the identity swap;
  copy-on-stamp (no false banners); zero-extra-queries when live at
  every door; `test_redirect_door_parity` pins the three doors.

**Gate ledger — ALL FOUR CLOSED:** (1) archived refs — shipped (above).
(2) backup policy (id:54160417) — `reap_by_ttl` + rotation wire live
(14d TTL, hardened pattern); corpus ruling landed (the v15–v29 brain.db
baks reap; their recovery-script consumers are retirement candidates,
flagged not deleted).
(3) sweep-block consolidation (id:23cd4d61) — folds into Step 11.
(4) audience guard — CLOSED via the recipient-set rename (logs v3,
merge b2adb5e): audiences renamed `first_session` / `every_session`
(the enum picks WHO is eligible; cardinality is the ledger's, repetition
is the epoch's), the contract test (`resolve_for_whom outputs ⊆
AUDIENCES` + predicate binds both) shipped with it, the runtime tripwire
in `file()` stays for future-new-branch drift.

**Locked:** append-only epoch ledger (re-arm = generation); change-gated
re-file (identical re-file refreshes window only, never re-delivers);
kept-count rendering (head/tail/ledger/count agree); sweeps ahead of the
S2 gate.

**Next builds:** Step 7 (door polish — before Phase 2 producers; the
audience contract test already landed with the v3 rename), then 8–11
below. Adjacent, non-thalamus follow-ons spun off as their own tasks:
LAF survivor-credit for pick/enc lanes (finding id:ad74941e, chip armed —
algorithmic pre/post eval, no Haiku); recovery-script retirement
(bak corpus reaped, scripts empty — awaiting the beat).

**Do not reopen:** ThalamusDAL; producer-facing kind vocabulary; sweep
placement (Step 3's directive); delete-on-defer; unconditional dedup bump.

**Scope.** Five-angle Opus review (placement, unification-across-callers, cohesion,
coupling, altitude) of the freshly shipped Phase 1 Thalamus: `servers/scales/thalamus/`,
`dispatch_thalamus.py`, and its seams (Stop hook, boot render, idle maintenance, schema,
trace contract, MCP surface). Boundary fully traced (the subsystem is one day old; the
reviewing session authored it); brain history supplied 7 settled constraints and the
Phase 2/3 in-flight scope as false-positive filters. The coupling reviewer verified the
write-connection claim **empirically** on the bundled SQLite rather than asserting it.

**Headline verdicts.**
- The package is **cohesive** — no file splits. `dispatch_thalamus.py` and every seam
  placement were independently cleared by multiple angles.
- **Module-owned SQL stays** (the signal.py courier pattern, now 3 instances). A
  `ThalamusDAL` was weighed and rejected: `pull()`'s predicate IS delivery policy —
  moving it puts business logic in the DAL or forces N+1 reads. The real protection is
  the bound-cursor guardrail (Step 2). Revisit trigger: a fourth logs-backed queue →
  shared `_LogsWriteBase` plumbing.
- Five reviewer findings are **correctness defects in the day-old code**, not
  architecture — they cluster in Steps 1–3 and should land before anything else.

**Dependency summary.** Execute in order 1 → 2 → 3, then 4–11 are largely independent:
- **1** defect batch (thalamus.py) → **2** guardrail that locks 1 → **3** sweep relocation.
- **5** (ledger epoch) and **6** (contract vocabulary) both touch `pull()`'s SQL — run 5
  before 6; **6 before 7** (the door refactor uses 6's constants).
- **4, 8, 9, 10, 11** independent (8 and 10 both touch `daemon_hooks.py` lightly — fine
  in either order, rebase-aware).

**Dropped findings** (and why — the brain's constraints earned their keep):
- Stop-hook two-source abstraction: dropped. Per-source failure isolation is the point;
  an N=2 abstraction must smuggle ref_type/summary/trace-when-n through callbacks. Only
  the composed cap survives (Step 10).
- Unifying the two ledgers / TTL config knobs / the terminal `sent` row / a route
  dispatch table / a kind table / collapsing the audience predicate: all explicitly
  ruled leave-as-is by their reviewing angle.
- `assemble_boot` extraction from `BrainVoice`: deferred to Phase 2's
  standing-items-retirement commit, which already edits that function.
- `is_session_id` relocation out of `self_contract`: noted; trigger is a third consumer
  (neutral home then: `servers/contract.py`).

---

## Step 1 — Fix the five correctness defects in `thalamus.py`

**Problem.** Review found five defects, all verified against the code (one empirically):
1. **Write-conn cursor invariant** (`thalamus.py:121-126`, `:138-141`): `file()` binds
   SELECT cursors on `logs_conn_w` before writing on it — the exact SQLITE_BUSY_SNAPSHOT
   shape `dal_logs.py:19-32` forbids. Reproduced: with **two** rows matching the dedup
   SELECT, the following write fails instantly with `database is locked` (busy_timeout
   does not apply). Nothing prevents two open `(source, dedup_key)` rows — no unique
   index, only single-process discipline.
2. **Expiry drops the `deliver_at` anchor** (`_default_expires`, `:47-59`): ask and
   audience-`all` branches anchor expiry to *now*, so `remind(needs_answer=True,
   when='3w')` expires (now+14d) before it is due (now+21d) — the item can never
   deliver, then fires a **false loud dead-letter**. `resolve()`'s defer branch already
   composes anchor+grace correctly; the door is the odd one out.
3. **Live route silently discards params** (`:93-117`): `for_whom='live'` ignores
   `when`, `needs_answer`, `dedup_key` without a word — in the door whose design rule is
   "reject loudly at the write boundary".
4. **Overflow count can never exceed 1** (`:180-198`): `LIMIT PULL_MAX_ITEMS + 1` makes
   `overflow ∈ {0,1}`; the render says "+1 more due" while hiding 15. The existing test
   asserts the substring, not the number.
5. **`answer=''` closes an ask with an empty payload** (`resolve`): `is not None` passes
   the empty string; `file()` rejects an empty body — same guard, missing at one door.
   Also: `file()`'s budget count has no `expires_at` filter, so expired-but-unswept items
   wedge a producer at its cap while invisible to delivery.

**Target state.** (1) Unbind both cursors (`conn.execute(...).fetchone()`, matching
`_get_item`/`withdraw` in the same file) + `LIMIT 1` on the dedup SELECT. (2) Compose
expiry as ANCHOR × SPAN: `iso_after(days=SPAN, at=fromisoformat(deliver_at))` when
`deliver_at` is set — one composition, no branch-dependent anchor. (3) The live route
rejects unsupported params loudly (`when`/`needs_answer`/`dedup_key` → `{'filed': False,
'error': ...}`). (4) Hoist the due predicate to a module-level `_due_sql(via)` shared by
`pull()` and a real `COUNT` (the `signal._PENDING_INBOX_SQL` precedent) so the overflow
tail names the true count. (5) Reject empty/whitespace `answer`; budget count gains
`AND expires_at > ?`.

**Files & call sites.** `servers/scales/thalamus/thalamus.py` only, plus
`tests/test_thalamus.py`: new cases for the two-dedup-row race (regression for #1), a
dated ask delivering inside its window (`when='3w'`, `needs_answer=True` → pull succeeds
after due), live+param rejection, overflow asserting the *number*, empty answer, and
budget-with-expired-items.

**Verification.** `./dev pytest tests/test_thalamus.py tests/test_mcp_roundtrip.py -q`.

**Blast radius.** One module + its tests; no seam changes. Daemon restart to deploy.

**Depends on.** None — first, and hotfix-grade.

**Respects.** Synchronous loud budget at file() (settled); loud dead-letter design
(id:dd2ad2e8) — #2 is what makes the dead-letter honest.

---

## Step 2 — Guardrail: no bound SELECT cursors on a write connection

**Problem.** The `dal_logs.py` snapshot invariant ("every statement on wconn fully
consumed; do not bind such a cursor to a name") is documented prose, hand-reimplemented
by three channel modules, and held by nothing. `thalamus.py` violated it on its first
commit; the existing raw-SQL ratchet counts DML only and cannot see this class.

**Target state.** A new assertion in `tests/test_raw_sql_guardrail.py`'s family: flag
`<name> = <write-conn>.execute('SELECT ...')` patterns outside `dal*.py` (write conns:
`logs_conn_w`, and the graph write conn if applicable). ~10 lines, same scan style as
the existing ratchet. Covers signal.py, rejection_table.py, thalamus.py, and every
future channel module.

**Files & call sites.** `tests/test_raw_sql_guardrail.py`.

**Verification.** The new test passes on post-Step-1 code and fails if either Step-1
cursor fix is reverted (check by temporary revert).

**Blast radius.** Test-only.

**Depends on.** Step 1 (the current code would fail the new guardrail).

**Respects.** CLAUDE.md "where a boundary already leaks, the rule is directional" — this
holds the line where it actually bites, instead of relitigating the DAL address.

---

## Step 3 — Move the idle sweeps out from behind the S2 fire gate

**Problem.** `daemon_server._run_idle_maintenance` early-returns when
`brain.run_maintenance_if_due()` declines (idle threshold, min-interval, encode gate,
`llm_available`) — and both sweeps (`signal.reap_expired`, `thalamus.expire_due`) sit
after that return. On a keyless brain they **never run**: the loud dead-letter for
unanswered asks never fires, and (pre-Step-1) expired items wedge producer budgets.
`run_maintenance_if_due` itself documents the correct pattern: `prune_payloads_if_due()`
runs at its top, above the S2 conditions, with a comment naming exactly this hazard.

**Target state.** Both sweeps move beside `prune_payloads_if_due()` at the top of
`Brain.run_maintenance_if_due()` (brain.py), each behind its own cheap time throttle
(e.g. hourly, a `brain_meta` ts like the payload prune uses). The daemon's two
try/except blocks in `_run_idle_maintenance` are deleted — the concern moves to its
owner instead of being abstracted in place.

**Files & call sites.** `servers/brain.py` (`run_maintenance_if_due`),
`servers/daemon_server.py` (`_run_idle_maintenance` — remove both blocks),
`tests/test_thalamus.py` (sweep fires without S2 conditions — call
`run_maintenance_if_due` on a brain with no LLM and assert expiry ran).

**Verification.** `./dev pytest tests/test_thalamus.py -q` + the maintenance/daemon
test files (`-k "maintenance or daemon"`).

**Blast radius.** Maintenance cadence for the courier reaper changes (more reliable, not
less); watch daemon log lines move from `_run_idle_maintenance` to brain.

**Depends on.** Step 1 (budget filter) — independent otherwise.

**Respects.** Loud-by-default; the dead-letter contract (id:dd2ad2e8).

---

## Step 4 — Refs resolve in `pull()`: batched, veil-aware; the contract returns to pure formatting

**Problem.** `thalamus_contract._resolve_refs` calls `brain.get_node(ref)` per ref
inside the render — up to 15 sequential canonical pulls (~60-75 queries + correction
walks) on the boot/Stop critical path, to read titles. It breaks the template's own rule
(`self_contract.py:211` — "the contract FORMATS, never reaching into clock / session
state"), renders differently depending on an optional `brain` param, and is
**veil-blind**: a globally-filed item can ref a walled node, printing its title into
every session's boot. The renderer Thalamus replaces threads `session_id` for exactly
this reason (`frame.py:129-145`). The veil deferral in the design covers *items*;
refs are *nodes* and carry walls — this corrects that premise, not the settled decision.

**Target state.** `pull()` (which holds `brain` **and** `session_id`) resolves all refs
for the block in **one** call: `brain.filter_nodes(field='id', include=all_refs,
rich=False, session_id=session_id, limit=len(all_refs))` — verified skinny, batched,
and veil-aware with default-deny. Attach `ref_lines` to each item dict; `render_item`
formats what it is handed; `render_block`/`render_item` lose the `brain` parameter.
Phase 2's journal-view join then inherits resolved refs for free.

**Files & call sites.** `servers/scales/thalamus/thalamus.py` (`pull`),
`servers/scales/thalamus/thalamus_contract.py` (`_resolve_refs` removed;
`render_item`/`render_block` signatures), `tests/test_thalamus.py` (walled-ref case:
a ref to an isolated node does not render into another session's pull; batched shape).

**Verification.** `./dev pytest tests/test_thalamus.py -q`; render tests updated.

**Blast radius.** Render internals only; MCP surface unchanged.

**Depends on.** None (rebase-aware with Steps 5/6 — same file).

**Respects.** Scope-veil ownership in `scopes.py`/`filter_nodes` (routes through the
existing veil door rather than growing a new one); "answerable without fetch"
(id:70016ed3) — refs still resolve at inject.

---

## Step 5 — Ledger append-only: re-arm is a generation, not a deletion

**Problem.** `resolve(defer_until=…)` does `DELETE FROM thalamus_deliveries WHERE
item_id = ?`. The design's contrast with the courier is "ledger …, **forever**", and
Phase 3 retry "gates on unacked" — the delete destroys exactly that evidence, resets
`list_items` delivery counts, and makes "never delivered" indistinguishable from
"delivered, then deferred". Cheapest to fix while the table is days old.

**Target state.** `armed_epoch INTEGER DEFAULT 0` on `thalamus_items` and on the ledger;
ledger PK becomes `(item_id, session_id, armed_epoch)`; defer increments the item's
epoch (no delete); the pull predicate's `NOT EXISTS` gains
`AND d.armed_epoch = thalamus_items.armed_epoch`. `INSERT OR IGNORE` idempotence per
epoch is preserved exactly.

**Files & call sites.** `servers/schema.py` (both tables — new tables are
`CREATE IF NOT EXISTS`; existing installs need the `_add_column_if_missing` pattern +
ledger PK note: SQLite can't alter a PK, so recreate `thalamus_deliveries` if it has
rows — it is days old), `servers/scales/thalamus/thalamus.py` (`pull` predicate,
`resolve` defer branch, `list_items` count semantics — decide: count current-epoch or
all-time; recommend all-time with per-epoch available), `tests/test_thalamus.py`
(defer preserves history; re-delivery after re-arm; counts).

**Verification.** `./dev pytest tests/test_thalamus.py tests/test_raw_sql_guardrail.py -q`.

**Blast radius.** Schema + one module. Existing deployed ledger has a handful of rows —
recreate is safe; note it in the commit.

**Depends on.** Step 1 (same file; keep diffs sequential). Land before Phase 3 retry.

**Respects.** "Thalamus owns durable delivery state" (id:8a170558); two-state-machines
(id:e63c41dd) — the item row owns re-arm, the ledger stays truthful history.

---

## Step 6 — Contract-first vocabulary: states/audiences/moments in the SQL, windows and kinds in the contract

**Problem.** Three drift classes, all one shape — the contract declares vocabulary the
mechanics inline as literals:
- `pull`/`list_items`/`expire_due` hardcode `'open'`/`'once'`/`'all'` in SQL while the
  same module binds `tc.STATE_OPEN` in write paths eight lines away.
- The delivery-moment vocabulary doesn't exist: `via` is an unvalidated free string
  written to the ledger (`pull(via='bot')` behaves as Stop and ledgers the typo); the
  ask/boot rule is `if via != 'boot'` inside SQL assembly. Unknown `audience` values
  match neither predicate branch — open forever, silent death at expiry.
- Window derivation lives in mechanics, twice, with function-local datetime imports
  (`_default_expires` + `resolve`'s defer branch), while the contract's docstring claims
  to own "caps and default windows". Three unnamed partitions of the item space
  (audience-default, expiry-span, render-verb) disagree — the root of Step 1's defect #2.

**Target state.** In `thalamus_contract.py`: `MOMENTS = (VIA_BOOT, VIA_STOP)`,
`ASK_MOMENTS = (VIA_BOOT,)`; `window_for(needs_answer, audience, deliver_at)` and
`extend_window(new_deliver, current_expires)` beside `resolve_when`; a single
`kind_of(item)` derivation feeding both the render verb and the span lookup (internal
and derived — does NOT reintroduce the rejected kind vocabulary). In `thalamus.py`:
all state/audience literals bound from `tc.*`; `pull` validates `via` against
`tc.MOMENTS` loudly; `file` validates `audience` output the same way; both
function-local datetime imports deleted.

**Files & call sites.** `servers/scales/thalamus/thalamus_contract.py`,
`servers/scales/thalamus/thalamus.py`, seam literals in `servers/daemon_hooks.py`
(`via='stop'`) and `servers/brain_voice.py` (`via='boot'`) → `tc.VIA_*`,
`tests/test_thalamus.py` (unknown via/audience are loud).

**Verification.** `./dev pytest tests/test_thalamus.py tests/test_mcp_roundtrip.py -q`.

**Blast radius.** Rename-safe refactor inside the subsystem; seams touch two lines each.

**Depends on.** Steps 1 and 5 (same SQL; sequential diffs).

**Respects.** Contract-first (CLAUDE.md); the dropped-kind-vocabulary decision — kind_of
is derived, never producer-set.

---

## Step 7 — Door polish before Phase 2 producers: split `file()`, one envelope, validated `source`

**Problem.** Three Phase-2-facing roughnesses at the door: (a) `file()` is ~100 lines
holding two storage stories (live delegation vs queue) with two hand-listed 13-column
INSERTs and duplicated id-minting; Phase 3's machine live-now rewrites exactly the live
branch. (b) Two result-envelope shapes in one module — `file()` → `{'filed': …}`,
`resolve`/`withdraw` → `{'ok': …}` — every agent-toolset caller must learn both.
(c) `source` is the budget key AND the withdraw-ownership key, and it's unvalidated
free text: a typo'd source gets a fresh budget and orphans its own items. The values in
play are exactly the `encoding_source` grammar (`category:process`) the repo already
governs in `servers/contract.py`.

**Target state.** (a) `file()` = validate → resolve grammar → dispatch to `_file_live()`
/ `_file_queued()`, sharing one `_insert_item(conn, **fields)`. (b) One envelope key
across the module (recommend `ok` + `id`, with `filed` kept as an alias for one release
if anything external reads it — nothing does today). (c) `file()` validates `source`
against the `category:process` grammar / a contract-owned producer set, loudly.

**Files & call sites.** `servers/scales/thalamus/thalamus.py`,
`servers/dispatch_thalamus.py` (envelope key), `tests/test_thalamus.py` +
`tests/test_mcp_roundtrip.py`.

**Verification.** `./dev pytest tests/test_thalamus.py tests/test_mcp_roundtrip.py -q`.

**Blast radius.** Module-internal + dispatch result shape (MCP result payload key
changes — call it out in the commit; no persisted data changes).

**Depends on.** Step 6 (uses its constants). Land before Phase 2 wires encoder toolsets.

**Respects.** Three-verb budget (unchanged); one-door design (routing stays inside
`file()`); `encoding_source` grammar ownership in `contract.py`.

---

## Step 8 — Boot deliveries write the s0 trace; reconcile the design doc

**Problem.** The Stop leg writes `thalamus_delivery`; the boot leg deliberately doesn't
("ledger + boot_renders is the record"). But asks are boot-only, so **100% of ask
deliveries are untraced** — against the design's own line ("untraced delivery IS the
visibility problem…"), and Phase 2's stated measurement ("does Anchor drain and
answer") cannot be made from `query_traces(ref_type='thalamus_delivery')`. The doc and
the code currently contradict each other (`docs/THALAMUS-DESIGN.md:104-106` vs
`trace_contract.py:66-68`).

**Target state.** Symmetric tracing: the boot leg in `render_boot_v2` writes the same
s0 K `thalamus_delivery` event (it already builds a `SessionContext` for the Frame, so
the caller-owns-the-chain rule is satisfiable there). Alternative if chain plumbing at
boot proves awkward: move the trace write into `pull()` with a caller-supplied ctx —
but keep ONE owner either way. Update `docs/THALAMUS-DESIGN.md` to state the final
shape. Fix `_s0_trace`'s docstring ("four S0 turn events" — `thalamus_delivery` is the
fifth).

**Files & call sites.** `servers/brain_voice.py` (boot leg), possibly
`servers/channels/thalamus/thalamus.py` (`pull`), `servers/daemon_hooks.py`
(`_s0_trace` docstring), `docs/THALAMUS-DESIGN.md`.

**Verification.** `./dev pytest tests/test_thalamus.py -k trace -q` (new: boot pull
writes the event) + `tests/test_trace_contract_sync.py`.

**Blast radius.** One trace event per boot-with-due-items; observability only.

**Depends on.** None (rebase-aware with Step 10 in `daemon_hooks.py`).

**Respects.** Traces-layer ownership (trace writes via the established `_s0_trace` /
TraceDAL doors; never raw SQL).

---

## Step 9 — One relative-time grammar in `clock.py`

**Problem.** `brain_traces._resolve_time_bound` (past-direction) and
`thalamus_contract.resolve_when` (future-direction) are the same grammar — identical
regex, unit table, ISO fallback, loud-ValueError contract — differing only in sign.
The grammar is a *published* contract, quoted verbatim in three MCP tool descriptions
(`recall_episodes`, `remind`, `thalamus_resolve`); the two error messages have already
diverged. Phase 3 retry backoff would mint a third copy.

**Target state.** `clock.resolve_offset(value, *, direction)` (or `parse_offset`
returning timedelta kwargs) in `servers/clock.py` — the file whose premise is "ONE
function for now" and which already owns the `iso_cutoff`/`iso_after` pair. Both callers
keep their own empty-value convention ('' vs None) and error prefix by catching and
re-raising.

**Files & call sites.** `servers/clock.py`, `servers/brain_traces.py`,
`servers/scales/thalamus/thalamus_contract.py`; tests:
`tests/test_thalamus.py` + whatever covers `recall_episodes` bounds
(`-k "episode or time_bound"` — verify with `--collect-only`).

**Verification.** `./dev pytest tests/ -k "thalamus or episodes or clock or time_window" -q`.

**Blast radius.** Two consumers re-routed; behavior identical by test.

**Depends on.** None.

**Respects.** clock.py as the single source of truth for time formats (CLAUDE.md).

---

## Step 10 — The composed Stop reason gets an owner (and one block composer)

**Problem.** Each Stop source caps itself at 4000 (`RECEIVED_BLOCK_MAX`, `BLOCK_MAX`),
then `daemon_hooks` joins them with **no cap over the join** — a Stop `decision:block`
reason can reach ~8000 chars and nothing owns that number; Phase 2 grows the thalamus
half. Separately, `self_contract.render_received_block` and
`thalamus_contract.render_block` are the same budgeted-block algorithm line-for-line.

**Target state.** (a) One `cap_text_loud` over the joined reason in the Stop composer,
constant named where the channel is composed. (b) `compose_block_loud(head, rendered,
cap, …)` extracted to `servers/loud_truncation.py` (which owns the loud-cap primitives
and states its below-both-domains placement rationale); both contracts call it.

**Files & call sites.** `servers/daemon_hooks.py` (join cap),
`servers/loud_truncation.py`, `servers/channels/self_channel/self_contract.py`,
`servers/channels/thalamus/thalamus_contract.py`; tests: existing self-channel truncation
tests + `tests/test_thalamus.py` render cases.

**Verification.** `./dev pytest tests/ -k "thalamus or self_ or truncation" -q`.

**Blast radius.** Render-identical if extracted faithfully; the join cap changes
behavior only for >cap composites (names the overflow instead of silence).

**Respects.** The truncation contract family (one truncation point, always loud —
node 8178593a's standing rule).

**Depends on.** None (rebase-aware with Step 8 in `daemon_hooks.py`).

---

## Step 11 — Docs and small placements batch

**Problem.** Review nits that lie to the next reader:
- ~~`thalamus/__init__.py` is empty~~ — DONE (e0f1ea8, corrected in cabef0e).
  Both package docstrings and `servers/channels/__init__.py` now carry the
  placement rationale; nothing here remains.
- `render_boot_v2`'s comment still claims "rendering is read-only"; boot now commits
  ledger rows (and health auto-fix already wrote before that). State what boot commits.
  Collapse the three identical try/except sections (Frame / standing / thalamus) into a
  `_boot_section(name, build)` helper while there.
- `hook_pre_edit`'s docstring claims it delivers pending self-messages; the Stop path is
  the only drain caller.
- `_row_to_item`'s positional tuple indexing + `list_items`' `r[:-1]` — switch to
  `sqlite3.Row`/named access or a single column-map, so a mid-list column add can't
  silently shift fields (Step 5 adds a column; do this alongside or right after).

**Files & call sites.** `servers/channels/thalamus/__init__.py`, `servers/brain_voice.py`,
`servers/daemon_hooks.py`, `servers/channels/thalamus/thalamus.py`.

**Verification.** `./dev pytest tests/test_thalamus.py -q` + boot renders in a live
session after deploy.

**Blast radius.** Comments/docstrings + one mechanical refactor.

**Depends on.** Best after Step 5 (row shape).

**Respects.** Docs-current-state-only rule; "comments carry the why".

---

## Step 12 — The Prompt moment and the first assist (clock re-anchor)

**Problem.** The entity has no clock: its "now" is the newest timestamp in
context, and nothing enters the context while wall time passes between turns.
Across an idle gap (operator away, `--resume`, compaction) the anchor goes stale
silently — on 2026-09-04 a 12-hour gap turned a correct `remind(when='12h')`
result into a suspected parser bug (id:8ece8811). Nothing that rides the prompt
carries a clock; a stamp on every prompt would be tuned out (the harness's own
date-change line was). The Thalamus has no prompt moment, and no way to say
something that is *computed* rather than *filed*.

**Target state.**
- `delivery.PROMPT = Moment('prompt', forcing=False)`; `MOMENTS = (BOOT, PROMPT,
  STOP)`. `thalamus_contract` derives `VIA_PROMPT`; `tc.MOMENTS` gains it
  (`pull` raises on an unknown `via`).
- **Assists** in the Thalamus: `tc.ASSISTS = {'clock_reanchor': (VIA_PROMPT,)}`
  — name → moments it may speak at. `pull(brain, session_id, via)` evaluates the
  assists registered for `via` after the queued items and folds their lines into
  the same block and the same `(block, n)` count, so `deliver()` keeps and
  traces it as one `thalamus_delivery` K with `ref_id='prompt'`. No row, no
  ledger. Queued kinds yield nothing at `prompt` — `_due_filter` is unchanged
  and a test pins `pull(via='prompt')` → assists only, with an open notice and an
  open ask in the table.
- **Clock re-anchor** (`thalamus.py`, first assist): anchor = the session's
  newest `assistant_message` trace — `dal_logs.session_activity` grows one
  aggregate `last_assistant_at` (+ `brain_traces` passthrough; presence/peek
  get "last spoke at" for free). None → silent (first prompt; boot Frame just
  stamped Now). Age ≤ `ROSTER_LIVE_WINDOW_MIN` (read from `self_contract` — a
  contract constant; hoist to a channels-level contract only if a second
  consumer appears) → silent. Else one line:
  `⏱ 12h 11m since your last turn — now 2026-09-04 14:52 UTC (Friday). Re-anchor
  before reasoning about time: streams, queue, repo may have moved.` Wall-clock,
  like every Thalamus timestamp (id:2c491848).
- **`hook_recall`** calls `deliver(brain, ctx, PROMPT)` once and prepends the
  block on all three return sites: surface produced context; surface produced
  nothing (today a bare `approve` — becomes `additionalContext=block`); the
  `register_only` short-answer fast path ("ok" after 12 hours is the case).
  `_traced` is ignored — the Stop-side continuation stamp is untouched. COURIER
  declines the passive moment by the existing predicate.
- **Replay guard.** `eval/frame_replay.py` references `hook_recall`; if any eval
  drives it end to end, wall-clock minus a historical stamp fires the assist on
  every replayed prompt — skip the PROMPT leg when a replay clock is injected
  (`conversation_now(brain)` off wall-clock by more than a minute), presence's
  existing exemption precedent.
- **Door echo.** `thalamus.file()` and the resolve path return `now` beside the
  deadline they resolved; `remind` / `thalamus_resolve` MCP results carry it.
  Result-shape change → `eval/mcp_batch_probe.py` + `eval/mcp_schema_gate.py`
  before restart.
- **Dial.** `thalamus_delivery` is already dial-off; the assist inherits that.
  When the dial flips, decide whether assists should enter the encoder timeline
  as brain speech or be marked out — name it on the flip-day checklist, do not
  decide here.

**Files & call sites.** `servers/channels/delivery.py`;
`servers/channels/thalamus/thalamus_contract.py`, `thalamus.py`;
`servers/dal_logs.py` (`session_activity`), `servers/brain_traces.py`;
`servers/daemon_hooks.py` (`hook_recall`, three return sites); `brain_mcp.py`
(result passthrough only if the door shape needs declaring). Tests:
`test_delivery.py` (moments, `serves`, empty-brain `deliver(PROMPT)` →
`('', ())`), `test_thalamus.py` (pull at prompt → assists only; re-anchor at
29/31 min; `now` in results), `test_self_presence.py` (`last_assistant_at`;
heartbeat tail does not move it), `test_daemon_hooks.py` (stamp precedes "Brain
activated"; both bare-approve paths return context when stamped),
`test_trace_contract_sync` / `test_clock_contract_sync` /
`test_time_window_contract` green (nothing under `scales/` changes).

**Verification.** `./dev pytest tests/ -k "thalamus or delivery or self_presence
or daemon_hooks or trace_contract or clock_contract or time_window" -q`, tier
checked with `--collect-only | grep`. Then live: idle a session past 30 min,
prompt, see the line first in the injected context; `query_traces(ref_type=
'thalamus_delivery', ref_id='prompt')` lists it. No "was it acted on" gate:
the effect wanted is awareness, which is what injected context produces (Tom's
two-channels ruling in THALAMUS-DESIGN.md §Delivery).

**Blast radius.** A new moment walked by two sources: COURIER declines, the
Thalamus renders assists only. Boot and Stop renders are bit-identical (no
assist registers for them). One added aggregate in `session_activity`. The
`register_only` path gains one cheap leg (two small queries).

**Respects.** Moment vocabulary lives in `delivery.py`; moment-as-`ref_id`; the
eligibility predicate (id:7c7e805c, id:bb0513ae) untouched; the Thalamus owns no
transport; `servers/scales/` gains no real-elapsed clock; wall-clock Thalamus
timestamps (id:2c491848); the admission test (id:6a11f45f: clock → Thalamus).

**Depends on.** Step 10(b) is independent; rebase-aware with it in
`thalamus_contract.py`. Best after the Step 11 boot-helper collapse only if that
lands first — no hard dependency.

**Named, not included.** Queued kinds riding the Prompt moment (cadence
ruling); `recall_episodes` bound echo (same shape as the door echo); the boot
Frame's conversation-time "Now" (untouched, grain-side).
