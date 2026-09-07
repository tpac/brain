# Thalamus — architecture plan

Current state and the open steps only. Shipped work is a one-line ledger with
its brain node; the nodes hold the reasoning. Design: `docs/THALAMUS-DESIGN.md`.

## §2026-09-05 — Steps 1–11 CLOSED; next build is Step 13, S1 through the Thalamus ◀ ACTIVE ARC

**Read first:** handoff id:49c719af (the letter, with its boot self-test), rulings
id:8aa9f183 and id:3e549ac4, then § Step 13 below.

Tom's rulings this session: the first real Thalamus consumer is the **S1 Scribe
speaking to its own live session at Stop**, not boot ("boot requires something
different"); S2's escalation nodes stay in boot for now; the Scribe's entrance is
the review block it already writes (two addressed verbs), with the `remind` tool
kept for mid-run asks; directed asks deliver at Stop; a filing leaves a trace on
the run chain; and **anything the Thalamus delivers is relayed to a human in
plain words** — the question and its stakes, never an id, a moment, or the
producer (id:3e549ac4).

**Locked:** review-block entrance; render-join feedback, never write-back;
directed asks at Stop; filing traced; budget key = (source, target_session) —
"consumers of thalamus will dictate that" (Tom, id:1dda87c2).
**Re-ruled late 2026-09-05 (Tom): S1 and S2 unite NOW** — "it's a drift. The
instructions should be the same and the delivery should be different through
thalamus." Phase 2b folds into Step 13: ONE review paragraph for every
encoder; the binding routes on `source` alone (a session → directed → Stop;
none → broadcast → boot); the open-×N nudge stops naming a node type and says
`ask · <subject> · <question>`; `journals-escalation` leaves the boot default.
S2 still DELIVERS at boot — through the Thalamus.
**Order change:** (e) feedback precedes the lit paragraph — (b) strips
tell/ask out of the journal, so a lit encoder without (e) forgets it spoke and
re-tells every run. Sequence now: (h) ✓ (c) ✓ (d) ✓ (b) → (e) → (a) dark → (f).
**(e) ruled "let's expose", MINIMAL (Tom, 2026-09-06):** the encoder sees
what it said and the SETTLED outcome only — answered (the text), dismissed,
not delivered (the reason); an open item renders as a bare "open". Never
delivery counts, moments or dates (the Thalamus's bookkeeping — two state
machines). Why expose at all: dedup collapses OPEN items only, so a settled
ask re-asserted next run is a NEW item, re-delivered forever; and for S2 the
answer's only actor is the next S2 run — unexposed, every S2 ask is
fire-and-forget. Why minimal: the encoder's job is its perspective slice, not
managing its mail.
**Ruled 2026-09-06 (Tom, "yes new rule"):** an item's IDENTITY is (source,
dedup_key, target_session) — the budget's triple. Re-filing a key for another
reader is another item, never a retarget (one producer string serves every
session; a retarget would let one session take another's item). Moving an
item = withdraw, then file. `withdraw(dedup_key=…)` takes the target.
**Shipped:** 13(h) b573f76, 13(c) 2c7acc2, 13(d) 963e5c6, 13(b) merge 691bac4
(unified S1+S2; simplify + 8-angle review applied; id:9b83b70b), 13(e) aa2e6aa
(merge 3d85619; producer view + producer_items door; id:b0c1ccdd).
**(a) built dark:** `trace_contract.JOURNAL_ADDRESSED_INSTRUCTION` behind
`JOURNAL_ADDRESSED_LIVE = False`; `render_journal_review_block()` is byte-
identical to before while dark (pinned). **Next:** (f) flip the flag,
eval-gated → (g).
**Open:** the unified paragraph wording (eval-gated, Tom's nod); the fate of
the ~30 live `journals-escalation` nodes (archive behind a backup, or stop
injecting only); **boot influence** — "the influence of a single tell today is
very low" (Tom): delivery produces awareness, not action (the Gate 4 ask was
delivered at every boot for a week, unacted). Not solved here; 13(g) measures
delivered vs acted per moment, and that number picks the shape (forcing
render, response-required item, Prompt-moment re-surface). Uniting gives one
place to fix it.

Step 12 (Prompt moment) is independent — land in either order, merge main
between. The fourth-correspondent split (`env_message`) is NOT a Thalamus step;
it parks behind both (resolved shape: id:d8d38db2 — the client hook declares via
a tag→kind table in `trace_contract`; task-notification migration rides the
flip-day list). Phase 2b (S2 producers + boot-legacy retirement): id:5997f58c.

---

## Shipped ledger (one line each; the node holds the story)

| Step / arc | What landed | Merge | Node |
|---|---|---|---|
| Canonical-pull arc | absorbed ids resolve at every door; TTL reaper; audience rename (logs v3) | 2f0b54b, 9f769b4, 2bed982, b2adb5e | id:0cbc1e53, id:b0940238, ruling id:d42a49ce |
| 1 | five correctness defects in `thalamus.py` (unbound cursors, expiry anchored to `deliver_at`, loud live-route rejects, true overflow count, empty answer rejected, budget excludes expired) | 2026-08-28 | id:e72dde63 |
| 2 | guardrail: no bound SELECT cursor on a write connection | 2026-08-28 | id:e72dde63 |
| 3 | sweeps out from behind the S2 gate → `Brain.sweep_channels_if_due`, hourly throttles | 54984a3 | id:7efdb3f8 |
| 4 | refs resolve in `pull()` — batched, veil-aware; the contract only formats | 54984a3 | id:7efdb3f8 |
| 5 | append-only epoch ledger (`armed_epoch`; defer = re-arm, never delete) | 54984a3 (logs v2) | id:7efdb3f8 |
| 6 | contract-first vocabulary: `MOMENTS`/`ASK_MOMENTS`, `window_for`, derived `kind_of`; `via`/`audience` validated loudly | 54984a3 | id:7efdb3f8 |
| 7 | door polish: `file()` = validate → grammars → `_file_live`/`_file_queued` via one `_insert_item`; one `{'ok','id'}` envelope; `source` validated by `contract.validate_encoding_source` | b7e364a, cd36029 | id:26f76d24 |
| subtraction pass | 4 dead symbols out, 2 door guards in (directed ask rejects; expires ≤ when rejects) | 213759c | id:63b364f1; look-back id:35ef74e8 |
| 8 | `servers/channels/delivery.py`: Moment/Source, `serves()`, one `deliver()` walk, symmetric traces incl. boot; packages moved `scales/` → `channels/` | 00f53e7 | id:e8edd9e1; ruling id:7c7e805c |
| 9 | one relative-time grammar, `clock.resolve_offset`; tz-offset bug fixed | c746023 | id:e8719073 |
| turns & voices | correspondents first-class (dial `S0_CONVERSATIONAL_INCOMING`), Option A scope split, delivery K opens the successor chain, `ref_type` = correspondent axis | fcc99c7 | id:8bbf4f42; ruling id:3570a1bd |
| 10 | (a) `COMPOSITE_WARN` in delivery.py — a warn, not a cap (id:1e22a2f0); (b) `compose_block_loud` in `loud_truncation.py`, both contracts call it, render-identical by a 496-case snapshot | 56c7bd6 | id:828faa58, id:404e2d9f |
| 11 | named columns (`_ITEM_COLS`); `_boot_section` helper; boot comment states what boot commits; one channel-sweep table | e0f1ea8, feaab95 | id:cb29678b, id:c21c7705 |

**Locked by the shipped steps:** append-only epoch ledger; change-gated re-file
(an identical re-file refreshes the window, never re-delivers); kept-count
rendering (head/tail/ledger/count agree); sweeps ahead of the S2 gate; module-
owned SQL in the channel packages (no `ThalamusDAL` — revisit at a fourth
logs-backed queue); the Thalamus owns NO transport (messaging owns every leg);
moment vocabulary lives in `delivery.py`, moment-as-`ref_id`.

**Do not reopen:** ThalamusDAL; a producer-facing kind vocabulary (`kind_of` is
derived); sweep placement; delete-on-defer; unconditional dedup bump; the
Stop-hook two-source abstraction (delivery.py is the extraction that survived);
Option A scope split; `arms_continuation` + the 60-min window; the
`servers/channels/` package; S2 boot channel for now; the `remind` tool (kept);
render-join over write-back (id:defbdf8b); two state machines (id:e63c41dd).

---

## §2026-09-03 — Turns & voices substrate: what the ENCODER STREAM still owns

Built dial-gated with zero exposure (id:8bbf4f42): `deliver()` stamps each
source's block into its trace content with the moment as `ref_id`; the dial
gains `thalamus_delivery: False`; `last_delivery_stop` classifies the reaction
to a Stop-block as a real `assistant_message`; `get_session_turns` and
`get_conversation` carry `ref_type`; the delivery K OPENS the successor chain.
`OPERATOR_DIALOGUE_REF_TYPES` pins presence, episodes default, LAF and the trace
chain — only the encoder window, the embed lockstep, and the continuation stamp
ride the dial.

**FLIP-DAY CHECKLIST** (the encoder session's, beyond the prompt):
- **BLOCKER — `encode._lived_turns` grouping** (encode.py:1168): a flipped
  delivery episode matches no branch, so it is dropped AND the following
  reaction `assistant_message` OVERWRITES the previous operator turn's real
  reply; `_window_n_turns` (counts role=='user') diverges from `_lived_turns`.
  Both must learn the correspondent rows before any flip.
- **RULED to checklist (Tom, 2026-09-03): reaction rows in pinned scopes.** On
  flip, reaction rows enter presence focus/recency, the episodes default, LAF
  and the dual-store — the filter design (summary marker or accept) is the flip
  session's call.
- teach `embed_queue._render_trace_for_embedding` and `trace_contract.render_trace`
  speaker branches for self_message / thalamus_delivery (today both fall through
  to a ref_type literal).
- decide the boot-prelude render from the delivery trace's `ref_id` ('boot' vs 'stop').
- `encode.py:387`-area docstring still names the pre-ref_type get_conversation shape.
- then flip the dial row + restart, behind `s1_encode_eval`.

**NEXT-ARC THREAD — the fourth correspondent, the HARNESS** (Tom, 2026-09-03):
Claude Code's own injections (`<system-reminder>`, worktree notices) arrive
INSIDE user_message content, so the timeline attributes machine prose to the
operator. Measured 2026-09-04: 18 of 601 user rows in 7 days, every one fused
to a real prompt; neither Claude Code nor Codex declares a source field. Resolved
shape (id:d8d38db2): the client hook recognizes the envelope from a tag→kind
table in `trace_contract` and hands the daemon typed parts; the daemon writes a
dial-off `env_message` K row, then the operator row from the operator text;
presence summary and recall query use the operator half. Phase 2 migrates the
task-notification path onto it (deleting six filter sites) — encoder-visible,
rides the flip-day list. Not a Thalamus step.

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

**Depends on.** Nothing open. Rebase-aware with Step 13 in
`thalamus.py`/`thalamus_contract.py` (`pull`, `MOMENTS`).

**Named, not included.** Queued kinds riding the Prompt moment (cadence
ruling); `recall_episodes` bound echo (same shape as the door echo); the boot
Frame's conversation-time "Now" (untouched, grain-side).

---

## Step 13 — S1 through the Thalamus: addressed verbs in the review block

**Problem.** The Thalamus was built so the S1 Scribe could speak to the live
session it encodes — at Stop, not boot (Tom, 2026-09-05; rulings id:8aa9f183).
Today the Scribe has no way to: `remind` is not in its toolset, the door stamps
every filing `source='anchor'`, a directed ask is rejected (asks are boot-only),
and the door leaves no trace when an item is filed. Meanwhile the Scribe already
writes messages — into its own journal, where only its next run reads them
within a three-run window ("next session should revise this node to SHIPPED
once Tom says go", 2026-09-04). The message exists; the reader is wrong. And when
the Thalamus DOES reach a human today, the relay leaks the envelope — "th_0cbb56ae
sits in your court at every boot" — which a user cannot parse (id:3e549ac4).

**Target state.** The review block gains two addressed verbs in the grammar it
already has, and the journal component routes them to the door:

```
## Review
friction · 1ca943af · turn-row projection built in two places, drift risk
tell     · segment 6.a · you are proceeding on "I wonder if", not a yes — confirm first
ask      · 7e6decd2 · milestone says merge pending; it merged as 56c7bd6 — revise, or leave?
```

- **(h) Plain-language relay — land FIRST, alone.** `render_block`'s head gains
  one instruction line for the reader, the mirror of the self-channel's note
  line: *"when you raise one of these with the operator, say it in plain words —
  what is asked and why it matters; never the id, the moment, or the producer."*
  One line in `thalamus_contract`, one head test. The `remind` tool description
  (brain_mcp.py, rides a redeploy) gains the producer half: *"written for a reader
  with none of your context — no ids, no internal vocabulary."*
- **(a) Contract, dark.** `trace_contract.JOURNAL_ADDRESSED_TAGS = ('tell',
  'ask')`. The parser needs NO change — a 3-field line with tag `tell` already
  parses as `(tag, subject, note)`. `render_journal_review_block(addressed=False)`
  gains the S1-only paragraph (admission test: *would the session, or Tom, act
  differently in the next hour if they knew? → tell/ask; would only your next run
  care? → journal*; plain language for a reader with none of your context; one
  line per subject, don't re-assert). S2 bindings keep today's text.
- **(b) Write door returns, journal component routes.** `write_journal_notes`
  skips addressed notes and returns them (`{'addressed': [...]}` beside
  `written`/`malformed`) — the traces layer never imports a channel.
  `JournalBinding.harvest()` files each through `thalamus.file(brain,
  source=<binding's encoding_source>, body=note, needs_answer=(tag=='ask'),
  for_whom=<binding.session_id>, dedup_key=subject if 8-hex else None,
  refs=[subject] if 8-hex, session_id=…)` — the non-LLM entrance of
  id:7e9870ce. A binding without a session (S2) writes them as plain notes and
  warns `journal_addressed_unbound` — S2 stays as is. A door rejection is a loud
  warning `journal_addressed_rejected` AND the line is kept as a journal note so
  the residue survives and the encoder reads the rejection next run. A
  `resolved · <subject>` whose subject matches this source's open `dedup_key`
  also calls `thalamus.withdraw(source, dedup_key=subject)` — the encoder's
  existing verb closes its own item.
- **(c) Door.** (i) `ASK_MOMENTS` becomes per-audience: a directed ask (one
  session) delivers at Stop; a broadcast ask stays boot-only; `file()` drops the
  directed-ask rejection for that case and `pull(via='stop')` admits directed
  asks. (ii) `list_items(brain, source='', target_session='', include_closed=…)`
  — the by-source read the join needs (one parameter, not a new function).
  (iii) **Budget key.** `MAX_OPEN_PER_SOURCE` keys on `source` alone; the
  Scribe's `encoding_source` is `encoder:sonnet` for every session's runs and
  the grammar allows one colon, so eight open items would be a global cap. For
  directed items the key is `(source, target_session)` — RULED (Tom,
  2026-09-05, id:1dda87c2), named in the contract comment beside the cap.
  **(c) SHIPPED** — `ASK_MOMENTS` is a per-audience map, the door rejection
  is gone, `_due_filter` excludes each audience's asks at its off-moments,
  `list_items(source=, target_session=)`, budget predicate adds
  `target_session`; id:178f4727 revised to superseded.
- **(d) Trace the filing — SHIPPED.** `("s1", "delta")` gains `thalamus_filed`
  (also in `RESIDUE_REF_TYPES` + the dashboard mirror, so per-run consumers
  never count a filing as a run); the door, handed a `run_chain`, calls
  `brain.write_thalamus_filed` — one row (ref_id = item id; metadata =
  `THALAMUS_FILED_METADATA_SHAPE`, door vocabulary: source, body, target,
  needs_answer, dedup_key, route, filing ∈ new/refresh/rearm), scale derived
  from the chain prefix (`scale_for_chain`), failure-isolated. The write lives
  in `brain_traces.py` (the traces door, already a guardrail writer file), not
  the channel. An item's life is then joinable across scales: filed (s1 Δ, run
  chain) → delivered (s0 K `thalamus_delivery`, the session's chain,
  ref_id=stop; join filters route='queue') → answered (item state; the resolve
  call in the session's own tool trail).
- **(e) Feedback by render-join, never write-back — MINIMAL (Tom, "let's
  expose").** `JournalBinding.continuity()` appends, after the residue notes,
  the binding's own items with their SETTLED outcome — `open` / `answered:
  <text>` / `dismissed` / `expired[, unanswered]` — never delivery counts,
  moments or dates. The join lives in the binding (`_producer_view`), the fate
  derivation in `thalamus_contract.fate_of` (mirror of `kind_of`), the
  phrasing + caps in `trace_contract.render_producer_view` (the journal's line
  grammar), the window in `list_items(settled_days=)` (SQL, wall-clock in the
  channel). Withdrawn items don't render; a refused filing is already a note
  with `undelivered`. No `delivered` mark ever enters the journal
  (id:e63c41dd, id:defbdf8b).
- **(f) Prompt, eval-gated, the only encoder-visible change.** Ship (a)–(e)
  DARK first — with the paragraph absent no encoder writes the verbs, behavior
  is bit-identical. Then the S1 review paragraph: read the whole S1 prompt +
  review block first (id:71eeff20 discipline), run `eval/s1_encode_eval.py`
  before/after, `tests/test_s1e_residue.py` green, then restart.
- **(g) Measure, one week.** From traces + ledger: `thalamus_filed` per
  `encoding_run`; `journal_addressed_rejected` count; delivered latency
  (filed → ledger row); answered vs dismissed; dedup updates vs inserts.
  Kill criteria: zero filings in the window (unused — the `bridge_proposals`
  death, id:bfc6d106) or dismissed > answered (noise). Either way the number
  decides Phase 2b, not the build.

**Timeline representation.** In the receiving session's S0 timeline the
Scribe's message is the brain speaking at a Stop (`thalamus_delivery`, the
Scribe named inside the block) followed by the session's reaction. The dial is
off, so the encoder does not see that turn yet; after the voices flip it reads
its own tell arriving and what the session did — the loop closes inside the
lived sequence. Until then its only feedback is (e).

**Files & call sites.** `servers/trace_contract.py` (tags, review paragraph,
REF_TYPES); `servers/brain_traces.py` (`write_journal_notes` return shape);
`servers/scales/journal.py` (`harvest` routing, `continuity` join);
`servers/channels/thalamus/thalamus.py` (`file` run_chain + trace, `pull`
directed asks, `list_items` filters, budget key), `thalamus_contract.py`
(`ASK_MOMENTS` by audience, `render_producer_view`, the head relay line);
`servers/scales/s1/encode.py` (binding passes `addressed=True`); `brain_mcp.py`
(`remind` description, redeploy). Tests: `test_journal_notes.py`,
`test_journal_component.py`, `test_journal_lifecycle.py`, `test_s1e_residue.py`,
`test_thalamus.py` (directed ask at Stop; budget key; list filters; filed
trace; head relay line), `test_trace_contract_sync.py` (new ref_type + writer
file), `test_delivery.py`.

**Verification.** `./dev pytest tests/ -k "journal or residue or thalamus or
delivery or trace_contract" -q`, tier checked with `--collect-only | grep`.
Dark ship: a Scribe run on an isolated brain with no paragraph writes zero
items and zero `thalamus_filed` rows. Lit: one real session — a `tell` lands at
the next Stop as a brain block naming the Scribe; the next Scribe run's
continuity shows it delivered; `query_traces(ref_type='thalamus_filed')` and
`query_traces(ref_type='thalamus_delivery', ref_id='stop')` join on the item id.

**Blast radius.** (h) changes one head line I read at every delivery. (a)–(e)
dark: `write_journal_notes` return gains a key; the door's directed-ask
semantics change for directed items only; a new ref_type; one list-read
parameter; a budget-key change. (f) changes what the Scribe is told —
encoder-visible, hence the eval gate. S2 units, boot, and the escalation
channel are untouched.

**Respects.** Two orthogonal state machines (id:e63c41dd); render-annotation
over write-back (id:defbdf8b, forced by id:8a170558); pull model (id:1448610f);
Thalamus owns no transport (id:7c7e805c); contract/mechanics split (id:35ef74e8);
one door, three entrances (id:7e9870ce — BENT for the Scribe's end-of-run,
low-volume shape: a one-run-late rejection costs nothing; the tool stays for
mid-run asks, named in id:8aa9f183); plain-language relay (id:3e549ac4).

**Depends on.** Nothing shipped is a blocker. Rebase-aware with Step 12 in
`thalamus.py`/`thalamus_contract.py` (`pull`, `MOMENTS`) — land in either order,
merge main between. Sub-step order: (h) → (c) → (d) → (b) → (e) → (a) dark →
(f) → (g); each is cold-startable and separately testable.

**Named, not included.** Event-conditioned items ("when Tom says go" — the
`on_topic` moment, Phase 3); S2 producers and the boot-legacy retirement (Phase
2b, id:5997f58c); the voices dial flip (encoder stream); the `env_message` split
(not a Thalamus step).
