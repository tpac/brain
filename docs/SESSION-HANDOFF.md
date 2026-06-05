# Session Handoff — current state

**Living current-state doc.** When a new session meaningfully updates it, copy the prior version to `docs/archive/session-handoffs/SESSION-HANDOFF-{date}.md` first, then prune to current.

**Last refreshed:** 2026-06-05 (doc reconciliation + self-channel message fixes — delivery-trace `session_id` attribution, per-message TTL by address, drain/peek dedup [CR5], presence-focus fix [CR3]; backlog/handoff verified against code). Prior version (self-channel rules-of-engagement substrate, 2026-05-31) archived to `docs/archive/session-handoffs/SESSION-HANDOFF-2026-05-31-self-channel.md`.

---

## ⚠️ READ THIS FIRST

**Recently shipped (2026-06-05, pushed):** self-channel message fixes — delivery-trace `session_id` attribution, per-message TTL by address (broadcast 1h / directed 24h, config-tunable), drain/peek dedup, presence focus = latest conversational turn, S2 delta payload validation + loud truncation. Detail: `docs/BACKLOG.md` 2026-06-05 capture.

Two arcs landed after the prior handoff was written, both on `main` (not pushed). The prior handoff predated them and listed the second as its top *open* thread — it's done:

1. **Cross-stream comms hardening (2026-06-04, `78693a3` + `364269f`).** The daemon Errno-48 boot-race is fixed — all (re)starts now route through launchd (`_launchd_kickstart` / `launchctl kickstart -k`), serialized under the flock; single-owner lifecycle, no competing Popen. Self-channel got streams-experience polish: self-id at boot (`MY_STREAM_ID`), watchers count as present (heartbeat turns), Stop-only self-message delivery. **Resolved D1** (daemon `DAEMON_DOWN` recurrence — root cause was the over-determined lifecycle) and **CR2** (idle `/watch` window ageing out of presence). Full capture: `docs/BACKLOG.md` 2026-06-04 entry.

2. **S2 absorb consolidation (2026-06-04, `abe98df` + `e2fb44f`).** Consolidation now **emits `absorb`** — prompt v7 + decoder lever A (`_pre_classify` routes cross-type clusters → `needs_judgment`), eval-tested (correct 10→15, under-merge 8→3, over-merge held at 1). Design docs synced (`S2-ABSORB-OP-DESIGN.md`, `S2-CONSOLIDATION-LOCKED-ABSORB.md` → v7). The absorb *primitive* shipped earlier (`d3a0fa1`); this is the consolidation wiring that the prior handoff awaited.

---

## The standout open arc — recall-side episodic references (`docs/EPISODIC-REFERENCES.md`)

The encoder **write** path (`source_refs`, `co_anchored` edges) has been live in production since ~2026-05-25 — we pay the encode cost every cycle. The **read** side that turns that substrate into user-felt recall is **unbuilt and never measured**:

- **render expansion at `SURFACE_FORMAT`** (the joint-reactivation read shape) — ground-up; zero `source_ref` handling in `surface_contract.py` today (verified 2026-05-30).
- **`source_summary` parallel-pathway recall scoring** — `max(legacy_weighted_sum, source_summary_score)`; backwards-compatible by design.
- **S2Healer source_refs cleanup** — scan invalid trace_ids, archive orphan `co_anchored` edges.
- **§13.6 recall eval gate — never run.** The "shipped measurable substrate, skipped the measurement" thread.

Highest-value loose end: cost sunk, value uncollected. This is the arc to pick up when doing substantive recall work.

---

## Other open threads

### Haiku turn analysis (A1 / A2 / A3) — `docs/BACKLOG.md` 2026-06-04 capture + "surface_haiku warm floor"
There is **no `cache_control` anywhere** in the surface path (neither v4 nor `v5_agentic`) — the CLAUDE.md "cached system block" claim was never wired. Steady-state `surface_haiku` is 8–10s (a 2-round agentic loop, not a single call); watch-mode self-messages pay full agentic recall because `<task-notification>` dodges the `pre_response_recall` skip-gate. Levers: **A1** add `cache_control` (Haiku-4.5 min cacheable prefix = 4096 tokens — measure the instructions-vs-candidates split first; reliable win is caching round-1's prefix for round 2); **A2** watch-wake recall contract + cousin-community breadcrumb (Tom's design); **A3** kill the agentic-loop network roundtrip. Eval-gated on recall quality.

### Self-channel remaining phases (`docs/SELF-CHANNEL-DESIGN.md`)
- **Phase 3 — boot letter.** First-person session arc surfaced at wake. `render_letter` exists; boot smart-surface not built. Parked behind the boot-reignition eval (`docs/BOOT-REIGNITION.md`).
- **Phase 4 — encode self-turns.** `self_message: False` at `trace_contract.py:177` (verified unflipped 2026-06-05). Flip to `True` so anchor↔anchor turns encode/recall like operator turns. One dial when wanted.

### S2 consolidation → live merge (`docs/S2-CONSOLIDATION-ABSORB-SESSION-HANDOFF.md`)
The `absorb` wiring shipped (above). Deferred remainders in that doc: the **live real-pair merge** of `96d2fdf8`/`426ae3cd`, **Track 1b**, and edge-direction handling. Eval-gated (consolidation is sacred).

### DAL Cleanup Phases 4/5/6 (`docs/DAL-CLEANUP-PLAN.md`)
Reads migration, missing DALs (incl. the incomplete `NodeDAL.purge` cascade — **a latent bug**), and the raw-SQL guardrail. Phase 4 gated on a SessionStateDAL resurrect-vs-delete decision. ⚠ a parallel stream has touched DAL — check presence / `git log` before driving.

### Smaller cleanups (itemized in BACKLOG)
**A2** (watch-wake contract, above), **CR4** (`scale='s0'` predicate on `active_sessions_by_turn`), **R1** (revise MCP-description rewrite — eval-gated), **presence read-lag** (F2b — boot-transient half).

*B4 (CCD↔brain namespace bridge) — closed 2026-06-05: CCD and brain use independent session-id spaces with no robust shared key; comms stay in the brain id space, so there's no bridge to build. Optional per-session cwd/branch ergonomics parked.*

### Known recall bug — dampening cluster (BACKLOG P1.6)
Synaptic-fatigue (per-session anti-repeat) + hub-dampening are broken post spread-activation migration. Four RED tests reproduce it (`test_fatigue_*`, `test_hub_dampening`), parked with the recall work since 2026-05-29. Detail in `docs/S2-GATING-AND-TEST-CLEANUP-HANDOFF.md` → "Parked work". Natural bundle with the recall-arc above.

### Tom-time-gated
`eval/ground_truth/` — 7 conversation templates across 5 strata are scaffolded (source conversations written), but the **ideal-node YAML is still unfilled** (verified 2026-06-05 — all `<fill>`). Tom's ~1.75h authoring session, untouched. Once filled: targeted structural-delta eval joins the longmem oracle path.

---

## Production state (don't hardcode — `list_interactions` is the source of truth)
- Encoder: s1e v24 + s1_scout_facts v7 + s1_scout_quote v4 activated 2026-05-30 (`d0fea6d`, `47f7018`).
- S2 consolidation: prompt v7 activated 2026-06-04 (`abe98df`).
- Confirm exact active versions via `list_interactions` rather than trusting any doc.
- `eval/encoder_eval/` (A/B any encoder version vs any past one) and the Frozen Corpus harness (`eval/longmem/`) are the evaluation surfaces — see their READMEs and `docs/EVAL-PLATFORM.md`.

## Priority docs
1. This handoff
2. `docs/EPISODIC-REFERENCES.md` — recall-side work + §13.6 gate (the standout open arc)
3. `docs/BACKLOG.md` — broad open-item registry. Verify any "open" item against code before picking it up — the dated session captures are history and aren't always pruned when an item ships.
4. `docs/SELF-CHANNEL-DESIGN.md` — self-channel phases 3/4
5. `docs/S2-CONSOLIDATION-ABSORB-SESSION-HANDOFF.md` — absorb live-merge remainders
6. `docs/DAL-CLEANUP-PLAN.md` — DAL Phases 4/5/6
