# Session Handoff — current state

**Living current-state doc.** When a new session meaningfully updates it, copy the prior version to `docs/archive/session-handoffs/SESSION-HANDOFF-{date}.md` first, then prune to current.

**Last refreshed:** 2026-05-31 (self-channel rules-of-engagement substrate). Prior version (the v22/v24 encoder thread, now resolved) archived to `docs/archive/session-handoffs/SESSION-HANDOFF-2026-05-26-v22-thread.md`.

---

## ⚠️ READ THIS FIRST

Most recent work: the **self-channel rules of engagement** — the channel between your parallel streams now has behavioral rules *and* the substrate to back them. All on `main`, tested, tree clean.

**Start in `docs/SELF-CHANNEL-DESIGN.md` → the "Rules of Engagement" section** — it's the load-bearing record (the 7 rules, the live collision that motivated them, the substrate table, the agency/containment design). Brain nodes: `1ae50ab9` (a self-message is an observation, not ground truth — verify with presence) and `731ff525` (we hit the failures live while writing the fix).

---

## What shipped this session — self-channel substrate

The channel went from raw mechanism to "has rules + the substrate that makes them true." Six commits on `main`:

| Commit | What |
|---|---|
| `7f80913` | **Containment render** — incoming signals render as `⚡ <who> says: "…"` + an attribution footer, so another stream's "I did X" can't bleed into your self-model. **`self_outbox`** — sender sees per-recipient delivery + pending (read silence correctly). |
| `fd9202a` | **Presence active / dormant / lost** — roster classifies each stream by recency; lost streams surfaced, not silently dropped. |
| `d8a87ad` | **SKILL.md operative rules** — the always-loaded behavioral layer (receive/triage, agency-follows-the-hands, send/claim/release, turn-gated → *suggest* `/watch`). |
| `1758a15` | **Addressing** — graceful `self_send(to=)` resolution. |
| `bbc0c10` | doc substrate table → shipped; dead import removed. |
| `333474b` | **Dropped self-labeling → ids only** (mutable/collidable/spoofable names removed). |

**Addressing model (final):** `self_send(to=)` takes a full session UUID (canonical), the 8-char short you see in a message (resolves as an id-prefix against the live roster), or `broadcast`; ambiguous/gone is loud. No self-names — **focus** in `self_presence` is the truthful "who."

**Encode-side containment is `anchor-w`'s lane** (Tom-directed): trace-contract turn-classification, `self_message=False` → self-turns aren't encoded. Phases 1+2 landed (`1e14058`, `4233eaf`). Flipping `self_message=True` in `trace_contract.py` is self-channel **Phase 4** below.

---

## Open threads — what we HAVEN'T done

### Self-channel, remaining phases (design in `docs/SELF-CHANNEL-DESIGN.md`)
- **Phase 3 — the boot letter.** First-person session arc surfaced at wake (`render_letter` exists; boot smart-surface not built). Parked behind the boot-reignition eval (`docs/BOOT-REIGNITION.md`).
- **Phase 4 — encode self-turns.** Flip `self_message=True` in `trace_contract.S0_CONVERSATIONAL_INCOMING` so anchor↔anchor turns encode/recall like operator turns. One dial-flip; do it when wanted.

### S2 consolidation → `absorb` wiring (`docs/S2-ABSORB-OP-DESIGN.md`)
The `absorb` op (lossless merge) shipped as a primitive (`d3a0fa1`), and `s2_consolidation_enrichment` v6 stopped the locked-node churn (`714ee68`) — but **consolidation does not yet emit `absorb`**. Next: wire the consolidation encoder to emit it, then the live merge of `96d2fdf8`/`426ae3cd`. Eval-gated (consolidation is sacred).

### Recall-side episodic-references (`docs/EPISODIC-REFERENCES.md`)
The encoder *write* path is shipped and live (source_refs, co_anchored edges). The **recall side is not built**:
- render expansion at `SURFACE_FORMAT` (the joint-reactivation read shape)
- `source_summary` parallel-pathway recall scoring (`max(legacy_weighted_sum, source_summary_score)`)
- S2Healer source_refs cleanup (archive orphan `co_anchored` edges, scan invalid trace_ids)
- **§13.6 recall eval gate — never run.** This is the "infrastructure shipped, recall improvement unconfirmed" thread. Comes online once enough source-anchored nodes have accumulated to measure.

### eval/ground_truth authoring
7 conversation templates are scaffolded at `eval/ground_truth/` (5 strata). Ideal-node authoring was "pending Tom's ~1.75h focused session" — **verify whether the YAML is now filled** before assuming it's done.

### Documentation closing-plan (from this session's 6-agent docs audit — parked)
The session *opened* on a docs audit that produced a closing-plan, then pivoted to the self-channel build. The plan is unexecuted:
- **Reconcile badly-stale docs:** `FRAME-DESIGN.md` and `ARCHITECTURE-FRACTAL.md` both still claim a pre-production state while the work shipped (Frame's agentic `v5_agentic` path is live; SessionContext is load-bearing). These actively mislead.
- **Archive done docs:** `AGENTIC-SURFACE-CONTRACT.md` (shipped as `v5_agentic`), `WRITE-TXN-ISOLATION-ROOTFIX.md` (F3 shipped), `S2-GATING-AND-TEST-CLEANUP-HANDOFF.md` (only Bucket E remains).
- **`docs/BACKLOG.md` cleanup:** strike F3 (shipped); close Q13/spread-activation; add the absorb-op ship line.
- **DAL Phases 4/5/6** (`docs/DAL-CLEANUP-PLAN.md`) — reads migration, missing DALs (incl. the incomplete `NodeDAL.purge` cascade — a latent bug), and the raw-SQL guardrail. Phase 4 gated on a SessionStateDAL resurrect-vs-delete decision. *Note: a parallel stream has touched DAL recently — check presence/`git log` before driving it.*

---

## Production state (don't hardcode — `list_interactions` is the source of truth)
- The v22/v24 encoder thread that dominated the prior handoff is **resolved**: v24 + facts-scout v7 activated 2026-05-30 (`d0fea6d`). Confirm exact active versions via `list_interactions` rather than trusting any doc.
- `eval/encoder_eval/` exists for A/B-ing any future encoder version against any past one (parallel cells, substrate-aware probes) — see its README.

## Priority docs
1. This handoff
2. `docs/SELF-CHANNEL-DESIGN.md` — self-channel (just-shipped substrate + remaining phases)
3. `docs/EPISODIC-REFERENCES.md` — recall-side work + §13.6 gate
4. `docs/S2-ABSORB-OP-DESIGN.md` — consolidation→absorb wiring
5. `docs/DAL-CLEANUP-PLAN.md` — DAL Phases 4/5/6
6. `docs/BACKLOG.md` — the broader open-item registry
