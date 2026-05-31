# Session Handoff — current state

**This is the living current-state doc.** When a new session updates it, the prior version moves to `docs/archive/session-handoffs/SESSION-HANDOFF-{date}-{slot}.md`.

**Last refreshed:** 2026-05-26 (v22/v24 encoder thread — everything below). · **2026-05-31:** a parallel S2 consolidation-merge / `absorb` thread was added — see the callout directly under READ THIS FIRST; full detail in `docs/S2-ABSORB-OP-DESIGN.md`. The v22/v24 content below is untouched and still open.

Prior version archived to `docs/archive/session-handoffs/SESSION-HANDOFF-2026-05-25-post-v22-ship.md`.

---

## ⚠️ READ THIS FIRST IF YOU JUST WOKE UP

> **Parallel thread added 2026-05-31 — S2 consolidation merge / `absorb`** (separate from the v22/v24 encoder thread below, which is still open). Shipped on `main`: `s2_consolidation_enrichment` **v6** (locked dupes → KEEP, kills the archive+churn loop; commit `714ee68`) + a new **`absorb`** brain_batch op — a lossless merge primitive (commit `d3a0fa1`). **That thread's next session starts in `docs/S2-ABSORB-OP-DESIGN.md`** → wire the consolidation encoder to emit `absorb`, then the live merge of `96d2fdf8`/`426ae3cd`. Everything below is the unrelated v22/v24 encoder workstream.

**Phase B substrate complete + v22 active in production.** The episodic-references write path (Steps 0–7) is live; v22 encoder prompt active since `2026-05-26T00:46:54Z` (commit `c144ddf`). v23/v24 + new scout versions are registered DORMANT pending one open eval-decision call.

**Stable state of `main`** (clean working tree):
- v22 s1e prompt active in production
- Phase B Step 7 (co_anchored auto-edge) shipped — dispatcher writes structural engram edges when source_refs overlap
- Layer 1 validator extensions (hex-format soft warn + sparsity threshold lowered from >10 to >5)
- `sync-prompts` fixed to mirror ACTIVE version, not highest registered (closes the DORMANT-leak bug)
- `eval/encoder_eval/` infrastructure shipped: 6 probes, multi-version harness, parallel + stratified runner
- `eval/ground_truth/` corpus scaffolded (7 conversations across 5 strata) — Tom-authoring pending
- 165 v22-eval-gate tests pass

**Active priority docs** (in order):
1. This handoff
2. **`docs/EPISODIC-REFERENCES.md`** — §0 execution log captures Phase B + v22 ship state
3. **`docs/BACKLOG.md`** — open items + deferred reviewer follow-ups + v24 experimentation thread
4. **`eval/encoder_eval/README.md`** — multi-version encoder eval module architecture
5. **`servers/scales/s1/quality_contract.py`** — 36-dim contract, version 3

**Next session's natural starting point**: **The v24/v7/v4 experimentation thread.** v22+v5+v2 is the proven production baseline (50-cell longmem eval: 23/25 correct, 100% source_refs coverage). v24+v7+v4 is a DORMANT candidate stack with one open decision: re-run gpt4_d31cdae3 (temporal hedging) a few times to determine stochastic-vs-deterministic regression before activate-or-revise.

---

## What shipped 2026-05-26

Terse — details live in commit messages + per-cell artifacts.

### Phase B Step 7 — co_anchored auto-edge
**Commit `07ab3f1`.** When a node is written with `source_refs`, the dispatcher queries `node_source_refs(trace_id)` for siblings sharing any ref and writes structural `co_anchored` edges via `GraphDAL.add_relation`. Fires in both `remember()` and `revise()` REPLACE paths. Excluded from candidate cosine ranking (same pattern as `co_accessed`). 5 new tests in `test_remember_source_refs.py`.

### Layer 1 — validator soft-warns
**Commit `07ab3f1`.** Two additions to `daemon_dispatch.py`:
- `_maybe_warn_source_refs_hex_format`: rate-limited soft warn when any ref doesn't match `^[0-9a-f]{8}$` (catches literal-placeholder-copy earlier than S2Healer). Reviewer F6.
- Sparsity threshold lowered from >10 to >5 — aligns with v22's §7.5 teaching ("1-3 typical, second-guess at 5-6").

Wired into all four dispatch entry points: `remember`, `remember_batch`, `revise`, `revise_batch`. 3 new validator tests pass.

### sync-prompts active-version fix
**Commit `1fe730e`.** Bug: `_fetch_latest` grabbed `ORDER BY version DESC LIMIT 1` — the highest registered version regardless of active state. Combined with the v22 DORMANT pattern, this would seed fresh-brain installs with untested candidates, **bypassing the eval gate**. Fix: renamed `_fetch_active`, JOIN against `interaction_active` pointer. Regression-locked by `test_sync_grabs_active_not_latest_version` in `test_prompt_sync.py`. CLAUDE.md updated with eval-gated workflow.

### eval/encoder_eval/ infrastructure
**Commits `5238c02`, `5a4e672`, `fe71ac4`.** New module: multi-version encoder quality eval composing existing `eval/longmem/` pieces with six probes (`brain_presence`, `specificity_preservation`, `source_refs_coverage`, `atomization_shape`, `edge_structure`, `voice_balance`), staged checkpointed harness, parallel cell execution (ThreadPoolExecutor), `--stratify` for axis-balanced sampling, default stop conditions, streaming `per_cell.jsonl`. README.md documents architecture. 9 smoke tests for probes.

Key architectural insight surfaced during dogfooding: `probe_specificity_preservation` was substrate-blind — only checked node content, not the trace_events.metadata.content that source_refs point at. Probe now reports `in_content` + `via_substrate` + `combined` scores separately. v22's "specificity drop" turned out to be a probe artifact, not an encoder regression.

### v22 active flip
**Commit `c144ddf`** at `2026-05-26T00:46:54Z`. Evidence: 50-cell longmem eval (v22 vs v19, 5 axes × 5 items × 2 versions):
- v22 23/25 (92%) vs v19 22/25 (88%) on answer correctness
- v22 100% source_refs coverage on all 25 cells uniformly; v19 averaged 72-96%
- v22 wins specificity (combined) on 4 of 5 axes
- v22 produces ~40% more typed edges per item on info_extraction; co_anchored cohorts form correctly
- v21 (the unevaluated yesterday-fix) was implicitly retired — Tom's directive ("v21 is a non-stable version from yesterday")

### eval/ground_truth/ corpus
**Commit `7e721eb`.** 7 conversation templates across 5 strata: 2 identity-bearing, 2 partnership voice, 1 technical correction, 1 methodology, 1 temporal. Each has fillable YAML for ideal-node authoring + rationale + expected failure modes. Authoring pending Tom's ~1.75h focused session.

---

## v23/v24 + scout v6/v7 + quote v3/v4 — current state (DORMANT)

Forensic probes on c2ac3c61 (multi_session precision-refinement failure in 50-cell run) surfaced three encoder-prompt gaps + one facts-scout misclassification pattern. Two iteration cycles:

### v23 / facts v6 / quote v3 (initial, DORMANT)

Three small changes drafted:
- s1e v23: one sentence in §7.4 (Anchored synthesis) — "when a fact surfaces twice in the window (vague earlier + precise later) on the same axis, BOTH are evidence-events for one node"
- facts_scout v6: 4 edits — supersession-scope clarifier on line 34, Cap ranking reformulated, Example 4 why_candidates tightened, NEW Example 5 (parallel-entity + same-axis refinement, languages-domain abstraction)
- quote_scout v3: new Skip bullet ("routine factual claims = facts scout's territory")

### v24 / facts v7 / quote v4 (post-probe refined, current DORMANT)

Ambiguity probe surfaced three real fixes:
- s1e v24: "originating phrase" disambiguated to "originating turn's verbatim phrasing stays in user_raw_quote — keep the vague phrase; don't overwrite with the refined wording"
- facts_scout v7: Cap statement preserves "recall-weight" framing (not exclusion rules) and adds refinement-pair clarification
- quote_scout v4: appended "When a turn mixes routine factual content with distinctive wording, isolate just the distinctive phrase as the handle"

Diff files at `/tmp/{s1e_v24,s1_scout_facts_v7,s1_scout_quote_v4}_proposed.txt`. Pre-state at `/tmp/{s1e_v22,s1_scout_facts_v5,s1_scout_quote_v2}_pre.txt`.

### Targeted eval — apples-to-apples A/B (5 items × 2 arms)

`eval/encoder_eval/reports/v24_targeted_20260525_235258/`

| Item | Axis | baseline v22+5+2 | candidate v24+7+4 |
|---|---|:---:|:---:|
| c2ac3c61 | multi_session | ✓ | ✓ |
| 5025383b | multi_session | ✗ | **✓** |
| 60159905 | multi_session | ✓ | ✓ |
| ce6d2d27 | knowledge_update | ✓ | ✓ |
| bbf86515 | temporal | ✓ | ✓ |

**Subtotal: baseline 4/5, candidate 5/5.**

### Follow-up eval — candidate-only on 5 new items

`eval/encoder_eval/reports/v24_followup_20260525_235653/`

| Item | Axis | v22 (50-cell) | v24+7+4 |
|---|---|:---:|:---:|
| 3fe836c9 | multi_session | ✓ | ✓ |
| gpt4_93159ced_abs | abstention | ✗ | **✓** |
| cc539528 | info_extraction | ✓ | ✓ |
| cc5ded98 | knowledge_update | ✓ | ✓ |
| gpt4_d31cdae3 | temporal | ✓ | **✗** |

**Subtotal: candidate 4/5; baseline known 4/5 from 50-cell run.**

### Combined v24 candidate signal

**9/10 (90%) across 10 distinct items, 15 total cells.** Two real qualitative wins:

1. **Answerer-overreach fixed (gpt4_93159ced_abs)**: both v22 AND v19 lost this in the 50-cell run with the NovaTech fabrication. v24 says *"I don't have information about your work history or when you started at Google."* Clean abstention. Likely driven by quote_scout v4's "routine factual claims = facts scout territory" rule reducing answerer's ambient-context volunteering.
2. **Hobby cohort retention (5025383b)**: v22 (this run) dropped one hobby — *"I only have one hobby connected — photography. I don't have a record of a second hobby."* v24 caught both photography AND cooking.

One real concern:

**Temporal hedging regression (gpt4_d31cdae3)**: v22 ✓ → v24 ✗. v24 over-cautious on inferring "a few years ago" relative to current-date. Could be encoder-side (didn't resolve to event_time) OR answerer-side (had the data, didn't compose ordering claim) OR N=1 stochastic noise.

### Methodological finding (the big one)

**c2ac3c61 wasn't a deterministic v22 failure.** Same prompt, fresh brain, succeeded this run. The 50-cell run captured one stochastic v22 failure. Same with 5025383b: v22 caught both hobbies in the 50-cell run but missed cooking this time. **LLM encoders are stochastic. N=1 results have noise.** Some 50-cell "fails" are tail outcomes; some are deterministic patterns. Distinguishing requires re-runs.

---

## Open eval-decision call

Three options for v24+v7+v4 disposition:

**A. Activate now.** +1/10 wins, real abstention improvement, no catastrophic regression. The temporal hedge is one item; could be noise; production exposure surfaces if it's systemic.

**B. Cheap re-run check first.** Run `gpt4_d31cdae3` three more times against v24 (3 cells, ~15 min wall, ~$2). Stochastic → activate. Deterministic → investigate the temporal-inference path before activating.

**C. Don't activate.** Net delta too small to justify the change; leave v24/v7/v4 DORMANT pending a more thorough eval or until additional failure modes accumulate.

**Tom's preference (per session close)**: continue experimenting with v24 + v7. The natural next step is B (cheap re-run check) followed by A or C depending on result.

---

## What lives in production right now

| Layer | Active | Notes |
|---|---|---|
| s1e prompt | **v22** (since 2026-05-26T00:46:54Z) | DORMANT v23, v24 await eval decision |
| s1_scout_facts | **v5** | DORMANT v6, v7 await eval decision |
| s1_scout_quote | **v2** | DORMANT v3, v4 await eval decision |
| s1_scout_temporal | v2 | unchanged |
| Schema | **v29** | trace_id hex live since 2026-05-25 |
| Encoder quality contract | **v3** | 36 dims, 12 CR rules, Group 9 example_authoring |
| §7.6 example library | wave-1 (7 examples) | All placeholder-compliant |
| Source_refs WRITE path | shipped | MCP schemas, brain.remember/revise, dispatch validation |
| Source_refs READ path | shipped | get_trace + get_traces |
| **co_anchored auto-edge** | **shipped (Step 7)** | Fires in `remember` + `revise` REPLACE |
| **Layer 1 validator** | **shipped** | Hex-format soft warn + sparsity >5 warn |
| **sync-prompts active-version** | **fixed** | DORMANT candidates excluded from seed |
| Identity stamping | shipped | trace_events.metadata identity fields |
| Trace embeddings | shipped | brain_logs.db trace_embeddings, embed worker live |
| Daemon | healthy | |

---

## eval/encoder_eval/ — the new infrastructure

**For multi-version encoder testing**: any future v25/v26/... can walk through this module's gate.

### CLI

```bash
./dev python3 -m eval.encoder_eval.runner \
    --versions 22,24 \
    --corpus longmem \
    --stratify 2 \
    --parallel 6 \
    --run-name vNN_eval_$(date +%Y%m%d_%H%M%S)
```

### Targeted eval scripts (reusable patterns)

- `eval/encoder_eval/targeted_v24_eval.py` — A/B with all three interactions overridden (s1e + facts + quote). Pattern: `apply_interaction_override(brain, name, template)` works for any interaction.
- `eval/encoder_eval/targeted_v24_followup.py` — candidate-only follow-up on new items, baseline known from prior run.

### Six probes (run on every cell)

1. **brain_presence** — does any node contain the gold answer's atomic value (field-weighted string match)
2. **specificity_preservation** — substrate-aware: numerics from haystack preserved in node content OR recoverable via source_refs → trace_events.metadata.content
3. **source_refs_coverage** — % of nodes carrying refs, sparsity violations, hex-format failures
4. **atomization_shape** — nodes per turn (sweet spot 0.3-0.8), type diversity
5. **edge_structure** — typed connect_to vs co_anchored, related_to overuse, aspect coverage
6. **voice_balance** — user_raw vs anchor_raw symmetry on identity-bearing types

### Brain + agent_call preservation

Each cell's brain DB lives at `~/AgentsContext/brain-eval-{run_name}-{stage}-v{version}/{qid}/`. Encoder + surface prompt/response dumps live at `eval/longmem/reports/{arm}/items/{qid}/agent_calls/`. Both gitignored, survive indefinitely. The per_cell.jsonl row carries `paths.brain_db`, `paths.artifacts_dir`, `paths.agent_calls_dir` for offline analysis.

---

## Health checks before resuming

```bash
# Active s1e + scouts
./dev python3 -c "
from servers.daemon_client import send_command
for name in ['s1e', 's1_scout_facts', 's1_scout_quote']:
    info = send_command('list_interactions', {})
    e = next(x for x in info['result'] if x['name'] == name)
    print(f'{name}: max=v{e[\"max_version\"]}, active=v{e[\"active_version\"]}')
"
# Expected: s1e max=v24 active=v22; facts max=v7 active=v5; quote max=v4 active=v2

# Targeted eval test suite
./dev pytest tests/test_remember_source_refs.py tests/test_encoder_eval_probes.py \
  tests/test_prompt_sync.py tests/test_episodic_refs_dal.py tests/test_schema_v27.py \
  tests/test_contract_sync.py tests/test_trace_contract_sync.py \
  tests/test_dispatch_contract_sync.py tests/test_mcp_roundtrip.py \
  tests/test_time_window_contract.py tests/test_dashboard_disconnection.py -q 2>&1 | tail -3
# Expected: 165 passed

# Production daemon health
./dev python3 -c "
from servers.daemon_client import send_command
print('count_traces:', send_command('count_traces', {}))
"
```

Signals that things are healthy:
- s1e active = 22, max = 24 (DORMANT candidate registered, not flipped)
- 165 v22-eval-gate tests pass
- Seed file `servers/scales/s1/encoding_prompt.py` header reads `Last sync: DB v22`

Signals that something regressed:
- s1e active != 22 unexpectedly
- Seed file out of sync with active (`./dev sync-prompts --check` non-zero)
- New `source_refs_hex_format` warnings in daemon log (suggests new encoder regression)

---

## The mission, restated

The episodic-references write path is shipped end-to-end. v22 is active, encoding source_refs uniformly (100% coverage on every cell of the 50-cell longmem real-test). co_anchored auto-edges form cohorts at write time. The encoder, scouts, dispatcher, validator, and substrate all participate in the joint reactivation contract per `docs/EPISODIC-REFERENCES.md`.

What's NOT yet live: render expansion at SURFACE_FORMAT (joint reactivation read shape), `source_summary` parallel-pathway recall scoring, S2Healer source_refs cleanup. These are recall-side work that comes online when v22's encoder has accumulated enough source-anchored nodes for the patterns to be measurable.

v24 + scout v7/v4 are the first dormant candidates after the Phase B substrate ship — they probe whether targeted prompt refinements (precision refinement vs. supersession, scout-scope boundaries) move the eval needle further. The methodological gain from this session is bigger than the candidate stack itself: **the encoder_eval infrastructure can now A/B any future version against any past version with parallel cells and substrate-aware probes**.

The discipline this session shipped:
- **Substrate-first, behavior-later** — Step 7 substrate (co_anchored auto-edge) lives independent of v22 prompt teaching
- **Probe-driven prompt iteration** — every prompt change goes through cold-read probes before activation
- **DORMANT-by-default with active-version sync** — dormant candidates cannot leak into the seed
- **Per-cell brain preservation** — every eval brain survives for offline analysis
- **N=1 noise awareness** — single-cell results have stochasticity floor; some 50-cell "fails" are tail outcomes

Re-eval gate for v24+v7+v4 is the next open call. Tom's preference: continue experimenting.
