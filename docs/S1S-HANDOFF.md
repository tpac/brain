# S1S rewrite — handoff for the testing session

**Status at handoff:** all scout infrastructure built and unit-tested
(182 tests green). S1S prompt rewrite drafted. Muster integration and
S1S prompt registration DEFERRED pending the encoding-quality A/B this
document prepares.

**Author:** Anchor session 2026-04-23. Work spanned multiple earlier
sessions — see S1 ARCHITECTURE.md for design history.

---

## What was built

### Code — all landed, all tested
- **[servers/scales/s1/orientation.py](servers/scales/s1/orientation.py)** — shared orientation labels (node catalog, surfaced nodes, conversation window, current date, session context, encoding journal). Single source of truth; both scouts and S1S read from here.
- **[servers/scales/s1/scouts/contract.py](servers/scales/s1/scouts/contract.py)** — scout I/O contract: `SCOUT_NAMES`, envelope schema, `build_shared_prefix`, `validate_scout_output`, `format_scout_report_for_s1s`.
- **[servers/scales/s1/scouts/base.py](servers/scales/s1/scouts/base.py)** — `run_llm_scout()` shared runner. Loads interaction, composes system+user content with cache layout, calls Anthropic, parses JSON, injects deterministic scout+category fields, validates, logs all failures to `brain_errors`. Never raises.
- **[servers/scales/s1/scouts/temporal.py](servers/scales/s1/scouts/temporal.py)** — algorithmic temporal scout. 6 extraction passes (dateparser + absolute Month-Day + modifier+weekday + modifier+unit + word-number + vague-quantifier + fuzzy-anchor). Detects relational markers (`just before`, `right after`, `during`). Catalog lookup for existing time_anchor reuse.
- **[servers/scales/s1/scouts/runners.py](servers/scales/s1/scouts/runners.py)** — uniform dispatch registry (`SCOUT_RUNNERS: {name → callable(brain, ctx)}`). Adapter shims for LLM scouts and temporal. Startup guard prevents drift vs `SCOUT_NAMES`.
- **[servers/scales/s1/scouts/muster.py](servers/scales/s1/scouts/muster.py)** — `build_muster_context()` + `run_muster()`. Parallel `ThreadPoolExecutor`, per-scout wall-clock deadline (90s), timeout stubs + exception stubs, shared Anthropic client across scouts, trace emission (O + K per scout on s1e chain).
- **[servers/scales/s1/scouts/prompts/](servers/scales/s1/scouts/prompts/)** — 4 seed prompts: quote_prompt.py, temporal_prompt.py (Haiku fallback reserved; not wired), facts_prompt.py, synthesis_prompt.py. Each exports `SYSTEM_PROMPT`.
- **[servers/interaction_seed.py](servers/interaction_seed.py)** — registers `s1_scout_quote`, `s1_scout_temporal`, `s1_scout_facts`, `s1_scout_synthesis` on fresh brain init with category_statement + config.
- **[servers/tools/sync_prompts.py](servers/tools/sync_prompts.py)** — extended `SEED_PROMPTS` list to sync scout templates DB ↔ seed files.
- **[servers/trace_contract.py](servers/trace_contract.py)** — added `scout_input` (S1 O) and `scout_findings` (S1 K) ref_types.
- **[servers/scales/s2/base.py](servers/scales/s2/base.py)** — `retry_on_transient_api_error` helper + wiring in community/consolidation encoders. Maintenance interval lowered 30 → 15 min ([servers/brain_constants.py](servers/brain_constants.py)).
- **[eval/longmem/classifier.py](eval/longmem/classifier.py)** — `PARTIAL_RECALL` split into `ENCODE_MISS` vs `RECALL_MISS` via direct brain scan. [eval/longmem/report.py](eval/longmem/report.py) updated to match.

### Prompt rewrite — drafted, NOT registered
- **[docs/S1S-PROMPT-REWRITE-DRAFT.md](docs/S1S-PROMPT-REWRITE-DRAFT.md)** — proposed `s1e` v13.
- Major moves (vs live v12):
  - `## What You Receive` pulls from `orientation.py` + adds `## Scout reports` bullet
  - Dropped 5 of 6 detection patterns (kept Corrections; folded Missing-grounding into a bullet)
  - Old temporal block (with time_anchor example) removed
  - New `## Temporal composition` section with 5 Allen relations + episodic parents + validity intervals
  - `## What Good Encoding Looks Like` — 4 transformations × 2 examples each (engineering + non-engineering pair: math, poetry, research, clinical)
  - Type-emergence paragraph in `## Fields` (reads catalog, reuses, invents, tag-inventory earns its shape by use)
  - Two conventional types named (`time_anchor`, `event`) as load-bearing-but-not-system
  - `concept` removed from special-behavior types (it had no system behavior)

### Tests — 182 green
`tests/test_scout_contract.py` · `tests/test_scout_llm_base.py` · `tests/test_scout_temporal.py` · `tests/test_scout_muster.py` · `tests/test_orientation_unified.py` · `tests/test_longmem_classifier.py` · `tests/test_prompt_sync.py` · `tests/test_s2_retry.py` · `tests/test_trace_contract_sync.py` · `tests/test_contract_sync.py`

---

## What remains — start the next session here

### Step 1: Register the new prompt
```bash
# Read the draft prompt
cat docs/S1S-PROMPT-REWRITE-DRAFT.md

# Register via MCP (in a session, use the brain tool directly):
# register_interaction(name='s1e', template=<paste prompt text>,
#                      parameters=<keep existing S1E_CONFIG>, created_by='anchor')

# Mirror DB → seed file
./dev python3 -m servers.tools.sync_prompts

# Verify
./dev sync-prompts --check
```

### Step 2: Wire muster into run_encoding
[servers/scales/s1/encode.py](servers/scales/s1/encode.py) — `run_encoding()` function.
Between `_build_user_content` and `run_llm_loop`, add:

```python
from servers.scales.s1.scouts.muster import build_muster_context, run_muster

# Extract catalog ids from the user_content build (minor refactor may be
# needed to expose them; currently _build_user_content returns only text).
muster_ctx = build_muster_context(
    brain=brain, messages=messages, session_id=session_id, counter=counter,
    catalog_rendered=catalog_text, catalog_node_ids=cataloged_ids,
    session_context=brain.get_config('session_context', ''),
    log_fn=_log,
)
scout_report, scout_outputs, muster_metrics = run_muster(muster_ctx)
user_content += "\n\n## Scout reports\n\n" + scout_report
```

Note: `_build_user_content` returns `user_content` string only. A small
refactor to also return `(catalog_text, cataloged_ids)` is needed, OR
call `build_node_catalog` a second time for muster.

### Step 3: Run the encoding-quality A/B

Existing frameworks to leverage — DO NOT rewrite these, adapt them:

| File | What it does | When to use |
|---|---|---|
| [eval/encoder_prompt_ab_eval.py](eval/encoder_prompt_ab_eval.py) | A/B two encoder prompts on same transcripts; compares node count, types, edge types, edge description quality | Closest fit. Adapt: A=v12 no-scouts, B=v13 with scouts |
| [eval/ab_test_prompts.py](eval/ab_test_prompts.py) | KPIs across variants: revise-vs-create, recall calls, tool efficiency, noise resistance | Use for broader KPI view after basic A/B passes |
| [eval/encoding_v3_compare.py](eval/encoding_v3_compare.py) | Side-by-side comparison using `InstrumentedBrain` from [eval/capabilities/base.py](eval/capabilities/base.py) | Reference for the InstrumentedBrain pattern; v3 era |
| [eval/test_extensive_encoding.py](eval/test_extensive_encoding.py) | 32KB — extensive encoding checks | Heavy; use later if quick A/B shows mixed signals |
| [eval/capabilities/base.py](eval/capabilities/base.py) | `InstrumentedBrain`, `CapabilityTest`, `CapturedAction` — the action-capture harness | Reuse directly for A/B |
| [eval/corpus/](eval/corpus/) | 6 conversation fixtures (architecture, debugging, philosophy, art/design, emotions, product) | Test set — spans domains per Tom's point |

### The A/B test to run

Target: ~5 conversations from `eval/corpus/` (or all 6 if cheap).

For each conversation:
1. Two isolated brains, both freshly seeded.
2. Feed the conversation to each:
   - **Control (A)**: current `s1e` v12 prompt, muster DISABLED. (Gates: set a flag or temporarily bypass the muster call.)
   - **Treatment (B)**: `s1e` v13 prompt, muster ENABLED.
3. Capture all encoder actions (remember, revise, connect) via `InstrumentedBrain`.
4. Compare on a quality rubric:

**Rubric dimensions** (shape inherited from Mar-Apr evals):
- **Node count** — B similar, higher, or lower than A?
- **Two registers** — do pair (principle + fact) combos appear with `grounds` edges?
- **Specificity retention** — numbers, names, exact quotes preserved?
- **Connection density** — edges per node; non-generic relations?
- **Edge description quality** — specific `why` vs generic/empty?
- **Node focus** — one concern per node, or compounding?
- **Operator voice** — `user_raw_quote` populated for distinctive phrasings?
- **Temporal composition** — time_anchor nodes created/reused correctly? `event_time` metadata populated? Allen relations used when appropriate?
- **Missing atoms** — is the operator's domain vocabulary being atomized? (e.g. if a conversation mentions a book title 3 times, does B create a node for the book?)

**Judge**: Haiku or Sonnet grading each dimension 0-2 (poor/ok/good) per
conversation. Or use the v12-era rubric if one exists in the prompt_ab_eval
outputs.

**Pass criteria** (proposal — confirm before running):
- B ties or beats A on ≥ 6 of 9 dimensions across ≥ 3 of 5 conversations.
- No single dimension drops by > 1 point on average.

### Step 4: 20-item longmem baseline with classifier split

After A/B passes:
```bash
./dev python3 eval/longmem/harness.py --items 4 --workers 5 --run_name v13_scouts_post_ab
```

Compare to the pre-scouts baseline from 2026-04-23
(`eval/longmem/reports/run_baseline_pre_scouts.*`): 11/20 = 55%,
dominated by PARTIAL_RECALL (8 of 9 failures). With the classifier fix,
the new run will split those into ENCODE_MISS vs RECALL_MISS. Target:
ENCODE_MISS lift from scouts' anchor / fact / quote atomization.

---

## Known risks / watchpoints

1. **Encoder-defers-to-scouts regression.** v13 prompt relies on priming
   over instruction — trusts the encoder to read the conversation AND
   the scout reports. If the encoder shortcuts to scout-only, we lose
   coverage. A/B rubric dimension "Operator voice + Node focus" will
   catch this (scout findings are narrow; full conversation reading is
   broader).

2. **Allen relations under-use.** Prompt has 5 relations (`before`,
   `after`, `meets`, `met_by`, `during`). If the A/B shows encoder only
   uses `before`/`after` and never `meets`, the adjacency precision is
   lost. Watch edge type distribution in the rubric.

3. **Temporal scout relational-marker false positives.** Regex for
   "before"/"after"/"during" can match non-relational uses. If
   `has_relational_marker` lights up for sentences that don't actually
   reference another event, S1S wastes work searching catalog. Inspect
   the scout trace output on 2-3 conversations before running the full
   A/B.

4. **Type-tag fragmentation.** v13 removed the `concept` directive. Fresh
   brain might end up with scattered grounding-node types (`term`,
   `vocabulary`, `entity`, `definition`). Not fatal — S2 community
   detection absorbs these into clusters — but worth tracking in the
   rubric.

5. **Muster ghost threads.** On a scout timeout, the blocked thread runs
   in the background until its own SDK timeout fires. Bounded by
   Anthropic's default (~600s). Verify no resource leak over many
   encoding cycles.

---

## Commands to run in the next session (cheat sheet)

```bash
# Full test suite
./dev pytest tests/test_scout_contract.py tests/test_scout_llm_base.py \
             tests/test_scout_temporal.py tests/test_scout_muster.py \
             tests/test_orientation_unified.py tests/test_longmem_classifier.py \
             tests/test_prompt_sync.py tests/test_s2_retry.py -q

# Sync prompts DB ↔ seed files
./dev sync-prompts --check

# Start the A/B (adapt encoder_prompt_ab_eval.py):
./dev python3 eval/encoder_prompt_ab_eval.py --transcripts 5 --verbose

# 20-item longmem baseline with new classifier split:
./dev python3 eval/longmem/harness.py --items 4 --workers 5 \
              --run_name v13_scouts_post_ab

# Dashboard while eval runs (read-only observer, separate window):
BRAIN_DB_DIR=~/AgentsContext/brain-eval-items \
  ./dev python3 dashboard/brain_dashboard_standalone.py
```

---

## References

- **S1 architecture & phasing**: [servers/scales/s1/ARCHITECTURE.md](servers/scales/s1/ARCHITECTURE.md) — Phase 1 (current, muster+scouts), Phase 2 (conditional — scribe enrichment path), Phase 3 (tune).
- **Scout prompt drafts** (seed files): [servers/scales/s1/scouts/prompts/](servers/scales/s1/scouts/prompts/)
- **Proposed new S1S prompt**: [docs/S1S-PROMPT-REWRITE-DRAFT.md](docs/S1S-PROMPT-REWRITE-DRAFT.md)
- **Longmem baseline pre-scouts**: [eval/longmem/reports/run_baseline_pre_scouts.md](eval/longmem/reports/run_baseline_pre_scouts.md) — 55% overall, 25% multi_session, 75% temporal+info_ext.
