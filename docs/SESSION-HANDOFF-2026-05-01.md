# Session handoff — 2026-05-01

Backup of the brain nodes (`c22ee2c4`, `6db31f12`, `f8459edb`). Read these first
when resuming. Three threads: shipped state, S2 fix, recall refining.

---

## Shipped this session (in production)

| | What | Commit |
|---|---|---|
| 1 | ORT arena leak fix — `enable_cpu_mem_arena=False` + `enable_mem_pattern=False` via flat fastembed kwargs (NOT `extra_session_options={...}` dict — that gets silently dropped). Plus `_EMBED_BATCH_CHUNK=64`. | `25502e4` |
| 2 | Facts scout v2 — "EITHER speaker" framing + Example 4 for assistant-introduced enumerations. Registered as prod `s1_scout_facts` v2. | `6ac31a8` |
| 3 | Hop-3 scrutiny gate — `HOP_SCRUTINY_DEFAULT = True`. At hop 3+ in spread, only top-half of activated nodes (median-derived) propagate. ~44% latency reduction. | `28b13c5` |
| 4 | Variant verification logging — one stderr line per process at first `_graph_expand` call. | `4ab5c23` |

**Net vs session start:**

- Recall p50 latency: ~10s → ~6s
- Recall p95: 22s+ (timeouts) → ~7-8s (no timeouts expected)
- Quality canonical (seed=42): 13/15 (87%) → 14/15 (93%)
- Quality different sample (seed=1): 12/15 (80%) → 12/15 (80%) (parity, no regression)
- Peak RSS during recall: 5.8 GB (leaking) → 2.6 GB (releases)

## Tried + rolled back

`lim=30` neighbor cap: shipped briefly in `7146b7f`, rolled back in `28b13c5`.

- iter_M (lim=30+scrutiny, seed=42): 14/15 — looked clean
- iter_N (lim=30+scrutiny, seed=1): 9/15 — different sample regression
- iter_O (lim=50, no scrutiny, seed=1): 12/15 — confirmed iter_N regression real
- Mechanism: lim=30 narrows muster recall during ingestion → encoder misses specific facts. The latency win lives in scrutiny alone.

## Lessons learned

1. **fastembed silent-drop API**: session options must be passed as flat kwargs to `TextEmbedding(model_name=..., **session_kwargs)`. Inside `extra_session_options={...}` they get dropped by `_select_exposed_session_options(kwargs)`. Three "fixes" (bucket-pad, spin keys, mem_pattern attempt-1) had been silent no-ops because of this.

2. **Sample variance at N=15 is ~20pp.** Cross-sample confirmation (M + N) is required before declaring quality holds. iter_M alone was overconfident.

3. **Spread-layer variants (D/E/G/I) all regressed below baseline.** Spread isn't the bottleneck.

4. **Recall during ingestion shapes encoding.** Anything that narrows muster recall risks degrading what gets encoded. lim=30 was an instance.

5. **Lexical bridge via pre-cosine Haiku expansion fails on brain-vocabulary gaps** (feed/scratch grains). Haiku doesn't know what's in the brain. Encode-time enrichment is the architecturally correct fix.

---

## NEXT SESSION → Thread 1: Fix S2 starvation

S2 fires rarely. Observed 2.5-day gap between 04-28 and 04-30 04-30 → 05-01 runs. Mechanism is correct, FREQUENCY is the problem.

### Why it stalls

`brain.run_maintenance_if_due(last_activity_ts)` (`servers/brain.py:1133`) gates on:
- `idle_seconds < 3 min` → return None
- `since_last_run < 15 min` → return None

`last_activity` is reset on every `_handle_client` call (`servers/daemon_server.py:323, 362`). Includes:
- `hook_recall` (real user activity ✓)
- `hook_pre_edit`, `hook_post_response_track` (Claude doing things, NOT Tom typing)
- internal IPC: `set_config`, `trace_append`, `ping`, `hook_idle_maintenance` polls themselves

In active dev sessions, these hit the daemon every <30s. The 3-min idle window almost never opens.

Aggravators:
- Daemon restarts reset `last_activity` (line 49 init)
- No safety valve for multi-day stale state
- `_s2_running` flag could deadlock if maintenance hangs

### Fix shape — three options

**Option A (recommended — surgical):** Filter what counts as "activity." Only `hook_recall` resets `last_activity`. Tool-use hooks, pings, internal commands don't.

Code locations:
- `servers/daemon_server.py:323, 362` — currently `self.last_activity = time.time()` unconditionally. Add a check: only reset for specific hook types. Or have hooks pass a flag indicating "this is a user action."
- Or: keep generic activity tracking but ALSO track `last_user_prompt_ts` separately, use THAT in `run_maintenance_if_due`.

**Option B (cheap polish):** Persist `last_activity` across daemon restarts via `brain_meta` alongside `s2_last_run_ts`. Restart shouldn't grant fresh 3-min grace.

**Option C (safety net):** Force-fire if `since_last_run > 24h` regardless of idle. Catches multi-day backlog.

**Recommendation:** A + C minimum. B if cheap.

### Implications of starvation today

- New nodes don't get community placement → recall doesn't have community signal
- Corrections/consolidations don't fire when relevant
- Healer doesn't fill missing fields on freshly-encoded nodes
- Edge family classification stays stale on new edge types
- "Brain feels stale" symptoms after multi-day gaps

---

## NEXT SESSION → Thread 2: Recall refining (after S2 fix)

In priority order:

### 1. Lexical bridge — encode-time, not query-time

Pre-cosine Haiku expansion failed on bc149d6b (feed/scratch grains) because Haiku doesn't know brain vocab. Right move:

- New stage in S1E: after main writes, generate 1-3 broader handles per node via Haiku
- Store as new vector type in `node_enrichments` (e.g. `_broader_handles`)
- Existing recall picks them up via STEP 3.5 multi-vector scoring
- Cost: ~1 Haiku call per encoded node, amortized forever

Skip if facts scout v2's broader extraction already captures these (worth checking).

### 2. Answerer aggregation + recency (eval-only impact)

Items where right node retrieved but answerer didn't compute:
- 8e91e7d9 (siblings 1+3=4) — needs aggregation across nodes
- 59524333 (gym 6PM newer than 7PM) — needs recency-prefer-newer
- 71017276 (chandelier "weeks ago") — needs date arithmetic

`eval/longmem/answerer.py` `ANSWERER_SYSTEM` prompt change. Improves measurement, not Anchor.

### 3. Render full-content for high-activation seeds

3b6f954b: right node retrieved, content truncated before "University of Melbourne." Render budget allocation in `format_surface_output_activation` should always include fact-shaped handles regardless of softmax budget pressure. Small render-layer change.

### 4. Hub dampening / specialized-node lift

Flat embedding space (top-25 spread ~0.09 per `dea1a002`). 93% of nodes never recalled (`0591813f`). Approaches:
- Per-edge-family z-score normalization (HippoRAG-inspired)
- Per-field cosine: max across title/content/situation, not weighted sum
- Hybrid keyword (BM25/SPLADE) for token-level specificity

### 5. Surface judge prompt: "no direct match but related"

Tom's flag: judge can abstain AND surface adjacent. For abstention queries, framing "no direct match for X but Y is closest" in selected_why. Surfacer prompt change.

### 6. Judge variance noise

bc8a6e93_abs flapped YES/NO with semantically-equivalent answers. Either accept as N=15 noise or move to multi-token judge.

---

## Files / commits to read first

- `servers/scales/s1/surface_contract.py` — spread_activation, scrutiny gate, contract constants
- `servers/scales/s1/scouts/prompts/facts_prompt.py` — facts scout v2 (synced from prod)
- `servers/brain_recall.py` — STEP 3 cosine, STEP 4.5 FTS5, conditional expansion gate
- `servers/brain.py:1133` — `run_maintenance_if_due`
- `servers/daemon_server.py:275-300, 645-672` — `_main_loop` polling + `_run_idle_maintenance`
- `eval/longmem/harness.py` — 15-item stratified eval
- `docs/archive/MEMORY-LEAK-INVESTIGATION-2026-04-27.md` — fastembed flat-kwargs postmortem

## Eval scoreboard (15-item LongMemEval, seed=42 unless noted)

| Variant | Score | Note |
|---|---|---|
| A baseline | 13/15 (87%) | reference |
| D L4 lane | 11/15 (73%) | regressed |
| E lim=15 | 9/15 (60%) | regressed |
| G lineage-pass | 11/15 (73%) | regressed |
| I edge thickness | 11/15 (73%) | regressed |
| **J facts v2** | **14/15 (93%)** | shipped |
| M lim=30+scrutiny | 14/15 (93%) | rolled back |
| N lim=30+scrutiny seed=1 | 9/15 (60%) | drove rollback |
| O baseline seed=1 | 12/15 (80%) | confirmed regression |

## Bench (prod-clone, 3,221 nodes, full hook_recall)

| Variant | avg | p95 |
|---|---|---|
| A lim=50, no scrutiny (old prod) | 11,170ms | 11,360ms |
| B lim=30, no scrutiny | 7,319ms | 11,802ms |
| C lim=30 + scrutiny | 5,944ms | 6,491ms |
| **D lim=50 + scrutiny** (live now) | 6,270ms | 7,902ms |

---

Brain node IDs for cross-reference: `c22ee2c4`, `6db31f12`, `f8459edb` (locked).
