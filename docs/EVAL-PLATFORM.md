# Eval Platform — Reference for Digging In

**Last updated:** 2026-07-29 — LAF walker section added (the recall-lane offline substrate now has a documented home here; role-expansion artifacts included). Prior: 2026-05-30 — Frozen Corpus matured: sweep now **scores every item** with a **recall-conditional** rate (the gate stopped hard-excluding composed answers), **`--interaction-override`** added for DORMANT-version A/B, first 20-item baseline run. Prior: 2026-05-29 Frozen Corpus two-stage architecture; 2026-05-10 artifacts/analyzer/run-diff/harness reliability.

Originally built 2026-04-25/26 as the bias-detection broader eval. Now expanded into a deeper diagnostic platform with per-item artifact bundles for post-hoc analysis without re-running.

---

## LAF walker — the recall-lane offline substrate (`eval/laf/walker/`)

The walker is the OTHER eval platform: offline replay of recall over cached
per-turn lane values, for LAF composition work (reach@K over the corpus-v2
verdicts; doors discipline: door-1 `cue` / door-2 `window`+`session`, fit and
scored per door — `laf_doors.py`). Full conventions live in the scripts'
docstrings and `docs/RECALL-SR-REDESIGN.md` §19–20; corpus truth in
`corpus_v2_synthesis.md`.

**Artifact chain (check `build_meta` stamps before trusting any of it):**
`walker.db` (turns/candidates/lanes, local artifact, never committed) →
`field_cache.npy` + `lane_cache.npy` + `field_cache_index.json`
(`field_cache_build.py`; dense per-turn per-node lanes — the reach substrate)
→ `cand_turn_episodic` (v1 pick/enc roles, `episodic_roles.py`) →
`cand_turn_episodic_roles` + `roles_lane_cache.npy` (v2 conn/auth roles,
`episodic_roles_v2.py` + `roles_lane_cache.py`, 2026-07-29 role-expansion
arc; backfill provenance + validation in `role_backfill_audit.py`).

**Before building new lane/role machinery:** the moment-selection, role-join,
support-z, per-door fit, and paired-bootstrap machinery all exist — extend
`laf_doors.py`/`role_arms.py` patterns, don't re-derive. Vectors never need
rebuilding for a role change (it's a re-join; see `roles_lane_cache.py`
docstring).

---

## Frozen Corpus — the two-stage architecture (2026-05-29)

**The problem it solves.** The single-pass harness (`harness.py:run_item`) threw the
encoded brain away and rebuilt it on every item, every A/B arm, every variance
replicate, every experiment. Encode is **~99%** of an item's cost (~231s/item;
the scored query→answer→judge is ~1%). That one fact caused both chronic eval
pains: **hours of runtime**, and **blurred A/B attribution** (both arms re-encode
*and* re-recall, so a delta mixes encoder + retriever + answerer + LLM noise —
you can't say which stage moved).

**The fix: split at the encode/recall seam, with a durable artifact between.**

```
build_corpus.py  (Stage 1, slow, ONCE)        sweep.py  (Stage 2, fast, MANY)
  replay haystack → full S0/S1/S2               load frozen brain (copy)
  freeze each per-item brain on disk    ──▶     run only query → answer → judge
  content-address by config hash               under a recall/surface config
  write manifest (answerability, S2 Δ,          ~100× cheaper; both A/B arms
  build errors)                                 start from byte-identical graphs
```

- **Stage 1 — `build_corpus.py`.** Replays each item's haystack through the real
  S0/S1/S2 loop and **keeps** the encoded brain at
  `~/AgentsContext/eval-corpus/{corpus_hash}/{qid}/`. Content-addressed by
  `(s1e, ingest-surface, s2_every_n, oracle, qids)` — a re-run with the same
  inputs is a **cache hit** (0 re-encoding). Manifest records, per item: the
  **answerability gate** (gold-scan on the frozen brain → answerable/ENCODE_MISS,
  Tom's spec dims 1+2), the **S2 Δ** (consolidation/community/healer
  fires/work/actions/errors — S2 is a subject under test), and **`build_errors`**
  (rows in `debug_log` during the build). Manifest is written **incrementally**
  after each item, so a crash mid-build preserves completed items.
- **Stage 2 — `sweep.py`.** Copies each frozen brain into a per-run work dir
  (so both A/B arms start from identical bytes and the query phase can't mutate
  the shared corpus), then runs only the cheap suffix, optionally `--variance N`.
  **Scores every item** — it does NOT hard-exclude on the gate's gold-scan, which
  single-node keyword-AND false-negatives on composed/multi-session answers (a
  13-node, on-topic item once scanned "unanswerable"). Instead the per-item failure
  bucket separates genuine `ENCODE_MISS` from recall failures, and the report carries
  both a raw pass rate and a **`recall_conditional`** rate (drops `ENCODE_MISS` from
  the denominator). Abstention items are scored too (bucket `ABSTENTION_FAIL` on a
  miss, not a spurious `ENCODE_MISS`). Emits **harness-shape artifacts**
  (`reports/{run}/items/{qid}/{result,meta,recall}.json`) so `compare_arms.py` +
  `cost_summary.py` work unchanged (encoder cost reads `$0` — correct, the sweep
  doesn't encode). Adds the **S2-reached-recall** probe: do S2-origin nodes
  (`encoding_source LIKE 's2:%'`) show up in candidates / selected? — the direct
  readout for the communities-not-in-recall bug.

**Run it:**

```bash
# Stage 1 — build once (content-addressed; cache-hit skips rebuild)
./dev python3 eval/longmem/build_corpus.py --items 4 --label dev20_baseline
#   → prints corpus_hash, e.g. a300d2

# Stage 2 — sweep many times against the frozen corpus
./dev python3 eval/longmem/sweep.py --corpus a300d2 --label armA_v8
./dev python3 eval/longmem/sweep.py --corpus a300d2 --surface eval/prompts/surface_v9.txt --label armB_v9 --variance 3

# Compare (unchanged — reads the harness-shape artifacts the sweep wrote)
./dev python3 eval/longmem/compare_arms.py armA_v8 armB_v9 --labels v8,v9 --out-dir eval/longmem/reports/ab_v8_v9
```

- **Recall experiment:** same `corpus_hash`, different `--surface` → delta is
  *pure recall* (encode held byte-identical). The baseline sweep persists by
  name, so the control is never recomputed.
- **Encode experiment** (e.g. v22 vs v24+v7): build *two* corpora, the treatment
  with `--interaction-override 's1e=24,s1_scout_facts=7'` — fetches the DORMANT
  versions from the live daemon and registers+activates them in each isolated eval
  brain (part of the corpus hash, so the arms get distinct addresses). Sweep both
  with the same recall config → delta is *pure encode*. (`--s1e <file>` still works
  for an unregistered draft prompt.)

**First baseline (corpus `a300d2`, v22/v5, 20 items, 2026-05-30):** 94.4%
recall-conditional (17/18), 85% raw. The only two clean misses were `ENCODE_MISS`
(encoder wrote 0 nodes), not recall — recall reads strong, consistent with v22's
50-cell headline. A 2-item v24+v7 A/B (`--interaction-override`) on those two
skip items showed facts-scout v7 going 0→6 candidates and v24 encoding nodes v22
dropped, which (with the prior 9/10 targeted eval) drove activation of the
**v24 + scout-facts v7 + scout-quote v4** bundle in production.

**Files:** `eval/longmem/corpus.py` (content-addressing + S2-Δ aggregation),
`build_corpus.py` (Stage 1), `sweep.py` (Stage 2). Reuses `replay.py`,
`answerer.py`, `judge.py`, `classifier.py`, `fresh_brain.py` from the layers
below. Tests: `tests/test_eval_corpus.py`.

**Known follow-ups:** (1) `build_corpus` lacks per-item try/except — one
unhandled exception halts the loop (incremental save preserves prior items, but
it won't *continue* past a bad item yet). (2) `sweep` opens one Brain per qid;
the embedder is already a process singleton so per-qid overhead is ~18ms (a
non-issue), but Brain construction is not shared. (3) The S2-reached-recall
`selected`-id match is best-effort (candidate-id match is solid).

**Surfaced during the build-out:** the S2 community encoder can leave a
transaction open on `self.conn` (deferred isolation), making the next
`brain_batch` `BEGIN IMMEDIATE` throw *"cannot start a transaction within a
transaction"*. Interim guard shipped in `brain_batch`; root fix in
[docs/WRITE-TXN-ISOLATION-ROOTFIX.md](WRITE-TXN-ISOLATION-ROOTFIX.md) (BACKLOG F3).

---

## TL;DR

**Primary path for recall/encode/surface experiments → the Frozen Corpus (section above).**
Build a content-addressed corpus once (`build_corpus.py`), then sweep it cheaply
and repeatedly (`sweep.py`). Encode is paid once; A/B attribution is clean.

The supporting layers (the diagnostic substrate the corpus build + sweep reuse):

1. **`eval/longmem/harness.py`** — the original single-pass harness (encodes +
   queries every run). Reliability guardrails: `_preflight()`, streaming JSONL,
   `--smoke-test`, `--qids`, `--s1e-override`. Still handy for a quick one-shot run.
2. **`eval/longmem/artifacts.py`** — durable per-item bundles captured BEFORE brain
   cleanup. Schema-stable, scale-agnostic. The sweep emits the same shape.
3. **`eval/longmem/analyzer.py` + `run_diff.py` + `compare_arms.py` + `cost_summary.py`**
   — analysis: refined-bucket classification, side-by-side run comparison, cost/latency.
4. **`eval/full_suite.py`** — legacy broad-eval orchestrator (3 sub-suites).
   Historical; see the 2026-04-25/26 section near the bottom.

**Single most important file for cold-start digging:**
- Corpus state (answerability + S2 Δ + build errors): `~/AgentsContext/eval-corpus/{corpus_hash}/manifest.json`
- Aggregate metrics: `eval/longmem/reports/run_{run_name}.json`
- Per-item investigation: `eval/longmem/reports/{run_name}/items/{qid}/` (full artifact bundle)

> **Current vs historical:** the Frozen Corpus section (above) and the 2026-05-10
> artifacts/analyzer section (below) are current. Everything from **"Legacy context
> (2026-04-25/26)"** onward — the component map, the 44.4% headline, the `full_suite`
> commands, the preserved run dirs — is a historical snapshot kept for methodology.
> Current longmem baseline is **~92%** (v22, 50-cell; brain `id:b8247ed2`), not 44.4%.

---

## What's new — 2026-05-10

### Artifacts subsystem (Phase 1, scale-agnostic)

Every eval run produces per-item bundles under `eval/longmem/reports/{run_name}/items/{qid}/`:

```
├── meta.json            — qid, axis, question, gold, haystack metadata
├── interactions.jsonl   — every interaction version this brain saw
├── traces.jsonl         — every trace_event (S0/S1/S2/...) with full metadata
├── nodes.jsonl          — every active node with full content + KV
├── edges.jsonl          — edge_relations with descriptions
├── recall.json          — query, top-N candidates with scores, selected,
                            classifier evidence augmented with
                            `fact_node_ranks_in_candidates`
└── result.json          — harness result + judge verdict
```

~1 MB per item. Auto-captured BEFORE per-item brain cleanup. Wrapped in try/except so artifact failures NEVER kill the eval. See [eval/ARTIFACTS.md](../eval/ARTIFACTS.md) for the full schema + investigation playbook + extension guide (S0/S2 use the same machinery via `dump_nodes(prefix='pre_unit')` checkpoints).

### Refined-bucket analyzer

`eval/longmem/analyzer.py` walks artifact bundles and produces 11-bucket refined classification, distinguishing for example:
- `encoder_filtered` — encoder ran but explicitly chose 0 nodes (e.g. "no stake/padding" journal language)
- `encoder_partial` — encoder wrote nodes, gold-bearing fact not in any of them
- `ranker_buried` — fact-bearing node exists in brain but NOT in top-N candidates
- `surface_skipped` — fact-bearing node was in candidates, surface didn't pick

This is sharper than the original 4-bucket classifier and is what should be used for new analyses.

### Run-diff for side-by-side comparison

`eval/longmem/run_diff.py <run_a> <run_b>` produces `eval/longmem/reports/diff_{a}_vs_{b}/comparison.md` with:
- Per-item pass/fail with movement (`unchanged_pass`, `unchanged_fail`, `fail→pass`, `pass→fail`)
- Per-axis pass rate shifts
- Failure bucket distribution shift
- Behavioral signal totals (anchor_raw_quote count, open-type node count, third-party quote node count, scout-handoff entity field count)
- Sanity check on which prompt version was actually used (from interactions.jsonl)

Falls back to `results_{run}.jsonl` for runs without artifacts (e.g. pre-Phase-1 runs).

### Harness reliability features

In `eval/longmem/harness.py`:
- **`_preflight()`** — checks `ANTHROPIC_API_KEY`, oracle, disk space, embedder import. Fails fast.
- **Streaming JSONL writes** — `hypotheses_{run}.jsonl` + `results_{run}.jsonl` append after each item. Crash at item N preserves items 1..N-1.
- **`--smoke-test`** — 1 item per axis × 5 axes serial, ~3-5 min wall, exits non-zero on pipeline failures. Run BEFORE any long run.
- **`--qids`** — comma-separated qid list overrides stratified sampling for targeted re-runs.
- **`--s1e-override <path>`** — registers an alternate s1e prompt over the seeded v1 in each fresh brain. Used to test prompt revisions without touching the seed file.

### Encoder prompt probe + diff

`eval/encoder_prompt_probe.py <path>` sends a prompt to 5 clean Sonnets in parallel, each interviewed on a different aspect (goal, edge_cases, emphasis, voice, bias_surface). `eval/encoder_prompt_diff.py <json_a> <json_b>` produces side-by-side reports.

Used 2026-05-10 for the v14 → v15.3 evolution. Per-iteration cost ~$0.50 (5 Sonnet calls × ~10K tokens each). Add an Opus review pass for stated-intent-vs-result verification at ~$2 per iteration.

### Hand-curated realchat corpus

`eval/longmem/realchat_extractor.py` reads `~/.claude/projects/.../*.jsonl`, strips system-reminders and `[BRAIN]` blocks and sidechain entries, produces clean per-session exchange lists. `eval/longmem/realchat_corpus.py` builds axis-aligned eval items from a hand-curated list. 15 items currently at `eval/longmem/data/realchat_oracle.json` (3 per axis). Gold-quality has known issues — gold tends to be about meta-framing while the brain encodes substance. Refinement deferred.

---

## Legacy context (the original 2026-04-25/26 platform)

> ⚠ **HISTORICAL — 2026-04-25/26 snapshot.** Everything from here to the end of the
> doc describes the original `full_suite` broad eval. It's kept for methodology and
> for the dig-in recipes (which generalize), but the numbers, run dirs, and exact
> commands are from that first run. **Current longmem baseline is ~92% (v22), not the
> 44.4% below**, and the `brain-eval-full_20260425_*` dirs have long since been cleaned
> up. For how to run today, use the Frozen Corpus section at the top.

Originally `eval/full_suite.py` ran three sub-suites against isolated brain copies. First run revealed **44.4% on broad longmem** vs **84% on cherry-picked N=5** — confirming the over-fitting risk Tom flagged. The 30 broad items × 3 variance brain dirs are preserved for forensic analysis.

---

## The platform (component map)

```
┌─────────────────────────────────────────────────────────────────┐
│ eval/full_suite.py        — orchestrator (THIS IS THE ENTRY POINT)
│   │
│   ├─ Suite 1: longmem_broad  (run via ProcessPoolExecutor, N workers)
│   │   └─ uses run_item from eval/s1s_full_e2e.py
│   │       └─ replay_item, query_brain, answer_question, judge_one
│   │
│   ├─ Suite 2: abstention_battery  (synthetic queries, same harness)
│   │   └─ build_abstention_items() — reuses 0862e8bf source haystack
│   │
│   └─ Suite 3: snapshot_replay  (3 subprocesses)
│       └─ eval/s1s_snapshot_replay.py per conversation
│           └─ replays a real session JSONL through v14+SPLIT against
│              an old prod snapshot
└─────────────────────────────────────────────────────────────────┘
```

### Key files

| File | What it does |
|---|---|
| `eval/full_suite.py` | Orchestrator. Hardcoded picks + harness wiring. |
| `eval/s1s_full_e2e.py` | `run_item()` — fresh seed brain, ingest, query, judge. |
| `eval/s1s_snapshot_replay.py` | Replay a real conversation against a snapshot copy. |
| `eval/longmem/replay.py` | `replay_item()` — feeds haystack sessions through encoder. |
| `eval/longmem/answerer.py` | `answer_question()` — Sonnet generates hypothesis from `additional_context`. |
| `eval/longmem/judge.py` | `judge_one()` — Haiku judges hypothesis vs gold answer. |
| `eval/longmem/fresh_brain.py` | `create_fresh_eval_brain()` + `per_item_brain_dir()` — brain isolation. |
| `eval/longmem/data/longmemeval_oracle.json` | The 500-item dataset (6 categories). |
| `eval/s1s_v13_prompt.py` | `extract_v13_prompt()` — pulls v14+SPLIT body from `docs/S1S-PROMPT-REWRITE-DRAFT.md`. |

---

## What got tested 2026-04-25/26

### Suite 1: Longmem broad (90 jobs)

Items per category (`LONGMEM_BROAD_PICKS` in `eval/full_suite.py`):

| Category | Picked qids |
|---|---|
| temporal-reasoning | `0bb5a684`, `08f4fc43`, `2c63a862`, `2a1811e2`, `bbf86515` |
| multi-session | `0a995998`, `6d550036`, `b5ef892d`, `e831120c`, `3a704032` |
| knowledge-update | `6a1eabeb`, `6aeb4375`, `830ce83f`, `852ce960`, `945e3d21` |
| single-session-preference | `8a2466db`, `06878be2`, `75832dbd`, `0edc2aef`, `35a27287` |
| single-session-assistant | `7161e7e2`, `c4f10528`, `89527b6b`, `e9327a54`, `4c36ccef` |
| single-session-user | `e47becba`, `118b2229`, `51a45a95`, `58bf7951`, `1e043500` |

5 items × 3 variance runs = 15 jobs/category × 6 = **90 jobs**.

### Suite 2: Abstention battery (NOT RUN this round)

5 synthetic abstention queries against the `0862e8bf` source (Luna
the cat conversation). Each at 3 variance = 15 jobs. Skipped per
operator direction.

| Query | Tests |
|---|---|
| `What is the name of my hamster?` | pet → pet adjacency |
| `What's my dog's name?` | pet → pet (different) |
| `What is my doctor's name?` | person adjacency (vet ≠ doctor) |
| `What city did I visit recently?` | place adjacency (no travel mentioned) |
| `What kind of car do I drive?` | item adjacency (no car mentioned) |

### Suite 3: Snapshot replay (3 conversations, all ended early)

| Conversation | Source date | Cycles in prod | Cycles captured |
|---|---|---|---|
| `71857713-2390-414d-9d51-1ef1de652d90` | Apr 24 | 11 | 0 (API connection error) |
| `eba17631-1caf-4f2c-a4ef-245a132f1862` | Apr 22 | 12 | 3 (stopped on direction) |
| `fd829e08-35b9-408b-9ea6-d50cc9e19aec` | Apr 24 | 13 | 1 (API connection error) |

Snapshot used: `~/AgentsContext/brain/brain.db.bak-pre-situation-migration`
(Apr 19 prod state, 83 MB).

**Failure mode worth noting:** running 3 snapshot replays in parallel with
25 longmem workers all hitting Sonnet caused API rate limiting / connection
errors. The replay's encoder doesn't have retry/backoff — needs to adopt
the pattern from commit `4763946` (`S2 resilience`). Logged as **B+1.19**
in PHASE-B+1-BACKLOG.md.

---

## Where everything is preserved (post-run)

### Run dir
```
eval/reports/full_suite/full_20260425_224942/
├── _run.log.preserved         # FULL log of the run — ALL job lines
└── (summary.json absent — script killed before aggregate phase)
```

### Per-item brain copies (the diggable gold)
```
~/AgentsContext/brain-eval-full_20260425_224942/
├── 0bb5a684-r0/                    # qid-r{variance_idx}
│   ├── brain.db                    # full encoded state
│   ├── brain_logs.db               # all traces
│   └── (eval-related artifacts)
├── 0bb5a684-r1/
├── 0bb5a684-r2/
├── 08f4fc43-r0/
├── ... (one dir per qid × variance, ≈90 dirs)
```

Each `brain.db` contains the encoded nodes, edges, vectors. Each
`brain_logs.db` has the full trace history (s0, s1r, s1e, s2 events).

### Snapshot replay outputs
```
eval/reports/snapshot_replay/
├── full_20260425_224942__71857713/
│   ├── brain.db                    # snapshot copy + partial replay state
│   ├── brain_logs.db               # traces
│   └── summary.json                # killed flag + 0 cycles
├── full_20260425_224942__eba17631/  # (3 cycles captured)
├── full_20260425_224942__fd829e08/  # (1 cycle captured)
├── replay_71857713/                 # earlier-today's good run (13 cycles)
└── replay_smoke_1/                  # smoke test
```

The `replay_71857713` dir from earlier today has the GOOD replay we
analyzed — 13 cycles, 72 nodes, full data. Compare against the
killed full_suite run to see what good replay output looks like.

---

## Headline result (the bias check)

```
Total: 40/90 = 44.4% (vs 84% on narrow N=5)

Axis                          Correct  Δ vs narrow N=5
single-session-user           11/15 (73%)  -27pp
temporal-reasoning            10/15 (67%)  -33pp
single-session-assistant       7/15 (47%)  (not tested earlier)
knowledge-update               5/15 (33%)  -67pp ←
multi-session                  4/15 (27%)  -53pp ←
single-session-preference      3/15 (20%)  (not tested earlier)
```

**The drops on knowledge-update and multi-session are dramatic** — the
original `LONGMEM_PICKS` had unusually favorable items. Real performance
is roughly half what we celebrated this morning.

---

## How to dig in (next session)

### 1. Find a specific item's verdict
```bash
grep "single-session-preference" \
  eval/reports/full_suite/full_20260425_224942/_run.log.preserved \
  | head -20
# Each line: [longmem_broad] N/90 ✓|✗|! axis qid Wms
```

### 2. Inspect what got encoded for a failed item
```bash
qid="0edc2aef"
brain_dir=~/AgentsContext/brain-eval-full_20260425_224942/${qid}-r0
sqlite3 $brain_dir/brain.db <<'SQL'
.headers on
SELECT type, title, length(content) AS clen
FROM nodes WHERE archived=0 AND encoding_source LIKE 'encoder:%'
ORDER BY created_at;
SQL
```

### 3. Pull the full trace chain for a job
```python
import sqlite3, json, os
qid = '0edc2aef'
v = 0
db = os.path.expanduser(f'~/AgentsContext/brain-eval-full_20260425_224942/{qid}-r{v}/brain_logs.db')
con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
for chain_id, rt, summary, md in con.execute(
    "SELECT chain_id, ref_type, summary, metadata FROM trace_events ORDER BY id"):
    print(f"[{rt:20s}] {summary[:120]}")
```

### 4. Find what was surfaced for the query (recall O event)
The s1r chain has the query and 25 candidates per recall. Look for
`ref_type='recall'` then parse the `metadata.candidates` field — list of
`id|title|score|type` strings.

### 5. Pull the full hypothesis + judge verdict for a job
The log has `✓`/`✗` marks but not the full hypothesis text. To recover
the full answerer output for a specific item, you'd need to:
- Re-run `answer_question` against the preserved brain (cheap — just 1
  Haiku call)
- OR re-run the full `run_item` with the preserved haystack

The harness's per-job result dict includes:
```python
{
    'qid': ..., 'axis': ..., 'arm': 'B',
    'question': ..., 'gold': ..., 'hypothesis': ...,
    'correct': bool, 'abstained': bool, 'has_context': bool,
    'judge_raw': ..., 'additional_context_chars': int,
    'query_s1r_ms': int, 'answer_ms': int,
    'n_nodes_created': int, 'n_edges_created': int,
    'brain_dir': str,
}
```
That's what gets written to `results.jsonl` at aggregate time. We
LOST those in-memory results when the script was killed before
aggregate. Recoverable by re-running individual items.

### 6. Per-axis quality dimensions (for preserved brains)
```bash
./dev python3 eval/s1s_ab_quality_analyzer.py \
  --run-dir eval/reports/snapshot_replay/replay_71857713
# Reports: noun retention, two-register %, edge desc median chars, etc.
```

---

## How to re-run

### Re-run the full broad eval (or a subset)
```bash
# Full medium suite (90 longmem + 15 abstention + 3 snapshot)
./dev python3 eval/full_suite.py \
  --run-name preflight_$(date +%Y%m%d_%H%M%S) \
  --variance 3 --longmem-workers 25 --abstention-workers 10

# Skip snapshot replays (avoid API contention)
./dev python3 eval/full_suite.py \
  --run-name longmem_only \
  --skip-snapshot

# Skip abstention specifically
./dev python3 eval/full_suite.py \
  --run-name no_abstention \
  --skip-abstention

# Just snapshot replays (recommended: serialize, not parallel)
for conv in 71857713 eba17631 fd829e08; do
  ./dev python3 eval/s1s_snapshot_replay.py \
    --snapshot ~/AgentsContext/brain/brain.db.bak-pre-situation-migration \
    --conversation ~/.claude/projects/-Users-tpac-brain/${conv}*.jsonl \
    --run-name solo_${conv}
done
```

### Re-run a single item (for forensic deep-dive)
The harness's `run_item` is callable directly:
```python
import sys; sys.path.insert(0, '/Users/tpac/brain')
import json
from eval.s1s_full_e2e import run_item

dataset = json.load(open('/Users/tpac/brain/eval/longmem/data/longmemeval_oracle.json'))
item = next(it for it in dataset if it['question_id'] == '0edc2aef')
item['_axis'] = 'single-session-preference'
item['_variance_idx'] = 0
result = run_item(item, run_name='dive_0edc2aef', arm='B')
print(json.dumps(result, default=str, indent=2))
```

This re-creates a fresh brain in `~/AgentsContext/brain-eval-dive_0edc2aef/`
and runs the full pipeline. Result dict includes hypothesis, judge_raw,
all metrics.

---

## Known gaps / what to dig into next

### A. Per-axis failure mode analysis (the big ask)

For each weak axis (preference 20%, multi-session 27%, knowledge-update 33%),
sample 2-3 failures, inspect:
1. **Did the brain encode the right info?** (read brain.db nodes)
2. **Did recall surface it?** (read s1r `recall` candidates trace)
3. **Did surfacer pick it?** (read `surface_selected` K event)
4. **Did answerer compose correctly?** (re-run `answer_question` and
   compare to hypothesis)

This separates encoder failures from recall failures from answerer failures.
We currently don't know the breakdown per axis.

### B. Abstention battery (deferred, unfinished)

Synthetic queries already coded in `eval/full_suite.py`. To run:
```bash
./dev python3 eval/full_suite.py --run-name abstention_only \
  --skip-snapshot --skip-longmem --abstention-workers 10
```
Earlier today's diagnosis (5 abstention runs of "What is name of my
hamster?"):
- 2/5 succeeded (Luna context surfaced)
- 2/5 failed because Haiku rejected adjacent context (top score 0.55-0.58)
- 1/5 failed because answerer ignored surfaced Luna context

The expanded battery would tell us if the failure mode generalizes
across entity types (pet/person/place/item).

### C. Snapshot replay redo (without API contention)

Run them ONE AT A TIME (no parallel longmem load):
```bash
# Wait for any longmem to finish first, then:
./dev python3 eval/s1s_snapshot_replay.py \
  --snapshot ~/AgentsContext/brain/brain.db.bak-pre-situation-migration \
  --conversation ~/.claude/projects/-Users-tpac-brain/71857713-*.jsonl \
  --run-name solo_71857713_$(date +%H%M)
```

The Apr 19 snapshot is preserved at
`~/AgentsContext/brain/brain.db.bak-pre-situation-migration`. Don't
delete it — needed for these replays.

### D. Verify fix-effects on broad eval

Today's hardening commits (`f90aed2`, `ea1d845`) added:
- Per-field vector invalidation surfaces in revise() result
- Encoding lock recovery on spawn-failure
- 3 silent `pass` blocks → `_log_error`

To verify these fixes don't introduce regressions on broad eval, re-run
`full_suite` after they're confirmed live. The 44.4% baseline is now
the comparison point, not 84%.

---

## Memory + safety notes

- **Production safety:** all eval brains are isolated copies. The live
  `~/AgentsContext/brain/brain.db` is never written to by eval scripts.
- **`brain-eval-*` dirs:** weekly cleanup is fine. Each is the
  output of one `--run-name`.
- **API contention:** running 25+ parallel Sonnet calls hits rate limits.
  For broad eval keep `--longmem-workers <= 25`. For snapshot replays,
  serialize them OR run when the daemon is idle.
- **`brain.db.bak-pre-situation-migration` is sacred.** Don't delete —
  it's the snapshot every replay reads from.

---

## Last edited: 2026-04-26 (post-broad-eval session)

Update this doc when:
- New sub-suites are added to `full_suite.py`
- Picks change (LONGMEM_BROAD_PICKS, ABSTENTION_BATTERY,
  SNAPSHOT_REPLAY_CONVERSATIONS)
- New analysis tools are added (e.g., a per-axis failure-mode classifier)
- Eval results trigger architectural changes worth recording
