# Surface v12 — ship-the-bundle handoff

Picks up the surface-layer arc from the 2026-07-02 session. The design work is
DONE; the job is to eval-gate the v12 prompt and ship the bundle. Read this
whole file, then recall the brain nodes at the bottom before touching anything.

## Already LIVE on main (verified 2026-07-02, don't redo)

- **recall_topical resurrection** — shared `recall_score()` in
  `surface_contract.py` (topical was silently dead 3 weeks on a score-key
  fork: tool read `'score'`, `brain.recall()` emits `effective_activation`).
- **Prompt caching** — `cache_control` on the round-1 user content; tools ride
  every round (stripping them invalidated the prefix); forced-finalize
  fallback if Haiku tool-uses on the final round. `CACHE_MIN_PREFIX_TOKENS`.
- **Trace attribution** — `tool_calls[].result_ids` / `dropped_ids`, and the
  `fetched_by` / `floored_by` roles in `nodes_for_traces` (the LAF P2 feed).
- **Judgment scoreboard** — `eval/oracle_audit/ab_tool_use_audit.py`.

Confirm still live: a fresh 2-round recall trace should show
`cache_read_tokens == cache_creation_tokens` and `result_ids` populated on
`tool_calls`.

## STAGED, NOT live (this branch / PR only)

Two commits, in order:
1. `cut recall_verbatim from TOOL_DEFINITIONS` (fetch_tools.py + test)
2. `surface prompt v12 draft` → `eval/surface_v12_prompt.txt`

**These MUST ship together and only after the eval passes.** Live v11 still
names `recall_verbatim` in its prompt text; cutting the tool while v11 is
active is the exact prompt/code contradiction we're removing. That's why this
PR is a **draft — do not merge until step 4.**

## Steps (each with its verify-check)

### 1. Post-deploy scoreboard — pre/post-caching read
```
./dev python3 eval/oracle_audit/ab_tool_use_audit.py 3 \
  --split 2026-07-02T19:23:00+00:00
```
Split point is the daemon restart that armed caching (+ LAF laf_v1, same
restart). Against baseline (591 recalls/7d: 88% fire, topical 341/341 zero,
cache 0%), the "post" column should show: high cache-hit share, recall_topical
`avg-res > 0` (resurrection), forced-finalize + empty-tool_use ≈ 0, and the
fetch-precision columns populated. Surface-tool metrics are clean of the LAF
confound (LAF changes the 25-candidate scoring, not the tool loop). **If
forced-finalize is NOT rare, the tools-in-round-2 cache trade-off needs a
rethink before shipping** — it may argue for a v12 tweak.

### 2. Probe-fidelity gate — do this BEFORE the A/B
The Frozen-Corpus sweep must fire the AGENTIC loop, not single-shot, or the
A/B silently tests the wrong path. Confirm `BRAIN_SURFACE_VARIANT=v5_agentic`
is exported in the sweep env AND assert the swept recalls' traces carry
non-empty `tool_trace`. If they don't, the A/B is invalid — fix the harness
first.

### 3. A/B v11 vs v12 on the Frozen Corpus (docs/EVAL-PLATFORM.md)
```
./dev python3 eval/longmem/build_corpus.py --items 20 --label surf_base
# sweep both arms against the SAME corpus (encode held fixed):
#   control  = current active surface
#   candidate = --surface eval/surface_v12_prompt.txt
./dev python3 eval/longmem/compare_arms.py <control> <cand> --labels v11,v12
```
v12 is a simplification + cost change, so the bar is **no regression** on
recall_conditional pass-rate, not "must beat." Review the per-item diff WITH
TOM before activating — his call on ship.

### 4. Ship the bundle (only after 3 passes + Tom's ok) — all together
1. `register_interaction('surface', <v12 text>)` → registers DORMANT vN+1
2. `set_interaction_active('surface', vN+1)` → flips the runtime pointer
3. **Update the seed to match**: hand-edit `servers/interaction_seed.py`
   `SURFACE_PROMPT_V1` to the v12 text. Surface is NOT covered by
   `./dev sync-prompts` (that mirrors the 4 encoder prompts only); its seed is
   this manual inline template, read by fresh brains.
4. Merge this PR to main (both staged commits).
5. Restart the daemon. Now recall_verbatim is gone from TOOL_DEFINITIONS AND
   the v12 prompt that omits it is active — no contradiction window.

Verify post-restart: recall_verbatim absent from a fresh recall's tool
options; v12 fire behavior matches the probe.

## Parked follow-ups (after the bundle ships)

- **Stop injecting the retrieval-stats block** in `build_surface_prompt` — v12
  removed its only consumer (the weak-retrieval trigger), so it's dead tokens
  + a latent escape hatch.
- **If a targeted episodic surface tool is ever wanted**, build
  `recall_episodic` (recall_episodes → nodes_for_traces bridge) on the
  **created/revised roles ONLY, never surfaced** — surfaced echoes past picks
  into the LAF training signal (the 4942bd35 circularity trap; decision
  eb6dc82b). But first check whether the always-on LAF episodic lane
  (d3480899) already covers it.

## Recall these first — they carry the WHY

- `eb6dc82b` — recall_verbatim cut decision + reintroduction constraint
- `232cde97` — surface production baseline (the numbers to beat/hold)
- `ba71904d` — Haiku needs directive patterns, not philosophical framing
- `0580129f` — cut for comprehension not cost; verbosity = escape-hatch doors
- `5337dce7` — state the default before the exceptions (Haiku salience)
- `63910b9c` / `0ff497c4` — the topical death + its fix (the drift lesson)

## Design stance for any v12 edits

It's Haiku. Directive, concrete, default-first, one door per tool. Fetching is
normal now (caching paid the cost) — guide it, don't suppress it. Verify against
real prompts (replay a `/tmp/brain-judge-result-*.json`: split off the v11
instructions at the `\n\n---\n\n` seam, keep the dynamic content, swap in v12)
rather than trusting a read-through.
