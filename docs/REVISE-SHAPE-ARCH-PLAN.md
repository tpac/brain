# Revise shape — architecture review and follow-on plan (2026-09-04)

Scope: the revise-shape subsystem as it sits on branch `claude/sweet-lichterman-ba9854`
(head `71b2fac`, steps 1–7 of `docs/REVISE-SHAPE-SPEC.md`, unmerged). Boundary
traced: `servers/contract.py` (REVISE_RULE, swappable, REVISE_FIELD_ALIASES,
field summary) → `servers/brain_mcp.py` (revise / revise_batch generators) and
`servers/brain_remember.py` (the write path, step 2); `servers/scales/s1/encode.py`
+ `encode_contract.py` + `encoding_gist_prompt.py` + `servers/interaction_defaults.py`
+ `interaction_collapse.py` (the `s1e_gist` boundary); `eval/encoder_ops.py` (the
op-dump reader) and its two consumers `eval/encoder_prompt_ab.py`,
`eval/encoder_ops_shape.py`, beside the older `eval/longmem/connect_ab.py`,
`eval/longmem/corpus_shape.py`, `eval/capabilities/base.py`; the guardrail
`tests/test_teaching_vocabulary_sync.py`. Coverage caveat: the trace was done by
the same-session code-review finders (callers of every changed symbol, the stub
brains, the delta-metadata readers, `AspectRegistry.from_dict`) plus the author's
own build, not by an independent dynamic trace; the brain surfaced no in-flight
plan over the eval harness or the capabilities suite. Report-only: nothing here
is applied.

Settled constraints respected: one revise shape, value-or-swap, edges via
`connect_to` (73d30b14); the gist is the `s1e_gist` interaction read through the
resolver (21c9f826); deploy is one merge together after the edge cell (07b830a1);
the 2026-09-04 rulings on the prompt rows, the gist catalog clause as wording
only, `partially_resolves` in `correction_improvement` only (c5ed32b2). The cell
result (9a9e35c1: edge 0/3, `connect_to` on 0/13 revise ops) is measured fact,
not a structural defect this plan can fix.

**What is clean (no step):** `s1e_gist` as its own learnable boundary follows
the registry recipe exactly (template module, config in the consumer's contract,
registry + validator + collapse rows); `eval/encoder_ops.py` is a justified new
module (two consumers that had diverged, one responsibility); contract → tools →
field summary → gist → prompt now state one rule with a guardrail holding them;
the aspect edit is in the owner file at the owner's granularity.

## Dependency summary

1. **Step 1** (harness correctness, from the code review) — independent; first,
   because every later measurement reads through it.
2. **Step 2** (one resolution of `s1e_gist` per encode; `enabled` as the
   off-switch) — independent of 1; needs Tom's ruling on the generic
   unknown-key check.
3. **Step 3** (the assembler owns the gist slot) — independent; small.
4. **Step 4** (alias map depth + `content_edits` retirement eligibility) —
   needs Tom's ruling; can follow 1.
5. **Step 5** (fold the shape metrics and the older extractors onto the one
   reader — spec §8 row 10's post-cell half) — after 1; needs Tom's ruling.
6. **Step 6** (`eval/capabilities/base.py` schemas derive from the generators, or
   the suite retires) — independent; needs a liveness check and Tom's ruling.

Steps 1, 2, 3 can run in parallel sessions. 4, 5, 6 wait on rulings.

## Step 1 — Fix the harness where the code review confirmed it lies

**Problem.** Five confirmed defects in the step-4 scorer, each of which can score
a cell wrong: (a) `eval/encoder_ops.edge_entries` treats any op carrying
`source_id`+`target_id` as an asserted edge, so a `disconnect` overwrites the
pair's real why with `''` in `_edge_pairs`; (b) `--gist` is `nargs='?'` beside a
`nargs='*'` positional — a capture path after `--gist` is deployed as the gist
override; (c) arm F's splice ignores `s1e_gist`'s `enabled`, so a gist-off arm
cannot be measured and disagrees with assembled arms; (d) `score_swap_fidelity`
misses swaps inside `connect_to` (the very swap the edge cell scores) and
re-implements `contract.apply_swaps` without its refusals; (e) `_corr_rels_offline`
hand-builds the seed path the owner exports and reads the repo seed while the
live scorer reads the working copy. Plus the guardrail's tool-surface rows match
whole-tool JSON, so a schema key alone counts as "taught" (tautological), and two
docstrings carry dates and dead-behavior narrative against CLAUDE.md's
"comments carry the why, not the history".

**Target state.** `edge_entries` emits a connect entry only for `kind(op) ==
'connect'`. `--gist` is a flag and `--gist-file FILE` a separate option (the
two-flag shape §8 row 2 argued against loses to the argparse trap); the splice
reads `get_interaction_effective('s1e_gist')` once and strips the slot when
`enabled` is false. `score_swap_fidelity` calls `contract.apply_swaps` per field
and also checks `connect_to` why/relation swaps against the stored edge
description (read through `GraphDAL`). `_corr_rels_offline` imports
`servers.aspect_store.SEED_ASPECTS_JSON_PATH` (or `aspects_json_path()` when the
run should match the live working copy — state which in the flag's help). The
guardrail's tool rows match against description strings (tool description +
each property's `description`), not `json.dumps(tool)`. Docstrings in
`eval/encoder_ops.py` and `tests/test_encoder_ops.py` lose the dates and the
old-behavior narrative.

**Files & call sites.** `eval/encoder_ops.py:127`; `eval/encoder_prompt_ab.py`
L816 (argparse), L921 (splice), L536 (`score_swap_fidelity`), L727
(`_corr_rels_offline`); `tests/test_teaching_vocabulary_sync.py:44–52`
(`SURFACES`); `tests/test_encoder_ops.py` (add a disconnect case, a connect_to
swap fidelity case, an argparse case); `tests/test_gold_surfaces.py`.

**Verification.** `tests/test_encoder_ops.py`, `tests/test_gold_surfaces.py`,
`tests/test_teaching_vocabulary_sync.py`; then `eval/encoder_prompt_ab.py --rescore`
over `ab_2026-09-01_03/ops3/*-v41gist/*.json` AND `ops5/*-shape/*.json` with both
golds — no gold number may move.

**Blast radius.** Eval only; ~80 lines. A moved gold number on re-score means a
fix changed the scorer's semantics — stop and report.

**Depends on.** None. **Respects.** c5ed32b2 (row 10's pre-cell half is what
this repairs). Tom's go on the findings.

## Step 2 — Resolve `s1e_gist` once per encode; state the off-switch

**Problem.** `run_encoding` stamps `get_interaction_stamp('s1e_gist')` before the
`enabled` gate and outside the lived branch, and `_build_user_content` resolves
config and prompt separately — three resolutions of one K per encode where
`brain.get_interaction_effective` exists for exactly this ("three separate
accessor calls can straddle a concurrent set_interaction_active"). The delta
records a gist fingerprint even when no gist entered the payload; `if gist:` is
dead because the resolver never returns an empty template. `validate_s1e_gist_config`
accepts unknown keys, so `{"enable": false}` passes and the gist stays on.

**Target state.** One `eff = brain.get_interaction_effective('s1e_gist')` in
`run_encoding`, threaded into `_build_user_content` (a `gist=` argument beside
`precomputed`; standalone callers resolve it themselves); the delta carries the
stamp only for a gist that was emitted, plus `gist_emitted: bool` for the
non-lived / disabled cases; the dead branch goes. `enabled` stays the off-switch
and the template module's docstring says why (an empty template override keeps
the code default by the resolver's rule). Unknown-key refusal belongs in the
resolver for every interaction — override keys must be a subset of the code
default's keys — not per-name validators.

**Files & call sites.** `servers/scales/s1/encode.py` L101–107, L345–353,
L1010–1017; `servers/brain.py` `_resolve_interaction` (the generic key check);
`servers/scales/s1/encode_contract.py` (validator simplifies or goes);
`tests/test_s1e_lived_sequence.py`, `tests/test_s1e_trace_links.py` stubs gain
`get_interaction_effective`; `tests/test_interactions_runtime.py` for the
resolver check.

**Verification.** `tests/test_s1e_*`, `tests/test_interactions_runtime.py`,
`tests/test_interaction_override.py`, `tests/test_trace_delta_shape.py`.

**Blast radius.** The generic key check is a resolver behavior change for every
interaction: an existing override carrying a key the code default lacks (grep
the interactions table via `list_interactions` / `get_interaction_effective`
before enabling) would start degrading to the default loudly. Otherwise local
to the encoder.

**Depends on.** None. **Respects.** 21c9f826 (the gist stays an interaction).
**Ruling needed:** the generic unknown-key refusal (changes every boundary).

## Step 3 — The assembler owns the gist slot

**Problem.** `eval/encoder_prompt_ab.splice_gist` keys the slot on a hand-listed
tag set (`GIST_SLOT_CLOSERS`) and replaces everything between the last
recognized closer and `<timeline`. A new block added to `_build_user_content`
after the catalog would be deleted from arm-F payloads silently while the tool
prints "replaced N chars".

**Target state.** The block order is one constant the assembler exports
(`encode_contract.PAYLOAD_BLOCKS` or a small `encode.gist_slot(payload) -> (start,
end)` helper); the harness imports it and REFUSES loudly when the slot text
contains any `<tag` it does not own, instead of deleting it.

**Files & call sites.** `servers/scales/s1/encode.py` L988–1017 (the block
emission), `servers/scales/s1/encode_contract.py`, `eval/encoder_prompt_ab.py`
L700–717; a test in `tests/test_encoder_ops.py` or a new harness test with a
capture carrying an unknown block.

**Verification.** The new test; `--gist` on one frozen capture prints the same
`[gist]` line as today.

**Blast radius.** Eval plus one exported constant; ~40 lines.

**Depends on.** None. **Respects.** T7 (free text for guide text; angle brackets
are payload structure) — no tag around the gist.

## Step 4 — Alias depth, and whether `content_edits` retires now

**Problem.** `contract.REVISE_FIELD_ALIASES` has one reader (the guardrail),
while `brain.revise`, `brain_mcp._generate_revise_schema` and
`eval/encoder_ops.swaps_of`/`revise_surfaces` each hardcode `content_edits`.
Adding an alias to the map exempts it from the taught set immediately while
nothing accepts or advertises it, and no test fails.

**Target state.** Either the map has no other readers because the alias is gone,
or every reader iterates it. The spec's own retirement trigger (§3: "zero uses
across a full A/B round") is now met — the five ops5 runs wrote `content` swaps
and zero `content_edits`. If Tom retires: drop `CONTENT_EDITS_SCHEMA`, the
normalization in `brain.revise` (~L1514), the `content_edits` prop in the revise
generator and `BATCH_OP_SPECS['revise']`, the map entry, and add the name to
`tests/test_retired_fields.RETIRED_NODE_FIELDS` — that scan then enforces
absence everywhere; the guardrail stays green by construction. If not yet: add
one guardrail assertion that every alias key IS a revise-spec property whose
description says "Deprecated alias", so the map cannot lie in either direction.

**Files & call sites.** `servers/contract.py` (L141–158, L290, L307),
`servers/brain_remember.py` ~L1514–1540, `servers/brain_mcp.py`
`_generate_revise_schema`, `eval/encoder_ops.py` `swaps_of` / `revise_surfaces`,
`tests/test_retired_fields.py`, `tests/test_revise_unified.py` (alias tests),
`tests/test_teaching_vocabulary_sync.py`.

**Verification.** `tests/test_retired_fields.py`, `tests/test_revise_unified.py`,
`tests/test_teaching_vocabulary_sync.py`, `tests/test_brain_batch_op_contract.py`;
`eval/mcp_batch_probe.py` + `eval/mcp_schema_gate.py` again (schema change).

**Blast radius.** Any live caller still sending `content_edits` (Anchor's own
MCP calls; S2 units — grep `content_edits` under `servers/scales/s2`) would be
refused loudly after retirement. Check before, not after.

**Depends on.** Step 1 for the re-score check. **Respects.** 73d30b14 (alias then
retire). **Ruling needed:** retire now, or keep the alias one more window.

## Step 5 — One reader, one metric vocabulary (spec §8 row 10, post-cell half)

**Problem.** `eval/longmem/connect_ab.extract_connect_entries` is still its own
flattener (ignores batch-level `connect_to`, the `relations` form, revise-side
`connect_to`, reads `title` directly) while `encoder_prompt_ab` imports
`WRITE_TOOLS` from that same module. `eval/encoder_ops_shape.py` duplicates
`corpus_shape`'s `GENERIC_RELATIONS`, `RESCUE_VERBS`, `WHY_BAND`, `HEX8` and
re-lists `encoder_ops.SURFACE_FIELDS`; its metrics live outside the harness, so
a run does not print them. `rescore` recomputes op counts through `score_arm`
with a synthetic log.

**Target state.** `encoder_ops` imports the constants from `corpus_shape` (the
owner) and exports `WRITE_TOOLS`; `connect_ab.extract_connect_entries` is a thin
adapter over `ops_of` + `edge_entries`; the shape metrics fold into
`encoder_prompt_ab.score_shape` so every run prints them and
`encoder_ops_shape.py` becomes a table over saved dumps that calls the same
functions; `rescore` counts with `Counter(kind(op) for …)`. Retention uses word
boundaries.

**Files & call sites.** `eval/encoder_ops.py`, `eval/encoder_ops_shape.py`,
`eval/encoder_prompt_ab.py` (`score_shape`, `rescore`, L325 import),
`eval/longmem/connect_ab.py` L43, L77–108.

**Verification.** Re-run `eval/encoder_ops_shape.py ab_2026-09-01_03` before and
after — the table must not move; `tests/test_encoder_ops.py`; `--rescore` on
both baselines.

**Blast radius.** Eval only; `connect_ab` is a legacy replay harness (a856873b) —
confirm it still runs after the adapter change or mark it retired.

**Depends on.** Step 1. **Respects.** 2ae1d407 (full-output shape metrics ride
the eval). **Ruling needed:** this is the fold-in Tom has not ruled.

## Step 6 — `eval/capabilities/base.py` stops hand-rolling the write schemas

**Problem.** `_build_capability_tools` builds `remember` and `revise` schemas by
hand with its own type map, appends "(replaces existing value)", and describes
revise as "content is appended with revision history" — two generations stale.
Five eval scripts import it; none of them can exercise a swap.

**Target state.** First check liveness (last run of `eval/encoding_v3_*`,
`eval/simulate_real.py`, `eval/test_extensive_encoding.py`,
`eval/capabilities/test_*`). If dead: delete the suite (recoverable in git) with
Tom's word. If live: `_build_capability_tools` derives `remember` / `revise` from
`brain_mcp._generate_remember_schema()` / `_generate_revise_schema()` (key
`inputSchema` → `input_schema`), the way `_build_revise_batch_schema` derives.

**Files & call sites.** `eval/capabilities/base.py` L147–195 and its five
importers.

**Verification.** `eval/capabilities/test_*` (whatever still runs);
`tests/test_retired_fields.py` (advertised fields).

**Blast radius.** Eval only.

**Depends on.** None. **Ruling needed:** delete vs derive.

## Considered, not recommended

- **Two module identities for `eval/encoder_ops.py`** (`eval.encoder_ops` in
  tests vs bare `encoder_ops` in the scripts): both modules are stateless
  functions and constants; an `eval/__init__.py` would change how every eval
  script is invoked. Not worth it.
- **Moving `REVISE_RULE` prose into `$defs`/`$ref` to shrink the tool blobs**
  (+26% on the encoder toolset): a real cost, but the generation behavior of
  `$ref` inside `input_schema` is unverified across the clients this plugin ships
  to, and brain_batch made the same inline choice deliberately. Measure before
  changing; not a structural defect.
- **The YAML connect_to scan in the guardrail:** it yields no extra keys today
  and is exactly the coverage the temporal example needs if its items ever carry
  a key the array examples do not. Keep.
