# Revise shape — value or swap, edges via `connect_to`

Status: ruled by Tom 2026-09-03. **Steps 1–2 (contract + write path) are
committed on the branch as `2ad74aa`**, unmerged — the daemon runs main and
nothing deploys until the surfaces (§4 prompt/gist/MCP rows) move and the edge
cell (§6 step 3) has a number; Tom ruled the deploy is one merge, together, never
the tool-layer half first. The §4 table marks each surface **done** or still
owed. The vocabulary guardrail is red on `gist` for `connect_to` (the MCP rows
landed 2026-09-04) — the lockstep working, green again when the gist row lands.
**Handoff for the next session: brain `71ce490b`** (boot self-test,
verify-before-use, gates — the six unruled recommendations are listed there);
working set outside git at
`/Users/tpac/AgentsContext/s1e-field-coverage-gold/ab_2026-09-01_03/WORKING-SET-next-session.md`.
**Before any prompt row is edited: the full read of `SYSTEM_PROMPT` with a
per-section audit artifact (brain id:71eeff20) — Tom's condition for the
prompt half.**

## 1. The rule

On `revise`, a field takes either its **new value** — the whole field is
replaced — or a **swap** `{old, new}` (a list of swaps for several spots) that
changes only what is stale. Fields not named are untouched. Edges are
`connect_to`, exactly as on `remember`: on a revise it changes the edge this
node already has to that target (its `why` or its relation, value or swap), or
creates it if there is none.

```
{op: "revise", node_id: "d827d22f", reason: "manifests moved 9.6.0 → 9.7.2",
 title:     {old: "brain/9.6.0", new: "brain/9.7.2"},
 content:   {old: "both manifests still say `9.6.0`.", new: "both manifests say `9.7.2` — still short of D-10's 0.9.0."},
 situation: "When picking up Phase 5 — plugin.json is still 'brain'; version is 9.7.2, short of D-10's 0.9.0",
 connect_to: [{target: "15bbfd64", relation: "gaps_in",
               why: {old: "both manifests still say 9.6.0", new: "manifests moved 9.6.0→9.7.2 and still haven't reached 0.9.0"}}]}
```

Why this shape (measured, id:42469273): three cold Sonnet readers were handed
the same node and stale value with three API shapes; this one was rated most
natural (4/5 vs 3/5) and was the only one where the stale edge description got
repaired inside the node's own op. It removes vocabulary rather than adding it:
`content_edits` becomes an alias, no `edge_edits`, no `revise_edge` for
encoders. The sweep — "one `revise` per node across every surface the stale
value sits in" — becomes literally the call.

## 2. Semantics

| element | rule |
|---|---|
| bare string on any replaceable field | replaces the whole field (the restructure case). Unchanged from today. |
| `{old, new}` on any replaceable field | `old` must occur **exactly once** in the stored value; replaced by `new`. Zero or several matches → the op fails loudly with the count, nothing written (today's `content_edits` contract, generalized). `old` may be any span — a token, a clause, a sentence. `old == new` → error. |
| list of swaps | applied in order, each under the same rule; a later swap may depend on an earlier one. |
| swap on a field with no stored value | error — nothing to match. Use a bare value. |
| swap-typed fields | every replaceable text field: `title`, `content`, `situation`, `question`, `reasoning`, `thought`, `their_raw_quote`, `my_raw_quote`, `correction_pattern`, open KV keys. Non-text fields (`confidence`, `emotion`, `event_time`, `type`, `evolution_status`, `source_refs`) take bare values only; a swap on them is a schema error. |
| bare value and swap for the same field in one op | error (today's `content` / `content_edits` mutual exclusion, generalized). |
| immutable fields (`id`, `created_at`, `locked`) | skipped with a warning, as today. |
| `connect_to` on revise — identity | `target` must be an existing node **id** (8-hex); sibling titles are not valid on revise — there are no siblings. The edge is identified by (this node, target, relation), either direction: outgoing is tried first, incoming second. |
| `connect_to` on revise — `relation` | a bare string identifies the relation row (and is the relation to create if the pair has no edge). A swap `{old, new}` renames it in place via the existing rename primitive — weight, `created_at`, history survive. `relation` is **required** when the pair carries more than one relation; with exactly one it may be omitted. |
| `connect_to` on revise — `why` | bare string replaces the description; swap patches it under the exactly-once rule. |
| `connect_to` on revise — no such edge | created, outgoing from this node, with the given relation and `why` (which must then be a bare string ≥30 chars). Direction matches `remember`'s `connect_to`. |
| `connect_to` on revise — `relations: [{relation, why}]` | accepted with the same item semantics, one row each. |
| ops other than `revise` | unchanged. `remember` gains only the `target` alias (§3). `connect` keeps its upsert semantics for the callers that rely on it; the prompt stops teaching it as the repair path. `revise_edge` stays a standalone deferred tool for Anchor and S2. |
| unknown keys on revise | still land in the node's KV store — **except `connect_to`**, which today is silently DROPPED: `brain.revise` lets it through field classification, `_store_node_metadata` skips it as a control field, and the result still lists it in `fields_updated` with a `node_revised` delta — an edge change is reported that never happened. After this change it routes to edges. No other key changes route. |

## 3. Aliases and renames

| old | new | policy |
|---|---|---|
| `content_edits: [{old,new}]` on revise | `content: [{old,new}]` | accepted as an alias, normalized in `brain.revise` before the swaps apply. Passing both is the same mutual-exclusion error. Retire when the encoder's op dumps show zero uses across a full A/B round; retirement = drop `CONTENT_EDITS_SCHEMA` and the alias handling, and add the name to `tests/test_retired_fields.py::RETIRED_NODE_FIELDS` — its scan (prompts, gist, field summary, every op description, every tool blob) then enforces absence. One retirement registry, not two. |
| `connect_to[].title` | `connect_to[].target` | on **both** `remember` and `revise`, so the item shape is identical. `title` stays an accepted alias for one deprecation window. Rationale: the field is defined as "an 8-char id for an existing node, a title only for a same-batch sibling" — a name that lies about its usual content, which the prompt spends sentences correcting ("copy the id into the `title` slot"). On revise it would be actively misleading. |

## 4. Lockstep — every surface that changes, or the change is not real

By E10 the tool layer outranks the prompt by position (the field summary and
tool descriptions are injected last). If the prompt says "value or swap" and
the `revise` description still says "`content_edits` … other fields REPLACED
wholesale", the prompt loses silently. Three prompt lines also dereference into
tool descriptions (id:1b7984f8). All rows move in one change.

| surface | owner | today | after |
|---|---|---|---|
| `BATCH_OP_SPECS["revise"]` | `servers/contract.py` | **done** | description states `REVISE_RULE` (the one rule, one string every surface quotes); props gain `content` typed `string \| swap \| swap[]` via `swappable()` (the exemplar of the union — other fields follow it, additionalProperties open) and `connect_to` with `REVISE_CONNECT_TO_ITEM_SCHEMA` (derived from the remember item schema: same vocabulary and `why` exemplars, `relation`/`why` and `relations[]` items swappable, `target` id-only); `content_edits` kept in props as the alias |
| `CONNECT_TO_ITEM_SCHEMA` | `servers/contract.py` | **done** | `anyOf: [{required: [target]}, {required: [title]}]`; `target` carries the description, `title` is "Deprecated alias of `target`"; `contract.connect_to_target(entry)` is the one place the alias is known — every reader (resolver, apply, batch-level label, the batch probe) calls it |
| `CONTENT_EDITS_SCHEMA` | `servers/contract.py` | **done** | `SWAP_SCHEMA` is the item (no prose of its own — it is inlined at every swappable field), `SWAP_LIST_SCHEMA` the list; `CONTENT_EDITS_SCHEMA` is the list with a one-line deprecated-alias description |
| swap-typed fields | `servers/contract.py` | **done** | every writable str field swaps by default (the open-KV default); `bare_only: True` marks the exceptions — `type`, `evolution_status`, `event_time`, `emotion_label`, `source_turn_id`; `get_swap_fields()` returns `{name: spec}` for the generators |
| `RETIRED_OP_FIELDS` | `servers/contract.py` | **deleted** | retirement lives in `tests/test_retired_fields.py` alone (§3) |
| `ENCODING_TOOLS` | `servers/scales/s1/encode_contract.py` | **done** | hoisted from a function-local set in `encode.py`; the guardrail test imports it |
| `brain.revise` | `servers/brain_remember.py` | **done** | `connect_to` popped first; `content_edits` normalized to `content: [swaps]`; swaps separated from bare values and refused on non-text / `bare_only` fields before any write; after the row is read every swap resolves against the stored value (`contract.apply_swaps`, title/content from the row, everything else from KV) — all of them before any write, so one bad `old` leaves the node untouched; edges routed last through `_apply_revise_connect_to` → `revise_edge` (update/rename, either direction) or `connect_typed` (create), results in `connect_to_result`, never in node deltas; warning when a NEW relation lands on an edge that points into the node (ruling 1c67e263); `encoding_source` param carries the caller's provenance for edges only |
| `contract.validate_field` | `servers/contract.py` | **done** | knows the swap shape: valid on any str field not `bare_only`, refused plainly elsewhere; `is_swap` / `is_swap_list` / `validate_swaps` / `apply_swaps` are the primitives every layer shares |
| `GraphDAL.get_edge_endpoints` | `servers/dal_graph.py` | **done** | stored (source, target) of an edge — the direction warning's evidence |
| `_handle_brain_batch` revise branch, `_handle_revise`, `revise_batch` | `servers/dispatch_write.py`, `brain_remember.py` | **done** | pass-through; `_handle_revise` passes `encoding_source` and turns `connect_to_result` into `edge_relation_revised` manifest rows (created under reason `connect_to`, revised under the revise's reason); `revise_batch` passes `encoding_source` |
| MCP `revise` description + `content_edits` prop | `servers/brain_mcp.py` | **done** | description = `REVISE_RULE` + bare-value note + alias note; every `get_swap_fields()` prop through `swappable()` (the "(replaces existing value)" suffix is gone); `connect_to` is the same object as `BATCH_OP_SPECS['revise']['properties']['connect_to']`; the blob grew 6.4K → 14.5K chars (inline swap shape at ten fields + the edge item schema — the brain_batch choice, kept) |
| MCP `revise_batch` description + `revisions` items | `servers/brain_mcp.py` | **done** | same rule; items inherit `connect_to` from the revise generator |
| MCP `brain_batch` oneOf | derived from `BATCH_OP_SPECS` | — | derives automatically; re-run `eval/mcp_batch_probe.py` + `eval/mcp_schema_gate.py` |
| MCP `remember` description (`connect_to` passage) | `servers/brain_mcp.py` | "use `connect_to` with a correction-aspect relation…" | unchanged text; item key `target` (alias `title`) |
| `generate_field_summary` content line | `servers/contract.py` | **done** | content line says value-or-swap; `REVISE_RULE` verbatim as its own line before RETURNS — the summary is injected last, so it states the rule the tools state |
| s1e prompt — Actions → revise bullet | `servers/scales/s1/encoding_prompt.py` | 48 lines: every-surface rule, `content_edits` default, "short fields have no patch form", REPLACE semantics | the one rule + every-surface sentence naming `connect_to` for edge descriptions; the "short fields have no patch form" clause is **deleted** (it becomes false); `source_refs` REPLACE note stays |
| s1e prompt — Actions → connect bullet | same | "wire edges between two existing catalog nodes" | creation only, stated as such; repair is `connect_to` on the node's revise |
| s1e prompt — connect_to targets section | same | "copy that 8-char id into the `title` slot" | `target` slot; sibling-title form on remember only |
| s1e prompt — Nodes → Anatomy content bullet | same | "content is replaced on revise … or patch … with `content_edits`" | "…or swapped in place: any field takes its new value or `{old,new}`" |
| s1e prompt — Reading → flavor 3 | same | "an in-place patch for a routine update" | "a swap for a routine update"; edge descriptions named as claims |
| s1e prompt — revise ladder example (97b1f24e) | same | `content_edits` + title/situation rewrites | swaps on title/situation where a token changed; bare values where the claim restructured; `question` untouched — the example teaches when to use which |
| s1e prompt — sweep example | same | 5 revises with `content_edits` + title rewrites; no edge op | same 5 revises in swap form; a45c88f1's revise carries `connect_to: [{target: "e91a6d05", relation: "implements", why: {old, new}}]` — the edge repaired inside the node's op |
| s1e prompt — Temporal → validity intervals | same | "`content_edits` on the changed claim" | "a swap on the changed claim" |
| gist (`ENCODER_GIST`) | `servers/scales/s1/encode_contract.py` | "…title, content (patched in place)…" | "one `revise` per node: a swap on every surface the stale value sits in, `connect_to` for its edge descriptions" |
| `tests/test_revise_unified.py` | tests | **done** | classes H (`TestValueOrSwap`, `TestSwapDispatch`) and I (`TestConnectToOnRevise`): swaps on title / situation / open KV / content, list order, exactly-once and all-or-nothing, bare_only refusal, no-stored-value refusal, alias conflict, dispatch validator; connect_to: why value and swap, relation rename, relation optional-when-one / required-when-several, create outgoing, bare-why floor, sibling-title rejection, incoming edge found, new relation on incoming edge rides it and warns, `title` alias, never a node field, field+edge in one op, revise_batch, dispatch emits `edge_relation_revised` and no node delta |
| `tests/test_brain_batch_op_contract.py` | tests | three sites derive from `BATCH_OP_SPECS` | unchanged assertions; `connect_to` on revise covered by the derived oneOf |
| `tests/test_teaching_vocabulary_sync.py` | tests (new, this session) | guards today's vocabulary | guards the new one; retirement of `content_edits` flips a contract tuple and the test enforces its absence on every surface |
| `docs/S1E-CHECKLIST.md` E10 | docs | four checks | unchanged; this change is E10's worked instance |

## 5. Drift guardrail

`tests/test_teaching_vocabulary_sync.py` pins the prompt, the gist, and the
tool layer to one vocabulary, from the contract outward:

1. every `op: "…"` in the s1e prompt's examples is a `BATCH_OP_SPECS` op;
2. every key inside a `connect_to` item in the examples is a
   `CONNECT_TO_ITEM_SCHEMA` property;
3. every revise-spec property beyond `node_id`/`reason` is taught in the
   prompt AND stated in the `revise` / `revise_batch` descriptions AND in the
   field summary — a name present in the contract but silent on any surface
   is the E11 defect;
4. every backticked identifier in the gist is an op, tool, writable field,
   `connect_to` key, or one of the relation verbs the gist deliberately names.

Retired names are `tests/test_retired_fields.py`'s job (one registry:
`RETIRED_NODE_FIELDS`), whose scan covers the prompts, the gist, the field
summary, every op description and every tool blob.

Adding `content_edits` to `RETIRED_NODE_FIELDS` without touching the prompt
fails that scan; renaming `title` → `target` in the schema without touching the
examples fails (2); adding `connect_to` to the revise spec without teaching it
fails (3).

## 6. Eval plan (next session)

1. Implement §2–§4 on this branch; `./dev pytest` on `test_revise_unified`,
   `test_brain_batch_op_contract`, `test_teaching_vocabulary_sync`, the
   interaction-defaults and bypass guards; `eval/mcp_batch_probe.py` and
   `eval/mcp_schema_gate.py`; daemon restart.
2. Rewrite the prompt rows in §4 on a candidate copy of the production text
   (`eval/candidate_prompts/`), the gist in the contract; run the vocabulary
   test against the candidate too.
3. Edge cell: canonical `d827d22f` ×3 and run-44 ×2, candidate + gist, versus
   the round-2 `v41 + gist A` baseline (3 runs/cell already). The number that
   matters: edge description repaired, currently **0/24** across every arm.
4. Only then the cross-prompt census (verb→op mismatches, surfaces named vs
   ops that reach them, example coverage of ops) over every registered
   default — the vocabulary it checks is what steps 1–2 change.

## 7. Out of scope, deliberately

`revise_edge` (standalone) stays for Anchor and S2. `connect`'s upsert stays.
S2 prompts are audited by the census, not edited here. The `confidence:`
open-key teaching and the situation Bad/Good pair from v-next.9 ride the same
candidate but are not part of this contract change.

## 8. Pre-merge review findings — fold into the implementation, do not ship around them

Ten-angle code review of the branch (2026-09-03, main...HEAD; 15 findings
filed). Tom's ruling: nothing merges from this branch; these become rows the
next session closes inside the same change, and the A/B measures the combined
result (gist + revise shape) against the round-2 `v41 + gist A` baseline.

| # | finding | row to close |
|---|---|---|
| 1 | **The gist bypasses the interaction resolver.** It is S1E-only instructional text hardcoded in `encode_contract.py`; a registered `s1e` override cannot see, edit, or disable it, the fingerprint does not cover it, and `--gist-file` exists only because `tests/interaction_override.py` cannot reach it. | Make it an interaction: `s1e_gist` with its code default indexed in `servers/interaction_defaults.py`, read in `_build_user_content` through `brain.get_interaction_prompt('s1e_gist')`, positioned by the assembler. The harness then A/Bs it through `inject_prompt` like `--s1e-template`; delete `--gist-file`. |
| 2 | **`--gist-file` double-splices.** The idempotence guard tests the *candidate* text after rebinding `ENCODER_GIST`; any capture carrying the production gist (or an older wording) gets a second gist. Empty file passes (`'\n' in text`). | Superseded by row 1. If a splice survives for pre-gist captures: one `--gist [FILE]` flag, a distinct local, strip-then-splice keyed on the contract text, non-empty assertion, and a `[gist]` line that names which text ran. |
| 3 | **The gist names a surface no revise op reaches.** "and any edge description" with no op; the natural encoding — `connect_to` on a revise — is silently dropped today and reported as written (§2, verified 2026-09-03: `_CONTROL_FIELDS` skip, not a KV row). | Exactly what §1–§4 implement. Until then the gist must not ship. |
| 4 | **Guardrail test defects.** `_teaching_surfaces` stringifies schema *values* (10 of 21 revise field names unfindable); taught-check is bare substring, retired-check is word-boundary; the "every field" check covers one field; retirement path self-contradicts (retire → both tests fire, or `assert fields` fires); `_teaching_surfaces` re-evaluated per field; gist allowlist split across three literals. | `json.dumps(tool)` for surfaces (as `test_retired_fields.py:152`); one `\b` matcher; bind the taught set to the union the new spec produces (`get_writable_fields()` ∪ `connect_to`); make retirement executable; hoist the surfaces once; one `GIST_OPEN_VOCABULARY`. |
| 5 | **Two duplications in the test.** `ENCODER_TOOL_NAMES` hand-copies `ENCODING_TOOLS` (function-local in `encode.py`); `RETIRED_OP_FIELDS` is a third retirement registry beside `tests/test_retired_fields.py::RETIRED_NODE_FIELDS` and `ALL_FIELDS`' `agent_writable: False`. | **Closed.** `ENCODING_TOOLS` lives in `encode_contract.py`, imported by `encode.py` and the guardrail. `RETIRED_OP_FIELDS` and its test are gone; `test_retired_fields` scans the gist, `generate_field_summary()`, and every `BATCH_OP_SPECS[*]['description']` beside the prompts. §3 states the one retirement policy. |
| 6 | **Gist emitted unconditionally while `<node_catalog>` is conditional** — "the catalog above" dangles on catalog-less payloads. | Gate the catalog bullets on `node_catalog`, or render the gist in two parts. |
| 7 | **The production prompt's payload legend is now false** ("`<scout_legend>` sits just before the timeline") and the gist has no legend entry. | Lockstep row: What I Receive names the gist block and its position. |
| 8 | **`partially_resolves` is in no aspect** (only `resolves` is in `correction_improvement`/`settlement`); the prompt, the gist, and the test allowlist all teach it. Predates this branch. Not a loss: the S2 aspect unit classifies unhomed relation verbs on its cadence — but a verb we teach on purpose should not wait for classification. | Tom ruled (2026-09-03): add it — human edit to `aspects_v1.json` beside `resolves`, plus the `REQUIRED_ASPECTS` line only if a new aspect is introduced. |
| 9 | **Agent-name literals.** v-next.7 removed three (`I am Anchor…`, the identity example's title and quote); v9 was built from production and inherited them, and its new situation Bad example added a fourth ("Anchor rename"). | Carry v7's three D-12 edits into the candidate; neutralize the example text. |
| 10 | **Op accounting is inconsistent across tools.** Harness `score_arm` counts connects and disconnects as creates and archive as revise (30/77 real dumps have wrong `creates`; one shows situ 40% where the true rate is 100%); `eval/encoder_ops_shape.py` classifies correctly but drops why-less edges from every edge metric (why-count is the denominator), counts the `relations: [...]` form as `''`, never iterates `remember_batch`'s batch-level `connect_to`, counts `absorb` as a full rewrite, substring-matches retention (`5` in `125`, no separator before the edits blob), and crashes on any non-F dump or unknown chain (`PAYLOAD[chain]`). Neither applies `contract.unwrap_operations`. | One shared `kind()` + `_ops_of` (with `unwrap_operations`, batch-level `connect_to`, `relations`) used by `score_arm`, `score_gold`, and the shape scorer; fold the shape metrics into `score_shape`; import constants and extractors from `corpus_shape` / `s1s_ab_quality_analyzer`; `d['chain']`; word-boundary retention. Update the `_edge_pairs` reader for `target`. |
| 11 | **Provenance.** The shipped `ENCODER_GIST` differs from the measured `gist_a.md` by one test-forced edit; arm-F dumps carry no gist flag or hash; `test_s1e_lived_sequence`'s `'now=' not in body` now spans editable prose. | Record the gist text hash in the dump JSON and run line; any wording change is a re-measure; scope the lived-sequence assertion to the `<timeline` tag. |
| 12 | **Checklist head** said "Uncommitted; not promoted" on a committed branch. | Fixed this session: "committed on the branch, unmerged, not promoted". |

What the review confirmed clean: arms A–D render through production and get the gist for free; the control-integrity check has a blind band, not a break; the splice is byte-exact with production in the nominal case; the gist's per-run cost is ~300 tokens on a body that is a cache write every run regardless; the new test's import chain is pre-paid by `conftest`.

