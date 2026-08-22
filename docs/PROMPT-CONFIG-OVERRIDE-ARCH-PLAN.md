# Prompt/Config Override Model — Architecture Plan

## § Steps 1–6 live; next is Step 7, and every gate is now ruled (2026-08-22) ◀ ACTIVE ARC

**Read first:** handoff node `id:080016bf`; plan index `id:700654c9`; session-4 record
`id:7775ec63`; the four rulings `id:0274bca8`.
"No pointer" is now **honest** (Step 5, main `d375868` — both MAX(version) fallbacks deleted) and
**reachable** (Step 6, main `eefc15b` — `clear_interaction_override` at DAL/Brain/MCP, unified
cache invalidation, registration never auto-activates). A 5-lens Opus fleet then found a **third**
MAX fallback in `sync_prompts._fetch_active` — the tool writes the seed `.py` that
`interaction_defaults` imports as the *code default*, so one clear plus a routine sync could have
deployed an un-eval'd dormant prompt fleet-wide; it now returns `None` ("no override deployed —
seed file is authoritative"). Surface fingerprint `af8471e407ef` byte-identical across the whole
migration, verified live post-restart.
**Locked:** overlay semantics (decision 1); accessor signatures; `source='override'` = row
*contributed*, fingerprint equality is Step 8's classifier; **Step 8's collapse runs DAEMON-ONLY**,
not as a `LOGS_MIGRATIONS` step (`id:ffc58bda` — a migration-step collapse would fire inside frozen
eval corpora and make them float); **Step 7 consolidates rather than repairs in place, and includes
the reach-around table**; review rhythm = no fleet for 7, full fleet for 8, narrow 2–3 lens for 9.
**Open:** Steps 7–9. No gate is currently blocked on a human.
**Do not reopen:** wiring s2_community's interaction read (post-Step-8); a sibling
`get_effective_*` accessor; whole-value override; guarding (instead of deleting) the schema
backstop — its unversioned re-run property is the bug; env-flag exemption or float-acceptance for
the corpus question (both rejected, `id:ffc58bda`).

**`SEED_PROMPTS_VERSION` bumps before Step 8 are a COST, not a prohibition.** A bump makes reconcile
register+point ~7 names with `RECONCILE_PROVENANCE`, enlarging the set the collapse must classify —
but those rows are pristine, so the collapse still handles them correctly. And a bump is *compelled*
by the append-only fingerprint ratchet whenever shipped template content changes, so it cannot be
deferred at will. **Sanctioned instance: 2026-08-22, the BRAIN_VERSION 31 voice-field rename**
(`user_raw_quote`→`their_raw_quote`, `anchor_raw_quote`→`my_raw_quote`) bumped it for s1e, s2_healer
and s1_scout_quote — a sibling stream's work, not this arc's. Step 8 should expect the larger
classify set and treat it as normal.

Migrate prompt/config storage from **DB-owned defaults** to **code-owned defaults + DB overrides**.
Direction approved by Tom 2026-08-17 (brain `id:63e6b1f8`). This doc is the executable worklist;
the reasoning lives in the brain. Each step is written to run cold in a separate session.

**Target shape.** `brain.get_interaction_prompt(name)` / `get_interaction_config(name)` return the DB
override when one exists, else the code default (a `SYSTEM_PROMPT` constant in a prompt `.py` file, or
a config dict in the consumer's contract file). Nothing is ever written between code and DB. A plugin
update changes the default; every name without an override follows on the next read.

## Settled decisions (do not re-litigate)

| # | Decision | Source |
|---|---|---|
| 1 | The accessor **overlays** the override onto the single code default. Partial overrides are the normal case. Strict whole-value replace was rejected: it would force an override of one knob to snapshot the whole config, freezing every other key against future code updates. | Tom, this session |
| 2 | `clear_interaction_override` is **required**, not optional. Without it an override is a one-way door and the model degrades back into the per-name freeze. | Tom, this session |
| 3 | The eval gate becomes a **process rule**: experimental changes land as overrides, get promoted into the code default after the eval passes. Accepted cost — a file edit + merge now deploys fleet-wide with no gate in the code path. | Tom, this session |
| 4 | Eval/A-B ergonomics is a **first-class requirement**, not cleanup: overriding a prompt for a test must not require edits in many places, and must revert automatically. | Tom, this session |
| 5 | Aspects (`servers/scales/s2/aspects_v1.json`) are **out of scope**. Already the after-model; deliberately no override layer. | `id:f8fb451a` |
| 6 | Register/activate/rollback survive as the override lifecycle. | `id:ad9c6ebb` |
| 7 | Exactly **one default home per interaction**, never a second fallback source. This is the invariant that prevents the `a6dfcfe3` trap. | `id:a6dfcfe3` |

## Blockers found by review (each has a step)

1. **Dropping an `interaction_active` pointer does not mean "no override."** `dal_logs.py:479-482`
   `get_active` falls back to `ORDER BY version DESC LIMIT 1`, and `schema.py:1725-1730` re-materializes
   a pointer at `MAX(version)` on **every** boot, committed at `:1745` *before* the migration runner at
   `:1749`. Clearing `s1e`'s pointer today leaves v35 active, silently, forever. → Step 5.
2. **The naive collapse predicate is actively harmful on `trace_recording`.** It is the only name where
   active (1) ≠ MAX (2). Its v1 params byte-match `TRACE_RECORDING_NORMAL`, so `_matches_shipped`
   returns True → "drop the pointer" → `get_active` returns v2 = `TRACE_RECORDING_DEBUG` =
   `{k: True for k in PAYLOAD_KIND_EXT}` → **full payload capture on every LLM round**. → Step 8 policy
   table (`PIN`).
3. **Five caller literals already disagree with their code default**, and the migration makes them
   reachable (today `seed_interactions` guarantees a row, so they are dead code). → Step 4.
4. **`shipped_prompts()` covers 7 of the 16 registered names.** Reusing it as the defaults registry
   silently drops the default for nine live names with readers, including `scopes` (governs isolation).
   → Step 3.
5. **`healer_contract.HEALER` already carries `max_tokens: 2048`** with zero readers of `HEALER[...]`,
   while production runs 4096. A session told "move the config into the contract file" will find the key
   already there and silently halve the cap. `aspect_contract.ASPECT` has the same dead-duplicate shape
   (values agree today — latent, not live). → Step 2.

## Measured ground truth (2026-08-17, live install)

- 21 interaction names. `s1e` 35 versions, `s2_community_enrichment` 24, `s2_edge_families` 16 (dead
  since 2026-05-04), `surface` 15, `s2_consolidation_enrichment` 13, `s1_scout_facts` 7, `s2_aspects` 5,
  `s1_scout_quote` 4, `s2_healer` 4, `encoding_agent` 2 (dead legacy), `s1_scout_temporal` 2,
  `trace_recording` 2 (active 1), 8 names at 1 version.
- **All 7 `shipped_prompts()` templates byte-match their code default**, and their params match. The
  collapse drops all 7 pointers here.
- Drift lives in the never-reconciled names: `boot` holds `tom_quotes_*` where code has
  `operator_quotes_*` (and **nothing reads either spelling**); `s1_scout_quote`/`s1_scout_temporal` hold
  an `output_schema` the code dicts deliberately omit *and* `tests/test_prompt_sync.py:95-101` asserts
  the omission, so code and DB can never converge for those two; `s2_community` has 25 code keys vs 8 DB
  keys with **one** key in common, and is never read by anything.
- `trace_events.interaction_id` is a bare `INTEGER` (`schema.py:875`, `:1545`) with **no `REFERENCES`**.
  193,164 trace rows; 8,438 (4.4%) carry a pointer; **2,022 of those point at `interaction_id = 7`,
  which resolves to nothing** — deleted `judge` rows, dangling since 2026-05-02, unnoticed for 3.5
  months. Nothing JOINs on the column anywhere; the only consumers are display.
- `interaction_id` is `last_insert_rowid()` and `version` is a per-install counter, so neither is stable
  across installs. Cross-install K comparability is a property the pointer **never had**.
- Eval monkeypatch surface is **4 sites**, not ~20: `interview_encoder_probe.py:76`,
  `s2_consolidation_eval.py:111`, `s2_locked_eval.py:130`, `s2_locked_probe.py:179`. The other ~16 hits
  are plain *reads* of the baseline — the larger and riskier population.
- `recall_query_expansion` v1 was created **2026-08-17**, proving create-only seeding still adds names
  to old installs. Do not assume an install-date cohort.
- `run_versioned_migrations` (`schema.py:551-618`) is **sound** — stamps at `:610` after the step loop,
  rolls back and re-raises with the stamp unwritten. The `dfc74ee` "stamp before run" defect is absent
  and `tests/test_versioned_migrations.py:61` guards it.

## Dependency order

```
1 (K-provenance stamp) ─┐
                        ├─→ 7 (eval ergonomics) ─┐
2 (relocate configs) → 3 (registry) → 4 (resolver) ─┬→ 5 (unload MAX fallbacks) ─┼─→ 8 (collapse) → 9 (delete machinery)
                                                     └→ 6 (clear verb) ──────────┘
```

Steps 1 and 2 are independent and can run in parallel sessions. Everything from 3 onward is a chain.

---

## Step 1 — Stamp K-provenance into traces so a default-run is attributable

**Problem.** Today "which K produced this Δ" is `trace_events.interaction_id` (a rowid) plus
`interaction_version` (a per-install counter). Under the new model a run on the code default has no DB
row, so both go NULL/0 — indistinguishable from "unrecorded." Neither value was ever stable across
installs, and 24% of existing pointers are already dangling.

**Target state.** Two new fields on the delta/selection trace metadata shape:
`interaction_fingerprint` = `sha256(name + template + canonical_json(config))[:12]` of the **effective**
value, always written; and `interaction_source` ∈ `{'default','override'}`. `interaction_version` keeps
its meaning (the override version, or 0 on a default run) and becomes *meaningful* rather than
ambiguous once `interaction_source` disambiguates it. `interaction_id` keeps taking the override rowid
when one exists. **No schema migration** — both ride the existing `metadata` TEXT column.

Add `Brain.get_interaction_stamp(name) -> {'fingerprint','source','version','id'}` beside the accessors,
backed by a pure `interaction_fingerprint(name, template, config)` in the contract file. This is the
owner-side computation that replaces three hand-rolled `(x or {}).get('id')` digs.

**Files & call sites.**
- Add: `servers/brain.py` near `:723-747` (`get_interaction_stamp`); `servers/trace_contract.py`
  (`interaction_fingerprint` + shape entry at `:303`, kwarg defaults at `:514`, coercion at `:580` —
  both fields need **defaults**, because `servers/mutation_emitter.py:225` raises against the registered
  shape and mutation traces legitimately carry no K).
- Change: `servers/scales/s1/encode.py:327-328` + `:345-351`; `servers/scales/s1/surface.py:125` +
  `:182-183` (the bare positional threaded through `:748, :939, :1046, :1069, :1285` collapses into one
  `stamp` dict — **five signatures shed a parameter**); `servers/scales/s1/surface_capture.py:87-112`.
- Unchanged: `servers/dal_logs.py:640-652, 703-707`; `servers/mutation_emitter.py:236`.
- Dashboard: `dashboard/static/lib/trace_detail.js` — add a `K#` chip; drive `K v` / `K default` off
  `interaction_source`; **retitle the `K id` tooltip at `:64`**, which claims "FK to the exact
  prompt/config row" and is already false for 24% of pointers.
- Precedent to promote rather than invent: `eval/longmem/corpus.py:66-80 source_token()` and
  `tests/test_seed_prompt_reconcile.py:460-466 _fingerprint()` already do exactly this hash.

**Verification.** `./dev pytest tests/test_trace_contract_sync.py tests/test_scribe_*.py -v`. Then
assert a real encode run's delta trace carries a non-empty `interaction_fingerprint` and
`interaction_source == 'override'` (this install has an active `s1e` pointer today).

**Blast radius.** Additive. Nothing reads the two new fields yet. ~6 files, medium diff, mostly the
`surface.py` parameter collapse.

**Depends on.** None — land first, so traces written during the migration are already attributable.

**Respects.** `id:b385ccea` (K provenance is what makes a boundary learnable). Contract-first: the new
fields are declared in `trace_contract.py`, not inline at write sites.

**Named follow-up (out of scope here).** S2 units have no K-attribution at all: their Δs ride
`mutation_emitter` (`interaction_id: None` hardcoded), so a healer mutation is unattributable to the
healer prompt that produced it. The organic fix is ONE edit, not per-unit wiring: the unit already
fetches its prompt centrally (`s2/base.py:526-527`) and every write flows through
`apply_encoder_attribution` (`s2/base.py:265`) — hand `get_interaction_stamp(name)` to that same
chokepoint and every S2 unit's traces gain the stamp at once.

---

## Step 2 — Relocate each config default into its consumer's contract file

**Problem.** Config defaults live in four kinds of home: `interaction_seed.py` `*_CONFIG_V1` dicts,
consumer contract modules, module-level dicts (`recall_laf.DEFAULT_CONFIG`), and caller literals. Once
seeding is deleted, `interaction_seed.py` — a file whose stated purpose is seeding — cannot be the home
for ~15 config dicts belonging to 8 different consumers.

**Target state.** One default home per interaction, in the consumer's `*_contract.py` where one exists.
Every consumer already has one.

| From `servers/interaction_seed.py` | To |
|---|---|
| `S1E_CONFIG_V1:74` | `scales/s1/encode_contract.py` beside `ENCODING_AGENT:146` (add `model`, `effort`) |
| `SURFACE_CONFIG_V1:64` | `scales/s1/surface_contract.py` as `SURFACE_INTERACTION_DEFAULT`, adjacent to `SURFACE_MODEL:214` with the same "tier" comment |
| `S1_SCOUT_*_CONFIG_V1:117-181` + the three `*_CATEGORY` strings | `scales/s1/scouts/contract.py` beside `FACTS_OUTPUT_SCHEMA` — collapses the cross-file identity invariant `tests/test_prompt_sync.py:93` currently polices |
| `S2_{COMMUNITY,CONSOLIDATION}_ENRICHMENT/HEALER/ASPECTS_CONFIG_V1:93-107` | the matching `*_contract.py` |
| `RECALL_QUERY_EXPANSION_CONFIG_V1:83` | `servers/recall_expansion_prompt.py` (sole reader; do not invent a contract file for two keys) |
| `BOOT/VOICE/PRE_EDIT/SIGNAL_CONFIG_V1:183-209` | **delete** — see below |
| `recall_laf.DEFAULT_CONFIG:91` | stays put; `recall_laf.py` is the sole consumer |

**⚠ Hazard — read before editing `healer_contract.py`.** `servers/scales/s2/healer_contract.py:13-16`
already contains `'model': 'claude-haiku-4-5', 'max_tokens': 2048`, with **zero readers** of
`HEALER[...]` anywhere in `servers/`, `eval/`, or `tests/`. Production runs **4096**
(`interaction_seed.py:101-103`). Reconcile to **4096** — do not trust the value already in the file.
`servers/scales/s2/aspect_contract.py:26-30` has the identical dead-duplicate shape; its values agree
with production today, so verify rather than assume. Also dead: `community_contract.py:158-166`
`COMMUNITY_ENRICHMENT` — same two values, imported by nobody. The contract module already declared
ownership and the seed forked it; **adopt the contract copy and delete the seed duplicate**.

**Four config dicts to delete outright** (zero readers, every key grepped): `BOOT_CONFIG_V1`,
`VOICE_CONFIG_V1`, `SIGNAL_CONFIG_V1`, `PRE_EDIT_CONFIG_V1`. `PRE_EDIT` is the trap — its two
plausible keys are read from a **different store**: `servers/brain_assembly.py:52,85` call
`self.get_config(...)` → `Brain.get_config` (`brain.py:1868`) → `brain_meta`, not `interactions`.
Deleting a reader-less *default* removes no override hook (the accessors stay generic), so this does not
violate decision 5. `interaction_seed.py:251-278` already concedes "Several are dead config with no
reader at all."

**Verification.** `./dev pytest tests/test_scout_contract.py tests/test_contract_sync.py tests/test_interactions_runtime.py -v`. Then, per interaction, assert the relocated dict is byte-equal to
what the live DB active row carries — except `s2_healer`, where the intended answer is 4096.

**Blast radius.** No behavior change if done correctly; a wrong `max_tokens` is silent. ~12 files.

**Depends on.** None.

**Respects.** Contract-first (CLAUDE.md). "Extend before creating" — every destination file exists.

---

## Step 3 — Build the complete defaults registry

**Problem.** There is no name→default index. `shipped_prompts()` (`interaction_seed.py:251-295`) is the
closest thing but returns **7** names, because its question was "what should the fleet be force-advanced
to" — a question this migration deletes. `seed_interactions:465-564` registers **16**. Reusing the
7-name roster silently drops the default for nine live names *with readers*, including `scopes`
(governs isolation — `servers/scopes.py:250`) and `trace_recording` (`brain_traces.py:954`).

**Target state.** `servers/interaction_defaults.py` — one concern: the name→`(template, config)` index
the accessors resolve against. Imports only; no content of its own; target under 80 lines. Deliberately
**not** named `shipped_prompts` so it cannot inherit advance semantics. Covers all 16 registered names.
Optional per-name validator hook (this is where `brain.py:788-800`'s `if name == 'scopes'` special case
goes to die).

**Names with no code default, and why:** `encoding_agent`, `s2_edge_families`, `s2_node_families` are
dead — absent from the registry, existing rows stay as inert history. `recall_laf`'s default is
`recall_laf.DEFAULT_CONFIG`; register it in the index so the shared resolver serves it (this is what
lets Step 4 delete the local merge). `s2_community` is registered but **never read** — decide: either
wire the read at `community_decoder.py:97` (`config or brain.get_interaction_config('s2_community') or
COMMUNITY_DETECTION`) or drop the name. Do not carry the seed's shape forward undecided.

**Files & call sites.** Add `servers/interaction_defaults.py`. Update the CLAUDE.md Map row at `:58`
(`Interactions (the K store)` currently points at `interaction_seed.py`). Relocate
`interaction_fingerprint()` here from `trace_contract.py` (Step 1 parked it there because this file
didn't exist yet; it is K-identity, not trace shape — its Step 7/8 consumers import it for eval
assertions and the collapse comparison).

**Verification.** New `tests/test_interaction_defaults.py`: every registry entry's template is a
non-empty `str` >100 chars; every config is a non-empty dict; and — the replacement for the deleted
fingerprint-history test — **every literal name argument to `get_interaction_prompt` /
`get_interaction_config` / `get_interaction` in the tree is a registry key**. That test is what makes
Step 4's raise safe to ship.

**Blast radius.** Additive; nothing reads it until Step 4. One new file.

**Depends on.** Step 2 (the registry points at the relocated homes).

**Respects.** "One concern per file." Decision 5 — aspects are not in the registry.

---

## Step 4 — Move resolution into the existing accessors

**Problem.** Five readers hand-roll their own default fallthrough, and **five of their literals already
disagree with the code default**. Today `seed_interactions` guarantees a DB row so the literals are
unreachable; the migration makes "no row" the normal path and turns each into a competing default.

| Site | Literal | Code default | Consequence on a default run |
|---|---|---|---|
| `scales/s2/community_encoder.py:409-412` | `'claude-haiku-4-5-20251001'` | `claude-sonnet-4-6` | requests a **dated, retired** model → API 404 |
| `scales/s1/scouts/base.py:140` | `2000` | `5000` | truncates every facts-scout response mid-JSON |
| `scales/s1/surface.py:130,133` | `'legacy'` | `xml_v13` | **wrong layout** on the recall hot path |
| `scales/s2/base.py:528-529` | haiku / `4096` | `ASPECT` = sonnet / `8192` | wrong model and cap for `s2_aspects` |
| `scales/s1/encode.py:111` | `None` | `"medium"` | encoder effort silently becomes API-default high |

The dated model is pointed: `interaction_seed.py:87-91` documents that exact value *as the rot it
fixed* — but only the seed dict was corrected, never the caller literal.

**Target state.** `Brain.get_interaction_prompt(name)` and `get_interaction_config(name)` **become the
resolvers**, at their existing addresses, with **unchanged signatures**. Body: read the DB override; if
present, overlay it onto the registry default (decision 1); else return the default. Then delete every
literal above and let readers subscript (`cfg['model']`) — a resolved config is total by construction.

> **Signature is pinned. Do not add a parameter and do not add a sibling method.** All four monkeypatch
> sites install a `lambda name:` of exactly one positional arg (`eval/interview_encoder_probe.py:76`,
> `eval/s2_consolidation_eval.py:111`, `eval/s2_locked_eval.py:130`, `eval/s2_locked_probe.py:179`), and
> ~16 other eval sites read this method to get the A-arm **baseline** against an `IsolatedBrain` copy. A
> new `get_effective_prompt()` breaks the four silently; leaving the old one raw-DB-only makes the ~16
> run their baseline arm with an empty system prompt. One method satisfies both only if it *is* the
> resolver.

**Four guard semantics, all enforced in the resolver; delete all five caller-side guards.**
1. **No DB row → normal, silent, return the default.** Kills `surface.py:119-123`'s `RuntimeError`
   (whose message names the deleted `interaction_seed`), the two `print` WARNINGs in the S2 encoders,
   `s2/base.py:535-544`'s `_log_error`, and `scouts/base.py:129-136`'s stub path.
2. **No code default for `name` → raise.** A typo'd name or an unregistered new boundary. Today
   `s2/base.py:526`, both S2 encoders, and `scouts/base.py:296` have **no truthiness check at all** —
   after the migration a name missing from the registry would send an empty system prompt to Sonnet
   with no error. One resolver-side raise covers all four.
3. **Row present but unparseable JSON → `_log_error` + fall back to the default.** *New guard, and the
   most dangerous gap.* `brain.py:733-736` currently swallows `JSONDecodeError` and returns `{}`
   silently. Today `{}` means "use my literal"; after the migration it means "use the default" — so a
   typo'd override silently reverts the boundary with zero signal. `recall_laf.py:404-412` already has
   the right shape ("a broken K-store must be distinguishable from an empty one") — generalize it.
4. **Row present with an invalid value → validate at the resolution seam, log, use the default for that
   key.** Three sites already do this locally: `recall_laf.py:414-424` (`z_norm`),
   `brain_traces.py:1141-1148` (`retention_days`), `brain.py:786-800` (`scopes`, at the write door).
   The registry's validator hook gives them one home.

**Also collapse here:** `scouts/base.py:288-311 _load_interaction` (its own duplicate `json.loads`
parse — `encode.py:108-109` already says "reuse it, don't re-hand-roll the parse"; 2 callers at
`scouts/base.py:128` and `scouts/temporal.py:638`); `s2/base.py:431-436 _get_interaction_config` (a
one-line passthrough, 3 call sites, no subclass overrides); `recall_laf.py:400-403`'s key-level merge
(the overlay moves into the accessor — **keep** the `z_norm` validation at `:417-425`, that is a
read-time guard, not a default); `encode.py:100` + `:110`, which resolves `s1e` twice.

Keep `brain.get_interaction(name, version=N)` (`brain.py:755-759`) as an **inspection** accessor for the
MCP door and eval version-fetch paths. Do not add a default fallthrough to it.

**Verification.** `./dev pytest tests/test_interactions_runtime.py tests/test_recall_query_expansion.py tests/test_scout_llm_base.py tests/test_recall_laf.py tests/test_s2_client_lifetime.py -v`. Per-interaction
sentinel test in the shape of `tests/test_recall_query_expansion.py:217-225` (assert a sentinel config
value reaches the LLM call), **not** source-text pinning.

> **Test-integrity — report, do not weaken.** `tests/test_scribe_model_resolution.py:26-31` pins the
> **exact source line** `enc_model = enc_cfg.get('model') or 'claude-sonnet-4-6'`, so this step fails it
> by construction. Its intent (`:37-40`: "a model literal in the call would make the config key dead")
> is right and belongs on the resolver, generalized to all readers. Raise it before changing it.
> Separately, `tests/test_interactions_runtime.py:362-363` asserts `get_interaction_prompt('judge') ==
> ''` — it passes only because `judge` is an orphan with no default, giving zero coverage of the new
> path. Add a real name.

**Blast radius.** Largest step. Touches every runtime reader. A wrong overlay is silent — hence the
sentinel tests.

**Depends on.** Step 3.

**Respects.** Decision 1, decision 7. `id:a6dfcfe3` — one default home, no chained fallback. The
three-level `config.get(k, self.config.get(k, LITERAL))` chain in both S2 encoders is that exact trap
and dies here.

---

## Step 5 — Unload the two MAX(version) fallbacks

**Problem.** "No pointer" cannot mean "no override" while two mechanisms resurrect `MAX(version)`:
`dal_logs.py:478-482` (`get_active`'s `ORDER BY version DESC` fallback, justified as "covers pre-seed
bootstrap windows" — a window that ceases to exist) and `schema.py:1714-1731` (an unconditional
`INSERT OR IGNORE … SELECT name, MAX(version) … GROUP BY name` stamped `BACKSTOP_PROVENANCE`, run at
**every** `ensure_logs_schema`, committed at `:1745` *before* the runner at `:1749`). Together: the
collapse would appear to succeed and change nothing, then a boot would re-install a spurious override on
every name. Ordering cannot fix this; only deletion can.

This is also a live constraint violation — the fallback is *policy* ("what runs when nothing is
deployed") living inside the DAL.

**Target state.** `InteractionDAL.get_active(name)` returns `None` when no `interaction_active` row
exists. Pointer presence becomes the single unambiguous "an override is deployed" bit. Delete
`schema.py:1714-1731` entirely — do not guard it; its unversioned-so-it-re-runs property is exactly what
flips from self-healing to self-corrupting. **Keep the `BACKSTOP_PROVENANCE` constant** — Step 8 needs
it to classify the 8 live rows that carry it.

**Latent bug this closes.** `trace_recording` is active=1/max=2, so any future path that deletes that
pointer silently enables `TRACE_RECORDING_DEBUG` (all payload kinds on). No such path exists today
(grep: no `DELETE FROM interaction_active` in `servers/`), which is why it is a loaded gun with no
trigger. Step 6 adds the trigger, so this step must land first.

**Files & call sites.** `servers/dal_logs.py:478-486`; `servers/schema.py:1714-1731`.

**Verification.** `./dev pytest tests/test_interactions_runtime.py tests/test_logs_schema*.py -v`. Add:
delete a pointer on a scratch brain, assert `get_active` returns `None` and the accessor returns the
code default; reopen the `Brain` and assert **no** pointer was re-created.

**Blast radius.** Small diff, high leverage. Any code relying on the MAX fallback breaks loudly.

**Depends on.** Step 4 (the accessor must have a default to fall through to, or every name resolves to
nothing).

**Respects.** "No policy inside the DAL." Blocker 1.

---

## Step 6 — Add the clear verb, at all three layers

**Problem.** Grep confirms there is **no `DELETE FROM interaction_active` anywhere in `servers/`** — the
only ones in the repo are in `tests/test_interactions_runtime.py`. So "clear the override, revert to the
code default" has no DAL method, no brain method, no MCP tool. Today that is correct (there is no
default to revert to). After the migration it is the exact inverse of `set_active`, and without it the
only way back is registering a *copy of the default as a new version* — the DB-owns-the-default disease
being removed. It is also what makes an auto-reverting eval override possible (Step 7).

**Target state.**
1. `InteractionDAL.clear_active(name) -> bool` beside `set_active` (`dal_logs.py:436`). Pure mechanics:
   delete, `commit_unless_batched`, return whether a row went. No policy.
2. `Brain.clear_interaction_override(name)` beside `set_interaction_active` (`brain.py:762-773`). This is
   where cache-invalidation policy lives.
3. `_handle_clear_interaction_override` in `servers/dispatch_observability.py` beside `:214`, plus a tool
   entry beside `servers/brain_mcp.py:726`. Report "no pointer existed" **distinctly** from "cleared" so
   "already on the default" is observable. No `set_by` is written, so the reserved-provenance check does
   not apply.

**Fix the pre-existing cache asymmetry in the same change.** `set_interaction_active` special-cases
`if name == 'trace_recording'` to call `invalidate_trace_recording_cache()` (`brain.py:766-772`), while
`recall_laf`'s 60s config cache (`recall_laf.py:397-399`, `CONFIG_TTL_S:136`) has **no invalidation hook
at all** — a `recall_laf` pointer flip is already up to 60s late today. Add one
`Brain.invalidate_interaction_caches(name)` that both `set_active` and `clear` call, and retire the name
special-case.

**Also here:** drop `AUTO_V1_PROVENANCE`'s v1 auto-activate (`dal_logs.py:425-437`). Its comment
("Otherwise nothing is reading anything for this name") is false once a code default exists, and
auto-activating v1 means "the first register silently deploys" — contradicting decision 6. Keep the
constant; it classifies 4 live rows.

**Verification.** `./dev pytest tests/test_interactions_runtime.py tests/test_dispatch_contract_sync.py tests/test_trace_modes_forensics.py -v`. Add: register+activate an override, clear it, assert the
accessor returns the code default and that a cached reader (`trace_recording`, `recall_laf`) sees the
change immediately rather than after the TTL.

**Blast radius.** Additive, plus one behavior change (no v1 auto-activate) that
`tests/test_interactions_runtime.py:190` may pin — check before editing.

**Depends on.** Step 4.

**Respects.** Decision 2, decision 6. "A missing function on the owner is the finding, not license to
bypass" — added at all three owners rather than reached around.

---

## Step 7 — Make eval and A/B override a prompt in one place, and revert automatically

**Problem (Tom's requirement, decision 4).** Overriding a prompt for a test today means touching one of
**six** hand-rolled register+activate implementations, and there is no way to revert — so nothing can
clean up after itself, and a forgotten override silently opts that name out of code defaults forever.

Six copies: `eval/longmem/build_corpus.py:223-241` (`_apply_interaction_override`, the most general —
`None` on either side preserves the active value); `eval/longmem/harness.py:224-247`
(`_apply_s1e_override`) and `:250-277` (`_apply_surface_override`);
`eval/encoder_eval/targeted_v24_eval.py:52-66` (a fourth copy whose docstring also claims to generalize
the harness one); `eval/longmem/ab_encode.py:45-58` (`inject_prompt`, + a read-back assert);
`eval/laf/walker/field_cache_build.py:144-153` and `eval/ab_community_model.py:132-147` (both inline,
both deliberately merging, each having independently rediscovered merge-vs-replace and read-back
verification). `eval/longmem/leg_b.py:127-130` imports the helper *from a corpus-builder script*, which
is the evidence it is in the wrong place.

**Target state.**
1. **One helper, one home** — an eval-support module (`eval/_support/` or beside
   `tests/isolated_brain.py`), **not** `servers/`:
   `override_interaction(brain, name, *, template=None, parameters=None, merge=False) -> int`.
   `None` preserves the active value; `merge=True` overlays keys onto the stored config; read-back
   verification always. Then `harness._apply_s1e_override` / `_apply_surface_override` become a
   two-line file read plus one call, and the other four copies are deleted.
2. **A context manager that cannot leak** — `with interaction_override(brain, name, template=X):`
   clearing on exit including on exception. This is only writable because Step 6 exists.
3. **Assert on the effective value, cheaply** — arms compare
   `brain.get_interaction_stamp(name)['fingerprint']` rather than string-comparing multi-KB templates.
   That is `id:a6dfcfe3` / `id:432d5105` (mutation-verify at the effective value) made ergonomic, and it
   replaces `ab_encode`'s read-back assert and `ab_community_model.py:141-147`'s `sys.exit(2)` with one
   shared check.
4. **A stray-override check.** Production should carry zero overrides after Step 8, so any pointer is
   either deliberate or a forgotten revert. `InteractionDAL.list_all` (`dal_logs.py:503-519`) already
   returns `active_set_by` — surface it as a `./dev` check or a boot log line, flagging `eval:`-tagged
   pointers on a production brain.
5. **Baseline arms must clear explicitly.** `IsolatedBrain` (`tests/isolated_brain.py:42,88-91`)
   `snapshot_to`-copies production `brain_logs.db`, so an isolated eval brain **inherits every
   production override** (`s1e` v35, `surface` v15 today). A baseline arm that does not clear would
   measure production's override and label it "the default."
6. **Content-address corpora on the fingerprint, not the version int.**
   `eval/longmem/build_corpus.py:286-291` stamps `interaction_overrides = {name: int(version)}` into the
   corpus hash. After the migration a version int is not a complete address — version-absent means
   "code default" — so two corpora built against *different code-default generations* hash
   **identically**, and `load_manifest` at `:304` returns the wrong arm's corpus. Same silent-arm-
   collapse class as `a6dfcfe3`. `eval/longmem/corpus.py:66-80` already uses the fingerprint idiom.

**Accessor reach-arounds to fix while here** (each bypasses validation and cache invalidation):

| Site | Now | Use |
|---|---|---|
| `eval/longmem/build_corpus.py:230` | `brain._interaction_dal.get_active` | `brain.get_interaction(name)` |
| `eval/longmem/harness.py:239-248` and `_apply_surface_override` | `_interaction_dal.register` / `.set_active` | `brain.register_interaction` / `set_interaction_active` (skipping them bypasses the `scopes` validation at `brain.py:788-798` **and** trace_recording cache invalidation at `:768-772`) |
| `eval/longmem/artifacts.py:139` | raw SQL on `interactions` | `brain.list_interactions()`; **and** dump resolved default fingerprints — `:131-133` labels it "which prompt versions the encoder used", which becomes wrong when it dumps overrides only |
| `eval/longmem/artifacts.py:161` | raw SQL on `trace_events` | a `TraceDAL` read |
| `eval/oracle_audit/ab_prompt_ablation.py:32` | raw `SELECT template FROM interactions` | `brain.get_interaction('surface', version=N)` |
| `scripts/count_surface_tokens.py:58` | raw SQL | `brain.get_interaction(...)` |
| `eval/s2_consolidation_eval.py:429-438` | imports `CONSOLIDATION_ENRICHMENT_PROMPT`, which **does not exist** (the module exports only `SYSTEM_PROMPT`) — an `ImportError` behind unreachable code, plus a `_interaction_dal.register` reach-around | delete the block; use `brain.get_interaction_prompt(...)` |

**Verification.** `./dev pytest tests/test_raw_sql_guardrail.py tests/isolated_brain*.py -v`, plus a
context-manager leak test: enter, raise inside the block, assert on exit that no pointer remains. Then
one real cheap sweep through `eval/longmem/sweep.py` to prove an arm's fingerprint assertion fires.

**Blast radius.** Eval tree only — no runtime code. Wide but shallow; `tests/test_raw_sql_guardrail.py`
holds the line.

**Depends on.** Step 6 (the clear verb) and Step 1 (the fingerprint).

**Respects.** Decision 4. "Route, don't reach." Repo rule: no raw SQL outside `dal*.py`.

---

## Step 8 — The one-time install collapse

**Problem.** Every install carries DB rows that were the *default*, not a deviation. Until they are
reclassified, the install reads its frozen rows and receives no code updates. Getting this wrong
silently freezes or unfreezes prompts, and nothing fails loudly.

**Target state — a pointer operation, never a row deletion.** `interactions` rows are the override
history *and* the target of 6,416 resolvable historical trace pointers; deleting them raises no error
(no `REFERENCES` clause) and instead silently orphans display data — which has already happened once
(`interaction_id = 7`, 2,022 traces, unnoticed since 2026-05-02). **Delete zero rows.**

**The predicate is `_matches_shipped`'s semantics** (`interaction_seed.py:309-319` — template equality
**and** parsed-params equality; both halves matter), re-derived against the Step 3 registry, then that
function is deleted with the rest.

**Per-name policy table in code — five verdicts.** The table is the deliberate deployment decision in a
reviewable diff, carrying forward the one genuinely good property `SEED_PROMPTS_VERSION` had
(`interaction_seed.py:215-219`). It is also the only place `RETIRE` can be recorded once
`shipped_prompts()`'s exclusion categories are gone.

| Verdict | Names | Behavior |
|---|---|---|
| `COMPARE` | the 7 shipped + `pre_edit`, `voice_surface`, `signal_assembler`, `scopes`, `recall_query_expansion` | run the predicate: match → drop pointer; differ → keep as override |
| `ADOPT` | `boot`, `s2_community`, `s1_scout_quote`, `s1_scout_temporal` | drop the pointer unconditionally — content is dead or inert |
| `PIN` | `trace_recording` | **never touch** (active ≠ MAX, documented debug recipe) |
| `RETIRE` | `encoding_agent`, `s2_edge_families`, `s2_node_families` | drop the pointer, keep version rows |
| `SKIP` | `recall_laf` | never tell the collapse it has a code default, or the measured `{"z_norm":"support"}` tuning is dropped |

**Why `ADOPT` rather than `COMPARE` for those four.** `boot`'s live keys are `tom_quotes_*` vs the code's
`operator_quotes_*` and **nothing reads either** — `docs/DISTRIBUTION-READINESS.md:255` already cites
this exact row as the proof that seeding froze installs, so freezing it would enshrine the artifact that
justified the mechanism being deleted. `s2_community` has 25 code keys vs 8 DB keys with one in common,
and no reader. `s1_scout_quote`/`s1_scout_temporal` carry an `output_schema` the code dicts omit *and*
`tests/test_prompt_sync.py:95-101` **asserts** the omission — so code and DB can never converge, and
`COMPARE` would guarantee permanent override status for two scouts that never run
(`exclude_scouts=('quote','temporal')`).

"Match any historical default" was considered and rejected: it needs a git-archaeology table of retired
dicts (new permanent accretion) and still fails for quote/temporal, whose drift came from a human
`register`, not a stale default.

**Also required for `scopes`:** the code default never passes through `register_interaction`'s door
validation (`brain.py:788-800`), so the accessor must run `validate_scopes_config` on the default too —
otherwise the door is bypassed for the one value that ships to everyone.

**Mechanics.**
- Home: `LOGS_MIGRATIONS` (`schema.py:1666`, currently `[]`), `LOGS_VERSION` 1 → 2. **Not**
  `seed_prompts_version` — that key is being deleted and reusing it leaves a dangling counter.
- **Backup twice.** The runner fires `backup_before_destructive(db_path, 'v%d' % current)` at
  `schema.py:588-595` — but that tag is keyed on the stream version, so a future step at the same
  version silently reuses the file (`db_backup.py:209-214`). Add an explicit
  `backup_before_destructive(logs_db_path, 'pre-override-collapse')` inside the step.
- **Write an audit record before deleting anything**: `(name, version, set_by, set_at,
  sha256(template), parameters, verdict)` as JSON. The deleted pointers are the only unrecoverable
  information, and it is 21 rows. This turns rollback into a pure replay (`INSERT OR REPLACE INTO
  interaction_active`, delete the `logs_schema_version` stamp, restart) and lets an operator tell a
  *dropped* pointer from one that never existed.
- One implicit transaction, all 21 verdicts inside it. Stamp goes after the work — the runner already
  does this correctly.

**The loud check that closes the detection gap.** A wrong collapse is invisible from the after-state,
because "effective value unchanged" *is* the predicate. No structural test can see it. So, inside the
step: (1) snapshot `{name: (prompt, config)}` for all 21 names **through the accessors** before any
write; (2) apply verdicts; (3) re-read all 21; (4) any name whose effective value changed and is not
`ADOPT` → `brain._log_error('interaction_collapse_drift', …)` and **raise**, so the runner rolls back
and retries next boot. That single assertion catches the `trace_recording` disaster automatically
(before = v1 NORMAL, after = v2 DEBUG, changed, not `ADOPT` → refuse). Log the per-name verdict on both
branches, per `interaction_seed.py:414-415`'s own reasoning.

**Frozen-corpus caveat.** `ensure_logs_schema` runs inside `Brain.__init__`, so a `LOGS_MIGRATIONS` step
also runs in eval corpora, `IsolatedBrain` copies, and tests — where `reconcile_seeded_prompts` was
deliberately daemon-only (`interaction_seed.py:427-433`). The collapse leaves effective values unchanged
*at that instant*, but a collapsed corpus then **floats with future code edits**, which a frozen corpus
must not do. Either exempt eval/IsolatedBrain via an env flag or state that corpora are re-pinned by
overrides. **Decide before shipping.**

**Verification.** Two `IsolatedBrain` runs, one per git ref, each dumping the 21
`(name, sha256(prompt), config)` triples to the scratch dir; diff them. Any difference other than an
intended `ADOPT` is a failure. `tests/isolated_brain.py:78-80` clones via the online backup API, safe
against a live daemon — which is why this is legal where `Brain(db_path=…/brain.db)` is not. Prefer this
over any structural check; structural checks pass by construction here.

**Blast radius.** The only step that writes to a production DB. Recoverable via the audit record.

**Depends on.** Step 5 (or the collapse is inert and then reverts) and Step 6 (the clear primitive).

**Respects.** `backup_before_destructive` before any destructive DB operation. "Loud by Default."
Blockers 1 and 2.

---

## Step 9 — Delete the machinery, fix the docs, add the bypass guard

**Problem.** ~940 lines of distribution machinery become dead, and roughly 30 doc surfaces assert the
model that was just inverted. CLAUDE.md is current-state-only by rule, and a doc that lies is a defect.

**Delete.** `servers/interaction_seed.py` (its last surviving concern moved to
`servers/interaction_defaults.py` in Step 3). `servers/tools/sync_prompts.py` **entirely** — checked for
a survivor: `check_configs`'s semantics *invert* (after the migration "code ≠ DB active" is the
definition of an override, so an auditor firing on it is noise), and `_fetch_active` exists only because
the tool opens a raw `sqlite3.connect` instead of a Brain. The concern behind it — "no install is
silently running something the repo doesn't know about" — is served by
`_handle_list_interactions` → `InteractionDAL.list_all`, since **an override's existence is the
divergence**. Remove the `sync-prompts` `case` block in `dev:36-44` and its echo at `:50`, returning
`dev` to the pure exec wrapper its own header claims. Remove the `seed_interactions` call at
`brain.py:322-327` (after which `Brain.__init__` stops writing `interactions` at all — this dissolves
the frozen-corpus/race hazard at its source), the `reconcile_seeded_prompts` call in
`daemon_server.py:556-560`, and `scripts/create_fresh_brain.py:45-50` (whose comment names two dead
interactions).

**Tests: delete two files, migrate 8 invariants.**

| Survivor | New form | Destination |
|---|---|---|
| `test_prompt_sync.py::test_exports_system_prompt` | roster from the registry, not `SEED_PROMPTS` | `tests/test_interaction_defaults.py` |
| `test_prompt_sync.py::test_seed_role_in_docstring` | **inverts** — it currently asserts every prompt docstring says "seed-only / DB is authoritative", i.e. it *enforces the lie*. New: assert **no** default file makes that claim or carries a `Last sync:` line | `tests/test_interaction_defaults.py` |
| `test_prompt_sync.py::test_facts_config_carries_the_contract_schema` + `test_only_the_mustered_scout_ships_a_schema` | keep the `is FACTS_OUTPUT_SCHEMA` identity check; reach the dicts at their new home | `tests/test_scout_contract.py` (the current file apologizes for its own placement at `:82-84`) |
| `test_prompt_sync.py::test_seed_doesnt_override_externally_registered_version` | **the key survivor** → "an override survives a default change": activate an override, change the code default underneath, assert the override still wins. That is the whole migration in one test | `tests/test_interactions_runtime.py`, new `TestOverrideResolution` |
| `test_seed_prompt_reconcile.py::test_brain_init_does_not_reconcile` | stronger: reopen a Brain on an existing DB, assert `list_interactions()` version counts are byte-identical. Catches any future write-on-boot | `tests/test_interactions_runtime.py` |
| `test_seed_prompt_reconcile.py::ReservedProvenanceTest` ×3 | ~verbatim; reword the docstrings from "the reconcile reads these" to "the collapse migration and pointer audits read these" | `tests/test_interactions_runtime.py` |
| `test_interactions_runtime.py::test_surface_has_prompt_and_config` | keep the template↔layout atomic-flip check; read the pair from the registry | `tests/test_interaction_defaults.py` |
| `test_interactions_runtime.py::test_all_have_config` | assert every *code default* is a non-empty dict | `tests/test_interaction_defaults.py` |

Two source-parsing tests **die outright**, deliberately: `test_only_scouts_production_actually_runs_are_shipped` (its subject is roster membership, and its `re.search` on `encode.py`
only earned its place because membership was a silent decision) and `test_daemon_load_brain_calls_reconcile` (`inspect.getsource(_load_brain)` — there is no boot call left to be
missing). Do not repoint either at another symbol.

**The bypass guard (new, and the only thing that keeps decision 5's override hook alive).** The
dangerous shape is `from ...surface_prompt import SYSTEM_PROMPT` appearing in a **runtime** path,
skipping the override check and silently removing a learnable boundary. Add a test that greps `servers/`
for imports of `SYSTEM_PROMPT` from `*_prompt.py` and of `*_CONFIG_V1` symbols, allowlisting only
`servers/interaction_defaults.py`. Highest-risk sites, all currently correct and all one line from a
direct import: `surface.py:118-124`, `s2/base.py:526`, both S2 encoders, `scouts/base.py:296`,
`recall_laf.py:402-403`, `scopes.py:250`.

**Docs.** Rewrite the CLAUDE.md section at `:94-115` ("Encoder prompts: DB is authoritative, sync to
`.py`") — every sentence becomes false. New title and content: *code owns the default, the DB holds
overrides*; editing `<name>_prompt.py` **is** the deployment; register+activate installs a per-install
override that outranks the default and survives default changes; `clear_interaction_override` reverts to
the shipped default. Current-state only — no tombstone of the reconcile. Update the Map row at `:58`.
Update the two accessor docstrings at `brain.py:723-747` — they are the runtime contract.

Rewrite the docstring of all 10 prompt files (`scales/s1/encoding_prompt.py`,
`scales/s1/surface_prompt.py`, `recall_expansion_prompt.py`, the four `scales/s2/*_prompt.py`, the three
`scales/s1/scouts/prompts/*_prompt.py`). Each currently opens "Seed for interaction X — DB is
authoritative at runtime" and says "Do NOT edit this file to change prompt behavior" — the exact
inverse. Drop the `Last sync: DB vN` footer and the eval-gate paragraph (that belongs in CLAUDE.md
once, not in ten near-identical copies). Target ~3 lines each. **Two files need care:**
`scales/s1/scouts/prompts/temporal_prompt.py:3-10` carries hand-written content (that the scout is
algorithmic-first and the real work is in `temporal.py`) which `_patch_header_version` existed to
protect — preserve it explicitly. `scales/s1/surface_prompt.py:9-10` cross-references
`SURFACE_CONFIG_V1 (interaction_seed.py)` and must repoint to its Step 2 home.

Also stale: `servers/brain_mcp.py:714,720` advertise `encoding_agent`, `voice_surface`, `boot` as example
interaction names — three dead names in operator-facing text at the MCP door; use `s1e`, `surface`,
`trace_recording`, `scopes`. Four provenance why-comments (`dal_logs.py:365-374`, `:507-509`;
`dispatch_observability.py:224-228`, `:271-273`) explain themselves in terms of the reconcile.
`docs/TRACE-MODES-DESIGN.md:125` carries the "activate v2 = enter debug" recipe, which needs restating
now that nothing pre-registers v2 (entering debug becomes an ordinary override carrying
`TRACE_RECORDING_DEBUG`). Historical plan docs (`DISTRIBUTION-READINESS.md:255-314`,
`LLM-CLIENT-ARCH-PLAN.md:285-325`) may keep their record but not their current-state claims.

**Verification.** Merge tier per `id:feedback_full_suite_before_merge` — a wide `-k` composed from the
layers this touches (contract, guardrail, sync, interactions, traces, scouts, s2, recall), not narrow
feature keywords. Plus `./dev python3 -c 'import servers.brain'` and a daemon restart to prove no import
depends on a deleted module.

**Blast radius.** Largest deletion; behavior-neutral if 1–8 landed. Import errors are loud.

**Depends on.** Steps 7 and 8.

**Respects.** CLAUDE.md current-state-only. "Clean as you go." Decision 5.

---

## What re-accretes, and where it lands

Five special cases dissolve structurally (they are the payoff): the generation counter and its burned
version 1; `_pointer_is_pristine`'s provenance-as-proxy-for-intent (replaced by *presence* as intent);
`_pristine_advance_target`'s crash-residue adoption; `sync_prompts`'s entire generated-file-provenance
problem class; and `check_configs` + the fingerprint HISTORY test.

Three requirements survive and need a named home, or they will re-accrete badly:

1. **The eval gate.** Today a code edit reaches only fresh installs, and `sync_prompts._fetch_active`
   refuses to mirror a dormant candidate. After the migration the same edit is an instant fleet-wide
   deployment. `scouts/contract.py:131` already says "an edit here IS a deployment" — the migration
   makes that literally true everywhere. Home: decision 3's process rule.
2. **`trace_recording`'s debug mode.** With `seed_interactions` gone, nothing pre-registers v2, so the
   documented "activate v2 = enter debug" recipe breaks for every new install. Home: an ordinary
   override carrying `TRACE_RECORDING_DEBUG` (both readers at `brain_traces.py:954,1136` already overlay
   the code constant, so they are the closest-to-target readers in the repo — leave them alone).
3. **"Does this name still exist?"** `shipped_prompts()`'s exclusion categories encoded real
   information. Home: the policy table's `RETIRE` row. This install has 5 of 21 names with no reader.
