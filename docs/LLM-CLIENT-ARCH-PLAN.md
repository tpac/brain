# LLM Client Construction — Architecture Plan

## Scope

Where the brain constructs and configures its provider (Anthropic) client: the live
`anthropic.Anthropic()` construction sites, the key-resolution path feeding them, the timeout
policy, and the failure taxonomy they raise into.

**Boundary traced (2026-08-14, against `a6bb54b`):** 5 live construction sites in `servers/`
(+1 dead in `scales/s2/archive/`), 7 SDK call sites, 2 key resolvers, `llm_available`'s 6 live
consumers, and the full import graph for `ANTHROPIC_CLIENT_TIMEOUT`. Everything resolves
statically — no dynamic dispatch was left unfollowed. `eval/` and `scripts/` construct ~40 more
clients; they are **out of scope** (separate lifecycle, not runtime).

**Line numbers are perishable.** They were re-derived against `a6bb54b` after a sibling stream
landed the rejection latch. Re-grep before acting.

### The shape, stated plainly

The encoder lane is **already converged**; everything unconverged is the **recall lane**.

| Site | Lane | Timeout | Lifetime |
|---|---|---|---|
| `scales/runner.py:107` `make_client()` | encoder (S1E + **all 4 S2 units**) | `ANTHROPIC_CLIENT_TIMEOUT` 600s | per run |
| `brain.py:2261` `_ensure_anthropic_client` | daemon shared (feeds S1 surface) | 600s | daemon lifetime, key-stamped |
| `scales/s1/scouts/base.py:181` | recall | `timeout_seconds` | fallback only |
| `scales/s1/scouts/muster.py:121` | recall | none at ctor | per encode cycle |
| `brain_recall.py:108` | recall | **none anywhere** | per call |

S2 has **zero** construction sites: `consolidation_encoder.py:131`, `community_encoder.py:430`
and `s2/base.py:541` all call `runner.make_client()`. That is the `945feba` seam (id:c29dc917),
and it is why the 2026-06-08 audit's "11 sites" (id:820b66dd) is **5 today** — that node is stale
on the count, sound on the principle.

**Essential differences** (do not unify): the three client lifetimes — daemon-singleton for TLS
pool reuse (id:0c78a639), per-cycle for the 5m shared scout cache (id:e49766ac), per-run for
encoders; 600s for long encoder batches (community round 2 legitimately reaches ~218s) vs the
~90s scout deadline; `max_retries=0` on scouts (best-effort).

**Accidental differences** (this plan's subject): a per-call throwaway client in the S2
single-shot path; one genuinely unbounded client on the recall hot path; a constant imported
upward out of the encoder lane; two key resolvers with divergent semantics.

### Settled constraints respected

- **The two-lane split is deliberate** (id:c29dc917; `make_client`'s docstring explicitly scopes
  the recall lane out). This plan does not merge the lanes.
- **Model selection is user config, not code constants** (id:23a321af). The interactions table is
  the runtime home. This plan does not centralize model IDs — separate thread.
- **Provider-specific naming is a deliberate breadcrumb** (id:0f593eb1, Tom: *"it will help a
  future refactoring session to have the name anthropic"*). No provider-neutral renames.
- **`classify_llm_failure` and the rejection latch are landed** (`a6bb54b`). This plan composes
  with them and changes neither.
- **Constructor timeout ≠ request timeout** (id:c270184d, revised 2026-08-14). `with_options` at
  a call site can bound a client that looks unbounded at its constructor — which is why muster is
  fine and `brain_recall` is not.

---

## Dependency summary

Steps 0 and 3 are **independent** and can run in parallel sessions in any order. Step 4 overlaps
Step 1's file, so Step 1 goes first (or both in one pass). Step 2 runs last — it touches
`llm_available`, which the just-landed latch also occupies.

```
Step 0 ──┐
Step 3 ──┼── (independent, any order)
Step 1 ──┴──▶ Step 4          (shared file: brain_recall.py:107-113)
                                   Step 2  ← last; largest blast radius
```

**If the provider replacement is the driver, Step 4 is the one that matters** — it is what makes
the swap a config operation instead of a source edit. Steps 0/1/3 are hygiene; Step 2 is
correctness.

---

## Step 0 — Hoist the S2 single-shot client out of the per-call path

**Problem.** `scales/s2/base.py:541` evaluates `make_client()` *inside* `_call_llm`, so the healer
and aspect units construct a fresh `anthropic.Anthropic()` — new httpx pool, cold TLS handshake —
on **every** single-shot LLM call. The two loop encoders do not: `consolidation_encoder.py:131`
and `community_encoder.py:430` each hoist `client = make_client()` once per `_encode` and reuse it
across rounds. This is the same throwaway-client anti-pattern already removed from `surface.py`
(id:7acac778 — *"cold TLS every single time, never cached"*); it survived the `945feba`
convergence because that change made construction **uniform**, not **shared**. The cost is small
(S2 maintenance cadence, not a hot path); the reason to fix it is that the codebase currently
disagrees with itself about client lifetime.

**Target state.** `_call_llm` receives or reuses a client for the unit's run rather than building
one per call, matching what consolidation and community already do.

**Files & call sites.** `servers/scales/s2/base.py:541` (the `make_client()` call inside
`_call_llm`). Reference implementations: `consolidation_encoder.py:131`, `community_encoder.py:430`.

**Verification.** `tests/test_trace_delta_shape.py` is the only suite that currently exercises
`_call_llm` — thin coverage, so add a test asserting one client is constructed per run rather than
per call. Then `tests/test_aspect_encoder.py`, `tests/test_s2_retry.py`.

**Blast radius.** One file, one method. The telemetry note at `base.py:503` (elapsed_ms no longer
including client construction) becomes more accurate, not less.

**Depends on.** None — independent.

**Respects.** id:c29dc917 (routes through the runner seam, does not bypass it); id:7acac778 (applies
the decision already made for `surface.py`).

---

## Step 1 — Bound the recall-lane query-expansion client

**Problem.** `servers/brain_recall.py:108` builds `anthropic.Anthropic()` with no timeout **and**
no per-request `with_options` — the only genuinely unbounded client in `servers/`. It sits on the
recall hot path. Its `try/except` catches exceptions, not hangs: a stalled socket blocks a
ThreadPoolExecutor recall worker regardless. It also hardcodes `model='claude-haiku-4-5'` at
`:113` while `SURFACE_MODEL` already exists in `scales/s1/surface_contract.py:192`.

**Target state.** The request is bounded (constructor timeout, or `with_options` at the call site
as `scouts/base.py:192` does), and the model comes from the existing contract constant rather than
a literal.

**Files & call sites.** `servers/brain_recall.py:107-113` (`_expand_query_via_haiku`).

**Verification.** **There is no existing direct test** — `_expand_query_via_haiku` is referenced
only from `eval/`, not `tests/`. Add one asserting the bound is applied. Regression:
`tests/test_recall_laf.py`, `tests/test_recall_quality.py`.

**Blast radius.** One function. The bound must be generous enough not to truncate legitimate
expansion (~1s typical); this is a hang guard, not a latency budget.

**Respects.** id:c270184d as revised — muster needs no equivalent fix; `scouts/base.py:192`
already binds every scout request via unconditional `with_options`.

---

## Step 3 — Move `ANTHROPIC_CLIENT_TIMEOUT` out of the encoder lane

*(Numbered 3 for dependency clarity; independent of 0 and 1, and can run before Step 2.)*

**Problem.** `ANTHROPIC_CLIENT_TIMEOUT` is defined at `scales/runner.py:29` — the encoder lane —
but `brain.py:2246` imports it **upward** out of that lane to build the daemon's shared client.
`brain.py` is a layer above `scales/`; a daemon-level policy constant living in an encoder module
is a contract-first violation, and the dependency is invisible unless you read the import graph.

**Target state.** The constant lives in a contract file both layers legitimately depend on;
`runner.py` and `brain.py` both import it downward.

**Files & call sites.** Definition `scales/runner.py:29`. Importers: `brain.py:2246`,
`runner.py:107`, plus `eval/s2_locked_probe.py:251` and `eval/s2_absorb_prompt_probe.py:95` —
repoint all four. While in the file, fix the stale comment at `scales/s1/scouts/muster.py:178`,
which claims ghost scout threads are *"bounded by ANTHROPIC_CLIENT_TIMEOUT for LLM scouts"*: they
are not — muster's client never receives that constant, and the real bound is the scout's own
per-request timeout via `with_options`.

**Verification.** `tests/test_contract_sync.py`; import-smoke the two eval probes.

**Blast radius.** Mechanical, 4 import sites. Choosing the destination contract file is the only
judgment call.

**Depends on.** None — independent.

---

## Step 2 — One key resolver; lift the side effects out of `llm_available`

**Problem.** Two resolvers with divergent semantics coexist. `load_env()`
(`scales/dispatch.py:133`) is env-wins/no-override — it silently no-ops when `os.environ` already
holds a key, including a **revoked** one — and has 6 live callers across the encoder lanes.
`resolve_api_key()` (`:163`) is file-wins/always-current, with 3 callers, all in `brain.py`.

The encoder lanes are correct today only by side effect: `s2/base.py:531` calls `load_env()` *only
when `os.environ` is empty*, so a stale-but-present key is never reloaded — and the value is fresh
solely because reading the `llm_available` **property** rewrites `os.environ` on the way past
(`brain.py:2081-2085`). Key freshness for S1E and all of S2 therefore depends on an attribute read
having a side effect, across 6 production consumers of that property (`daemon_server.py:468`,
`daemon_hooks.py:480`, `brain.py:912/1624/2029`, `brain_voice.py:411`). The landed latch added a
second hidden effect to the same read (it can now clear the rejection latch).

**Target state.** The side effects live in an explicit refresh method that `llm_available` calls;
reading the property is a query. Encoder lanes obtain a current key by calling that method, not by
depending on someone else having read a boolean first.

**Files & call sites.** `servers/brain.py:2061-2110` (`llm_available` — the sibling stream left a
docstring paragraph naming this smell explicitly; **delete it when this lands**, it exists as a
hook, not a defence). `resolve_api_key` callers: `brain.py:2081`, `:2157`, `:2247`. `load_env`
callers: `scales/s1/encode.py:64`, `scales/s1/surface.py:1032`, `scales/s2/base.py:531`,
`scales/s2/consolidation_encoder.py:128`, `scales/s2/community_encoder.py:426`.

**Verification.** `tests/test_llm_rejection_latch.py` (landed with the latch — must stay green),
`tests/test_daemon_hooks.py`, `tests/test_keepalive.py`, `tests/test_contract_sync.py`,
`tests/test_dispatch_contract_sync.py`. This step changes the import surface and a shared
predicate, so per feedback on test tiering it warrants the **full suite** before merge, unlike
Steps 0/1/3.

**Blast radius.** Largest of the four. `llm_available` is process-wide with 6 consumers; the
latch's fingerprint logic reads the same key. Measure before assuming free: `resolve_api_key`
does a stat + read per call, so routing 6 more sites through it adds file I/O to the encode path.

**Depends on.** Land after Steps 0/1/3, and after the rejection latch has settled — it occupies
the same function.

**Respects.** The latch's ordering is deliberate — the `os.environ` sync at `brain.py:2083-2085`
runs *before* the latch check at `:2100`, so encoder-lane freshness is unaffected by latch state.
Preserve that ordering through the lift.

---

## Step 4 — Make model resolution table-driven at the three remaining sites

**Problem.** Three LLM call sites take their model from source, not config, violating id:23a321af
(*"model should be part of a config controlled by the user"*):

| Site | Today | Interaction row |
|---|---|---|
| S1 Scribe `scales/s1/encode.py:165` | `model="claude-sonnet-4-6"` literal | **`s1e` exists**, config has `max_tokens` but no `model` |
| S1 surface `scales/s1/surface_contract.py:192` | `SURFACE_MODEL` module constant | **`surface` exists**, config is `{"layout": "xml_v13"}` only |
| Query expansion `brain_recall.py:113` | `'claude-haiku-4-5'` literal | **none — unregistered interaction** |

The Scribe case is the sharpest: `encode.py:99` already reads `effort` from the table, then passes
a hardcoded model literal into the same `run_llm_loop` call. The primary encoder's model is the one
thing that cannot be configured. By contrast all four S2 units (`s2/base.py:511`) and all three LLM
scouts (`scouts/base.py:139`) already resolve model from `parameters`.

**Why this is now a prerequisite, not cleanup.** Under a provider **replacement** with per-unit
model choice, every model value must be a config row or the swap requires source edits. Provider is
global under that topology, so the table needs **no schema change** — the same `parameters.model`
rows just carry `gpt-…` values. These three literals are the entire difference between "switch
provider by editing config" and "switch provider by editing code".

**Target state.** Every LLM call resolves its model from its interaction's `parameters`, with the
contract value as seed/fallback only. Query expansion becomes a registered interaction — it is a
real learnable boundary with a prompt (`_EXPANSION_PROMPT`, currently a module constant in
`brain_recall.py`), and per id:66699ad5 the table is the source of truth for interactions at every
scale.

**Files & call sites.** `scales/s1/encode.py:165` + `S1E_CONFIG_V1` (`interaction_seed.py:157`);
`scales/s1/surface.py` model threading + `SURFACE_CONFIG_V1` (`:153`) + `surface_contract.py:192`;
`brain_recall.py:107-113` (register the interaction, read template + model from it).

**Verification.** `tests/test_prompt_sync.py` (seed/active mirror contract — a new interaction must
be seeded), `tests/test_contract_sync.py`, `tests/test_surface_transitions.py`,
`tests/test_daemon_hooks.py`. Follow the registered-DORMANT → activate → `./dev sync-prompts`
discipline for the new query-expansion interaction. **Guard against the known resolution trap**
(id:a6dfcfe3): interaction `parameters` beat a passed `config=` dict, so verify the *effective*
model inside the run, not the value passed in.

**Blast radius.** Medium. Registering a new interaction touches the seed roster and its contract
test. Changing where surface's model comes from touches the recall hot path — the value must not
change in the process; this is a plumbing change, not a model change.

**Depends on.** Step 1 overlaps `brain_recall.py:107-113` — do Step 1 first, or do both in one pass.

**Respects.** id:23a321af (model is user config); id:66699ad5 (interactions table is source of truth
at every scale); id:a6dfcfe3 (verify the effective model, not the passed one).

---

## Deliberately excluded

**An `LLMClient` / provider-adapter interface.** No second provider is chosen, so its shape would
be guesswork — and streaming and tool-use loop semantics diverge far more between providers than
client construction does, so a seam drawn at construction buys the least while appearing to solve
the problem. Forbidden by CLAUDE.md's rule against speculative abstraction. Steps 0–3 leave any
future adapter strictly easier and none harder.
**Trigger to revisit:** a second provider is actually selected.

**Converging `runner.py`'s retry tuple onto `classify_llm_failure`.** Tempting and wrong. The
retry tuple (`runner.py:223-227`) matches by `isinstance`, including `httpx.TimeoutException`,
which covers `ReadTimeout` / `ConnectTimeout` / `WriteTimeout` / `PoolTimeout` **by subclass**.
The landed classifier matches by class *name* (`dispatch.py:_TRANSIENT_NAMES`) — deliberately, so
the module imports no SDK. The two are not weaker/stronger but **incomparable**: name-matching is
broader (any third-party class named `ReadTimeout` would newly retry) *and* narrower — the landed
name set omits `WriteTimeout` and `PoolTimeout`, which the isinstance check covers today. Pointing
runner at it would silently change retry behaviour in both directions on a live path, for no gain.
**Trigger to revisit:** when an adapter exists, it owns `is_transient(exc)` — isinstance where the
SDK is importable, names where it isn't — and *both* runner and the classifier consume it. One
taxonomy, precision preserved, import-freedom intact.

**Merging the encoder and recall lanes into one factory.** The split is deliberate (id:c29dc917)
and the three lifetimes are real requirements, not preference. Merging trades a measured
cache-warmth benefit for symmetry.

**Splitting the provider concern out of `scales/dispatch.py`.** The file is now 319 lines in two
halves: ~180 of provider seam (`LLM_*` vocabulary, `_classify_one`, `load_env`, `resolve_api_key`,
`key_fingerprint`) and ~120 of write-command classification — two audiences, two lifecycles, and a
module docstring that advertises only the second. The co-location rationale is sound as written
(*"key resolution already lives in this module, so interpreting the refusal of the key we resolved
belongs beside it"*), but it argues for classification living beside key resolution, not for either
living in `dispatch.py`. Not worth churning a file that landed the same day.
**Trigger to revisit:** a second provider's mapper arrives — the landed comment already names it
(*"adding a second provider means adding a second mapper here"*). That is when the file stops
being cohesive enough.

*(Model-ID centralization was excluded here on the grounds that it was a separate thread. That
exclusion is **withdrawn** — see Step 4. The operator confirmed a provider **replacement** with
per-unit model choice, which promotes it from cleanup to prerequisite.)*

---

## Open questions

1. **Step 3's destination.** Which contract file should own `ANTHROPIC_CLIENT_TIMEOUT`? It is
   daemon-level policy consumed by two layers; `brain_constants.py` and `contract.py` are both
   plausible.
2. **Step 2's cost.** Routing 6 encoder-lane sites through `resolve_api_key` adds a stat + read per
   call. Acceptable, or measure first?
