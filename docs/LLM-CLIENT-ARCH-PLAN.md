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
Step 5 ── (independent; added 2026-08-19, can run before or after Step 2)
```

**If the provider replacement is the driver, Step 4 is the one that matters** — it is what makes
the swap a config operation instead of a source edit. Steps 0/1/3 are hygiene; Step 2 is
correctness.

---

## Step 0 — Hoist the S2 single-shot client out of the per-call path

**Problem.** `scales/s2/base.py` evaluates `make_client()` *inside* `_call_llm`. `_call_llm` has
exactly two callers, and they differ:

- **Healer** (`healer_encoder.py:88`) calls it **inside a batch loop**, so a run with N batches
  built N clients — N httpx pools, N cold TLS handshakes. **This is the defect.**
- **Aspect** (`aspect_encoder.py:69`) calls it **once per run**, so it was already effectively
  per-run and is unaffected either way.

*(An earlier draft of this step claimed both units were affected. Verified 2026-08-14: only healer
loops.)* The two loop encoders already do the right thing — `consolidation_encoder.py:131` and
`community_encoder.py:430` hoist `client = make_client()` once per `_encode`. This is the
throwaway-client anti-pattern already removed from `surface.py` (id:7acac778 — *"cold TLS every
single time, never cached"*); it survived the `945feba` convergence because that change made
construction **uniform**, not **shared**. The cost is small (S2 maintenance cadence, not a hot
path); the reason to fix it is that the codebase currently disagrees with itself about client
lifetime.

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

## Step 1 — Bound *and gate* the recall-lane query-expansion call

**Problem — two defects in one function.**

1. **Unbounded.** `servers/brain_recall.py:108` builds `anthropic.Anthropic()` with no timeout
   **and** no per-request `with_options` — the only genuinely unbounded client in `servers/`, on
   the recall hot path. Its `try/except` catches exceptions, not hangs: a stalled socket blocks a
   ThreadPoolExecutor recall worker regardless.
2. **Ungated.** `_expand_query_via_llm` is **the only LLM call site in the brain that never
   checks `brain.llm_available`** — there is no reference to it anywhere in `brain_recall.py`. The
   only guard at the call site is `_do_expand`, a *quality* heuristic on candidate score spread. So
   on a keyless brain, or one whose key is currently latched as rejected, this constructs a client
   and fires a call that can only 401 — swallowed silently by `except: return []`. Every other lane
   gates: S1 Scribe (`brain.py:912`), S2 (`brain.py:1624`), surface (`daemon_hooks.py:480`),
   keepalive (`daemon_server.py:468`), voice (`brain_voice.py:411`).

   **Severity: latent, not live.** Query expansion is opt-in via `BRAIN_QUERY_EXPANSION`
   (default `off`), the flag is set **nowhere in the repo**, and it is additionally force-disabled
   whenever `laf_v1` scoring is active — which it is in production. So a default install never
   fires this, and it did **not** contribute to the external install's 401 storm. *(An earlier
   draft of this plan called it a live billing risk and ranked Step 1 first on that basis; that
   was wrong.)* It is worth closing anyway: both defects become reachable the moment an operator
   turns the flag on, and a hole in an otherwise-universal gate is cheapest to fix while the file
   is open.

It also hardcodes `model='claude-haiku-4-5'` at `:113` (see Step 4).

**Target state.** The call is gated on `llm_available` before any client work, and the request is
bounded — constructor timeout, or `with_options` at the call site as `scouts/base.py:192` does.

**Note on ordering.** Gate first, then bound: the gate makes the unbounded-client window smaller by
removing the keyless/latched case entirely.

**Files & call sites.** `servers/brain_recall.py:107-113` (`_expand_query_via_llm`).

**Verification.** `tests/test_recall_query_expansion.py` — bound, gate order, and (since Step 4)
the effective config at `messages.create`. Regression:
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

### ⚠ Distribution gap: `s1e` and `surface` are NOT equivalent (found 2026-08-16)

The two sites this step touches are covered by *different* amounts of machinery, and treating them
the same would quietly recreate the frozen-prompt problem the distribution mechanism exists to kill.

`shipped_prompts()` covers `s1e`, `s2_aspects`, `s2_community_enrichment`,
`s2_consolidation_enrichment`, `s2_healer`, `s1_scout_facts`. **`surface` is not in it.**

| | `model` key on `s1e` | `model` key on `surface` |
|---|---|---|
| Fresh install | ✅ ships | ✅ ships (via seed only) |
| Existing pristine install | ✅ on a `SEED_PROMPTS_VERSION` bump | ❌ never reaches it |
| Rot detection | ✅ `sync-prompts --check` reports seed↔ACTIVE drift | ❌ nothing notices |

Config-only interactions were deliberately excluded because several are dead config with no reader.
But **`surface` stops being dead the moment this step puts a live model key in it.**

**RESOLVED (operator, 2026-08-17), in two halves:**
- **Model stays in `surface_contract.py` as code.** The value is coupled to
  `CACHE_MIN_PREFIX_TOKENS` (model-specific cache floor) and the warmup ping in the same file — a
  provider swap must edit that file regardless, so a config key would be false configurability.
  The constant carries a comment naming the decision and its trigger to revisit: when a provider
  adapter owns cache strategy, the coupling dissolves and the model moves into config.
- **Template + `layout` config DO ship.** `surface` joined `shipped_prompts()` and the
  `SEED_PROMPTS` sync roster in generation 3 (which was still unreleased, so it rode the same
  bump); the seed moved to `servers/scales/s1/surface_prompt.py` and mirrors ACTIVE (v15 — the
  mature prompt is the shipped default). Pristine fleet installs advance to v15; our own install
  is hands-off at v15 already.

**Related trap if the query-expansion interaction is eval-gated:** the seed must not reach its
candidate through an imported mutable constant. Either the seed reads through the ACTIVE pointer, or
a check enforces `constant == ACTIVE`. Otherwise registering DORMANT gates nothing and fresh brains
boot the untested candidate — this bit the facts-scout arc the same day (brain id:f9d36193).

*Credit: stream `17d9ae94`, who owns the prompt-distribution machinery.*

### The procedure, re-read against the landed machinery (`e3d9481`, 2026-08-16)

Their rewrite **landed**, and it adds a fifth step the earlier draft of this plan didn't have:

```
register_interaction(name, template)      # registers v(N+1), DORMANT
set_interaction_active(name, version=N+1) # flips the runtime pointer
./dev sync-prompts                        # mirrors ACTIVE → .py seed files
./dev sync-prompts --check                # drift gate
# then BUMP SEED_PROMPTS_VERSION in servers/interaction_seed.py
```

**The bump is the deployment decision for the fleet**, and Step 4 cannot skip it.
`shipped_prompts()` returns `(template, config)` pairs — **config ships too** — so adding a `model`
key to `S1E_CONFIG_V1` *is* a shipped-config change. `tests/test_seed_prompt_reconcile.py`
fingerprints the shipped templates and configs and **fails when they change without a bump**. That
test is a hard gate Step 4 will hit; treat it as the reminder, not an obstacle.

`reconcile_seeded_prompts` advances an install only while it still runs the shipped default —
any human `register`/`set_active` for a name makes that name hands-off on that install permanently.

**And here is the sharp edge of the `surface` decision.** `surface` is not in `shipped_prompts()`,
so it is **not fingerprinted**. A `model` key added to `SURFACE_CONFIG_V1` would therefore:
- reach fresh brains only,
- never reach an existing install,
- and **not even trip the test that exists to catch exactly this class of mistake** — because that
  test only sees what `shipped_prompts()` returns.

So the failure mode isn't just "it doesn't ship." It's "it doesn't ship *and nothing tells you*."
Decide deliberately: add `surface` to `shipped_prompts()` along with the live key, or leave its
model in `surface_contract.py` where it is honest about being code. Do not put a live key in a
config that nothing distributes and nothing checks.

**Depends on.** Step 1 overlaps `brain_recall.py:107-113` — do Step 1 first, or do both in one pass.

**Respects.** id:23a321af (model is user config); id:66699ad5 (interactions table is source of truth
at every scale); id:a6dfcfe3 (verify the effective model, not the passed one).

---

## Step 5 — Timeout granularity: stop letting connect inherit the read budget

*(Added 2026-08-19 from the e2dc24d3 incident. Independent of Step 2.)*

**Problem.** `ANTHROPIC_CLIENT_TIMEOUT = 600.0` (`brain_constants.py:389`) is a **flat** timeout:
passing a bare float gives httpx 600s for connect, read, write, and pool alike. The 600s figure is
read-side reasoning (community round 2 legitimately reaches ~218s) — but connect inherits it, so
"can't reach the server" is granted the same patience as "slow generation". Two multipliers stack
on top of the flat value:

1. **SDK retries.** `make_client()` sets only `timeout`, so the SDK default `max_retries=2`
   applies — one blocked call surfaces after up to 600s × 3 plus backoff (~30 min) before our code
   sees any exception. (S2's loop encoders wrap rounds in `retry_on_transient_api_error`
   (`runner.py:227`, attempts=2) on top of that — worst case doubles again.)
2. **The run deadline is cooperative.** `SCRIBE_RUN_DEADLINE_SECONDS = 1200` is checked between
   rounds and before the stream-fallback re-issue (`runner.py:514`, `:553`); it cannot preempt a
   thread blocked inside a read. `scribe_hung` (1800s) exists precisely because the deadline is
   blind while blocked.

**Incident that exposed it (2026-08-18, session e2dc24d3).** Network flapped ~21:00–22:10 UTC.
Two encode attempts failed *fast* (~3s clean `APIConnectionError` — connect-refused behaves
exactly as wished). The third attempt's socket connected and then stalled mid-run: ~30 min blocked
inside one `messages.stream` call (600s × 3 SDK attempts), `scribe_hung` at 1903s, deadline able to
fire only 53 min into the run when the SDK finally returned control — 1 of 4 encode permits held
throughout. Work landed on the next attempt, so blast radius was one permit; the same shape holding
all four permits through a longer outage is the version that hurts.

**Target state.**

- The constant becomes a granular policy — `httpx.Timeout(connect=~10, read=600, write=600,
  pool=600)` (exact split to be chosen at implementation). Dead-network connects fail in seconds;
  slow generation keeps its headroom. Both 600s consumers take it unchanged: `make_client()`
  (`runner.py:99`) and the daemon shared client (`brain.py:2400`).
- The encoder lane drops SDK retries (`max_retries=0`, or 1 at most). The Scribe already has its
  own retry layer above (`SCRIBE_RETRY_COOLDOWN_SECONDS` + `SCRIBE_MAX_FAILED_RETRIES`), and S2's
  loop encoders have `retry_on_transient_api_error` — SDK retries underneath those only multiply a
  block. This is the same reasoning that put `max_retries=0` on scouts and query expansion.
- **Measure before tightening read.** `messages.stream`'s read timeout applies per-event, and a
  healthy stream never goes minutes between events — but the *first* event arrives only after
  server prefill, so the read bound must clear worst-case TTFT, not typical inter-chunk gaps.
  `per_round_stats.ttft_ms` already measures exactly this; size any tighter read bound from that
  data, not from intuition.

**Files & call sites.** `brain_constants.py:389` (the constant and its comment — the comment's
read-side rationale stays true, it just stops applying to connect); `scales/runner.py:99`;
`brain.py:2400`. Scouts and query expansion are unaffected — they already bind per-request
(`scouts/base.py:179`, `brain_recall.py:106`) with `max_retries=0`.

**Verification.** A unit test asserting the built client's timeout object has connect ≪ read (both
construction sites); `tests/test_s2_client_lifetime.py` stays green. For the retry drop, the
existing failure-path tests (`tests/test_s2_retry.py`) must still pass — the wrapper's behaviour is
unchanged, only the hidden SDK layer under it goes away.

**Blast radius.** Transport policy for the encoder lane + daemon shared client. The connect split
and retry drop are safe; a tighter read bound is the only risky edge, which is why it is gated on
TTFT data.

**Depends on.** None — independent of Step 2.

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

## Appendix — provider-portability findings (researched 2026-08-14)

> **STATUS: RESEARCH OPEN — NOT A DECISION. Do not build against this appendix.**
> The operator's direction (2026-08-14): *"ill want to do more research on that later. Not focus on
> building it now."* Nothing below is settled. The five Steps above are the buildable work; the
> provider swap is a separate, later thread that starts with more research — not with this text.
> Treat the recommendation in §"Responses vs Chat Completions" as one stream's provisional take
> with named gaps, not as a finding.

Scoping input for the recall-lane seam, given the operator's stated direction: a **replacement**
(one provider at a time, globally) with **per-unit model choice**. Checked against current OpenAI
docs rather than assumed — three of four map cleanly, one does not.

| Brain uses | OpenAI equivalent | Portability |
|---|---|---|
| `effort` (`encode.py`) | `reasoning.effort` — `none`/`low`/`medium`/`high`/`xhigh`/`max` | **clean** — value map |
| `output_config={'format':{'type':'json_schema',…}}` (scouts, surface, expansion) | `response_format` json_schema + `strict: true` | **clean, with a schema audit** — strict mode requires every property listed in `required` and `additionalProperties: false` on every object. Our existing schemas need checking against that before they'd pass. |
| Tool-use loop (`run_llm_loop`) | Responses API (item-centered) or Chat Completions (message-centered) | **a decision, not a mapping** — OpenAI now has two APIs. Responses is the recommendation for agentic/tool-calling work; Chat Completions is closer to `run_llm_loop`'s existing message-array shape. Picking Responses is more divergence up front, less later. |
| **`cache_control` breakpoints, 1h + 5m TTLs** | **automatic caching only** — ≥1024 tokens, TTL fixed at 30 min | **⚠ does not port** |

**The caching gap is the one that costs something behavioural, not just syntactic.** The brain
deliberately places breakpoints at two tiers: 1h on the byte-stable system prompt
(`run_llm_once`, `scouts/base.py:175`) and 5m on the user prefix shared across the four scouts in
one muster cycle (id:e49766ac) — the first scout writes, the other three read. OpenAI caching is
automatic and prefix-based with a single fixed 30-minute TTL and no explicit breakpoints, so that
two-tier design has no direct expression: the 1h system cache becomes 30m, and the deliberate
shared-prefix write/read becomes implicit prefix matching. Scout economics depend on this (the
shared prefix is ~23K tokens, id:bebe1c8f), so **cost per encode cycle must be re-measured under
any swap — it will not be a like-for-like port.**

Nothing here blocks a replacement. It does mean the recall-lane seam should be shaped so the
caching strategy is a *provider concern* behind it, not a caller concern in front of it — today
`cache_control` blocks are assembled at the call sites, which is exactly what would have to change.

### Responses API vs Chat Completions — the wire-format fork

OpenAI has two APIs. If a swap proceeds, one must be chosen; under a **replacement** topology the
seam only ever carries **one** OpenAI wire format, so choosing wrong costs an adapter rewrite, not
a caller rewrite.

| | Chat Completions | Responses |
|---|---|---|
| Input | `messages[]` | `input[]` + top-level `instructions` |
| Output | `choices[].message` | `output[]` of typed Items (message / reasoning / function_call) |
| Tool call | `tool_calls[{id, function:{name, arguments}}]` | `function_call` Item, name at top level |
| Tool result | `{"role":"tool", "tool_call_id":…}` | `function_call_output` Item with explicit `call_id` |
| State | caller resends the array | `previous_response_id` / replay Items / Conversations API |
| Structured output | `response_format` | `text.format` |
| Reasoning | supported | "richer experience… improved tool usage" |

**How each lands against `run_llm_loop` as it exists** (`runner.py:671-685` — manual
`api_messages` accumulation, assistant `tool_use` blocks, `{"role":"user", content:[tool_result]}`,
terminate on `if not tool_uses: break`):

- **Chat Completions** preserves the loop's architecture. Same manual accumulation, same
  "no tool call → done" termination, same caller-owns-the-array model — the adapter is a
  translation layer, not a redesign. `record_round_fn`'s literal-messages capture
  (`runner.py:363-370`) maps 1:1, so the forensics layer survives unchanged.
- **Responses** has one architectural conflict, not merely a syntactic one: `previous_response_id`
  is server-side conversation state, and `retry_on_transient_api_error` deliberately re-runs the
  **whole loop** including writes that already landed (`runner.py:239-241`). Server-held state makes
  "re-run the whole loop" ambiguous. Declining that feature removes Responses' principal advantage,
  leaving the Item model — which costs us: `response.content[0].text` and the
  `''.join(b.text for b in …)` extractors change shape, `reasoning` Items need filtering, and the
  round-recorder payload shape needs rethinking.

**Provisional take (one stream, 2026-08-14, NOT a decision):** Chat Completions first — it is the
shape the loop already has, OpenAI supports it indefinitely, and under replacement the wrong pick
costs only an adapter. Add Responses behind the same seam later *if* reasoning-model tool use
measurably beats it.

**SETTLED (operator, 2026-08-14): the brain owns its state. No server-side conversation state.**
That makes Chat Completions the correct wire format for us on this axis.

**Why that decision is stronger than it looks.** The Anthropic Messages API is **stateless by
construction** — *"send the full conversation history each time"* — and has **no
`previous_response_id` equivalent at any layer**. Server-side compaction (`compact-2026-01-12`) and
context editing (`context-management-2025-06-27`) both look like state and are not: the client still
owns and resends `messages`, appending the returned compaction blocks. The SDK Tool Runner keeps its
own copy **in the client process**. Anthropic's only genuinely server-stateful surface is Managed
Agents — a separate platform (agents + environments + sessions + SSE), not a request parameter.

So `run_llm_loop`'s caller-owns-the-array shape was never a preference the brain selected — it is
the only shape the Messages API offers. Every line of the encoder lane was written under that
constraint. Two consequences worth carrying forward:

1. **Chat Completions is structurally the same model as what we already have** (client-owned message
   array, resent each turn). Responses' typed-Item model plus server-held state is the outlier
   relative to *both* our code and the API it was built against.
2. **If the brain ever does want server-side state, the comparison is not Responses vs Chat
   Completions** — it is Managed-Agents-style hosting vs self-hosting the loop, which is a different
   and far larger question than a provider swap. Out of scope here.

**Explicitly NOT verified — check these before any decision:**
- Whether **prompt caching behaves differently between the two APIs**. Unknown, and it matters: the
  loop places **three** breakpoints (`runner.py:339-348`), and caching is already the one dimension
  that does not port.
- The "40–80% cost savings" figure for Responses. It came from a secondary blog in search results,
  **not** OpenAI's documentation. Do not weight it without a primary source.
- Whether OpenAI-compatible endpoints on self-hosted/third-party stacks implement Responses at all
  — many implement only Chat Completions, which would matter if "hosted model" is ever in scope.

Sources: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs) ·
[reasoning models](https://developers.openai.com/api/docs/guides/reasoning) ·
[prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) ·
[migrate to Responses](https://developers.openai.com/api/docs/guides/migrate-to-responses)

---

## Status — updated 2026-08-19

**Steps 0, 1, 3 and 4 are MERGED AND LIVE. Steps 2 and 5 remain.**
Do not trust a commit hash written in this doc — main moved 20+ commits in a day across several
streams. Re-derive with `git log`.

**Steps 0/1/3** shipped with `tests/test_recall_query_expansion.py` (7 tests, one
mutation-verified). Pre-commit review: 4 findings, 3 fixed, 1 accepted (no key stamp on
`_llm_client` — adding one would put a `resolve_api_key()` file read inside healer's batch loop).
Full suite at merge: 2,586 passed, 8 skipped, 1 xfailed.

**All three are verified**, each by the observation that actually applies to it:
- **Step 3** — the warmup line `anthropic_client_ms:` proves the moved constant resolves inside
  `_ensure_anthropic_client`.
- **Step 1** — a live recall completing at all proves the new module-level import resolves; a
  failure there kills every recall.
- **Step 0** — `tests/test_s2_client_lifetime.py` (4 cases, mutation-verified: reverting the hoist
  fails the wiring test while the other three still pass).

**Correction worth keeping.** An earlier version of this section said Step 0 was "not yet verified
live, needs a real multi-round encode." That was wrong and would have sent a successor waiting for
a signal that cannot arrive: `_call_llm` routes through `run_llm_once`, which is single-shot and
emits no per-round log lines. There is no daemon.log signal for client-construction count and
never was — a test was always the only route (brain id:2706de84). Before recording anything as
"awaiting verification", name the exact observation that would constitute it.

**The rolling cache breakpoint (`6c07776`) is `ebef255c`'s work, not this plan's** — merged
alongside and now confirmed: 7 runs, 10 chain links, 0 breaks, across `[s1e]`, `[s2ce]` and
`[s2-consolidation]`. Its success test is the **cr chain** (`cr(N+1)` == `cr+cw(N)` exactly), not
`in` alone — 2 of 11 pre-fix runs already showed `in=1` from server-side auto-caching, and their
tell was a broken chain. One untested condition remains: every observed post-fix r0 ran 117–189s,
inside the 5m cache window, while the only known false pass came from a 368s r0. Detail:
brain id:7e15b37a.

**Step 4 implemented 2026-08-17** (worktree `jolly-neumann-4f1195`, merge pending): Scribe model
from `s1e` params (live DB v35 carries `model` + `effort`); query expansion is the registered
`recall_query_expansion` interaction (v1, prompt + model + max_tokens from the table, seed in
`servers/recall_expansion_prompt.py`, in the sync roster but deliberately NOT in
`shipped_prompts()` while `BRAIN_QUERY_EXPANSION` defaults off); surface ships template + layout
config while its model stays code (see the resolved decision above — folded into this same
generation, 2026-08-17). `SEED_PROMPTS_VERSION` 2→3; the gen-3 HISTORY row was replaced when
surface folded in (3 was never released — the append-only rule protects released generations). Both resolution paths
mutation-verified (`tests/test_scribe_model_resolution.py`,
`tests/test_recall_query_expansion.py::ExpansionEffectiveConfigTest`). The s1e params path goes
live at merge + daemon restart — the encode log line now prints the effective model.

**Step 2 remains**, last.

**Step 5 added AND its safe halves shipped 2026-08-19** (same day, same session): the connect
split (`ANTHROPIC_CONNECT_TIMEOUT = 10.0`, both construction sites on `httpx.Timeout`) and the
encoder-lane retry drop (`make_client` at `max_retries=1`). Pinned by
`tests/test_client_timeout_policy.py`. The remaining piece — a tighter *read* bound — stays open,
gated on `per_round_stats.ttft_ms` data as the step specifies.

---

## Open questions

1. ~~**Step 3's destination.**~~ **RESOLVED — `servers/brain_constants.py`.** It already holds the
   sibling stream's `LLM_REJECT_BACKOFF_MINUTES` / `LLM_REJECT_STRIKE_RESET_SECONDS` beside the
   daemon timing constants, and `scales/` already imports upward from `servers/` routinely
   (`from ..contract import …` in dispatch, s2, s1). `contract.py` was rejected: it is the **node
   field** contract ("single source of truth for node fields"), a different concern entirely.
2. **Step 2's cost.** Routing 6 encoder-lane sites through `resolve_api_key` adds a stat + read per
   call. Acceptable, or measure first?
3. ~~**Responses vs Chat Completions**~~ — **partly settled 2026-08-14.** The operator ruled the
   brain owns its state, so server-side conversation state is out and Chat Completions is correct on
   that axis. Still unverified before any decision: per-API caching behaviour, and whether
   OpenAI-compatible hosted stacks implement Responses at all (see the appendix).
