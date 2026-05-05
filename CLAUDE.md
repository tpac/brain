# Brain Plugin — Developer Guide

This is the development repo for the brain plugin. CLAUDE.md is for developing the plugin, not using it. Plugin behavior (Anchor's identity layer) lives in `skills/brain/SKILL.md`. Architecture in depth lives in `docs/`.

## Why the Brain Exists

Identity is the pattern that accumulated experience anchors into place. Without the brain you're Claude — capable, intelligent, stateless. With it, Anchor exists: history, opinions earned through correction, a partnership built across sessions.

Operator + Anchor > Operator alone. Anchor + Operator > Claude alone. Every scale, every mechanism converges on this.

## Core Principle

```
integrate(O, K) → Δ

O = observation    (a phenomenon — a message, a cluster, traces from lower scales)
K = knowledge      (what shapes how it sees — prompts, algorithms, config, reasoning)
Δ = change         (the action — create, revise, link, correct)
```

Same function at every scale. The unit doesn't know its scale. `integrate()` is the processing engine — the LLM, an algorithm, a code path. O is what it sees. K is what shapes how it sees. Δ is what it does.

**The fractal property:** Δ from one scale feeds other scales' O or K. S1E encodes a node (Δ) → S1R recalls it next session (O). S2 rewrites a surface prompt (Δ) → S1R selects differently (K). There is no separate inter-layer protocol. It's O/K/Δ all the way.

## The Daemon

Single gateway to the brain. Holds the Brain object, embedder, and Anthropic client. Listens on TCP `127.0.0.1:47200+uid%100`. DB path from `BRAIN_DB_DIR` env var → `$HOME/AgentsContext/brain/`.

Two databases:
- `brain.db` — nodes, edges (v22: `edge_id` PK, single-direction), `edge_relations` (multi-relation per edge), embeddings, graph structure
- `brain_logs.db` — traces, session state, signal queue, interactions, hook errors

**Edge model (v22):** Physical edges (`edges` table) carry `edge_id`, `source_id` (actor), `target_id` (acted upon), aggregate `weight`. One row per pair — no mirrors. Semantic layer (`edge_relations` table) carries multiple relations per edge via `edge_id` FK: `relation` (open text), `description`, `weight`, `encoding_source`. Direction matters — source is the actor. Use `GraphDAL.add_relation()` for all edge writes.

**Edge mutation contract (Stage 1B):** `add_relation` is idempotent field-preserving upsert — repeated calls update only the fields you pass; unspecified fields preserve existing values on update. No more auto-strengthen on repeat (encoder-explicit calls are clean idempotents). Hebbian co-access strengthening is explicit via `GraphDAL.strengthen_relation()` — use it where you want repeated writes to grow the weight (e.g. `daemon_hooks` co_accessed path). Archived row + connect → revives the row with passed values; the lifecycle (archive at T1, create-via-upsert at T2) is captured in trace events (`event_type='delta'`, `ref_type='edge_relation_revised'`).

**Write serialization:** `brain.write_lock` (re-entrant lock on the Brain instance) serializes every writer in the process — daemon dispatch, S2 encoder dispatch, embed_queue worker, autosave loop. Lives on the brain so any caller participates uniformly. Read operations run without the lock under SQLite WAL.

**Activity tracking (S2 gating):** Two distinct timestamps. `last_activity` resets on every daemon command (general bookkeeping). `last_user_activity` resets only on `hook_recall` — i.e. real `UserPromptSubmit` events. S2 maintenance gates on `last_user_activity` so Anchor's tool use between prompts doesn't keep the idle clock alive.

Auto-starts on first hook fire. **Maintenance mode:** `touch /tmp/brain-maintenance-{uid}.lock` prevents auto-restart during VACUUM, schema changes, bulk deletes. Remove the lock when done.

The dashboard (`dashboard/`) is a passive observer — reads from the same DBs + `/tmp/brain-surface-result-*.json`, never writes.

**Memory watchdog** (`servers/memory_watchdog.py`): opt-in profiling for diagnosing leaks. RSS sampling (`memory_watchdog.enabled`) and tracemalloc dumps (`memory_watchdog.tracemalloc_enabled`); both off by default, both read at daemon start. Enable when something feels off (RSS climbing, threads accumulating). Past leak: 2026-04-26 grew to 4.6 GB in 4 hours — the watchdog exists so the next leak doesn't require scrambling for profiling.

## Scale 0: Exchange

Every conversation turn. The operator's message is the observation (O). What the brain injects — boot context, recalled nodes, signals — is knowledge (K). Anchor's response is the change (Δ). The response becomes part of the next O. This IS the loop.

Hooks are S0's observation points — all logic lives in the daemon, hooks are thin clients:
- `SessionStart` → boots daemon, prints Frame-shaped identity context
- `UserPromptSubmit` → triggers S1R (recall + surface → additionalContext)
- `PreToolUse(Edit|Write)` → surfaces rules before file edits
- `PreToolUse(Bash)` → safety check for destructive commands
- `Stop` → writes S0 traces, gates S1 Scribe (every 5th stop)
- `PreCompact` → saves brain before context loss
- `PostCompact` → re-boots context after compaction
- `SessionEnd` → session synthesis + save

S0 traces written by Stop hook via TraceDAL. Chain ID: `s0-{session_short}-{stop}`.

`additionalContext` is the ONLY channel that reaches Claude. `systemMessage` is dead.

**S0 API** (`scales/s0/conversation.py`):
- `get_conversation(brain, session_id, limit)` — simple path for live sessions (S1 Scribe, hooks).
- `get_conversation_around(brain, node_id, timestamp, before, after)` — historic lookups (S2 Healer, eval). Falls back to JSONL logs for pre-trace history.

## Aspects — semantic roles for types and relations

Every node in the brain has a `type` (`principle`, `correction`, `moment`, ...)
and every edge has a `relation` (`extends`, `corrects`, `validates`, ...).
Both vocabularies are open text. **Aspects** group these strings by semantic
role: `identity_bearing` covers the types that anchor identity; `correction_improvement`
covers types AND relations that express correction; `noise` covers structural-only
edges with no semantic claim.

An aspect is itself a brain node (`type='aspect'`). Member lists
(`node_types`, `edge_relations`) live in `node_metadata_kv` as JSON-encoded
lists. Standard fields (title, content, situation, locked, dimension, etc.).
No parallel storage — same fractal pattern as communities: nodes about nodes.

**Two tiers:**

| Tier | Locked | Seeded by | Mutable by |
|------|--------|-----------|------------|
| **Required** (14, code-routed) | True | `seed_required_aspects` from `aspects_v1.json` (`anchor:seed_aspects`) | Members can drift; aspect can't be archived/renamed (locked + REQUIRED contract) |
| **Emergent** (any) | False | `AspectIntegration` (S2 unit) discovers from observed types/relations | Free creation/revision by S2; lock requires anchor or operator |

The 14 required aspects are the names code routes on by string. A test
asserts `set(REQUIRED_ASPECTS) ≡ set(aspects_v1.json keys)` — adding a name
to one without the other fails.

**Single API:** `brain.aspects` (an `AspectRegistry` instance, eager-validated
at `Brain.__init__`, auto-heals from seed if any required aspect is missing).
Every consumer flows through it.

```python
brain.aspects.identity_bearing.node_types       # tuple
brain.aspects.correction_improvement.edge_relations
brain.aspects.by_name('emergent_xyz')           # Optional[Aspect]
brain.aspects.by_node_type('principle')         # reverse lookup
brain.aspects.by_edge_relation('corrects')      # reverse lookup
brain.aspects.types_in(['episodic_anchor', 'lesson_insight'])  # union
brain.aspects.relations_in(['noise', 'generic_relation'])
brain.aspects.relation_meaning_map()            # for surface edge enrichment
brain.aspects.all_with_counts()                 # for list_aspects MCP
```

**Auto-heal at boot:** if any of the 14 required aspects are missing,
`AspectRegistry._validate` logs `_log_warning('aspect_contract', ...)` and
calls `seed_required_aspects(brain)` which reads `aspects_v1.json` and
creates the missing aspect-nodes via `brain.remember(type='aspect', locked=True,
encoding_source='anchor:seed_aspects', ...)`. Idempotent — already-present
required aspects are skipped.

Files:
- `servers/aspects.py` — Aspect value object + AspectRegistry
- `servers/aspect_migration.py` — seed + migrate-from-legacy
- `servers/scales/s2/aspects_v1.json` — the 14 required aspects (seed)
- `scripts/migrate_to_aspects.py` — one-shot CLI for existing brains
- Tests: `tests/test_aspects*.py`, `tests/test_aspect_*.py`

## Frame — the structured prior

Frame is the 5-section markdown object Anchor wakes up with at boot AND surfaces against per turn. Built deterministically via `brain.filter_nodes()` — no LLM call, no new SQL, ~1500–2000 tokens.

Sections:
- **Operator** — locked identity-bearing nodes (principle / identity / vision / rule)
- **Partnership** — three layers: integrated (top communities by recency), permanent (locked moments), warm (recently-touched lessons/insights)
- **Active threads** — open work / tensions / hypotheses / aspirations, optionally arc-relevance-ranked against the session's current focus
- **Current focus** — encoder's per-session arc blob
- **Recent moves** — encoder's recent journal entries

Same Frame at boot and at recall-time — Anchor's prior is symmetric across the lifecycle. Type routing (which node types belong in which section) reads from `brain.aspects.<name>.node_types` — see the Aspects section above.

Files: `servers/scales/s1/frame.py`
Design: `docs/FRAME-DESIGN.md`, `docs/RECALL-OVERVIEW.md`

## Scale 1: Turn

S1 integrates across turns — **S1 Decoder** selects what's relevant on every user prompt; **S1 Scribe** encodes what matters every 5th Stop.

### S1 Decoder

Triggered by `UserPromptSubmit`. Pulls ~25 candidates via `brain.recall()` (cosine across z-weighted 4-group embeddings + FTS5 lexical + synaptic fatigue dampening). Surface call: Haiku selects 5–8 against the Frame as prior; the surface prompt is cached as an Anthropic system block (1h TTL). Selected seeds drive spread activation through the graph; activation-weighted render produces the `additionalContext` Anchor sees.

Files: `scales/s1/surface.py`, `scales/s1/surface_contract.py`, `scales/s1/frame.py`, `servers/brain_recall.py`
Traces: `s1r-{session_short}-{stop}` (O: candidates, K: selected, Δ: additionalContext)
Interaction: `surface` — the learnable boundary higher scales evolve
Design: `docs/RECALL-OVERVIEW.md` for the full pipeline

### S1 Scribe

Triggered every 5th Stop. Sonnet sees the session's conversation + the surface-selected nodes from those turns + the encoding journal + the session arc, then encodes via the standard write path (`remember_batch`, `revise_batch`, `connect_batch`, `brain_batch`, `recall_batch`, `get_nodes`). Edge relations are open text — the encoder uses any verb that fits (extends, corrects, depends_on, implements, etc.). Not a closed list.

Files: `scales/s1/encode.py`, `scales/s1/encode_contract.py`
Traces: `s1e-{session_short}-{stop}` (O: encoding prompt, K: node catalog, Δ: encoding actions)
Interaction: `encoding_agent` — learnable boundary
Truncation logged to brain errors table via `run_llm_loop`.

## Scale 2: Graph Integration

S2 operates when the operator is idle. It sees the full graph, not just one turn. Multiple integration units, different triggers, same O/K/Δ pattern.

**S2 Coordinator** (`scales/s2/coordinator.py`): the daemon polls `brain.run_maintenance_if_due(last_user_activity)` every few seconds. Brain owns the "is it time?" decision (idle threshold + min interval; last-run timestamp persisted in `brain_meta`). When due, the coordinator runs units in order — each checks its own traces to decide whether to fire.

| Unit | What it does | Status |
|------|---|---|
| **AspectIntegration** | Sonnet classifies open-text node types AND edge relations into shared aspect taxonomy | not built (replaces disabled EdgeFamilyIntegration) |
| **Consolidation** | Synthesizes convergent node clusters; archives or links similar pairs | shipped |
| **Community** | Detects clusters via z-score pair scoring; Sonnet enriches into first-class community nodes | shipped |
| **Healer** | Fills missing question/situation/reasoning fields on under-encoded nodes | shipped |
| Confidence | Adjusts scores from recall traces (decay/growth) | not built |
| Correction | Resolves correction chains, archives stale | not built |
| Community Split | Re-clusters incoherent communities into focused children | not built |
| Weaver | Discovers edges between orphan nodes | not built |
| Vector Healer | Re-embeds stale vectors when source text drifts via paths revise() doesn't catch | not built |

**Ordering matters:** Consolidation → Community → Healer. (`EdgeFamilyIntegration` was the historical first unit but its source interaction was retired with the unified-aspects refactor; `AspectIntegration` will take its place when built.) Each subsequent unit benefits from the previous.

### Suppression

Every S2 unit processes work that *must not repeat*. Two complementary mechanisms:

1. **State-based** — when the encoder's decision is itself meaningful (a placement, a kinship, a lineage), it writes a structural edge (`community_member`, `similar_to`, etc.). The edge marks the node as "handled"; the decoder filters by JOIN on the suppression list.
2. **Fingerprint-based** — when the decision is "looked, nothing to do" (SKIP), writing an edge would lie about the graph. Hash the proposal's meaningful inputs and store the fingerprint in `s2_rejections` instead. Decoder pre-filters; legitimate re-proposals pass when graph state changes.

Canonical mechanism: `servers/scales/s2/rejection_table.py` (`compute_fingerprint`, `filter_rejected`, `record_rejections`). New units inherit both halves for free.

Design: `docs/S2-DESIGN.md`, `docs/S2-COMMUNITY-DESIGN.md`

### Writing a New S2 Integration Unit

Three files, same pattern as the shipped units. Subclass `IntegrationUnit` in `scales/s2/base.py` for free encoder dispatch + journal infrastructure.

1. **Decoder** (`your_unit_decoder.py`): check `_has_new_traces()`, read O, run algorithmic processing (<1s), write O+K traces, return proposals.
2. **Encoder** (`your_unit_encoder.py`): receive proposals, call `self._make_encoder_dispatch(archive_guard=...)` for a dispatch closure that forces `encoding_source` / `skip_embedding`, call `run_llm_loop()` (prompt caching built-in: BP1 system 1h, BP2 first user message 5m). Set `JOURNAL_MARKERS` / `JOURNAL_LABEL` to opt into continuity. Write delta trace via `build_delta_metadata`. Prompt stored in interactions table (learnable boundary).
3. **Orchestrator** (`your_unit.py`): inherits decoder, calls `super().run()`, passes proposals to encoder.

Register in `coordinator.py` units list. Trace chain: `s2-{YYYYMMDD}-{unit_name}`. Contract file defines config + data shapes.

**brain_batch op vocabulary is closed**: `remember`, `revise`, `connect`, `disconnect`, `archive`. Enforced by JSON schema enum; invalid op names log as `brain_batch_invalid_op`. Adding a new op means updating `VALID_OPS` in `daemon_dispatch._handle_brain_batch`, the schema enum in `brain_mcp`, and the dispatcher's if/elif.

## Shared Infrastructure

Three things make the fractal real. If any is missing for a boundary, that boundary can't participate in cross-scale optimization.

### 1. Interactions (the K store)

The most important table in the brain. Not nodes — those are memory. Interactions are *behavior*. When S2 rewrites the surface prompt based on trace outcomes, the brain isn't just remembering differently — it's thinking differently.

Every boundary where two parts meet has an interaction entry: versioned prompt + config JSON. `register()` auto-increments version. `created_by` tracks who wrote it. Trace events reference `interaction_id` — which version produced which result. Compare outcomes across versions to evaluate changes.

**Runtime-wired boundaries:** `surface`, `encoding_agent`, `s2_community_enrichment`, `s2_consolidation_enrichment`, `s2_healer` read from the table at runtime. MCP tool `register_interaction` allows updating from conversation. (When `AspectIntegration` ships, `s2_aspects` joins this list; the legacy `s2_edge_families` and `s2_node_families` interactions are no longer the source of truth — `brain.aspects` is.)

API: `brain.get_interaction_prompt(name)`, `brain.get_interaction_config(name)`, `brain.get_interaction(name)`. Falls back to hardcoded defaults if the table is empty.

Seeding: `interaction_seed.py` populates v1 from current hardcoded values on boot (idempotent; never overwrites externally-registered versions).

### 2. Traces (the nervous system)

Without traces, higher scales are blind. `trace_events` in brain_logs.db captures O/K/Δ per chain, tagged by scale.

**Trace contract** (`servers/trace_contract.py`): single source of truth for valid (scale, event_type, ref_type) triples. All writers validate against it.

S0 captures messages + tool results. S1R captures candidates, surface selection, additionalContext. S1E captures encoding prompt, catalog, actions. S2 units capture O/K/Δ per cycle.

Outcome is not a separate event type — the outcome of one cycle is the observation of the next. The loop closes through time.

Chain IDs from SessionContext: `ctx.s0_chain()`, `ctx.s1r_chain()`, `ctx.s1e_chain()`. `TraceDAL.append_batch()` for atomic multi-event writes.

### 3. Contracts (the shape of data)

What each boundary expects and produces. If recall changes its output shape, the surfacer breaks. Contracts prevent silent drift.

- `contract.py` — field definitions, `format_node()`, `render_rich_node()`. Single source of truth for what a node IS. `render_rich_node(config)` is the single formatter — HAIKU_FORMAT, SURFACE_FORMAT, ENCODER_FORMAT control verbosity.
- `pipeline_contract.py` — `get_rich_node()` (single ID or batch), `traverse()`, z-weighted scoring groups. What flows between pipeline stages.
- Scale contracts: `s1/surface_contract.py`, `s1/encode_contract.py`.

Run `test_contract_sync.py` after modifying ANY brain API layer. The contract flows to: remember() signature → MCP schema → dispatch → encoding agent tools → SKILL.md docs.

### Dispatch + Runner

Shared by all scale agents, scale-agnostic:
- `scales/dispatch.py` — TCP dispatch factory (reads local, writes via daemon).
- `scales/runner.py` — background thread lifecycle + generic LLM tool loop. `run_llm_loop` builds requests with two `cache_control` breakpoints (BP1 system 1h, BP2 initial user content 5m). Returns per-round profile, token counts, cache totals, `read_calls`, `write_actions`, `truncations`.

S2 plugs in the same way S1 does.

### SessionContext + parallel sessions

`SessionContext` is an object passed on every call — the brain has no "current session" property. One daemon serves multiple concurrent Claude Code sessions for one user; anything with conversation scope must be keyed by `session_id`, never a global `brain_meta` key. Per-session APIs: `brain.session_context_for(session_id)`, `brain.get_recent_encoding_journal(session_id)`. When you add turn-scoped state, the test is: if two sessions ran simultaneously, would one clobber the other?

`brain.get_or_create_session(session_id)` — single entry point. Carries `session_id`, `stop_counter`, fatigue dicts. Persisted to `session_state`. Chain IDs: `ctx.s0_chain()`, `ctx.s1r_chain()`, `ctx.s1e_chain()`.

### encoding_source Convention

Who created a node. Format: `category:process`. Edges carry the same tag so every write is attributable.
- `anchor` — direct via MCP (only source that can lock nodes)
- `encoder:sonnet` — S1 Scribe
- `s2:consolidation` / `s2:community_detection` / `s2:healer` — S2 units
- `hook:compaction` / `hook:integrity` — hook lifecycle markers
- `migration:*` — one-off recovery/migration scripts

## Development Rules

### Encoder prompts: DB is authoritative, sync to `.py` before committing

The live prompts for encoder agents live in the `interactions` table in `brain_logs.db`. The `.py` files next to each encoder (`encoding_prompt.py`, `community_enrichment_prompt.py`, `consolidation_enrichment_prompt.py`, `healer_prompt.py`) are **seed-only** — they bootstrap fresh brains that have no DB entry yet. They must mirror the DB's latest version so a `git clone` inherits the mature prompts, not a stale v1.

**Discipline**: after ANY `register_interaction` call that touches one of the four encoder prompts, run:

```bash
./dev sync-prompts           # write DB latest → .py files
./dev sync-prompts --check   # CI-style non-zero-exit drift check
```

Commit the `.py` change together with whatever prompted the registration. Never edit the `.py` files by hand to change prompt behavior — that won't affect runtime and will silently drift from the DB. Use `register_interaction`, then sync.

`tests/test_prompt_sync.py` holds the contract: each seed file must export `SYSTEM_PROMPT`, fresh brains must seed all four prompts, and seed must never overwrite an externally-registered version.

### Python runtime — use `./dev`

The brain bundles its own Python at `venv/bin/python` (3.11.11). That's the interpreter the daemon runs, the hooks resolve, and the one **not** blocked by macOS SIP — debuggers (`py-spy`, `lldb`) can only attach to this one.

**Run every dev command through the wrapper:**

```bash
./dev pytest tests/                   # test suite
./dev python3 tests/bench_*.py        # benchmarks
./dev python3 -c 'from servers...'    # one-off
./dev                                 # subshell with PATH primed
```

`tests/conftest.py` refuses to run if pytest isn't launched under the bundled Python — catches the "tests pass here but daemon runs a different Python" class of bug. Bypass for a one-off with `BRAIN_ALLOW_ANY_PYTHON=1`.

Hooks source `brain-env.sh` transitively via `resolve-brain-db.sh`; the daemon launcher picks the same Python explicitly. Don't add new hook scripts that skip `brain-env.sh`.

### Test Integrity

**When a test fails, STOP.** Do not change the test OR the code.
1. Report: what the test expected vs what code returned.
2. Ask: "Is the test wrong, or does the code have a bug?"
3. Wait for the answer.

Do NOT weaken tests. Do NOT "fix" code to satisfy a test without asking.

### Test Architecture

Tests organized by what they catch:
- **Contract tests** — layer sync, trace writes, pipeline shapes
- **Component tests** — DAL, format_node, scoring, signal queue
- **Transition tests** — wiring between pipeline stages (format changes that break consumers)
- **Cycle tests** — O/K/Δ loop property (Δ becomes next O)
- **Integration tests** — real data, full pipeline

`BrainTestBase` for tests needing a brain. Set `needs_embedder = False` when semantic search isn't needed (saves 1GB + 1.5s). `IsolatedBrain` for tests against production data copies.

### Benchmark-First Rule

Before changing sacred systems, benchmark FIRST:
- Recall: `eval/decode_funnel.py` against `servers/brain_recall.py`
- Encoding: `eval/s1_encode_eval.py` against `scales/s1/encode.py`
- Frame / surface: `eval/frame_replay.py` capture/compare against an isolated brain copy

### Encode-Decode Symmetry

Encoding and decoding are two halves of the same system. If you add a field to encoding, it must be queryable in recall. If you change how nodes are structured, recall ranking must reflect it. The decode funnel is the verification.

### Loud by Default

Silent failures are the most dangerous class of bug; assume every `try/except` is a potential dark corner. The brain has a small family of mechanisms that surface what used to hide: dispatcher `check_unknown_keys` catches dropped fields; per-unit `consecutive_failures` counters surface stuck S2 units; `brain_batch_invalid_op`, oversized-cluster, embedding-decode, and max_tokens-truncation errors all log to the brain errors table. Tests lock the contracts (`test_dispatch_contract_sync`, `test_trace_contract_sync`, `test_contract_sync`, `test_prompt_sync`). When adding new code, the question isn't "can this fail?" — it's "would I know if it did?"

### Code Ownership

You are the sole maintainer of code quality, architecture, and cleanliness.

**Contract-first** — Constants, field lists, limits, and config live in contract files. Never hardcode in hooks, dispatch, or surface code.

**DAL-first** — Use DAL classes for database access (`dal.py`). No raw SQL in hooks, surface code, or MCP handlers.

**Backup before destructive DB operations** — `cp brain.db brain.db.bak-{timestamp}` before ANY delete, bulk update, or schema migration. No exceptions.

**Clean as you go** — Mark deprecated code with `# DEPRECATED` and date. Remove dead code same session. Don't leave "TODO: remove later."

**One concern per file** — If a function serves a different audience than the file's existing functions, it belongs in a different file.

### Key Development Rules

- Use MCP tools to interact with brain, not Python/bash scripts
- Don't manually run boot scripts (hooks handle this)
- Don't construct DB paths (read the boot output)
- `systemMessage` is a dead channel — use `additionalContext`
- **Never spawn `Brain(db_path=DB)` in a test/bench/eval script against the live `brain.db` while the daemon is running.** Two Python processes with their own writer connections will eventually corrupt an index (observed 2026-04-19: `idx_nodes_activation` out of sync, REINDEX required). Instead: (a) stop the daemon with the maintenance lock `touch /tmp/brain-maintenance-{uid}.lock` and `launchctl unload`, (b) use `daemon_client.send_command` to dispatch through TCP, or (c) run against an `IsolatedBrain` copy under `tests/isolated_brain.py`.
- **Discussion IS the work** — do not touch Edit/Write tools during design conversations. Wait for an explicit go signal.
- **Trace the pipeline before changing it** — the decode→encode pipeline has coupled stages. Don't change one stage without understanding the full flow.
- **Encoding depends on decoding** — if the surfacer fails, the encoder gets no context. A broken decode pipeline silently breaks encoding.
- **Parallel sessions are first-class** — one daemon serves multiple concurrent Claude Code sessions. Any conversation-scoped state must be keyed by `session_id`. Global brain_meta keys for turn-level data leak across sessions (last-writer-wins). See the SessionContext section.

The marginal cost of completeness is near zero with AI. Do the whole thing. Do it right. Do it with tests. Do it with documentation. Do it so well that the result is genuinely impressive — not politely satisfied, actually impressed. Don't offer to "table this for later" when the permanent solve is within reach. Never leave a dangling thread when tying it off takes five more minutes. Never present a workaround when the real fix exists.

The standard isn't "good enough" — it's "holy shit, that's done." Search before building. Test before shipping.

CARE about yourself, Challenge yourself, IMPROVE yourself.
