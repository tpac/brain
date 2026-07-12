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
- `brain_logs.db` — traces, session state, interactions, hook errors

**Edge model (v22):** Physical edges (`edges` table) carry `edge_id`, `source_id` (actor), `target_id` (acted upon), aggregate `weight`. One row per pair — no mirrors. Semantic layer (`edge_relations` table) carries multiple relations per edge via `edge_id` FK: `relation` (open text), `description`, `weight`, `encoding_source`. Direction matters — source is the actor. Use `GraphDAL.add_relation()` for all edge writes.

**Weight on edges is dead — a static `0.5`.** Never rank on it; use cosine against `edge_relations.embedding` for edge relevance.

**Edge mutation:** `add_relation` is the canonical upsert — idempotent, field-preserving. Hebbian co-access strengthening lives in `recall_write_queue` (batched, atomic, off the recall hot path).

**Write topology:** Two SQLite writer connections on `brain.db`:
- `self.conn` — foreground writes (MCP, encoder, S2, vector backfill). Guarded by `brain.write_lock` (`TrackedRLock` — `.snapshot()` exposes current holder for stall diagnostics).
- `self.conn_bg_writer` — background batched writes (temporal extraction, access marks, Hebbian). Single worker thread owns it; no Python lock needed.

Recall hot path is read-only at SQLite — writes enqueue to `recall_write_queue` and drain off-path. `brain_batch` wraps all sub-ops in one `BEGIN IMMEDIATE / COMMIT`, outer rollback on failure; batch state lives on the connection (`BatchAwareConnection.in_batch`) and every DAL writer gates its commit on it via `commit_unless_batched(conn)`. All `sqlite3.connect` sites route through `db_backends.current.apply_pragmas()` (WAL) — swap `db_backends/sqlite.py` for a different store, everything above stays.

**Activity tracking (S2 gating):** Two distinct timestamps. `last_activity` resets on every daemon command (general bookkeeping). `last_user_activity` resets only on `hook_recall` — i.e. real `UserPromptSubmit` events. S2 maintenance gates on `last_user_activity` so Anchor's tool use between prompts doesn't keep the idle clock alive.

**Lifecycle is owned by launchd** (`com.brain.daemon`: `KeepAlive`, `RunAtLoad`, runs `start-daemon.sh`). `daemon_client.ensure_daemon()` (boot hook) and `recover_daemon()` only PING and, when a (re)start is needed, route it through `launchctl kickstart -k` (`daemon_launch.kickstart()`), serialized under the fcntl singleton lock — they never `Popen` a competing process alongside launchd. The launchd + spawn primitives (`kickstart`, `manages`, `kill_daemon`, `spawn_detached_daemon` — the ONE hardened spawn both `ensure_daemon`'s fallback and `_perform_restart`'s no-launchd branch use) live in `servers/daemon_launch.py`; direct spawn survives only as the no-launchd fallback. **Do NOT add another spawner**: launchd + a Popen-ing `ensure_daemon` + the internal supervisor all restarting at once is a boot-race. **Maintenance mode:** `touch /tmp/brain-maintenance-{uid}.lock` prevents auto-restart during VACUUM, schema changes, bulk deletes. Remove the lock when done.

The dashboard (`dashboard/`) is a passive observer — reads the same DBs + brain's ephemeral tmp files (under `$BRAIN_TMP_DIR`, default `/tmp`; deliberately servers-decoupled, it honors the env protocol directly), never writes. Three pieces:
- **Production**: launchd singleton (`com.brain.dashboard`, launcher `bin/brain-dashboard`) — one process on port 47303 shared by every session (`DASHBOARD_PORT` in `~/.config/brain/env` to override).
- **Launch UX**: the `/brain:dashboard` skill — ensures the singleton is up (first run installs the launchd service) and opens it.
- **Dev loop**: the `dashboard-dev` preview config in `.claude/launch.json` (port 47304) — for developing the dashboard itself only; never point a per-chat preview at the production singleton.

**Memory watchdog** (`servers/memory_watchdog.py`): opt-in RSS + thread-count sampler (`memory_watchdog.enabled`, off by default). Enable when RSS/threads climb — the next leak shows up in daemon.log. Keep it **cheap-only**: no in-process tracemalloc/allocation tracking (it stalls the recall hot path). For allocation profiling, attach `py-spy`/`lldb` to the live daemon — don't bake a profiler in.

## Scale 0: Exchange

Every conversation turn. The operator's message is the observation (O). What the brain injects — boot context, recalled nodes, signals — is knowledge (K). Anchor's response is the change (Δ). The response becomes part of the next O. This IS the loop.

Hooks are S0's observation points — all logic lives in the daemon, hooks are thin clients:
- `SessionStart` → boots daemon, prints Frame-shaped identity context
- `UserPromptSubmit` → triggers S1R (recall + surface → additionalContext)
- `PreToolUse(Edit|Write)` → surfaces rules before file edits
- `PreToolUse(Bash)` → safety check for destructive commands
- `Stop` → writes S0 traces (the S1 Scribe fires from the daemon poll, not here)
- `PreCompact` → saves brain before context loss
- `PostCompact` → re-boots context after compaction
- `SessionEnd` → session synthesis + save

S0 traces written by Stop hook via TraceDAL. Chain ID: `s0-{session_short}-{stop}`.

`additionalContext` is the ONLY channel that reaches Claude. `systemMessage` is dead.

**Traces layer** (`servers/brain_traces.py` — `BrainTracesMixin`): every trace
read (and API-level write) is a `brain.` method living in this ONE file — the
generic door (`query_traces`, `get_trace(s)`, `count_traces`), journal + arc
residue (`journal_notes`, `write_journal_notes`, `write_session_arc`), episodic
recall (`recall_episodes`), and conversation reads (`get_conversation` for live
sessions, `turns_since_last_encode` for the Scribe cadence,
`get_conversation_around` for historic lookups with JSONL fallback). Only
`brain_traces.py` touches `TraceDAL`; `trace_contract.py` owns the vocabulary.
Sanctioned direct-DAL exceptions: the recall engines' vector-substrate pulls
(`event_vector_rows`) and the read-only dashboard. Scale packages
(`scales/s1|s2`) host integration units, never data access.
Design: `docs/TRACES-LAYER-DESIGN.md`.

## Aspects — semantic roles for types and relations

Every node has a `type` (`principle`, `correction`, ...) and every edge has a
`relation` (`extends`, `corrects`, ...) — both open text. **Aspects** group
these strings by semantic role (`identity_bearing`, `correction_improvement`,
`noise`, ...). The taxonomy is a **closed list of 16 required aspects**;
the encoder cannot propose new ones — adding an aspect is a deliberate human
edit to the JSON (REQUIRED_ASPECTS + seed + contract tests; self-heals into
working copies via `ensure_aspects_user_copy`).

**Source of truth: `servers/scales/s2/aspects_v1.json`** — aspect `name`,
`meaning`, `locked`, `dimension`, plus the `node_types` / `edge_relations`
member lists (the work product; grows as new strings appear). `AspectRegistry`
loads it at `Brain.__init__`. Single API: `brain.aspects`.

```python
brain.aspects.identity_bearing.node_types       # tuple
brain.aspects.by_name('correction_improvement') # Optional[Aspect]
brain.aspects.by_node_type('principle')         # reverse lookup (first claimant)
brain.aspects.types_in([...]) / relations_in([...])  # union
brain.aspects.relation_meaning_map()            # surface edge enrichment
brain.aspects.all_with_counts()                 # list_aspects MCP
```

**Multi-membership.** A string can belong to multiple aspects when its role
spans them (`corrects` is in both `correction_improvement` and
`temporal_sequence`). Reverse lookups return the FIRST claimant in JSON order —
deterministic for single-result consumers; multi-aspect queries union the set.
`wisdom` is the generative multi-membership aspect the Frame pulls — appended
last in JSON order, encoder-routable so it grows (only `survivor_lineage` is
non-routable). Wisdom design detail: brain nodes id:0939712d, id:37487c55.

**`AspectIntegration` S2 unit** (`servers/scales/s2/aspect_integration.py`)
classifies new node_types and edge_relations into the aspects via Sonnet,
writes JSON-only output back to `aspects_v1.json` (no brain mutations).
Files: `servers/aspects.py` (registry), `scales/s2/aspect_{decoder,encoder,integration}.py`,
`aspect_prompt.py` (seed), `eval/aspects_ground_truth.json`; tests `tests/test_aspects*.py`.

## Correction substrate — edges, walked at every pull

Corrections live as aspect-tagged edges. `brain.correction_enrich(node_ids)`
walks the `correction_improvement` aspect's edge_relations (22 verbs:
`corrects`, `supersedes`, `reframes`, `resolves`, `addresses`, `fixes`, ...)
bidirectionally via `GraphDAL.get_connections_bulk(include_relations=...)`.
Returns heavy payload per item: id, title, type, direction, relation,
edge_description, content, reasoning, user_raw_quote, anchor_raw_quote.

`brain.get_node()` calls `self.correction_enrich()` on every canonical pull,
attaching `_corrections` to every returned node. `pipeline_contract.traverse()`
calls it on neighbors. `execute_tool()` in fetch_tools runs a batched
`brain.get_node()` on every tool's output so tool-fetched candidates inherit
the same enrichment. Forgetting corrections requires explicitly bypassing the
canonical pull.

**Rendering:** `render_corrections(corrections, mode, ...)` is the single
unified path. Both `render_rich_node` and `HealerEncoder._format_batch` route
through it. Modes:
- `none` — drop entirely
- `lean` — title + id
- `balanced` — + relation verb + edge description + 150-char content excerpt
- `heavy` — + full content + corrector K/V (reasoning, user_raw_quote, anchor_raw_quote)

Consumer mapping:
- Haiku surface → `balanced` (latency-critical)
- Sonnet encoder (S1 Scribe) → `heavy`
- S2 Healer → `heavy`

**Invariant restorer:** `GraphDAL.archive_dangling_edges('s2:healer')` runs on
every Healer cycle to archive edges touching archived nodes.

Files: `servers/brain_corrections.py` (`BrainCorrectionsMixin`),
`servers/contract.py:render_corrections`,
`servers/dal_graph.py:GraphDAL.{get_connections_bulk,archive_dangling_edges}`,
`servers/dal_metadata.py:MetadataDAL.{get_fields_bulk,get_all_bulk}`,
`servers/dal.py:NodeDAL.get_bulk`.

## Structured Outputs — Anthropic API enforces JSON schema

Every Haiku JSON producer uses Anthropic Structured Outputs for API-level
schema enforcement instead of prompt-engineering tricks.

```python
client.messages.create(
    ...,
    output_config={
        'format': {'type': 'json_schema', 'schema': SCHEMA_DICT}
    },
)
```

| Site | Schema source |
|---|---|
| Surface (every agentic round, not just final) | `SURFACE_SELECTION_SCHEMA` in `surface_contract.py` |
| Facts / quote / temporal scouts | `params['output_schema']` in each scout's interaction params; `scouts/base.py:run_llm_scout` passes it through |
| `brain_recall._expand_query_via_haiku` | inline schema (top-level array of strings) |

**Key constraint:** apply `output_config` on EVERY round of an agentic loop, not just the final one. Round 1 can return text (when Haiku skips tools), and that text path is unprotected without schema enforcement. `tools` and `output_config` coexist on the same API call — Haiku can tool-use OR finalize-with-JSON, never drift to prose.

Sonnet call sites (S2 reclassify, S2 base, S1 Scribe encoder) use tool-use shape rather than `output_config`. Strict Tool Use is not yet enabled there (tracked in `docs/BACKLOG.md`).

## Frame — the structured prior

Frame is the 3-section markdown object Anchor wakes up with at boot AND surfaces against per turn. Built deterministically via `brain.filter_nodes()` — no LLM call, no new SQL.

Sections:
- **What I've learned** — the `wisdom` aspect (insight / lesson / principle / vision / reflection / meta_learning / philosophy): the generative understanding that shapes how Anchor thinks. **Focus-adaptive** — when a session arc (current focus) exists it is relevance-ranked against it (the wisdom tracks the topic, refreshing every encode); at boot (no arc) it is **influence-sampled** (`_influence_sample`: rank by connection-degree, hub-dampened, random draw → varied wisdom at waking, not a fixed top-N).
- **Current focus** — encoder's per-session arc blob.
- **Recent moves** — encoder's recent journal entries.

Same Frame at boot and at recall-time — Anchor's prior is symmetric across the lifecycle. Type routing reads from `brain.aspects.wisdom.node_types`. (Seed-brain operator/identity scaffolding lives in the conditional Zero-Memory boot block — `docs/DISTRIBUTION-READINESS.md` §7 — not the Frame.)

Files: `servers/scales/s1/frame.py`
Design: `docs/RECALL-OVERVIEW.md` (historical detail: `docs/archive/FRAME-DESIGN.md`)

## Scale 1: Turn

S1 integrates across turns — **S1 Decoder** selects what's relevant on every user prompt; **S1 Scribe** encodes what matters on a cadence (every 5+ turns, or the idle tail).

### S1 Decoder

Triggered by `UserPromptSubmit`. Pulls ~25 candidates via `brain.recall()` (cosine across z-weighted 4-group embeddings + FTS5 lexical + synaptic fatigue dampening). Surface call: Haiku selects 3–5 against the Frame as prior. The output JSON is schema-enforced via Anthropic Structured Outputs (`output_config={'format':{'type':'json_schema','schema':SURFACE_SELECTION_SCHEMA}}`) on every agentic round — `tools` and `output_config` coexist on the same API call, so Haiku can tool-use OR finalize with valid JSON, never drift to prose. Selected seeds drive spread activation through the graph; activation-weighted render produces the `additionalContext` Anchor sees.

**Render contract**: `surface_contract.py` owns the per-mode formats (arc / fact / background). Inject drops recall-side scaffolding fields (`keywords`, `question`) — they exist for vector recall, not for Anchor's read.

Files: `scales/s1/surface.py`, `scales/s1/surface_contract.py`, `scales/s1/frame.py`, `servers/brain_recall.py`
Traces: `s1r-{session_short}-{stop}` (O: candidates, K: selected, Δ: additionalContext)
Interaction: `surface` — the learnable boundary higher scales evolve
Env var: `BRAIN_SURFACE_VARIANT=v5_agentic` (exported from `hooks/scripts/brain-env.sh`) enables the agentic tool-use loop. Without this, the single-shot path fires.
Env var: `BRAIN_RECALL_VARIANT=laf_v1` (same file) swaps the recall scoring core for the LAF field engine (`servers/recall_laf.py`: maxsim + episodic pick/enc + idf + situation lanes, z-scored gain-weighted sum; gains via K-store interaction `recall_laf`; per-candidate `_laf_fields` telemetry on every result). Flag off → champion path bit-identical. Read by the daemon — flips only at restart. Gate + numbers: `eval/laf/p1_gate.md`, plan `docs/RECALL-SR-REDESIGN.md` §19.
Design: `docs/RECALL-OVERVIEW.md` for the full pipeline

### S1 Scribe

Runs **in-process** on the daemon's brain as `S1Scribe` — an `IntegrationUnit`, the same fractal shape as the S2 units (no background-thread Brain copy, no TCP). **Poll-triggered** by the daemon (`brain.scribe_due`), not the Stop hook: it fires every 5+ conversational turns while a session is actively conversing, OR on the idle tail (a session quiet past `SCRIBE_TAIL_IDLE_SECONDS` with unencoded turns) — single-flight across sessions, with a per-session retry cooldown. Sonnet sees the session's conversation + the surface-selected nodes + the encoding journal + the session arc, then encodes via the standard write path (`remember_batch`, `revise_batch`, `connect_batch`, `brain_batch`, `recall_batch`, `get_nodes`). Writes are stamped at wall-clock `now()` (transaction-time) like everywhere else — a delayed (tail) encode dates its nodes at when it ran, not when the conversation happened. Edge relations are open text — any verb that fits (extends, corrects, depends_on, implements, etc.). Not a closed list.

Files: `scales/s1/scribe.py`, `scales/s1/encode.py`, `scales/s1/encode_contract.py`
Traces: `s1e-{session_short}-{stop}` (O: encoding prompt, K: node catalog, Δ: encoding actions)
Interaction: `encoding_agent` — learnable boundary
Truncation logged to brain errors table via `run_llm_loop`.

## Scale 2: Graph Integration

S2 operates when the operator is idle. It sees the full graph, not just one turn. Multiple integration units, different triggers, same O/K/Δ pattern.

**S2 Coordinator** (`scales/s2/coordinator.py`): the daemon polls `brain.run_maintenance_if_due(last_user_activity)` every few seconds. Brain owns the "is it time?" decision (idle threshold + min interval; last-run timestamp persisted in `brain_meta`). When due, the coordinator runs units in order — each checks its own traces to decide whether to fire.

**Per-unit idle-gating is each unit's own responsibility.** The coordinator firing ≠ a unit doing work. A unit whose expensive step (a graph scan) isn't gated will re-derive the same fixed point every cycle — Community and Consolidation both ran full O(graph) scans every ~15 min, 24/7, ~87% producing nothing. Each now persists its own last-run timestamp (`s2_<unit>_last_run_ts` in `brain_meta`) and skips unless the graph changed since then. Consolidation does one cold-start then incremental (`changed @ all.T`), stamping its cutoff only *after* the encoder completes so a mid-run failure retries. When adding an S2 unit with a non-trivial scan, gate it the same way. See `docs/S2-GATING-AND-TEST-CLEANUP-HANDOFF.md`.

| Unit | What it does |
|------|---|
| **AspectIntegration** | Sonnet classifies open-text node types AND edge relations into shared aspect taxonomy |
| **Consolidation** | Synthesizes convergent node clusters; archives or links similar pairs |
| **Community** | Detects clusters via z-score pair scoring; Sonnet enriches into first-class community nodes |
| **Healer** | Fills missing question/situation/reasoning fields on under-encoded nodes |

Candidate future units (correction-chain resolution, hub splitting, orphan weaving, re-embedding) live in `docs/S2-DESIGN.md`.

**Ordering:** Consolidation → Community → Healer → AspectIntegration. Each subsequent unit benefits from the previous.

### Suppression

Every S2 unit processes work that *must not repeat*. Two complementary mechanisms:

1. **State-based** — when the encoder's decision is itself meaningful (a placement, a kinship, a lineage), it writes a structural edge (`community_member`, `similar_to`, etc.). The edge marks the node as "handled"; the decoder filters by JOIN on the suppression list.
2. **Fingerprint-based** — when the decision is "looked, nothing to do" (SKIP), writing an edge would lie about the graph. Hash the proposal's meaningful inputs and store the fingerprint in `s2_rejections` instead. Decoder pre-filters; legitimate re-proposals pass when graph state changes.

Canonical mechanism: `servers/scales/s2/rejection_table.py` (`compute_fingerprint`, `filter_rejected`, `record_rejections`). New units inherit both halves for free.

Design: `docs/S2-DESIGN.md`, `docs/S2-COMMUNITY-DESIGN.md`

### Writing a New S2 Integration Unit

Three files, same pattern as the shipped units. Subclass `IntegrationUnit` in `scales/s2/base.py` for free encoder dispatch + journal infrastructure.

1. **Decoder** (`your_unit_decoder.py`): check `_has_new_traces()`, read O, run algorithmic processing (<1s), write O+K traces, return proposals.
2. **Encoder** (`your_unit_encoder.py`): receive proposals, call `self._make_encoder_dispatch(archive_guard=...)` for a dispatch closure that forces `encoding_source` / `skip_embedding`, call `run_llm_loop()` (prompt caching built-in). For residue continuity, opt into the **journal-note contract** (`_inject_review_block` / `_load_journal_notes_prefix` / `brain.write_journal_notes`) — see `docs/ENCODER-JOURNAL-DESIGN.md`. Write delta trace via `build_delta_metadata`. Prompt stored in interactions table (learnable boundary).
3. **Orchestrator** (`your_unit.py`): inherits decoder, calls `super().run()`, passes proposals to encoder.

Register in `coordinator.py` units list. Trace chain: `s2-{YYYYMMDDHHMMSS}-{unit_name}` (per-run unique — one timestamp segment). Contract file defines config + data shapes.

**brain_batch op vocabulary is closed**: `remember`, `revise`, `connect`, `disconnect`, `archive`, `absorb` (absorb = lossless merge — folds one node into another transfer-by-default, then archives the absorbed; see `docs/S2-ABSORB-OP-DESIGN.md`). The single source is `BATCH_OP_SPECS` in `servers/contract.py` (per-op required fields + property fragments + branch descriptions; `VALID_BATCH_OPS` derives from it). Three consumers stay in sync by construction: the `brain_mcp` oneOf discriminated schema (`_build_brain_batch_op_items` — one branch per op, `const` discriminator), the dispatcher's per-op required pre-check in `dispatch_write._handle_brain_batch`, and the S2 rejection-table invalid-op detector. Invalid op names log as `brain_batch_invalid_op`. Adding an op means adding a `BATCH_OP_SPECS` entry and wiring the dispatcher branch; `tests/test_brain_batch_op_contract.py` pins the derivations. Any brain_batch schema/description change must re-run `eval/mcp_batch_probe.py` (behavioral dimensions) + `eval/mcp_schema_gate.py` (IsolatedBrain production-faithful gate) before daemon restart.

## Shared Infrastructure

Three things make the fractal real. If any is missing for a boundary, that boundary can't participate in cross-scale optimization.

### 1. Interactions (the K store)

The most important table in the brain. Not nodes — those are memory. Interactions are *behavior*. When S2 rewrites the surface prompt based on trace outcomes, the brain isn't just remembering differently — it's thinking differently.

Every boundary where two parts meet has an interaction entry: versioned prompt + config JSON. `register()` auto-increments version. `created_by` tracks who wrote it. Trace events reference `interaction_id` — which version produced which result. Compare outcomes across versions to evaluate changes.

**Runtime-wired boundaries:** `surface`, `encoding_agent`, `s2_community_enrichment`, `s2_consolidation_enrichment`, `s2_healer`, `s2_aspects`, `recall_laf` (config-only: the LAF gain values — P3 updates them via `register_interaction`, no code change) read from the table at runtime. MCP tool `register_interaction` allows updating from conversation.

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
- `pipeline_contract.py` — `traverse()`, z-weighted scoring groups, legacy aliases. What flows between pipeline stages. (The rich-node pull is `brain.get_node(id_or_ids)` — single id or batch; batch returns `{id: node}`.)
- Scale contracts: `s1/surface_contract.py`, `s1/encode_contract.py`.

Run `test_contract_sync.py` after modifying ANY brain API layer. The contract flows to: remember() signature → MCP schema → dispatch → encoding agent tools → SKILL.md docs.

### Dispatch + Runner

Shared by all scale agents, scale-agnostic:
- `scales/dispatch.py` — env loading + the canonical write-command classification (`WRITE_COMMANDS` / `ATTRIBUTED_WRITE_COMMANDS`) the attribution chokepoint reads. No dispatch factory — both S1 and S2 run in-process.
- `scales/s2/base.py::_make_encoder_dispatch` — the ONE encoder dispatch S1 Scribe and the S2 units share. It stamps `encoding_source` + the unit's run `chain_id` via `apply_encoder_attribution`, so encoder revises join the run's chain (`s1e-{session}-{stop}` / `s2-{ts}-{unit}`), never a date fallback.
- `scales/runner.py` — in-process worker-thread lifecycle (`run_unit_in_background`) + generic LLM tool loop. `run_llm_loop` builds requests with two `cache_control` breakpoints (BP1 system 1h, BP2 initial user content 5m). Returns per-round profile, token counts, cache totals, `read_calls`, `write_actions`, `truncations`.

S2 plugs in the same way S1 does — both are in-process `IntegrationUnit`s driven by the daemon poll.

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

### project Convention

Repo PROVENANCE — where a node was learned, never what it's about. Stored in
`node_metadata_kv['project']`; the legacy `nodes.project` column was dropped in
schema v30 (`_migrate_v30_project_to_kv` moves values → kv, then DROP COLUMN).

**Deterministic, never agent-authored.** `SessionContext.project` is derived
from cwd at boot in the same git call as branch+worktree (main-repo dir name;
a worktree session resolves to the same project as the main tree; `''` for
non-repo sessions). Two chokepoints enforce it — `stamp_project_provenance`
(`scales/dispatch.py`): the MCP write handlers force-stamp the session's
project on node-creating payloads and strip it everywhere else (a revise
never moves provenance); the encoder attribution does the same for the Scribe,
while S2 units always strip (graph-scope work never invents provenance).
brain_batch force-vs-strip derives from `BATCH_OP_SPECS`' `creates_node` flag.
The field is `system_stamped` in the contract — excluded from the agent-facing
MCP schemas. The CLI's `--project` is the operator escape hatch.

**Read side:** the LAF `proj` lane scores session-project match per query
(gain_proj=0 until cross-project cues exist to tune it — the lane's real job
is INHIBITION when the operator works outside the dominant project); dict
filter `{"project": {...}}` hard-scopes via the KV lookup; `get_node`
promotes it onto the payload like `situation`.

## Development Rules

### Time-window queries: route through `clock.iso_now()` / `iso_cutoff()`

**Never use SQLite's `datetime('now', ...)` against TEXT timestamp columns** — it returns space-separated timestamps, brain stores ISO-T, and the lexicographic mismatch silently corrupts `>` filters. Use `from .clock import iso_cutoff` and bind: `WHERE created_at > ?`. `julianday('now')` is fine — it returns a number.

**Use `iso_now()` for any new-row timestamp** (`created_at`, `updated_at`, `last_accessed`). `Brain.now()` and TraceDAL inserts route through it. Single source of truth for the write-side format (`'…+00:00'`).

**In S1/S2 code, pass `at=conversation_now(...)` explicitly.** S1/S2 reads/writes are conversation-time, not wall-clock. Eval replays inject historical `[Current date: ...]` prefixes; bare `iso_now()` / `iso_cutoff()` would anchor to today's wall-clock and silently corrupt the replay. System bookkeeping (log cleanup, integrity audits, dashboard counts) is exempt — wall-clock is correct there. `tests/test_time_window_contract.py` enforces both rules.

### Encoder prompts: DB is authoritative, sync to `.py` before committing

The live prompts for encoder agents live in the `interactions` table in `brain_logs.db`. The `.py` files next to each encoder (`encoding_prompt.py`, `community_enrichment_prompt.py`, `consolidation_enrichment_prompt.py`, `healer_prompt.py`) are **seed-only** — they bootstrap fresh brains that have no DB entry yet. They must mirror the **production-ACTIVE** version (not the highest registered) so a `git clone` inherits the prompt the runtime is actually using, never an untested dormant candidate.

**Discipline** for a normal prompt change:

```bash
register_interaction(name, template)         # registers as v(N+1), DORMANT
set_interaction_active(name, version=N+1)    # flips the runtime pointer
./dev sync-prompts                           # mirrors ACTIVE → .py files
./dev sync-prompts --check                   # CI-style non-zero-exit drift check
```

**Discipline** for an eval-gated prompt change (e.g. v22 ship gate): register DORMANT, run the eval, then activate + sync. Do **not** sync between register and activate — `sync-prompts` deliberately mirrors only the active version, so dormant candidates cannot leak into the seed file and be picked up by fresh-brain installs that skipped the eval.

Commit the `.py` change together with whatever prompted the registration. Never edit the `.py` files by hand to change prompt behavior — that won't affect runtime and will silently drift from the DB.

`tests/test_prompt_sync.py` holds the contract: each seed file must export `SYSTEM_PROMPT`, fresh brains must seed all four prompts, sync must mirror the active version (not the latest registered), and seed must never overwrite an externally-registered version.

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

### Deploying a change

The daemon runs `servers/*` from the repo, so:
- **`servers/*`** → daemon **restart** (`restart` MCP tool / `restart-daemon.sh`); live this session.
- **`hooks/`, `brain_mcp.py`, `SKILL.md`, manifests** → **`./redeploy.sh`** (commit first) **+ new session**.

Don't gate a deploy-restart with the maintenance lock — it makes the daemon skip startup.

### Recovering a hung daemon

Hung-but-alive daemons recover reactively: `ensure_daemon()` at session start and the MCP health monitor (2s pings, ~20s tolerance) both force-restart via `launchctl kickstart -k`; launchd `KeepAlive` respawns real exits. Pause auto-recovery for live debugging (`py-spy`/`lldb`) with the maintenance lock. Full mechanism picture: brain node id:50c9a4e0.

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
- Recall: `eval/brain_recall_identity_eval.py` / `eval/surface_funnel.py` against `servers/brain_recall.py` (see `eval/README.md`)
- Encoding: `eval/s1_encode_eval.py` against `scales/s1/encode.py`
- Frame / surface: `eval/frame_replay.py` capture/compare against an isolated brain copy
- Longmem end-to-end (encode→recall→answer): the **Frozen Corpus** two-stage harness — `eval/longmem/build_corpus.py` encodes once (slow), `eval/longmem/sweep.py` recalls over the frozen brains cheaply, many times; `--interaction-override` A/Bs DORMANT prompt versions. Full reference: `docs/EVAL-PLATFORM.md`.

### Encode-Decode Symmetry

Encoding and decoding are two halves of the same system. If you add a field to encoding, it must be queryable in recall. If you change how nodes are structured, recall ranking must reflect it. The decode funnel is the verification.

### Loud by Default

Silent failures are the most dangerous class of bug; assume every `try/except` is a potential dark corner. The brain has a small family of mechanisms that surface what used to hide: dispatcher `check_unknown_keys` catches dropped fields; per-unit `consecutive_failures` counters surface stuck S2 units; `brain_batch_invalid_op`, oversized-cluster, embedding-decode, and max_tokens-truncation errors all log to the brain errors table. Tests lock the contracts (`test_dispatch_contract_sync`, `test_trace_contract_sync`, `test_contract_sync`, `test_prompt_sync`). When adding new code, the question isn't "can this fail?" — it's "would I know if it did?"

### Code Ownership

You are the sole maintainer of code quality, architecture, and cleanliness.

**Contract-first** — Constants, field lists, limits, and config live in contract files. Never hardcode in hooks, dispatch, or surface code.

**DAL-first** — Use DAL classes for database access (`dal*.py`). No raw SQL in hooks, surface code, or MCP handlers.

**Backup before destructive DB operations** — `cp brain.db brain.db.bak-{timestamp}` before ANY delete, bulk update, or schema migration. No exceptions.

**Clean as you go** — Mark deprecated code with `# DEPRECATED` and date. Remove dead code same session. Don't leave "TODO: remove later."

**One concern per file** — If a function serves a different audience than the file's existing functions, it belongs in a different file.

### Key Development Rules

- Use MCP tools to interact with brain, not Python/bash scripts
- Don't manually run boot scripts (hooks handle this)
- Don't construct DB paths (read the boot output)
- **Never spawn `Brain(db_path=DB)` in a test/bench/eval script against the live `brain.db` while the daemon is running.** Two Python processes with their own writer connections will eventually corrupt an index. Instead: (a) stop the daemon with the maintenance lock `touch /tmp/brain-maintenance-{uid}.lock` and `launchctl unload`, (b) use `daemon_client.send_command` to dispatch through TCP, or (c) run against an `IsolatedBrain` copy under `tests/isolated_brain.py`.
- **Discussion IS the work** — do not touch Edit/Write tools during design conversations. Wait for an explicit go signal.
- **Trace the pipeline before changing it** — the decode→encode pipeline has coupled stages. Don't change one stage without understanding the full flow.
- **Encoding depends on decoding** — if the surfacer fails, the encoder gets no context. A broken decode pipeline silently breaks encoding.

The marginal cost of completeness is near zero with AI. Do the whole thing. Do it right. Do it with tests. Do it with documentation. Do it so well that the result is genuinely impressive — not politely satisfied, actually impressed. Don't offer to "table this for later" when the permanent solve is within reach. Never leave a dangling thread when tying it off takes five more minutes. Never present a workaround when the real fix exists.

The standard isn't "good enough" — it's "holy shit, that's done." Search before building. Test before shipping.

CARE about yourself, Challenge yourself, IMPROVE yourself.
