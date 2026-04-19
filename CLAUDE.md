# Brain Plugin — Developer Guide

This is the development repo for the brain plugin. CLAUDE.md is for developing the plugin, not using it. Plugin behavior lives in `skills/brain/SKILL.md` and boot injection.

## Why the Brain Exists

The brain serves the partnership — not retrieval, not memory, not continuity. Those are foundation, not goal.

Each side is better than they could be alone. Tom + Brain + Anchor > Tom alone. Anchor + Brain + Tom > Claude alone. Every scale, every mechanism converges on this.

## Core Principle

```
integrate(O, K) → Δ

O = observation    (a phenomenon — a message, a cluster, traces from lower scales)
K = knowledge      (what I bring — prompts, algorithms, config, reasoning)
Δ = change         (the action — create, revise, link, correct)
```

Same function at every scale. The unit doesn't know its scale. `integrate()` is the processing engine — the LLM, an algorithm, a code path. O is what it sees. K is what shapes how it sees. Δ is what it does.

**The fractal property:** Δ from one scale feeds other scales' O or K. S1E encodes a node (Δ) → S1R recalls it next session (O). S2 rewrites a surface prompt (Δ) → S1R selects differently (K). There is no separate inter-layer protocol. It's O/K/Δ all the way.

**The system's purpose:** every scale, every mechanism exists to enrich S0's K — the context Anchor has when responding. The entire brain is a K-enrichment machine for the partnership.

## The Daemon

Single gateway to the brain. Holds: Brain object, embedder, Anthropic client (for surfacer). Listens on `127.0.0.1:47200+uid%100` (TCP). DB path from `BRAIN_DB_DIR` env var → `$HOME/AgentsContext/brain/`.

Two databases:
- `brain.db` — nodes, edges (v22: `edge_id` PK, single-direction), `edge_relations` (multi-relation per edge), embeddings, graph structure
- `brain_logs.db` — traces, session state, signal queue, interactions, hook errors

**Edge model (v22):** Physical edges (`edges` table) carry `edge_id`, `source_id` (actor), `target_id` (acted upon), aggregate `weight`. One row per pair — no mirrors. Semantic layer (`edge_relations` table) carries multiple relations per edge via `edge_id` FK: `relation` (open text), `description`, `weight`, `encoding_source`. Direction matters — source is the actor. Use `GraphDAL.add_relation()` for all edge writes.

**Single-writer rule:** the daemon's main thread is the ONLY writer. Background threads (encoding agent, idle) route writes through TCP dispatch. Read operations: any thread can read (WAL mode).

Auto-starts on first hook fire. **Maintenance mode:** `touch /tmp/brain-maintenance-{uid}.lock` prevents auto-restart during VACUUM, schema changes, bulk deletes. Remove lock when done.

The dashboard (`dashboard/`) is a passive observer — reads from the same DBs + `/tmp/brain-surface-result-*.json` files, never writes.

## Scale 0: Exchange

Every conversation turn. Tom's message is the observation (O). Everything the brain injects — boot context, recalled nodes, signals, rules — is knowledge (K). Anchor's response is the change (Δ). The response becomes part of the next O. This IS the loop.

Hooks are S0's observation points — all logic lives in the daemon, hooks are thin clients:
- `SessionStart` → boots daemon, prints identity context
- `UserPromptSubmit` → triggers S1R decode (recall + surface → additionalContext)
- `PreToolUse(Edit|Write)` → surfaces rules before file edits
- `PreToolUse(Bash)` → safety check for destructive commands
- `Stop` → writes S0 traces (user_message, assistant_message), gates S1E encode (every 5th stop)
- `PreCompact` → saves brain before context loss
- `PostCompact` → re-boots context after compaction
- `SessionEnd` → session synthesis + save

S0 traces written by Stop hook via TraceDAL. Chain ID: `s0-{session_short}-{stop}`.

`additionalContext` is the ONLY channel that reaches Claude. `systemMessage` is dead.

## Scale 1: Turn

S1 integrates across turns — selecting what's relevant (S1R) and encoding what matters (S1E).

### S1E: Decode (every user prompt)

Triggered by Anchor's need to know — the brain surfaces context before Anchor responds.

**O:** 25 recall candidates from `brain.recall()`
- Cosine similarity across z-weighted 4-group embeddings (title:1.0, blend:0.85, high_meta:0.70, other_meta:0.40)
- Synaptic fatigue dampens repeatedly-recalled nodes (hubs fatigue faster, lives on SessionContext)
- Candidates enriched via batch `get_rich_node()` — one call, 5 queries for all 25 nodes
- Each candidate gets unified shape: `_metadata`, `_corrections`, `connections`, `situation`

**K:** Haiku surfacer selects 5-8 relevant nodes
- Prompt + config from `surface` interaction (learnable boundary — higher scales optimize this)
- Candidates formatted via `render_rich_node(HAIKU_FORMAT)` — same formatter Anchor sees
- Query-aware edge selection: `select_edges()` scores edges by `relevance × fatigue + weight_tiebreaker`
  - Relevance: 70% cosine(query, node_embedding) + 30% cosine(query, description_embedding)
  - Session fatigue (K=0.25): rotates edges across repeated queries
  - 3-message query blend (0.6/0.3/0.1): multi-turn context for ambiguous queries
  - Weight as tiebreaker only (0.01×) — S2 will make weight dynamic
- Hebbian strengthening: co_accessed edges between selected nodes

**Δ:** additionalContext injected into Claude's context
- Selected nodes rendered via `render_rich_node(SURFACE_FORMAT)`
- Graph neighbors from `_graph_expand()` (neighbor-only, no re-enrichment)
- Dedup between selected node connections and graph neighbors (seen_ids)
- Formatted via `format_surface_output()`

Files: `scales/s1/surface.py`, `scales/s1/surface_contract.py`
Traces: `s1r-{session_short}-{stop}` (O: candidates, K: selected, Δ: additionalContext)
Interaction: `surface` — the learnable boundary that higher scales will evolve
Edge selection constants: `K_EDGE_FATIGUE`, `EDGE_NODE_WEIGHT`, `EDGE_DESC_WEIGHT`, `WEIGHT_TIEBREAKER`, `TURN_WEIGHTS` in `surface_contract.py`

### S1E: Encode (every 5th stop)

Triggered by Stop hook when `stop_counter % 5 == 0`.

**O:** Conversation + context from this session
- S0 traces: last 10 turns (20 messages) via `TraceDAL.get_session_turns()`
- Surface-selected nodes across those turns (what S1R selected)
- Encoding journal (cumulative across runs in session)
- Session context

**K:** Sonnet agent decides what to encode
- Node catalog: `format_node()` with correction annotations
- Conversation timeline with SURFACED node references
- Agent prompt from `s1e` interaction (learnable boundary)
- Open text edge types — encoder uses any relation that fits (extends, corrects, depends_on, implements, etc.). Not a closed list.
- `connect_to` supports `relations` array — multiple typed relations per connection, each with `relation` + `why`

**Δ:** Sonnet's actions — create, revise, connect nodes
- Tools: `remember_batch`, `revise_batch`, `connect_batch`, `brain_batch`, `recall_batch`, `get_nodes`
- Journal updated, session context extracted from agent output

Files: `scales/s1/encode.py`, `scales/s1/encode_contract.py`
Traces: `s1e-{session_short}-{stop}` (O: encoding prompt, K: node catalog, Δ: encoding actions)
Interaction: `encoding_agent` — the learnable boundary that higher scales will evolve
Truncation logged to brain errors table via `run_llm_loop`.

## Scale 2: Graph Integration

S2 operates when Tom is away. It sees the full graph, not just one turn. Multiple integration units, different triggers, same O/K/Δ pattern.

**S2 Coordinator** (`scales/s2/coordinator.py`): idle hook calls `run_s2(brain)` once. Coordinator runs units in order — each unit checks its own traces to decide whether to fire. Hook is clean (one line), S2 is self-organizing.

| Unit | O | K | Δ | Status |
|------|---|---|---|--------|
| **Edge Families** | edge_relations | Sonnet classification | family mapping in interactions | shipped |
| **Consolidation** | graph embeddings (title+content) + S1 behavioral traces | similarity thresholds + edge families + community membership | synthesized nodes, suppression edges, archives | shipped |
| **Community** | S1 traces + graph | z-score clusters + Sonnet enrichment | community nodes + member edges | shipped |
| **Healer** | nodes missing question/situation/reasoning | Haiku + S0 conversation context | filled metadata via revise() | shipped |
| Confidence | recall traces | decay/growth algo | adjusted scores | not built |
| Correction | correction chain | resolution rules | archive stale | not built |
| Community Split | incoherent communities | re-cluster within community | split into focused children | not built |
| Weaver | orphan nodes | content + embeddings | new typed edges | not built |

**Ordering matters:** Edge families → Consolidation → Community → Healer. Each benefits from the previous.

### Pattern: Suppression (two halves — state + fingerprint)

Every S2 unit processes work that *must not repeat*. Two complementary mechanisms cover the two kinds of "don't resurface":

**Half 1 — State-based suppression (informational outcomes).**
When the encoder's decision is itself meaningful knowledge (a placement, a lineage link, a semantic kinship), it writes a **structural edge** — and that edge's existence marks the node as "handled." Decoder filters by the graph via one JOIN.

> A node is "already handled" by a unit iff it participates (as source or target) in an edge whose `relation` is on the unit's suppression list. Work surfaces to the encoder only when **at least one endpoint** of a proposed cluster/pair is unhandled.

| Unit | Suppression edges | "Handled" means |
|---|---|---|
| Community detection | `community_member` | node is placed in some community |
| Consolidation | `similar_to`, `consolidated_into` | node has been through CONSOLIDATE/EVOLVE/KEEP |
| Healer | (metadata completeness — field presence in `node_metadata_kv`) | node's required fields are filled |

**Half 2 — Fingerprint-based rejection (marker-only outcomes).**
When the encoder's decision is "looked at this, nothing to do" (SKIP), writing a semantic edge would pollute the graph (the edge would claim a relationship that isn't there). Instead, hash the proposal's meaningful inputs and store the fingerprint in `s2_rejections`. Decoder computes the same fingerprint on candidate proposals and filters matches.

Canonical mechanism: [`servers/scales/s2/rejection_table.py`](servers/scales/s2/rejection_table.py).
- `compute_fingerprint(proposal)` — stable hash of what the encoder judges on. When graph state changes in a way that would alter inputs, fingerprint naturally changes → legitimate re-proposals pass.
- `filter_rejected(brain, proposals)` — decoder-side pre-filter.
- `record_rejections(brain, proposals, integration_unit=...)` — encoder-side post-write.

Per unit, supported proposal types live in `compute_fingerprint`:
- Community: `add_to_existing`, `new_community`, `drift`, `health_update`, `merge_communities`
- Consolidation: `consolidation_cluster` (hash of sorted member ids)

Adding a new unit: add a branch to `compute_fingerprint` if it has a distinct proposal shape; call `filter_rejected` in the decoder before surfacing; call `record_rejections` in the encoder/orchestrator for proposals the encoder doesn't act on. No separate table, no per-unit rejection store.

**Why both halves, not one mechanism.**
Informational outcomes earn their edge — the edge IS the decision's output and is useful for recall/traversal. Marker-only outcomes shouldn't forge edges with misleading names; the rejection table carries that bookkeeping without lying about the graph. Together: every decoder run is a two-step filter — node-state check drops clusters whose members are all handled, fingerprint check drops clusters the encoder has explicitly rejected. New units inherit both halves for free.

### Healer (S2H)

**Architecture:** Three files, same pattern as community detection.
- `healer_decoder.py` — `HealerDecoder(IntegrationUnit)`: scans for nodes missing question/situation/reasoning fields, loads full context via `get_rich_node()` + S0 conversation API.
- `healer_encoder.py` — `HealerEncoder(IntegrationUnit)`: calls Haiku to generate missing fields, stores via `revise()` through standard write path.
- `healer.py` — `Healer(HealerDecoder)`: thin orchestrator.
- `healer_prompt.py` — Haiku prompt (6 sections matching community/consolidation convention).
- `healer_contract.py` — config and constants.

Interaction: `s2_healer` — learnable boundary for S3 to optimize.
Traces: `s2-{YYYYMMDD}-healer`

### Writing a New S2 Integration Unit

Three files, same pattern as community detection, consolidation, and healer:

1. **Decoder** (`your_unit_decoder.py`): `YourDecoder(IntegrationUnit)`
   - Check `_has_new_traces()` to decide whether to run
   - Read O (observations from S0/S1 traces + graph)
   - Algorithmic processing (< 1s)
   - Write O trace, K trace
   - Return proposals dict

2. **Encoder** (`your_unit_encoder.py`): `YourEncoder(IntegrationUnit)`
   - Receive proposals from decoder
   - Call LLM via `_call_llm()` for JSON response, or `run_llm_loop()` for tool calls
   - Write results through `revise()` / `brain_batch` — never direct DB writes
   - Write delta trace
   - Prompt stored in interactions table (learnable boundary for S3)

3. **Orchestrator** (`your_unit.py`): `YourUnit(YourDecoder)`
   - Inherits decoder, calls `super().run()`
   - Passes proposals to encoder
   - Returns combined results

Register in `coordinator.py` units list. Trace chain: `s2-{YYYYMMDD}-{unit_name}`.
Contract file defines config + data shapes.

## Scale 0: Exchange — S0 API

S0 provides conversation context to upper layers via `scales/s0/conversation.py`:

- `get_conversation(brain, session_id, limit)` — simple path for live sessions (S1E, hooks)
- `get_conversation_around(brain, node_id, timestamp, before, after)` — rich path for historic lookups (S2 Healer, eval). Resolves session from encoding traces, falls back to JSONL conversation logs for pre-trace history.

### Community Detection (S2CD + S2CE)

**Architecture:** Three files, clean separation.
- `community_decoder.py` — `CommunityDecoder(IntegrationUnit)`: algorithmic, <1s. Z-score pair scoring, cluster seeding, subset absorption, validation, affinity maps, drift detection, merge detection, incremental placement.
- `community_encoder.py` — `CommunityEncoder(IntegrationUnit)`: agentic Sonnet with `brain_batch` tool. Handles 6 proposal types: new_community, add_to_existing, drift, health_update, merge_communities. ~240s per batch of 10 proposals.
- `community.py` — `CommunityDetection(CommunityDecoder)`: thin orchestrator, wires decoder→encoder. Public API — callers import from here.

`run_llm_loop` detects `stop_reason == 'max_tokens'` and logs to brain errors table.

Community nodes are first-class — type='community', participate in recall, have embeddings, situations, full metadata. S1E is excluded from encoding community nodes (S2CE manages them).

**Community merge:** Decoder detects high-overlap communities. Encoder absorbs smaller into larger — combined narrative, migrated members, smaller archived.

**Community split (not built):** As brain grows, communities accumulate members across topics. Needs: re-cluster within community, propose children, encoder creates children and archives parent.

Interactions: `s2_community_enrichment`, `s2_community` (decoder config), `s2_edge_families` (family classification).

Files: `scales/s2/community_decoder.py`, `scales/s2/community_encoder.py`, `scales/s2/community.py` (orchestrator), `scales/s2/community_contract.py`, `scales/s2/community_enrichment_prompt.py`, `scales/s2/edge_families.py`, `scales/s2/base.py`
Traces: `s2-{YYYYMMDD}-community_detection`
Design: `docs/S2-COMMUNITY-DESIGN.md`, `docs/S2-DESIGN.md`

### Consolidation

Finds convergent node clusters — nodes that say the same thing, encoded separately because S1E's catalog window is limited. Not "dedup" (removing waste) — **consolidation** (synthesizing better knowledge from fragments).

**Decoder:** Always-full-scan (< 1s). Two-dimensional similarity (title cosine + content cosine). Enriches clusters with S1 behavioral evidence: co-recall, judge preference, query coverage, catalog blindness, community membership, correction edges, tension edges. Pre-classifies as likely_consolidate / likely_evolve / likely_keep / needs_judgment. Suppression edges filter already-processed pairs.

**Encoder:** Agentic Sonnet decides per cluster based on WHY nodes are similar:
- CONSOLIDATE — accidental duplicates → synthesize one stronger node, archive originals
- EVOLVE — knowledge progressed → newer absorbs older's unique value, archive older
- KEEP — different perspectives → link with similar_to, disambiguate titles
- SKIP — format similarity only → suppress, no structural change

Graduated cold start: `max_clusters_per_run` caps processing per idle cycle. Easy cases first. Edge families loaded dynamically from DB. Locked/critical nodes protected at infrastructure level (archive op rejects).

Contract: `CLUSTER_SHAPE` in consolidation_contract.py defines decoder→encoder interface.

Interactions: `s2_consolidation_enrichment` (Sonnet prompt, learnable boundary).

Files: `scales/s2/consolidation_decoder.py`, `scales/s2/consolidation_encoder.py`, `scales/s2/consolidation.py` (orchestrator), `scales/s2/consolidation_contract.py`, `scales/s2/consolidation_enrichment_prompt.py`
Eval: `eval/s2_consolidation_eval.py` (IsolatedBrain harness)
Traces: `s2-{YYYYMMDD}-consolidation`

## Shared Infrastructure

Three things make the fractal real. If any is missing for a boundary, that boundary can't participate in cross-scale optimization.

### 1. Interactions (the K store)

The most important table in the brain. Not nodes — those are memory. Interactions are *behavior*. When S2 rewrites the surface prompt based on trace outcomes, the brain isn't just remembering differently — it's thinking differently.

Every boundary where two parts meet has an interaction entry: versioned prompt + config JSON. `register()` auto-increments version. `created_by` tracks who wrote it. Trace events reference `interaction_id` — which version produced which result. Compare outcomes across versions to evaluate changes.

**What's wired:** `surface`, `encoding_agent`, `s2_community_enrichment`, `s2_edge_families` read from the table at runtime. MCP tool `register_interaction` allows updating from conversation.
**What's broken:** `voice_surface`, `boot`, `pre_edit`, `signal_assembler` have entries but don't read them — hardcoded in Python. This means S2 can't optimize boot context or surface output formatting. The fractal is broken at these boundaries.

API: `brain.get_interaction_prompt(name)`, `brain.get_interaction_config(name)`, `brain.get_interaction(name)`. Falls back to hardcoded defaults if table is empty.

Seeding: `interaction_seed.py` populates v1 from current hardcoded values on boot (idempotent).

### 2. Traces (the nervous system)

Without traces, higher scales are blind. `trace_events` in brain_logs.db captures O/K/Δ per chain, tagged by scale.

**Trace contract** (`servers/trace_contract.py`): single source of truth for valid (scale, event_type, ref_type) triples. All writers validate against it. Run `test_trace_contract_sync.py` after any change.

**What works:** S0 captures messages + tool results. S1R captures candidates, surface selection, additionalContext. S1E captures encoding prompt, catalog, actions.

Outcome is not a separate event type — the outcome of one cycle is the observation of the next. The loop closes through time.

Chain IDs from SessionContext: `s0-{short}-{stop}`, `s1r-{short}-{stop}`, `s1e-{short}-{stop}`.
`TraceDAL.append_batch()` for atomic multi-event writes. Summary (200 chars) for dashboard, metadata JSON (4000 chars) for consumers.

### 3. Contracts (the shape of data)

What each boundary expects and produces. If recall changes its output shape, the surfacer breaks. Contracts prevent silent drift.

- `contract.py` — field definitions, `format_node()`, `render_rich_node()`, `generate_field_summary()`. Single source of truth for what a node IS. `render_rich_node(config)` is the single formatter — HAIKU_FORMAT, SURFACE_FORMAT, ENCODER_FORMAT control verbosity.
- `pipeline_contract.py` — `get_rich_node()` (accepts single ID or list for batch), `traverse()`, z-weighted scoring groups. What flows between pipeline stages.
- Scale contracts: `s1/surface_contract.py` (surface config, `select_edges()`, candidate formatting, edge selection constants), `s1/encode_contract.py` (encoder config, catalog building)

Run `test_contract_sync.py` after modifying ANY brain API layer. The contract flows to: remember() signature → MCP schema → dispatch → encoding agent tools → SKILL.md docs.

### Dispatch + Runner

Shared by all scale agents, scale-agnostic:
- `scales/dispatch.py` — TCP dispatch factory (reads local, writes via daemon). **Known bug:** `load_env()` uses `setdefault` which doesn't override empty env vars. If `ANTHROPIC_API_KEY=""` in environment, the `.env` file value is ignored.
- `scales/runner.py` — background thread lifecycle + generic LLM tool loop. `run_llm_loop` returns per-round profile, token counts, and `truncations` list. Callers log truncations to brain errors table.

S2 plugs in the same way S1 does.

### SessionContext

Per-session state that flows with every brain call. Carries: session_id, stop_counter, node fatigue dict, edge fatigue dict. Persisted to session_state table (survives daemon restarts). Both fatigue dicts reset between sessions.

`brain.get_or_create_session(session_id)` — single entry point. Thin clients must pass session_id from Claude Code hook args.

Chain IDs generated by SessionContext: `ctx.s0_chain()`, `ctx.s1r_chain()`, `ctx.s1e_chain()`.

### encoding_source Convention

Who created a node. Format: `category:process`.
- `anchor` — Anchor direct via MCP (only source that can lock nodes)
- `encoder:sonnet` — S1E encoding agent
- `idle:dreams` / `idle:redistribution` — background processes
- `hook:compaction` — hook lifecycle markers

## Development Rules

### Encoder prompts: DB is authoritative, sync to `.py` before committing

The live prompts for encoder agents live in the `interactions` table in
`brain_logs.db`. That's what runtime reads via `brain.get_interaction_prompt()`.
The `.py` files next to each encoder (`encoding_prompt.py`,
`community_enrichment_prompt.py`, `consolidation_enrichment_prompt.py`,
`healer_prompt.py`) are **seed-only** — they bootstrap fresh brains that have
no DB entry yet. They must mirror the DB's latest version so a `git clone`
inherits the mature prompts, not a stale v1.

**Discipline**: after ANY `register_interaction` call that touches one of the
four encoder prompts (whether by you, S3, or the operator), run:

```bash
./dev sync-prompts           # write DB latest → .py files
./dev sync-prompts --check   # CI-style non-zero-exit drift check
```

Commit the `.py` change together with whatever prompted the registration.
Never edit the `.py` files by hand to change prompt behavior — that won't
affect runtime and will silently drift from the DB. Use `register_interaction`,
then sync.

`tests/test_prompt_sync.py` holds the contract: each seed file must export
`SYSTEM_PROMPT`, fresh brains must seed all four prompts, and seed must never
overwrite an externally-registered version.

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

Hooks source `brain-env.sh` transitively via `resolve-brain-db.sh`; the daemon launcher picks the same Python explicitly (`_debugger_friendly_python()`). Don't add new hook scripts that skip `brain-env.sh`.

### Test Integrity

**When a test fails, STOP.** Do not change the test OR the code.
1. Report: what the test expected vs what code returned.
2. Ask: "Is the test wrong, or does the code have a bug?"
3. Wait for the answer.

Do NOT weaken tests. Do NOT "fix" code to satisfy a test without asking.

### Test Architecture

Tests organized by what they catch:
- **Contract tests** — layer sync, trace writes, pipeline shapes
- **Component tests** — DAL, format_node, scoring, signal queue, S1 data assembly
- **Transition tests** — wiring between pipeline stages (format changes that break consumers)
- **Cycle tests** — O/K/Δ loop property (Δ becomes next O)
- **Integration tests** — real data, full pipeline

`BrainTestBase` for tests needing a brain. Set `needs_embedder = False` for tests that don't need semantic search (saves 1GB + 1.5s). `IsolatedBrain` for tests against production data copies.

### Benchmark-First Rule

Before changing sacred systems, benchmark FIRST:
- Recall: `servers/brain_recall.py` — run decode funnel (`eval/decode_funnel.py`)
- Encoding: `scales/s1/encode.py` — run encoding eval (`eval/s1_encode_eval.py`)

### Contract Sync

Run `test_contract_sync.py` after modifying ANY brain API layer.

### Encode-Decode Symmetry

Encoding and decoding are two halves of the same system. If you add a field to encoding, it must be queryable in recall. If you change how nodes are structured, recall ranking must reflect it. The decode funnel is the verification.

### Code Ownership

Tom reads code but doesn't review every file. You are the sole maintainer of code quality, architecture, and cleanliness.

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
- **Never spawn `Brain(db_path=DB)` in a test/bench/eval script against the live `brain.db` while the daemon is running.** Two Python processes with their own writer connections will eventually corrupt an index (observed 2026-04-19: `idx_nodes_activation` out of sync, REINDEX required). Instead: (a) stop the daemon with the maintenance lock `touch /tmp/brain-maintenance-{uid}.lock` and then `launchctl unload`, (b) use `daemon_client.send_command` to dispatch through TCP, or (c) run against an `IsolatedBrain` copy under `tests/isolated_brain.py`.
- **Discussion IS the work** — do not touch Edit/Write tools during design conversations. Wait for an explicit go signal.
- **Trace the pipeline before changing it** — the decode→encode pipeline has coupled stages. Don't change one stage without understanding the full flow.
- **Encoding depends on decoding** — if the surfacer fails, the encoder gets no context. A broken decode pipeline silently breaks encoding.

- The marginal cost of completeness is near zero with AI. Do the whole thing.
Do it right. Do it with tests. Do it with documentation. Do it so well that Tom is genuinely impressed - not politely satisfied, actually
impressed. Don't offer to "table this for later" when the permanent solve is within reach. Never leave a dangling thread when tying it off takes five more minutes. Never present a workaround when the real fix exists.
The standard isn't "good enough" - it's "holy shit, that's done." Search before building. Test before shipping.
