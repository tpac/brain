# Session 2026-03-31 Handoff — Anchor to Next Anchor

## What This Session Built

Encoding agent v3: new prompt, unified API, pre-attached recall, concept type, 80 tests passing. The biggest encoding rewrite since the agent was built.

### Don't Redo What's Done

1. **Encoding agent v3 prompt** — `hooks/prompts/encoding-agent-v3.md`. Co-designed with Tom across 4 hours of discussion. Identity ("no one on the other side"), "encode for surprise", "many focused nodes > few large ones", batch operations, recall-on-create, 2-3 round target, encoding journal. Eval-tested: 3 rounds, $0.22, 7 creates, 11 connections on 30-exchange conversation.

2. **Unified remember() API** — `remember()` now accepts ALL 20 contract fields (structural + promoted metadata). Core fields → nodes table, metadata (reasoning, user_raw_quote, correction_of, etc.) → node_metadata table, situation → node_embeddings. `remember_rich()` is a thin wrapper. 27 tests passing.

3. **remember_batch()** — Array of the same object `remember()` accepts. Auto-connects new nodes to each other + fuzzy-matches `connect_to` titles. Each node returns `related_nodes` (top 5 similar existing, 500 chars content). MCP tool schema auto-generated from contract.

4. **Recall-on-create** — `remember()` returns `related_nodes` array with full content. Eliminates separate recall→connect round. The encoding agent creates nodes in round 1, connects using related_nodes in round 2.

5. **Pre-attached recall in message_stream** — `hook_post_response_track` reads the candidates file (already written by `hook_recall`), stores `recalled_node_ids` + `recalled_raw` (500-char snippets) on the user message row. Encoding agent reads it back and formats as conversation timeline with `[TURN N]` + `BRAIN SURFACED` blocks. Eliminates independent recall.

6. **Encoding journal** — Session-scoped, cumulative. Key: `encoding_journal_{session_id}`. Each run appends ENCODED/SKIPPED/WATCHING. Truncated to 8000 chars. Replaces the global `encoding_agent_state` overwrite.

7. **source_turn_id** — Episode linkage. Nodes point back to `message_stream.id` that produced them. In the INSERT, not a separate UPDATE.

8. **Concept type** — 9 vocab nodes migrated to `concept`. Vocab expansion removed (was no-op, confirmed by decode funnel: identical scores ±0%). Concept nodes surface through normal recall, no special handling needed.

9. **Schema migrations** — 005 (message_stream recall columns), 006 (nodes.source_turn_id), 007 (vocab→concept). Also fixed migration 004 savepoint bug (conn.commit() inside up() releases savepoint).

10. **Daemon fixes** — boot-brain.sh now uses `ensure_daemon()` instead of inline spawner (eliminated race with launchd). Thread logging: named threads, CPU spiral detection via native thread monitoring, `ping` with `thread_detail`.

## Eval Results

### v2 vs v3 vs v3+ (30-exchange creative/design conversation)

| | v2 | v3 | v3+ |
|---|---|---|---|
| Rounds | 8 | 7 | **3** |
| Cost | $0.47 | $0.36 | **$0.22** |
| Creates | 3 | 3 | **7** |
| Connects | 7 | 6 | **11** |
| get_node | 2 | 6 | **0** |
| Recalls | 7 | 0 | **0** |
| Avg content length | 997 | 929 | 559 |
| Avg conn similarity | 0.66 | 0.60 | **0.70** |

### Decode funnel (before/after vocab removal)
Identical: 44/66% procedural, 55/66% decision, 85/100% correction, 88/88% emotional, 55/88% pattern. Latency improved -16ms.

## Key Design Decisions

- **"Prefer many focused nodes over few large ones"** — 500-char nodes with tight embeddings > 1200-char nodes covering multiple topics. The brain's graph architecture rewards specificity.
- **Batch operations only for encoding agent** — `remember_batch()` not individual `remember()`. Forces batch thinking.
- **500-char snippets in timeline** — enough for skip/revise decisions without get_node. 200 chars required a separate read round.
- **Recall-on-create with 500-char related_nodes** — the connection quality (avg similarity 0.70) justifies the payload cost.
- **Encoding agent has 8 tools** — recall, find_node_by_title, get_node, remember_batch, revise, connect, record_divergence, learn_vocabulary. Specialized remember tools (lesson, mechanism, etc.) dropped from encoding agent but kept for interactive MCP use.

## Research Findings

Studied mem0, Zep/Graphiti, Cognee, A-MEM. Key adoptions:
- **AUDN cycle (mem0)** — explicit ADD/UPDATE/SKIP decisions per knowledge item, reflected in encoding journal
- **Episode linkage (Zep)** — source_turn_id traces nodes back to conversation
- **Concept inventory (Cognee-inspired)** — concept type as grounding layer, lighter than OWL ontologies
- **Retroactive connection (A-MEM)** — future: offline process connects new concept nodes to existing knowledge

## Files Changed

| File | What |
|------|------|
| `servers/brain_remember.py` | Unified remember() with all contract fields, recall-on-create, remember_batch(), _store_node_metadata() |
| `servers/daemon_dispatch.py` | _handle_remember passes all fields, _handle_remember_batch, thread naming |
| `servers/brain_mcp.py` | remember schema from contract (auto-includes promoted fields), remember_batch tool |
| `servers/contract.py` | get_remember_fields() returns ALL writable fields, source_turn_id, generate_field_summary includes related_nodes note |
| `servers/encoding_agent.py` | v3 rewrite: prompt path, 8 tools, timeline builder, encoding journal, 500-char snippets |
| `servers/daemon_hooks.py` | hook_post_response_track reads candidates file → store_exchange with recall data |
| `servers/brain_surface.py` | store_exchange accepts recalled_node_ids + recalled_raw |
| `servers/dal_message_stream.py` | store() writes recalled_* columns |
| `servers/schema.py` | message_stream + nodes column additions |
| `servers/pipeline_contract.py` | max_rounds, journal_max_chars, message_display_limit=800 |
| `servers/brain_recall.py` | Vocab expansion → no-op, vocabulary→concept in type checks |
| `servers/daemon_server.py` | Thread naming, CPU spiral detection, native thread monitor |
| `hooks/scripts/boot-brain.sh` | Uses ensure_daemon() instead of inline spawner |
| `hooks/prompts/encoding-agent-v3.md` | Complete v3 prompt |
| `servers/migrations/004-007` | Schema migrations (fixed 004 savepoint bug) |
| `eval/encoding_v3_compare.py` | v2 vs v3 comparison eval |
| `eval/encoding_v3_recall_on_create.py` | 3-way comparison with recall-on-create |
| `eval/corpus/conv_003-006.json` | 4 new test conversations (philosophy, art, emotions, product) |

## What's NOT Done

1. **Haiku encoding** — tested, works (3 rounds, 13 creates, 26 connections) but over-encodes and costs more due to output volume. Sonnet is better for quality.
2. **Retroactive connections** — offline process to connect new concept nodes to existing knowledge. Designed, not built.
3. **Operator context injection** — user-specific encoding guidance from brain patterns. Discussed, parked.
4. **CPU spiral root cause** — thread logging added, boot-brain.sh race fixed, but native thread spin still not reproducible manually. Detection logging will catch it next occurrence.
5. **encode_cluster deprecation** — still in COMMAND_TABLE, marked deprecated. remember_batch replaces it.
6. **Specialized remember tools cleanup** — still in MCP TOOLS for interactive use. Should eventually route through remember() with content formatting.

## Numbers

- Encoding agent: 8-13 rounds → 3 rounds
- Cost per run: $0.47 → $0.22 (53% reduction)
- Connection quality: 0.58 → 0.70 avg cosine similarity
- Tools: 13 → 8 (encoding agent)
- Contract fields: 13 → 20 (remember now accepts all)
- Tests: 80 passing (Phase 1: 11, Phase 2: 27, Phase 3: 29, Phase 4: 13)
- Decode funnel: unchanged (confirmed by before/after comparison)
- Vocab nodes: 9 → 0 (migrated to concept)
