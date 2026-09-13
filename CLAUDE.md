# Entity — Developer Guide

This repository develops the Entity brain plugin. Keep this guide short and current: project architecture, development constraints, and references. History belongs in the brain and git; consumer instructions belong in the plugin skills.

## Architecture

The core model is `integrate(O, K) → Δ`: observations and knowledge produce change at each scale. One unit's output becomes another's input; the graph and traces carry that continuity.

- **S0** records host conversation and tool activity. **S1** connects conversation to memory: the Scribe encodes turns into nodes/edges; recall and surface select graph knowledge for the current conversation.
- **Frame** is the deterministic session prior: session context, current focus, and recent moves. It is composed from session state; recall/surface handle memory selection separately.
- **S2** integrates accumulated graph knowledge across sessions. The coordinator runs aspect classification → community detection → healer → consolidation. Consolidation runs last so earlier writes do not immediately invalidate its rejection fingerprints.
- **Community** separates algorithmic discovery from LLM judgment: typed-edge structure and incremental placement produce proposals; the encoder creates/revises community nodes through shared dispatch. Membership is represented by `community_member` edges; structural metrics are derived from the graph. Decoder configuration (`s2_community`) and encoder prompt/config (`s2_community_enrichment`) have separate interaction entries.
- **S2 suppression** remembers examined proposals using fingerprints of meaningful inputs. Unchanged rejected work stays suppressed; changed evidence allows reconsideration. Run gating and rejection suppression serve different purposes.
- **Shared execution**: S1 Scribe and S2 use `IntegrationUnit` and the shared encoder dispatch for attributed writes and run traces. The daemon owns write serialization; vector updates run through the embedding queue.

Before recommending an architectural change, read the owning implementation, its contract, callers, and relevant tests. Establish what already exists and which boundary the proposal changes. Design documents provide rationale and may include proposals; verify current behavior against executable code, including when docstrings disagree.

- Each concern has one owner. Use its API; add a missing operation there instead of bypassing it. SQL belongs in `dal*.py`; use `brain_traces.py` for traces and `dal_graph.py` for edges.
- Constants, field lists, limits, and configuration belong in contract files. Consumers derive from them.
- Existing boundary leaks are not permission to add bypasses. Identify the owner before proposing or placing code.
- Keep host-specific setup in adapters; shared brain functions remain host-neutral.
- Extend existing owners before adding modules. New files need a distinct responsibility, audience, or lifecycle.
- `servers/scales/` owns integration grains and shared machinery; `servers/channels/` owns correspondents addressing live streams. Placement rules live in `servers/scales/__init__.py`.
- `brain.db` stores nodes, edges, and embeddings; `brain_logs.db` stores traces, session state, interactions, and errors.

## Code Map

Paths below are repository-relative. Module docstrings and the linked documents provide detail.

| Concern | Entry points / reference |
|---|---|
| Host integration and hooks | `servers/host_contract.py`, `hooks/hooks*.json`, `hooks/adapters/`; `docs/HOST-CONTRACT-DESIGN.md` |
| Daemon lifecycle and transport | `servers/daemon_server.py`, `servers/daemon_launch.py`, `servers/daemon_client.py` |
| Storage, writes, and backups | `servers/brain.py`, `servers/db_backends/sqlite.py`, `servers/db_backup.py` |
| Recall and Frame | `servers/brain_recall.py`, `servers/recall_laf.py`, `servers/scales/s1/frame.py`; `docs/RECALL-OVERVIEW.md` |
| Encoding and surface | `servers/scales/s1/`; `docs/ENCODE-ON-IDLE.md`, `docs/RECALL-OVERVIEW.md` |
| S2 units and coordination | `servers/scales/s2/coordinator.py`, `servers/scales/s2/base.py` |
| Community pipeline | `servers/scales/s2/community.py`, `servers/scales/s2/community_decoder.py`, `servers/scales/s2/community_encoder.py`, `servers/scales/s2/community_contract.py` |
| S2 suppression | `servers/scales/s2/rejection_table.py` |
| Aspect taxonomy | `servers/aspects.py`, `servers/aspect_store.py`, `servers/scales/s2/aspects_v1.json` |
| Corrections and edges | `servers/brain_corrections.py`, `servers/dal_graph.py` |
| Stream communication | `servers/channels/`; `docs/SELF-CHANNEL-DESIGN.md`, `docs/THALAMUS-DESIGN.md` |
| Traces and sessions | `servers/brain_traces.py`, `servers/trace_contract.py`, `servers/session_context.py`; `docs/TRACES-LAYER-DESIGN.md` |
| Interaction defaults and overrides | `servers/interaction_defaults.py`, `tests/interaction_override.py` |
| Scope provenance and visibility | `servers/scopes.py`, `servers/scales/dispatch.py` |
| Node and pipeline contracts | `servers/contract.py`, `servers/pipeline_contract.py` |
| Python runtime and process environment | `dev`, `hooks/scripts/brain-env.sh` |

## Development Constraints

- New shell hooks must source `hooks/scripts/resolve-brain-db.sh`, which loads `brain-env.sh`; use the resolved runtime and database location instead of hardcoding either.
- Pass `SessionContext` through session-scoped calls. Key conversation state by `session_id`, never a global `brain_meta` key; concurrent sessions must not clobber one another.
- Use `servers/clock.py` helpers: `iso_now()` for row timestamps and bound `iso_cutoff(...)` values for time-window queries. SQLite `datetime('now', ...)` produces a different TEXT format and silently breaks comparisons.
- In `servers/scales/`, conversation-time data uses `at=conversation_now(...)`; transaction timestamps and system bookkeeping use wall-clock time. See `tests/test_time_window_contract.py` and `tests/test_clock_contract_sync.py`.
- `as_of` replay filters today's surviving data; it cannot restore archived vectors. Treat historical replay measurements accordingly.
- Read interaction prompts/configs through `brain.get_interaction_prompt/_config`. Code owns defaults; the DB holds overrides. The recipe for adding a boundary is in `servers/interaction_defaults.py`. Use `tests/interaction_override.py` for isolated A/B overrides; promote evaluated winners into code defaults, then clear the corresponding experimental override with `brain.clear_interaction_override(name)` so it cannot mask future defaults. Inspect overrides with `./dev check-overrides`; preserve policy-managed pointers in `servers/interaction_collapse.py`.
- Activate S2 through `brain.run_s2()`, including evals and benchmarks; it owns the single-flight lock. Never call the coordinator directly. Preserve coordinator ordering and each unit's gating when changing scheduling.
- Attribution and trace chains come from the execution context, not model-authored arguments. Automated encoders/S2/hooks cannot grant node locks; preserve the interactive-source restriction at the write boundary (`servers/contract.py`, `servers/scales/dispatch.py`).
- Apply `output_config` on every round of an agentic loop.
- Gate each new S2 unit's graph scan on its own `s2_<unit>_last_run_ts` to avoid repeatedly deriving the same fixed point.
- The aspect encoder classifies into existing aspects. Taxonomy changes belong in `aspects_v1.json`; required names are owned by `servers/aspect_store.py`.
- Scope provenance is stamped by `stamp_scope_provenance`, never authored by an agent. New recall entry points must route through `brain.canonicalize_results` to preserve corrections, canonical attachments, and visibility rules.
- Encoding and recall must stay aligned: new encoded fields need recall support; structural changes need ranking verification.
- Log failures to the brain errors table; do not silently drop fields, invalid operations, or failed processing.
- Before destructive DB operations, call `backup_before_destructive(db_path, tag)` from `servers/db_backup.py`. Never copy a live WAL database with `cp`.
- Never open a second `Brain` writer against the live database in tests, benchmarks, or evals. Use `IsolatedBrain` from `tests/isolated_brain.py`, or dispatch live operations through `daemon_client.send_command`. Do not run experimental mutations on production data.
- Remove dead code within the changed scope. Comments explain current rationale, not history. Follow the change through its callers, tests, and documentation without expanding into unrelated work.
- Design discussions do not authorize edits; wait for an explicit implementation request.

## Validation

Run development commands through the bundled runtime:

```bash
./dev pytest tests/                   # test suite
./dev python3 path/to/script.py       # scripts, benchmarks, evals
./dev                                # subshell with runtime on PATH
```

- Use `BrainTestBase` from `tests/brain_test_base.py`; set `needs_embedder = False` when semantic search is unnecessary. Use `IsolatedBrain` for production-data copies.
- Do not widen guardrail allowlists, exclusions, or frozen baselines merely to make a change pass. Resolve the boundary violation; justify any intentional exception separately.
- Record the tested revision, isolated dataset, effective model/config, and interaction fingerprints. Compare equivalent starting states and verify intended overrides reached the resolver before interpreting results.
- When a test fails, stop and report expected versus actual behavior. Ask whether the test or implementation is wrong before changing either.
- Benchmark before changing recall, encoding, or Frame/surface. Entry points: `eval/brain_recall_identity_eval.py`, `eval/surface_funnel.py`, `eval/s1_encode_eval.py`, `eval/frame_replay.py`. Longmem and broader evaluation workflows: `eval/README.md`, `docs/EVAL-PLATFORM.md`.
- For community changes, check decoder proposals and encoder outcomes separately, including membership reconciliation and suppression/retry behavior. Start with `tests/test_s2_community.py`, `tests/test_community_membership_reconcile.py`, `tests/test_community_unplaceable.py`, and `eval/s2_community_decoder_eval.py` on isolated data. That eval simulates encoder acceptance; validate real encoder writes and orchestrator behavior separately when changing them.
- Batch operation schemas derive from `BATCH_OP_SPECS` in `servers/contract.py`. Run `eval/mcp_batch_probe.py` and `eval/mcp_schema_gate.py` after schema or description changes, before restarting.
- Interaction resolver changes are guarded by `tests/test_interaction_defaults.py` and `tests/test_interaction_bypass_guard.py`. Host integration changes require `tests/test_host_contract.py` and `tests/test_hooks_manifest_sync.py`.

## Deployment

**Production runs only from `main`.** The designated production checkout must stay on `main`; never switch it to a feature branch or develop in it. Develop and commit in separate worktrees, validate against isolated databases, then merge reviewed changes into `main` before deploying.

- Before merging or deploying, verify the target checkout's actual branch, revision, working-tree changes, daemon source directory, and database. A directory name does not establish its branch. Do not include unrelated branch commits in a merge.
- A feature worktree isolates code, not the shared daemon or production data. Its MCP calls can still reach production; tests and evals must explicitly use isolated data.
- The daemon follows its configured directory, not the Git branch name. Updates there can become live on restart or a source-directory startup check. After deployment, verify the daemon's reported source directory and loaded-code fingerprint against the intended `main` tree.

A restart reloads the daemon's configured code tree; it does not install checkout edits into plugin caches. Check the target installation before deploying.

- `build-plugin.sh` packages tracked runtime files; ensure new runtime files are tracked before packaging.
- `redeploy.sh` refreshes the configured Claude plugin installation and the packaged Codex source, then restarts the daemon. Its destination is install-specific; inspect it before use on another environment.
- `scripts/codex-install.sh` builds and refreshes the Codex plugin cache for the selected `CODEX_HOME`.
- Daemon-side code needs a restart after the running tree is updated. MCP proxy changes (`servers/brain_mcp.py`), tool schemas, hooks, skills, and manifests require a fresh session after installation.
- Do not hold the maintenance lock during a deploy restart; it suppresses daemon startup.
