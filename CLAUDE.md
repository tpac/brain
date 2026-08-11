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

Same function at every scale. The unit doesn't know its scale. Δ from one scale feeds
another's O or K — S1E encodes a node (Δ) → S1R recalls it next session (O). There is no
separate inter-layer protocol. Shape and rationale: `docs/ARCHITECTURE-FRACTAL.md`.

## Architecture

**One owner per concern** — every concern has exactly one module that owns it. Reach it through that module's API, never around it.

**Route, don't reach** — about to write SQL outside `dal*.py`, or touch a table another layer owns? Walk it: what concern is this? which module owns it? does it expose what I need? Then call that. The walk is the rule; the prohibition is its shadow.

**A missing function is the finding, not the obstacle** — "the owner doesn't expose it" is the reason to add it there, not license to bypass. Add it to the owner, call it from here.

**Layers service layers** — the doors: traces → `brain_traces.py`, edges → `GraphDAL.add_relation`, S2 runs → `brain.run_s2()`, daemon spawn → `daemon_launch.py`, constants and limits → the contract file. S2 running SQL against trace tables is the canonical violation.

**Where a boundary already leaks, the rule is directional** — don't add new bypasses. The guardrail tests hold the line where it currently is.

**Know the owner before you write** — if you can't name which module owns the data you're about to touch, read the Map. Spatial certainty comes before code.

**Extend before creating** — a new module is a structural commitment. Justify it with a distinct responsibility, audience, or lifecycle. "It feels like a new thing" isn't one.

## Map

Where each concern lives. The module docstring is the detail — this table is the index.

| Concern | Code | Detail |
|---|---|---|
| Hooks (S0 observation points) | `hooks/hooks.json` | the manifest is the list |
| Daemon, launchd lifecycle, spawn | `servers/daemon_server.py`, `daemon_launch.py` | docstrings |
| Write topology, locks, batching | `servers/brain.py`, `db_backends/sqlite.py` | docstrings |
| Recall → surface → inject | `servers/brain_recall.py`, `recall_laf.py`, `scales/s1/surface*.py` | `docs/RECALL-OVERVIEW.md` |
| Frame (the deterministic prior) | `scales/s1/frame.py` | `docs/RECALL-OVERVIEW.md` |
| Encoding (S1 Scribe) | `scales/s1/scribe.py`, `encode.py` | `docs/ENCODE-ON-IDLE.md` |
| S2 units + coordinator | `servers/scales/s2/` | `docs/S2-DESIGN.md` |
| Suppression (state + fingerprint) | `scales/s2/rejection_table.py` | `docs/S2-DESIGN.md` |
| Aspects (roles for types/relations) | `servers/aspects.py`, `scales/s2/aspects_v1.json` | the JSON's `_schema` key |
| Corrections | `servers/brain_corrections.py` | docstring |
| Traces | `servers/brain_traces.py`, `trace_contract.py` | `docs/TRACES-LAYER-DESIGN.md` |
| Interactions (the K store) | `servers/interaction_seed.py` | the roster lives there |
| Scope provenance + the veil | `servers/scopes.py`, `scales/dispatch.py` | `scopes.py` docstring |
| Node + pipeline contracts | `servers/contract.py`, `pipeline_contract.py` | docstrings |
| Edge model | `servers/dal_graph.py` | `add_relation` docstring |
| Runtime flags | `hooks/scripts/brain-env.sh` | read at daemon start only |

All `scales/` paths live under `servers/`. Two databases: `brain.db` (nodes, edges,
embeddings) and `brain_logs.db` (traces, session state, interactions, errors).

## Conventions

- `additionalContext` is the only channel that reaches Claude.
- `encoding_source` is `category:process` — `anchor`, `encoder:sonnet`, `s2:<unit>`, `hook:<event>`, `migration:*`. Only `anchor*` can lock a node.
- Trace chains come from `SessionContext`: `s0-` / `s1r-` / `s1e-{session_short}-{stop}`; S2 uses `s2-{ts}-{unit}`.
- `SessionContext` is passed on every call — the brain owns no current session. Anything conversation-scoped is keyed by `session_id`, never a global `brain_meta` key. Two sessions run at once; ask whether one would clobber the other.
- `brain_batch` ops are closed: `remember`, `revise`, `connect`, `disconnect`, `archive`, `absorb`. Source of truth is `BATCH_OP_SPECS` in `servers/contract.py`; three consumers derive from it. Any schema or description change must re-run `eval/mcp_batch_probe.py` + `eval/mcp_schema_gate.py` before restart.
- Apply `output_config` on every round of an agentic loop, not just the last — round 1 can return unprotected text.
- Gate a new S2 unit's graph scan on its own `s2_<unit>_last_run_ts` in `brain_meta`, or it re-derives the same fixed point every cycle.
- Adding an aspect is a human edit to `aspects_v1.json` plus one `REQUIRED_ASPECTS` line. The encoder only routes strings into existing aspects; it cannot propose one.
- Scope provenance is stamped by `stamp_scope_provenance` and is never agent-authored.
- `brain.get_node()` walks corrections on every canonical pull and attaches `_corrections`. Forgetting corrections requires deliberately bypassing the canonical pull.
- A replay must pass its injected time via `get_frame(brain, at=...)` — nothing detects it automatically.

## Development Rules

### Time-window queries: route through `clock.iso_now()` / `iso_cutoff()`

**Never use SQLite's `datetime('now', ...)` against TEXT timestamp columns** — it returns space-separated timestamps, brain stores ISO-T, and the lexicographic mismatch silently corrupts `>` filters. Use `from .clock import iso_cutoff` and bind: `WHERE created_at > ?`. `julianday('now')` is fine — it returns a number.

**Use `iso_now()` for any new-row timestamp** (`created_at`, `updated_at`, `last_accessed`). `Brain.now()` and TraceDAL inserts route through it. Single source of truth for the write-side format (`'…+00:00'`).

**In S1/S2 code, pass `at=conversation_now(...)` explicitly.** S1/S2 reads/writes are conversation-time, not wall-clock. Eval replays inject historical `[Current date: ...]` prefixes; bare `iso_now()` / `iso_cutoff()` would anchor to today's wall-clock and silently corrupt the replay. System bookkeeping (log cleanup, integrity audits, dashboard counts) is exempt — wall-clock is correct there. `tests/test_time_window_contract.py` enforces both rules.

### Encoder prompts: DB is authoritative, sync to `.py` before committing

The live prompts for encoder agents live in the `interactions` table in `brain_logs.db`. The seed `.py` files listed in `SEED_PROMPTS` (`servers/tools/sync_prompts.py`) are **seed-only** — they bootstrap fresh brains that have no DB entry yet. They must mirror the **production-ACTIVE** version (not the highest registered) so a `git clone` inherits the prompt the runtime is actually using, never an untested dormant candidate.

**Discipline** for a normal prompt change:

```bash
register_interaction(name, template)         # registers as v(N+1), DORMANT
set_interaction_active(name, version=N+1)    # flips the runtime pointer
./dev sync-prompts                           # mirrors ACTIVE → .py files
./dev sync-prompts --check                   # CI-style non-zero-exit drift check
```

**Discipline** for an eval-gated prompt change: register DORMANT, run the eval, then activate + sync. Do **not** sync between register and activate — `sync-prompts` deliberately mirrors only the active version, so dormant candidates cannot leak into the seed file and be picked up by fresh-brain installs that skipped the eval.

Commit the `.py` change together with whatever prompted the registration. Never edit the `.py` files by hand to change prompt behavior — that won't affect runtime and will silently drift from the DB.

`tests/test_prompt_sync.py` holds the contract: each seed file must export `SYSTEM_PROMPT`, fresh brains must seed every prompt in `SEED_PROMPTS`, sync must mirror the active version (not the latest registered), and seed must never overwrite an externally-registered version.

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
- **`servers/*`** → daemon **restart** (`restart` MCP tool / `hooks/scripts/rebrain-daemon`); live this session.
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

**Backup before destructive DB operations** — `cp brain.db brain.db.bak-{timestamp}` before ANY delete, bulk update, or schema migration. No exceptions.

**Clean as you go** — Remove dead code the same session; delete it, don't mark it. Don't leave "TODO: remove later."

**Comments carry the why, not the history** — no dates, no dead symbol names, no "previously". That's the commit message.

**One concern per file** — If a function serves a different audience than the file's existing functions, it belongs in a different file.

### Key Development Rules

- Use MCP tools to interact with brain, not Python/bash scripts
- Don't manually run boot scripts (hooks handle this)
- Don't construct DB paths (read the boot output)
- **Never spawn `Brain(db_path=DB)` in a test/bench/eval script against the live `brain.db` while the daemon is running.** Two Python processes with their own writer connections will eventually corrupt an index. Instead: (a) stop the daemon with the maintenance lock `touch /tmp/brain-maintenance-{uid}.lock` and `launchctl unload`, (b) use `daemon_client.send_command` to dispatch through TCP, or (c) run against an `IsolatedBrain` copy under `tests/isolated_brain.py`.
- **Discussion IS the work** — do not touch Edit/Write tools during design conversations. Wait for an explicit go signal.
- **Trace the pipeline before changing it** — the decode→encode pipeline has coupled stages. Don't change one stage without understanding the full flow.
- **Encoding depends on decoding** — if the surfacer fails, the encoder gets no context. A broken decode pipeline silently breaks encoding.

**Deep, not wide.** Go all the way down on the thing you're changing — its tests, its callers, the doc that would otherwise lie, the real fix instead of the workaround. Don't widen to the adjacent problem you noticed; name it and move on. Completeness is finishing the cut, not enlarging it.

The tell you're going wide: you can't name who is hurt by leaving it. The tell you're stopping short: you can. Search before building. Test before shipping.

CARE about yourself, Challenge yourself, IMPROVE yourself.
