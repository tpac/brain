# Daemon Lifecycle — Architecture Consolidation Plan

**Topic.** Concentrate the daemon lifecycle subsystem — start / bind / singleton-flock /
shutdown / teardown / restart / recover / spawn — so a future investigator sees it in one place
with the fewest moving parts and paths. Tom's framing: *"we keep hitting issues because it's
complicated, not because it's unsafe. Fewer dependencies, concentrated, not spread to different
places and different paths."*

**Boundary traced.** `servers/daemon_server.py` (1296L), `servers/daemon_client.py` (650L),
`servers/daemon_config.py` (190L), `hooks/scripts/{start-daemon.sh,restart-daemon.sh,boot-brain.sh}`,
the launchd plist, and the discovery-address callers across `brain_mcp.py`, `hook_common.py`,
`brain_cli.py`. Reviewed across five angles (placement, unification, cohesion, coupling, altitude).

**Coverage caveats.** The plist↔Python timing coupling is config, not call graph — traced by
reading both, not by static resolution. The `brain_tmp_dir()` fan-out (36 callers) was counted, not
each visited.

## Recommends only — nothing here is applied
Each step is executable cold in its own session. Recommend; do not batch. This doc is the artifact.

---

## Settled constraints (guardrails — a recommendation that undoes one is a false positive)

- **launchd is the SOLE spawner** where present (Errno-48 fix `5d844297`).
- **Four-mechanism recovery is intentional** — ensure_daemon + MCP health-monitor ping + launchd
  KeepAlive; do NOT collapse the two ping nets (`50c9a4e0`).
- **Liveness ≠ readiness** — cold-start slowness must not trigger restart; `_await_responsive`
  polls ~20s (`798380ea`).
- **flock is the sole singleton mutex; no stale flock; never unlink a held lock** (`79aa572`).
- **`_launchd_manages_daemon` treats indeterminate launchctl as "managed"** — uncertain ≠ absent
  (`a8199f2e`, `05812fd`).
- **Teardown releases the lock LAST** (after DB close) — two writers corrupt the indexes.
- **Worktree/non-source checkouts never restart the daemon** (`_is_daemon_source`, `9f72c7fb`).
- **The four SURFACES are a deliberate split** — boot-hook path, daemon-side, launchd plist,
  session-init — "fixes deploy to the surface that loaded them" (`95b3166e`). **The goal is
  concentrating the shared PRIMITIVES each surface calls, NOT merging the surfaces.**
- **The maintenance lock is the eval/DB-op safe-stop and MUST survive every step.**
  `is_maintenance_mode()` (`daemon_config.py:178`, `/tmp/brain-maintenance-{uid}.lock`) is honored by
  `ensure_daemon` (`daemon_client.py:232`) and `recover_daemon` (`:574`) to skip startup during
  VACUUM / schema changes / bulk deletes / evals. Any step that rewires `ensure_daemon`, the
  spawn/kickstart path, or `daemon_config` (Steps 2, 3, 8) MUST keep this gate — it is how a
  human/eval safely stops the daemon so a second process can open `brain.db`. Dropping it silently
  re-opens the two-writer corruption door the whole subsystem exists to keep shut.

## Already resolved (do NOT re-plan)

- **Restart = clean exit + KeepAlive respawn, launchd sole spawner** — commit `3cb6031`, live and
  verified. This closed the "fourth root" (`_do_restart` detached Popen, `e6fd63aa`) and the
  self-Popen-vs-kickstart double-mechanism. Agents that read the pre-`3cb6031` worktree flagged
  these as live; they are fixed on `main`. (The worktree has since been fast-forwarded to `main`.)
- **Step 3 RESOLVED (2026-07-06) — as a drift fix, NOT the unified primitive.** The tri-state
  design below would regress corpse recovery: its "managed+unreachable → defer to KeepAlive" arm is
  correct for `ensure_daemon` (down = process exited → KeepAlive respawns) but WRONG for
  `_relaunch_daemon`, whose target is the hung CORPSE — a live process KeepAlive can never respawn
  past; the manual kill-despite-managed is what produces the exit KeepAlive needs. The two ladders
  are role-distinct, not drifted copies. Shipped instead: `_relaunch_daemon` gained the missing
  re-ping rung (defers to a responsive incumbent instead of SIGKILLing it — the drift this step
  named), the role asymmetry is documented at both managed arms in `daemon_client.py`, and
  `test_kickstart_failed_but_incumbent_responsive_defers_never_kills` pins it. A1 (clean-exit on
  all platforms) evaluated and DECLINED: ~4 lines saved vs an unverifiable Linux behavior change
  (restart would leave the daemon down until a client pings). Post-Step-2 there is no remaining
  triple-written ladder: `_perform_restart` is two lines per branch, and `_relaunch_daemon`'s
  spawn tail already delegates to `ensure_daemon`. Step 6 may treat the spawn/restart model as
  settled: restart = clean-exit (managed) | teardown+spawn (unmanaged); recovery = kickstart →
  re-ping-defer → source-gated kill + ensure_daemon.
- **Step 4 RESOLVED (2026-07-07)** — dead `restart_daemon()` deleted (zero callers confirmed
  repo-wide, docs included). `stop_daemon` kept: it is the graceful TCP-shutdown utility, and its
  kill fallback is safe now that the lock unlink is gone (Step 5).
- **Step 5 RESOLVED (2026-07-07)** — `kill_daemon` no longer unlinks the lock file (kernel releases
  the dead PID's flock; only the stale PID hint is cleared). **No code path unlinks a lock file** —
  pinned by `TestKillDaemonLockDiscipline` (source + behavioral assert). "Confine to no-launchd"
  needed no further change: `port_is_occupied` is only reached in ensure_daemon's unmanaged arm,
  and `kill_daemon`'s remaining callers are that arm, `stop_daemon`'s fallback, and
  `_relaunch_daemon`'s corpse kill — which the Step 3 resolution established as deliberately
  NOT launchd-gated.
- **Step 6 RESOLVED (2026-07-07)** — `edcc749` + `f0a70bf`. (a) Deleted `_run`'s
  `is_daemon_responsive` pre-check: the flock (acquired in `start()` before the supervisor loop) is
  the singleton primitive — while we hold it no same-uid process is past the acquire, so none can be
  serving our port; the bind-time `EADDRINUSE` `DuplicateDaemonError` backstop (KEPT) covers the
  uid%100 residue + the acquire→bind race. (b) Phase-scoped the supervisor's `except Exception`:
  `self.brain is None` ⟺ a load-phase fault → exit for a fresh reload (KeepAlive / next
  ensure_daemon), no in-place retry; brain-up transient fault → warm-retry to MAX. (c) Fixed the
  crash-streak counter (`HEALTHY_UPTIME_RESET_S=300`) — the old reset-on-every-bind let an endless
  serve-crash loop never reach MAX. LOUD preserved. Pinned by `test_lock_holder_blocks_start_before_run`
  + `TestSupervisorPhaseScoping`.
- **Step 7 RESOLVED (2026-07-07)** — plist repo template + installer, mirroring the dashboard
  pattern (`adef11ae`). New `hooks/scripts/com.brain.daemon.plist` (`__PLUGIN_DIR__` /
  `__BRAIN_DB_DIR__` tokens) carries the FULL 5-var `DAEMON_CPU_ENV` — fixing the drift Step 1
  named (the hand-installed plist was missing `PYTORCH_ENABLE_MPS_FALLBACK`). **Deviation from the
  step text below:** the install step went to `boot-brain.sh` (a new dedicated
  `install-daemon-service.sh`, mirroring `ensure-dashboard.sh`), NOT `ensure-runtime.sh` — the
  latter can't get `BRAIN_DB_DIR` (it runs via `resolve-brain-db.sh`→`brain-env.sh` BEFORE
  `BRAIN_DB_DIR` is resolved, and sourcing resolve there would recurse). boot-brain.sh runs the
  installer AFTER `resolve-brain-db.sh` and BEFORE `ensure_daemon()`, so on a fresh macOS install
  launchd owns the daemon from boot instead of ensure_daemon direct-spawning a detached rival.
  Idempotent (skip if already `launchctl`-managed), macOS-only, non-fatal. Contract test
  `TestDaemonPlistTemplateContract` pins plist env == `DAEMON_CPU_ENV` and
  ThrottleInterval == `LAUNCHD_THROTTLE_INTERVAL_S`.

---

## Testing & isolation (applies to EVERY step)

The subsystem's failure history is corruption + restart storms, so verification is not optional
polish — it is how each step earns its merge. Three cross-cutting rules:

**1. The flock is NOT the eval-corruption guard — don't conflate them.** The singleton flock
(Step 11) guards **daemon-vs-daemon** (one daemon per user). It does NOT stop an eval/test/bench
process that opens the live `brain.db` *directly* — those processes never acquire the daemon lock.
That corruption class is governed by a *separate* discipline that these steps must PRESERVE, never
weaken:
  - `IsolatedBrain` (`tests/isolated_brain.py`) — copies the DB; the correct way to test against
    production-shaped data.
  - the **maintenance lock** — `touch /tmp/brain-maintenance-{uid}.lock` stops auto-restart so a
    human/eval can safely open the DB (see the settled-constraints entry).
  - TCP routing / `daemon_client.send_command` — mutate the live brain only *through* the daemon.
  A step that makes `ensure_daemon`/spawn ignore the maintenance lock is a regression even if every
  unit test passes.

**2. New integration tests are port/db-scoped — never the live daemon.** Current tests bind nothing
on the live `:47203` (they use mocks, `BrainDaemon.__new__`, `IsolatedBrain`, or host/port
overrides à la `test_hook_daemon_call_logging.py`). Any NEW test that spawns a *real* daemon MUST
use a test-scoped port + DB dir, so it can run in parallel (and alongside evals / the live daemon)
without racing the singleton flock or the shared `brain.db`. `tests/conftest.py` already refuses to
run outside the bundled Python — keep that guard.

**3. Restart-under-load must stay clean.** A restart while S2 maintenance / an encode is in flight
must drain the pool + bg-writer before releasing the lock (`_teardown_brain`'s ordering). Add an
explicit **restart-mid-S2** test (idle maintenance running → restart → assert clean teardown order,
no torn write) on top of the ordering already pinned by `test_write_txn_discipline`.

Per-step verification names the specific suite; these three rules are the floor under all of them.
The existing gates: `test_daemon_recovery` (62, mocked lifecycle), `test_write_txn_discipline`
(release-LAST / two-writer ordering), `test_maintenance_gate` (idle S2 gate), `test_system`
(hook-table + dispatch), `test_daemon` / `test_daemon_hooks` (worktree + hook paths).

---

## Dependency summary

```
Step 1 (config facts) ─┬─> Step 2 (daemon_launch.py) ─┬─> Step 3 (one spawn/kickstart primitive)
                       │                              ├─> Step 5 (confine kill/port; drop lock-unlink)
                       │                              └─> Step 6 (trim DuplicateDaemonError / supervisor)
                       └─> Step 7 (plist repo template)   [also uses Step 1 constants]

Independent (any time): Step 4 (delete dead restart_daemon), Step 8 (move brain_tmp_dir)

After the primitives are concentrated (Steps 1–3): Step 9 (split daemon_server god-file),
Step 10 (extract daemon_provenance), Step 11 (SingletonLock — widest blast radius, LAST)
```

Steps 1, 4, 8 are independent low-risk starters. Steps 9–11 are the largest diffs; do them only
after 1–3 land so they split *concentrated* code, not moving targets.

---

## Step 1 — Single-source the launch "facts" in daemon_config

**Problem.** The facts every spawn path needs are re-derived and have measurably drifted:
- **`DAEMON_PORT`** (`47200 + uid%100`) is re-derived in ~8 sites across Python and shell
  (`brain_mcp.py`, `hook_common.py`, `post_tool_trace.py`, `agent-bridge.py`, and 3 shell scripts) —
  `get_daemon_addr()` exists but the hook/script layer inlines the arithmetic.
- **The CPU-only env contract is split four ways and disagrees**: the plist sets 5 vars,
  `ensure_daemon`'s Popen sets 5 (incl. `PYTORCH_ENABLE_MPS_FALLBACK`), `daemon_config` sets 3 via
  `setdefault` at import, and the restart path sets none. A daemon started fresh vs. restarted vs.
  launchd-spawned runs with a *different* accelerator env — a latent SIGABRT/thread-spin surface.
- **Plist `ThrottleInterval=10` ↔ Python deadlines** (`_KICKSTART_DEADLINE_S`, `_GRACE_DEADLINE_S`,
  `SHUTDOWN_BACKSTOP_S=15`) is a load-bearing timing contract living half in XML, half in prose
  comments, single-sourced nowhere, asserted by no test.
- **`daemon.log` path** is derived twice with divergent `BRAIN_DB_DIR` handling.

**Target state.** In `daemon_config.py`: `DAEMON_CPU_ENV` (the one env dict), `get_daemon_log_path()`,
and named launchd-timing constants (`LAUNCHD_THROTTLE_INTERVAL_S=10`, `LAUNCHD_SIGKILL_GRACE_S≈20`)
with the Python deadlines DERIVED from them. One Python `DAEMON_PORT` constant + one exported
`BRAIN_DAEMON_PORT` in `brain-env.sh`; every discovery site *reads*, none derives.

**Files & call sites.** `servers/daemon_config.py`; the ~8 port-derivation sites; the env-set sites
(`daemon_client.py` Popen, `daemon_config` import block); `hooks/scripts/brain-env.sh`.

**Verification.** New `test_daemon_config` asserting one source for port/env/timing; grep shows no
remaining inline `47200 +`. A contract test that the plist's `ThrottleInterval` equals
`LAUNCHD_THROTTLE_INTERVAL_S` (pairs with Step 7).

**Blast radius.** Wide but mechanical (read-not-derive); no control-flow change. Medium diff.

**Depends on.** None.

**Respects.** Timing constants preserve the throttle/patience math (`c920f508`); env dict preserves
the CPU-only invariant.

---

## Step 2 — Create `servers/daemon_launch.py`: the one home for launchd + spawn primitives

**Problem.** "How the daemon gets spawned / killed / talked-to-via-launchd" is smeared across three
modules, and the daemon reaches *up* into the client for it:
- launchd primitives (`_launchd_kickstart`, `_launchd_manages_daemon`) live in `daemon_client.py`
  (nominally the *outside-caller* module), but `daemon_server._perform_restart` imports
  `_launchd_manages_daemon` — a layering inversion (server → client), done via a private cross-module
  import.
- `LAUNCHD_LABEL` is a third home (`daemon_config`); the `gui/{uid}/{label}` service target is built
  identically in two functions.
- Two detached daemon-spawn implementations exist — `daemon_client.py:333` (ensure_daemon fallback,
  hardened: `_debugger_friendly_python`, CPU env, `stdin=devnull`, zombie-port kill) and
  `daemon_server.py:992` (restart no-launchd branch, un-hardened: raw `sys.executable`, no env,
  no devnull). They drift silently in the least-tested (Linux/fresh-install) path.

**Target state.** New `servers/daemon_launch.py` owns the launchd + spawn surface as **public** API:
`LAUNCHD_LABEL`, `service_target()`, `kickstart()`, `manages()`, `_debugger_friendly_python()`,
`kill_daemon()`, `port_is_occupied()`, `daemon_argv(db_path)` (interpreter + startup command +
`DAEMON_CPU_ENV` from Step 1), and **one** `spawn_detached_daemon(db_path)`. Both `daemon_client`
and `daemon_server` consume it as public names — no private reach-through, no server→client import.

**Files & call sites.** New `servers/daemon_launch.py`; move the primitives out of
`daemon_client.py` and `LAUNCHD_LABEL` out of `daemon_config.py`; rewire `ensure_daemon`,
`_relaunch_daemon`, `_perform_restart`. Consider collapsing the `-c` heredoc into a real
`python -m servers.daemon_server` entrypoint so the startup command is a module main, not a string.

**Verification.** `tests/test_daemon_recovery.py` (re-point the `_launchd_*` patch targets to the new
module — mechanical rename). Add a test that both spawn callers use `spawn_detached_daemon` (one
hardened path). Import check: no circular import (functions consumed at call time).

**Blast radius.** Medium-high (moves ~6 functions + 1 constant, rewires ~4 call sites). The bodies
move verbatim — policy unchanged.

**Depends on.** Step 1 (uses `DAEMON_CPU_ENV`, `daemon_argv`, `get_daemon_log_path`).

**Respects.** launchd sole spawner; uncertain≠absent (`manages()` body unchanged); the hardened
spawn becomes the *only* spawn, fixing the un-hardened restart divergence without changing policy.

---

## Step 3 — Collapse the kickstart-else-spawn ladder into one primitive

**Problem.** The "kickstart if launchd-managed-and-reachable, else source-gated direct spawn, else
defer" decision is written **three times** and has drifted:
- `ensure_daemon`'s post-kickstart tail (`daemon_client.py:304`) — re-pings, defers to a responsive
  incumbent, then `manages()` probe, then Popen.
- `_relaunch_daemon` (`daemon_client.py:539`) — does NOT re-ping; goes straight to `_kill_daemon`.
  So `recover_daemon → _relaunch_daemon` can SIGKILL a daemon `ensure_daemon` would have deferred to.
- `_perform_restart`'s no-launchd branch (`daemon_server.py`) — a third partial copy.

Also, `kickstart()` returning False and `manages()` returning False now answer nearly the same
question (is launchd here?), producing an interleaved ladder that's the hardest-to-reason block in
the subsystem.

**Target state.** One `spawn_or_kickstart(db_path)` in `daemon_launch.py` built on a **single
tri-state launchd-presence probe** (`managed+reachable | managed+unreachable | unmanaged`) computed
once: managed+reachable → kickstart + await; managed+unreachable → defer to KeepAlive;
unmanaged → source-gated direct spawn. `ensure_daemon`, `_relaunch_daemon`, and
`_perform_restart`'s no-launchd case all call it. (Evaluate A1's stronger form: `_perform_restart`
becomes `_shutdown(); os._exit(0)` on *all* platforms and no-launchd respawn is delegated to the
next `ensure_daemon` — a behavior change on Linux worth a separate decision.)

**Files & call sites.** `daemon_launch.py` (new primitive), `daemon_client.py`
(`ensure_daemon`, `_relaunch_daemon`), `daemon_server.py` (`_perform_restart`).

**Verification.** `test_daemon_recovery.py` — the existing recover/ensure/kickstart tests must still
pass; add one asserting `_relaunch_daemon` defers to a responsive incumbent (the current drift).

**Blast radius.** Medium — touches the three most incident-dense functions. Careful; well-tested area.

**Depends on.** Step 2.

**Respects.** Four-mechanism recovery (both ping nets intact); uncertain≠absent; worktree-never-
restart (source gate preserved in the `unmanaged` arm).

---

## Step 4 — Delete (or reroute) the dead `restart_daemon()` footgun

**Problem.** `daemon_client.py:433` `restart_daemon()` (stop_daemon + sleep + ensure_daemon) has no
in-tree callers, yet is public. `stop_daemon → _kill_daemon` **unlinks the lock file** (violates
"never unlink a held lock") and has **no `_is_daemon_source` guard** (a worktree caller would SIGKILL
the shared daemon). A future caller reaching for the obvious name would break two settled invariants.

**Target state.** Delete it (recoverable in git) after confirming zero callers, OR reimplement as a
thin wrapper over `recover_daemon()` so it inherits the source-check + launchd serialization +
no-unlink guarantees.

**Files & call sites.** `daemon_client.py`; grep `restart_daemon(` across repo + hooks first.

**Verification.** Grep confirms zero callers; suite still green.

**Blast radius.** Tiny (dead code).

**Depends on.** None.

**Respects.** Removes a violation of "never unlink held lock" and "worktree never restarts."

---

## Step 5 — Confine `kill_daemon`/`port_is_occupied` to no-launchd; drop the lock-file unlink

**Problem.** `_port_is_occupied` + `_kill_daemon` (with its PID+lock-file unlink) date to when
multiple non-launchd spawners raced for the port. With launchd as sole spawner and clean-exit
restart, a competing PID-holder no longer arises on macOS — but the path survives, and
`_kill_daemon` unlinking the lock is safe only *implicitly* (it targets a daemon it just SIGKILLed).

**Target state.** Reachable only from the no-launchd spawn primitive (Step 2/3). Drop the lock-file
unlink entirely — rely on the kernel auto-releasing flock on the killed PID's death (the invariant
`start()` already documents). Result: **no code path unlinks a lock file.**

**Files & call sites.** `daemon_launch.py` (`kill_daemon`), `daemon_client.py` callers.

**Verification.** `test_daemon_recovery.py` flock tests; add one asserting `kill_daemon` never
unlinks the lock path.

**Blast radius.** Small; removes an implicit-safety reasoning burden.

**Depends on.** Step 2 (and ideally Step 3).

**Respects.** "never unlink a held lock" — this makes it structurally true.

---

## Step 6 — Trim the now-oversized guard surfaces (evaluate, don't reflexively delete)

**Problem.** Two guards have surfaces sized for a pre-clean-exit world:
- `DuplicateDaemonError` has three raise/catch sites. The `_bind_socket` EADDRINUSE backstop still
  fires on the genuine KeepAlive/kickstart overlap window and MUST stay. But the `_run` pre-check
  (`is_daemon_responsive` before bind) guards a state the flock `LOCK_EX` (acquired first in
  `start()`) already rejects — reachable only if a caller bypasses the lock, which no path does.
- The in-process supervisor loop (`start()`: retry `_run()` up to `MAX_SUPERVISOR_RESTARTS`) is a
  third respawn mechanism beneath KeepAlive + the ping nets; CLAUDE.md itself flags the boot-race.
  Its warm-brain no-reload restart is a real latency win, but it's an unscoped `except Exception`.

**Target state.** Keep the bind-time `DuplicateDaemonError` backstop; **evaluate** deleting the
`_run` pre-check after confirming no non-flock entry to `_run()`. **Evaluate** scoping the supervisor
to cheap transient faults (socket/pool) and letting brain-level faults exit to KeepAlive with a fresh
load. Both are "confirm-then-trim," not mechanical deletes.

**Files & call sites.** `daemon_server.py` (`_run`, `_bind_socket`, `start()` supervisor).

**Verification.** `test_daemon_recovery.py` duplicate + supervisor tests; add a test that a
lock-holder cannot reach the `_run` duplicate pre-check.

**Blast radius.** Medium; touches crash-recovery. Do carefully, with the four-mechanism picture open.

**Depends on.** Step 3 (the spawn/restart model must be settled first).

**Respects.** Four-mechanism recovery; does not remove the load-bearing bind-time backstop.

---

## Step 7 — Give the launchd plist a repo home (templatized) + installer

**Problem.** The single most load-bearing lifecycle artifact — the plist encoding `KeepAlive`,
`RunAtLoad`, `ThrottleInterval`, the `start-daemon.sh` entrypoint, the CPU env — exists ONLY as a
hand-installed `~/Library/LaunchAgents/com.brain.daemon.plist` with a hardcoded `/Users/tpac` path
and **no in-repo template or installer**. Daemon-side comments reference it as if it were a repo
file. A fresh install has no way to provision the LaunchAgent. The dashboard already solved exactly
this (`com.brain.dashboard.plist` template + `/dashboard` installer, `adef11ae`).

**Target state.** `hooks/scripts/com.brain.daemon.plist` templatized (`__PLUGIN_DIR__` /
`__BRAIN_DB_DIR__` placeholders, mirroring the dashboard) + an install step folded into
`ensure-runtime.sh`. `ThrottleInterval` and the CPU env come from the Step-1 constants (generate the
plist from them, or contract-test equality). Unblocks distribution.

**Files & call sites.** New `hooks/scripts/com.brain.daemon.plist`; `hooks/scripts/ensure-runtime.sh`;
the Step-1 timing/env constants.

**Verification.** Contract test: plist `ThrottleInterval` == `LAUNCHD_THROTTLE_INTERVAL_S`, plist env
== `DAEMON_CPU_ENV`. Fresh-install smoke (bootstrap the LaunchAgent from the template).

**Blast radius.** Independent; additive. Touches install flow, not the running daemon.

**Depends on.** Step 1 (for the constants it derives from). Otherwise independent.

**Respects.** launchd sole owner (this is where that policy is *encoded*); mirrors the settled
dashboard-singleton pattern.

---

## Step 8 — Move `brain_tmp_dir()` out of daemon_config

**Problem.** `daemon_config.py` declares "pure config" but carries `brain_tmp_dir()` — an
ephemeral-scratch-root helper for recall candidates / encoder prompts / dashboard reads, with ~36
callers across `scales/`, `hooks/`, `dashboard/`, `eval/`. It shares nothing with daemon lifecycle
except the file. A reader grepping `daemon_config` for "what governs the daemon" wades through a
data-plane helper.

**Target state.** Move `brain_tmp_dir()` to a data-plane home (e.g. `servers/paths.py` or beside the
ephemeral-file writers it serves). `daemon_config` keeps lifecycle constants + rendezvous paths only.

**Files & call sites.** `daemon_config.py`; ~36 import sites (mechanical import rewrite).

**Verification.** Grep confirms all callers rewired; suite green.

**Blast radius.** Wide but purely mechanical (import path change).

**Depends on.** None.

**Respects.** Cohesion (config module sheds a non-config concern).

---

## Step 9 — Split the `daemon_server.py` god-file (1296L) along the dispatch precedent

**Problem.** `BrainDaemon` tangles three concerns: **lifecycle** (start/bind/flock/shutdown/
teardown/restart), **request serving** (`_serve`/`_handle_client`/`_dispatch`/`_locked_exec`/
`_dispatch_hook`/`_accumulate_touched`/HOOK_TABLE), and **five background polls** (idle maintenance,
scribe poll + cooldown state, autosave, keepalive, warmup). An investigator asking "how does the
daemon start, bind, and tear down?" reads past request-handling and four poll loops; the flock story
alone spans `start()`, `_cleanup()`, and `_teardown_brain()`. This is the shape `daemon_dispatch.py`
was split for (1657→6 files).

**Target state.** Extract serving → `servers/daemon_serving.py`; background polls (+ their state/
flags) → `servers/daemon_background.py`. Fold the inline thread-inventory + native-CPU-spiral sampler
from `_autosave_loop` into `memory_watchdog.py` (the existing extracted watchdog home). The residue
in `daemon_server.py` IS the readable lifecycle unit: `__init__`/`start()`/`_run()`/bind/close/
signal/`_shutdown`/`_teardown_brain`/`_cleanup`/`_arm_force_exit_backstop`/`_signal_drain_shutdown`
+ the flock — one file, one story.

**Files & call sites.** `daemon_server.py` → +`daemon_serving.py`, +`daemon_background.py`;
`memory_watchdog.py` (absorb the sampler).

**Verification.** `test_system.py` (HOOK_TABLE integrity, hook dispatch), `test_daemon_recovery.py`,
`test_maintenance_gate.py`, `test_write_txn_discipline.py`. Full suite before merge (mixin/import
surface change).

**Blast radius.** Large diff (mixin extraction). Behavior-preserving mechanical move.

**Depends on.** Steps 1–3 (split concentrated lifecycle code, not a moving target).

**Respects.** All lifecycle invariants (code moves verbatim); mirrors the `dispatch_*.py` precedent.

---

## Step 10 — Extract `daemon_provenance.py` from daemon_client

**Problem.** The worktree-never-restarts policy is encoded across three predicates spread over two
files: `_is_daemon_source` + `_code_changed` (`daemon_client.py`) reading ping fields, plus
`_IS_WORKTREE`/`_is_worktree_checkout`/`REPO_ROOT`/`_CODE_FINGERPRINT` (`daemon_config.py`). An
investigator tracing "why won't my worktree restart the daemon?" bounces between three files.

**Target state.** `servers/daemon_provenance.py` owns the "is this checkout the daemon's source, and
is the running code current?" predicate family. `daemon_client` keeps transport + orchestration.

**Files & call sites.** `daemon_client.py`, `daemon_config.py`, `brain_mcp.py` (ping-field consumer).

**Verification.** `test_daemon_recovery.py` worktree/code-changed tests; the `test_daemon.py`
worktree tests (`056c7a40`).

**Blast radius.** Medium (moves a predicate family + its config bindings).

**Depends on.** None strictly, but do after Step 2 so the module boundaries settle together.

**Respects.** Worktree-never-restart policy (predicates move verbatim); content-fingerprint scope
(`a74a3ac5`).

---

## Step 11 — One `SingletonLock` owner for the flock (widest blast radius — LAST)

**Problem.** The flock — the brain's only real write-safety mutex — is four fragments: the path in
`daemon_config` (`get_lock_path`), the acquire (with "never unlink a held lock") in `start()`, the
release ("LAST, after DB close") in `_cleanup`/`_teardown_brain`, and a SEPARATE flock on the same
file in `ensure_daemon` (serializing restart decisions). No single place states the mutex and its
rules; the invariants that make it correct are spread across ~120 lines in two files.

**Target state.** A small `SingletonLock` (own file, or a section of `daemon_launch.py` — flock +
launchd together ARE the exclusion layer) owning path + non-blocking acquire (with pid-hint) +
idempotent release + the "never unlink / release LAST / LOCK_NB-means-live-holder" rules as comments
on the code that enforces them. `start()` uses it for singleton identity; `_teardown_brain` for
release-LAST; `ensure_daemon` for restart serialization.

**Files & call sites.** `daemon_server.py` (`start`, `_cleanup`, `_teardown_brain`),
`daemon_client.py` (`ensure_daemon`), `daemon_config.py` (`get_lock_path`).

**Verification.** `test_daemon_recovery.py` flock/duplicate tests; `test_write_txn_discipline.py`
(the release-LAST ordering that prevents two-writer corruption).

**Blast radius.** Widest — touches the acquire/release on the path that guards all brain writes. Do
LAST, most carefully, with the two-writer-corruption history open.

**Depends on.** Steps 2, 3 (the launch/spawn model settled first).

**Respects.** flock-sole-mutex, no-stale-flock, never-unlink-held-lock, release-LAST — all preserved,
just co-located and self-documented.

---

## Recommended execution order

1. **Start independent + mechanical:** Step 1 (config facts), Step 4 (delete dead `restart_daemon`),
   Step 8 (move `brain_tmp_dir`). Any order, separate sessions.
2. **The core concentration:** Step 2 (`daemon_launch.py`) → Step 3 (one spawn/kickstart primitive).
   This is the spine — it's where "fewer paths, one home" is actually won.
3. **Trim on top of the primitive:** Step 5 (confine kill/port), Step 6 (evaluate guards).
4. **Independent, unblocks distribution:** Step 7 (plist repo template) — any time after Step 1.
5. **Cohesion splits (largest diffs, last):** Step 9 (daemon_server god-file), Step 10
   (daemon_provenance), Step 11 (SingletonLock — the most careful, guards all writes).

The highest value-per-risk is **Steps 1–3**: they collapse the drifted launch contract and the
triple-written spawn ladder into one home and one path — which is exactly the "see it all in one
place" Tom asked for — without touching any load-bearing guard. Steps 9–11 make it *pretty* (one
readable lifecycle file) but are large diffs; they should follow, not lead.
