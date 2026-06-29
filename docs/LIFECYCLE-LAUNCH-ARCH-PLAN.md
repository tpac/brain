# Process Lifecycle & Launch — Architecture Plan

**Status:** Partially executed 2026-06-28 — the dashboard plist was templatized, the two dashboard
launchers were consolidated to `bin/brain-dashboard` (`start-dashboard.sh` removed), the dashboard
port was made env-configurable (`DASHBOARD_PORT` in `~/.config/brain/env`), `userConfig.brain_path`
was added, and the cold-install bootstrap was validated (Layers 1–3). The **launchd installer that
substitutes the plist tokens (Step 2) remains the main open item.** The steps below were the original
recommendations, each written to be executed cold in a separate session.

## Scope

**Topic:** how the daemon and dashboard are *spawned, supervised, recovered, and installed*.

**Boundary traced (Prong A — code):** `ensure_daemon` / `recover_daemon` / `_relaunch_daemon` /
`_launchd_kickstart` / `is_launchd_manages_daemon` in `servers/daemon_client.py`; `BrainDaemon.start`
/ `_run` / `_shutdown` + the internal supervisor in `servers/daemon_server.py`; the `_health_monitor`
watchdog in `servers/brain_mcp.py`; the two launchers `hooks/scripts/start-daemon.sh` +
`start-dashboard.sh`; `hooks/scripts/brain-env.sh` + `ensure-runtime.sh`; the two launchd plists; and
the deploy path (`build-plugin.sh`, `redeploy.sh`). Full spawner inventory + call graph mapped by an
Explore agent.

**Boundary recalled (Prong B — brain):** 4 recall queries (~30 nodes) on daemon lifecycle, recovery,
dashboard separation, and install/parity, plus 5 targeted code-verification greps.

**Coverage caveats — read these:**
- The **recovery/spawner core is settled and clean** — brain history *and* current code agree. I did
  **not** recommend changes there; doing so would re-litigate locked decisions (see "Not recommended"
  below). The live debt is concentrated in the **install/launch layer**, much of it exposed by this
  same session (the dashboard-singleton work, 2026-06-28).
- I ran the five review angles (placement / unification / cohesion / coupling / altitude) **inline
  rather than as a 5-agent fan-out.** Discovery showed the core settled and the live surface small and
  focused; a fan-out would have re-derived the one real finding from five directions or confirmed
  cleanliness I'd already established. This is a deliberate proportionality call, not skipped work.
- I did **not** deeply trace the 5 migration scripts' `launchctl` usage — judged one-off, not
  maintained runtime (see Step 3 note).

## Dependency summary

- **Step 1 → Step 2** (sequential): templatize the plists *before* wiring the install that materializes
  them.
- **Step 3** is independent and low-value — fold into Step 1's session or skip.
- Net: this is a **2-step plan** (+1 optional). Small on purpose.

---

## Step 1 — Templatize both launchd plists (and version-control the daemon plist)

**Problem.** Two coupled defects in the same place:
1. The **daemon plist is not in the repo at all** — it lives only at
   `~/Library/LaunchAgents/com.brain.daemon.plist`, hand-installed, un-version-controlled. The
   dashboard plist `hooks/scripts/com.brain.dashboard.plist` is the *only* tracked plist (added
   2026-06-28).
2. The tracked dashboard plist **hardcodes absolute user paths** (`/Users/tpac/brain`,
   `/Users/tpac/AgentsContext/brain` — confirmed at lines 21/27/39/42/45) and **ships** via
   `git ls-files hooks/`. So a clean end-user install would receive a plist pointing at *my* machine —
   actively wrong, not merely absent. This is the parity-violation class that
   `start-daemon.sh`-missing-from-manifest was (resolved 2026-06-15, id:722a4832); the plist contents
   are the unfixed remainder.

**Target state.** Both plists exist in-repo as **templates with placeholders**, never literal paths:
- `hooks/scripts/com.brain.daemon.plist` (NEW) and `hooks/scripts/com.brain.dashboard.plist` (edit
  existing) carry tokens like `__PLUGIN_DIR__`, `__BRAIN_DB_DIR__`, `__LOG_DIR__` in place of every
  `/Users/tpac/...` path.
- The daemon template **must preserve** its extra env block (`ORT_DISABLE_ALL_ACCELERATORS`,
  `ONNX_PROVIDERS`, `VECLIB_MAXIMUM_THREADS`, `PYTORCH_MPS_DISABLE`) — the dashboard template has only
  `BRAIN_DB_DIR` + `DASHBOARD_PORT`.
- Both keep `KeepAlive`, `RunAtLoad`, `ThrottleInterval=10`, and `ProgramArguments` →
  `__PLUGIN_DIR__/hooks/scripts/start-{daemon,dashboard}.sh`.
- Templates are the single source; the live `~/Library/LaunchAgents/*.plist` become *materialized
  outputs* (Step 2), never hand-edited.

**Files & call sites.** `hooks/scripts/com.brain.dashboard.plist` (edit → placeholders);
`hooks/scripts/com.brain.daemon.plist` (create from the current live
`~/Library/LaunchAgents/com.brain.daemon.plist`, then placeholder it). No Python call sites change in
this step.

**Verification.** `xmllint --noout hooks/scripts/*.plist` (well-formed). Grep both templates for
`/Users/` → must return nothing. Diff a manually-substituted daemon template against the current live
`~/Library/LaunchAgents/com.brain.daemon.plist` → must be byte-identical after substitution (proves
the template faithfully reproduces today's working plist).

**Blast radius.** Small and inert on its own — templates aren't consumed until Step 2. The one live
effect: the shipped dashboard plist stops carrying my paths. ~2 files, ~90 lines.

**Depends on.** None.

**Respects.** "Daemon managed by launchd, not hooks/MCP" (id:7e5d965d) — templates don't change *who*
owns lifecycle, only how the plist is sourced. Parity principle (id:76b2df70, locked value) — this is
the fix it demands.

---

## Step 2 — Add an idempotent, sentinel-guarded plist install to `ensure-runtime.sh`

**Problem.** **Nothing in the repo installs or loads the plists** (verified: no `launchctl
load|bootstrap|LaunchAgents` in `build-plugin.sh`, `redeploy.sh`, `ensure-runtime.sh`, or
`boot-brain.sh`). They are assumed pre-installed. On Tom's machine they were hand-installed once; a
fresh end-user install gets the shipped launcher scripts (good, since 2026-06-15) but **no launchd
agent**, so neither daemon nor dashboard ever starts. This blocks the cross-project / clean-install
goal (id:b6381ebf, locked).

**Target state.** `ensure-runtime.sh` — already the first-install bootstrap, already sentinel-guarded
via `$PLUGIN_DIR/.runtime-ready` (lines 13/33) — gains a **second, independent sentinel** (e.g.
`$PLUGIN_DIR/.launchd-installed` or `${XDG_CONFIG_HOME:-$HOME/.config}/brain/.launchd-installed`) that,
once after the venv is ready:
1. Resolves `__PLUGIN_DIR__` (from `$PLUGIN_DIR`), `__BRAIN_DB_DIR__` (from `resolve-brain-db.sh`),
   `__LOG_DIR__` into the Step 1 templates via `sed`.
2. Writes the materialized plists to `~/Library/LaunchAgents/`.
3. `launchctl bootstrap gui/$(id -u) <plist>` for each (modern form; `bootout` first only if a
   *different* definition is already loaded).
4. Touches the sentinel.

**Hard constraints (the part that makes this safe, not a boot-race):**
- **Idempotent + one-time.** Sentinel fast-path on every subsequent boot — this must NOT run per-boot.
- **Provision ≠ spawn.** It loads the launchd *agent*; it never `Popen`s a daemon. `ensure_daemon`
  remains the sole ping+kickstart coordinator. This is the line that keeps it compatible with the
  "do NOT add another spawner" rule.
- **No-op when already correct.** If a plist with equivalent content is already loaded (Tom's existing
  machine), do not bootout/bootstrap — never bounce a healthy daemon just to re-install an identical
  plist.

**Files & call sites.** `hooks/scripts/ensure-runtime.sh` (add the guarded block + a small
`_install_launchd_agent` helper). Possibly a tiny `hooks/scripts/install-launchd.sh` if the logic is
big enough to warrant its own file (cohesion) — decide at implementation time; ensure-runtime is the
natural home since it owns first-install provisioning and already resolves `$PLUGIN_DIR`.

**Verification.**
- **Open question to resolve FIRST:** does Claude Code's plugin-install mechanism already provision
  launchd agents? If it does, this finding narrows to "hook into that" rather than "install
  ourselves." Confirm before building (grep the plugin framework / ask).
- Fresh-install simulation: remove a sentinel + `launchctl bootout` a test-labeled copy → run
  `ensure-runtime.sh` → assert the plist materialized with correct substituted paths, `launchctl print`
  shows it loaded, daemon binds its port.
- Regression guard from the lifecycle work: **3 simultaneous boots → exactly one daemon process**
  (the invariant id:5d844297 / id:6e31bc41 were about). Run `tests/test_daemon.py`,
  `tests/test_daemon_recovery.py`, `tests/test_daemon_hooks.py`.
- Idempotency: second `ensure-runtime.sh` run is a no-op (sentinel) and does not bounce the daemon.

**Blast radius.** Highest in this plan — it touches first-boot provisioning of the
highest-blast-radius process. Mitigated by the sentinel + no-op-when-correct constraints. On *existing*
machines it must be a no-op. ~40–80 lines in one shell file.

**Depends on.** **Step 1** (needs the templates to materialize).

**Respects.** id:7e5d965d (launchd stays sole owner — install only provisions it); "do NOT add another
spawner" (CLAUDE.md — guarded by sentinel + provision-not-spawn); id:3328499d (must not run while the
maintenance lock is live — check the lock before any bootout/bootstrap); id:76b2df70 + id:b6381ebf
(the parity / clean-install goal this serves).

---

## Step 3 (optional, low value) — De-duplicate the launcher preamble

**Problem.** `start-daemon.sh` and `start-dashboard.sh` share ~17 lines of identical preamble
(`set -e`, `SCRIPT_DIR=...`, `source brain-env.sh`, the `BRAIN_PYTHON` executable check + FATAL
message). This duplication was introduced 2026-06-28 when `start-dashboard.sh` was written by mirroring
`start-daemon.sh`.

**Target state.** One of: (a) a sourced `hooks/scripts/_launcher-preamble.sh` both scripts source after
`brain-env.sh`; or (b) a single parametrized `start-service.sh <daemon|dashboard>`. (a) is the smaller,
clearer change — the two `exec` tails genuinely differ (daemon runs an inline `python -c
BrainDaemon(...).start()` + a `BRAIN_DB_DIR` check; dashboard execs the standalone script).

**Files & call sites.** `hooks/scripts/start-daemon.sh`, `start-dashboard.sh`, + new preamble file. If
the script *names/paths* change (option b), the two plists' `ProgramArguments` and `restart-daemon.sh`
must update too — which is why **option (a) is preferred** (no plist churn).

**Verification.** `tests/test_daemon.py` + `tests/test_daemon_hooks.py` still green; manual daemon +
dashboard restart both bind their ports.

**Blast radius.** Tiny — but touches the daemon's launch path, so any error fails the daemon. Marginal
benefit (17 lines). **Recommendation: skip unless touching these files anyway.** Flagged for honesty,
not urgency.

**Depends on.** None.

**Respects.** "Don't refactor what isn't broken" — hence low priority.

---

## Migration-script `launchctl` (noted, not a step)

5 one-off scripts hand-roll `launchctl load/unload`: `scripts/{archive_legacy_aspect_nodes,
backfill_edge_embeddings, migrate_trace_identity, reembed_edges_drop_meaning,
scrub_archived_edge_embeddings}.py`. The canonical safe-stop is the maintenance-lock pattern
(id:2eb161b3). But these are throwaway migration one-offs, not maintained runtime — routing them
through a shared primitive is low-value churn on dead-ish code. **Leave them.** If a *new, maintained*
tool ever needs to stop the daemon, it should use the maintenance lock, not copy these.

---

## Not recommended — and why (the brain's guardrails)

These surfaced during discovery and were **dropped** because they contradict settled/locked decisions
or are already shipped. Verified against current code, not taken on trust:

| Tempting change | Why dropped | Evidence |
|---|---|---|
| "Consolidate the 3+ daemon spawners" | Already done — launchd is sole owner; hooks/MCP only connect. | id:7e5d965d, id:81b3b0ad |
| "Unify the duplicated recovery code" | Already done — `recover_daemon()` is the single primitive in `daemon_client.py`. | id:7a7ee3ec (shipped 2026-05-29) |
| "Remove the `Popen` path in `ensure_daemon`" | Intentional **no-launchd fallback**, not debt. The launchd-sole-owner consolidation shipped (9a91f08); Popen is the deliberate remainder. | id:5d844297, id:1f45ffb5, CLAUDE.md |
| "Move lifecycle logic out of `daemon_client.py`" | That **is** its correct home by explicit decision. | id:3087d519 |
| "Move the health monitor into the daemon" | Must live *outside* the daemon process (external timer watchdog) — architecturally mandatory. | id:c219791d |
| "Make the dashboard route everything through daemon TCP (drop direct SQL)" | Deliberate reliability-over-purity tradeoff; acknowledged debt, kept on purpose. | id:5ddb6b7f |
| "Make the dashboard write / push live events" | Locked: dashboard is a passive async observer, never writes. | id:eb263281 (locked) |
| "Add back a suspend detector / dev-mode gate" | Removed 2026-06-25 as a net-negative (116 restarts/week); ping-based nets are sufficient. Code verified clean of residue. | id:ddc99931, id:eaec221c |

**Bottom line:** the lifecycle *machinery* is mature and battle-tested. The only real structural debt
is that we can launch the processes but can't yet **install** them reproducibly — Steps 1–2 close that,
and nothing else here is worth doing.
