# Distribution Architecture Plan

Output of the 2026-08-11 architecture review of the distribution/deployment boundary:
`build-plugin.sh`, `redeploy.sh`, the manifests, `tests/test_deploy_contract.py`,
`hooks/scripts/` (boot/runtime/launchd/resolution chain), `servers/daemon_launch.py`,
`daemon_client.py`, `daemon_config.py`, and Phase 5 of `docs/DISTRIBUTION-READINESS.md`.
Five review angles (placement, unification, cohesion, coupling, altitude) over a fully
traced boundary — the boot chain, daemon lifecycle, and build/deploy chains were walked
end-to-end; nothing dynamic was left unresolved. Checked against settled decisions
(D-11/D-12, 5.0a "never auto-move data", the locked shape-scan principle id:22b3cfa4,
prune-before-unzip); no finding below contradicts one.

**Verdict on the existing Phase 5 queue: the order holds.** No symptom-before-cause
inversion. Two exceptions worth attention: (a) **5.0a as currently specced has two gaps**
(steps 3–4 below) and should absorb them as acceptance criteria before it is built;
(b) one **operator decision** (step 0) could shrink the entire plugin-rename-data-loss
class rather than netting this one rename.

**Dependency summary.** Step 0 is a decision that reshapes 5.0a and therefore precedes it.
Steps 1, 2, 5–10 are independent of each other and of Phase 5 sequencing — any can run in
its own session now. Step 3 precedes step 4. Steps 1, 5, and 6 shrink 5.1's scrub/export
surface, so they pay best before 5.1.

---

## Step 0 — DECISION (operator): default new brains outside the adapter namespace

**Problem.** New brains are created at `$CLAUDE_PLUGIN_DATA/brain` (`resolve-brain-db.sh`
step 5). `$CLAUDE_PLUGIN_DATA` is keyed to the *plugin name* — the thing D-11 says changes
for positioning reasons. D-11's own rationale ("a service name and data dir should never
change; coupling them makes a marketing decision trigger a data migration") is contradicted
by the default birth location. 5.0a as queued nets *this* rename; the next rename or an
uninstall/reinstall re-runs the fire drill, and "does CC wipe `$CLAUDE_PLUGIN_DATA` on
uninstall" stays load-bearing and unanswered.
**Target state.** New-brain creation defaults to the service namespace (XDG data dir, e.g.
`~/.local/share/brain/`), which never renames. Existing steps stay as read-only *adoption*
of brains found at the old locations. 5.0a shrinks from a permanent ladder feature to a
transition net for the two existing default-path installs.
**Files & call sites.** `hooks/scripts/resolve-brain-db.sh` (create branch, step 5);
`docs/DISTRIBUTION-READINESS.md` §5.0a.
**Verification.** Sandbox matrix from the 2026-08-08 5.0a test, plus fresh-install case
landing at the XDG path.
**Blast radius.** Changes where a stranger's brain is born — semi-irreversible product
decision; existing installs unaffected (never-auto-move holds).
**Depends on.** Nothing. Gates the 5.0a build.
**Respects.** Tom's "never auto-move data" (untouched — governs existing data);
the deferred runtime-relocation decision (opposite direction: runtime, not data).

## Step 1 — Delete the dev-machine install path from `servers/daemon_launch.py`

**Problem.** `debugger_friendly_python()` candidate 2 hardcodes
`~/.claude/plugins/marketplaces/local-desktop-app-uploads/brain/venv/bin/python` — one
machine's install channel inside the shipped, host-neutral service layer. The 5.0b gate is
structurally blind to it (its shapes are plugin.json-derived; this is an install path). On
public installs it's dead weight; on the dev machine it can run *repo* code under the
*plugin's* venv (interpreter/dependency skew). Flagged HIGH independently by two angles.
**Target state.** Candidate deleted. `$BRAIN_PYTHON` (exported on every launch path) and
`REPO_ROOT/venv` (which *is* the plugin venv when running installed) already cover every
legitimate case. If a cross-checkout hint is ever needed, read `PLUGIN_ROOT` from
`resolved.env` via a helper — never a literal.
**Files & call sites.** `servers/daemon_launch.py:debugger_friendly_python` (candidates
list). Callers unchanged.
**Verification.** `tests/test_daemon_recovery.py`; grep the tree for
`local-desktop-app-uploads` afterward (should hit only dev-local settings files).
**Blast radius.** Tiny — a fallback candidate; the two preceding candidates dominate.
**Depends on.** None — independent.
**Respects.** D-11 host-neutrality (this *strengthens* it).

## Step 2 — Make `BRAIN_DAEMON_PORT` a real contract: env-first in `daemon_config`

**Problem.** `brain-env.sh` documents `BRAIN_DAEMON_PORT` as the override and every
shell/hook client honors it — but `daemon_config.DAEMON_PORT` (the daemon's own bind, plus
`is_daemon_responsive`/`recover_daemon`) and `dashboard/daemon_client.py` compute the bare
formula, ignoring the env. Setting the documented override in `~/.config/brain/env` splits
the system: clients on one port, daemon on another, kickstart storms against a healthy
daemon. The formula is also copy-pasted in ~8 places.
**Target state.** `daemon_config.DAEMON_PORT = int(os.environ.get("BRAIN_DAEMON_PORT") or
47200 + uid % 100)`. Hook-side Python (`hook_common.py:305`, `post_tool_trace.py:93`,
`agent-bridge.py:18`) and `dashboard/daemon_client.py:13` drop local formulas for
env-var + constant (dashboard keeps an env-first read, no bare formula). Shell keeps
`brain-env.sh` as its one source (restart/brain/daemon-client.sh already defer or die in
step 5).
**Files & call sites.** `servers/daemon_config.py:48`; the four Python copies above.
**Verification.** `tests/test_daemon_recovery.py` (port-derivation assertions); a manual
`BRAIN_DAEMON_PORT` override smoke: daemon binds it, hooks reach it.
**Blast radius.** Default behavior identical (env unset → same formula). Risk only in the
override path, which is currently broken anyway.
**Depends on.** None — independent.
**Respects.** Contract-first constants doctrine; `brain-env.sh` stays the shell owner.

## Step 3 — One Python DB resolver, reading `resolved.env`

**Problem.** `resolve-brain-db.sh` is the declared owner of DB resolution and persists
`resolved.env` *specifically so non-hook consumers can find the brain* — yet `resolved.env`
has **zero Python readers**, and the legacy `~/AgentsContext/brain` fallback is
re-implemented in 7+ Python files in three divergent variants (`daemon_client.py:398`
`_relaunch_daemon`, `brain_mcp.py:1237` health monitor, `aspect_store.py:58`,
`tools/sync_prompts.py:275`, `dashboard/db.py:27`, `dashboard/queries/aspects.py:33`,
`hooks/scripts/idle_maintenance.py:11` — which doesn't even check `BRAIN_DB_DIR`;
`scripts/consolidation_edge_recovery/*` invert the owner's precedence). On any standard
install (brain at `$CLAUDE_PLUGIN_DATA`): the MCP health monitor's DAEMON_DOWN write
silently no-ops, daemon recovery can spawn a **fresh shadow brain** at the legacy path, the
idle-fires observability log is dark exactly where it's needed.
**Target state.** `daemon_config.resolve_db_dir()` (its Path Helpers section already owns
rendezvous paths): `BRAIN_DB_DIR` env → parse `~/.config/brain/resolved.env` → legacy path.
All servers/ + hook-Python + scripts/ sites call it. Dashboard, per its disconnection
contract, keeps ONE mirrored copy in `dashboard/db.py` (same three steps);
`dashboard/queries/aspects.py` derives from it. `idle_maintenance.py` logs under the
resolved dir (its wrapper already exports `BRAIN_DB_DIR`) — or folds into the hook logging
channel and drops the side-file.
**Files & call sites.** The 9 files above + `servers/daemon_config.py`.
**Verification.** New unit test: `resolve_db_dir` precedence incl. resolved.env parse;
existing `tests/test_daemon_recovery.py`; grep `AgentsContext` in servers/ + dashboard/ +
hooks/ afterward — remaining hits should be `resolve-brain-db.sh`/`boot-brain.sh` (the
shell owner) only.
**Blast radius.** Behavior change only where the old 2-step fallback was already wrong
(standard-location installs). Legacy installs resolve identically.
**Depends on.** None — independent. Precedes step 4.
**Respects.** Route-don't-reach; resolve-brain-db.sh remains the inference owner —
Python reads the *persisted contract*, it doesn't re-run the ladder.

## Step 4 — Close the daemon's frozen-DB-path hole (5.0a acceptance criterion)

**Problem.** The materialized launchd plists bake `__BRAIN_DB_DIR__` once (installers
no-op forever after `launchctl print` succeeds); `start-daemon.sh` requires the baked env
and never runs the ladder; `ensure_daemon` checks `source_dir` + code fingerprint but
**never db-path divergence**. Consequence for 5.0a as specced: after a rename the user
adopts via `userConfig.brain_path`, hooks resolve the new path — and the daemon +
dashboard keep writing the old brain forever (the old brain.db still exists per
never-auto-move, so the baked path stays "valid"). Two silently diverging half-brains —
the exact amnesia class 5.0a exists to close. Related: the installed plist is a frozen
snapshot generally (a future `ThrottleInterval` change never reaches installed machines).
**Target state.** (a) `start-daemon.sh` sources `resolve-brain-db.sh` (baked env becomes a
hint the ladder's fast-path confirms, not a verdict). (b) Daemon ping response includes its
`db_dir`; `ensure_daemon` treats mismatch vs the session's resolved dir like stale code —
kickstart. (c) Installers re-materialize the plist template to a temp file, diff against
the installed copy, re-bootstrap on drift.
**Files & call sites.** `hooks/scripts/start-daemon.sh`, `install-daemon-service.sh`,
`ensure-dashboard.sh`, `servers/daemon_server.py` (ping payload),
`servers/daemon_client.py:ensure_daemon`.
**Verification.** `tests/test_daemon_recovery.py` (extend: db-path mismatch triggers
kickstart; plist drift re-materializes); the 5.0a sandbox matrix must include an
adoption-while-daemon-running case.
**Blast radius.** Daemon lifecycle — the sensitive zone; review-before-commit with the
recovery tests as the gate. Diff is moderate.
**Depends on.** Step 3 (the daemon-side resolution it confirms against). Feeds 5.0a's
spec — land before or with 5.0a.
**Respects.** Never-auto-move (detection + restart, no data movement); D-11 (labels
untouched).

## Step 5 — Dead shipped scripts: surface for deletion, then gate the class

**Problem.** Zero-referrer files ship in every package: `encoding-hook.sh` +
`encoding_hook.py` (retired pre-Scribe encoding path; looks wired, isn't in `hooks.json`),
`brain-client.sh`, `daemon-client.sh` (a source-me lib with no sourcers),
`agent-bridge.py`, `extract-session-log.py` (referenced only by a contract test). Also
`scripts/seed_brain.py` — the ONE remaining hand-listed manifest file, and it has rotted:
zero runtime referrers; live seeding is `servers/seed_pack.py:seed_baby_brain` +
`interaction_seed.py`. Dead code ships to a public repo whose pitch is inspectability
(D-1). Five instances = a class, not litter.
**Target state.** (a) Operator confirms, then delete the dead six (git-recoverable);
`build-plugin.sh` drops the `scripts/seed_brain.py` line; `redeploy.sh` drops `scripts`
from its prune list. (b) New 5.0b-gate assertion: every shipped `hooks/scripts/*` file is
referenced by `hooks.json`, another shipped file, a plist, or a small external-wiring
allowlist (`brain-dashboard`, `brain-watch`, `brain-statusline.sh` — wired via user
settings). Shape-scan, allowlist names where wiring legitimately lives outside the tree.
**Files & call sites.** The six files; `build-plugin.sh`; `redeploy.sh:40`;
`tests/test_deploy_contract.py`.
**Verification.** The new gate assertion itself; full-suite run (import-surface change).
**Blast radius.** Deletion of unreferenced files — verified zero referrers; the risk is a
consumer outside the tree, which the allowlist mechanism is built to name.
**Depends on.** None. Do (a) before (b) or the new assertion fails on day one.
**Respects.** Deletion gated on operator look (recoverable-but-non-trivial rule);
seed-pack design (D-5) untouched — `seed_brain.py` is the *dead predecessor*, verify
against `seed_pack.py` before deleting.

## Step 6 — Unify the duplicated shell mechanisms (four small extractions)

**Problem.** Four mechanism-level duplications, each with demonstrated or imminent drift:
(a) API-key resolution (env-file source + dual-casing `CLAUDE_PLUGIN_OPTION` fallback)
copied verbatim in `boot-brain.sh:30-45` and `brain-env.sh:21-45` — forced by bootstrap
ordering, but only the key block is needed, not all of brain-env; a casing fix landing in
one copy re-creates the 2026-07-15 keyless-daemon failure. (b) plist-install ritual
(sed template → bootstrap → verify) copied between `install-daemon-service.sh` and
`ensure-dashboard.sh`, already drifted: only the daemon side re-verifies. (c) daemon boot
incantation written twice (`daemon_launch.daemon_argv` vs `start-daemon.sh`'s inline
heredoc — they already differ on env pinning, and the heredoc interpolates `$DB_PATH` into
Python unquoted). (d) `restart-daemon.sh` + `hook_common.daemon_call_raw` hand-roll the
TCP wire protocol that `daemon_client.send_command` owns (hook_common already imports
daemon_client — the decoupling excuse doesn't apply).
**Target state.** (a) `hooks/scripts/api-key-env.sh`, sourced by both — the
`runtime-state.sh` precedent (created to kill exactly such a five-site copy). (b) a
sourceable `launchd-install.sh` helper: template-materialize + bootstrap + verify, label
as argument; callers keep their policy (install-only vs ensure-up). (c) `servers/
daemon_server.py` gains `__main__`; `daemon_argv()` returns `[python, '-m',
'servers.daemon_server', db_path]`; `start-daemon.sh` becomes `exec "$BRAIN_PYTHON" -m
servers.daemon_server "$DB_PATH"`. (d) `hook_common.daemon_call_raw` wraps
`send_command` (keeps its stderr logging); `restart-daemon.sh` calls a
`python3 -c "from servers.daemon_client import send_command; …"` one-liner (the
boot-brain.sh pattern).
**Files & call sites.** As named; plus `dashboard/daemon_client.py` adopts newline framing
to match the wire owner (its copy stays, per the disconnection contract).
**Verification.** `tests/test_daemon_recovery.py`, `tests/test_run_hook_contract.py`;
manual: `./redeploy.sh` end-to-end on the dev machine (restart path), launchd reinstall.
**Blast radius.** Boot/lifecycle shell — moderate; each extraction is independently
committable; do (c) with extra care (both spawn paths must exec identically).
**Depends on.** None between them; (d) after step 2 avoids re-touching port lines.
**Respects.** D-11 (labels are arguments, not renamed); policy/mechanism split that
`daemon_launch.py` declares.

## Step 7 — Slim `boot-brain.sh`: move the boot diagnostics into `boot_brain.py`

**Problem.** 258 lines, ~13 concerns, three audiences (UX copy, install orchestration,
diagnostics), five `python3` forks inside a 15s hook budget. Two pieces are misplaced
outright: the hooks.json schema check (lines 159–179) is **unreachable by construction** —
boot-brain.sh is itself registered in the manifest it validates; if the wrapper were
malformed, CC would drop all hooks and the check never runs. The MCP import smoke
(229–255) is Python logic inlined in shell with a confused sys.path computation.
**Target state.** hooks.json schema check: delete from boot; its real home is a gate-style
contract test (shape family of `test_deploy_contract.py`). MCP import smoke + crash-sentinel
report: functions in `boot_brain.py` (the warm path always reaches the exec). Key
resolution: shared include from step 6a. Cold-install orchestration, heredocs, ensure_daemon,
exec — stay; that skeleton is the file's actual job.
**Files & call sites.** `hooks/scripts/boot-brain.sh`, `hooks/scripts/boot_brain.py`,
`tests/test_deploy_contract.py` (or a new hooks-manifest contract test).
**Verification.** `tests/test_run_hook_contract.py`; a real SessionStart on the dev
machine; assert the new contract test fails on a deliberately malformed hooks.json copy.
**Blast radius.** SessionStart path — visible immediately if broken; keep the diff staged
(delete dead check → move smoke → adopt include).
**Depends on.** Step 6a (the key include).
**Respects.** Hooks stay brain-env-routed; the 15s budget (this reduces forks).

## Step 8 — Derive `redeploy.sh`'s prune list from the package

**Problem.** The prune list (`for _d in servers hooks skills dashboard data scripts
.claude-plugin`) is a hand enumeration beside a build that derives from `git ls-files`.
Already drifted benignly (`data/` ships nothing). The biting direction: a new top-level
package dir won't be in the list → its deleted files orphan-drift across deploys — the
class prune-before-unzip was built to kill, and the third hand-list instance of the
locked shape-scan principle.
**Target state.** Prune set = top-level dirs of `brain.plugin` (`unzip -Z1 | cut -d/ -f1 |
sort -u`) minus the runtime allowlist (`venv py bin .runtime-ready` — the one enumeration
that's correct, because the package can't know it). Alternatively (cheaper): a 5.0b-gate
assertion `prune list ⊇ package top-level dirs`.
**Files & call sites.** `redeploy.sh:40` (or `tests/test_deploy_contract.py`).
**Verification.** `./redeploy.sh` end-to-end; deliberately add a scratch top-level dir to
a test zip and confirm it's pruned.
**Blast radius.** Dev machine only, but it *is* the deploy path.
**Depends on.** Step 5a (drops `scripts` from the list first, avoiding churn).
**Respects.** Prune-before-unzip (settled — this hardens it); id:22b3cfa4.

## Step 9 — Split `daemon_config.py` back to actual config (opportunistic)

**Problem.** The "pure config" module carries: an import-time `os.environ` mutation,
identity binding (`get_operator_name`/`get_agent_name` — a traces concern), and the code
fingerprint/worktree-detection mechanism (`_fingerprint_dir` walks + md5s every
`servers/**.py` at import, in every importer that wanted a constant).
**Target state.** Identity functions → the traces/identity side (or `servers/identity.py`);
fingerprinting + `_IS_WORKTREE` → `daemon_launch.py` (self-declared home of spawn/staleness
mechanisms; its consumer `daemon_client._code_changed` is the natural peer). Constants and
path helpers stay.
**Files & call sites.** `servers/daemon_config.py`, `daemon_launch.py`,
`brain_traces.py`, `daemon_client.py` imports.
**Verification.** `tests/test_daemon_recovery.py`, trace-identity tests; import-surface
change → full suite before merge.
**Blast radius.** Import graph shuffle; no behavior change intended.
**Depends on.** After steps 2–4 (they touch the same file; avoid conflicts).
**Respects.** One-owner-per-concern; 5.0c untouched (identity *values* stay config-driven).

## Step 10 — Hygiene batch (each trivial, bundle into any adjacent session)

- `dashboard/server.py:41` comment claims the dashboard plist pins `DASHBOARD_PORT` — the
  plist says the opposite; fix the comment (the map must not lie).
- Pin `Label == daemon_launch.LAUNCHD_LABEL` in `test_daemon_recovery.py`'s plist test
  (one line; makes the D-11 freeze executable).
- When 5.1 is built: denylist + scrub patterns live in ONE importable contract constant
  consumed by both the export script and `test_deploy_contract.py` (today the gate
  hand-copies the denylist; two enumerations of the same set the day 5.1 ships).
- `docs/DISTRIBUTION-READINESS.md`: fold inline CORRECTION blocks into corrected text,
  move §6/6b/6c session narratives to `docs/archive/` (doc stays the plan owner;
  current-state style per the standing docs rule). Low urgency — the doc never ships.
- `boot-brain.sh` "Searched locations" message re-derives the ladder and has rotted
  (omits 4b and `userConfig.brain_path`) — render from one source or trim to the doc
  pointer. Subsumed by step 7 if done there.

---

**Not queued, deliberately:** the launchd label strings in 5 places (frozen forever by
D-11, gated by shape-scan at rename — machinery to unify them protects a value that cannot
change); the dashboard port default (env-first everywhere, consistent); the statusline
`/tmp` literal (the fixedness is the documented contract); `ensure-runtime.sh` and
`resolve-brain-db.sh` internal structure (dense but single-responsibility);
`_is_worktree_checkout`'s git heuristic for restart authority (correct for single-copy
public installs; revisit only if install identity gets formalized).
