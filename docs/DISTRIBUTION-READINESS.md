# Distribution Readiness — Sharing Anchor

**Status:** Install/launch layer **shipped + cold-validated (Layers 1–3)** as of
2026-06-29 — see **§6b**. Near-term target is **Goal A (hand it to a trusted friend)**;
remaining gates: depersonalize 3 non-s1e seed prompts (+ the s1e encoder prompt, owned by
another stream), relocate `aspects_proposed.json`, and the **untested Layer-4 live install**.
· **Started:** 2026-06-14 · **Audience:** dev-facing
**Placement:** lives in the **private** dev repo only. This document names personal-data
findings and internal file paths — it must **never** be copied into the public
distribution repo.

---

## 1. Goal

Make Anchor installable by someone who isn't Tom, on their own machine, to the
standard of a public open-source repo. Each new user gets their **own** brain
(their own daemon, their own DB) that becomes their own Anchor over time. We are
not sharing Tom's brain — we are sharing the *substrate* that grows one.

---

## 2. Locked decisions

These forks are settled. Future sessions should not relitigate them without a
stated reason.

| # | Decision | Rationale |
|---|----------|-----------|
| D-1 | **Full open source**, fresh public repo (clean history). | The tool's pitch is "trust me with your identity layer" — inspectable code *is* the credibility. Current repo history carries personal data and can't be the public one. |
| D-2 | **Repo separation, not develop-in-public, not a coupled mirror.** Private dev repo stays the daily driver and never goes public. A distinct public distribution repo carries only the clean shippable artifact, fed by a release step. | Tom: "I want separation." Develop-in-public imposes a permanent discipline tax (no personal commit ever again); a mirror imposes a sync tax. Separation keeps the working mess private at the cost of a deliberate release step. *(Open sub-question: does the public side carry full source history or just the built plugin — see §8.)* |
| D-3 | **Cross-platform v1 now, v2 deferred.** Ship the cheap ability to run on Linux (graceful degradation + first-class Popen fallback). Defer systemd parity / supervisor abstraction until a real Linux user exists. | v2's true cost isn't the build — it's the permanent obligation to validate the *most dangerous subsystem* (daemon lifecycle) on two OSes forever. Don't pay that tax for users who may not exist yet. |
| D-4 | **`userConfig` is additive, never a replacement.** Add the CC-native prompt-on-enable + keychain path for the API key, but keep the `~/.config/brain/env` fallback. | `userConfig` only exists inside the plugin runtime; the daemon running standalone / in Cowork still needs the env file. Best UX in-plugin, still works out-of-plugin. |
| D-5 | **Seed pack / persona design is its own session.** Approach: mine Tom's real brain for nodes that (a) *teach mechanisms* in detail and (b) make *good live encoder examples*, then genericize them into the shipped seed pack. | Whatever a stranger's Anchor wakes up as is a lasting artifact every new brain grows from — semi-irreversible. Deserves a dedicated design pass, not a find-replace. Overlaps with the encoder's few-shot examples (§4, Phase 1.2). |

---

## 3. Current state (grounded findings, verified 2026-06-14)

> Snapshot of the original (2026-06-14) investigation. Many findings below are now
> **resolved** — Linux degradation (§4 3.1), realchat untracked (§4 1.3), dashboard
> shipped (§4 4.1), userConfig (§4 2.1), and (2026-06-29, **§6b**) the dashboard
> launchd installer + path-clean package + `userConfig.brain_path`. The
> personalized-seed-prompts blocker is **partly** done — s1e encoder prompt is in
> progress on another stream; **3 non-s1e prompts + `aspects_proposed.json` still
> remain**. **§6b is the current status; read it first.**

The 2026-05-01 audit (`cd4a99f`) fixed hardcoded paths, the API-key gate, and
`.claude/settings.local.json`. **Those held.** New issues since:

**Blocker — the plugin is still personalized in its seed prompts.** These `.py`
files seed `interactions` on every fresh brain via `servers/interaction_seed.py`,
and `build-plugin.sh` ships all of `servers/`, so they're in the `brain.plugin`
artifact today:
- `servers/scales/s2/community_enrichment_prompt.py:12` — `"...persistent brain shared with operator Tom."` (and `:84`).
- `servers/scales/s1/encoding_prompt.py` — 57 "Tom" references, incl. a full
  worked example built on an ACL surgery / "Dr. Chen" / 5K-charity-run cluster
  (`:636` onward). These are **few-shot teaching fixtures** (they teach the
  encoder how to handle temporal clusters + corrections), not a DB dump — but
  they're Tom-named and seed into every fresh brain regardless.

**High — real conversation data is tracked.** `eval/longmem/data/realchat_sessions.json`
(5.6 MB, 27 real Tom↔Anchor sessions) is **not** gitignored.

**Daemon supervision is launchd-only and not actually installed.**
- ~530 lines of `servers/daemon_client.py` are built around `launchctl kickstart -k`
  (`_launchd_kickstart`, `_launchd_manages_daemon` — macOS `gui/<uid>` domain).
- **No `.plist` file is tracked in the repo, and nothing generates/installs one.**
  It exists only on Tom's machine, set up by hand. So every fresh install — Mac
  *or* Linux — already runs on the platform-neutral `Popen` fallback, with no
  `KeepAlive`/`RunAtLoad`.
- **Zero systemd anywhere.** On Linux the `launchctl` calls must degrade to
  `False` (not raise `FileNotFoundError`) or the lifecycle code breaks — needs
  verification.

**Dashboard is not shipped.** `dashboard/` is excluded from `build-plugin.sh`. A
plugin-only user (including Tom outside the repo) has no dashboard at all. It's a
read-only observer of the DBs + `/tmp/brain-surface-result-*.json`.

**First-run friction.** ~~API key missing → boot exits~~ **RESOLVED (a1a620e,
2026-07-15):** keyless boot now runs the full chain (runtime, embedder, brain.db,
daemon, traces, recall); `brain.llm_available` gates surface/encode/S2 with a
single logged marker, and the key heals the running daemon with no restart.
Remaining friction: embedder (~100–200 MB) downloads on first boot and can block
the 15s SessionStart hook.

**Low — "Tom" in shipped code comments.** ~30 files under `servers/` and `hooks/`
carry the operator's name in *comments* (e.g. `# Tom's convention: pass the object`,
`# Tom corrected prior framing`). Not a privacy leak and not behavior — but reads
as internal for a public repo. Cosmetic genericization → "the operator"; rides
with the D-5 prompt pass or Phase 5 polish. (Audit 2026-06-14.)

**Conventions — mostly idiomatic.** `.claude-plugin/plugin.json`, `.mcp.json`,
`hooks/hooks.json`, `skills/*/SKILL.md` all correct; uses `${CLAUDE_PLUGIN_ROOT}`
/ `$CLAUDE_PLUGIN_DATA`. The explicit 133-file allowlist in `build-plugin.sh` is a
feature (prevents leaking DBs/conversations) — keep it. Gaps: `marketplace.json`
is named `local-desktop-app-uploads` (a Cowork-upload dev artifact); API key is
hand-rolled vs `userConfig` (D-4).

---

## 4. The plan, by phase

Legend — **Reversible?** / **No-regret?** (worth doing even if distribution were
shelved) / **Risk** (blast radius) / **Effort** (S/M/L).

### Phase 0 — Safety net

**0.1 Encoding eval baseline.** Capture current encoding quality before any prompt
edit. Run the longmem frozen-corpus build + sweep and/or `eval/s1_encode_eval.py`
on the *current* prompts; save numbers.
- *Why:* the seed prompts are tuned; the few-shot examples are pedagogically
  load-bearing. Benchmark-first rule. This is the safety net for all of Phase 1.
- *Reversible:* additive · *No-regret:* **yes** · *Risk:* none (costs tokens — corpus encode) · *Effort:* S
- *Verify:* a saved baseline score we can diff Phase-1 prompt edits against.

### Phase 1 — De-personalize (the gate)

**1.1 Parameterize the operator name.** Replace literal "Tom" as *operator
identity* with a variable/placeholder across seed prompts (`community_enrichment_prompt.py:12,84`,
sweep all four seed prompts + `interaction_seed.py`).
- *Why:* "Tom" hardcoded in a shared prompt is already a latent correctness bug.
- *Reversible:* yes · *No-regret:* **yes** · *Risk:* low · *Effort:* S
- *Verify:* grep for operator-identity "Tom" in shipped `servers/` returns only
  legitimately-generic hits; eval (0.1) shows no regression.

**1.2 Neutralize personal specifics in encoder few-shot examples.** Minimal pass:
replace ACL/Dr. Chen/charity-run personal specifics in `encoding_prompt.py` with
neutral-but-still-instructive fixtures, so nothing personal ships. **The deeper
"design *great* teaching examples" work is the seed-pack session (D-5)** — this
overlaps; do the minimal neutralize now, the excellent redesign later.
- *Reversible:* yes · *No-regret:* only matters for shipping · *Risk:* **medium**
  (encoding quality — gated by 0.1) · *Effort:* M
- *Verify:* no personal narrative remains; eval (0.1) shows no regression.

**1.3 Relocate real conversation data.** Move `eval/longmem/data/realchat_sessions.json`
(+ any sibling real-session corpora) to a private location and gitignore the path.
**Move, do not delete.**
- *Reversible:* yes · *No-regret:* **yes** (privacy hygiene) · *Risk:* low · *Effort:* S
- *Verify:* `git ls-files | grep realchat` is empty; file still exists at the
  private location.

**1.4 Drift audit sweep.** Re-run the personal-reference grep
(`/Users/tpac`, `\btom\b`, `playbuzz`, `Pachys`, `AgentsContext`) across shipped
surfaces (excluding dev-only CLAUDE.md/docs/tests/eval) and triage anything new.
- *Reversible:* n/a · *No-regret:* **yes** · *Risk:* none · *Effort:* S
- *Verify:* every user-facing hit resolved or explicitly waived.

### Phase 2 — Onboarding correctness

**2.1 `userConfig` for the API key (additive — D-4) — DONE (2026-06-14).**
`plugin.json` declares an optional `userConfig.api_key` (`sensitive: true` →
keychain). `brain-env.sh` + `boot-brain.sh` gained an additive fallback: if
`ANTHROPIC_API_KEY` is still empty after the env-file/shell check, take it from
CC's injected `CLAUDE_PLUGIN_OPTION_API_KEY` (both casings checked — doc doesn't
pin `<KEY>`'s case). **Precedence: env-file/shell > userConfig** (existing setups
unchanged). Convention confirmed against plugins-reference (env-var + keychain).
**2.3 folded in here:** the "key not set" boot message now names *both* the
plugin-config prompt and the env file. Verified: uppercase/lowercase fill,
existing-key-wins, JSON valid, `bash -n` clean. End-to-end (CC injection +
keychain prompt) is verified-by-construction; needs a deploy + re-enable to
exercise live.
**Known limitation (code-review #2, 2026-06-14) — RESOLVED 2026-07-15.** The
prediction came true on the first laptop install: the daemon launchd installer
(Step 7, 2026-07-07) made every fresh install supervisor-spawned, so a
userConfig-only key never reached the daemon (`llm_unavailable` in the
dashboard while boot looked healthy). Fix: `boot-brain.sh` now MIRRORS a
userConfig-resolved key to `~/.config/brain/env` (mode 600, never overwrites
an existing key line) so `dispatch.load_env` resolves it on every spawn path;
the boot message says so honestly (keychain + mirrored file), and
`render_boot_v2` states the DAEMON's LLM state (`LLM layer: PAUSED`) inside
[BRAIN] when it boots keyless — the hook's view can no longer mask the
daemon's. The plist stays key-free (one plaintext location, not two).
Empirically confirmed same install: the userConfig prompt DOES appear and
inject on the claude.ai-upload channel.

**2.2 First-run embedder cost — DONE (2026-06-14) via pre-fetch at install (option B).**
Rejected option A (async `load_model` in a background thread) — it adds correctness
risk to the recall + baby-brain-seeding hot path for a one-time first-run delay.
Instead: `ensure-runtime.sh` step 4 pre-fetches the embedding model into fastembed's
cache during the (already-blocking) bootstrap, so the daemon's first `load_model()`
is cache-fast. Non-fatal (offline → daemon downloads on first recall, as before);
model name read from `plugin.json` (no drift). Verified at the command level
(extraction + cache-find both exit 0); NOT tested via a full fresh bootstrap
(would require nuking the runtime sentinel).

**2.3 Visible first-run state.** Surface "key missing / brain warming" through
`additionalContext` (the only channel that reaches Claude), not stdout.
- *Reversible:* yes · *No-regret:* yes · *Risk:* low · *Effort:* S

### Phase 3 — Cross-platform daemon (v1 only)

**3.1 Linux-safe `launchctl` degradation — DONE (2026-06-14): already handled.**
Both `_launchd_kickstart` and `_launchd_manages_daemon` already wrap the
`subprocess.run(["launchctl", …])` in `except Exception: return False`
([daemon_client.py:485](servers/daemon_client.py:485),
[:505](servers/daemon_client.py:505)) — `FileNotFoundError` on Linux is caught, so
they degrade to `False`. **No code change** (don't fix working code). Added a
regression test (`TestLaunchdHelpersDegradeWhenAbsent`, test_daemon_recovery.py)
locking the contract, since a narrowed `except` would silently break Linux. 35/35
pass.

**3.2 Popen fallback first-class — DONE (verified): already the cross-platform
baseline.** On Linux both helpers return `False` → `ensure_daemon` takes the
"launchd not managing → spawn directly" branch → detached `Popen`
(`start_new_session=True`, `ONNX CPUExecutionProvider`,
[daemon_client.py:333](servers/daemon_client.py:333)) that survives the hook exit;
`recover_daemon` routes through the same path for lazy crash-restart. Covered by
`test_no_launchd_falls_back_to_direct_spawn`. **Caveat:** verified by trace + unit
test, NOT on real Linux hardware — a real-Linux smoke is the remaining
confirmation.

**3.3 [DEFERRED v2 — D-3]** systemd user unit + plist, both auto-installed behind
a `supervisor` abstraction so `ensure_daemon` isn't hard-wired to `launchctl`.
Buys `RunAtLoad` warmth + immediate `KeepAlive` on both OSes. **Not this cycle.**

### Phase 4 — Dashboard access

**4.1 Dashboard access — DONE (2026-06-14): ship it + a launcher (not a daemon route).**
Trace showed the daemon is raw JSON-over-TCP (a `/dashboard` route would be real new
surface), while the dashboard is already a standalone threaded HTTP server **already
bound to `127.0.0.1`** ([dashboard/server.py:332](dashboard/server.py:332)),
daemon-proxying with a read-only SQLite fallback — just not shipped. So: `build-plugin.sh`
now ships the **git-tracked** `dashboard/` subtree (`git ls-files`, dev notes excluded)
+ a `bin/brain-dashboard` launcher mirroring `bin/brain-watch` (sources `brain-env.sh`
+ `resolve-brain-db.sh`, execs the standalone server). **No daemon change; the
127.0.0.1 binding was already correct.** Verified: test-build packs it junk-free,
launcher `bash -n` clean, dashboard imports under the venv (self-contained — no
`servers/` imports). Plugin ~575K→720K. A new user runs `brain-dashboard` from any repo.
*(Code-review corrections: the first cut used a `find` over the working tree, which would
ship a developer's untracked scratch/secrets and silently ship nothing if the dir moved —
switched to `git ls-files` (tracked-only, fails loud if empty). Also caught: the new
launcher was gitignored by `bin/*` — added `!bin/brain-dashboard` so it actually ships.)*

### Phase 5 — Packaging & distribution

**5.1 Real `marketplace.json`** (drop `local-desktop-app-uploads`; proper owner +
plugin entry). *Reversible · no-regret · S.*
**5.2 README + CONTRIBUTING to public standard.** *Reversible · no-regret · M.*
**5.3 Build hygiene** — confirm the allowlist ships nothing personal post-Phase-1.
*S.*
**5.4 [LAST — one-way door] Fresh public distribution repo + clean history.**
Only after everything above is verifiably clean. **Irreversible: git history is
forever; the current repo's history carries the personal data.** *L.*

---

## 5. Sequencing & dependencies

```
0.1 baseline ──► 1.1 / 1.2 (prompt edits gated by baseline)
                 1.3 / 1.4 (independent, anytime)
2.x onboarding  (independent of Phase 1)
3.1 ──► 3.2     (3.1 first; 3.3 deferred)
4.1             (independent; needs the 127.0.0.1 guard)
5.1–5.3 ──► 5.4 (publish LAST, after Phases 1–4 clean + seed-pack session done)
D-5 seed pack   (blocks a *polished* 1.2 and the quality of every new brain)
```

The publish (5.4) must not happen before the seed-pack session (D-5): a stranger's
first brain should wake up *well*, not with neutralized-placeholder fixtures.

---

## 6. Session log (2026-06-14) — executed + committed

**Decisions taken:** baseline → smoke-test now, full 20-item baseline deferred to the
head of D-5; **all** of 1.1 (operator-name + few-shot genericization) folded into D-5,
so prompt de-personalization happens once, holistically, with the baseline in hand.

**Executed — all committed in `0d2c1e7` (path-scoped on `main`):**
- **Plan doc** created (this file).
- **1.3** — `realchat_sessions.json` (5.6M) + `realchat_oracle.json` (598K) untracked
  (`git rm --cached`, kept on disk) + gitignored; public LongMemEval stays tracked.
- **1.4** — conclusive shipped-surface audit: no hard path/identity leaks (`cd4a99f`
  held; `~/AgentsContext` is `$HOME`-relative); "Tom Pachys" attribution stays (MIT).
  New low-sev finding: "Tom" in ~30 code comments (§3) → D-5/Phase-5 polish.
- **2.1 + 2.3** — `userConfig.api_key` (optional, keychain) + additive shell fallback
  (env-file/shell wins) + clearer first-run message. (Supervisor-daemon limitation: §2.1.)
- **2.2** — embedder pre-fetch at install (`ensure-runtime.sh` step 4; option B).
- **3.1 / 3.2** — Linux daemon: found *already handled* (launchctl absence caught, Popen
  fallback runs); added a regression test, **no code change**.
- **4.1** — shipped `dashboard/` (git-tracked subtree) + `bin/brain-dashboard` launcher.
- **Eval smoke-test** — 5/5 pipeline-health pass; harness good for the D-5 baseline.
- **Code-review pass** (max-effort recall) — 8 findings; **fixed 4**: the critical
  `bin/brain-dashboard` gitignore build-break, `find`→`git ls-files` (ship tracked-only),
  the stale daemon-mirror comment, and `python -c` path-interpolation hardening.
  **Deferred 4** (noted): #2 supervisor-daemon key gap (→ §2.1), cache_dir pre-fetch,
  eval FileNotFound on clone, launcher double-source.

**Deploy state:** committed but **NOT live** — hook/manifest/dashboard changes load from
the *built* plugin copy, so going live needs `redeploy.sh` + a new session (+ one plugin
re-enable for the userConfig keychain prompt). **Hold the redeploy until the tree
settles** — it rebuilds from the working tree, which still carries another stream's
in-flight dispatch refactor.

**Process learning — multi-session shared tree.** This work ran alongside another active
stream in the **same working tree**, and it bit twice: (a) `git add` / `git rm --cached`
staging got silently wiped by the other stream's `git reset`; (b) `git checkout -b` is
unsafe (shared HEAD would yank the other stream onto the branch). Working pattern that
held: commit **directly on `main`, path-scoped, atomically** (stage + commit in one shot,
no pre-stage-and-wait), with a guard that aborts unless the staged set is *exactly* this
session's files. See `0d2c1e7`.

**Not this session:** all prompt edits (→ D-5), Phase 5 packaging, the publish.

---

## 6b. Session 2026-06-28/29 — install/launch layer shipped; Goal-A path

This session built + validated the **install/launch layer** Phases 2–4 sketched, and
fixed the framing: **Goal A = hand it to a trusted friend** (reversible, the real proof —
do first) vs **Goal B = public OSS publish** (§5.4, the one-way door — later). Goal A does
**not** need the clean-history repo or public marketplace.

**Shipped + committed (main, path-scoped atomic commits):**
- `c6d4b82` **userConfig.brain_path** — bring-your-own-brain at enable. `resolve-brain-db.sh`
  reads `CLAUDE_PLUGIN_OPTION_BRAIN_PATH`; a not-yet-existing path is created (honored), not
  silently dropped. (Resolves the DB-path half of open-fork #3.)
- `b87f455` **dashboard: single launcher + env-configurable port + path-clean package** —
  consolidated to `bin/brain-dashboard`; port = `server.py` default 47303, overridable via
  `DASHBOARD_PORT` in `~/.config/brain/env` (the single user-editable place); plist
  templatized → **package has no `/Users/tpac`**. `/dashboard` skill added.
- `a11bbb9` **`/dashboard` self-installs the launchd singleton** (`hooks/scripts/ensure-dashboard.sh`):
  first run materializes the plist for the machine + `launchctl bootstrap`; KeepAlive/RunAtLoad
  thereafter. **This is §3.3's deferred launchd installer — now shipped for the dashboard**
  (the daemon still Popen-fallbacks; daemon launchd self-install is optional parity).
- `d165bce` Claude-pane open option (Claude-in-Chrome `navigate` at the singleton — no 2nd
  server); `4990a69` `dashboard-dev` preview config (separate port) — restores the
  screenshot/inspect UX-dev loop the singleton had cost; `20755f4` substitution-safe plist comment.
- `4bea8e8` `docs/LIFECYCLE-LAUNCH-ARCH-PLAN.md` (lifecycle arch-review plan).

**Cold-install validated — the 4-layer test (`f2a29b23`):** Layers 1–3 proven by isolated
probes — bootstrap (uv → Python 3.11 → venv → 6 deps → model, ~14s from zero), resolve
adopt-vs-create fork, `seed_pack` (16 generic nodes + 14 seed prompts), entry-point imports
(`servers.brain`, `brain_mcp`, `daemon_server`). **Layer 4 — live Claude Code (daemon boot +
MCP connect + hooks fire + recall/encode) — is UNTESTED. The key remaining unknown.**

**Goal-A checklist — status as of 2026-07-11 (all gates but the live test CLOSED):**
1. ~~Depersonalize seed prompts~~ **DONE** — s1e (other stream), s2_community_enrichment v22 +
   s2_aspects v3 (registered → activated → synced; `s2_aspects` added to `sync_prompts` SEED_PROMPTS).
   Every interaction template a fresh brain seeds is operator-agnostic; only 2 code comments in
   `surface_contract.py` still say "Tom" (never seeded).
2. ~~Relocate `aspects_proposed.json`~~ **DONE** — untracked + out of the package; runtime path
   resolves to `$BRAIN_DB_DIR`.
3. **Layer-4 live test — the one OPEN gate.** Partially proven in production 2026-07-08 (first
   friend install: boot, brain creation, seeding, daemon, MCP, PostToolUse traces all worked).
   Exposed + fixed: CLAUDE_PLUGIN_* vars are hook-execution-only (→ `resolved.env` persistence),
   silent no-brain bail (→ ANCHOR OFFLINE in `hook_common.daemon_unavailable_error`), seed
   idempotency dup bug (exact-title match). UNRESOLVED: friend's UserPromptSubmit/Stop hooks not
   firing on the ex.co machine — silent-gate (now impossible) vs CC plugin-hook bug
   (#10225/#29767/#53643); his reinstall with the current zip disambiguates.
4. **Friend install** — `brain.plugin` is fully self-contained (ships `.claude-plugin/marketplace.json`):
   unzip → `claude plugin marketplace add <dir>` → install → API key via userConfig. No repo access needed.
- Daemon launchd self-install shipped 2026-07-07 (Step 7, `install-daemon-service.sh`). Seed pack
  gained 3 exemplar nodes (correction/decision/lesson) + mechanism-staleness fixes; seeds 19 nodes.
  Seed-pack *quality* redesign stays D-5 (deferred).

---

## 6c. Bootstrap race (2026-07-17 laptop report) — fixed

Second laptop attempt produced a precise report (`archives/anchor-mcp-bootstrap-
race-report.md`): on a clean install the MCP spawn and the SessionStart hook both
ran the cold `ensure-runtime.sh` concurrently; one racer's tar overwrote `bin/uv`
in place while the other executed it → macOS SIGKILL (`Killed: 9`) → MCP
connection closed (CC never retries) → tool-less session; the hook racer burned
its 15s timeout bootstrapping → identity injection + key notice dropped. Fixes
(same session): mkdir-lock serializes the bootstrap (winner runs, losers wait on
the sentinel; stale-lock steal); uv extracted to temp + atomic `mv`;
`boot-brain.sh` detaches the bootstrap on cold installs and prints a static
warming notice instantly; `mcp-launch.sh` never bootstraps inline — kicks the
locked bootstrap detached and polls the sentinel ≤25s (connects in-session when
fast, exits cleanly for a next-session fast path otherwise).

## 7. Deferred / separate sessions

- **D-5 seed pack / persona design** — mine the brain for mechanism-teaching +
  good-encoder-example nodes; genericize into the shipped seed pack. Folds in a
  proper 1.2 redesign.
- **Zero-Memory boot block** — a conditional block `brain_assembly` injects at boot
  *only while the brain is sparse* (fresh install), dropped once the graph is rich
  enough that character emerges from it. Gate on an identity signal (e.g. count of
  locked identity/principle nodes), not raw node count. It can be *detailed* for free:
  when it fires the dossier is near-empty, so it doesn't contend for the 10k boot cap
  (the two are anti-correlated). Frame it as scaffolding that fades, not fixed
  character — it holds the anti-sycophancy axioms (useful-not-liked, hold-a-position)
  as bootstrap, while the epistemic-integrity stance (don't assert unchecked memory)
  stays always-on in SKILL.md, since a mature brain needs it *more*. Parked 2026-06-15
  during the SKILL.md instinct rewrite; sibling of D-5 (both = what a near-empty Anchor
  wakes up as).
- **Runtime relocation to `$CLAUDE_PLUGIN_DATA`** — QUEUED NEXT (Tom, 2026-07-17:
  "definitely intending to update often"). The runtime (`bin/uv`, `py/`, `venv/`,
  `.runtime-ready`) lives inside the install dir, so every plugin
  update/reinstall wipes it and forces a cold bootstrap (the exact race window
  §6c closed). Moving it to the survives-updates plugin-data dir makes every
  upgrade a warm ~8ms boot. Touches every `$PLUGIN_DIR/venv` reference
  (brain-env.sh, launchers, plists, ensure-runtime) — deliberate refactor, one
  variable at a time, after §6c proves out on the laptop.
- **3.3 cross-platform v2** — systemd + supervisor abstraction; trigger = first
  real Linux user.
- **Cowork support [DEFERRED 2026-06-14 — diversion from the Code goal]** — Tom
  deferred it: shipping to others via Claude Code is the goal; Cowork is platform-
  blocked and separable. **Findings to preserve** (June-2026 research, secondary-
  source, **UNVERIFIED**): (a) *plugin-manifest* hooks may not fire in Cowork
  (#27398 — settings resolution excludes plugin scope), and the brain is hook-driven
  (boot/recall/encode), so a plugin-install user's Anchor could be inert there;
  (b) the Cowork VM (local, Apple Virtualization) is network-isolated from the host
  — reaching a local daemon needs an HTTP bridge (supergateway), not a
  `BRAIN_DAEMON_HOST` one-liner; (c) the VM is ephemeral (no daemon persistence).
  **Contradicts** the 2026-06-12 conclusion (a4c4879d: "only client literals block
  Cowork sharing") — that described the server binding all interfaces, not the
  actual VM→host path or the hook situation, and was never tested. **But** Tom
  recalls hooks working in his Cowork — likely because his hooks are
  `settings.json`-scoped (which fire) vs plugin-manifest-scoped (which don't).
  **Open empirical question** (resolve via a live Cowork test before any work): do
  hooks fire in Cowork, and from which scope? Revisit when Anthropic ships
  plugin-hook support (#63360 pending).

- **5.4 the publish** — the irreversible step; gated on all of the above.

---

## 8. Open forks still to decide

1. **D-2 granularity:** does the public side carry the full (scrubbed) source with
   its own fresh history, or only the built plugin artifact + `marketplace.json`?
   "Full open source" (D-1) implies source, but the *mechanism* of separation
   (release step: squash-export? curated subtree push to a clean repo?) is unsettled.
2. **Seed identity (D-5):** what *is* a stranger's Anchor at birth — how much
   persona, how much blank slate?
3. **Operator-name mechanism (1.1):** runtime-detected (from CC user / git config)
   vs a `userConfig` field vs a generic constant. Affects 2.1.

---

## 9. Risk register

| Risk | Where | Mitigation |
|------|-------|------------|
| Silent encoding-quality regression | Phase 1 prompt edits | Baseline-first (0.1); diff every edit |
| Breaking every session today | `ensure_daemon` / spawn path (3.x) | Minimal touch; `test_daemon_recovery` + `test_keepalive` |
| Brain exposed on the network | Dashboard route (4.1) | Bind `127.0.0.1` explicitly; verify not LAN-reachable |
| Leaking personal data permanently | Publish (5.4) | Fresh repo, clean history, post-Phase-1 audit; do last |
| `userConfig` coupling the daemon to the plugin host | 2.1 | Keep env-file fallback (D-4) |
| Accidental data deletion | 1.3 relocate | Move, never delete; verify file exists at new path |
