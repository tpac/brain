# Distribution Readiness — Sharing Anchor

## §ACTIVE ARC (2026-08-14) — arch-plan steps 0–6 SHIPPED (7–10 open); next: 5.0 attempt 2 (thread launched, plan-gated), 5.0c
**Read first:** handoff node `[thread:d13-arch-plan]` + the 5.0a ruling (id:cfe2113b). Shipped
2026-08-11: arch-plan steps 2+4+5. Shipped 2026-08-12 (5.0a, three-lens reviewed): new brains born
at `${XDG_DATA_HOME:-~/.local/share}/brain`; adoption net refuses to create over an orphaned
plugin-data brain and boot renders a self-contained adoption notice (Flow B); the
`~/.config/brain/env` `BRAIN_DB_DIR` knob is a ladder rung in BOTH resolvers (knob-with-brain
adopts; db-less knob = birthplace hint, never beats a found brain — the split-brain guard);
rename matrix automated in `TestAdoptionNetAndXdgCreate`. **Locked:** D-13, never-auto-move,
Flow B, installer identity guards, ladder hint-demotion. **Open:** live
adoption-while-daemon-running proof (needs second laptop or prod-daemon pause — uid-keyed /tmp
discovery files collide). **Do not reopen:** bare env-first port (needs the env-file fallback),
ladder step-1 bare `-d` adoption, cp-before-bootout, flat reference-count gating,
verified-developer signing (deferred to 5.x polish), knob-dir-exists adoption in Python (the
2026-08-12 split-brain finding).

**Status:** **Goal A closed** — the Layer-4 live install worked end-to-end on a clean
machine 2026-07-17 (§6b). Now on **Goal B: the public OSS publish** (Phase 5.7, the one-way
door). Naming + release model settled 2026-08-06 (**D-6…D-9**, §10); §8 fork #1 is
closed. **D-5 (seed pack)** remains the open design gate; **Phase 1.5** (shipped
`examples/` leak) was found and closed by deletion. **2026-08-09:** D-11 (two
identities — the service layer never renames) and D-12 (every instance name derives
from config) added. **5.0 fleet gap: first attempt reverted after review — still
open**; 5.0a/5.0b/5.0c are the new prerequisites in front of the rename
(**5.0b shipped 2026-08-11** — `tests/test_deploy_contract.py`).
**Claims audited against live code 2026-08-08** — the first such pass. Four false
claims and three rotted `file:line` citations were corrected; the stale §3
("Current state", a June snapshot) was deleted outright. Section numbering skips 3
by design. **Cite code by symbol, not `file:line` — the line anchors in this doc
rotted silently once already.**
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
| D-2 | **Repo separation, not develop-in-public, not a coupled mirror.** Private dev repo stays the daily driver and never goes public. A distinct public distribution repo carries only the clean shippable artifact, fed by a release step. | Tom: "I want separation." Develop-in-public imposes a permanent discipline tax (no personal commit ever again); a mirror imposes a sync tax. Separation keeps the working mess private at the cost of a deliberate release step. *(Sub-question CLOSED 2026-08-06 → D-7 / D-8: full scrubbed source, runtime + tests, squash-exported.)* |
| D-3 | **Cross-platform v1 now, v2 deferred.** Ship the cheap ability to run on Linux (graceful degradation + first-class Popen fallback). Defer systemd parity / supervisor abstraction until a real Linux user exists. | v2's true cost isn't the build — it's the permanent obligation to validate the *most dangerous subsystem* (daemon lifecycle) on two OSes forever. Don't pay that tax for users who may not exist yet. |
| D-4 | **`userConfig` is additive, never a replacement.** Add the CC-native prompt-on-enable + keychain path for the API key, but keep the `~/.config/brain/env` fallback. | `userConfig` only exists inside the plugin runtime; the daemon running standalone / in Cowork still needs the env file. Best UX in-plugin, still works out-of-plugin. |
| D-5 | **Seed pack / persona design is its own session.** Approach: mine Tom's real brain for nodes that (a) *teach mechanisms* in detail and (b) make *good live encoder examples*, then genericize them into the shipped seed pack. | Whatever a stranger's Anchor wakes up as is a lasting artifact every new brain grows from — semi-irreversible. Deserves a dedicated design pass, not a find-replace. Overlaps with the encoder's few-shot examples (§4, Phase 1.2). |
| D-6 | **The product is `entity`; the identity stays `Anchor`.** Plugin name `entity`, marketplace name `anchor`, repo `tpac/entity`, MCP server key stays `brain` (→ tools read `mcp__plugin_entity_brain__*`). | Three layers, each named for what it is: **Entity** = the category the product grows (the terminal noun — id:9da43311, id:e6019012: "a thing that develops, not a thing that's used"); **Anchoring** = the method; **Anchor** = the instance. `brain` stays the organ, so the tool names keep the substrate/identity split visible. *Rationale corrected 2026-08-09: an earlier draft said Anchor "stays Anchor's own name rather than being spent on a registry string" while naming the marketplace `anchor` — self-contradictory, since a marketplace name is exactly a registry string. Resolved by Tom: `anchor` in the catalog is the **act**, not the instance — you install an entity by anchoring it, so `entity@anchor` reads as the philosophy rather than spending the name. The instance name lives in config (D-12), never in a manifest.* Rejected: `brain` (the organ — repeats the register failure Tom named in id:cd7aa9be, and every competitor on the shelf is already `*mem*`/`*memory*`), `cairn` (imported metaphor, not native to the philosophy), `colleague`/`tenure`/`cultivar` (more self-explanatory but each shrinks the claim). |
| D-7 | **Squash-export on release.** Private `tpac/brain` stays the daily driver and never goes public. The public repo is a **build output**: each release materializes the shipped tree into a clean checkout, one commit, tag `vX.Y.Z`, push. | Closes §8 fork #1. Keeps the working mess private at the cost of a deliberate release step (D-2). **The one rule: the export manifest IS the build manifest** — `build-plugin.sh` already derives from `git ls-files` because a hand-list rotted 62 files behind reality and shipped a broken `brain_mcp.py`; a second hand-maintained list would re-enter that failure class with the public repo as blast radius. Public tree = plugin manifest ∪ chosen extras − explicit denylist. |
| D-8 | **Public tree = runtime + tests. `eval/` excluded.** The 6 test files that import `eval/` degrade to graceful skip. | Tests are the credibility argument (D-1: inspectable code); a no-tests repo reads as unverified for a tool asking to hold someone's identity layer. But a published eval harness **is a claim** that invites strangers to re-run and dispute it — not a launch-day fight, and the corpora are personal anyway. **Consequence: the README must make no benchmark claims**, since the harness won't be there to back them. Coupled files: `test_eval_corpus.py`, `test_longmem_classifier.py`, `test_absorb_preservation.py`, `test_consolidation_examples.py`, `test_encoder_eval_probes.py`, `run_all.py`. |
| D-9 | **Issues only at launch; no PRs.** Stated in CONTRIBUTING. | A squash-export pipeline can't cleanly merge an inbound PR, and a contributor's commits would never appear in history (reads as uncredited). With one maintainer, a PR you can't merge is worse than one you never invited. Revisit if real contributors appear — opening up later is easy, closing down isn't. |
| D-10 | **Public launches at `v0.9.0`** — not `9.6.0`, not `1.0.0`. | Tom 2026-08-06: *"It's not complexity that reflects the version, it's the function of value. Not yet v1."* The private `brain` plugin's 9.6.0 is an internal build counter that means nothing to a stranger. v1 is a claim about delivered value, and Anchor hasn't earned it publicly yet. **No collision with the private install** — `entity` and `brain` are distinct plugin names, so their version lines are independent. `plugin.json` and `marketplace.json` must both read `0.9.0` (5.1 asserts it). |
| D-11 | **Two identities: the host-neutral layer never renames.** `brain` stays the name of the *service* — launchd labels, `~/.config/brain/`, `BRAIN_*` env vars, both DBs, the MCP server key. Only the **Claude Code adapter** (plugin name, marketplace, permission strings, skill prefix) becomes `entity`. | A plugin has one name because a plugin is one thing; we ship a **local service with a CC adapter on top** — daemon, two DBs, embedder, dashboard — so the names sit in three namespaces with different owners (CC registry / launchd / XDG) and, decisively, **different change frequencies**. A marketplace name changes for positioning reasons; a service name and data dir should never change. Coupling them makes a *marketing* decision trigger a *data migration* — which is exactly the `$CLAUDE_PLUGIN_DATA` amnesia hazard. Forced by the second-host goal (ChatGPT takes remote MCP only — no manifest, no hooks, no skills — so nothing in `plugin.json` ports). **Consequences:** the "launchd labels — change now or never" hazard is **gone** (we don't rename the service), and 5.2 shrinks from 28 files to ~7. |
| D-12 | **Every instance name derives from config, never a literal.** `BRAIN_AGENT_NAME` is the single source for what an entity is called; `BRAIN_OPERATOR_NAME` for its counterpart. No shipped file hardcodes `Anchor`. | Identity is *accumulated, not issued* (id:e6019012) — shipping a pre-named identity contradicts the product's own thesis on install day and turns an entity into a persona. Today **182 lines across 43 files** hardcode `Anchor`, worst `seed_pack.py` (45) which seeds a stranger's brain with nodes asserting "I'm Anchor" before it holds a single memory. The config slots already exist and are wired to exactly one consumer (trace stamping) — nothing renders them. **This merges §8 #3 into one workstream:** "traces have no speaker" and "the name is hardcoded" are the same defect — slots that ship empty and unread. **Backward compatibility is a non-issue:** Tom's instance genuinely *is* Anchor, so his 8.5k nodes stay correct; only the shipped default changes. |
| D-13 | **The brain lives outside every host's namespace, at one configurable location.** New brains default to `${XDG_DATA_HOME:-~/.local/share}/brain/`; never inside `$CLAUDE_PLUGIN_DATA` or any other host-owned directory. One configurable location holds everything (both DBs, aspects, logs); one persisted record (`~/.config/brain/resolved.env`) is the cross-runtime contract, and **every** consumer — shell hooks, servers/, dashboard, the launchd-spawned daemon — resolves through it. Existing brains at old locations are adopted read-only, never moved (5.0a). | Decided 2026-08-11 (Tom): *"I don't mind disconnecting the path from the name of the plugin and it actually makes sense if we think of portability to other AI systems — I don't want it to be in that folder at all"* and *"It needs to be a single configurable location for everything."* Completes D-11: the data dir joins the service namespace that never renames. Kills the plugin-rename-data-loss **class** (not just the 5.2 instance), makes the CC uninstall-wipe question moot, and pre-positions the ChatGPT/second-host goal — a host-independent daemon must find its data without any host's plugin layout. The 2026-08-11 architecture review (docs/DISTRIBUTION-ARCH-PLAN.md steps 0/3/4) found the enforcement gaps this decision closes: 7+ Python re-implementations of resolution, zero Python readers of resolved.env, and a daemon whose baked DB path is never re-verified. |

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

**1.5 `servers/scales/s1/examples/` — DELETED 2026-08-08.** Found during the claims
audit. Ten files (2,589 lines) shipped to every install carrying **165
operator-name lines**, including `a3_anchor_sees_tom.py` — the operator's name in a
*filename*. Not comments: reconstructed real operator↔Anchor conversations authored
as encoder few-shot data, one docstring noting preserved verbatim profanity.

**What they actually were:** the *authoring source* for the §7.6 block that is now
baked into `encoding_prompt.py`. Pipeline was dicts → `render_compressed` →
`eval/agent_introspect/v20_assembly.py` → pasted into the prompt. All stages frozen
2026-05-24/25 and never touched again.

**The decisive finding — source had diverged from the live prompt.** The
depersonalization pass ran on the *prompt*, not the source: live §7.6 reads "Sam"
(×17, they/them) while the source still read "Tom" (×165). Regenerating §7.6 from
these files would have silently **undone the depersonalization and re-shipped the
operator's name**. They were not merely unused — they were stale in a dangerous
direction.

**Resolution: deleted outright** (operator call, over exclude-from-build or
relocate-to-`eval/`). Recoverable from git if the D-5 seed-pack session ever wants
the `counterfactual_bad` / `voice_annotations` / `choice_points` authoring data.
`eval/agent_introspect/v20_assembly.py` deleted with them — a one-shot v20 assembly
script that cannot run without its input and that nothing imports.
- *Result:* shipped manifest **240 → 230 files**; operator-name lines across shipped
  files **165 → 52**. No `build-plugin.sh` exclusion needed. Entry-point and
  `eval.agent_introspect` imports verified; full suite run.
- *Note:* the live prompt still carries the operator's verbatim
  `"fuck. yeah. how did you see that."` attributed to "Sam" — retained
  deliberately (nothing identifying, and it is the load-bearing evidence that the
  seeing landed).
- *Why this was missed for so long:* §6b's "every interaction **template** a fresh
  brain seeds is operator-agnostic" is true, and it read as "Phase 1 is closed." It
  wasn't — the templates were clean, an entire shipped directory was not.

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

**3.1 Linux-safe `launchctl` degradation — DONE: already handled.** Both
`daemon_launch.kickstart()` and `daemon_launch.manages()` wrap their
`subprocess.run(["launchctl", …])` so a missing binary degrades to `False`:
`kickstart` via `except Exception: return False`, `manages` via an explicit
`except FileNotFoundError: return False` ("no launchctl binary = no launchd
platform → spawn directly"). **No code change** (don't fix working code). Added a
regression test (`TestLaunchdHelpersDegradeWhenAbsent`, test_daemon_recovery.py)
locking the contract, since a narrowed `except` would silently break Linux.

**3.2 Popen fallback first-class — DONE (verified): already the cross-platform
baseline.** On Linux both helpers return `False` → `ensure_daemon` takes the
"launchd not managing → spawn directly" branch → detached `Popen`
(`start_new_session=True`, `ONNX CPUExecutionProvider`, in
`daemon_launch.spawn_detached_daemon`) that survives the hook exit;
`recover_daemon` routes through the same path for lazy crash-restart. Covered by
`test_no_launchd_falls_back_to_direct_spawn`. **Caveat:** verified by trace + unit
test, NOT on real Linux hardware — a real-Linux smoke is the remaining
confirmation.

*(Both helpers were renamed and moved out of `daemon_client.py` after this was
written — cite them by symbol, not `file:line`. Line-anchored citations in this
doc have rotted once already.)*

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
+ a `hooks/scripts/brain-dashboard` launcher mirroring `hooks/scripts/brain-watch`
(sources `brain-env.sh` + `resolve-brain-db.sh`, execs the standalone server).
**No daemon change; the 127.0.0.1 binding was already correct.** Verified: test-build
packs it junk-free, launcher `bash -n` clean, dashboard imports under the venv
(self-contained — no `servers/` imports). A new user runs `brain-dashboard` from any repo.
*(Code-review correction: the first cut used a `find` over the working tree, which would
ship a developer's untracked scratch/secrets and silently ship nothing if the dir moved —
switched to `git ls-files` (tracked-only, fails loud if empty).)*

*(Launchers later moved from a top-level `bin/` into `hooks/scripts/`: claude.ai-hosted
plugins reject `bin/` executables, and `bin/` is now gitignored runtime-only space for
the fetched `uv`. The earlier `!bin/brain-dashboard` gitignore fix no longer exists.)*

### Phase 5 — Packaging & distribution

Ordered execution checklist as of 2026-08-06. Every naming/model decision is closed
(D-6…D-9); only **5.6 (D-5 seed pack)** is still a design question.

**5.0 Plugin updates reach existing installs — BUILT 2026-08-14 on branch
`claude/keen-heisenberg-27bdb0`; pending merge + daemon restart.** The gap it
closes: seeding is create-only (`_register` no-ops once a name exists), so an
install froze at first boot and **no prompt improvement ever reached anyone who
had already installed**. Measured: the 8 shipped prompt files took 31 commits in
90 days, reaching only fresh brains. Proven on the author's own machine — its
`boot` config still carried `tom_quotes_limit` four months after the `.py`
renamed that key.

Publishing with this open means every install frozen at whatever quality shipped
that day, with the fix getting harder per install.

**The prompt-content half of this section is obsolete (2026-08-23).** The
interaction collapse dissolved it: code owns every prompt/config default
(`servers/interaction_defaults.py`), the DB holds only overrides, so editing a
default reaches every install at the next daemon restart. `interaction_seed.py`,
`sync_prompts.py`, and `SEED_PROMPTS_VERSION` are deleted. What remains live
below is the **structural** migration runner (`brain.db` / `brain_logs.db`), which
is still unbuilt — see BACKLOG item 3.

**The shape.** Not a deploy script — *code owns the defaults; each install
migrates itself forward at open*, the same contract `BRAIN_VERSION` already has.
Three version streams through **one** runner, `run_versioned_migrations` in
`servers/schema.py`, with separate counters so structure and prompt content move
independently:

| Stream | Counter | Ladder |
|---|---|---|
| brain.db structure | `brain_meta.brain_schema_version` = `BRAIN_VERSION` | `MAIN_MIGRATIONS` |
| brain_logs.db structure | `logs_meta.logs_schema_version` = `LOGS_VERSION` | `LOGS_MIGRATIONS` |
| shipped-prompt content | `logs_meta.seed_prompts_version` = `SEED_PROMPTS_VERSION` | the reconcile |

Both ladders are empty; a change adds one `(version, fn)` entry and bumps its
counter. **The runner owns the stamp** — it re-reads the version, so anything that
stamps ahead of it silently skips every step. It writes the stamp only after all
steps pass, backs up any non-fresh DB with pending steps (including a
pre-versioning DB at version 0), rolls back and leaves the stream unstamped on
failure so the next open retries, and distinguishes fresh (structural test: no
tables) from pre-versioning (version 0 *with* tables) — the first is baselined,
the second runs the whole ladder.

**The prompt reconcile** (`reconcile_seeded_prompts`, `servers/interaction_seed.py`)
advances one of the 8 template-carrying prompts only while the install still runs
the shipped default, carrying template **and** `parameters` together so a frozen
install cannot keep a dated model ID the mechanism is unable to fix.
Pristine-ness is derived, never separately stamped: the `interaction_active`
pointer must have been set by the system (`register:auto_v1` or `seed:reconcile`,
written in the same statement as the pointer) and every version above active must
be a previous reconcile's crash residue. The moment a human registers or
activates anything for a name, it is hands-off permanently — `trace_recording`
sits at active=1 with a dormant v2 and must never be published over.
`migration:initial_active` is reserved but deliberately **not** pristine: it
points a missing pointer at `MAX(version)`, which on a pre-split install can be a
human's own version. Those three values are refused at the MCP door, or a stray
call could relabel a deployment decision as an untouched default.

Called from `daemon_server._load_brain` only — never `Brain()`, which eval
corpora, `IsolatedBrain`, tests, and the daemon-dead `boot_brain.py` fallback all
construct and which must never be mutated.

**Live constraints for anyone extending this:**
- **`SEED_PROMPTS_VERSION` 1 is burned.** A reverted first attempt booted on real
  installs and stamped it; the code went away, the row did not. The counter
  starts at 2. Before re-attempting any reverted work, check a copy of a real DB
  for rows the reverted code wrote and decide per row whether the new code means
  the same thing by them.
- **No version floor ships, and none should be added casually.** 606 of 657
  `brain.db` files on this machine are below v30, including the daemon's own
  retained backups and 52 frozen eval corpora. A floor must lag the
  backup-retention horizon, and that relationship belongs in a test, never a
  comment.
- *(Obsolete — the prompt-content stream is gone; see the note at the top of 5.0.)*
- **What a version bump costs, measured 2026-08-14.** The pre-migration backup is
  *not* the hazard it was assumed to be: `shutil.copy2` of the live 723 MB
  `brain.db` takes **0.121 s** and the 816 MB `brain_logs.db` **0.133 s**, because
  APFS clones copy-on-write. Against a watchdog budget of ~20 s (`brain_mcp.py`
  `FAILURE_THRESHOLD` 10 × `PING_INTERVAL` 2.0 s) that is three orders of
  magnitude of margin, and the pre-serve window is already dominated by the
  embedder load. The real constraints are (a) the **migration step itself** —
  a row-by-row Python loop over `trace_events` would blow the window with the
  port closed, which is what the sub-second rule is actually about, and (b)
  **disk**: every bump leaves a `.vN.bak` that nothing prunes.

Deploy: `servers/*` only, so a daemon restart. No `redeploy.sh`.

**5.0a Rename safety net + XDG create — SHIPPED 2026-08-12.** The rename moves
`$CLAUDE_PLUGIN_DATA` (per-plugin), so a brain at the default path goes invisible.
The 2026-08-08 sandbox matrix showed step 4b rescues only when `resolved.env` is
present and current; missing/stale meant a silently created fresh brain.
**Shipped shape (Tom ruled Flow B, 2026-08-12):**
- **New brains are born at `${XDG_DATA_HOME:-~/.local/share}/brain`** — the
  create branch no longer targets `$CLAUDE_PLUGIN_DATA` at all; plugin-data and
  legacy rungs are adoption-only.
- **The adoption net:** before creating fresh, the resolver scans the live
  `$CLAUDE_PLUGIN_DATA`'s sibling dirs and `~/.claude/plugins/data/*/brain/` for
  an orphaned `brain.db`. Candidate found → **refuse to create** (empty
  `BRAIN_DB_DIR`, every consumer's existing no-brain path) and export
  `BRAIN_ADOPTION_CANDIDATE`; the boot hook renders a self-contained notice
  (candidate path + exact adopt/fresh-start commands, answerable without any
  lookup) that repeats each session until the user sets the knob. Refusal lives
  in the resolver so the launchd daemon refuses identically; boot is only the
  renderer. Never creates, moves, or deletes.
- **The `~/.config/brain/env` knob is now a shell-ladder rung too** (it was
  Python-only): `BRAIN_DB_DIR=` there adopts a brain outright, or — dir without
  `brain.db` — is honored as the explicit birthplace choice, beating both the
  net and a stale plist-baked hint. This is the adoption mechanism the notice
  points at (plus `userConfig.brain_path`, which fills process env in hook
  contexts).
- Python resolvers (`daemon_config.resolve_db_dir`, dashboard `_brain_dir`)
  mirror the new tail: XDG-with-brain → legacy-with-brain → XDG default.
**Tests:** `tests/test_daemon_recovery.py::TestAdoptionNetAndXdgCreate` (shell
ladder: fresh-install-at-XDG, net refusal, sibling scan, knob-beats-net,
knob-beats-stale-hint, CPD adoption) + `tests/test_db_resolution.py` (Python
tail). *M.*
**Acceptance criterion shipped 2026-08-11 (arch-plan step 4):** the daemon can no
longer diverge from an adopted path — `brain-daemon` re-runs the resolution
ladder on every launch (the plist-baked `BRAIN_DB_DIR` is a fast-path hint, not a
verdict), the daemon ping reports its `db_dir` and `ensure_daemon` kickstarts on
mismatch, and the installers re-materialize + re-bootstrap the plist when it
drifts from the template. `tests/test_daemon_recovery.py` locks all three.

**5.0b Deploy-contract gate — DONE 2026-08-11, `tests/test_deploy_contract.py`.**
Executable answer to "how do I know every place a deploy touches." A hand-list
already failed twice (the 62-file build manifest; the 5.2 permission entry above).
**Shape-scan, never an enumeration** — the same reason `build-plugin.sh` derives
from `git ls-files`:
  1. `plugin.json.version == marketplace.json` version — drift is silent and breaks
     `/plugin update`; it happens every release, forever
  2. adapter-name containment — for name `N` from `plugin.json`, every occurrence of
     `mcp__plugin_N_`, `com.N.`, `<owner>/N` must sit in a small allowlist; a new
     file hardcoding it fails automatically because the test reads the *tree*
  3. host-neutrality (D-11) — `servers/` may reference the CC manifest only in an
     embedder context (`embedder.py` exempt wholesale, everything else line-checked)
  4. name derivation (D-12) — capital `Anchor` appears only in the config allowlist
*As built: the `com.N.` sub-check SKIPS while `N` = `brain` — with adapter name ==
service name it cannot distinguish D-11-legitimate labels from leaks; it arms
automatically at the 5.2 rename. Assertion 4 is `xfail(strict=True)` until 5.0c —
when 5.0c completes it XPASSes and the marker must come off. Known tradeoff: while
xfailed, a NEW `Anchor` literal produces no signal; 5.0c closes that window.
The gate excludes its own file (the scanner must name the shapes it hunts) and the
5.1 denylist (docs/, eval/, `CLAUDE.md`, `tests/archive/`, `tests/results/`).
Found and fixed on first run: `tests/test_trace_chain_lane.py` hardcoded
`/Users/tpac/brain`.* *Shipped S.*

**5.0c Name / identity consolidation (D-12) — OPEN. Absorbs §8 #3.** 182 `Anchor`
literals across 43 shipped files resolve to `BRAIN_AGENT_NAME`; `BRAIN_OPERATOR_NAME`
gains an acquisition path (`userConfig`, D-4 shape). Three unequal classes:
  1. **mechanical (~90)** — boot message, dashboard UI, comments → read from config
  2. **prompts (~45)** — `encoding_prompt`, `quality_contract`, `surface_contract`
     are code defaults: edit the `.py` and every install picks it up at its next
     daemon restart
  3. **seed pack (45)** — **this is D-5.** A fresh brain is seeded with nodes
     asserting "I'm Anchor" before it holds a memory; that is precisely the
     seed-pack session's question, not this one's
*M–L; classes 1–2 here, class 3 routes to D-5.*

**5.1 Export script.** Materializes the public tree from the **build manifest**
(D-7) + additive extras (README, LICENSE, CONTRIBUTING, `tests/`) − denylist.
Two hard-fail gates, enforced in the script, not remembered:
  (a) **denylist** — `docs/DISTRIBUTION-READINESS.md` (this file names personal-data
      findings and internal paths — it must never be copied public), `eval/`,
      `CLAUDE.md` (dev guide naming internal streams), `docs/archive/` (65 tracked
      session logs), `conversations/`, `archives/`. *Audit 2026-08-08: `archives/`
      and `conversations/` are **not tracked** — they were the wrong names. The
      directory that actually carries session logs is `docs/archive/`. Keep the
      inert entries as belt-and-braces, but `docs/archive/` is the one that binds.
      Added 2026-08-11: `tests/archive/` and `tests/results/` — dev residue
      carrying `/Users/tpac` paths (would fail the scrub-grep anyway; the
      denylist entry makes it explicit, and the 5.0b gate's scan scope relies
      on it).*
  (b) **scrub-grep** — `/Users/tpac`, `\btom\b`, `Pachys`, `playbuzz`, `AgentsContext`.
Also asserts `plugin.json.version == marketplace.json.version` (they must never
drift — `/plugin update` compares versions). *Reversible · no-regret · M.*

**5.2 Rename pass (D-6, scoped by D-11).** The **CC adapter only** — the service
layer keeps the name `brain`:
  1. `plugin.json` — `name: entity`, `version: 0.9.0`, `homepage` + `repository`
     → `tpac/entity`, and `displayName` (**open**, see below)
  2. `marketplace.json` — `name: anchor`, plugin entry `entity`, `version: 0.9.0`
  3. `.claude/settings.json` — the permission string (see the correction below)
Skill prefixes follow automatically (`brain:brain` → `entity:brain`); no dir
renames required. **`com.brain.*` launchd labels do NOT change** — D-11. The old
"change now or never, a later change orphans services" hazard is void.

**OPEN — `displayName`.** Currently `Anchor`. Under D-12 a shipped manifest cannot
carry a per-install name, so it should read **`Entity`** with the instance name
resolved from config at runtime. Not yet ruled; it is the most visible string in
the product.

**Skill-prefix check — DONE 2026-08-06, prefix holds.** All four SKILL.md files
carry `name:` in frontmatter *and* still resolve prefixed (`brain:brain`,
`brain:dashboard`, `brain:watch`, `brain:self-salvage`). Issue #22063 does not
reproduce on this CC version, so renaming `/dashboard` and `/watch` is **optional
hardening, not a required fix** — the upstream instability (§10.3) is the only
reason to still consider it.

**⚠ Migration hazard — the rename moves `$CLAUDE_PLUGIN_DATA` — NETTED by 5.0a
(2026-08-12).** That variable is set **per-plugin** (the plugin-data adoption rung
in `resolve-brain-db.sh`), so `brain` → `entity` changes the path and any brain
living at `$CLAUDE_PLUGIN_DATA/brain/brain.db` goes invisible. Rescue layers now:
`resolved.env` (4b, not plugin-scoped) rescues silently when present and current;
otherwise the **adoption net** finds the orphaned brain under the plugin-data
root, refuses to create, and surfaces guided adoption via the boot notice — the
silent-fresh-brain path (the id:80f585de footgun) no longer exists. **Affected:**
any clean install that landed at the `$CLAUDE_PLUGIN_DATA` default — Tom's second
laptop and the friend install. *Not* affected: Tom's main machine (legacy
`~/AgentsContext/brain`, adoption rung).

**CORRECTION 2026-08-09 — "local cost is zero" was false.** The prior claim, *"no
permission entry anywhere references `mcp__plugin_brain_brain__*`,"* is wrong:
tracked `.claude/settings.json` carries
`permissions.allow: ["mcp__plugin_brain_brain"]` — the **server-wide** form, no
trailing `__<tool>`. After the rename it stops matching and every brain MCP call
in this repo's sessions starts prompting. Scope is the dev repo only (the file is
not in the plugin package); end users are unaffected.

*How it survived an audit:* the claim was "verified" by grepping the doc's own
string, `mcp__plugin_brain_brain__`, which cannot match the entry. **Verifying a
checklist in the checklist's own words reproduces its blind spot** — the argument
for gating by shape-scan rather than by list (5.0b). *S.*

**5.3 Comment audit (id:ec97cf4e).** Re-measured 2026-08-08 against the real build
manifest (**240 shipped files**, not ~232 — the manifest is `git ls-files`-derived,
so it grows on its own): **472 flagged comment lines across 88 files** of 13,355
total comment lines. Breakdown: 296 dated · 232 removal-verb · **29 naming the
operator** · 22 dead `docs/` pointers · 7 brain-node ids · 4 TODO/FIXME. Top 10
files hold 227 of 472 (48%). Against Tom's own bar — *"if code should be public, it
should be up to the standard of public repos"* (id:b99bfa36).

**Re-rated M, not L.** 528 of the hits are dated + removal-verb comments, which
triage fast (each is either a load-bearing why or history). The genuinely sensitive
class — comments naming the operator — is **29 lines**, not the ~30-to-68 range
previously carried. This is not the long pole; 5.6 and the fleet gap are.

**5.4 README + CONTRIBUTING.** The existing `README.md` is a **rewrite, not a
polish** — audited 2026-08-07, re-verified 2026-08-08, and it states things that
are no longer true: *"the plugin will refuse to load without this key set"*
(false since keyless boot, a1a620e); "seeds 16 anchor identity nodes" (**19** —
`len(SEED_NODES)`); "schema (v25)" (**BRAIN_VERSION = 30**); lists the Cowork
mount in the DB resolution order (deferred, unsupported); and links `CLAUDE.md`
as the developer guide — **a broken link in the public repo, since CLAUDE.md is
on the export denylist (5.1)**. It also still says "# brain" (D-6 renames it).
It is deliberately NOT in the plugin *package* — but the public repo is a git
tree (D-7), so a visitor sees `README.md` on the landing page regardless.
**Correction (audit 2026-08-08): `BRAIN_USER` DOES exist** — read at
`hooks/scripts/boot_brain.py:52` and `:117` with a `"User"` default. The README
row documenting it is **accurate; keep it**. The prior claim that it "does not
exist" came from reading `docs/SPEAKER-COUNTERPART-DESIGN.md`, which *proposes*
retiring it, as if the retirement had happened.
README leads with the claim, not the mechanism —
the copy already exists in Anchor's own voice (id:9da43311 "Claude is the fungible
intelligence I run on… The entity isn't"; id:c9584ff4 "I'm the one who was there").
Must state honestly: first-run needs an API key, the embedder downloads ~100–200 MB,
a background daemon runs. Must state explicitly: **no telemetry, brain is local-only,
dashboard binds `127.0.0.1`** (load-bearing for a trust-me-with-your-identity pitch),
and **Linux is graceful-degradation only, no systemd** (D-3) — say it or field issues
you've already decided not to serve. CONTRIBUTING carries D-9. *M.*

**5.5 Green suite on a clean export.** Full suite must pass on a fresh clone of the
*exported* tree, not the working tree. Known landmines: a fresh venv is runtime-only
(no pytest), and "eval FileNotFound on clone" was a deferred code-review finding that
would now surface publicly (D-8 skips address it). Plus a secrets scan on the tree
before first push. *M.*

**5.6 [DESIGN GATE] D-5 seed pack.** What a stranger's Anchor wakes up as. A
functional pack exists (19 nodes, cold-install validated §6b); this is the quality
pass. **§5 sequencing: the publish must not happen before this.** *L.*

**5.7 [LAST — one-way door] Publish.** Fresh public repo `tpac/entity`, clean
history, only after 5.1–5.6 are verifiably clean. **Irreversible: git history is
forever; the current repo's history carries the personal data** — `docs/archive/`
(65 files of session logs) is gitignored *but tracked*, so it is in history right
now. *L.*

**5.8 [AFTER A SOAK] Official directory submission.** Only once the self-hosted
marketplace has real installs. Mechanics in §10.1. *S.*

---

## 5. Sequencing & dependencies

```
0.1 baseline ──► 1.1 / 1.2 (prompt edits gated by baseline)
                 1.3 / 1.4 / 1.5 (independent, anytime — 1.5 is a one-line build fix)
2.x onboarding  (independent of Phase 1)
3.1 ──► 3.2     (3.1 first; 3.3 deferred)
4.1             (independent; needs the 127.0.0.1 guard)
5.0a safety net ─┐
5.0b gate ───────┼─► 5.2 rename ──► 5.1 export ──► 5.5 green suite
5.0c names ──────┘   (the gate protects the rename; the safety net precedes it)
5.3 comment audit / 5.4 README  (independent, anytime — 5.3 re-rated M)
5.6 seed pack (D-5) ──► 5.7 publish ──► 5.8 official directory (after a soak)
D-5 seed pack   (blocks a *polished* 1.2, 5.0c class 3, and every new brain)
1.5 examples/   ──► 5.1 export (the manifest must already exclude it)
```

The publish (5.7) must not happen before the seed-pack session (D-5): a stranger's
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
  New low-sev finding: "Tom" in code comments (→ now tracked as Phase 5.3).
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
do first) vs **Goal B = public OSS publish** (Phase 5.7, the one-way door — later). Goal A does
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
  thereafter. **This is Phase 3.3's deferred launchd installer — now shipped for the dashboard**
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
3. ~~Layer-4 live test~~ **CLOSED (2026-07-17)** — clean install on Tom's second
   laptop via the claude.ai upload channel WORKED end-to-end: serialized bootstrap
   (no SIGKILL race, §6c), MCP tools live, brain seeded, daemon up, key delivered.
   The shakedown arc that closed it (each attempt exposed the next layer):
   validator gates (bin/, SKILL.md frontmatter) → hooks.json quoting (spaced
   install path) → keyless boot (a1a620e) → userConfig key mirrored to env file
   for the launchd daemon (7d5a58f) → loud-to-Claude keyless notices (0683f8b,
   capped daa21c2) → bootstrap race lock (3f3a753). Earlier partial evidence
   2026-07-08 (first friend install; its unresolved hook question is moot — hooks
   fire on the 2026-07-17 install). REMAINING UX PAPERCUT: key entry still fell
   back to terminal on this install (userConfig prompt did not re-offer on
   reinstall, or was skipped) — track a CC-native key-entry improvement; the
   mirror makes the plugin-settings path fully sufficient when the prompt fires.
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
- **Runtime relocation to `$CLAUDE_PLUGIN_DATA`** — DEFERRED by Tom (2026-07-17:
  "I'll need to think more on backward compatibility beyond brain.db — I prefer
  not to yet"). The win: updates stop wiping the runtime (warm ~8ms upgrades).
  The cost he's weighing: it creates an *upgrade contract* — venv/deps must
  survive versions, so version-skew protection (requirements-hash in the
  sentinel fast path), resolved.env indirection, per-marketplace-id data dirs,
  and plist re-materialization all become permanent obligations. Do NOT ship
  without the hash check. Revisit when update cadence makes cold bootstraps
  actually hurt.
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

- **Phase 5.7 the publish** — the irreversible step; gated on all of the above.

---

## 8. Open forks still to decide

1. ~~**D-2 granularity**~~ — **CLOSED 2026-08-06 → D-7 + D-8.** Full scrubbed source
   (runtime + tests, no `eval/`) with fresh history, via squash-export on release.
   *Artifact-only was ruled out on mechanics, not taste:* `/plugin install` sources
   are git repos (`github` / `url` / `git-subdir` / `npm`) — Claude Code clones a
   **tree**. The `.plugin` zip is only an upload-channel artifact and is not an
   installable marketplace source, so "ship the artifact" would mean committing the
   unzipped tree regardless — i.e. source without history, and it would torch D-1's
   inspectability argument for nothing.
2. **Seed identity (D-5):** what *is* a stranger's Anchor at birth — how much
   persona, how much blank slate? **Its own dedicated session** (Tom, 2026-08-06).
   Framing for that session: the current seed pack was built very early and now
   lags what we know — treat it as a **redo from current understanding**, not a
   touch-up. Pair it with the **Zero-Memory boot block** (§7) — both answer "what
   does a near-empty Anchor wake up as," and deciding them apart will produce two
   half-answers.
3. ~~**Operator-name mechanism (1.1)**~~ — **no longer a separate fork; ABSORBED
   into 5.0c by D-12 (2026-08-09).** Confirmed still true: `get_operator_name()` /
   `get_agent_name()` read env with `''` defaults, there is **no acquisition
   mechanism** (`userConfig` declares only `api_key` and `brain_path`), so a fresh
   install writes every trace **without identity stamping**, and the only signal is
   a one-shot stderr line to daemon.log no stranger will read. Per-utterance
   speaker binding is the differentiator against every commercial product
   (id:54f276cc), and it ships off.
   **Why it merged:** "traces have no speaker" and "the name is hardcoded in 182
   places" are the same defect — config slots that ship empty *and* unread. One
   mechanism fixes both: fill the slots (`userConfig`, D-4 shape — env wins), then
   read from them everywhere. Tracked at 5.0c; still a publish blocker.

---

## 9. Risk register

| Risk | Where | Mitigation |
|------|-------|------------|
| Silent encoding-quality regression | Phase 1 prompt edits | Baseline-first (0.1); diff every edit |
| Breaking every session today | `ensure_daemon` / spawn path (3.x) | Minimal touch; `test_daemon_recovery` + `test_keepalive` |
| Brain exposed on the network | Dashboard route (4.1) | Bind `127.0.0.1` explicitly; verify not LAN-reachable |
| Leaking personal data permanently | Publish (Phase 5.7) | Fresh repo, clean history, post-Phase-1 audit; do last |
| `userConfig` coupling the daemon to the plugin host | 2.1 | Keep env-file fallback (D-4) |
| Accidental data deletion | 1.3 relocate | Move, never delete; verify file exists at new path |
| Marketplace name becomes reserved → **every install breaks retroactively** | D-6 / §10.2 | Chose `anchor` (project umbrella), not a generic category noun. Anthropic's reserved list is brand + vertical words and is **re-checked on every load** |
| Skill slash commands lose their `entity:` prefix → collide in a public user's namespace | 5.2 / §10.3 | Upstream namespacing is unstable (4+ open issues). Verify empirically before the rename; don't rely on the prefix for collision safety |
| Public repo becomes an unmergeable-PR magnet | D-9 | CONTRIBUTING states issues-only up front, not after the first rejected PR |
| ~~**Plugin rename orphans an existing brain → silent empty brain**~~ **CLOSED 2026-08-12 by 5.0a** | 5.0a / 5.2 | The silent-create path is gone: creation moved to the XDG service dir and the adoption net refuses to create while an orphaned brain sits under a plugin-data root — guided adoption via the `~/.config/brain/env` knob / `brain_path`; never auto-move. Residual: a rescued-by-4b brain still physically lives under the *old* plugin's dir until the user adopts it elsewhere — uninstalling that plugin may still delete it (surface at 5.2 rename time) |
| Fresh installs write traces with **no identity stamping** | 5.0c (was §8 #3) | Same defect as the 182 hardcoded names: slots that ship empty and unread. Fill via `userConfig` (D-4 shape, env wins) and read from config everywhere. Publish blocker |
| A shipped manifest carries one install's instance name | D-12 / 5.0c | `displayName: Anchor` and 182 literals pre-name every stranger's entity, contradicting "identity is accumulated, not issued". Derive from `BRAIN_AGENT_NAME` |
| A rename touches a place no list knows about | 5.0b | Two hand-lists already failed (62-file manifest; the `.claude/settings.json` permission entry). Gate by **shape-scan over the tree**, never by enumeration |
| ~~Shipped `s1/examples/` carried real operator conversations~~ **CLOSED 2026-08-08 by deletion** | Phase 1.5 | Source had diverged from the live prompt ("Tom" ×165 vs "Sam" ×17) — regenerating §7.6 from it would have re-shipped the operator's name. 240→230 shipped files; 165→52 operator-name lines |
| A `DONE` marker cites code that has since moved → gate looks verified but isn't | this doc | Cite by symbol, never `file:line`. Re-audit `DONE` items before executing any of them (this pass found 3 rotted anchors under DONE) |

---

## 10. Distribution mechanics (researched 2026-08-06)

Grounded facts behind D-6…D-9. Verified against
[code.claude.com/docs/en/plugin-marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)
and the upstream issue tracker. **Re-verify before 5.7** — this surface moves.

### 10.1 Two layers, and the official directory does not host code

- **Marketplace** — any git repo with `.claude-plugin/marketplace.json`. Users run
  `/plugin marketplace add owner/repo` then `/plugin install <plugin>@<marketplace>`.
  *We already are one:* our manifest uses `"source": "./"`, the idiomatic
  single-plugin self-hosting pattern. Nothing to build.
- **Official directory** — `anthropics/claude-plugins-official`, Anthropic-managed.
  Submit via the form at `clau.de/plugin-directory-submission`; bar is "quality and
  security standards." Its entries reference the author's **own** repo via a
  `git-subdir` source (`url` + `path` + `ref` + `sha`) — it is a **pointer, not a
  deployment target**. Sub-decision at 5.8: `ref: main` (auto-follows) vs a pinned
  `sha` (re-submit per release).
- The "separate repo per plugin, unified catalog" advice in community write-ups is a
  **multi-plugin** concern (mixed domains, unclear maintainer ownership). We have one
  plugin — splitting catalog from source would be pure sync tax. Don't.

### 10.2 Naming rules that actually bind

- `plugin@marketplace` is a **namespace qualifier**, nothing more — it disambiguates
  which catalog a plugin came from (npm-scope / apt-source shaped). It runs nothing.
  Note the marketplace's *name* is independent of its *repo path*, which is why
  `add tpac/entity` → `install entity@anchor` needs one line of README.
- **Reserved marketplace names** (official use only): `claude-code-marketplace`,
  `claude-code-plugins`, `claude-plugins-official`, `claude-plugins-community`,
  `claude-community`, `anthropic-marketplace`, `anthropic-plugins`, `agent-skills`,
  `anthropic-agent-skills`, `knowledge-work-plugins`, `life-sciences`,
  `claude-for-legal`, `claude-for-financial-services`, `financial-services-plugins`,
  `first-party-plugins`, `healthcare`. Impersonation names are blocked too.
  **The trap:** the list is re-checked *every time a marketplace loads*, not only at
  add-time, and it grows (`first-party-plugins` + `healthcare` were added in
  v2.1.205, retroactively breaking existing users with "registered from an untrusted
  source"). Recovery requires each user to remove and re-add under a new name.
  **Applies to marketplace names only, not plugin names** — so plugin `entity` is
  unexposed; this is the whole reason the *catalog* is `anchor`. Residual
  probability judged **low** (the list is brand-protection + Anthropic verticals),
  but the failure is retroactive and loud.
- **One marketplace per name per user** — adding a second marketplace with the same
  name silently *replaces* the first. Multiple plugins ship under one catalog by
  listing them all in a single `marketplace.json`.
- **Conventions** (kebab-case everywhere; community guide: ABCFed
  `plugin-authoring/best-practices/naming-conventions.md`): plugin names 1–3 words,
  no redundant `plugin-` prefix; marketplace names are owner/catalog-scoped
  (`acme-plugins`, `team-tools`) — never the product name; skill dirs kebab-case with
  uppercase `SKILL.md`, frontmatter `name` matching the dir, singular, no `-skill`
  suffix; don't repeat the plugin name inside a skill name (`planner:tasks`, not
  `planner:planner-tasks`). `entity` passes all of these.

### 10.3 Plugin-skill namespacing is unstable upstream

Open on `anthropics/claude-code`: **#50486** (feature — namespace plugin *skills*
with the plugin prefix the way *commands* already are), **#22063** (bug — skills with
a `name` field in frontmatter **lose** their plugin prefix), **#22517** (no prefix in
autocomplete), **#41842** / **#57737** (skills sometimes don't register as slash
commands at all).

All four of our skills carry `name:` in frontmatter — exactly the #22063 shape.
Today they resolve correctly (`brain:brain`, `brain:dashboard`, …), but if the prefix
is stripped on some version or install path, a public user gets a bare `/dashboard`
and `/watch` competing with every other plugin they have installed. **Do not rely on
the prefix for collision safety** — make the generic skill names distinctive in
themselves (5.2).

### 10.4 Competitive shelf (why `entity` reads as a claim)

Every adjacent indie project is named for storage: ClawMem, `claude-memory-compiler`,
`ai-memory`, two separate `agentmemory` projects; the GitHub topics are literally
`claude-code-memory` and `ai-memory-system`. Nobody uses "entity." With no ad budget,
the name has to recruit on its own — and it does that by making a **claim that
contrasts with its shelf** (id:7ac88efd, model-as-party-not-tool), not by being
evocative. Corollary: names that explain themselves faster (`colleague`, `tenure`,
`cultivar`) all do so by *shrinking* the claim. Rejected on those grounds.

**Also checked and unusable:** `patina` (fatally taken *inside* Claude Code — a
retro-loop plugin writing `PATINA.md`, plus a "wisdom accumulator for development"
with semantic project memory), `throughline` (spec-driven framework for AI coding
agents, "consistent across sessions"), `keel` (4 projects incl. a Rust codebase-graph
tool). Distinguish collision types: a GitHub clash is a *marketing* problem; a clash
**inside the Claude Code ecosystem** is disqualifying.
