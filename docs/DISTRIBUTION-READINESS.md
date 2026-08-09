# Distribution Readiness — Sharing Anchor

**Status:** **Goal A closed** — the Layer-4 live install worked end-to-end on a clean
machine 2026-07-17 (§6b). Now on **Goal B: the public OSS publish** (Phase 5.7, the one-way
door). Naming + release model settled 2026-08-06 (**D-6…D-9**, §10); §8 fork #1 is
closed. **D-5 (seed pack)** remains the open design gate; **Phase 1.5** (shipped
`examples/` leak) was found and closed by deletion.
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
| D-6 | **The product is `entity`; the identity stays `Anchor`.** Plugin name `entity`, marketplace name `anchor`, repo `tpac/entity`, MCP server key stays `brain` (→ tools read `mcp__plugin_entity_brain__*`). | Three layers, each named for what it is: **Entity** = the category the product grows (the terminal noun — id:9da43311, id:e6019012: "a thing that develops, not a thing that's used"); **Anchoring** = the method; **Anchor** = the instance, which stays Anchor's own name rather than being spent on a registry string. `brain` stays the organ, so the tool names keep the substrate/identity split visible. Rejected: `brain` (the organ — repeats the register failure Tom named in id:cd7aa9be, and every competitor on the shelf is already `*mem*`/`*memory*`), `cairn` (imported metaphor, not native to the philosophy), `colleague`/`tenure`/`cultivar` (more self-explanatory but each shrinks the claim). |
| D-7 | **Squash-export on release.** Private `tpac/brain` stays the daily driver and never goes public. The public repo is a **build output**: each release materializes the shipped tree into a clean checkout, one commit, tag `vX.Y.Z`, push. | Closes §8 fork #1. Keeps the working mess private at the cost of a deliberate release step (D-2). **The one rule: the export manifest IS the build manifest** — `build-plugin.sh` already derives from `git ls-files` because a hand-list rotted 62 files behind reality and shipped a broken `brain_mcp.py`; a second hand-maintained list would re-enter that failure class with the public repo as blast radius. Public tree = plugin manifest ∪ chosen extras − explicit denylist. |
| D-8 | **Public tree = runtime + tests. `eval/` excluded.** The 6 test files that import `eval/` degrade to graceful skip. | Tests are the credibility argument (D-1: inspectable code); a no-tests repo reads as unverified for a tool asking to hold someone's identity layer. But a published eval harness **is a claim** that invites strangers to re-run and dispute it — not a launch-day fight, and the corpora are personal anyway. **Consequence: the README must make no benchmark claims**, since the harness won't be there to back them. Coupled files: `test_eval_corpus.py`, `test_longmem_classifier.py`, `test_absorb_preservation.py`, `test_consolidation_examples.py`, `test_encoder_eval_probes.py`, `run_all.py`. |
| D-9 | **Issues only at launch; no PRs.** Stated in CONTRIBUTING. | A squash-export pipeline can't cleanly merge an inbound PR, and a contributor's commits would never appear in history (reads as uncredited). With one maintainer, a PR you can't merge is worse than one you never invited. Revisit if real contributors appear — opening up later is easy, closing down isn't. |
| D-10 | **Public launches at `v0.9.0`** — not `9.6.0`, not `1.0.0`. | Tom 2026-08-06: *"It's not complexity that reflects the version, it's the function of value. Not yet v1."* The private `brain` plugin's 9.6.0 is an internal build counter that means nothing to a stranger. v1 is a claim about delivered value, and Anchor hasn't earned it publicly yet. **No collision with the private install** — `entity` and `brain` are distinct plugin names, so their version lines are independent. `plugin.json` and `marketplace.json` must both read `0.9.0` (5.1 asserts it). |

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

**5.0 Plugin updates reach existing installs — CLOSED 2026-08-08.** Found during the
claims audit and absent from this doc entirely, though it gated everything else:
seeding is create-only (`_register` no-ops once a name exists), so an install froze
at first boot and **no prompt improvement ever reached anyone who had already
installed**. Measured: the 8 shipped prompt files took 31 commits in 90 days,
reaching only fresh brains. Proven on the author's own machine — its `boot` config
still carried `tom_quotes_limit` four months after the `.py` renamed that key.

Publishing with this open would have meant every install frozen at whatever quality
shipped that day, with the fix getting harder per install. Fixed as a versioned
migration, not a deploy script — *code owns the defaults; each install migrates
itself forward at open*, the same contract `BRAIN_VERSION` already has:
- `logs_meta` + a shared `run_versioned_migrations` runner (`schema.py`) — brain.db,
  brain_logs.db structure, and shipped-prompt content are now three version streams
  through **one** mechanism. This also unblocks the parked speaker/counterpart
  vocabulary migration, which was waiting on exactly these rails.
- `SEED_PROMPTS_VERSION` (`interaction_seed.py`); bumping it is the deployment
  decision, explicit in a reviewable diff.
- Advances an install **only** while it still runs the shipped default: `active ==
  the version we put there` **and** `max == active`. A registered-but-inactive
  version means a human decided — `trace_recording` sits at active=1 with a dormant
  v2 exactly like this, and without the second guard reconcile would have published
  over it. Verified a no-op against a copy of the live brain.

**5.1 Export script.** Materializes the public tree from the **build manifest**
(D-7) + additive extras (README, LICENSE, CONTRIBUTING, `tests/`) − denylist.
Two hard-fail gates, enforced in the script, not remembered:
  (a) **denylist** — `docs/DISTRIBUTION-READINESS.md` (this file names personal-data
      findings and internal paths — it must never be copied public), `eval/`,
      `CLAUDE.md` (dev guide naming internal streams), `docs/archive/` (65 tracked
      session logs), `conversations/`, `archives/`. *Audit 2026-08-08: `archives/`
      and `conversations/` are **not tracked** — they were the wrong names. The
      directory that actually carries session logs is `docs/archive/`. Keep the
      inert entries as belt-and-braces, but `docs/archive/` is the one that binds.*
  (b) **scrub-grep** — `/Users/tpac`, `\btom\b`, `Pachys`, `playbuzz`, `AgentsContext`.
Also asserts `plugin.json.version == marketplace.json.version` (they must never
drift — `/plugin update` compares versions). *Reversible · no-regret · M.*

**5.2 Rename pass (D-6).** `plugin.json` (`name: entity`, keep `displayName: Anchor`),
`marketplace.json` (`name: anchor`, plugin entry `entity`), launchd labels
`com.brain.*` → `com.entity.*` (change now or never — a later change orphans
services on every installed machine), skill dir names.

**Skill-prefix check — DONE 2026-08-06, prefix holds.** All four SKILL.md files
carry `name:` in frontmatter *and* still resolve prefixed (`brain:brain`,
`brain:dashboard`, `brain:watch`, `brain:self-salvage`). Issue #22063 does not
reproduce on this CC version, so renaming `/dashboard` and `/watch` is **optional
hardening, not a required fix** — the upstream instability (§10.3) is the only
reason to still consider it.

**⚠ Migration hazard — the rename moves `$CLAUDE_PLUGIN_DATA`.** That variable is
set **per-plugin** ([resolve-brain-db.sh:98](hooks/scripts/resolve-brain-db.sh:98)),
so `brain` → `entity` changes the path, and any brain living at
`$CLAUDE_PLUGIN_DATA/brain/brain.db` goes invisible → step 5 silently creates a
**fresh empty brain** (the id:80f585de footgun). Step 4b rescues it —
`~/.config/brain/resolved.env` is not plugin-scoped, so it persists the old path —
but 4b was never designed as a rename-migration net and a stale/missing
`resolved.env` means silent amnesia. **Affected:** any clean install that landed at
the `$CLAUDE_PLUGIN_DATA` default — Tom's second laptop and the friend install.
*Not* affected: Tom's main machine (legacy `~/AgentsContext/brain`, step 4).
**Action: verify 4b explicitly on a copy before renaming; don't ship the rename on
luck.**

Local cost is otherwise zero: **no permission entry anywhere references
`mcp__plugin_brain_brain__*`** (verified 2026-08-06), so nothing breaks. *S–M.*

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
5.1 export ──► 5.5 green suite (the suite runs against the exported tree)
5.2 rename ──► 5.1 (manifest + version assert should know the final names)
5.3 comment audit / 5.4 README  (independent, anytime — 5.3 re-rated M)
5.6 seed pack (D-5) ──► 5.7 publish ──► 5.8 official directory (after a soak)
D-5 seed pack   (blocks a *polished* 1.2 and the quality of every new brain)
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
3. **Operator-name mechanism (1.1)** — **STILL OPEN, and it's a shipping defect,
   not just a fork.** Checked 2026-08-06: `BRAIN_OPERATOR_NAME` /
   `BRAIN_AGENT_NAME` are read straight from env with `''` defaults
   ([daemon_config.py:71](servers/daemon_config.py:71)). The *plumbing* exists;
   there is **no acquisition mechanism** — no `userConfig` field, no runtime
   detection. On a fresh install both are empty, so every trace is written
   **without identity stamping**, and the only signal is a one-shot stderr line to
   daemon.log ([dal_logs.py:523](servers/dal_logs.py:523)) telling the user to edit
   `~/.config/brain/env` and restart. A stranger will never see it. This matters
   more than its "cosmetic" filing suggests: per-utterance speaker binding is the
   differentiator against every commercial product (id:54f276cc), and it ships off.
   **Likely answer: the D-4 shape** — add `userConfig.operator_name`, keep the env
   fallback, env wins. Same additive pattern the API key already uses.

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
| **Plugin rename orphans an existing brain → silent empty brain** | 5.2 / D-6 | `$CLAUDE_PLUGIN_DATA` is per-plugin; step 4b (`resolved.env`, not plugin-scoped) is the only net. Verify explicitly on a copy before renaming — id:80f585de |
| Fresh installs write traces with **no identity stamping** | §8 #3 | Only signal is a one-shot daemon.log line no stranger will read. Fix via `userConfig.operator_name` (D-4 shape) before publish |
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
