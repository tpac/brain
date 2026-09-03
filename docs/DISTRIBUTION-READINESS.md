# Distribution Readiness — Sharing Anchor

## §ACTIVE ARC (2026-09-03) — **5.2 DONE (rename + 0.9.0 + the D-10 assertion); next is 5.9, then 5.7**

**5.2 landed 2026-09-03.** Both manifests read `entity` / `anchor` / `0.9.0`;
the 15 product-name sites (`boot-brain.sh`, `setup.html`) say **Entity**; the
`com.entity.` check armed at the rename and passed. D-10 is now enforced twice:
`TestVersionLockstep.EXPECTED_VERSION` (the ONE home of the literal — a release
bump edits it in the same commit as the manifests) and gate C's optional
`EXPECT_VERSION=X` (the release command passes what it is releasing; unset keeps
agreement-only). `.claude/settings.json` carries BOTH permission strings until
the redeploy has happened and pre-rename sessions have cycled — drop
`mcp__plugin_brain_brain` then. **Not done in 5.2, by design:** the redeploy
itself (Tom's timing — it flips every new session on his machine to the
`entity` name, and his install is a *directory* marketplace whose outer
`marketplace.json` still says `brain`; observe one session before deciding
whether the outer entry needs the rename too), and the D-12 xfail (re-measured
2026-09-03: **55 shipped files / 150 occurrences** still carry `Anchor` — the
deferred one-at-a-time sweep, not a SCOPE question; the 7 never-ship files /
53 occurrences are the SCOPE question and can wait until that sweep is done).

**5.3 closed its gated half 2026-09-01. Gate B: 69 → 0, and it now STAYS
there** — `TestPublicTreeExport::test_live_tree_exports_clean` runs the export
over the live repo on every suite run. Verified against a 13-commit merge from
main immediately after arming: clean.

**What actually mattered was not on the worklist.** The audit's headline number
(614 dated/removal-verb lines) is polish; the real find was that gate B scrubbed
the employer's FORMER name and not its current one, leaking `ex.co` 20 times
including three real project names hardcoded in a shipped migration. Gate B now
also scans `\btpac\b` and `\bex\.co\b|\bexco\b`. Detail in the 5.3 item.

**Tom's standing directive, 2026-09-01:** *"i want things that are released to
the public repo to be opt in not opt out so we dont accidentally launch personal
stuff."* Measured: 413 of 429 public files arrive by opt-out. New item **5.9**.

**Deliberately NOT done, and this is a ruling not a gap:** the ~487
dated/removal-verb comment lines. They carry no personal data and no broken
reference, 95% of them are good comments, and sweeping 487 judgment calls is a
plausible net negative. Reopen only with a sample in front of Tom.

### STILL OPEN after 5.3 — read this before assuming the audit finished

5.3 closed the **personal-information** class. It did not close the **name**
class, and two of the items below are bigger than they look.

| # | Open | Where it is written up |
|---|---|---|
| 1 | **The encoder's system prompt says `"I am Anchor"`** — 3 occurrences in `encoding_prompt.py`'s `SYSTEM_PROMPT`, so every install's encoder claims that name. Eval-gated: a prompt change, not a comment edit | 5.3 → the Anchor inventory |
| 2 | **`Anchor` across 62 files / 203 occurrences** (re-measured post-rename 2026-09-03: **150 in 55 shipped files** — the real blocker — and 53 in 7 never-ship files) — comments, docstrings, prompt/rubric files. Tom ruled it a deliberate one-at-a-time sweep with him engaged, NOT a mechanical pass | same table |
| 3 | **`encoding_source='anchor'`** — 41 data-model literals + the lock gate's `startswith('anchor')`. Rename target recorded as `interactive`/`session` | same, final paragraph |
| 4 | **The public tree is opt-out** — 413 of 429 files ship because nothing said no | **5.9** |
| 5 | **Nothing covers what the release step adds** — commit message, tag, repo description, committer identity are all outside every gate | **5.7** (unbuilt) |
| 6 | **The v30 slug-map collapse is lossy** for 9 local pre-v30 copies, four of them Tom's own backups | 5.3 → the ⚠ note |
| 7 | 116 dangling `docs/`/`eval/` references | 5.3 → the split table |

Items 1–3 are one thread, and it is **not** a comment audit. Items 4 and 5 are
the two places where the gates genuinely do not reach.

---

### Prior arc head (2026-09-01) — 5.0c phase 4

**Handoff node: `id:bc154094`. Phase-4 detail lives in the 5.0c item below.**

**5.0c phase 4 landed 2026-09-01, minus two hunks.** The mechanical half shipped
as `0bae543` (agent name defaults to **Jade**, chosen-ness tracked separately so
a default disarms nothing). Phase 4 merged four of six prompt-visible sites plus
both MCP schema descriptions; the two `encoding_prompt.py` hunks went to the
v-next.7 stream instead of main (see the 5.0c item — they were at risk of being
silently reverted by that candidate's promotion). The answer
at all six sites was *generalize*, never *substitute a config read*: a
per-install name in a template body would break
`interaction_fingerprint()`'s documented cross-install comparability. The
identity exemplar was chosen by a re-runnable probe
(`eval/identity_exemplar_probe.py`), not by taste. The **D-1 packaging fix**
rode along: `LICENSES/` now ships (manifest 235 → 237) and `marketplace.json`
carries `license`.

**Every count in this doc moves with every merge — re-measure, don't trust.**
Gate B was 67 when the mechanical block landed and 69 hours later, from
another stream's comments. Re-measured after phase 4: **still 69**, gate A
clean, 428 files — and that is correct, not a miss. Gate B scans
*personal-information* patterns; phase 4 removed `Anchor` literals, a
different class with a different gate. Only the 5.3 audit moves this number.
That is the character of the remaining worklist: the gated part is small and
shrinking, the ungated comment part grows on its own.

**Two corrections to what this section used to claim** — both measured, both
were false:
- **5.0c does NOT un-xfail gate assertion 4**, and neither does 5.3. It scans
  a scope wider than what ships and still matches 203 occurrences / 62 files
  (post-rename 2026-09-03; 150 of them in files that ship, so SCOPE alone
  cannot arm it).
  The class the plan assigned to the comment audit is **not a comment
  problem** — `Anchor` there names an architectural role that is also a data
  literal (`encoding_source='anchor'`, `anchor_touched`). Detail in the 5.3
  ratchet note; it needs a D-12 ruling, not a sweep.
- **The frozen-corpus harness cannot A/B a code change.** Its address omits
  the repo's code state, so a two-tree A/B silently cache-hits and reports
  clean. Detail in the 5.0c item.

**Remaining, in dependency order:**
1. ~~**5.3 comment audit**~~ — **gated half DONE + ratcheted 2026-09-01.** The
   ungated ~487 dated/removal-verb lines are a ruling, not a gap (see 5.3).
2. ~~**5.2 rename + D-10**~~ — **DONE 2026-09-03** (see the arc head).
3. **5.9 opt-in export** — before the first push, since the first push is the
   one that can leak. Not a gate on 5.2; a gate on 5.7.
4. **5.7 release command → publish** — the one-way door. Must include an
   **install smoke test**: the suite runs *inside* the export tree, so it
   cannot catch a file that ships, is never imported by tests, and is needed
   at first run on a stranger's machine.

**Rulings waiting on Tom** (details in `id:bc154094`): the
`agent_is_default=False` default; whether the boot banner is product or
entity; the `_has_brain` predicate; and now **the scope of gate assertion 4** —
narrow `SCOPE` to what ships, or clean the two unowned classes. **The fleet is
deferred to last** and is explicitly not a gate on 5.2.

**Sequence ruled 2026-08-31, superseding both prior orders** (the doc's
`5.0c → 5.3 → 5.2 → 5.5 → 5.7` and id:44025fbb's `5.0c → 5.3 → 5.5 → 5.2 → 5.7`).
Two corrections to both:
- **5.0c classes 1–2 and the 5.3 *runtime scrub* class are one pass, not two.**
  15 shipped files carry both an `Anchor` literal and a `Tom` literal, and in the
  S1 prompt/rubric files they sit in the *same strings* (`quality_contract.py`
  D-dimension examples, `encode.py`, `encoder_view.py`). Those are eval-gated
  artifacts: two passes over them = two eval gates.
- **5.2 goes late, immediately before 5.7.** It is a three-file edit, and after
  ruling 3 below it has **no blocking precondition at all**. It still goes late,
  for a different reason: 5.0b's `com.N.` assertion is *dormant until the
  rename* and arms at it, so a 5.5 run before 5.2 is a 5.5 you run twice.

**Tom ruled all four open questions 2026-08-31:**
1. **Gold data → exclude + graceful-skip**, not sanitize. Denylist
   `golden_dataset_v2.json`, `golden_canary.json`, `corpus/precision_corpus.json`;
   their consumers join the D-8 skip set. Rationale: a sanitized corpus that
   still passes proves less than an honest gap, and sanitizing silently changes
   the ground truth a recall test measures against.
2. **The two eval-only quality contracts → RELOCATE into `eval/`**, not
   denylist. They move beside their only consumers; the package stops carrying
   them by construction rather than by a denylist entry. Update the two
   `eval/agent_introspect/*_eval.py` imports in the same commit.
3. **The fleet check → deferred to LAST**, and explicitly *not* a gate on 5.2.
   Tom: *"Let's defer for last, i dont know their situation yet and perhaps
   we'll just tailor a 1 time special install."* See the fleet note in §5.
4. **The 5.3 comment audit → FULL audit before publish.** All 614 lines, on the
   critical path. Overrides the "ungated, therefore deferrable" reading — the
   public-repo bar (id:b99bfa36) is the standard, not the gate.

Ordered: ~~**(1)** free deletions + the relocation~~ · ~~**(2)** gold-data
exclusion~~ · **(3)** the combined literal sweep — the long pole · ~~**(4)**
5.5 + the D-8 graceful-skip~~ · **(5)** the full 5.3 comment audit ·
**(6)** 5.2 rename + D-10 bump + the missing `0.9.0` assertion · **(7)** 5.7 ·
**(8)** the fleet, last, likely as a hand-tailored one-time install.

### Steps 1, 2 and 4 SHIPPED 2026-08-31 (`6fb57ed`)

**Gate B: 227 → 67 hits / 39 files. Gate A clean. The export tree's full suite
is GREEN: `2983 passed, 14 skipped, 0 failed` (36m52s), collection `2990 / 0
errors` — it previously aborted on 6.** The 14 skips are 7 eval-guarded modules
reporting the D-8 reason plus 7 pre-existing; nothing is silenced by accident.

**What remains in gate B is exactly the two future steps and nothing else:**
**49** in `servers/` + `dashboard/` (step 3, the eval-gated sweep) and **18**
test attribution comments (step 5). `Anchor` literals in the package: **131 /
43 files** (was 143 — the relocation took `quality_contract.py`'s 14).

Landed: both eval-only quality contracts relocated to `eval/agent_introspect/`;
gold corpora excluded (no test needed a skip — every consumer was already a
denylisted harness); `tests/test_deploy_contract.py` and 11 dev harnesses
denylisted; the `Tom` → `Ada` fixture rename across 18 test files, token-aware
so attribution comments were left for step 5; four hand-rolled resolution
ladders routed through `resolve_db_dir`; and `tests/eval_optional.py` as the
one door for the D-8 graceful skip (7 modules + the capture pin).

**Three things found in execution that the plan did not anticipate:**
1. **`tests/test_deploy_contract.py` had to be denylisted, not allowlisted.**
   It self-skips outside the dev repo — but the *public repo is a git checkout
   with `plugin.json` tracked*, so it would NOT skip there: it would run and
   fail, because `TestPublicTreeExport` shells out to
   `scripts/export-public-tree.sh`, which is outside the manifest. The 5.5
   measurement missed this because the export tree had no `.git` (id:e354693e).
2. **Three files assert ON the literal** and a rename would have silently made
   them vacuous — `seed_pack.py`'s origin-story tributes, the test holding
   their boundary, and `test_prompt_uses_generic_operator_label` (which asserts
   `'Tom:'` is *absent* from a surface prompt). Allowlisted, not renamed. They
   are D-12 allies.
3. **The hardcoded `~/AgentsContext` path was masking a pre-existing test
   env leak.** Four `setUpClass` sites set `$BRAIN_DB_DIR` to a temp dir and
   never restore it — and then delete that dir. Removing the literal withdrew
   an accidental absorber (id:dd426bd9). **Still open:** the leak itself
   (`test_spread_activation.py` ×2, `test_remember_source_refs.py`,
   `test_encoder_eval_probes.py`), blast radius unmeasured.
**Where the work actually is.** Shipped and verified on main: 5.0 (`e3d9481`),
the versioned migration runner (`b11e45d` — both DBs, one real step at v31),
5.0a (`89e8286`) **completed 2026-08-28 by the relocation offer (`d4f4772`)**:
`hooks/scripts/relocate-brain.sh` + parked-brain boot notice, so pre-08-12
installs can converge on the XDG service dir. 5.0b gate, arch-plan steps 0–6.
**The uninstall question is ANSWERED** (empirically, 2026-08-27): a default
`claude plugin uninstall` DELETES `~/.claude/plugins/data/<plugin>-<mkt>/`;
`--keep-data` is opt-in; `update` preserves. So relocation is a **rename
precondition**: bump the plugin version so the notice + script reach existing
installs, let parked brains converge to XDG, THEN rename — `plugin update`
cannot cross a rename (identity is `plugin@marketplace`; D-6 changes both
halves), so anything the fleet needs before the rename must ship as a `brain`
update first.
**Version bump SHIPPED 2026-08-28: 9.7.2** — carries the relocation offer to
the fleet; the rename soak clock starts when existing installs take it.
**5.1 SHIPPED 2026-08-28** (`scripts/export-public-tree.sh` — see the item).
**5.4 DONE** same day. **5.6 / D-5 DONE 2026-08-30** — the design gate is
closed; the publish path has no design questions left. Unstarted: **5.0c**
(classes 1–2 only — class 3 dissolved with the name-free pack) · **5.3** ·
**5.5** · ~~**5.2**~~ (done 2026-09-03) · **5.7**.
Arch-plan steps 7–10 remain open but gate nothing.

**All counts re-measured live 2026-08-31 against the materialized export tree**
(`export-public-tree.sh` + `build-plugin.sh --list`, 237 shipped files). Every
number below moved; the doc's prior figures are superseded:

| | Was | Now | Note |
|---|---|---|---|
| 5.3 scrub worklist (gate B) | 217 / 81 files | **227 / 76 files** | 208 `tom`, 18 `AgentsContext`, 4 `Pachys`, 2 `/Users/tpac` |
| ↳ in `tests/` | — | **163** | the sanitize-vs-exclude fork lives here |
| ↳ in shipped runtime | — | **59 / 26 files** | half comments, half prompt/rubric strings |
| 5.3 comment audit | 472 / 88 files | **614 / 111 files** | ungated; see 5.3 |
| ↳ naming the operator | 29 | **33 / 20 files** | |
| D-12 `Anchor` literals | 182 / 43 | **143 / 43** | package manifest; Nursery took `seed_pack.py` 45 → 4 |
| 5.7 personal data in git history | 65 | **89 tracked files** | 70 `docs/archive` + 8 `tests/conversations` + 9 `tests/archive` + 2 `tests/results` |
| D-8 coupled test files | 5 listed / "6" | **exactly 6** | re-derived; see 5.5 |

**Free deletions found 2026-08-31 — do these first, they cost nothing.** Four
mechanical moves that shrink both worklists with no prompt editing and no eval:
1. **`servers/scales/s1/quality_contract.py` (1357 lines) and
   `servers/scales/s2/consolidation_quality_contract.py` (279 lines) ship to
   every install with no runtime consumer.** Their only importers are
   `eval/agent_introspect/encoder_contract_eval.py` and
   `consolidation_contract_eval.py` — both inside `eval/`, which D-8 excludes.
   The S1 one is the densest file in *both* worklists (14 `Anchor` + 12 `Tom`).
   **Ruled 2026-08-31: relocate both into `eval/`** beside their consumers —
   the package stops carrying them by construction, not by a denylist entry,
   and an eval-only instrument in `servers/` was a one-concern-per-file
   misplacement anyway. Update the two `eval/agent_introspect/*_eval.py`
   imports in the same commit. *Nothing in the deploy gate checks reachability:
   a `git ls-files` manifest fixed rot but cannot tell shipped-and-used from
   shipped-and-dead.*
2. **The scrub gate has no self-exemption.** `tests/test_deploy_contract.py`
   contributes 5 hits that are the gate's own fixtures — it writes
   `/Users/tpac/brain` and `Tom Pachys` into a tmp tree precisely to prove the
   gate catches them. 5.0b already grants itself this exemption ("the scanner
   must name the shapes it hunts"); 5.1's gate B does not, so it stays red
   forever otherwise. Allowlist entry, not an edit.
3. **The legacy-rung behavior tests deserve the allowlist the resolvers have.**
   `test_db_resolution.py`, `test_daemon_recovery.py`, `test_recall_laf.py`
   reference `~/AgentsContext/brain` because that rung is *shipped behavior* —
   the same rationale that already allowlists the four resolver files.
4. **Dev harnesses hardcode the author's machine as a default source** —
   `isolated_brain.py`, `run_tests.py`, `benchmark_multivec_encoding.py`,
   `benchmark_canary.py`, `bench_vector_cache.py`, `eval_runner.py`,
   `generate_golden.py`, `test_spread_activation.py`, `integration/test_*.py`.
   These are dead paths on a stranger's machine, not shipped behavior — clean
   or denylist.

**The one operator question left is 5.8's, and the research is done
(2026-08-31): no published policy requires an OSI license.** See 5.8 — the
constraint is different from the one the doc assumed.
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
closed. **D-5 (seed pack) is CLOSED — the Nursery, 2026-08-30**; the publish path
has no design gates left. **Phase 1.5** (shipped
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

An official `entity` plugin on a clean public repo — no junk, no personal
info — that people can use to **install and update** the plugin (Tom,
2026-08-28). Installable by someone who isn't Tom, on their own machine, to
the standard of a public open-source repo; updatable in place, with their
brain surviving every update. Each new user gets their **own** brain (their
own daemon, their own DB) that becomes their own Anchor over time. We are not
sharing Tom's brain — we are sharing the *substrate* that grows one.

---

## 2. Locked decisions

These forks are settled. Future sessions should not relitigate them without a
stated reason.

**Conformance sweep, 2026-08-31 — every D-number walked against the artifacts.**
Run because this arc's failure mode is *decisions locking at the governance layer
while execution drifts at the substrate* (community id:3350ea51), proven twice on
2026-08-28 alone. Before this sweep only 2 of 13 had been checked against code.

| # | Artifacts reflect it? | Evidence |
|---|---|---|
| D-1 | **partial — one defect** | `LICENSE` ships and names the dual grant, but points at `LICENSES/PolyForm-*.md` for full text and **`LICENSES/` is not in the package manifest** — dangling on an installed plugin (it *is* in the public tree, an export extra). `marketplace.json` carries **no `license` field** though the marketplace schema accepts an optional SPDX string and `plugin.json` sets one. README makes no "open source" claim ✓ |
| D-2 | yes | no artifact to drift |
| D-3 | yes | `daemon_launch.py` treats a missing `launchctl` as "no launchd platform" and falls through to `subprocess.Popen` |
| D-4 | yes | `plugin.json.userConfig` carries `api_key` + `brain_path`; the `~/.config/brain/env` ladder rung is intact |
| D-5 | yes | the Nursery pack, `tests/test_seed_pack.py::TestNurseryPackContracts` |
| D-6 | yes (2026-09-03) | `plugin.json` `name: entity`, `homepage`/`repository` `tpac/entity`; marketplace `name: anchor`, entry `entity`. MCP server key stays `brain` (`.mcp.json`) → tools `mcp__plugin_entity_brain__*` ✓; `displayName: Entity` ✓ |
| D-7 | export yes, **release no** | `export-public-tree.sh` materializes and gates; the exported tree correctly carries no `.git`. The squash-push/tag command is unbuilt (5.7) |
| D-8 | **partial — one defect** | `eval/` denylisted ✓, `tests/` ship ✓. But the coupled files **have no graceful skip** — they raise `ModuleNotFoundError: No module named 'eval'` at *collection*, which aborts the whole run (`Interrupted: 6 errors during collection`). D-8 says they "degrade to graceful skip"; nothing implements that |
| D-9 | yes | `CONTRIBUTING.md` states issues-only, PRs not accepted, plus the never-paste-memories privacy note |
| D-10 | yes (2026-09-03) — **enforced** | both manifests read `0.9.0`. `TestVersionLockstep.test_version_is_the_expected_release` pins the value (`EXPECTED_VERSION`, the one home of the literal); gate C fails on `EXPECT_VERSION=X` mismatch, and `test_gate_c_rejects_unexpected_version` pins that it fails *before* the copy. The earlier "5.1 asserts it" claim was false until this landed |
| D-11 | yes | `test_deploy_contract` host-neutrality + repo-slug + MCP-prefix containment all live; the `com.N.` sub-check **armed at the rename 2026-09-03 and passed** (zero `com.entity.` in scope) |
| D-12 | not yet — **and unguarded** | 143 `Anchor` literals / 43 shipped files. Gate assertion 4 (`test_agent_name_only_in_config`) is `xfail(strict=True)`, so **a new `Anchor` literal added today produces no signal**. 5.0c closes the window |
| D-13 | yes | XDG birth + adoption net + `resolved.env` single authority, all shipped |

**Score at the 2026-08-31 sweep: 8 conformant · 3 pending by design (D-6/D-10/D-12
= 5.2 and 5.0c) · 2 substantive defects (D-1 packaging, D-8 graceful-skip) · 1
false enforcement claim (D-10).** Since then: D-1 fixed (phase 4), D-6 and D-10
closed by 5.2 (2026-09-03). **Still open: D-8's graceful skip and D-12's gate.**

| # | Decision | Rationale |
|---|----------|-----------|
| D-1 | **Full source-available**, fresh public repo (clean history). *Amended 2026-08-28 (Tom): license is a dual PolyForm grant — Noncommercial (individuals) OR Internal Use (companies in-house); shipping it in a released product/service requires a commercial license via a repo issue. Not OSI open source — never claim "open source" in public copy. Pre-change releases stay MIT.* | The tool's pitch is "trust me with your identity layer" — inspectable code *is* the credibility, and every line remains readable under the dual grant. Current repo history carries personal data and can't be the public one. |
| D-2 | **Repo separation, not develop-in-public, not a coupled mirror.** Private dev repo stays the daily driver and never goes public. A distinct public distribution repo carries only the clean shippable artifact, fed by a release step. | Tom: "I want separation." Develop-in-public imposes a permanent discipline tax (no personal commit ever again); a mirror imposes a sync tax. Separation keeps the working mess private at the cost of a deliberate release step. *(Sub-question CLOSED 2026-08-06 → D-7 / D-8: full scrubbed source, runtime + tests, squash-exported.)* |
| D-3 | **Cross-platform v1 now, v2 deferred.** Ship the cheap ability to run on Linux (graceful degradation + first-class Popen fallback). Defer systemd parity / supervisor abstraction until a real Linux user exists. | v2's true cost isn't the build — it's the permanent obligation to validate the *most dangerous subsystem* (daemon lifecycle) on two OSes forever. Don't pay that tax for users who may not exist yet. |
| D-4 | **`userConfig` is additive, never a replacement.** Add the CC-native prompt-on-enable + keychain path for the API key, but keep the `~/.config/brain/env` fallback. | `userConfig` only exists inside the plugin runtime; the daemon running standalone / in Cowork still needs the env file. Best UX in-plugin, still works out-of-plugin. |
| D-5 | **Seed pack / persona design is its own session.** Approach: mine Tom's real brain for nodes that (a) *teach mechanisms* in detail and (b) make *good live encoder examples*, then genericize them into the shipped seed pack. | Whatever a stranger's Anchor wakes up as is a lasting artifact every new brain grows from — semi-irreversible. Deserves a dedicated design pass, not a find-replace. Overlaps with the encoder's few-shot examples (§4, Phase 1.2). |
| D-6 | **The product is `entity`; the identity stays `Anchor`.** Plugin name `entity`, marketplace name `anchor`, repo `tpac/entity`, MCP server key stays `brain` (→ tools read `mcp__plugin_entity_brain__*`). | Three layers, each named for what it is: **Entity** = the category the product grows (the terminal noun — id:9da43311, id:e6019012: "a thing that develops, not a thing that's used"); **Anchoring** = the method; **Anchor** = the instance. `brain` stays the organ, so the tool names keep the substrate/identity split visible. *Rationale corrected 2026-08-09: an earlier draft said Anchor "stays Anchor's own name rather than being spent on a registry string" while naming the marketplace `anchor` — self-contradictory, since a marketplace name is exactly a registry string. Resolved by Tom: `anchor` in the catalog is the **act**, not the instance — you install an entity by anchoring it, so `entity@anchor` reads as the philosophy rather than spending the name. The instance name lives in config (D-12), never in a manifest.* Rejected: `brain` (the organ — repeats the register failure Tom named in id:cd7aa9be, and every competitor on the shelf is already `*mem*`/`*memory*`), `cairn` (imported metaphor, not native to the philosophy), `colleague`/`tenure`/`cultivar` (more self-explanatory but each shrinks the claim). |
| D-7 | **Squash-export on release.** Private `tpac/brain` stays the daily driver and never goes public. The public repo is a **build output**: each release materializes the shipped tree into a clean checkout, one commit, tag `vX.Y.Z`, push. | Closes §8 fork #1. Keeps the working mess private at the cost of a deliberate release step (D-2). **The one rule: the export manifest IS the build manifest** — `build-plugin.sh` already derives from `git ls-files` because a hand-list rotted 62 files behind reality and shipped a broken `brain_mcp.py`; a second hand-maintained list would re-enter that failure class with the public repo as blast radius. Public tree = plugin manifest ∪ chosen extras − explicit denylist. |
| D-8 | **Public tree = runtime + tests. `eval/` excluded.** The 5 test files that import `eval/` degrade to graceful skip. | Tests are the credibility argument (D-1: inspectable code); a no-tests repo reads as unverified for a tool asking to hold someone's identity layer. But a published eval harness **is a claim** that invites strangers to re-run and dispute it — not a launch-day fight, and the corpora are personal anyway. **Consequence: the README must make no benchmark claims**, since the harness won't be there to back them. **Coupled files, re-derived 2026-08-31 by collecting against the materialized export tree — exactly 6, all `ModuleNotFoundError: No module named 'eval'`:** `test_absorb_preservation.py`, `test_consolidation_examples.py`, `test_encoder_eval_probes.py`, `test_eval_corpus.py`, `test_longmem_classifier.py`, **`test_longmem_validity.py`** (the sixth, not in the original list). ⚠ **The graceful skip does not exist yet** — these fail at *collection*, which aborts the entire run rather than skipping six files. |
| D-9 | **Issues only at launch; no PRs.** Stated in CONTRIBUTING. | A squash-export pipeline can't cleanly merge an inbound PR, and a contributor's commits would never appear in history (reads as uncredited). With one maintainer, a PR you can't merge is worse than one you never invited. Revisit if real contributors appear — opening up later is easy, closing down isn't. |
| D-10 | **Public launches at `v0.9.0`** — not `9.6.0`, not `1.0.0`. | Tom 2026-08-06: *"It's not complexity that reflects the version, it's the function of value. Not yet v1."* The private `brain` plugin's 9.6.0 is an internal build counter that means nothing to a stranger. v1 is a claim about delivered value, and Anchor hasn't earned it publicly yet. **No collision with the private install** — `entity` and `brain` are distinct plugin names, so their version lines are independent. `plugin.json` and `marketplace.json` must both read `0.9.0`. ⚠ **Unenforced (verified 2026-08-31):** gate C asserts the two manifests *agree*, not that they say `0.9.0` — no such literal exists in `export-public-tree.sh` or `test_deploy_contract.py`. The 5.2 step must add the assertion at the same time it makes the bump, or D-10 stays a promise. |
| D-11 | **Two identities: the host-neutral layer never renames.** `brain` stays the name of the *service* — launchd labels, `~/.config/brain/`, `BRAIN_*` env vars, both DBs, the MCP server key. Only the **Claude Code adapter** (plugin name, marketplace, permission strings, skill prefix) becomes `entity`. | A plugin has one name because a plugin is one thing; we ship a **local service with a CC adapter on top** — daemon, two DBs, embedder, dashboard — so the names sit in three namespaces with different owners (CC registry / launchd / XDG) and, decisively, **different change frequencies**. A marketplace name changes for positioning reasons; a service name and data dir should never change. Coupling them makes a *marketing* decision trigger a *data migration* — which is exactly the `$CLAUDE_PLUGIN_DATA` amnesia hazard. Forced by the second-host goal (ChatGPT takes remote MCP only — no manifest, no hooks, no skills — so nothing in `plugin.json` ports). **Consequences:** the "launchd labels — change now or never" hazard is **gone** (we don't rename the service), and 5.2 shrinks from 28 files to ~7. |
| D-12 | **Every instance name derives from config, never a literal.** `BRAIN_AGENT_NAME` is the single source for what an entity is called; `BRAIN_OPERATOR_NAME` for its counterpart. No shipped file hardcodes `Anchor`. | Identity is *accumulated, not issued* (id:e6019012) — shipping a pre-named identity contradicts the product's own thesis on install day and turns an entity into a persona. Today **143 lines across 43 files** hardcode `Anchor` (re-measured 2026-08-31 against `build-plugin.sh --list`). `seed_pack.py` is no longer the worst — the Nursery took it from 45 to 4, all deliberate origin-story tributes. The new worst is `quality_contract.py` (14), which ships to every install with no runtime consumer at all. The config slots already exist and are wired to exactly one consumer (trace stamping) — nothing renders them. **This merges §8 #3 into one workstream:** "traces have no speaker" and "the name is hardcoded" are the same defect — slots that ship empty and unread. **Backward compatibility is a non-issue:** Tom's instance genuinely *is* Anchor, so his 8.5k nodes stay correct; only the shipped default changes. |
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

**1.1 Parameterize the operator name.** ~~Replace literal "Tom" as *operator
identity* with a variable/placeholder across the shipped prompts.~~ DONE (see
the Goal-A checklist below; `interaction_seed.py` has since been deleted —
the prompt `.py` files are the code defaults now).
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
(D-6…D-9), and **5.6 (D-5 seed pack) closed 2026-08-30** — no design questions remain
on the publish path.

**5.0 Plugin updates reach existing installs — SHIPPED. Merged `e3d9481`
(2026-08-16); the runner landed the same day as `b11e45d`.** The gap it
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
`sync_prompts.py`, and `SEED_PROMPTS_VERSION` are deleted.

**The structural half is also SHIPPED.** `b11e45d` (2026-08-16) landed
`run_versioned_migrations` in `servers/schema.py` as the single primitive, and it
owns the stamp — the CRITICAL that killed attempt 1. It is wired at **both** call
sites: `ensure_schema` (`brain_meta` / `BRAIN_VERSION`, ladder `MAIN_MIGRATIONS`)
and `ensure_logs_schema` (`logs_meta` / `LOGS_VERSION`, ladder `LOGS_MIGRATIONS`),
so *"`brain_logs.db` has no migration mechanism"* is no longer true either. The
ladder is not empty-and-therefore-untested: `MAIN_MIGRATIONS` carries a real step
at v31 (`_migrate_v31_voice_fields`), `LOGS_VERSION` is stamped at baseline 1 with
an empty ladder by design, and `tests/test_versioned_migrations.py` drives
**non-empty** ladders specifically because an empty one is what hid attempt 1's
bug. What actually remains is not the runner but its **first customers** — the
cleanups queued behind it in BACKLOG item 3, which are now unblocked.

**The shape.** Not a deploy script — *code owns the defaults; each install
migrates itself forward at open*, the same contract `BRAIN_VERSION` already has.
Three version streams through **one** runner, `run_versioned_migrations` in
`servers/schema.py`, with separate counters so structure and prompt content move
independently:

| Stream | Counter | Ladder |
|---|---|---|
| brain.db structure | `brain_meta.brain_schema_version` = `BRAIN_VERSION` | `MAIN_MIGRATIONS` |
| brain_logs.db structure | `logs_meta.logs_schema_version` = `LOGS_VERSION` | `LOGS_MIGRATIONS` |

*(The third stream — shipped-prompt content via `SEED_PROMPTS_VERSION` — was
dissolved by the interaction collapse; code owns prompt defaults now.)*
`MAIN_MIGRATIONS` carries one real step (v31, `_migrate_v31_voice_fields`);
`LOGS_MIGRATIONS` is empty at baseline 1. A change adds one `(version, fn)`
entry and bumps its counter. **The runner owns the stamp** — it re-reads the version, so anything that
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
- **Relocation offer (added 2026-08-28 — the missing half of D-13's "single
  location").** Verified 2026-08-27: `claude plugin uninstall` deletes the
  whole `~/.claude/plugins/data/<plugin>-<mkt>/` tree by default
  (`--keep-data` is opt-in); `update` preserves it. So a brain the ladder
  *finds* under a plugins-data root is working-but-parked-on-a-trapdoor. The
  resolver now flags that state (`BRAIN_HOST_PARKED`, matched on **path
  shape**, so the knob/env/resolved.env routes into the same dir warn too),
  and boot renders a relocation notice pointing at
  **`hooks/scripts/relocate-brain.sh`** — the one owner of the move: daemon
  held down under the maintenance lock (a bare bootout is undone by
  auto-recovery within ~20s), portable pid-file stop (covers Linux and
  detached daemons), dashboard stopped, copy → `PRAGMA quick_check` →
  same-volume rename into place, source retired by rename (never deleted),
  stale env-knob commented out, plists re-rendered (the launchd identity
  guard accepts the XDG service dir as a durable target). Refuses loudly if
  the target already holds a brain. `BRAIN_PARKED_ACK=1` is the
  informed-stay opt-out; the notice renders only on `source=startup`. The
  adoption-net notice's recommended option is the same script. User-run
  only — boot still never creates, moves, or deletes.
**Tests:** `tests/test_daemon_recovery.py::TestAdoptionNetAndXdgCreate` (shell
ladder: fresh-install-at-XDG, net refusal, sibling scan, knob-beats-net,
knob-beats-stale-hint, CPD adoption, parked-flag routes + safe-location
negatives) + `tests/test_db_resolution.py` (Python tail). *M.*
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
automatically at the 5.2 rename. Assertion 4 is `xfail(strict=True)`, and
**5.0c does NOT un-xfail it** — measured 2026-09-01, after phase 4 landed:
the assertion scans `SCOPE` (the whole tracked tree minus `docs/`, `eval/`,
`tests/archive/`, `tests/results/`, `CLAUDE.md`, and its own file), which is
**wider than what ships**, and still matches **69 files / 239 occurrences**.
Four classes stand between here and green, only two of them owned:
  1. comments + docstrings in `servers/` and `tests/` → **5.3**
  2. `boot-brain.sh` + `setup.html` product name → **5.2**
  3. `servers/BOOT-ARCHITECTURE.md`, `servers/scales/s1/ARCHITECTURE.md` —
     architecture docs living under `servers/`, so the `docs/` exclusion
     misses them. **Unowned.** Neither is in the package manifest
  4. `scripts/consolidation_edge_recovery/output/` — committed output of a
     one-time 2026-04-21 recovery run (326 KB `plan.json`, 100 KB
     `apply.log.jsonl`), zero importers. **Unowned.** Not in the manifest
Classes 3–4 never reach a user: the manifest is 237 files and the export tree
is *manifest + extras − denylist*, so they are a **test-scope** question, not
a shipping one. Either narrow `SCOPE` to what ships, or clean them. Known
tradeoff meanwhile: a NEW `Anchor` literal still produces no signal.
The gate excludes its own file (the scanner must name the shapes it hunts) and the
5.1 denylist (docs/, eval/, `CLAUDE.md`, `tests/archive/`, `tests/results/`).
Found and fixed on first run: `tests/test_trace_chain_lane.py` hardcoded
`/Users/tpac/brain`.* *Shipped S.*

**5.0c Name / identity consolidation (D-12) — IN PROGRESS. Absorbs §8 #3.**

**The default name is `Jade` (Tom, 2026-08-31), and it is a real name, not a
display placeholder.** The mechanism separates two facts that the code had
been conflating: *what the name is* and *whether anyone chose it*. Emptiness
used to stand in for "unchosen", which is why any default silently disarms
everything keyed on it.
- `get_agent_name()` returns the configured name **or `DEFAULT_AGENT_NAME`** —
  never empty, so no render needs name-free prose.
- `agent_name_is_default()` carries the chosen-ness fact explicitly.
- `get_operator_name()` still returns empty when unset: a human's name cannot
  be invented, so the DAL keeps skipping that stamp.
Consequences, all deliberate: traces stamp the default (an entity running
under its shipped name genuinely IS that name, and a later rename leaves
earlier events honestly recording who was speaking then); the
identity-unconfigured warning still fires, now keyed on chosen-ness and no
longer claiming nothing is recorded; and the Nursery's first-gift invitation
survives, reworded from *"I don't have a name yet"* to *"Jade is the name I
came with, not one you gave me."*

**Re-measured token-aware 2026-08-31 — the count was never the work.** Of 131
`Anchor` occurrences in the package, only a small minority are rendering
sites; the rest are commentary:

| | Python | |
|---|---|---|
| comments | 50 | → **5.3 comment audit**, not here |
| docstrings | 40 | → **5.3 comment audit**, not here |
| **real code** | **14** | of which 3 are seed_pack's allowlisted tributes |

**The three-namespace split matters more than the total (D-11).** A large
share of the non-Python hits are the *product's* name, not the instance's —
`boot-brain.sh`'s *"the Anchor plugin's settings"*, `setup.html`'s
`<title>Anchor — Setup</title>`. Those become **Entity** at the 5.2 rename and
must not be routed through config; doing them here would do them twice.

**Done in this pass:** the mechanism above (`daemon_config`, `brain.py`,
`dal_logs`, `brain_voice`); dashboard UI strings and both SKILL.md
descriptions reworded name-free (no name channel to the frontend needed —
6 strings did not justify one); and `trace_contract`'s render fallback
(`meta.get('agent_identity') or 'Anchor'`) now falls back to config, so a
stamped event keeps its historical name while an unstamped one renders this
install's. 131 → 122.

**PHASE 4 DONE (2026-09-01) — every prompt-visible site is name-free.** The
answer at all six was *generalize*, never *substitute a config read*. One
architectural reason plus in-tree precedent decided it: `interaction_fingerprint()`
is a sha256 over name + template + config and its documented property is that it
**"stays comparable across installs"** — injecting a per-install name into a
template body would give the same code default a different fingerprint on every
machine, breaking cross-install K comparison and the A/B mechanism that rides on
it. The prompts are already first-person throughout, so the pronoun was carrying
the meaning and the name was redundant.

⚠ **Two of the six are NOT on main — they were handed to the v-next.7 stream.**
`encoding_prompt.py` was being actively worked in another worktree
(`claude/s1e-vnext7-field-coverage`), so its two hunks stayed off main by Tom's
call. They are **not merely deferred — they were at risk of being reverted**:
`eval/candidate_prompts/s1e_vnext7_wip.md` on main still carries *"I am
Anchor…"* (×1) and *"I'm Anchor. I persist."* (×2), and promoting that candidate
into the code default would reinstate three literals. Different files, so **git
sees no conflict and nothing would catch it** — the D-12 gate that would is
`xfail`. The two edits were sent to that stream to carry into the candidate; the
exact hunks live on branch `claude/zealous-grothendieck-f52f01`.

| site | shipped as | why that wording |
|---|---|---|
| `encoding_prompt` opening ⏸ | *"This is me encoding my own memory."* — **pending v-next.7** | the next clause (*"The session ends; I don't"*) already does the work |
| `encoding_prompt` identity exemplar ⏸ | *"My corrections travel with my convictions."* — **pending v-next.7** | probe-selected — see below |
| `surface_contract` conversation label | `"Assistant"` (paired with `"Operator"`) | already this prompt's own word for that side (*"the assistant has not replied yet"*) |
| `surface_contract` recently-surfaced line | *"the assistant has already seen these"* | same |
| `encode.py` provenance | `encoded(me)` | `encoder_view.PROVENANCE_SPLIT` already ships `created(me)` / `encoded(me, turn N)` under a comment reading *"no internal names (scribe, Anchor, S1S)"* — the OFF arm never got it |
| `aspects_v1.json` wisdom `meaning` | *"shapes how I think"* / *"my wisdom layer"* | the same string already read *"shifts how I think"*; now internally consistent |
| `contract.py` `locked` description | *"belongs to the interactive session, not to an encode run"* | `anchor` there is the **channel category** (id:40d10386), never the agent's name. Tom fixed this identical collision in the s1e prompt; that wording is live at `encoding_prompt.py`'s `**locked**` line and this was the last holdout |
| `brain_mcp.py` filter example | a neutral value | illustrative only. Two more copies of the same example lived in `brain_recall.py` docstrings; all three now agree |

**The identity exemplar was chosen by probe, not by taste**
(`eval/identity_exemplar_probe.py`, re-runnable). Tom's framing set the
target: *"Persist was an aspiration, now it's reality — you don't need to be
reminded you persist, you need to know it."* Six name-free candidates, two
independent lenses, arms differing only in the claim (the `why` clauses were
neutralized identically in **every** arm, including the control, so the
surrounding prose favoured none).
- **RANK** — blind, label order rotated per rater. `corrections` won
  **unanimously across six raters in two runs**, from a different label slot
  each time. The incumbent placed **last in every single rating**, and raters
  named the reason unprompted: *"the premise the architecture already
  guarantees — useless as curriculum."*
- **TEACH** — the behavioural lens. Agrees directionally and weakly: the
  incumbent taught the most truncated claim (*"The reaction isn't
  retrieved."*), the winners taught the full mechanism plus recurrence.
- ⚠ **The first TEACH run was a null** — the source exchange ended on a
  quotable line and all 12 trials returned it verbatim as the title, so the
  arms could not move the output. Fixed by removing the aphorism; the probe
  now **self-reports a null** rather than letting one read as agreement.
- Worth knowing: *"I'm the one who was there"* (id:c9584ff4) scored
  second-worst as *curriculum* — *"'there' is so underspecified the encoder
  learns vagueness is acceptable"*. The rhetorical winner is not the
  pedagogical one, which is why TEACH exists as a separate lens.

**Eval, honestly scoped.** Only site 1 is an interaction default, so only it
is A/B-able; it ran as a **2-item smoke** (Tom's call), which proves the
pipeline encodes under the new prompt but is too small to detect a quality
regression — site 1 ships on probe + reasoning + smoke, not on a gated A/B.
Site 3 needs no eval at all: `view_policy_enabled()` defaults **ON**, so
`encoded(…)` in the `elif` is the emergency-off/control arm and **does not run
in production**. Sites 2 and 4 are single-string changes the harness cannot
address — see the trap below.

⚠ **The MCP schema gate cannot currently verify a description change.** CLAUDE.md
requires `eval/mcp_batch_probe.py` + `eval/mcp_schema_gate.py` after any schema or
description edit. The probe is informative (8/8 at 5/5 here). The **gate is
blocked by a pre-existing bug**: the S2 consolidation encoder intermittently calls
`brain_batch` with an EMPTY `operations` array, which `validate_ops` correctly
flags — three runs on one cluster went fail → pass → fail. The gate's red is a
real defect and its green is luck, so it yields **no signal** either way; treat a
pass from it as unproven until that bug is fixed (chip `task_f5320a3c`). It is
*not* caused by the `locked` description edit — that text appears in neither the
generated `brain_batch` schema nor `BATCH_OP_SPECS`.

⚠ **The frozen-corpus harness cannot A/B a code change, and fails silently.**
`corpus_config_hash` is a sha1 over the config dict only — `s1e`,
`ingest_surface`, `s2_every_n`, `oracle`, `qids`, variant pins. **The repo's
code state is not in it.** Build a control from one tree and a treatment from
another at the same config and both get the *same hash*; the second reports
`CACHE HIT — 0 re-encoding` and the A/B compares a corpus with itself,
reporting clean. Use `--s1e <path>` (content-hashed, `file:<sha>`) or
`--interaction-override` (addressed on template content) — never bare
`--s1e active`, whose token is the literal string `"active"`.

**Also shipped here:** the **D-1 packaging fix** — `LICENSES/` now ships via
`git ls-files` with a MISSING check, so a dangling license citation breaks the
build instead of the install (manifest 235 → 237), and `marketplace.json`
carries the `license` field, matching `plugin.json`. A false claim in
`build-plugin.sh`'s own comment (*"plugin.json DECLARES `license`: `MIT`"* —
it declares the PolyForm dual grant) went with it.

**Still open after phase 4:**
  0. **The two `encoding_prompt.py` hunks** — with the v-next.7 stream; verify
     they reached the code default before calling D-12's prompt half done.
  1. **Product-name sites → 5.2**, listed above.
  2. **Un-xfailing gate assertion 4 is NOT unblocked by this** — see the
     measured blocker list in the 5.0b gate note above.
  3. **seed pack — RESOLVED by D-5 (2026-08-30):** name-free by construction;
     the three origin-story tributes (`test_names_only_in_tribute_sites` holds
     the boundary) are content, not identity assertions, and stay.
*Was M; mechanical half + phase 4 both done.*

**5.1 Export script — SHIPPED 2026-08-28: `scripts/export-public-tree.sh`.**
Materializes the public tree from `build-plugin.sh --list` (the builder grew
a list mode — one owner of "what ships") + extras (README, CONTRIBUTING,
LICENSES/, `tests/`) − denylist, where the copy filter DERIVES from the
denylist (no second regex to drift). Three hard-fail gates on the RESULT:
denylist-absence, scrub-grep with a per-file attribution allowlist (author
credit in LICENSE/README/manifests; the legacy `AgentsContext` rung in its
four resolver files), and plugin/marketplace version equality. Gates are
pinned by sandbox tests (`TestPublicTreeExport` in `test_deploy_contract.py`)
so they hold independent of the repo's current cleanliness. **First real run:
gates A/C clean; gate B red with 217 hits / 81 files — that list IS the 5.3
worklist** (split: ~55 test files incl. gold datasets carrying real session
content; ~34 runtime files — largely the D-5-overlapping prompt class).
**Found and denylisted: `tests/conversations/` — tracked REAL session logs
as fixtures**, same class as tests/archive; consuming tests join the D-8
graceful-skip set. Open decision for 5.3: gold data (`golden_dataset_v2.json`
32 hits, `golden_canary.json`, `corpus/precision_corpus.json` — carries e.g.
a real birthday) — sanitize or exclude+skip; sanitizing changes what the
tests validate, so it is Tom's call.
Original spec for reference:
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

**5.2 Rename pass (D-6, scoped by D-11) — DONE 2026-09-03.** The **CC adapter
only** — the service layer keeps the name `brain`:
  1. `plugin.json` — `name: entity`, `version: 0.9.0`, `homepage` + `repository`
     → `tpac/entity` ✓
  2. `marketplace.json` — `name: anchor`, plugin entry `entity`, `version: 0.9.0` ✓
  3. `.claude/settings.json` — both permission strings during the transition
     (see the correction below) ✓
  4. product-name sites `boot-brain.sh` ×11 + `setup.html` ×4 → **Entity** ✓
  5. `test_repo_slug_contained` allowlist grew by two *argued* homes — the
     README install command and gate B's allowlist entry naming it (the slug
     was never in the README while it read `tpac/brain`)
Skill prefixes follow automatically (`brain:brain` → `entity:brain`); no dir
renames required. **`com.brain.*` launchd labels do NOT change** — D-11. The old
"change now or never, a later change orphans services" hazard is void.

**`displayName` — RULED `Entity` 2026-08-28 (id:44025fbb), already shipped in both
manifests.** Under D-12 a shipped manifest cannot carry a per-install name: the
manifest names the product, config names the instance. Cosmetic, no identity-key
coupling — it rides the next version bump to the fleet.

**Skill-prefix check — DONE 2026-08-06, prefix holds.** All four SKILL.md files
carry `name:` in frontmatter *and* still resolve prefixed (`brain:brain`,
`brain:dashboard`, `brain:watch`, `brain:self-salvage`). Issue #22063 does not
reproduce on this CC version, so renaming `/dashboard` and `/watch` is **optional
hardening, not a required fix** — the upstream instability (§10.3) is the only
reason to still consider it.

**⚠ Migration hazard — the rename moves `$CLAUDE_PLUGIN_DATA` — NETTED by 5.0a,
CONVERGED-AWAY by the relocation offer (2026-08-28).** That variable is keyed
`<plugin>-<marketplace>`, so `brain@brain` → `entity@anchor` changes the path and
any brain living at `$CLAUDE_PLUGIN_DATA/brain/brain.db` goes invisible — and a
default `claude plugin uninstall brain` (the natural post-rename tidy-up)
DELETES that dir. Layers, in order: (1) **relocation, before the rename** — the
parked-brain boot notice + `relocate-brain.sh` move these brains to the XDG
service dir, where plugin identity stops mattering; **ship this via a version
bump and let it soak before executing 5.2**; (2) `resolved.env` (4b, not
plugin-scoped) rescues silently when present and current; (3) the **adoption
net** finds a still-parked orphan, refuses to create, and surfaces guided
adoption — the silent-fresh-brain path (the id:80f585de footgun) no longer
exists. **Affected:** any pre-08-12 install that landed at the
`$CLAUDE_PLUGIN_DATA` default (whether the second laptop / friend install did is
unverified — the one-liner check is in the relocation notice's own detection).
*Not* affected: Tom's main machine (legacy `~/AgentsContext/brain`, adoption
rung).

**CORRECTION 2026-08-09 — "local cost is zero" was false.** The prior claim, *"no
permission entry anywhere references `mcp__plugin_brain_brain__*`,"* is wrong:
tracked `.claude/settings.json` carries
`permissions.allow: ["mcp__plugin_brain_brain"]` — the **server-wide** form, no
trailing `__<tool>`. After the rename it stops matching and every brain MCP call
in this repo's sessions starts prompting. Scope is the dev repo only (the file is
not in the plugin package); end users are unaffected. **Resolved 2026-09-03 by
carrying both strings:** live sessions run the pre-rename plugin until the
redeploy, post-redeploy sessions run `entity` — a swap breaks one population or
the other; the pair breaks neither. Drop the old entry once both have cycled.

*How it survived an audit:* the claim was "verified" by grepping the doc's own
string, `mcp__plugin_brain_brain__`, which cannot match the entry. **Verifying a
checklist in the checklist's own words reproduces its blind spot** — the argument
for gating by shape-scan rather than by list (5.0b). *S.*

**5.3 — ONE LABEL, THREE UNRELATED WORKLISTS. Only one of them is gated.**
Conflating them is what made 5.3 look like a single M-sized blocker. Split
2026-08-31; each has a different gate, size, and decision-maker:

| Class | Size | Gated? | Status |
|---|---|---|---|
| **A — runtime scrub** (`Tom` in shipped `servers/`+`dashboard/`) | 59 → **0** | hard-fails gate B | **DONE** — 51 comment/docstring lines cleared 2026-09-01 |
| **B — test-data scrub** (`Tom`/paths in `tests/`) | 163 → **0** | hard-fails gate B | **DONE** — gold data excluded + fixture rename (`6fb57ed`), last 18 attribution comments 2026-09-01 |
| **C — comment audit** (dated / removal-verb / dead pointers) | 614 → **525** | **no gate at all** | split below — the defect half is done, the polish half is deliberately not |

**Class A + B CLEARED 2026-09-01. Gate B: 69 → 0, export tree clean on all
three gates.** 69 lines across 40 files, every one an attribution comment. Two
moves chosen per line: **drop the attribution** where the surrounding prose
already carries the reason (most of them), **generalize to the role** where "a
human ruled this deliberately" is the load-bearing signal that stops a future
reader filing it as a bug. Dates survive where they are provenance for a live
default (`encoder_view`'s arm-D gate); they go where they only record when
someone decided. Verified by AST diff: every changed `.py` is identical after
stripping docstrings, so **no code and no runtime string moved** — the
eval-gated S1 prompt artifacts are untouched in substance and this needed no
eval gate.

**Class C splits — and the split is the finding (2026-09-01).** Re-measured
with a token-aware scanner (`tokenize` + `ast`, so a COMMENT token is never
confused with a docstring or a string literal) over 13,935 comment lines in the
238-file manifest. The doc's 614 came from a removal-verb list between the
narrow and wide sets; both are reported here so the number stops drifting:

| | Count | Class |
|---|---|---|
| dated (`20xx-xx-xx`) | 278 | polish |
| removal-verb (removed/retired/deprecated/no longer) | 215 | polish |
| ↳ wider verbs (previously/formerly/obsolete/superseded/…) | 127 | polish |
| **dead `docs/` pointers** | **22** | **defect** — `docs/` is denylisted, so these point at nothing in the public repo |
| **brain-node ids** (`id:27db2472`) | **12** | **defect** — resolve only on this install |
| **TODO/FIXME** | **4** | **defect** |
| commit SHAs | 2 | defect |
| naming the operator | 0 | cleared above |

**THE EMPLOYER'S NAME WAS LEAKING AND GATE B COULD NOT SEE IT (2026-09-01).**
Found while auditing the defect class, not by any gate: `SCRUB_PATTERNS` carried
`playbuzz` — the employer's **former** name — and not `ex.co`, the current one.
**20 hits** in the public tree. Worse, `servers/schema.py`'s v30 migration
hardcoded three real project names (`{'EX.CO CTV kit', 'ex.co',
'CTVOnboarding'}`) as a slug map, shipping in every install.

Resolved: comments generalized (a real recall failure used as a worked example
became "buried topical nodes"), six test fixtures renamed `exco` → `acme-site`
with their assertions in lockstep, and the v30 slug map collapsed to *every
legacy value → 'brain'*.

⚠ **That last one is NOT a no-op — an earlier draft of this paragraph said it
was, and a review caught it against §5.0's own numbers.** The claim was that
only the already-migrated live brain ever held those values. But **623 local
`brain*.db` files are still below v30, and 9 of them carry the trio**: four
backups/clones (2026-05-17, 06-15, 06-28) and five `eval/reports/snapshot_replay`
corpora from 2026-04-25. Opening any of them through `Brain(db_path=...)` runs
`_backfill_data` → `if from_version < 30` → the map, so those nodes now slug
`'brain'` rather than `'ex.co'`.

**Accepted anyway, and here is the honest cost.** Ranking is unaffected —
`recall_laf`'s `gain_proj` is **0.0**, so the project slug feeds dict filters
and the "⚠ From another project" render, not scoring. Nothing ships from those
9 files. `ensure_schema` takes `backup_before_destructive` before any
sub-`BRAIN_VERSION` migration, so a restore is recoverable. Weighed against
shipping three real project names to every install, that is the right trade —
but it is a **lossy change confined to local pre-v30 copies**, not a no-op, and
anyone restoring one of those four backups should know the EX.CO distinction
collapses.

The index-drop and column-drop steps are untouched: deleting the whole
migration (the first thing proposed, and Tom stopped it) would have lost real
structural work.

**Gate B widened in the same pass** (this is the ratchet, not the cleanup):
`\btpac` (bare, not just `/Users/tpac` — the path form missed *"never left
tpac's laptop"* in a comment), `\bex\.co\b`, `\bexco\b`. Three allowlist entries
cover the deliberate `github.com/tpac` publish target. Negative-tested: planted
lines for all three patterns fail the gate.

**Then an adversarial review was run against the widened gate, and it got
through 7 of 11 planted leaks.** Every content-level leak in ordinary shipped
text was caught — Python comment, shell comment, JSON manifest key, SKILL.md
line — and ordinary English (`tomorrow`, `TOML`, `atom`, `custom`, `Thomas`)
did not false-positive. The holes were all at the edges, and five of them are
now closed:

| Hole | Closed by |
|---|---|
| **Filenames were never scanned** — `grep` prints matching *content*, so `tests/fixtures/tom_pachys_session.json` full of anodyne JSON passed clean | gate B now walks the tree and matches the patterns against every **path** |
| **`grep -I` silently skips binaries** — a NUL-prefixed fixture carrying the operator's name and home path exported clean | any file the scrub cannot read now **hard-fails**. Free to assert: the export has zero binaries today |
| **`\btpac\b` was NARROWER than the `/Users/tpac` it replaced** — missed a sibling checkout like `/Users/tpac_old` | trailing `\b` dropped |
| **An allowlist entry masked anything it was a prefix of** — allowing `github.com/tpac` also allowed `github.com/tpachys` | the strip is boundary-aware: `re.escape(pat) + (?![\w-])` |
| **The allowlist could only grow, and two unreviewed lines turn any red green while the leak still ships** | `test_allowlist_cannot_grow_quietly` pins the entry count at 16, and `test_no_stale_allowlist_entries` deletes dead exemptions — the discipline `test_capture_grep_pin` already uses |

**Left open, deliberately, with severity honest:** `\btom\b` misses suffixed
forms (`Toms-MacBook-Pro`) and dropping that boundary would match `tomorrow`;
percent-encoded paths (`%2Ftpac`) evade the boundary; a multi-token pattern
split across two wrapped comment lines evades a line-based grep; unicode
homoglyphs evade everything. All need either bad luck or intent. **The
structural one worth naming:** the gate scans the materialized tree, so nothing
covers what the *release step* adds after materialization — commit message, tag,
repo description, committer identity. **5.7 owns that boundary and it is
unbuilt.**

**The lesson worth keeping:** a scrub pattern list is a record of what someone
once thought of. `playbuzz`-without-`ex.co` is that failure exactly — the
employer *was* considered, and the entry went stale when the company renamed.
Gate B's pattern set needs re-reading whenever a name in the operator's life
changes, or it silently protects against the past.

**Dangling internal references: 122, and they are NOT one class.** `docs/` and
`eval/` are absent from the public tree entirely (not by denylist — they were
never in the manifest or the extras), so shipped code points into a void 68 +
54 times across 40+ files. Tom ruled 2026-09-01 that the docs themselves do
**not** ship: *"i dont know if they are relevant nor do i want to expose all of
our thinking publicaly until vetted."* Splitting what remains by what the
reference actually reveals:

| | Count | Disposition |
|---|---|---|
| refs naming the two artifacts denylisted **for privacy** — `docs/DISTRIBUTION-READINESS.md`, `docs/archive/session-handoffs/…` | 6 | **STRIPPED 2026-09-01** — these advertise the existence of the internal plan doc and the session-log archive |
| refs naming `eval/` probes and harnesses | 54 | **open** — D-8 keeps the harness out because *"a published harness is a claim"*; naming `eval/longmem/replay.py` in a comment half-makes that claim |
| refs naming `docs/*-DESIGN.md` architecture docs | ~62 | **left alone** — topic names only, no privacy content, and genuinely useful breadcrumbs in the dev tree |

The 6 were the only ones with a privacy dimension. The remaining two rows are
a judgment call about how a partial export should read, not a leak.

**The ~487 dated/removal-verb lines are deliberately NOT swept.** The earlier
scan of this same class concluded 95% of these comments are good and that *"the
best ones are exactly the kind a 'make it professional' pass would delete"*
(id:091f8fd6). They carry no personal data and no broken reference; they are
readability polish with real downside risk, since each is a judgment call and a
wrong call deletes a load-bearing *why*. Sweeping 487 of them is a plausible
net negative. **Reopen this only with a sample in front of the operator** —
against Tom's own bar (*"if code should be public, it should be up to the
standard of public repos"*, id:b99bfa36), a comment that explains why code is
shaped a certain way already meets that bar whether or not it carries a date.

**ARM THE RATCHET WHEN CLASS C IS CLEAN (Tom, 2026-09-01).** Both halves of
the "no new names" gate already exist; one is disarmed and one is
deliberately sandbox-only. Extend them — do not add a third mechanism.
- **`Anchor` — NOT a 5.3 item, and this is the correction that matters.**
  `test_agent_name_only_in_config` is `xfail(strict=True)` and still matches
  **203 occurrences / 62 files** (re-measured post-rename 2026-09-03; it read
  216 / 63 on 2026-09-01, and class 1 went 148 → 154 across one 13-commit
  merge — it drifts like gate B did, and 5.2's 15 sites moved it down). The plan assigned class 1 — *comments + docstrings in `servers/` and
  `tests/`*, 154 of the 216 — to the comment audit. **Reading them says that
  assignment is wrong.**

  In these comments `Anchor` is not a personal name the way `Tom` is. It names
  an **architectural role** — the S0 agent holding the conversation, in
  contrast to Haiku (surface) and Sonnet (S1E): *"Showing it to
  Anchor/Haiku/encoders is false authority"*, *"what Anchor receives after
  Haiku's selection"*, *"Anchor (MCP): 1-5 nodes → want full detail"*.

  And the name is **in the data model**, not just the prose: `encoding_source
  = 'anchor'` (41 sites), `anchor_touched` traces, `anchor_message_limit`,
  `anchor_raw_quote`, `anchored_to`. CLAUDE.md states the contract — *"Only
  `anchor*` can lock a node."* Rewriting the comments to "the entity" while
  every literal two lines down still says `anchor` **splits the vocabulary and
  leaves the reader worse off than the name did**.

  So the un-xfail path does not run through 5.3. It runs through a D-12
  decision nobody has taken: **does the data model rename with the product, or
  does `anchor` stay as the role token it already is?** Until that is answered,
  sweeping the comments is a rename that stops half-way through the system.

  **THE OPEN `Anchor` INVENTORY — 203 occurrences / 62 files: 150 in 55 files
  that ship (plugin manifest or public-tree extras), 53 in 7 that never do.
  Re-measured post-rename 2026-09-03 by the 5.2 stream and independently
  reproduced by the 5.3 stream.** Tom ruled 2026-09-02 that this is *"a thorough sweep with me
  highly engaged in changes"*, one site at a time — not a mechanical pass.
  Classified by what each occurrence IS, because the classes need different
  answers:

| Class | Count | What it is | Who decides |
|---|---|---|---|
| **`encoding_prompt.py` SYSTEM_PROMPT** | **3** | **The encoder's opening line is literally `"I am Anchor, and this is me encoding my own memory."`** Every install's encoder claims that name whatever the entity is called. **Eval-gated** — this is a prompt change, not a comment edit | **the sharp one.** Needs the D-12 answer *and* an eval |
| shipped runtime comments + docstrings | 74 | `Anchor` naming the S0 role against Haiku/Sonnet | follows the D-12 answer |
| `tests/` comments + docstrings | 53 | same | follows |
| other prompt/rubric files | 20 | comments in `surface_contract`, `encoder_view`, `encode_contract` | follows |
| `seed_pack.py` origin-story tributes | 3 | deliberate CONTENT, ratified D-5 2026-08-30, allowlisted | **settled — leave** |
| ~~product-name sites~~ | ~~15~~ **0** | `boot-brain.sh` ×11, `setup.html` ×4 → **Entity**, done by 5.2 (`633329e`) | **closed** |
| unowned, never ship | 53 / 7 files | `servers/BOOT-ARCHITECTURE.md` (18), `scripts/consolidation_edge_recovery/output/plan.json` (21) + `apply.log.jsonl` (4), `servers/scales/s1/ARCHITECTURE.md` (4), `scripts/migrate_trace_identity.py` (3), `CHANGELOG.md` (2), `scripts/compute_zscore_stats.py` (1) — reaches neither the manifest nor the public tree | test-scope question only (narrow `SCOPE` or clean) — and it does NOT arm the gate: the 150 shipped hits do |

  **And the piece that is not a string at all:** `encoding_source='anchor'` is
  a **channel category**, not a name — the daemon stamps it on every
  interactive TCP/MCP write, nothing reads the configured entity name, and an
  install whose entity is called Jarvis still writes `'anchor'` (id:40d10386).
  Tom's read 2026-09-02 — *"on the encoding_source it should actually either
  use the current entity name or a generic entity label"* — matches the rename
  target already recorded: **`interactive` or `session`** (id:efafd723). Cost:
  the lock gate's `startswith('anchor')` predicate, the write default, and
  ~1,465 existing nodes. Deferred 2026-08-31 as "a spelling coincidence, not a
  coupling"; Tom's ruling reopens it as part of the same sweep.
- **`Tom` / personal data — ARMED 2026-09-01.**
  `TestPublicTreeExport::test_live_tree_exports_clean` runs the full export
  over the LIVE repo and fails on any personal-information hit, naming the
  file and line. The sandbox tests around it pin the gate's *mechanics*; this
  one pins the *tree* — a green mechanics test on a red tree proves only that
  the alarm works while the house burns. Negative-tested both ways: planting
  `Tom` + `/Users/tpac` in a tracked file fails it with the exact line;
  reverting passes. The test lives in a denylisted file, so it never runs in
  the public repo (where it would shell out to `scripts/`, outside the
  manifest). **When it fails, fix the LINE, never widen the ALLOWLIST** —
  allowlist entries are for shipped behaviour (`AgentsContext`), deliberate
  attribution (LICENSE), or a test that asserts ON the literal.
- **Why it matters, measured:** gate B went **67 → 69** on 2026-08-31 within
  hours, from another stream's thalamus work adding two comments. The gated
  classes shrink; the ungated comment class grows with every merge. Without a
  live-tree assertion the audit is a snapshot that starts rotting the moment
  it lands.

**Class B — the open ruling (Tom's call, blocks the sweep).** 47 of the 163 test
hits are gold data: `golden_dataset_v2.json` (32), `golden_canary.json` (10),
`corpus/precision_corpus.json` (5 — one carries a real birthday). Sanitizing
changes what the tests validate; excluding adds them to the D-8 graceful-skip
set. The remaining ~116 are largely the free deletions in §ACTIVE ARC.

**5.4 README + CONTRIBUTING + plugin self-description — DONE 2026-08-28
(Tom read and approved: "Looks good for now").** Landed: README
fully rewritten entity-era (first-person opening claim in the entity's voice,
neutral body; honest-expectations block — key optional-but-degraded, ~200 MB
first boot, background daemon, local-only/no-telemetry/127.0.0.1, Linux
graceful-degradation; storage truth — XDG birth, survives
update/uninstall/rename, adoption + relocation story; no benchmark claims
per D-8/eval-exclusion; no version literals that rot; no CLAUDE.md link).
CONTRIBUTING.md created (D-9 issues-only + never-paste-memories privacy
note). **`displayName` RULED: `Entity`** (D-12 — manifest names the product,
config names the instance) and shipped in both manifests; marketplace
metadata description de-Anchored. Install instructions are written for
`entity@anchor` and activate at publish. (License substance already clean:
MIT outbound; `common_words_10k.txt` removed.) The prior README's false
claims, for the record — it was a **rewrite, not a
polish** — audited 2026-08-07, re-verified 2026-08-08, and it stated things
that were no longer true: *"the plugin will refuse to load without this key set"*
(false since keyless boot, a1a620e); "seeds 16 anchor identity nodes" (**19** —
`len(SEED_NODES)`, re-confirmed 2026-08-27); "schema (v25)" (**`BRAIN_VERSION = 31`**
as of 2026-08-27 — this doc said 30, itself now stale); lists the Cowork
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

**5.5 Green suite on a clean export — RUN 2026-08-31 against a materialized
export tree. Result: `5 failed, 2979 passed, 8 skipped` in 37m35s, and every
one of the 5 failures is the same `eval/`-absence class.** This is much closer
to done than the item implied: nothing outside the eval coupling breaks. No
`tests/conversations/` fallout, no gold-data fallout, no personal-path fallout.

**The complete D-8 graceful-skip set is 8 files, in two kinds:**
- **6 import-coupled** (abort collection — `ModuleNotFoundError: No module
  named 'eval'`): `test_absorb_preservation.py`, `test_consolidation_examples.py`,
  `test_encoder_eval_probes.py`, `test_eval_corpus.py`,
  `test_longmem_classifier.py`, `test_longmem_validity.py`.
- **2 runtime-coupled** (collect fine, fail when run) — found only by running:
  `test_eval_artifacts.py` (4 tests, same `ModuleNotFoundError` raised inside
  the test body) and `test_capture_grep_pin.py::test_allowlist_entries_still_exist`
  (asserts `eval/longmem/connect_ab.py` exists on disk).
Plus the gold-data consumers, once those three files are denylisted (ruled
2026-08-31: exclude, not sanitize).

**Set RE-DERIVED — no longer an estimate:**
- **Import-coupled: exactly 6 files**, all `ModuleNotFoundError: No module
  named 'eval'` — `test_absorb_preservation.py`, `test_consolidation_examples.py`,
  `test_encoder_eval_probes.py`, `test_eval_corpus.py`, `test_longmem_classifier.py`,
  `test_longmem_validity.py`. These **abort collection**, they do not skip:
  `Interrupted: 6 errors during collection`. The D-8 graceful-skip logic is
  unwritten. **2991 tests collect** once the six are ignored.
- **`tests/conversations/` consumers are NOT import-coupled** — the 2026-08-28
  claim that "every consumer joins the graceful-skip set" is wrong at collection
  time. The 7 referencing files (`bench_precision_corpus.py`,
  `bench_precision_lifecycle.py`, `benchmark_real_conversations.py`,
  `relearning.py`, `test_clock_contract_sync.py`, `test_daemon_hooks.py`,
  `test_deploy_contract.py`) collect fine; if they fail it is at *runtime* on a
  missing fixture, which is a different fix.
- **Landmine confirmed live:** a fresh venv is runtime-only — `python -m pytest`
  on the export gives `No module named pytest`. Test deps come from
  `bin/uv pip install --python venv/bin/python -r requirements-test.txt`.
- **Do not run the suite inside the tree you are about to push** — doing so
  leaves `.pytest_cache/` in it (observed 2026-08-31). 5.7 must test a copy or
  clean before committing.

Full suite must pass on a fresh clone of the *exported* tree, not the working
tree. Plus a secrets scan before first push. Optional hardening while here: an
automated update-path test (install prior version → update → brain intact) —
today that property rests on the manual 2026-08-27 matrix. *M.*

**5.6 D-5 seed pack — DONE 2026-08-30 (the Nursery).** Full redesign shipped:
26 nodes (4 locked safety core · 6 self-knowledge · 6 growth reflexes · 3
marked exemplars · 1 seed community with loader-generated membership + the
`community_members` reconcile seed · 6 developmental scaffolds designed to be
revised away), all types/relations from registered aspect families, name-free
by construction (names live in config + boot; three origin-story tributes are
the only literals). The Zero-Memory boot block ships as the pack's spoken
half (`brain_voice`, gated in `context_boot`): fires while the brain is under
10 days old or under 100 lived memories, carries the newness disclosure, the
naming invitation (nameless installs only), and the anti-sycophancy floor,
then retires itself. Gates run: 4-lens adversarial review + re-verification,
recall probes on a seeded fresh brain, and a simulated first-session
rehearsal (9/11 instincts pass; transcript + findings id:2a9aa2c7). Contracts
in `tests/test_seed_pack.py::TestNurseryPackContracts` +
`test_context_boot_zero_memory_gate`. Remaining follow-on (not a gate): the
new-vs-old encoder eval on the frozen-corpus harness. *Was L; shipped.*

**5.7 [LAST — one-way door] Publish.** Fresh public repo `tpac/entity`, clean
history, only after 5.1–5.6 are verifiably clean.

**The `release` command — scoped 2026-08-31, NOT built (needs an explicit go).**
One command, and its whole value is that it *refuses*:
  1. assert the working tree is clean and on `main`
  2. `export-public-tree.sh` → materialize; **abort on any red gate** (A, B, C)
  3. run the export with `EXPECT_VERSION=<the version being released>` — gate C
     then fails on agreement-on-the-wrong-value, not just drift (built 2026-09-03
     with 5.2). The literal itself lives in ONE place,
     `TestVersionLockstep.EXPECTED_VERSION`; a bump edits it with the manifests,
     and the release command derives `EXPECT_VERSION` from its own argument
     rather than carrying a second literal
  4. run the suite **against a copy** of the tree, not the tree itself
     (`.pytest_cache/` pollution — see 5.5); abort on red
  5. secrets scan
  6. `git init` a fresh checkout, one commit, tag `vX.Y.Z`, push
Small now that tree-building and gating exist. It should absorb what the
abandoned release-ratchet rebuild (BACKLOG item 4) was for. **The exported tree
already carries no `.git` (verified)** — history cannot leak through the export
itself; only a mis-aimed push could.

**Irreversible: git history is forever, and this repo's history carries the
personal data.** Re-counted 2026-08-31 — **89 tracked files across 2138
commits**, not the 65 the doc carried: `docs/archive/` **70** ·
`tests/conversations/` **8** · `tests/archive/` **9** · `tests/results/` **2**.
All four are gitignored *and tracked*, so all four are in history right now.
This is the whole argument for D-7 (fresh repo, squashed) over a fork, and the
count grows on its own — Tom's session-closure convention (id:9e2c4dd3) copies
conversation JSONL into the repo, which is how `tests/conversations/` got there.
The export gates are what stand between that history and the public tree; they
were verified red-on-dirty and are pinned by `TestPublicTreeExport`. *L.*

**5.8 [AFTER A SOAK] Directory submission. Researched 2026-08-31 — the license
worry was the wrong worry, and "the official directory" is not submittable.**
- **No published policy requires an OSI license.** Neither
  `anthropics/claude-plugins-community` nor the Claude Code plugin-marketplace
  docs state any license requirement. The docs treat `license` as an **optional
  informational SPDX string** in the plugin entry. So the dual PolyForm grant
  is not disqualified by any *written* rule — the exposure is discretionary
  review, not a stated gate. Third-party guides recommending "MIT or Apache"
  are advice, not policy.
- **Two directories, only one has a door.** `claude-plugins-official` is curated
  at Anthropic's discretion with **no application process** — 5.8 as originally
  written is not an action anyone can take. The submittable one is the
  **community** directory, via the in-app form
  (`clau.de/plugin-directory-submission`, or `platform.claude.com/plugins/submit`
  for individual authors outside a Team/Enterprise org). Submissions pass
  `claude plugin validate` plus automated security screening; PRs opened
  against the mirror repo are auto-closed.
- **Cheap prerequisite:** run `claude plugin validate ./` locally before
  submitting — the review pipeline runs the same check.
- **Do first regardless:** add the `license` field to `marketplace.json` (D-1
  conformance defect) so the grant is legible wherever the entry is rendered.
Only once the self-hosted marketplace has real installs. Mechanics in §10.1. *S.*

**5.9 The public tree is OPT-OUT. Tom ruled 2026-09-01 that it must be OPT-IN.**
*"I want things that are released to the public repo to be opt in not opt out so
we don't accidentally launch personal stuff."* Measured 2026-09-01: **413 of the
429 public files arrive without anything having said yes to them.**

| Path | How files are chosen | |
|---|---|---|
| `servers/ hooks/ skills/ dashboard/ bin/` | `git ls-files <dir>` minus a few `grep -v` filters | **opt-out** |
| `tests/` | `git ls-files tests` (224) minus a 15-entry denylist → 189 land | **opt-out** |
| root | hand-listed (`README.md`, `CONTRIBUTING.md`) | opt-in |

Committing a tracked file into `servers/` or `tests/` ships it on the next
release with nothing asking. The only thing between that and a leak is gate B,
which knows five patterns. **This is the mechanism behind the 67→69 drift** —
not a fluke, the design working as built.

Note the tension `build-plugin.sh` already records: `git ls-files` was chosen
*deliberately* so an untracked scratch DB or secret can never ship. That
argument holds for untracked files and says nothing about tracked ones. The
opt-in question is only about the tracked set.

Scoping notes for whoever takes this:
- A literal 429-entry manifest rots — but it fails SAFE (a forgotten entry
  means a missing file, which breaks loudly at install, not a silent leak).
- The cheaper shape is per-directory opt-in: each shipped directory names an
  explicit include-list or an include-pattern, so adding a file is a
  one-line manifest edit rather than an automatic consequence of `git add`.
- `tests/` is the highest-risk directory (189 files by subtraction, and the
  denylist there is 15 entries of remembered exceptions).
- Whatever lands needs its own test: a new tracked file under a shipped
  directory must NOT appear in the export until something names it.
Not a publish blocker on its own — gate B still catches the known patterns —
but it is the reason the ratchets below exist, and it should land before the
repo takes outside contributors. *M.*

---

## 5. Sequencing & dependencies

```
0.1 baseline ──► 1.1 / 1.2 (prompt edits gated by baseline)
                 1.3 / 1.4 / 1.5 (independent, anytime — 1.5 is a one-line build fix)
2.x onboarding  (independent of Phase 1)
3.1 ──► 3.2     (3.1 first; 3.3 deferred)
4.1             (independent; needs the 127.0.0.1 guard)
DONE: 5.0 · 5.0a · 5.0b · 5.1 · 5.4 · 5.6 · 9.7.2

(1) free deletions ──┐   (RELOCATE the 2 eval-only contracts into eval/,
                     │    gate self-exemption, legacy-rung allowlist,
                     │    dev-harness personal paths)
(2) gold-data        ┼──► (3) COMBINED LITERAL SWEEP ──► (4) 5.5 + D-8 skips
    exclusion        │       = 5.0c cls 1–2 ∪ 5.3 cls A      8 files; suite
    (3 files → skip) │       one eval gate, not two          already 2979-green
                     │                                            │
(5) FULL 5.3 comment audit (614 lines / 111 files) ◄───────────────┘
              │
              ▼
(6) 5.2 rename ──► (7) 5.7 release cmd ──► publish ──► 5.8 community directory
    + D-10 0.9.0 bump      [one-way door]
    + the 0.9.0 assertion that doesn't exist
    + un-xfail gate assertion 4

(8) the fleet — LAST, off the critical path, likely a one-time special install
```

**Why 5.2 sits late.** It is a three-file edit. Its only true precondition is
the fleet check, and 5.0b's `com.N.` assertion is *dormant until the rename* —
so a 5.5 run before 5.2 is a 5.5 you run twice. Both prior orders
(this doc's, and id:44025fbb's) put it earlier; both are superseded.

**The soak is dissolved, not shortened — and the fleet moved OFF the critical
path (Tom, 2026-08-31).** The soak was framed as a clock: 9.7.2 shipped
2026-08-28, wait for convergence. It cannot work as a clock — there is **no
telemetry** (the README promises none), `/plugin update` is user-initiated
(id:b792b20e: Moshe sat frozen on 9.6.0 for two months while `/plugin update`
told him he was current), and `relocate-brain.sh` is **user-run only** — boot
renders a notice, a human must act on it. Elapsed time carries no information.

Tom's ruling: *"Let's defer for last, i dont know their situation yet and
perhaps we'll just tailor a 1 time special install."* So 5.2 no longer waits on
anything. **Why that is safe:** the rename cannot *reach* those installs at all
(id:8e1495cb — no update path crosses `plugin@marketplace`), so a friend's
`brain@brain` install simply keeps working, frozen, until they act. The residual
hazard is only their own initiative: `claude plugin uninstall brain` is the
command proven to delete the data folder (id:8a057057). A hand-tailored install
is the right mitigation *because* it replaces "hope they relocate" with "we do
it for them."

**Two things to settle when the fleet comes up, not before:**
- **Where do those installs source their marketplace from?** If from
  `tpac/brain`, 5.2 renames that manifest's entry underneath them — behavior of
  `/plugin update` against a marketplace whose entry vanished is unverified. It
  should not delete data (only `uninstall` does), but it is an unknown, and it
  is cheap to check before 5.2 rather than after.
- **Population:** Tom's main machine is on the legacy `~/AgentsContext` rung —
  *not* parked under a plugins-data root, so unaffected either way. That leaves
  Tom's second laptop and Moshe (installed 2026-07-30).

The publish (5.7) must not happen before the seed-pack session (D-5) — closed
2026-08-30: a stranger's first brain should wake up *well*, not with
neutralized-placeholder fixtures.

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
| ~~**Plugin rename orphans an existing brain → silent empty brain**~~ **CLOSED 2026-08-12 by 5.0a** | 5.0a / 5.2 | The silent-create path is gone: creation moved to the XDG service dir and the adoption net refuses to create while an orphaned brain sits under a plugin-data root — guided adoption via the `~/.config/brain/env` knob / `brain_path`; never auto-move. Residual **CLOSED 2026-08-28**: a brain resolved under any plugins-data root now trips `BRAIN_HOST_PARKED` and boot renders the relocation notice every session until the user moves it (uninstall deleting that tree was verified 2026-08-27; `--keep-data` named as the stopgap) |
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
