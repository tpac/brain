# S1E Prompt Checklist — the boxes every revision must check

## Walk state — SHIPPED 2026-08-25: v-next.6 IS the production default ◀ ACTIVE ARC

**v-next.7 candidate open (2026-09-01), UNPROMOTED.**
`eval/candidate_prompts/s1e_vnext7_wip.md`, 113,365 chars (+2,913 over the default) — field-coverage
redistribution across the existing revise examples, the narrow-to-residue
closure branch (`partially_resolves`, previously prose-only), plus the new
**A10** law and the Field-coverage matrix that would have caught the defect it
fixes. Not registered, not deployed as an override; the production default is
byte-unchanged at 110,452 / `3817564d21c4`. Needs Tom's ruling **and** a
field-coverage eval instrument that does not exist yet — run-44 scores sibling
reach, not which surfaces a revise touches. Diagnosis: §6a of
docs/STALENESS-PROPAGATION-FINDINGS.md.

⚠ **PROMOTE BY APPLYING THE DIFF, NEVER BY COPYING THE FILE.** The candidate is
a full-file snapshot of v6, so overwriting `SYSTEM_PROMPT` with it silently
reverts anything that landed in `encoding_prompt.py` in the meantime — and git
sees no conflict, because the candidate is a *different file*. Promotion =
apply v-next.7's 14 hunks to whatever `encoding_prompt.py` says at that moment.
Flagged by stream `831149ce` (2026-09-01), who has two D-12 agent-name edits
queued for that file: as of this writing `SYSTEM_PROMPT` still equals the v6
base exactly, so nothing has been lost — but a file-copy promotion after their
merge would reinstate 3 capital-`Anchor` literals with **zero signal**, since
`test_deploy_contract.test_agent_name_only_in_config` is `xfail(strict=True)`
while 5.0c is open. Diff-application makes the merge order irrelevant.

**The arc closed.** `SYSTEM_PROMPT` in `scales/s1/encoding_prompt.py` is v6 (110,452 chars,
was 86,833); merged as main `c9205a7`, daemon restarted, verified from the daemon itself —
`get_interaction_effective('s1e')` → fingerprint `3817564d21c4`, `source=default`,
pointer-less, `check-overrides` still at the permanent 2. Registered copy: **v41**.
Ship record with full numbers: brain id:4f201dd1.

**What shipped on top of the measured v40:** Tom's read of the prompt text produced four
wording fixes — "3+ distinct turns" + trace ids (was "turn anchors", one step from the
turn-coordinate leak a82d7b7d); the `locked` line stating code truth instead of colliding
with the persona name; `## Speed` → `## Cadence and worked examples`; six `§7.6.Ax`
headings renamed to what each teaches. Five of his six catches were production text we had
been shipping for versions.

**v-next.7 opens here, all measured, none speculative:**
1. **Tier-1 claims-check banner** (assembly layer) — v6 moved stale-sibling *reach* (3/3 vs
   1/3); fact-carriage is the proven residual, and the 2×2 says banner+template is what
   wins (7/15; either alone ≈ baseline).
2. **The atoms bar** — "keeps referencing" is stricter than the ruled retrieval-divergence
   test (id:9f072492); the Rachel item (830ce83f, third-party fact in passing, skipped by
   all three arms) is its ready-made probe.
3. **The agent name** — hardcoded 7× incl. an identity *example* node, while
   `BRAIN_AGENT_NAME` binding exists and every other layer respects it. Needs a token plus
   substitution; where the substitution belongs (shared resolver vs s1e call site) is the
   design question.
4. **source_refs under-production** (~50% of nodes, 3 items at zero; phrasing ruled out).
5. **The twin write-boundary warn** — a new node title-matching an unexpanded catalog title
   with no correction edge; minted 3/3 in every arm ever tested.

## Prior head — three-arm night (2026-08-25), superseded by the ship above

**Overnight three-arm result (lived substrate, repaired instruments, validity-partitioned;
full detail: brain id:7d301e3d, report artifact below):** gate-10 — prod 80.0 / v5 88.3 /
**v6 90.0, perfect 1.00 on all 9 scorable items** (only fails = the user-turns-only
abstention gold artifact); batch B — 57.1 / 71.4 / 73.8 (v5-v6 tie at valid-only);
batch C (2-arm) — 71.4 / 81.0; run-44 — all arms fail gold, but v6 moves sibling REACH
3/3 vs 1/3 (fact-carriage is the residual; twin stands). v6 = v5 + four surgical example
revisions (registered v40 DORMANT, eval/candidate_prompts/s1e_vnext6_wip.md).
**Recommendation: promote v6.** Ship act unchanged: template → SYSTEM_PROMPT in
scales/s1/encoding_prompt.py, merge, install stays pointer-less. NOT promoted — Tom's pass.
**v-next.7 spine (all measured):** Tier-1 claims-check banner (assembly layer — supplies
the fact-carriage half v6 proved missing); source_refs under-production (~50%, deeper than
phrasing); third-party-in-passing skip class (830ce83f, all arms); twin write-boundary warn.

## Prior head — GATE RAN 2026-08-24 (superseded by the three-arm night above)

**Where this stands.** v39 IS REGISTERED (dormant, 109,878 chars, header stripped) and the
gate eval ran end-to-end: baseline (pointer-less code default, corpus fcc338, --force) vs
v39 (corpus a3be7a), 10 gate-v29 qids, sweep --variance 3. **v39: 60→83.3% overall,
recall-conditional 64.3→86.2%, do-no-harm PASS (wins or ties all 10 items), 2→0
whole-conversation zero-encodes, event_time +37pp, catalog-targeting 22→40%, all three
journal guardrails PASS 9/9** (baseline's journal is entirely dead — 0/9; v39 arcs and
notes every item, residue-only, mean 857/942 chars). Full report:
https://claude.ai/code/artifact/54dcd1af-3d8f-4851-8869-d49c80ee3c45
Gate-run harness fixes landed on main (merge 1433756): gold-scan seed/digit contamination,
baseline "active" cache addressing (k_fingerprints now in the corpus content address +
manifest), Brain.get_source_refs, corpus_shape.py (the L1 encode-shape scorer). A false
over-production flag was retracted mid-run (id:2f149c13 — measurement artifact).

**REMAINING:**
1. **Tom's pass on promotion** (recommendation: promote v39 as-is).
2. On pass: promote the template into `SYSTEM_PROMPT` in `scales/s1/encoding_prompt.py`,
   merge (daemon runs main), install stays pointer-less (v39 was never activated).
3. v-next.6 seeds: why-length overshoot (mean 193 vs 120–180 band), emotion pair never
   fires (0/90). Harness follow-ups: abstract-gold scan blindness, answerer
   world-knowledge fabrication on abstention items (benchmark semantics — Tom rules).

**Order of work, and why:**

DONE 2026-08-23 — all Tom-ruled, all probe-verified (arms Q positive, R negative
control, S single-variable re-test; every arm production-faithful, incl. `## Arc`):
- **ad7e68e** — name-pass (44 sites → `their_raw_quote`/`my_raw_quote`, gloss shrunk);
  `reasoning` boundary in the three leaking canonical fields (this was also D7's real
  fix); the **open-node closure teaching** folded into corrections flavor 3 — three
  branches (retype+`resolves` / stay-open+narrow+`partially_resolves` / close+mint);
  canonical excerpt +2 catalog lines with `completes` and `resolves` edges; the sixth
  action-derived node; D1 (arrow moved onto the mechanism node — **no** global `grounds`
  direction asserted, live usage runs both ways), D2 (three sites, not two), D3, D8.
  **D5 and D6 dropped** — both premises failed measurement.
- **2a5acbd** — `reasoning` re-aimed in contract.py from "why this was encoded" to
  evidence strength. Reaches live v38 encoding on daemon restart, not gated behind v39.
  Verified by probe S: pipeline narration and rule-citation gone, and the meta-observation
  relocated itself to `thought` + the Review fence unprompted.
- The killed pilot: **the live-question teaching does NOT ship** (id:99b81b25 — the
  encoder authors 169 of 225 `open` nodes, so its premise inverted). Its replacement is
  the closure teaching above (id:99f5d84b).

DONE 2026-08-23 (continued):
- **The weave plan** (E2/E3 section) — all 5 moves landed (aedb73d, 6b7545e); the move
  table below carries per-move detail.
- **MCP membership change** (b59c694) — `question`/`event_time` writable (named
  remember() params), `critical`/`personal`/`personal_context` retired via
  `agent_writable=False` (read paths stay), `encoding_source` system-stamped +
  force-stamped at apply_encoder_attribution (setdefault let an LLM-authored value
  win), stale `content` line fixed, every remaining bare field described, brain_mcp's
  local description fallbacks removed. Tom waived the eval gate (stale surface;
  168 unit tests green). Residual, named: brain_batch per-op `encoding_source`
  still wins over top-level inside the handlers (serves legitimate multi-scale
  emitters; nothing advertises the key to any LLM). Held back as before: `locked`
  (needs the audience split — one `get_writable_fields()` serves both MCP callers
  who *can* lock and an encoder whose locks are silently demoted), `source_refs`
  (wants its 8-hex shape gate first), `thought` (1 use — let the teaching earn it),
  the `emotions` migration fork.

DONE 2026-08-23 — **E1 + E8 + E10 ran over the settled draft.** Method: E8 as a
mechanical census (author-run); E1 and E10 by BLIND Opus readers (draft + the live
post-membership tool surface, no priming) — the author of the text never audited it.
E1 returned 15 dead-conflicts / 6 optionality; E10 returned 26 findings incl. two
double-hits with E8/E1 (temporal why-length, the trace-marker form). The mechanical
fix set landed under already-ruled law: 043ea84 (draft — examples now obey their own
rules; 105,074 chars) + 1da0129 (tool-side: connect_batch's related_to default
killed, revise-side source_refs REPLACE text, marker form covers the lived arm,
field summary carries source_refs). E8's fresh-reader probe (real/example/slot)
verifies the grammar unification.

DONE 2026-08-23 — **Bucket-B rulings (Tom) landed.** (1) `emotions` fork: minimal
move only — the three example sites now write the declared `emotion`+`emotion_label`
pair (recall reads it); the array/migration design is DEFERRED, owner: a future
emotion session ("we'll deal with emotion later, as long as it starts writing").
(2) `thought` PROMOTED into the contract + named remember() param; delivery verified
real on both surfaces (generic KV render). (3) `my_raw_quote` scope per Tom: moments
important to ME — about myself or about the information — scope line added to the
voice rule; the two editorializing quotes rewritten as said-shaped realizations.
(4) Cadence unified: read-when-needed → encode → close (~2-3), with the anti-loop
guard (one read round at most). (5) `surfaced` softened to context-not-obligation.
(6) Residue route narrowed to the under-3-anchors case. (7) `locked` trimmed to
anchor-only-not-mine. (8) `confidence` DELEGATED(MCP — the field description
teaches selectivity at decision time); `evolution_status` deliberate encoder
silence (closure teaching's type-mutation is the ruled mechanism);
absorb/archive/disconnect + a `rule`-type example → v-next.6; flavor 3 now points
at the Validity-intervals discriminator. (9) The write boundary refuses a missing
edge `relation` loudly (was: silent `related_to`) at _handle_connect (covers
brain_batch's connect op) + the connect_batch loop.

DONE 2026-08-23 — **The completion pass: ALL challenges ran over the settled draft.**
E3 matrix walked (all boxes → carriers; empty cells named with owners); all 30 R-rows
re-stamped blind (8 ABSENT → 0; row 13's question-scarcity was the one teach-against-
evidence and is fixed — 3-of-6 selectivity, better-question-not-no-question); E2 example
sweep ran blind (36 findings; 4 assets stamped fully clean; A5 sound; voice ratio 59%
my-voice vs the 6% production baseline). The fix batch landed under Tom's
go-with-your-recs ruling: canonical now revises what its own edges falsify (the
closure teaching demonstrated at the highest-attention slot); temporal example fenced,
reasoning on all 6 nodes, deictics out; §7.6 reasonings rewritten to evidence
strength; ladder grounded by its own excerpt; future-target event_time doctrine
reconciled; sensitivity floor + richness ceiling added; sweep refs justified as the
correction-scene class. Ship-package code landed: source_refs 8-hex gate at the DAL
(node writes unaffected — refs refused loudly via source_refs_persist*), locked-strip
demotion now logs, connect_to description guardrail test in test_contract_sync.
Considered-and-left: §7.6.A8's placeholder (retyping the title would model the
anti-pattern). E7 redundancy re-audit running.

DONE 2026-08-23/24 — **both held doubts resolved** (bbadf24): (a) the rule-type premise
was false in code — rule+decision share the pre-action safety surface and never decay,
'open' has no system machinery (the type paragraph now states the truth; retype
dissolved; node id:6537771e); (b) Tom ruled their_raw_quote the SAYER's lane, speaker
weighting later (id:bb94905b). E7 re-audit banked at
eval/candidate_prompts/audits/e7_redundancy_vnext5.md — 28 scars DEFERRED post-gate
(E5 + eval attribution; id:4f9a7e20). Draft: 111,498 chars. Completion-pass milestone:
id:b5e358b7. Handoff: id:bb0740dd.

DONE 2026-08-24 — **the gate ran** (see the ACTIVE ARC head at the top for results and
what remains). The KPI framework became the four-level L0–L3 ladder (arm integrity /
encode-shape via corpus_shape.py / brain-presence / recall-differentiated), v39 registered
dormant, both arms built fresh and swept at variance 3.

**Deferred by design, with owners named:** the vertical axis beyond the pilot (six more
teachings shaped like Emerging patterns — evidence in id:56631bce); the journal→graph
route (id:7b2d67e8, constrained by ruling 68063517: residue into the Frame is *hurtful*);
retire/archive (id:9eaed612).

**Probe fidelity note:** probes A-O were built WITHOUT the `## Arc` block (production S1E binds
`arc=True` — encode.py `_journal`). Probe P is the first production-faithful arm; rebuild future
arms with field summary → arc → review → closure, in that order. **The recipe and the reusable
scenarios now live in `eval/candidate_prompts/probe_scenarios/`** (README carries the arm build,
the tool-ban clause, and the settled/unsettled pair — they only mean something together, since
the positive arm alone can't separate a working teaching from one that fires on everything).

**Read first:** handoff node id:477ddcc9; the rulings chain runs 7c87589c (fundamentals) →
13f72658 (three connections) → 4bb5b1e8 (edge honesty + prune-lexicon example).

The walk: section-by-section co-review with Tom (he rules, I hold the learnings), every
substantive edit probe-verified by stateless Sonnets before it lands. All stops closed
(probes A-P, then Q/R/S on the fix set); draft = eval/candidate_prompts/s1e_vnext5_wip.md,
100.3K vs v37's 86.9K — over v37 since Stop 8's sweep example (Tom-ruled earned size;
package eval is the backstop), ~30 measured teachings added, NOT registered.

**Locked:** target function; banner v1 never ships (position carries, text was
staleness-biased); [associated]-in-catalog (not a separate section); edge honesty over
floors; Allen sequencing verbs KEPT (measured top rescue paths — only the ceremony cut).
**Open:** E1/E2/E8/E10 global audits (E1's two-reads entry resolved at Stop 8; E10's first
entries are the MCP findings in REMAINING item 2); ship gate (override eval
→ package eval → Tom approves → the candidate lands as the code default); ship-package code items: the
Assembly matrix row, id:477ddcc9, plus the source_refs 8-hex shape gate at
add/replace_source_refs (dal.py:714 — type-checked only today; placeholders store
silently; loud write boundary), and the locked-strip log (brain_remember.py:1152
silently demotes non-anchor locked:true — the demotion is right, its silence isn't).
Lock doctrine (Tom, Stop 10): locking is a rare act — the brain's 352 locked nodes
need their own review session someday (set_node_lock is the door, d80fc0a).
**Ship-gate precondition — MET 2026-08-22 (ruling id:52c8eb6d).** The voice-field rename
landed on main via the rename stream (their work, not this walk's): BRAIN_VERSION 31 renames
`user_raw_quote`→`their_raw_quote` and `anchor_raw_quote`→`my_raw_quote` in BOTH
`node_metadata_kv.key` and `node_enrichments.vector_type`; no re-embed needed. That stream
also registers **s1e v38 = a mechanical name-swap of deployed v37** (two strings only) and
activates it, so live encodes keep writing voice fields across the schema move.
**Consequences for this walk:** (1) the WIP's name-pass is DONE (ad7e68e — 44 sites, gloss
shrunk 12→10 lines since the possessive now rides in the field name); (2) our
candidate registers as **v39+, and its diff base is v38, not v37** — any diff review against
v37 will show phantom voice-field churn; (3) `generate_field_summary()` inherits the new
names automatically (it derives from the contract), and that stream already rewrote the two
voice descriptions — so the field-summary ship-package item below starts from the new names
and must NOT re-touch those two lines; (4) contract.py is ours to edit now — they are out.

**Ship-gate RUN notes** (reported by the eval stream e2f19c82 and the rename stream
02e4b5a9, 2026-08-22 — their findings, not independently verified here):
- `--interaction-override` keeps its name and semantics on `build_corpus.py` and `leg_b.py`.
- The corpus content address moves from `{name: int(version)}` to version + template sha1
  (two arms on different code-default generations used to collide on one hash, and
  `load_manifest` could hand back the wrong arm's corpus). **Consequence: our first
  override build after their merge is a CACHE MISS and re-encodes once — budget for it,
  it is not a fault.** Baseline (no-override) corpora keep their hashes.
- **Pre-registered failure hypothesis for our gate run:** the frozen corpora still carry the
  OLD voice-field kv keys and depend on `sweep.py`'s copy-then-open migrating each work copy
  to v31. Migrate-on-open was verified but never exercised by a real sweep — ours may be its
  first. If the run reports voice fields **missing rather than renamed**, suspect that path,
  not the rename.

**THE SHIPPING PATH CHANGED UNDER US — 2026-08-23, mid-session.** The override-migration
stream's Step 8 (the install collapse) landed on main and the daemon ran it, so **`s1e` now
has NO active pointer**: `list_interactions` showed `active_version: 38` (set_by `anchor`)
at this session's start and `null` by its end, and `get_interaction('s1e')` now answers
*"no override deployed — the runtime reads its code default (`servers/interaction_defaults.py`)"*.
**This is that arc's INTENDED end state, not a breakage** — pointer-less is normal now
(id:bd39c56c). Consequences, and they replace this runbook's baseline and final step:

- **The live prompt is the code default** — 86,833 chars, already carrying the v31 voice
  names, i.e. v38's text. So the **baseline arm runs with NO override**, not `s1e=38`. An
  `s1e=38` baseline would be override-vs-override and would not measure against what
  production actually runs.
- **`v39` is still the right number** — max_version is 38, and registered versions stay on
  record through the collapse.
- **Activating v39 is NOT the ship act any more.** Tom's ruling in that arc
  (`docs/PROMPT-CONFIG-OVERRIDE-ARCH-PLAN.md`, decision 3): *"the eval gate becomes a
  process rule: experimental changes land as overrides, get promoted into the code default
  after the eval passes."* A permanent override would also freeze `s1e` against every
  future code default — precisely the condition the collapse existed to clear.
- **Ship sequence:** register v39 → deploy as a *temporary* override for the eval arm →
  package eval → on Tom's pass, **promote the template into the code default**
  (edit `SYSTEM_PROMPT` in `scales/s1/encoding_prompt.py` — that file IS the default
  `interaction_defaults.py` serves; no sync, no version bump since Step 9 deleted both)
  → then **clear the override** so the install returns to pointer-less.
  `clear_interaction_override` (their Step 6) is that verb.
- Unchanged and still correct below: `--interaction-override` mechanics, `--pooled`
  refusing to compose, `check-overrides` as the leak check, multi-rep over single runs,
  and BOTH arm-integrity checks.

**THE GATE RUNBOOK** (eval mechanics confirmed by the eval stream after their Step 7
reshape — re-anchor here, not to any older command; read the shipping-path block above
first, it changes the baseline arm and the final step):

```bash
./dev python3 eval/longmem/build_corpus.py --interaction-override "s1e=39" --label gate-vnext5 [--items N | --qids a,b,c]
./dev python3 eval/longmem/sweep.py --corpus <hash> --label gate-vnext5
```

- `--interaction-override` keeps its `name=version[,name=version]` format and semantics:
  it fetches that DORMANT version's template from the live daemon and applies it to each
  fresh per-item eval brain only. `39` assumes our candidate registers behind v38 (the
  mechanical name-swap) — confirm the real number at register time.
- `--pooled` still refuses to compose with an override.
- `./dev check-overrides` reports whether a previous run left a pointer behind — run it
  before the gate, since a stray pointer silently contaminates the arm.
- Multi-rep, never a single run (E4): same-capture variance exceeds arm deltas.

**Two arm-integrity checks — do both, they catch different failures.**

*Free, on the build:* `override_interaction` fingerprints the effective K before and after
applying. If the override changed nothing it prints to stderr —
`[override] WARN s1e v39 is byte-identical to what was already effective … an A/B across
this override compares a K against itself`. **Watch the build's stderr; don't let it scroll.**

*Manual, and the one no single build can make* — each arm only ever sees itself, so the
cross-ARM comparison is ours to run. Per arm, after its build:

```bash
./dev python3 -c "
import sys; sys.path.insert(0,'.')
from servers.brain import Brain
b = Brain(db_path='<corpus_item_dir>/brain.db')
print(b.get_interaction_stamp('s1e'))"
```

Then assert **`fingerprint_A != fingerprint_B`**. The stamp is 12 hex over the RESOLVED
(overlaid) template *and* config, so one `!=` is the complete check — two arms with
different version ints can still resolve to the same K when the difference lives in the
config half. Our gate should differ by construction (the pointer-less code default vs v39
are genuinely different text); **if the fingerprints match, stop** — the override didn't
take, and the config half
is now the likely cause.

⚠ That snippet spawns `Brain(db_path=…)` deliberately against a **corpus item's isolated
copy**. Never repoint it at the live `brain.db` while the daemon runs (CLAUDE.md).

**Do not cross the two values:** `corpus.interaction_token(version, template)` →
`v39:<sha1[:8]>` is the CACHE-side address and deliberately excludes config (matching how
the non-override path addresses `s1e="active"`); the 12-hex `get_interaction_stamp` is the
RUN-side integrity check and includes config. Different functions, different purposes.

**Collision audit (theirs, reported 2026-08-22).** 11 frozen manifests, 4 carrying overrides,
all 4 version maps distinct — no measurement we cite shows a visible collided hash. Their own
caveat is the load-bearing half and is worth preserving: the failure is **silent by
construction** (a collision CACHE HITs and writes no second manifest), so "no duplicates on
disk" is equally consistent with "never happened" and "happened and silently reused build #1."
Unfalsifiable after the fact. Exposure is narrower than it first looks — a template override
replaces the template wholesale, so the risk sits in the CONFIG half (model/effort/max_tokens
inherited from the eval brain's default) differing between two same-map builds.
**Do not reopen:** ledger method/home; §-numbering; degree floors; <memories_beyond_catalog>
as a section; the Allen full-cut.


> The working instrument for the s1e prompt co-review (2026-08-21). Distilled
> from the challenge ledger (docs/challenges/, 1,813 rows), the 37-version
> genealogy, and Tom's fundamentals. Every proposed edit names the boxes it
> closes AND the boxes it risks re-opening. The final draft passes when every
> box has a named **carrier**: a prompt line, an example dimension, an MCP
> description, code, or a deliberate `open-by-design`.

## How we use it

1. **Walk the prompt section-by-section.** Per section: which boxes apply,
   current status — `HOLDS` / `VIOLATES` / `ABSENT` / `DELEGATED(where)` /
   `OPEN-BY-DESIGN` — and the shape we want.
2. **Edits accumulate into one draft; nothing registers piecemeal.** After the
   last section, run the global audits (E-boxes): full example sweep,
   contradiction audit, coverage matrix.
3. **Coverage matrix fills as we walk.** An empty cell at the end is a gap —
   deliberate or caught.
4. **Fresh-eyes pass before any edit lands (anti-accretion, Tom 2026-08-21).**
   Per section: state its purpose in one line, draft what a blank page would
   say, THEN diff against the current text. Additions are the accretion bias
   showing — prefer merges, rewrites, and deletions that leave the section
   the size its purpose earns. Counterweight (E5): "I'd phrase it
   differently" is not a reason to churn text the probes show working —
   rewrite what's wrong or duplicated, keep what's measured.
5. **Reverse pass after each stop (Tom's principle, 2026-08-21).** Design to
   the spec, then ask from the other direction: "what is the thing I actually
   did?" — derive the law each edit instantiates. A law not on this list is a
   new box (add it); a law that fights a box challenges the spec (surface it).
   Stop-1 yield: A9, B7, D14, E7 — four laws the 1,813-row ledger never
   stated.
6. **Checkboxes per example (T1 operationalized).** Every example asset gets
   its own coverage row in the Example inventory below: the dimensions it
   currently teaches, its spare capacity (dimensions it could carry), and its
   risks (every dimension leaks, A4). Most box-fixes should land as example
   weaves, not new rules or sections (the v33 lesson) — the inventory is how
   we know which example has room.
7. **Ship gate:** A/B the candidate as an override (`tests/interaction_override.py`)
   → eval (run-44 staleness set + longmem sweep, multiple reps — no single-run
   conclusions) → Tom approves → the candidate replaces `SYSTEM_PROMPT` in
   `encoding_prompt.py`.

## T. Tom's fundamentals (the frame — 2026-08-21)

- **T1. Examples are golden, always with true AND false.** A single example
  holds several shapes at once, for better or worse. Every word matters.
- **T2. MCP is a teaching channel, often stronger than the prompt.** Tools
  give rich information to the agent even undescribed; descriptions and
  schemas can influence more than prompt text.
- **T3. Location of instructions matters a lot.**
- **T4. Contradictions can hurt or give optionality.** Audit each one and
  mark which it is — an unmarked contradiction resolves by position, not
  by intent.
- **T5. Keep things open. We cannot engineer full brain encoding — we guide.**
- **T6. Encoding succeeds in two forms, and both get checked:** (1) did we
  encode the information we expected to? (2) did we make it easy to recall in
  a DIFFERENTIATED way? A node so rich it contains "everything", beside a
  neighbor holding much of the same, defeats recall's ability to choose
  between them. Richness vs separability is a balance game — guide it.
- **T7. One sign system, chosen for the LLM — human-friendliness is not a
  priority at all.** Free prose = my guidance. `<angle-tags>` = ONLY structures
  that literally appear in the runtime payload. Examples = one distinct,
  consistent marker. Placeholders = one grammar. What's best for the LLM is
  the right shape; a mixed sign system is noise that costs comprehension
  every run. (Tom, 2026-08-21)

## A. Example laws (under T1)

- **A1. Every behavioral ask gets a worked example; instruction-only = dead.**
  `thought` 0/120 nodes, `emotional_context` 0 brain-wide, event_time ~0% for
  three prompt generations until the example rebuild. (encode-write #136,
  encode-absent #13, #38)
- **A2. True/false contrast is load-bearing.** Removing the "Wrong action
  (the trap)" paragraph silently killed correction-node emission entirely;
  restraint needs the wrong path SHOWN, not described. (genealogy #223)
- **A3. Examples override rules — audit every example after any rule change.**
  v15.7: 3 of 5 high-severity findings were stale examples overriding newer
  rules; the prompt's own maintenance kept half-revising itself (v20/v21/v22).
  (genealogy #15, #10)
- **A4. Example shape leaks on every dimension, not just the named lesson.**
  Title-length norms transfer (~110c examples → 110c output); thin fields in
  examples become missing fields in output; v20 found 4 real bugs in canonical
  examples that taught anti-patterns for 7+ days. (genealogy #161, #124, v20 row)
- **A5. Fictional ids/targets leak into production writes.** Copied example
  titles → `connect_to_unresolved`; literal `<trace-...>` strings. Only two
  safe forms: grounded excerpt with copyable ids, or disclaimed placeholder.
  (genealogy #74, v21/v22/v31 rows)
- **A6. Concrete example nouns anchor and shadow the live conversation.**
  {placeholder} templates for shape-teaching; concrete nouns for
  behavior-teaching; domain-spanning so engineering examples don't narrow a
  fresh brain's vocabulary. (genealogy #89, #172)
- **A7. Never build examples FROM the eval corpus** — contaminates the retest
  signal. (genealogy #221)
- **A8. Quality spread calibrates:** all-pristine examples produce
  overconfidence; strong/typical/weak spread routes confidence. No CoT blocks
  in examples for small models — they train format imitation. (genealogy #61, #44)
- **A9. The whole prompt is an example — its own conduct teaches.** The
  document must practice the epistemics it teaches: the opener's "surfaces at
  95% where a three-topic node musters 70%" was an invented statistic (traced
  to a v1 illustration, no eval behind it) — modeling exactly the overclaim
  habit the ledger convicts the encoder of (encode-write #17). Fabricated
  precision, unearned confidence, or sloppy hedging anywhere in the prompt
  trains that register regardless of any rule against it. (Stop-1
  reverse-pass, 2026-08-21)

- **A10. Field coverage is a MEASURED distribution across the example set,
  not a checkbox any single example can tick.** A field named in prose but
  exercised in ~1 of N examples is taught at the rate the examples show it,
  not the rate the prose asserts it. The distribution — not the presence —
  is the deliverable, so it gets counted, not eyeballed.
  **Evidence (the miss this law exists to prevent):** E2/E3 gap 5 caught
  *"falsified claims hiding in metadata fields — the sweep runs 5 revise ops,
  5/5 patch content only"*, the weave plan closed it by adding **one**
  `situation` patch to **one** op, and the box was ticked. Nothing re-counted
  the set afterwards. v-next.6 shipped with 11 revise ops: 2 multi-field,
  8 title/content-only, `question` 0/11, edge descriptions 0/11,
  `revise_edge` 0 occurrences in 110,452 chars — and in production on
  2026-08-31 the encoder repaired `d827d22f`'s title and content while
  leaving the same dead value in its `situation` and an edge description,
  quoting the prompt's title rationale back in its own `reason` string.
  The prose said four fields; the examples said two; the output followed
  the examples. (§6a of docs/STALENESS-PROPAGATION-FINDINGS.md; nodes
  450650d5, 08913f27)
  **The test — run it whenever an example is added, split, or edited:**
  1. Build the op × field grid for every worked op in the prompt (the
     Field-coverage matrix below). Count, don't recall.
  2. **No field named in prose may sit at 0 exercised ops** — that is the
     A1 failure with a coat on, and it is invisible without the count.
  3. **No single field may carry the set's only rationale sentence.** Count
     where consequence clauses ("…embeds and ranks against itself") attach.
     A field with a rationale is taught; a field in a list is mentioned.
     Three title rationales against one situation rationale is the v-next.6
     defect stated exactly.
  4. **The shape spread must include the inverse case.** For revise that
     means at least one op touching NEITHER content nor title — otherwise
     "revise" silently means "rewrite the claim" and the aspect fields read
     as decoration.
  5. **Deliberate zeros are recorded as deliberate, with the reason.** An
     absence nobody wrote down is indistinguishable from an oversight — and
     gap 5 became a repeat defect precisely because a closed gap left no
     standing count behind it.

  **A10 is the revise-shaped instance of a wider family.** `E11`–`E16` state
  the general forms — rationale attachment, before-state depiction,
  graded-vs-binary checks, standing counts, numeric self-conformance, and
  teaching economy. Point those at every example family in the prompt, not
  just the revise ops; this defect was only *found* in revise.

## B. Placement laws (under T3)

- **B1. Placement beats content.** Same idea: 30% as a buried bullet, 60% as
  a named top-level section — a named section changes the schema of attention.
  (genealogy #101; v4→v5)
- **B2. Position resolves conflicts — last/later wins.** The Speed line beats
  the Actions line by position; final-position directives are what land for
  Haiku; the last line is the highest-attention slot (concrete beats earnest).
  (genealogy #155, #13, #48)
- **B3. You cannot instruct attention — you position it.** Five system-prompt
  arms failed where one recency-positioned user-content banner broke through
  (7/15 vs 4/15, interaction with template). The system prompt has an
  attention ceiling; the assembly layer is prompt real estate too.
  (genealogy #57, #85)
- **B4. A corrective principle placed late loses to the document's first-half
  gravity.** (genealogy #91)
- **B5. Earlier + canonical-marked examples dominate later elaborate ones.**
  source_refs taught only in late §7.6 never fired; taught in prose + early it
  hit 100% coverage. (genealogy #129)
- **B6. The gate sentence is the prompt's measured center of gravity.** Both
  Stop-1 probes (current AND revised arms) independently ranked "Encode what
  earns its place — new AND useful. That's the whole gate" as the single
  most behavior-controlling sentence in the document — every other rule reads
  as downstream of it. A teaching that must enter the per-exchange decision
  LOOP (not just shape output) attaches there; opener placement alone did not
  put two-registers into either probe's question loop. (Stop-1 probes,
  2026-08-21. Post-final note: Probe D re-ranked "integration, not recording"
  as most-controlling with the gate still #1 in its question loop — paradigm
  governs, gate executes.)
- **B7. Attach new teaching to measured high-attention carriers, not new
  sentences.** The 7-word equality fix placed inside the probe-ranked #2
  attention sentence entered the top-5 attention list itself; the same idea
  as a standalone bolt-on clause never cracked top-5 (Probe B vs D). Probes
  give us attention rankings — additions ride the sentences that measurably
  carry. B6 is the special case (the gate sentence for loop-entering
  teachings). (Stop-1 reverse-pass, 2026-08-21)

## C. Channel laws (under T2)

- **C1. Fix at the layer that binds:** schema > code guard > MCP description >
  prompt example > prompt rule. Hard constraints go in code (learned 3×
  independently); tool schemas are a hard contract, prompts are advisory.
  (genealogy #11, #76, #17)
- **C2. Tool-mechanics rules work best AT the tool** — the connect_to rule in
  the MCP description took violations 1/run → 0 where the same rule at the top
  of the prompt never did; the absorb description fix re-framed every caller
  at once. (genealogy #58, encode-write #40)
- **C3. But MCP is shared infrastructure** — encoder-specific teaching in a
  description contaminates every other caller; mechanics→MCP, craft→prompt
  (SPLIT won on every dimension). The prompt currently DELEGATES sibling
  resolution + edge vocabulary to MCP at three points: that coupling must be
  deliberate and tested, or inlined. (genealogy #22, #108)
- **C4. A field the system reads must be taught, delegated, or dropped.**
  `question` is one of recall's four embedding groups and is taught nowhere in
  the prompt (healer is the accidental sole filler); `personal`/
  `personal_context` sit in the writable surface at 5/8434 nodes.
  (encode-absent #45, genealogy #183)
- **C5. Encode-decode symmetry at the lane level (Tom, 2026-08-21):** every
  recall lane we build needs its encode-side teaching, or the lane starves.
  The machinery-without-guidance class, currently four deep: the `project`
  field (added, encoder never told); the `<actions>` channel (built as
  preparation for tool-use / file-touch recall — the encoder was never told
  to write work-state handles recall could match); situation work-context
  expansion (Tom's ruling b4ba57af, never taught); `question` (4th embedding
  group, taught nowhere). The 3-attention-moments vision (prompt / stop /
  tool-use, ec0af09d) makes this structural: tool-time recall has zero
  encode-side substrate today. Nuance: "actions = outcomes, not mechanics"
  stays right — the synthesis is outcomes STAMPED with work-state handles
  (files, tools, project), not encoding mechanics.
- **C6. Document the opaque, never the derivable — and put glosses where
  economics and attention agree.** The render is self-teaching for format
  (ledger: bracket types, ids, edge grammar "read for free"; only invented
  notations like `×N`/`(N more...)` are opaque and MUST be glossed — the
  undocumented-[aged] conviction). A prompt sample that mirrors what the live
  payload already shows every run is duplication, not teaching. Gloss
  placement trade-off: prompt-side glosses are near-free (BP1 cache, 1h TTL)
  but far from the point of use; payload-side glosses (the `<scout_legend>`
  precedent, the trimmed-stub's own self-description) land at point of use
  but cost tokens EVERY run. Decide per gloss, name the choice. (Tom's
  right-spot challenge + probe evidence, 2026-08-21)

## D. Content & voice laws

- **D1. Restraint lands ~100% and over-generalizes; generative lands ~20%.**
  Every restraint rule needs an equally prominent generative counterpart; read
  any generative regression as a restraint rule added somewhere. (genealogy #83)
- **D2. More prompt = more compliance, less depth.** +681 chars moved one axis
  while specifics elsewhere abstracted away; one principle beats an enumerated
  list; char growth is itself a defect to justify. (genealogy #208, #65, #212)
- **D3. Voice EQUALITY — Tom's ruling (2026-08-21): the two voices weigh the
  same.** Speaker info repeatedly out-weighing Anchor's is "totally wrong";
  both voices' information, insights, and quotes are equal-value encoding
  material — and **when the two voices CONTRADICT, the contradiction is itself
  interesting information** (encode it as signal, don't auto-defer to the
  speaker). The one scoped exception stands: the other side's explicit wording
  is the authority over MY PARAPHRASE of THEIR experience (temporal-authority
  case) — that's paraphrase-vs-source, not voice rank. Evidence of the
  standing failure: anchor_raw_quote ~6% vs user_raw_quote 80%; "prompt's
  gravitational field pulls toward operator-driven content" survived three fix
  waves. First-person Anchor voice; "the other side" as role; brain as medium
  never subject. (genealogy #49, #91; encode-notice #24; rule 4bb3fc03)
- **D4. State the positive principle, not the negation of the observed
  mistake.** (genealogy #106)
- **D5. Ground every referent inline for a stateless reader;** no internal
  jargon the reader never saw; verbs are real function names. (genealogy #204,
  #151, #175)
- **D6. Every word matters — single words have flipped behavior.** "filter" →
  summarizing; a "(rightly)" aside self-undermines; "encode" as an action verb
  opens the instinct gap. (T1; genealogy #222, #12, #175)
- **D7. Detail AND meaning, both encoded, linked** — and atomization has a
  cost: tight atoms collapsed the multi_session axis vs narrative bundles.
  Guide the tension, don't force one pole. (genealogy #101; encode-write #135)
- **D8. Cross-redundancy for facts** (number/name in title + content + quote);
  situation is a MOMENT, not a topic — the single biggest recall lever.
  (genealogy #51; v16 rows)
- **D9. Preserve uncertainty:** contradictory sources → `open` node; never a
  fabricated point value. (encode-write #71)
- **D10. Temporal: event_time default-on for the other side's experiences;
  resolve relative→ISO; the other side's explicit wording is the authority;
  temporal self-containment (nothing inherits turn/age/now).** (v16/v26/v36 rows)
- **D11. Staleness is a claims problem:** status lines are claims that rot; a
  state change falsifies claims across MANY nodes — sweep them (patch-mode
  makes it affordable); revise re-audits title AND type, carries forward
  still-true detail. The volatility family (BRAIN-CHALLENGES #1) has zero
  shipped fixes — v-next.4 is its first prompt-text response. (genealogy #27;
  encode-write #26, #145)
- **D12. Corrections are the load-bearing read:** flavors cover both voices
  (first-person parity — my own "done, deleted it" falsifies claims too);
  correction triple; lineage refs to both moments; revise the wrong node so it
  stops propagating. (v16 + v-next.4 rows)
- **D13. Differentiation at encode (T6 form 2).** Every write is also a
  recall-field edit: does the local neighborhood stay separable after this
  node lands? Evidence for the failure: true-duplicate pairs recalled 41×/42×
  together and selected 0× (encode-write #56); twin clusters steal top-5
  slots from gold — dedup is a retrieval operator (encode-write #112);
  crowds dim, singletons shine (divisive normalization, encode-notice #35).
  Current carriers cover only slices: the retrieval-divergence test governs
  the 1-vs-3 SPLIT decision, "revise rather than mint a twin" governs exact
  duplication — nothing guides the richness ceiling (the "contains
  everything" node) or overlap with existing catalog neighbors. The prompt
  pushes richness hard (be expansive, cross-redundancy) with no
  separability counterweight.
- **D14. Replace by contrast, not deletion.** When a word carries behavior
  you must keep but a frame you must lose, negate it in place — "integration,
  not recording" kept the write-compulsion and fidelity priming that Probe A
  traced to the bare word "recording", while flipping the paradigm; naive
  deletion would have lost both. Word-level corollary of A2's true/false law.
  (Stop-1 reverse-pass, 2026-08-21)

## E2/E3 OUTPUT — the example system, reverse-derived (2026-08-22)

Method (Tom's, this session): reverse-pass every example into what it actually SHAPES →
segment the shapings → find the shapes no segment covers → weave the missing ones into
existing examples rather than adding assets. Run **blind**: three cold Opus readers got an
examples-only corpus (18 fenced assets + 16 inline pairs, all prose stripped) so nothing
told them what the examples were meant to teach; I wrote my own derivation first and diffed.
Author-blindness is the point — I know the intent, which is exactly what hides the leaks.

### The segment map — axes the example set covers

| axis | covered | NOT covered |
|---|---|---|
| op shape | remember (heavy), revise-patch, revise-fieldwalk, multi-node sweep | standalone `connect`, `archive`, `absorb`, `disconnect`, **skip** |
| knowledge kind | fact/event, decision, correction, mechanism, principle, moment, identity, quote, interpretation | open/hypothesis; `lesson`/`finding`/`milestone` (read in catalogs, never authored) |
| catalog relation | net-new, extends, contradicts→patch, abstracts-two-instances | would-duplicate→revise as a *decision*; near-twin deliberately kept |
| voice source | theirs, mine, both, an agent's | **neither** (action-derived node); third party / >2 voices |
| certainty | asserted flatly | hedged, contested-unresolved, **conditional verdict** |
| register | cool technical, hot correction, affective/identity | uncertain, mundane/logistics |
| temporal | dated event, relative→ISO, sequence, hub | perishable/expiring state |
| domain | infra-eng (heavy), quantified personal ×2, research ×1 | everything else |
| scale | 1-node (modal), 2-node, 5-node | selecting from many candidates |
| **field coverage** (added 2026-09-01, A10) | **was: title, content, situation, reasoning, event_time, type** | **was: `question`, `thought`, `confidence`, `evolution_status`, edge descriptions — all 0 ops.** This axis was absent from the map entirely, which is why the E2/E3 pass could enumerate nine axes and still miss the defect that shipped. See the Field-coverage matrix below. |

### Ranked gap list (b=blind reader, m=my derivation, s=sibling stream af64db80)

1. **Skip / zero nodes** (b,m) — no example; every exemplar is max-density; `encoded="false"` reads as an obligation. All three readers ranked it top.
2. **Degree taught at 1.47 mean / 3 max, three nodes at zero** (b) — against R3's measured −20pp for degree 0–2. The rule says draw the edges; the set models the danger band.
3. **Modal batch = 1 node** (b, 7 of 10 calls) — contradicts the gate's "most turns produce several, I don't ration". A3: examples win.
4. **Conditional verdict without its truth condition** (s) + **perishable state nothing re-visits** (b) — one family: claims with no attack surface. Distinct from the as-of stamp (that's time-rot; this is condition-rot).
5. **Falsified claims hiding in metadata fields** (s) — verified: the sweep runs 5 revise ops, **5/5 patch content only**.
6. **Uncertainty / contested-unresolved** (b,m) — every node asserts flatly; `confidence` never set; the one hedge is a `thought` hanging off a confident correction.
7. **Third-party attribution** (b) — every asset is dyadic; likely error is flattening a colleague's words into the counterpart's voice lane.
8. **The human is wrong** (b) — `correction` is defined 3/3 as MY behavioral flaw with a self-diagnostic pattern.
9. **Action-derived node carrying neither voice field** (b,m) — never shown; inverse defect live in the lexicon node (see D-list).
10. **Retire / merge / archive** (b,m) — zero examples. **Needs Tom's ruling: deliberate (S2's job) or gap?**
11. **Flat durable facts** (b) — inflation to reach exemplar density; abstraction-ladder pressure turns a checklist into a pseudo-principle.
12. **Sensitive material** (b) — no floor anywhere, and Priya's anxiety is precedent for encoding a third party's health.
13. **Within-session position change** (b) — supersession only ever shown cross-run.
14. **`similar_to` for deliberately-kept twins** (b,m) — measured 3.4× rescue lift, zero uses; and the canonical plants a near-twin pair it never resolves.

### Verified coherence defects (checked against the text, not taken on report)

- **D1. Pair `grounds` runs backwards.** The principle node emits `grounds` → the mechanism, while its own `why` argues the recipe is the findable handle and the principle "the meaning it serves". Detail should ground meaning. **In the T6 carrier — and I edited that `why` at 9b without seeing it.**
- **D2. Prompt-deictic refs inside stored edge `why`s** — "A7 declares my continuity; A4 names the structural limit…", "A6's encoding rule … gets its philosophical justification HERE." Section numbers and HERE die on storage. **E9's second conviction, in the zone I audited at Stop 10.**
- **D3. Sam/tom split** — 8 `<trace-tom-*>` placeholders against 32 "Sam" in node bodies. A rename that reached prose, not placeholders.
- **D4. Canonical quote node is dated 2026-03-20 and claims to be the moment insight b7e2054d "became conscious" — b7e2054d is dated 2026-03-02 in the excerpt.** 18 days earlier.
- **D5.** An `insight` node carries `correction_pattern` while two `correction` nodes carry it — type/field incoherence on a filter axis.
- **D6.** 2024 date island (11 instances, the auth-rewrite cluster) against 2026 everywhere else.
- **D7.** Marcus reasoning claims the number is cross-redundant across title/content/quote, but the title says `27:12` and the other two spell it out — the node's own rationale is lexically false.
- **D8.** The BAD batch's node is also missing situation/reasoning/edges, but only hub-only propagation is flagged — unflagged badness reads as acceptable.

### Weave plan (no new assets — each gap lands in an example that exists)

| move | closes | into | status |
|---|---|---|---|
| add a `connect` op drawing `similar_to` between the two planted ring-buffer twins | gap 14 + standalone-`connect` + the unresolved near-twin | canonical batch | DONE — Tom ruled convert. Canonical is now `brain_batch` with 6 `remember` ops + 1 standalone `connect` (9c04e7a1 `similar_to` 5d11c0a7), which is what the prompt's own rule at ~798 prescribes for a mix. Teaches `connect` vs `connect_to` and that the field is `description`, not `why`. **Self-check caught a contradiction the conversion created:** the round-1 line named two single-purpose batches directly above an example doing the opposite — tool names now stripped from it and from the sibling-title rule, so ~798 is the single owner of tool selection (also retires an E7 redundancy). |
| make one sweep patch a `situation` patch; give one patched node its truth condition | gaps 4, 5 | sweep example | DONE — b8e05f92 gains a `situation` replacement in the same op; the why-list now carries "staleness is not only in `content`" + the truth condition ("any rebuild", not "before merge") |
| name what the window did NOT earn a node for, in one line | gap 1 | canonical framing | DONE — demonstrates-list gains "What this round did NOT encode": the TCP migration got an edge, not a node; six nodes is an outcome, not a target |
| fix D1–D8 | coherence | in place | DONE (ad7e68e) — D5/D6 dropped on measurement, D7 dissolved into the reasoning fix |
| enrich the catalog excerpt so honest edges exist for the two zero-edge nodes | gap 2 (without manufacturing edges — Tom's honesty ruling holds) | canonical excerpt | DONE (ad7e68e) — +2 catalog lines, `completes` and `resolves` |

## E. Global audits & process gates

- **E1. Contradiction audit (T4).** Enumerate every in-prompt tension; stamp
  each `DEAD-CONFLICT` (fix: one side wins, by position) or `OPTIONALITY`
  (keep: mark the judgment it grants). Current known: anchor_raw_quote
  required-vs-selective; residue-deferral vs window-slides; expansive vs
  2-rounds. Resolved: Actions-two-reads vs Speed-"everything I need"
  (Stop 8, DEAD-CONFLICT — the Speed line now defers to the two reads).
- **E2. Example sweep after the draft settles** — every example re-audited
  against every rule that changed (A3), on all dimensions (A4), for leakable
  ids (A5).
- **E3. Coverage matrix:** every box → carrier (line/example/MCP/code/
  open-by-design). Empty cell = caught gap.
- **E4. Eval gate:** register DORMANT → eval → activate → sync. Never sync
  between register and activate. No single-run conclusions — same-capture
  variance exceeds arm deltas (079d9736); pendulum-test both directions;
  check the axes you didn't target (v15.9 fixed one item, cost the cohort
  -30%). (genealogy #18, #33)
- **E5. A rule earns its place only if removing it degrades behavior with
  examples still present.** (genealogy #142)
- **E6. Guide, don't engineer (T5).** Emergent types, open KV, lazy promotion,
  "or any other type that fits" stay open. Reflective questions beat
  procedural checklists in the adjacent heartbeat eval (100% vs 83%±29) — the
  prompt guides a judge, it doesn't compile a procedure. This box vetoes
  over-specification the other boxes might invite.
- **E7. Redundancy audit — T4's third sibling.** Every idea stated more than
  once in the prompt is either deliberate reinforcement (mark it — e.g.
  keep-when-unsure appears in opener AND gate region, and the opener instance
  ranked #1 for attention) or a layering scar (merge it — the P2/P3
  details-then-meaning seam was two eras writing one thought twice). Like E1
  for contradictions: unmarked repetition is accretion by default. (Stop-1
  reverse-pass, 2026-08-21) Marked deliberate so far: opener↔pair E=mc²
  callback (wires rule to demo); defaults-brevity instinct vs Speed
  be-expansive (catch-myself register vs write-moment posture); skip-list
  stated at Skip bullet + gate paragraph (under-encoding's double guard,
  Stop-8 fold cut the third).
- **E8. Format-grammar audit (T7 operationalized).** Census of the current
  assembled prompt: ~10 sign families in use — bare ``` fences, ```json fences
  on pseudo-JSON (comments, unquoted keys — a mildly WRONG signal), real
  `<tags>`, `{curly}` shape-placeholders, `<id-of-...>` and `<trace-...>`
  placeholders (angle-bracket COLLISION with real-tag grammar), `[bracket]`
  provenance/type tags, Bad:/Good: prose pairs, bold rule-leads, §-headers,
  // and # comments inside examples. Target grammar: angle brackets mean
  real-payload-structure ONLY (placeholders unify to one curly form — this
  directly serves A5's leakage problem: one placeholder grammar, one
  disclaimer); one honest example fence; Bad/Good pairs standardized. Runs as
  a dedicated draft-wide sweep in the global-audit stage, probe-verified
  (ask a fresh reader: what here is real, what is example, what is slot).
- **E9. Probe emissions get the invariant sweep before celebration.** A new
  teaching's output must be audited against ALL standing invariants — temporal
  self-containment, voice derivation, placeholder discipline, id copying —
  not just the new behavior. The miss that minted this box: Probe I's first
  living `thought` was showcased while carrying "turn 9 just showed…" — a
  window coordinate, the exact leak class three prompt versions fought
  (a82d7b7d ratchet). Tom caught it; the verifier had checked only what was
  new. New fields have no catalog ratchet yet — their first examples fully
  define the register, which makes the sweep cheapest exactly when it
  matters most. (2026-08-21)

- **E10. MCP surface audit — does the tool layer support the prompt, or fight
  it?** (Tom, 2026-08-23.) Runs AFTER the prompt settles, because the prompt is
  what defines the support the MCP layer owes. **This is not a symmetric
  contradiction check:** the contract field summary is injected last under
  "from contract", so by A3 the MCP surface WINS any disagreement by position —
  a contradiction here is the prompt losing, not a tie. Proved twice
  (68fd6e05: probes emitted `locked`/`encoding_source` that no example sets;
  17c604ad: the `reasoning` leak survived an example-side fix because the field
  description was driving it). Four checks:
  1. **Coverage, both directions.** Every field the prompt teaches appears in
     `get_writable_fields()`; every field the list advertises is either taught
     or deliberately silent. Known 2026-08-23: taught-but-absent — `question`
     (10,662 nodes, own vector at weight 0.90), `event_time` (430),
     `source_refs` (26), `thought` (1), `emotions` (0); advertised-but-dead —
     `critical` (0 of 12,041), `personal` (4), `personal_context` (2),
     `encoding_source` (spurious emission; wants `system_stamped` like
     `project`), `locked` (silent no-op for `encoder:sonnet`).
  2. **Agreement on shared fields.** Where both surfaces describe one field,
     they must not contradict. Fixed 2026-08-23: `reasoning` ("why this was
     encoded" → evidence strength, 2a5acbd). Open: the `emotion`+`emotion_label`
     pair (1,070/981 nodes) vs the prompt's `emotions` array (0) — two live
     representations of one dimension, a migration question.
  3. **Dereference.** Every prompt line that POINTS into a tool description
     must resolve to text still saying what the pointer promises — the prompt
     doesn't merely coexist with MCP, it depends on it (1b7984f8 found three
     such delegation lines).
  4. **Cross-caller safety.** A description aimed at S1E also teaches S2 units,
     the healer, and Anchor-via-MCP (807394de). Membership changes ship as their
     own eval-gated change (154793b6), never folded into a prompt package where
     the eval can't attribute the delta.

  Why it needs its own box: the section-by-section walk audits the prompt
  against itself, so this entire class is structurally invisible to it. Tom had
  asked for it once before as a one-off (8a9f5996, clean at the time); it never
  became a step, and everything above accumulated in the gap.

### E11–E16 — the v-next.7 boxes (2026-09-01)

Six audits generalized out of one defect: v-next.6's prose named four revise
fields, its examples showed two, and production followed the examples
(`450650d5`). Each is stated for the whole prompt, not for revise, and each
names where else to point it. **A10 is the revise-shaped instance; these are
the general forms.**

- **E11. Rationale-attachment census — the emphasis is where the CONSEQUENCES
  are, not where the mentions are.** Wherever the prompt offers a menu of N
  options, count where the *consequence sentences* attach ("…embeds and ranks
  against itself", "…is the failure mode"). An option with a named consequence
  is TAUGHT; an option in a list is MENTIONED, and output follows the taught
  one. **Conviction:** v-next.6 attached three consequence sentences to `title`
  and one to `situation` (the dead-referent case only) — the encoder repaired
  title+content, quoted the title rationale back verbatim in its own `reason`
  string, and left `situation` stale. Prose that named four fields lost to a
  3:1 rationale ratio. **Run it on:** the four correction flavors (does one
  carry all the why?), the three closure branches, the two registers, the
  atomization tie-breakers, voice fields vs each other, `connect` vs
  `connect_to`, the skip-vs-encode gate, the two reads. Ratio, not presence,
  is the output.

- **E12. Before-state depiction audit — an example teaches recognition only if
  its INPUT shows the thing to be recognized.** Every excerpt in the prompt
  renders a SUBSET of what the assembly actually renders. Where a lesson is
  "notice X, then act", X must be visible in the depicted input, and the
  depiction must not be lossier than production on that surface.
  **Conviction:** ZERO catalog excerpts rendered a `situation:` line and the
  one excerpt edge line carried no description — while `build_node_catalog`
  renders both (full-rich; `edge_style: 'oneline'` is the *surface* path, not
  the encoder's). So the encoder had no template for a stale situation or a
  stale edge `why`. **Run it on:** every catalog excerpt, the timeline sample,
  the `<continuity>`/residue render, the `<scout_notes>` sample, the
  `[associated]` stub render. Test per lesson: is its trigger visible in the
  depicted input, and does the depiction match what assembly emits?

- **E13. Graded-vs-binary check audit — does each self-check catch PARTIAL
  compliance, or only total absence?** **Conviction:** the mandatory `sweep:`
  line is the prompt's own completeness guard, and it fires only on
  `sweep: none`. A repair reaching two of four stale surfaces wrote an honest
  `sweep:` line, passed, and closed `DONE`. A check that can only see "did
  nothing" cannot see "did half" — and half is the common failure, not the
  rare one. **Run it on:** the `sweep:` line, the Review fence, the
  atomization same-batch and edge-description tests, the two-reads gate, the
  3-anchor bar, the skip verdict. Method: construct the half-done case for
  each and ask whether the check's own wording rejects it.

- **E14. A gap closed by an instance fix leaves a standing count, or it
  reopens.** **Conviction:** E2/E3 gap 5 caught this exact defect
  ("the sweep runs 5 revise ops, 5/5 patch content only"), the weave plan
  closed it by adding ONE `situation` patch to ONE op and marked it DONE, and
  nothing re-measured the set. v-next.6 shipped with the distribution
  unchanged in kind. Fixing the instance was right; recording it as closed
  without leaving a measurement is what made it recur. **Run it on:** every
  DONE row in any weave plan — each must name either the invariant now
  enforced (a law plus its count) or an accepted one-off with its reason.

- **E15. The prompt obeys its own stated numbers.** A9 says the whole prompt
  is an example; this is its checkable half. Census every explicit numeric
  rule and test the prompt's OWN example payloads against it.
  **Conviction:** the `partially_resolves` why drafted for v-next.7 landed at
  310 chars against the prompt's stated 120–180 band — a violation sitting
  inside the document that states the band. **Run it on:** edge-why length
  (120–180), `source_refs` 1–3, "3+ distinct turns", title-length norms,
  situation trigger-register form. Cheap and scriptable; belongs in E2's
  sweep.

- **E17. Correction shape — does the fix STAND ON ITS OWN, or ride along
  beside what it corrects?** (Tom, 2026-09-01: *"sometimes there is value in
  keeping the history but in many cases leaving the stale history within the
  fresh node is a disaster and overloading of information."*) Census every
  revised value in every example: is the old value inline in the new text, and
  in WHICH field? **Inline history is legitimate only in `content`, and only
  where it is load-bearing** — a validity interval, a measurement baseline, a
  forensic row. Everywhere else it is a dead claim competing with the node that
  is now right. Note the architecture: the `supersedes` edge and
  `correction_enrich` already annotate lineage on every canonical pull
  (`ab0ecacd`), and `event_time` carries validity — so an inline `(was X)` is a
  THIRD copy, and the only one that pollutes the embedding.
  **Why it is a standing bias, not an incident:** appending is cheaper than
  re-authoring, and it satisfies "revise" without forcing a re-decision of what
  is true — `e05a071f`'s operation-vs-disposition line. Tom caught the same
  reflex in prompt EDITING a week earlier (`bed31596`, "why only additions and
  not changes"); this is the same bias one level down, inside the content the
  examples teach.
  **Three convictions, all 2026-09-01:**
  1. The sweep gave `a45c88f1` the title *"Rollout order (superseded …): was
     auth-rewrite → api-gateway → cli"* — the whole dead sequence embedded,
     sitting beside its successor *"Rollout order after auth-rewrite was
     scrapped: api-gateway → cli"*. Shared opening words, shared tail: exactly
     the twin-competition the prompt's own atomization rule warns about, while
     a `supersedes` edge already carried the lineage. Fixed in v-next.7.
  2. **The style made a real defect undetectable.** A removal-diff detector
     over 206 encoder revises found nothing for the canonical item, because
     production wrote `9.7.2 (was 9.6.0 …)` — nothing was removed. The detector
     had to be reframed from removal to supersession to see its own case.
  3. 5 of 8 `content_edits` in the shipped examples are additive — the style
     transmits, and none of them had been audited on this axis.
  **The test is position and tense, not presence:** is the old value in the
  ASSERTION slot, or subordinate and past-tense? *"X (was Y)"* in a title
  asserts both. *"Y until Z"* in an edge description narrates one.
  **And the boundary, or this rule gets over-applied** (found by running the
  census, 2026-09-01): a retrieval surface that names a dead thing *because the
  deadness is the node's SUBJECT* is not a violation. The sweep's successor
  node carries `situation: "When picking up the rollout queue — auth-rewrite no
  longer exists as a step"` — correct, and rewriting it would lose the warning
  a future asker needs. The violation is a surface still ASSERTING a value that
  has been replaced, competing with the node that now holds the live one. Dead
  subject: keep. Dead claim: cut.
  **Run it on:** every `content_edits` new-string, every revised title, edge
  descriptions, the §7.6 identity exemplars — and the temporal
  validity-interval examples, where inline history is CORRECT and must be left
  alone (the yoga node's *"(was twice a week from 2023-08-11)"* is the model of
  a legitimate one).

- **E16. New teaching rides an existing explanation.** A new bullet or comment
  block is the expensive default; a clause folded into the explanation that
  already covers that ground is the cheap one, and examples outrank the prose
  around them anyway (`8225980e`). **Conviction:** the first v-next.7 draft
  added +7,698 chars, of which the six largest hunks (~5,600) were new
  explanatory prose while the behavioral levers — the added fields — were
  100–220 char hunks. Folding every added block into an adjacent existing
  explanation produced +2,913 with zero teaching lost; one op got THREE new
  fields for −2 chars. **Run it at draft close:** for each added block, name
  the existing explanation that could carry it as a clause. If one exists,
  fold. Report the net-per-hunk table, not just the total.

## R. Recall→encode pointers (Opus scout, 2026-08-21)

Full 30-row table: [challenges/recall-to-encode.md](challenges/recall-to-encode.md).
The recall-side cross-examination every stop verdict must now pass (the
encode-decode symmetry check the first six stops lacked). Headliners: R2
trigger-register situation (14%→54%), R3 degree floor (0-2 edges = −20pp),
R5 why-length admission threshold (<80 chars filtered before semantics),
R8 live defect (no correction example carries a real corrects edge),
R23 measured verb lift (after 5.0×, instantiates 4.1×, extends top rescuer).

## Fields — the aspect model (Tom, 2026-08-21) vs measured reality

Each field is a different ASPECT of the memory. Two standing decisions ride
above the table: **what lives in main content vs in fields**, and per field
**teach / delegate / recalibrate / drop**.

| Field | The aspect (intent) | Measured reality | Taught today | Decision needed |
|---|---|---|---|---|
| `question` | **Association** — the query this memory answers | Weakest standalone signal (4% hit@5) yet HIGHEST production weight 0.90 — inverted (2afa20d8); 94% missing; healer is sole filler | Nowhere | Teach in canonical example / recalibrate weight (code) / delegate to healer deliberately |
| `situation` | **Recall gate** — when this surfaces; a moment, not a topic | Strongest lever: +0.15-0.35 cosine over content for right nodes (5881538c); independent LAF lane, best combined configs (c7821ad2) | Required + examples — strong | **Gap: Tom's work-context expansion (project+files+libs+task, b4ba57af) never reached the prompt** — encode-side implication untaught |
| `emotion`/`emotion_label` (schema) + `emotional_context` (KV) | **Emotional register** — moments of happiness, frustration, 100 types, WITH reasoning; true also to MY perceived emotions | Non-zero emotion 39%→0.7%; emotional_context 0 nodes brain-wide | Field-list only + set-dressing | Revive via example (A1), both voices' emotions; unify the three surfaces |
| `user_raw_quote`/`anchor_raw_quote` | **Voice anchors — equal weight (D3)** | 80% vs ~6% asymmetry | Required-rule vs selective-example tension | Equality rewrite + contradiction-as-signal |
| `thought` | **My own read** — value as a thinking thing | 0/120 emitted | Subsection, no example | Example or drop |
| open KV (open_question, doubt, trigger, impact_scope…) | **Any aspect the named fields don't hold** — the field name is itself an encoding prompt | event_time 308 nodes (taught) vs emotional_context 0 (set-dressing) — teaching decides | Open-fields para | Which keys get worked-example status; `open` type vs `open_question` field — one home |
| `source_refs` | **Episodic anchor** — the moment it was learned | 0.3% of live nodes — dead in production | §7.4 prose + §7.6 placeholders only | Real-id demonstration in canonical, or accept-sparse deliberately |
| `event_time` (KV) | **Temporal anchor** | 308 nodes; 16%→59% after v26 examples | Doctrine + examples — strong | Holds (proof A1 works) |
| `project` | **Work-scope handle** — which project/codebase the memory belongs to | Machinery shipped (field exists, renders on nodes); encoder never guided (Tom 2026-08-21) | Nowhere | Teach as one work-state cluster with situation work-context + tool/file handles (C5) |
| **edges** (relation + why) | **Association with meaning — first-class encoding output, not plumbing** | why-craft taught; 96% of S1E edges point new→old (structural bias, untaught); vocabulary delegated to MCP | Edges section + examples | Direction bias; is MCP delegation deliberate (C3)? |
| content ↔ fields split | What carries substance vs what carries aspects | — | §7.4 judgment + KV para | Needs one crisp rule both can apply |

## Example inventory (checkboxes per example — fills first)

Every example asset in the assembled prompt, audited as a multi-dimensional
teaching surface. Columns: what it teaches now / spare capacity / risks.

| Example asset | Location | Teaches now | Spare capacity | Risks |
|---|---|---|---|---|
| Timeline sample + Bad/Good title pair | What I Receive | Timeline grammar (turn/other/me/actions/provenance/scout_notes); encoded-stub semantics; temporal self-containment (the Bad/Good pair) | An actions line judged encode-worthy vs not (the actions-guidance gap, C5); work-state handle demo; a `<memories_beyond_catalog>` stub if that ships | It's orientation-format first — teaching load competes with the reader learning the input shape; engineering nouns (A6) |
| Flat→Rich templates ×4 | Nodes | Shape transformation via {placeholder} (A6-safe); verbatim+meta; emotional register (#3); term-vs-misreading (#4) | A 5th template: status-claim → self-contained claim w/ date (D11); RICH templates could name situation/question so shape includes aspects | Implicit-only contrast (FLAT=false is unlabeled); more templates = dilution (D2) |
| Edge why Bad×4 / Good×4 | Edges | why-craft contrast — the prompt's only pure Bad/Good bank (T1 model); generic-gloss ban | A direction demo (an old→new edge Good — counters the untaught 96% new→old bias); a twin-separation why (T6) | Low — abstract placeholders keep it safe; the bank's form should be copied elsewhere, not grown |
| Temporal Ex1 (Grandma hub) | Temporal | time_anchor hub gate (3+ events, date-as-topic); anchored_to; sibling title form | Could carry full fields on one event node | **Live A4 violation: event nodes carry NO situation/reasoning and anchor content is `"..."` — teaches thin fields, the exact v15.10 failure class (genealogy #18)** |
| Temporal Ex2 (Nadia ACL, wholistic) | Temporal | 3 resolution paths; temporal authority (paraphrase-vs-source); correction node on my own wrong gloss (D12); Allen composition; fact atoms; open node for future date; inline Bad-content comment (T1 ✓) | Emotion register on the injury/recovery arc (natural home, D-emotion); question field on the fact atoms | 154 lines — the D2 poster child; several nodes lack reasoning (A4 leak); Tom already flagged length-class edits here twice |
| Canonical 5-node batch + catalog excerpt | Speed | situation+reasoning ×5; voice symmetry w/ selective anchor_raw_quote; cross-redundancy (27:12); event_time; id-copy from grounded excerpt + sibling title beside; **REAL corrects edge to d94f07b2 (R8 fix — placeholder+escape deleted, Stop 9a)**; question ×2 selective (C4, asker idiom, named restraint bullet); ONE thought on the correction (A1 revival); refs-absent named deliberate (13f72658); interpret/expand Bad beat on the quote node (A2, 5a's queued obligation); near-twin id-pick; open keys; N=3 earned principle; 5-type spread | title/type re-audit teaching (encode-write #26 — probes do it unprompted, still untaught); work-state handles demo (C5 — open, 9b decision) | Highest-attention asset (B5); probe L watches: question emitted 5/5 (all asker-idiom, scenario lacked a paraphrase-bait class — retest in 9b), full-rewrite chosen on total-premise-reversal (defensible restructure case) |
| Detail+meaning pair | Speed | Two-register law as tool-call; grounds edge why names TWO retrieval surfaces + "kept deliberately separable so recall can choose by intent" (THE T6 carrier, Stop 9b); question on the mechanism, absent on the principle (selectivity's second demonstration); anchor_raw_quote on a mechanism | — | RESOLVED (9b): opener↔pair E=mc² callback marked DELIBERATE — the callback wires rule to demonstration; cutting it orphans the demo. Probe M: pair behavior transferred fully (fact+principle, grounds why naming both surfaces, discriminating titles) |
| revise_batch patch/patch/field-walk (ghi789) | Speed | Patch→field-walk progression (T1 via comments); content_edits as correction default; preservation-by-construction demo; ladder line names the three rungs (claim → fields → neighborhood) | title/type re-audit on revise (encode-write #26 — probe K did it unprompted; still untaught in text); question on the dosage-class node (Stop 9 zone) | Split RESOLVED (Stop 8): ghi789 = one node's fields, sweep = one event's neighborhood — each names its lesson |
| Sweep example (BAD hub-only + 4-patch GOOD batch) | Actions/Speed seam | Event→multi-node propagation (R16, real ledger case); falsified-referent law (verdict node); edge-line id targeting + supersedes-over-remint; grounded source_refs copied from a trace= attr (R7 + three-connections); restraint clause (patch only what the event falsified); lookup question on the decision node (9b) | UPDATE-append counter-clause landed in the content_edits paragraph instead (stop-8 addendum) | Dev-domain skin nearer our corpus (A7) — accepted risk, package eval is backstop; +5.5K chars, first asset to push draft over v37 |
| Lexicon example (second misreading → interpretation) | §7.6, after A2 | The commissioned upgrade move (4bb5b1e8 §2): second occurrence visible via catalog = the signal; person-lexicon entry with trigger-register situation firing at the utterance; twin-incident trap shown (A2 law); emergent `interpretation` type demonstrates T5 open taxonomy; abstracts edge by copied id; question = the future moment's phrasing | voice anchor on the interpretation (defining quote rides the grounding event instead — detail/meaning split, decide at Stop 10 zone pass) | Probe N: full transfer on a fresh surface (retire/starter) — upgrade minted, trap avoided, honest sweep:none, self-directed thought about the reflex-vs-record gap. §7.6.A2 deliberately UNCHANGED: forcing `corrects` there would violate edge honesty (no prior-belief node exists in its scene); R8 satisfied by the canonical since 9a |
| §7.6 A6/A7/A4/A2/A3/A8 | §7.6 | Identity/hot-register encoding; anchor-voice depth; locked + trigger usage; correction at register; agent-as-other-side (A8); source_refs shape (placeholders) | **A voice-disagreement example — me holding my ground with evidence and encoding the contradiction as signal (new D3): none of the six shows it; every §7.6 example is me being corrected or seeing, never me disagreeing** | Big real estate; locked generosity needs its disclaimer kept; placeholder discipline must survive edits (A5) |
| v-next.4 sweep example (BAD hub-only + full sweep) | candidate | Sweep discipline (D11); content_edits patch form; labeled BAD contrast (T1 ✓); first-person falsifying evidence (D3/D12); supersedes lineage; edge-visible neighbor walk | Fix the wrong error name (`connect_to_bad_id`) in the adjacent prose while landing it; differentiation beat | ~100 added lines (D2); territory overlap with ghi789 (above) |
| MCP description examples (brain_batch, connect_to, absorb) | MCP | connect_to resolution scopes; sibling-vs-catalog forms; forward-reference example; empty-why anti-pattern; vocabulary + never-generic ban | Correct error-name semantics live here too (mechanics = MCP home, C2); absorb's content-destructive warning held — keep | Shared across ALL callers (C3) — no encoder-specific teaching; changes need the 8-step MCP eval gate |

## Field-coverage matrix — revise ops (A10; rebuild on every example edit)

Counted, not recalled: `./dev python3` over the assembled prompt, one row per
worked revise op. **v-next.6 = the shipped default; v-next.7 = candidate,
unpromoted.**

| op | node | v-next.6 fields | v-next.7 fields |
|---|---|---|---|
| canonical | `2b8ef0c1` | title | title *(deliberate: edge-line-only, title is all the excerpt shows)* |
| canonical | `7c1a4d93` | title, type | title, type, **evolution_status** |
| ladder 1 | `4a9f21c7` | content_edits | content_edits *(deliberate: the one-claim patch rung)* |
| ladder 2 | `d0e4b856` | situation | **situation, question, thought** ← the no-content/no-title op |
| ladder 3 | `97b1f24e` | title, content_edits, situation, reasoning, event_time | unchanged + situation now visibly de-stales; question deliberately untouched (not falsified) |
| canonical | `6ba3f17d` | **op did not exist** | **NEW op** — title, situation; keeps `type: "open"` *deliberately*, takes `partially_resolves`. The narrow-to-residue branch (Tom, 2026-09-01) |
| BAD | `e91a6d05` | content_edits | content_edits *(deliberate: labeled BAD)* |
| sweep | `e91a6d05` | content_edits | content_edits |
| sweep | `7d21c4aa` | title, content_edits | title, content_edits, **evolution_status** |
| sweep | `b8e05f92` | title, content_edits, situation | title, content_edits, situation, **question, confidence** |
| sweep | `c37d10be` | title, content_edits | title, content_edits |
| sweep | `a45c88f1` | title | title *(+ its edge description, via the new `connect` op)* |

**Regression guard (Tom's constraint, 2026-09-01): title and content must not
go backwards.** Verified by script over both files — `title` dropped from 0 ops,
`content_edits` dropped from 0 ops, and every retained title/content string is
byte-identical to v6. Title *rises* 7→8 on the new `6ba3f17d` op. (One scripted
"content TEXT CHANGED" flag on `e91a6d05` is a false positive — that id appears
on two ops and an id-keyed dict collides; both patches diff byte-identical.)

| field | v6 ops | v7 ops | note |
|---|---|---|---|
| title | 7 | **8** | +1 from the narrow-to-residue op; no existing title touched |
| content_edits | 7 | 7 | all 7 byte-identical |
| situation | 3 | 4 | |
| question | **0** | 2 | A1 violation in v6 — highest production weight (0.90), 94% missing |
| reasoning | 1 | 1 | |
| event_time | 1 | 1 | |
| type | 1 | 1 | |
| thought | **0** | **5** | 0/120 emitted in production. Four SHAPES, deliberately (Tom, 2026-09-01): a question left open, an idea it suggests, a warning for next time, a read/curiosity worth following. One example would have defined the whole register with no catalog ratchet to correct it (E9) |
| confidence | **0** | **0** | **DELIBERATE ZERO** (Tom, 2026-09-01): *"i dont know what to do with confidence and whether it should even be a number and not text (not now)."* A modelled value transfers as a default (A4), so seeding one ahead of that ruling would decide the question by example. E2/E3 gap 6 stays open and is now recorded rather than half-answered |
| evolution_status | **0** | 2 | two values shown (`resolved`, `dismissed`) so the vocabulary transfers, not just the key |
| `partially_resolves` | **0 (prose only)** | 1 | **A1 violation in v6** — the prose taught three closure branches and demonstrated one. Governs the census's 8 partially-answered-never-narrowed nodes (`644dc1e0`) |
| edge description | **0** | 1 | via `connect` upsert — `revise_edge` is NOT in `ENCODING_TOOLS`; `GraphDAL.add_relation` is a field-preserving UPDATE (verified live, node ff95dde0) |
| `source_refs` | 0 | 0 | **DELIBERATE ZERO.** Prose (REPLACE semantics, "never pass `[]` as a no-op") warns about a footgun whose correct behavior is *omission* — and an omitted field cannot be shown in an example. Recorded per A10.5 rather than forced. |
| voice fields, emotion | 0 | 0 | **DELIBERATE ZERO** — revise-time voice/emotion rewriting is not a behavior we want; they belong to the authoring moment. |

**Rationale-attachment count** (A10.3) — where consequence clauses attach:

| | v-next.6 | v-next.7 |
|---|---|---|
| title | **3** | 2 — one generalized surface clause + one title-specific line (restored on Tom's no-regression constraint) |
| situation | 1 (dead-referent case only) | shares the generalized surface clause |
| question / edge description | 0 | 1 each |

**Closure-branch coverage** (the `open` node's three exits — prose at flavor 3
teaches all three; A10.2 says none may sit at zero examples):

| branch | v-next.6 | v-next.7 |
|---|---|---|
| fully answered → retype + `resolves` | `7c1a4d93` ✅ | `7c1a4d93` ✅ (+ `evolution_status: resolved`) |
| partly answered → stays `open`, narrowed, `partially_resolves` | **0 — prose only** | `6ba3f17d` ✅ |
| answer opened a NEW question → close-and-mint | **0 — prose only** | **still 0 — KNOWN OPEN GAP** |

Close-and-mint is deliberately left for a later pass rather than bundled here:
the two branches now shown sit in the same round, which is what makes the
*discrimination* teachable ("is anything still unknown?"). A third branch in
the same batch would dilute that contrast. Recorded per A10.5 so it is a
tracked absence, not an oversight — the exact failure mode A10 exists to stop.

v-next.7 replaces the three title-specific rationales with one surface-general
clause ("a stale value survives in every surface I leave it in… whichever
surface I habitually skip is the one that goes stale") — so the set no longer
teaches a favourite field. **That is the change most likely to matter and the
one the eval must isolate.**

## Coverage matrix (fills during the walk)

| Section | Boxes applied | Status notes |
|---|---|---|
| Opener | B1, D3, D6, D14, A9, E7 (+B6/B7 discovered) | **DRAFTED** (scratchpad opener_final.txt, probes A/B/C/D): B1 restored via named 'Two registers, every exchange' lead; D3 carried by 7-word equality line at probe-#2 attention + disagreement in vigilance list; D14 'integration, not recording' contrast-foil (write-drive verified kept); A9 fake 95/70 → true mechanism; E7 P2/P3 seam merged. +168 chars vs current. Deferred out: two-registers loop-entry → gate sentence (Speed stop, B6); evidence-lean sentence → flavor 4 (Reading stop). Residual, deliberate: verbatim-priming now arrives downstream, not in opener. |
| What I Receive (+ assembly: banner, catalog, timeline) | B3, C5, E1/E7, stale-gloss class, T5 | **IN PROGRESS.** Caught: line-53 `encoded="true"` gloss contradicts line 62 (stale pre-B4 text — fix in draft). Queued draft edits: line-53 fix; `<attention>` gloss line in channel list; C5 pointer clause on the actions bullet. Banner: v1 text superseded (staleness-biased, Tom no-ship); v2 probe-failed (3:1 tilt); **v3 passed interview tier** (Probe F: symmetric read, no manufactured revisions; residual: additive-info-misread-as-supersedes → wide-sweep watch dimension). Stubs: ruled IN (subconscious framing); placement leaning [associated]-tag-in-catalog, Tom's nod pending. Both assembly changes flag-gated, behavioral gates (staleness set + wide sweep) still ahead. |
| Scout | E7, C6, T5 | **DECIDED (subtraction-only):** cut triple-stated scout_notes arrival (in-payload legend is the point-of-use gloss); trim handoff format-block to its three action-semantics lines (weave anchors / check catalog first / [me]=possible paraphrase); cut the vague 'primed for one kind of atomization' filler. Deliberate E7 reinforcement marked: anti-deference in posture section + defaults list. Section otherwise HOLDS (retirements verified-current). |
| Reading the conversation | D3, D12, D11, D4, E1, E7 | **DRAFTED + PROBE-VERIFIED (G):** v-next.4 first-person parity intro adopted; state-claims widen on flavor 3; NEW evidence-lean sentence on flavor 4 — Probe G's (d) flipped from 'none found' to quoting it, (e) named the lean while keeping type open ('treating a traced count and a recollection as equally weighted would itself misrepresent'); residue deferral now carries turn anchors across the window (E1 tension dissolved); 2 double-the sweep artifacts fixed; role-explanation E7 dupe with WIR — trim intro restatement. Variance note: G emitted one open node where C emitted fact+open pair — n=1, shape-noise per 079d9736, watch in eval. |
| Nodes | D3, D6, D5, D2, A9, E7 (5a) · C4, A1, T5 (5b pending) | **5a DONE (probe H):** voice anchors unified under one derivation rule — test is derivation not importance; probe emitted user-only/anchor-only/neither across the 3-turn calibration scenario, felt and rejected the ceremony reflex on the action-derived node ('the derivation test isn't did-I-say-something but does-the-node-exist-because-of-a-said-thing'), and withheld user_raw_quote on low-substance 'good point' — symmetric restraint both directions. Floating-quote label dissolved (named 2×, defined 0×). Interpret/expand compressed 20→10 (inertness verdict RETRACTED as confounded by surface abstention, 0099ac43 — compression on D2/E7 grounds only; Bad/Good beat queued for canonical, Stop 9). Contract appendix revision_history fix shipped (2ba8dfb). Anatomy revise-bullet needs content_edits caveat when v-next.4 merges (A3 obligation, Actions stop). **5b DONE (probe I):** three-connections anchoring rewrite (92→30 lines; refs = rare surface-the-moment flag — probe flagged exactly 1/4 nodes, the correction with both moments, and restated the rule as 'the scene is itself part of what the node means'); living thought (cross-turn hunch emitted, delivered-contract cited verbatim); emotions array on the right node; question 1/4 (lookup-fact only — reasoned selectivity, healer backfills the rest; eval-watch, not a defect); work-state identifiers in situation. Fossil sweep: §7.6 labels + 'for v19+' out. Soft spot on record: question under-emits on principle/correction types. **Reopen risk from recall scout: trigger-register situation finding (14%→54% sit-lane entry, acfb8596) may sharpen the situation teaching — awaiting the scout's table.** |
| Edges | C3, E5, T1 (craft boxes only) | **CLOSED (probe J) — failed the recall cross-exam, redrafted, verified.** Six R-rows landed under Tom's honesty ruling (sometimes 9, sometimes 2 — never floors): reachability physics + old-graph obligation, why admission mechanics (120-180 chars, cue nouns), verb functions (corrects/supersedes demote, similar_to dedups, measured rescue verbs inline, related_to 0.2×). Probe J: 3 whys at 214/199/233 chars, 2/3 edges on catalog nodes, extends/grounds/parallels. Earlier craft-pass notes stand: First pass (craft boxes) found: MCP delegation measured-deliberate (04ff3d58), needs a guardrail test on the connect_to description's load-bearing strings (ship package); new→old bias judged physics not defect; why-bank untouched. Tom's challenge: the section was never cross-examined against recall-side evidence (recall-rank/recall-surface classes, 357 rows unread in the survival check). Opus scout dispatched 2026-08-21 for write-side implications — known seeds: rescuing-edge profile (well-described, extends-dominant), orphan-gold encode-side bridges (~53% endo misses unreachable read-side), edge-inhibition. Section verdict waits on its table. |
| Temporal | C6, D2, E7, A4, D3, R23 | **CLOSED (probe J).** Distilled ~290→~160 lines: Allen ceremony → 'Sequence between events' (sequencing verbs KEPT — measured top rescue paths; the scout reversed my full-cut plan); Ex1 deleted (A4 violator, hub rule survives as prose); Ex2 trimmed (fact-atom dupes + recap out) + first-person voice fixes ('My since-November gloss'); validity intervals got the in-place-vs-supersedes discriminator (probe J flagged the two-patterns ambiguity, honest judgment call). event_time doctrine untouched (the measured A1 win). |
| Actions | A2, A3, E1, E7, R7, R16, R17, R19, R21 | **DONE (probe K, 2026-08-21):** the v-next.4 merge — content_edits default + short-fields scope clause; Anatomy patch caveat (A3 obligation cleared); ghi789 → patch forms + ladder line; sweep example + BAD hub-only batch ported (R16 carrier; R7 carried-by-example — supersede lands on an old id; first grounded source_refs demo); `sweep:` close line; two-reads vs Speed dead conflict FIXED in the Speed line (E1 entry resolved); error names split (connect_to_bad_id for mis-copied ids, connect_to_unresolved for sibling titles — F2 of 9dc3efad) + source_refs silent-store told truthfully (was false reassurance; code gate → ship package); E7 fold — triplicate skip/zero-nodes merged into the Skip bullet, last third-person 'the assistant' → first person. Probe K (hydroponics scenario, off-corpus): 6/6 patches via content_edits, edge-line id targeted AND superseded, close line correct with all ids, 4 old-node edges, restraint decoy untouched, unprompted type re-audit. FINDINGS: UPDATE-append idiom on 2-3 patches (stale text left standing under an appended correction — one-clause fix awaiting Tom); loose supersedes on a patched-and-living hub (eval-watch). Harness caveat: probe collapsed read/write rounds by instruction — sequencing ungraded. |
| Speed + canonical example | A1, A2, C4, C5(partial), T6, R8, R13 | **9a DONE (probe L) + 9b DONE (probe M):** canonical carries real corrects (R8), question ×2 + named selectivity, ONE thought, refs-absent-deliberate, Bad interpret/expand beat; pair carries THE T6 why + question-on-mechanism; sweep example gained its lookup question (A3 consistency). Probe M baits: 1/5 question (landed on principle via genuine idiom — the REASON transferred over the class pattern, correct per the bullet's own wording), moment stayed question-free, pair minted with grounds+discriminating titles, ONE self-aware thought, ZERO refs (watch: over-restraint on a true moment candidate — package eval arbitrates vs 0.3% baseline), empty Review block (format watch for E-audits). C5: sweep situation carries branch-name handles — weakly carried-by-example; Required-fields prose stays primary carrier. 9d pre-decided: R1 stays mechanism-not-numbers; defaults-brevity↔be-expansive marked deliberate (E7). Remaining in zone: 9c prune-lexicon example (draft for Tom). |
| §7.6 identity examples | A4, A9, D5, E7, T5, D1 | **DONE (probe O, 2026-08-21):** Tom's set-coherence directive executed — the six entries audited as a SYSTEM, not entries. Dropped `locked: true` ×5 (VERIFIED silent no-op for encoder:sonnet at brain_remember.py:1152 — ghost-field class, third conviction; ship-package: log the strip). Dropped `trigger:` field ×5 (not a contract field; E7-duplicated situation-in-trigger-register since 5b) — best clauses folded into each situation. Zone intro: fossil D-codes from a dead instrument replaced with plain words (D5); refs-density class clause added ("rarity lives in the class, not the habit"); NEW derive-teaching (Tom's directive): "every example is material, not a menu — build the shape this knowledge actually needs" (D1's generative counterpart to mirror-the-patterns). Open-fields `trigger:` KEY kept (distinct use: names what set a reflex off). A2/A8 untouched (stamped). Probe O: zero locked, zero trigger, trigger-register situations, REAL trace-id refs dense on identity nodes, hot register both voices, and the composition — residue note pre-armed the lexicon upgrade for a second fold. Watches: derive-a-shape inconclusive (bait too rule-shaped; eval-watch), question 3/3-genuine-idiom pattern continues. |
| Closing + injected blocks + field summary | B2, C1, C3, C4, D5, E1, T4 | **DONE (probe P — first production-faithful arm, arc block INCLUDED; probes A-O omitted it).** Closure paragraph rewrote: dead jargon ("judgment half") out, the actual injected sequence named (field list → `## Arc` → `## Review` → finishing rule), and the closing ORDER fixed — `sweep:` line, then Arc, then Review, then DONE. Probe P emitted exactly that order (earlier probes placed the sweep line randomly — the closure never named it). Anatomy: content line drops "revision history lives in trace events" (Tom); question line reframed — NOT "the asker's own words" but a tiny generalization above any single asking, keeping discriminating words (Tom; brackets R13's generic-paraphrase failure 035ac88f and the healer's natural-asking principle 218151a0). **Journal ride REJECTED after check (Tom's caution was right):** sweep-as-Review-note parses fine (open tag vocabulary) but (a) `render_journal_notes_prefix` feeds notes back as the next runs' continuity bounded to the last K note-bearing runs — a note every run floods it and ages out real residue, and (b) it contradicts the block's own "changes are recorded automatically; don't restate them". Clean ride kept: sweep DOUBT is a Review note; the sweep ledger stays a plain line. Shared-infra untouched (decorate_system serves 3 S2 encoders + S1E; S1E's only opt-in is arc=True). Measurability: the sweep line rides the run's final text, which the eval captures — no journal pollution needed. **Field summary = CONTRACT-side, handed to the rename stream 02e4b5a9** (omits question/thought/emotions/event_time/source_refs; advertises superseded emotion+emotion_label, dead personal/personal_context, no-op locked; stale content line). Watches: refs zero outside identity register 3 probes running; probe P left an ×3 open residue item untouched where the new mechanism arguably bears on it (soft miss, eval-watch). |
| MCP descriptions (brain_batch, connect_to, absorb) | — | — |
| Assembly layer (preamble, banner, [associated] stubs) | B3, C5, T6 | Stubs SHIPPED to main by the stubs stream (0b22bda + f613b64), flag OFF. Ship-package obligations from its handoff (full record: their node f5027961): (1) **eval-arm wiring** — encoder_prompt_ab.py/reassembly call build_node_catalog directly, never pass associated_ids → flag-on changes nothing in A/B until the arm plumbs retrieval (API: build_node_catalog(..., associated_ids), encode._associated_stub_ids(...), K=5); (2) **[associated] gloss line is REQUIRED before any flag-on eval** — a loud s1e_associated_stubs_untaught error fires when flag-on + active prompt lacks the tag; the gloss is now in the WIP; (3) replays against future-dated brains must pin as_of themselves (retrieval passes none, deliberately — three alternatives verified broken). Banner v3 interview-clean, wiring into _build_user_content still to code (flag-gated). |
