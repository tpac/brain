# S1E Prompt Checklist — the boxes every revision must check

## Walk state — WALK COMPLETE (stops 1-11, probes A-P); next: 4 rulings → defects → weave → contract → audits → gate (2026-08-22) ◀ ACTIVE ARC

**Where this stands.** The section-by-section walk is finished. What remains is not more
walking — it is a fix list, one contract change, two audits, and the gate. Draft is
97,944 chars; NOTHING is registered. Diff base is **v38** (the rename stream's mechanical
name-swap); our candidate registers as **v39**.

**Order of work, and why:**
1. **Four rulings from Tom** — drafts written and shown 2026-08-22, awaiting his word:
   (a) the live-question pilot (does one named vertical teaching ride in v-next.5?);
   (b) canonical excerpt enrichment (2 catalog lines so the zero-edge nodes gain honest
   targets); (c) the sixth canonical node (action-derived, no voice fields — closes the
   derivation rule's negative case, the `<actions>` encode-side demo, and C5 at once);
   (d) RULED: retire/archive is a real gap, its own thread, not a v-next.5 item (id:9eaed612).
   Full draft text in the working set; rationale in id:a6c13aad.
2. **The 8 verified defects** (E2/E3 section below) — mechanical once (b)/(c) are ruled,
   since both touch the canonical.
3. **The weave plan** (E2/E3 section) — 5 moves, no new assets. The `similar_to` connect op
   between the planted ring-buffer twins closes three gaps at once.
4. **contract.py field summary — REGISTRATION BLOCKER, not cleanup** (id:68fd6e05). Probes
   emitted `locked` and `encoding_source` — fields NO example sets — purely on the contract
   list's authority. contract.py is free (rename stream merged out).
5. **E1 contradiction audit + E8 format census** — unrun, and they must run over the
   SETTLED draft, not before the weave.
6. **Ship gate** — the runbook below is complete: invocation, check-overrides pre-step,
   expected one-time cache miss, migrate-on-open pre-registered as first suspect, and both
   arm-integrity checks.

**Deferred by design, with owners named:** the vertical axis beyond the pilot (six more
teachings shaped like Emerging patterns — evidence in id:56631bce); the journal→graph
route (id:7b2d67e8, constrained by ruling 68063517: residue into the Frame is *hurtful*);
retire/archive (id:9eaed612).

**Probe fidelity note:** probes A-O were built WITHOUT the `## Arc` block (production S1E binds
`arc=True` — encode.py `_journal`). Probe P is the first production-faithful arm; rebuild future
arms with field summary → arc → review → closure, in that order.

**Read first:** handoff node id:477ddcc9; the rulings chain runs 7c87589c (fundamentals) →
13f72658 (three connections) → 4bb5b1e8 (edge honesty + prune-lexicon example).

The walk: section-by-section co-review with Tom (he rules, I hold the learnings), every
substantive edit probe-verified by stateless Sonnets before it lands. Stops 1-9 closed
(probes A-N); draft = eval/candidate_prompts/s1e_vnext5_wip.md, 98.1K vs v37's 86.9K —
over v37 since Stop 8's sweep example (Tom-ruled earned size; package eval is the
backstop), ~30 measured teachings added, NOT registered.

**Locked:** target function; banner v1 never ships (position carries, text was
staleness-biased); [associated]-in-catalog (not a separate section); edge honesty over
floors; Allen sequencing verbs KEPT (measured top rescue paths — only the ceremony cut).
**Open:** stops 9b-11 (9a closed, probe L; UPDATE-append clause landed via micro-probe);
E1/E2/E8 global audits (E1's two-reads entry resolved at Stop 8); ship gate (DORMANT →
package eval → Tom activates → sync + SEED_PROMPTS_VERSION); ship-package code items: the
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
**Consequences for this walk:** (1) the WIP's name-pass is now DUE — swap the two names,
then SHRINK the voice-equality gloss, since the names finally carry the frame; (2) our
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

**THE GATE RUNBOOK** (confirmed unchanged by the eval stream after their Step 7 reshape —
re-anchor here, not to any older command):

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
config half. Our gate should differ by construction (v38 vs v39 are genuinely different
text); **if the fingerprints match, stop** — the override didn't take, and the config half
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
7. **Ship gate:** register DORMANT → eval (run-44 staleness set + longmem
   sweep, multiple reps — no single-run conclusions) → Tom activates →
   sync-prompts.

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

| move | closes | into |
|---|---|---|
| add a `connect` op drawing `similar_to` between the two planted ring-buffer twins | gap 14 + standalone-`connect` + the unresolved near-twin | canonical batch |
| make one sweep patch a `situation` patch; give one patched node its truth condition | gaps 4, 5 | sweep example |
| name what the window did NOT earn a node for, in one line | gap 1 | canonical framing |
| fix D1–D8 | coherence | in place |
| enrich the catalog excerpt so honest edges exist for the two zero-edge nodes | gap 2 (without manufacturing edges — Tom's honesty ruling holds) | canonical excerpt |

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
