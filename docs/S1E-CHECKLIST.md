# S1E Prompt Checklist — the boxes every revision must check

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

## E. Global audits & process gates

- **E1. Contradiction audit (T4).** Enumerate every in-prompt tension; stamp
  each `DEAD-CONFLICT` (fix: one side wins, by position) or `OPTIONALITY`
  (keep: mark the judgment it grants). Current known: Actions-two-reads vs
  Speed-"everything I need"; anchor_raw_quote required-vs-selective;
  residue-deferral vs window-slides; expansive vs 2-rounds.
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
  reverse-pass, 2026-08-21)
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
| Canonical 5-node batch + catalog excerpt | Speed | situation+reasoning ×5; voice symmetry w/ selective anchor_raw_quote; cross-redundancy (27:12); event_time; id-copy from grounded excerpt + sibling title beside; corrects placeholder; near-twin id-pick; open keys (correction_pattern, emotional_context); N=3 earned principle; 5-type spread | `question` (C4 — one line/node); ONE `thought` (A1 revival + selectivity); `source_refs` with real trace id; a differentiation beat in one reasoning (T6); work-state handles (C5) | Highest-attention asset (B5) — every addition leaks on all dimensions (A4); char growth (D2); keep additions off-corpus (A7) |
| Detail+meaning pair | Speed | Two-register law as tool-call; grounds edge whose why names TWO retrieval surfaces (the closest thing to a T6 carrier today); anchor_raw_quote on a mechanism | Name the separability logic explicitly in the why (make it THE T6 carrier); question field | Duplicates the opener's E=mc² frame — deliberate reinforcement or redundancy, decide once |
| revise_batch light/light/FULL (ghi789) | Speed | Light→FULL progression (T1 via comments); field-walk on stale value; reason discipline | `content_edits` patch form (or cede revise-teaching to the sweep example); title/type re-audit on revise (encode-write #26, untaught); preservation demo (currently prose-only) | Overlaps v-next.4's sweep example — TWO revise-teaching assets after activation; consolidate or split cleanly |
| §7.6 A6/A7/A4/A2/A3/A8 | §7.6 | Identity/hot-register encoding; anchor-voice depth; locked + trigger usage; correction at register; agent-as-other-side (A8); source_refs shape (placeholders) | **A voice-disagreement example — me holding my ground with evidence and encoding the contradiction as signal (new D3): none of the six shows it; every §7.6 example is me being corrected or seeing, never me disagreeing** | Big real estate; locked generosity needs its disclaimer kept; placeholder discipline must survive edits (A5) |
| v-next.4 sweep example (BAD hub-only + full sweep) | candidate | Sweep discipline (D11); content_edits patch form; labeled BAD contrast (T1 ✓); first-person falsifying evidence (D3/D12); supersedes lineage; edge-visible neighbor walk | Fix the wrong error name (`connect_to_bad_id`) in the adjacent prose while landing it; differentiation beat | ~100 added lines (D2); territory overlap with ghi789 (above) |
| MCP description examples (brain_batch, connect_to, absorb) | MCP | connect_to resolution scopes; sibling-vs-catalog forms; forward-reference example; empty-why anti-pattern; vocabulary + never-generic ban | Correct error-name semantics live here too (mechanics = MCP home, C2); absorb's content-destructive warning held — keep | Shared across ALL callers (C3) — no encoder-specific teaching; changes need the 8-step MCP eval gate |

## Coverage matrix (fills during the walk)

| Section | Boxes applied | Status notes |
|---|---|---|
| Opener | B1, D3, D6, D14, A9, E7 (+B6/B7 discovered) | **DRAFTED** (scratchpad opener_final.txt, probes A/B/C/D): B1 restored via named 'Two registers, every exchange' lead; D3 carried by 7-word equality line at probe-#2 attention + disagreement in vigilance list; D14 'integration, not recording' contrast-foil (write-drive verified kept); A9 fake 95/70 → true mechanism; E7 P2/P3 seam merged. +168 chars vs current. Deferred out: two-registers loop-entry → gate sentence (Speed stop, B6); evidence-lean sentence → flavor 4 (Reading stop). Residual, deliberate: verbatim-priming now arrives downstream, not in opener. |
| What I Receive (+ assembly: banner, catalog, timeline) | B3, C5, E1/E7, stale-gloss class, T5 | **IN PROGRESS.** Caught: line-53 `encoded="true"` gloss contradicts line 62 (stale pre-B4 text — fix in draft). Queued draft edits: line-53 fix; `<attention>` gloss line in channel list; C5 pointer clause on the actions bullet. Banner: v1 text superseded (staleness-biased, Tom no-ship); v2 probe-failed (3:1 tilt); **v3 passed interview tier** (Probe F: symmetric read, no manufactured revisions; residual: additive-info-misread-as-supersedes → wide-sweep watch dimension). Stubs: ruled IN (subconscious framing); placement leaning [associated]-tag-in-catalog, Tom's nod pending. Both assembly changes flag-gated, behavioral gates (staleness set + wide sweep) still ahead. |
| Scout | — | — |
| Reading the conversation | — | — |
| Nodes | — | — |
| Edges | — | — |
| Temporal | — | — |
| Actions | — | — |
| Speed + canonical example | — | — |
| §7.6 identity examples | — | — |
| Closing + injected blocks + field summary | — | — |
| MCP descriptions (brain_batch, connect_to, absorb) | — | — |
| Assembly layer (preamble, banner, beyond-catalog stubs) | — | — |
