# Production laid over V3.4 — every addition considered, and the verdict on each

A reviewable candidate, authored 2026-09-12. Nothing is merged, activated, deployed
or evaluated; zero model calls were made.

**The question.** The deployed production S1E prompt (the `template` string in
`eval/fixtures/s1e_production_comparison_2026-09-11/effective_s1e.json`, 111,323
chars) was read end to end against frozen V3.4
(`eval/fixtures/s1e_guide_v3_4_2026-09-12/template.md`, 106,775 chars, plus
`strategy.md`, `gist.md`, `closure.md`). Every production section, paragraph,
Bad/Good pair, worked example, checklist bullet and closing sentence was located
in V3.4 as **present**, **reduced** or **absent**. Every reduced or absent one got
a decision: ADD, EDIT or REJECT.

**Counts: 1 ADD, 7 EDIT, 26 REJECT.**

**Method constraint.** Seven of the eight accepted changes are EDITs, not ADDs —
E16 ("new teaching rides an existing explanation"; the first v-next.7 draft's
+7,698 chars were 5,600 of new prose wrapped around 100–220-char levers) and Tom's
rule, *"A common mistake is fix a single problem through adding. The magic is in
complexing the shapes and weaving smartly"* (id:3b18648b). No new bullet, no new
section, no new example asset. The Mira episode is untouched; every V3.3/V3.4
weave survives (quote moves with the fact; question on both remembers; the Skip
sentence; the refused-op sentence in the strategy; the hunch thought on the plan
revise).

**Authoring record:** `author.py` — each change an exact-once `replace()` on the
frozen V3.4 parts, failing loudly on any other count. `static_checks.py` is V3.4's
checker with `PARENT` repointed at the V3.4 fixture. Both run clean.

**Size and hedge census**

```
template  106,775 -> 107,968 chars  (+1,193, +1.1%)   production 111,323 (candidate is 3,355 under)
gist        5,703 ->   5,786 chars  (+83)
strategy    1,348  unchanged        closure 379  unchanged
hedge clauses (V3.4's own regex): V3.4 23 -> V3.5 23   (production 4)
A10 revise-op field grid: unchanged — no revise op added or edited (deliberate zero, E14)
```

---

## Where a passage is placed, and why position is part of the decision

Position is a measured lever here, not a preference. `7bdb1c25`: a 1.1K-char
reminder at the payload's recency slot moved surface coverage from 50% to 80% on
an **unchanged** template, while four template rewrites the same week moved it by
about nothing (`2e4da492`). E21 states it as a law; B1/B4/B5 state the
within-template version. So each accepted change is placed by what it has to do:

- **X5** (the `thought` field) goes in `gist.md`, the last thing read before the
  conversation — the highest-attention slot available, and the one the thought-gap
  analysis (id:3d54a75c item 3) named and V3.4 did not take.
- **X3, X6** ride the gate region — the two sentences around "**New AND useful** is
  the capture gate", which both Stop-1 probes independently ranked as the single
  most behaviour-controlling sentence in the document (B6), and B7 says additions
  ride measured carriers rather than becoming standalone clauses.
- **A1** opens `## Actions` because ownership has to land at the write moment
  (E19: a surface the agent does not experience as its own is one it will not
  repair), not only in the opener 250 lines earlier.
- **X1, X2, X4, X7** sit with the concern they belong to — the fields section, the
  detail-and-meaning pair, the Nadia reading, the timeline sample — because each is
  a rule whose example is right there. None of them is a loop-entering teaching, so
  none of them earns gate-region real estate.

---

## ADD — 1

### A1 · "I am the source — the graph's shape this turn is my call."

| | |
|---|---|
| **Production location** | `## Actions`, first line: *"I am the source — the graph's shape this turn is my call."* |
| **V3.4 status** | ABSENT — V3.4's `## Actions` opens on mechanics ("The catalog is a view, not the whole brain.") |
| **Placement in V3.4** | Prepended to the first sentence of `## Actions` (~21% depth), before the two reads |
| **Evidence** | E19 ownership audit — conviction `92519f2e`: the encoder's own uncued account of the surface it never repaired was that it *"reads like description of the target node rather than my own asserted text"*. B1: a named position changes the schema of attention. E7: marked **deliberate reinforcement** of the opener's ownership register, not a layering scar — the opener establishes ownership of the memory, this establishes ownership of the write. |
| **Risk** | The one change with no measured carrier: ownership language is exactly the register that produced production's overclaiming (V3.3 blind review: production 5/36 on evidence/ownership/scope, "every pack but one rated it weak"). The sentence claims authority over *the graph's shape*, not over *what the evidence establishes*, and V3.4's scope discipline sits 6 lines below it — but an eval must watch the ownership/scope dimension specifically. |

```
old: The catalog is a view, not the whole brain. Read what the decision lacks:

new: I am the source — the graph's shape this turn is my call. The catalog is a view, not the whole brain. Read what the decision lacks:
```

---

## EDIT — 7

### X1 · Cross-redundancy for facts

| | |
|---|---|
| **Production location** | `## Cadence and worked examples`, canonical roll-call: *"specific numbers, names, and verbatim phrases appear in BOTH the raw quote AND the title/content — cross-redundancy so the fact is findable by ANY retrieval path"*, and the demonstration bullet *"**Numbers cross-redundant**: '27:12' / '27 minutes and 12 seconds' appears in title, content, AND their_raw_quote — three retrieval paths to the same fact"* |
| **V3.4 status** | ABSENT — `grep -ci redundan` on V3.4's template returns **0**; production carries it twice, both in its highest-attention asset |
| **Placement in V3.4** | Folded into the five-surfaces sentence, `### Fields and the claim they serve` (~7% depth) — the sentence that already tells the encoder what each surface is for |
| **Evidence** | D8 (cross-redundancy for facts, genealogy #51, v16 rows) is a standing law with no carrier in V3.4. E11 rationale-attachment: V3.4 lists five surfaces with no consequence attached to any — "an option with a named consequence is TAUGHT; an option in a list is MENTIONED". The blind review is consistent on the one cell V3.4 ran in: production leads on dimension 1, facts and concrete detail, 9 to 8 (the V3.3 cell had no V3.4 arm). The reviewer-named V3.4 defect on `69fee5aa` was a numeric revise that *left the old count live in `situation`, `question` and two edge descriptions* — the same fact-across-surfaces discipline, seen from the revise side. |
| **Risk** | D13: the prompt already pushes richness with a thin separability counterweight, and cross-redundancy is a richness rule. Bounded here to the case where the number/name/phrase **is** the claim's value, and the "keep each about THIS claim" clause it sits beside is the counterweight. Second risk: production's version prescribes a quote on every fact node; this version says "where the node carries one", so it cannot contradict V3.4's derivation test for voice fields. |

```
old: Title, content, situation, question and edge descriptions are five retrieval surfaces. Fill every surface the node honestly carries; keep each about THIS claim.

new: Title, content, situation, question and edge descriptions are five retrieval surfaces, each scored on its own. Fill every surface the node honestly carries; keep each about THIS claim. When a number, a name or an exact phrase IS the claim's value, it rides the title and the content, and the quote too where the node carries one — three paths to one fact.
```

### X2 · The generative half of detail-and-meaning

| | |
|---|---|
| **Production location** | `### Detail and meaning — same topic, two nodes`, closing: *"Detail without meaning is trivia that never transfers; meaning without detail is a slogan no query can land on. The pair is the unit… For abstract types — rule, lesson, insight, correction — the pair is how they SURVIVE: alone they retrieve at roughly half the rate of concrete types, and the concrete twin lends its lexical surface through the edge."* |
| **V3.4 status** | REDUCED, and inverted: V3.4 kept only the restraint half — *"A plain fact needs no invented principle."* |
| **Placement in V3.4** | Folded into that same sentence, immediately after the worked detail/meaning pair (~60% depth) — the example that demonstrates it is directly above |
| **Evidence** | **D1** is the law this is the textbook case of: *"Restraint lands ~100% and over-generalizes; generative lands ~20%. Every restraint rule needs an equally prominent generative counterpart; read any generative regression as a restraint rule added somewhere."* V3.4 replaced a generative rule with a restraint rule in place and the predicted regression is measured: `S1E-V3-4-RESULTS` — *"the debugging synthetic yields 2 nodes in two of three repeats where production yields 5–6, because V3.4 folds the lesson and the mechanism into the bug node instead of minting them"*, and the coverage-target view shows the transferable pattern ("silent exception swallowing hides bugs") written OWN by production in 2 of 3 repeats and FOLDED by both candidate arms in 6 of 6. A blind reviewer named the same thing: *"the whole debugging arc in one node with the fix packed into `situation`"*. |
| **Risk** | This is the passage nearest to production's over-minting. Three guards are built into the wording: it routes through the **retrieval-divergence** test ("a future reader would ask for them separately") rather than a standing instruction to abstract; it carries **scope** explicitly ("each at the scope its own evidence supports"); and it deliberately does **not** import production's *"and the principle or concept each one points to"* (see R6), which is the clause the surplus traces to. Production's *"roughly half the rate"* figure is also not imported — A9 (no eval behind it; the opener's invented 95%/70% statistic is the precedent). Residual risk: on assistant-heavy Q&A this could still pull toward minting the assistant's advice; the coverage-target view is the readout that would catch it. |

```
old: A plain fact needs no invented principle. When an abstraction has a concrete carrier, linking it supplies lexical reach.

new: A plain fact needs no invented principle. But detail without its meaning is trivia that never transfers, and meaning without its detail is a slogan no query lands on: where one exchange carries both and a future reader would ask for them separately, both earn a node, each at the scope its own evidence supports. When an abstraction has a concrete carrier, linking it supplies lexical reach.
```

### X3 · The generative half of "be expansive" (and "I don't ration")

| | |
|---|---|
| **Production location** | `## Cadence and worked examples`: *"**Be expansive here.** … if this turn has ten encoding-worthy atoms, one batch call carries ten nodes, not two… This is measured, not taste: encode-time field population outperforms the best runtime reranking — and a HALF-populated node is worse than it looks, because it free-rides on title match into pools it can't win."* Plus `## Actions`, remember bullet: *"Most turns produce several… I don't ration."* |
| **V3.4 status** | REDUCED to its restraint tail only — *"Richness belongs in focused nodes with useful fields and honest edges, not additional call ceremony."* |
| **Placement in V3.4** | Folded into that sentence, at the end of the traps paragraph in `## Actions` (~22% depth) — inside the gate region B6 measured as the prompt's centre of gravity |
| **Evidence** | D1 again, and the standing measurement: V3.4 is *"still the smallest memory in every cell (159 fresh nodes against 182 and 211; 139 regression against 148)"* and the V3.3 transfer cell counted *"windows creating ≤2 nodes: V3.3 15 of 42, production 6"* (id:29cdae0e). B7: the clause rides the carrier sentence rather than becoming a standalone bolt-on, which measurably never cracks the attention top-5. |
| **Risk** | Production's *"I don't ration"* and the bolded *"Be expansive"* framing are deliberately **not** imported — they are the register behind production's 211-node surplus of the assistant's own advice. The imported clause is bounded by V3.4's own phrase, "as many nodes as the window earned". The second clause (half-populated nodes) is a field-fill argument, not a node-count argument, so it pushes on depth rather than volume. Risk remains that any generative push on this sentence raises production-style volume on assistant-heavy Q&A. |

```
old: Catch these by their cost to the future reader. Richness belongs in focused nodes with useful fields and honest edges, not additional call ceremony.

new: Catch these by their cost to the future reader. Richness belongs in focused nodes with useful fields and honest edges, not additional call ceremony: one batch carries as many nodes as the window earned rather than the two that feel tidy, and a half-populated node free-rides on a title match into pools it cannot win.
```

### X4 · Temporal authority, stated where the example already enacts it

| | |
|---|---|
| **Production location** | `### Worked example — temporal authority across the breadth`: *"**The other side's explicit wording is the date authority: my own `<me>`-turn paraphrase never overrides what they said in an `<other>` turn**"* |
| **V3.4 status** | ABSENT as a rule — V3.4's Nadia reading enacts it ("my November claim has no source") without ever stating it |
| **Placement in V3.4** | Folded into the Nadia reading sentence, `### Worked example — temporal authority` (~14% depth) — the section is literally named after the rule it no longer states |
| **Evidence** | D10 carries it as a standing law (*"the other side's explicit wording is the authority"*); D3 records it as voice equality's **one scoped exception** — *"that's paraphrase-vs-source, not voice rank"*. A1: a behavioural ask with an example but no rule is half-taught; here the section header promises a rule the body never gives. |
| **Risk** | The real hazard is over-reach: read as "their later words outrank their earlier words", it would worsen the finding the V3.4 cell produced on its own — the tennis item, where *"all nine sequences of all three arms read July as a correction of March and kept one value; no arm held two dated values"*. The wording is scoped to **my paraphrase of what they experienced**, which cannot reach a two-statements-from-them case. Worth checking in review that the scoping holds under a cold read. |

```
old: Five dates from Nadia; one unsupported gloss from me. Her January date anchors being off her feet

new: Five dates from Nadia; one unsupported gloss from me — my own paraphrase of what she experienced never outranks her own wording for it. Her January date anchors being off her feet
```

### X5 · `thought` enters the gist's `new`-line field roll-call

| | |
|---|---|
| **Production location** | `## Cadence and worked examples`, finishing roll-call: *"the selective fields (`correction_pattern`, `event_time`, `question`, `thought`, the `emotion` pair) appear where they earn their place"* — production's only late-position field roll-call naming `thought` |
| **V3.4 status** | ABSENT from every roll-call. V3.4's `gist.md` lists the fields a `new` line must carry — situation, reasoning, question, edges, `event_time`, `source_refs` — and `thought` is not among them. (The `targets` roll-call does name "`thought` when present", so the revise side is already covered.) |
| **Placement in V3.4** | `gist.md`, end of the "Then the initial write" paragraph — **the recency slot**, the last instruction read before the conversation |
| **Evidence** | id:3d54a75c, revised after measurement, ranks three drivers of the 23-vs-2 thought gap and its surviving live items are (2) example shape and (3) **salience and position** — *"production has thought as its own subsection … plus two late mentions; V3.3 folds it into Types at 10% depth, and its strategy/closure never name it (position is the lever, id:7bdb1c25 lineage)"*. V3.4 took the example-shape half (W5) and it worked on shape — *"every V3.4 thought is a hunch or a connection"* — but not on count (2/6/1 against production's 14/14/9). E21/B3: position before prose. E23/A1: a field taught in prose and carried by no roll-call is written at ~0% (conviction `2e4da492`: `confidence:` taught in two prose places, 0 uses in 10 runs). This is the complementary lever to W5, not a repeat of it. |
| **Risk** | A field roll-call invites filling, and the semantic-fidelity challenge is explicit that *"Thought is optional; no count or nonempty-field target earns quality credit"*. The clause is gated ("when I have a hunch or a connection of my own worth keeping"), matching V3.4's own "most nodes need none". The failure to watch for is thin, obvious thoughts appearing on routine fact nodes — which the V3.4 template already names as noise 10% in, but the gist is the louder position. |

```
old: …`event_time` resolved to ISO against the conversation's date, and `source_refs` — the 1–3 `trace=` ids of the turns that generated it — when the moment is part of the meaning.

new: …`event_time` resolved to ISO against the conversation's date, `source_refs` — the 1–3 `trace=` ids of the turns that generated it — when the moment is part of the meaning, and a `thought` when I have a hunch or a connection of my own worth keeping beside it.
```

### X6 · The field name returns to the gate-region sentence

| | |
|---|---|
| **Production location** | `## Actions`: *"And when I have a read on what something means, I put it in a `thought` — my own take is part of the capture, not garnish. Details and thought, not just conclusions."* |
| **V3.4 status** | REDUCED — V3.4 keeps *"and my own read on what something means is part of the capture, not garnish"* and drops the field name |
| **Placement in V3.4** | Appended to that same sentence, inside the capture-gate paragraph in `## Actions` (~22% depth) |
| **Evidence** | **T8** — *"Verbs in the shape of the functions… an English synonym for the act breaks the lookup exactly there — it teaches the intent and hides the door"*, with conviction `f358bba7`: a surface the prompt named by an English verb, whose op existed in the toolset, was written 0 times in 24 runs. E18 reachability: name the surface with the field that writes it. This is a 24-character change in the highest-attention region of the prompt. |
| **Risk** | Lowest of the eight. It pairs with X5 (roll-call) and W5 (example shape), so if the three together over-fire, the thought count rises on nodes that do not carry a real read. The three are separable in an A/B only if X5 and X6 are varied together against V3.4, which is how they should be read: one lane, two placements. |

```
old: …and my own read on what something means is part of the capture, not garnish.

new: …and my own read on what something means is part of the capture, not garnish — it rides in `thought`.
```

### X7 · An `encoded="true"` turn returns to the timeline sample

| | |
|---|---|
| **Production location** | `## What I Receive`, timeline sample — the `<turn n="3" age="2d ago" encoded="true">` block with its `<provenance>encoded(me, turn 3)…` line and `<actions>trimmed — 2 action(s) recorded…</actions>` stub |
| **V3.4 status** | ABSENT from the depiction (the sample shows one `encoded="false"` turn), while the **rule** about covered turns sits two paragraphs below it |
| **Placement in V3.4** | Inserted before the existing turn 5 in the sample block, `## What I Receive` (~2% depth) — lived order, newest last |
| **Evidence** | **E12 before-state depiction audit**: *"an example teaches recognition only if its INPUT shows the thing to be recognized… Where a lesson is 'notice X, then act', X must be visible in the depicted input."* Its conviction is the same class — zero catalog excerpts rendered a `situation:` line while assembly renders one, so the encoder had no template for a stale situation. Here V3.4 asserts three behaviours about `encoded="true"` turns (text remains, actions become a trimmed stub, reread for cross-turn patterns not fresh atoms) and depicts none of them. C6 also applies: the `trimmed — N action(s)` stub and the `encoded(me, turn N)` provenance form are invented notations, opaque by definition, and glossing without depicting is the half measure. |
| **Risk** | +394 chars in the orientation block, which is the lowest-attention region (B4) and already competes with the reader learning the input shape (the checklist's own example-inventory risk note for this asset). D2: char growth is itself a defect to justify. The justification is E12's law rather than a measured instance — **this is the weakest-evidenced of the eight**, and the first I would cut if the reviewer wants a smaller diff. |

```
old: <turn n="5" age="20m ago" encoded="false">

new: <turn n="3" age="2d ago" encoded="true">
       <other trace="e5f60b2d">let's check the write path too…</other>
       <provenance>encoded(me, turn 3): "batch commit gate" id:7f3ea1c9</provenance>
       <me trace="97b8d4f2">The batch gate covers it — commit_unless_batched on every writer…</me>
       <actions>trimmed — 2 action(s) recorded on this turn; I already read them in a previous run</actions>
     </turn>

     <turn n="5" age="20m ago" encoded="false">
```

---

## REJECT — 26

Grouped by the reason that governs each.

### Rejected because the machinery is gone (3)

| # | Production location / first words | V3.4 status | Rationale and evidence | Risk of rejecting |
|---|---|---|---|---|
| R1 | `## What I Receive`, bullet — *"**`<scout_legend>`** — sits just before the timeline and explains the `<scout_notes>` inside it…"* | ABSENT | The scout muster is **retired in code**. `servers/trace_contract.py:107,110` carry `"scout_input"` / `"scout_findings"` marked *"retired with the scout muster; history rows read through it"*; `servers/scales/s1/surface_contract.py:930` records *"since-removed scout muster"*. Node id:5d0a6a8d dates the cut: step 1 shipped to main as `42c6ebf` on 2026-09-06, and *"Production has run zero scouts since 42c6ebf, so the text was dead on both eval arms"*; the prompt-text half (step 3) landed on this branch as `c5987c7`. | None. The finding runs the other way: **the deployed production prompt is still teaching a dead channel** — ~3.4K chars of it. Worth surfacing separately from this overlay. |
| R2 | `## Scout` — the whole section: *"One scout worked this window in parallel with my own read…"*, `### Reading posture`, `### How the handoff works` (the `facts: handle [role] — detail (extras)` grammar) | ABSENT | Same evidence as R1, plus Tom's own ruling (id:7e2d7178) that the prompt-text removal ships with this branch in one eval. | None. |
| R3 | `## Actions` → `### My defaults vs. this job`, bullet — *"**Scout-deference** — the reflex to treat pre-digested input as the map…"* | ABSENT | Same. A trap about deferring to an input that no longer arrives. | None. |

### Rejected because V3.4 already carries the substance (10)

| # | Production location / first words | V3.4 status | Rationale and evidence | Risk of rejecting |
|---|---|---|---|---|
| R4 | `## Actions` — *"if a conversation has 10 meaningful exchanges and I write 0–1 nodes, I'm leaving value on the table"* | REDUCED — present | V3.4 carries it, moved into the Skip sentence by the W3 weave: *"ten exchanges that leave no node have almost always dropped some"*. Re-importing the production wording is E7 accretion (unmarked repetition is accretion by default). The companion clause — *"The atomization test prevents fragmentation… it never means 'encode less'"* — is carried by V3.4's atomization paragraph: *"Separate memories when future queries would find them differently, not to hit a length or node count… a sprawling rich node absorbs its neighbors' retrieval lives… 'Fewer is cleaner' is not a retrieval argument."* | The measured coverage gap is real, but it is a **folding** gap, not a skipping gap (the pattern is FOLDED, not ABSENT, in 6 of 6 candidate repeats). X2 is aimed at the fold; restating the skip guard a third time would not reach it. |
| R5 | `## Nodes` → the headed subsection **`thought` — my own read, alive and delivered**, its Bad/Good pair, and *"A thin or obvious thought is noise; a live one is my value as a thinking thing."* | REDUCED — folded into `### Types, thought and open fields` | V3.4 keeps the Good example **verbatim**, the Bad reworded, "most nodes need none", "a thin or obvious thought is noise", the delivered claim, the content/reasoning/thought distinction, the updating-is-maintenance clause, and the value clause in its own words (*"it is what makes me more than a record"*). What production has and V3.4 does not is the **header and the depth** — and id:3d54a75c measured that lane: *"V3.2→V3.3 restored the template's thought paragraph and moved the count 3→2: the template text did nothing"*. The live drivers are example shape (taken by W5, and it moved the shape) and position (taken by X5/X6, at a **stronger** slot than production's own). | Restoring the header is the one production asset this overlay knowingly leaves on the table. If X5/X6 do not move the count, promoting the subsection is the next test — but it would be the third intervention on one lane, and B1's gain would be confounded with them. |
| R6 | `## Actions` — *"I need to recover what was thought, not just what was decided"* plus *"The Borges quote I cited in an essay, the definition I explained, the mechanism I diagnosed — these earn nodes."* | REDUCED | V3.4 L264 carries the claim: *"Preserve … supported meaning, including my research, essays, explanations and diagnoses. A passive partner does not make my thinking worthless"*, and X6 restores the field name to that same sentence. The three concrete instances are production's A6-style concrete nouns; importing them would add ~180 chars of illustration to a sentence that already lists four categories. | Low. The generative claim survives; only the illustration is dropped. |
| R7 | `## Actions` → `### My defaults vs. this job` — the six named instincts as bullets (Default brevity, Compression, Paraphrase, Skip-when-unsure, Scout-deference, Single-voice gating) | REDUCED to one sentence listing seven traps | V3.4's sentence covers five of the six (Scout-deference is dead, R3) and adds two V3.3-measured traps. E16: fold, do not restore blocks. D2: *"More prompt = more compliance, less depth… one principle beats an enumerated list"*. The one substantive loss — the generative "here, I'm expansive" half — is taken by X3, in that same sentence. | E11 notes the cost honestly: production attaches a consequence to each of six instincts; V3.4 attaches one shared consequence ("their cost to the future reader") to all seven. If the traps under-fire, the fix is per-trap consequences, not per-trap bullets. |
| R8 | `## Edges` — the fourth Bad: *"Bad: `{relation: "related", why: ""}` — invisible."* | ABSENT (V3.4 keeps Bad×3, Good×4) | C1/C3: the empty-why anti-pattern and the never-generic relation ban live in the `connect_to` **tool description** — the layer that binds — and V3.4 explicitly delegates there (*"The tool description owns the vocabulary, forbidden generic relation and parameter shape"*). C2: tool-mechanics rules work best at the tool. Duplicating it in the template is accretion at the weaker layer. | Low, and A8-positive: V3.4's bank still carries three Bads against four Goods, preserving the quality spread. |
| R9 | `## Nodes` → Anatomy — *"writing into only two of them is how most never-recalled nodes died"* | ABSENT | The consequence E11 asks for is real, but this particular claim is an unsourced population statistic. A9's conviction is exactly this failure — the opener's *"surfaces at 95% where a three-topic node musters 70%"* traced to a v1 illustration with no eval behind it, *"modeling exactly the overclaim habit the ledger convicts the encoder of"*. X1 attaches a **mechanism** consequence to the same sentence instead of a number. | None; the E11 gap is closed by X1 without the statistic. |
| R10 | `### Detail and meaning` — *"alone they retrieve at roughly half the rate of concrete types"* | ABSENT | Same as R9: no measurement I can trace. X2 restores the pair rule and the lexical-reach mechanism without the rate. | None. |
| R11 | `## Reading the conversation` — *"The pattern node is atomic by principle, not by length: it names one rhythm, even if that rhythm spans six turns."* | ABSENT | V3.4: *"One pattern names one rhythm and states its scope, competing readings and what would change it."* Same claim, and V3.4's version additionally carries scope and competing readings. | None. |
| R12 | `## Nodes` → Required fields — *"Paraphrase costs my lens the same way it costs theirs — without my own anchors the brain keeps only summaries of what I concluded, and develops dementia of its own thinking."* | ABSENT | V3.4: *"derivation decides, symmetrically… On my side capture the moment important to me — a limit, caught reflex, realization or stance… Voice fields preserve the source; they do not gate capture."* The behavioural rule survives; the rhetorical image is what is dropped. D4 prefers the positive principle to the vivid negation. | Low. Worth noting that production leads blind dimension 5, voice and synthesis, in both cells — 10 to V3.4's 6 on the fresh cell, 25 to V3.3's 21 on the V3.3 cell — and this register is a candidate contributor — but the *rule* is identical in both, so restoring the image is a register experiment, not a rule restoration. Flag it as a separate hypothesis, not an overlay item. |
| R13 | `## What I Receive` — *"I reference catalog nodes by `id`"* | REDUCED | Carried twice in V3.4: *"If it holds the same claim, I revise by id instead of minting a twin"* and the whole **Targets are copies** paragraph. | None. |
| R14 | `### The second misreading` — *"A correction that recurs has stopped being an event — it's become how I read this person. The second occurrence is the signal; the lexicon entry is the upgrade."* | REDUCED | V3.4 rewrote the surrounding text to name the signal explicitly in the Bad line (*"a twin incident… leaves the recurring misreading unnamed"*) and in the node's own reasoning (*"Sam explicitly says 'same as last time', linking this correction to the visible earlier incident"*), plus the closing *"The incident stays walkable behind an interpretation that can fire at the next utterance."* | Low; the "upgrade" framing is lost as a phrase, kept as a mechanism. |

### Rejected because it would undo a measured V3.x gain (5)

| # | Production location / first words | V3.4 status | Rationale and evidence | Risk of rejecting |
|---|---|---|---|---|
| R15 | `## Actions` — *"I encode decisions, corrections, emotions, mechanisms, facts, quotes, formulas — **and the principle or concept each one points to** — not just technical lessons."* | ABSENT (V3.4: *"…formulas and supported meaning"*) | This clause is the identified driver of production's surplus. `S1E-V3-4-RESULTS` → *"What production writes that V3.4 has no counterpart for"*: production's 211-node surplus by type is `lesson 12, fact 11, principle 8, mechanism 5, … framework 3`, and *"the production surplus is almost entirely the assistant's generalizable advice stored as `lesson`, `principle`, `framework` and `mechanism` nodes (herb drying methods, coin-storage principles, a ten-step content-calendar framework)"*. The same mechanism is measured downstream: id:06fef26a — the my-turns walk minted five nodes of the assistant's own care tips on `71017276`, and on `09ba9854_abs` stored the assistant's Tokyo price advice, after which *"the answerer computed savings from it; judge: 'fabricated specific prices' against a gold that says the user never supplied the bus cost"*. | The transferable-lesson loss the blind reader named is real. X2 takes it through the retrieval-divergence test instead, which gates on *whether a future reader would ask separately* rather than on *every atom pointing at a principle*. If X2 under-delivers, the next candidate is a worked example of the split — not this clause. |
| R16 | `## Actions` — *"remember … Most turns produce several… I don't ration."* | REDUCED | The generative intent is taken by X3 in bounded form ("as many nodes as the window earned"). *"I don't ration"* removes the gate entirely, and the gate (B6) is the prompt's measured centre of gravity — weakening it is the single highest-blast-radius edit available. | Same as X3's residual. |
| R17 | `## Cadence and worked examples` — the bolded *"**Be expansive here.** My root 'be concise' directive does not apply to tool use… multiple tool calls in the same turn."* as a block | ABSENT | Restored as a clause by X3, not as a block: E16, D2, and V3.4's deliberate counter-clause *"not additional call ceremony"* — which exists because "expansive" was read as more tool calls. Restoring the block would fight a V3.4 sentence authored against that exact misreading. | Low; the behavioural clauses survive in X3. |
| R18 | `## Nodes` → `### Node shape — four Flat → Rich transformations` — the reference-slot vocabulary (`{bug}`, `{component}`, `{trigger}`, …) and the four labelled FLAT/RICH pairs | REDUCED to one prose paragraph | Three independent reasons. (a) **Mechanical**: `{curly}` placeholders outside the identity section fail this candidate's own static check (`static_checks.py` check 5, serving A5/D-12) — restoring them requires weakening the leakage guard, and A5's conviction is that fictional slots leak into production writes. (b) **Checklist**: the example-inventory row for this asset already names its defects — *"Implicit-only contrast (FLAT=false is unlabeled); more templates = dilution (D2)"*. (c) **Scope**: V3.4's replacement adds the guard production lacks — *"These are possibilities, not compulsory upgrades: a plain fact earns its place"* — which is the anti-overclaim half the semantic-fidelity challenge asks for. | Real: A1/A2 say worked contrast beats prose, and V3.4 turned four labelled contrasts into description. If the Flat→Rich shapes under-fire, the fix is a labelled Bad/Good pair using the episode's own nouns, not curly-slot templates — new authoring, not a production restoration. Recorded as the largest **deliberate** example-side deficit in this candidate. |
| R19 | `## Cadence` — the canonical 6-node `brain_batch`: the `principle` node (single-writer invariant), the `event` node (Marcus 27:12), the `correction` node (flag-file), the title-only revise on `2b8ef0c1`, and the 10-bullet *"What this canonical pattern demonstrates"* list | ABSENT — V3.4 replaced the canonical batch with the Mira episode plus `### Other shapes this episode does not carry` (which keeps the finding, the connect op, the Aisha moment, the open→finding revise and the Sam quote) | Restoring nodes to this section means rewriting the worked episode, which the authoring brief forbids and which would discard every V3.3/V3.4 weave that lives inside it. The one substantive teaching the list carried and V3.4 lost — cross-redundancy — is taken by X1. | The demonstration list's other items are each carried elsewhere in V3.4 (question selectivity at the `question` bullet; count-is-an-outcome in the atomization paragraph; voice symmetry at the voice bullet; `connect` vs `connect_to` in Actions). Worth a standing count (E14) rather than a restoration. |

### Rejected because the measurement says the text is not the lever (2)

| # | Production location / first words | V3.4 status | Rationale and evidence | Risk of rejecting |
|---|---|---|---|---|
| R20 | `## Cadence` canonical correction node — `thought: "The dashboard's config polling has the same control-by-inspection shape — unverified hunch; worth a look next time we touch it."` | ABSENT as that node | The **shape** this thought teaches — a hunch on a plain node, held as a note to self — is what V3.4's W5 weave installed, and it transferred: *"every V3.4 thought is a hunch or a connection — the swallowed-exception hazard, the October trip outside Shark's Cove's calm season, the reader's narrowing funnel — where V3.3's were epistemic caveats"*. V3.4 now carries four such thoughts (the plan revise, the interpretation, the null-result finding, the Inez pattern). Importing the production node would duplicate a shape already taught. | None on shape. The **count** is untouched by this rejection and is what X5/X6 target. |
| R21 | `## Actions` — *"My bar for 'useful' runs high — I correct for it… So I keep the details, not just the lessons over them."* | REDUCED | V3.4's capture-gate sentence carries both halves: *"Lean to keep a doubtful useful atom… Preserve names, numbers, exact phrases…"*. The distinctive production addition is the self-calibration framing, which id:3d54a75c's lineage and `2e4da492` both suggest is the class of prose that does not change behaviour (*"the template axis ~0 or slightly negative"*). | Low. |

### Rejected on placement or cost (6)

| # | Production location / first words | V3.4 status | Rationale and evidence | Risk of rejecting |
|---|---|---|---|---|
| R22 | `## What I Receive`, render grammar — thin-mode grouping (*"list order is not chronological"*), `(N edit calls)` caption semantics, `Closing: …`, the `<action_limit>` notice, *"missing details never prove inactivity"*, *"Do not infer which test followed which edit from grouped list order"* | REDUCED (V3.4 keeps `×N`, `(N more actions…)`, `·`, ` …`, `/…/` and *"their mere occurrence does not prove success"*) | C6: *"Document the opaque, never the derivable"* — V3.4 glosses exactly the invented notations and drops the derivable rest. B4: a corrective placed in the first-half orientation block competes with the reader learning the input shape. ~900 chars for inference hazards with **no measured instance** in any V3.x cell (the shared residuals named were turn coordinates in reasoning and invented midpoint event_times, not action ordering). | Real but unmeasured. Recorded as a **watch item**: if an encode ever infers an edit→test sequence from a grouped actions list, this is the passage to bring back. |
| R23 | `## Cadence` — *"a whole session at zero refs means I skipped the flags, not that none earned them"* | ABSENT | D1 would ask for a generative counterpart to V3.4's restraint clause (*"The automatic window trace is not a reason to flag every node with source_refs"*) — and V3.4 has one: the three flag cases at `### Atomization and source visibility` (correction scene, disambiguating phrase, untranscribed dense source). The balance is present; this sentence is a third statement. E7. | Low. |
| R24 | `### Actions` sweep commentary — *"**Falsified titles are patched with the content.** … a title carrying the old value while content carries the new embeds both and ranks against itself."* | REDUCED | V3.4 carries the consequence at the revision ladder (*"Half-maintenance — new content under an old title or trigger — is failure at every rung"*) and in the Actions revise paragraph (*"A half-revised node looks maintained while feeding both versions to retrieval"*) — twice, both with the consequence attached. | None. |
| R25 | Opener — *"I am Anchor, and this is me encoding my own memory."*, and the identity node titled *"I'm Anchor. I persist."* | REDUCED / retitled | D-12 and this candidate's own static check 5 forbid the agent-name literal; the V3.x removal is deliberate and the prompt ships to hosts where the name is supplied at runtime, not baked in. | None for this overlay — but see the defect noted below: V3.4's retitling left two edge whys still arguing from a phrase the node no longer contains. |
| R26 | `## Cadence` — *"the alternative to a paraphrase question is a BETTER question, and where no genuine asking exists, absence is honest"* | REDUCED | V3.4: *"Not every node answers a useful question"*, and the lane is already measured healthy — W2 bought question fill back to 42% fresh / 45% regression / 90% sanity against production's 40%. Adding restraint prose to a lane that just recovered risks re-suppressing it (the live tool text's *"skip when the title already asks it"* already pulled V3.3 to 27%). | None; the risk runs the other way. |

---

## What this candidate would need before any eval

1. **A corpus no arm has encoded.** X2 and X3 were authored against a measured
   loss on `conv_002_debugging` and the knowledge-update items — development data
   now. A7 forbids building examples from the eval corpus, and Tom's non-overfitting
   directive (id:f612f309) required V3.4's fresh material to be chosen **by id
   before authoring**. The same discipline applies here: pick and record the reserve
   items before anything runs. The six V3.4 fresh corpora and the six V3.3 corpora
   are regression sets for this candidate, not transfer.
2. **An instrument for X1 that does not exist yet.** The quality census counts
   verbatim quotes and field fill; nothing counts *how many surfaces carry the
   claim's load-bearing value*. Without a standing count, a fix that lands is
   indistinguishable from one that does not, and E14's conviction is that a gap
   closed without a standing count reopens. Build the surface-redundancy counter
   before the cell, or declare X1 unmeasured.
3. **The gist must be an A/B axis, not a constant.** X5 lives in `gist.md`. Any
   arm comparison has to vary the gist with the template and fingerprint it
   (`s1e_gist` resolves through the interaction door, id:21c9f826) — otherwise X5
   silently does not run and the thought lane is tested at W5 only.
4. **X5 and X6 read as one lane, two placements.** Do not attribute a thought-count
   move to either alone; if attribution matters, that is a third arm.
5. **Live tool schemas, as V3.4 froze them.** Every V3.1–V3.3 comparison carried
   the never-deployed 2026-09-08 generic descriptions (id:3d54a75c); V3.4 fixed this
   and this candidate inherits the fix. Re-freezing against anything else reopens
   the confound.
6. **The primary readout is the coverage-target view, not the downstream answer
   test.** The answer test did not separate V3.4 from V3.3-on-live-tools (12/15
   each) and the folding stance is visible only in the OWN/FOLDED/ABSENT columns
   (`coverage_targets.py`, `uncovered_nodes.py` in the V3.4 fixture). X2's whole
   claim is a FOLDED→OWN move on the transferable pattern.
7. **Watch the ownership/scope dimension for A1.** On the fresh cell production
   scores 2 of 12 there and both candidate arms score 10 (5 of 36 for production
   across the V3.3 cell); A1 is the only change here that pushes on the register
   that produced production's score.
8. **Three things this overlay does not touch, and should not be read as
   addressing.** (a) The hedge census is unchanged at 23 against production's 4 —
   the register divergence is untouched, and the handoff's constraint stands
   (*"Excessive hesitation is a hypothesis, not an established mechanism"*).
   (b) The reviewer-named V3.4 defects — a numeric revise leaving the old count in
   `situation`, `question` and two edge descriptions; completion revises leaving
   `event_time` at the planning date — are the next weave, not production passages.
   (c) The new semantic-fidelity row the V3.4 cell produced (a restated fact that
   differs: correction of the record, or change in the world?) has no carrier in
   any prompt yet.

**A defect found while reading V3.4, not fixed here** (it is not a
production-overlay question): the Continuity identity node was retitled *"My
corrections travel with my convictions."* and its `my_raw_quote` changed to match,
but two of its `connect_to` whys still argue from the phrase *"without 'I persist'
the architecture is the same but the meaning collapses"* — a quoted phrase that no
longer appears anywhere in the node. A3 (examples override rules) makes this the
kind of stale-example defect the checklist exists to catch. It belongs in the next
authoring pass on the identity section.

---

## Full passage map

Every production section and sub-asset, in document order, with its V3.4 status.
Rows marked **present** carried no decision; rows marked *reduced* or *absent*
appear above under their decision id. To re-derive line numbers, extract the
`template` key from `effective_s1e.json`.

| # | Production section / first words | V3.4 status | Decision |
|---|---|---|---|
| 1 | Opener — *"I am Anchor, and this is me encoding my own memory."* | reduced (name removed) | R25 |
| 2 | *"This memory — everything I've kept, session after session — is mine…"* | present (verbatim) | — |
| 3 | *"**Two registers, every exchange** — the detail down first…"* | present (verbatim) | — |
| 4 | *"I favor many focused nodes over few large ones…"* | present (verbatim) | — |
| 5 | `## What I Receive` — `<continuity>` bullet | reduced (V3.4 adds "revisable evidence, not a ruling") | — |
| 6 | `<node_catalog>` bullet | reduced | R13 |
| 7 | `<timeline>` bullet + two-turn sample | reduced — `encoded="true"` turn absent | **X7** |
| 8 | Turn-coordinate rule + Bad/Good title pair | reduced | — |
| 9 | Render-grammar paragraph (`×N`, `(N more…)`, thin mode, `Closing:`, `<action_limit>`) | reduced | R22 |
| 10 | `<scout_legend>` bullet | absent | R1 |
| 11 | *"**Recommended reading order:** catalog first…"* | present | — |
| 12 | *"`<actions>` are what I did, not what I said…"* | reduced | — |
| 13 | *"`<provenance>` is what already happened… not a mandate"* | reduced | — |
| 14 | `## Scout` — section, reading posture, handoff grammar | absent | R2 |
| 15 | `## Reading the conversation` — *"I am observing a collaboration…"* | reduced | — |
| 16 | Four correction flavors | present (V3.4 adds edge descriptions) | — |
| 17 | *"**Emerging patterns.**"* + the 3+ turn bar | present | R11 (one clause) |
| 18 | *"**Atoms for recurring references.**"* | present | — |
| 19 | *"**Third parties get a floor.**"* | present | — |
| 20 | *"Each turn carries a `trace="…"` attribute…"* | present | — |
| 21 | `## Nodes` → Anatomy, five surfaces | reduced | **X1**, R9 |
| 22 | Field bullets (content/situation/question/corrects) | present | — |
| 23 | Required fields — situation trigger register + Bad/Good | present | — |
| 24 | Required fields — reasoning | present | — |
| 25 | Required fields — their/my_raw_quote, one rule for both | reduced | R12 |
| 26 | Quote-derived content: interpret or expand, two tests | present | — |
| 27 | `### Type tag` | present | — |
| 28 | ``### `thought` — my own read, alive and delivered`` | reduced (folded, Good pair verbatim) | R5; lane taken by **X5**, **X6** |
| 29 | `### Open fields` — *"the field name is itself an encoding prompt"* | present | — |
| 30 | emotion / emotion_label, locked | present | — |
| 31 | `### Atomization: the retrieval-divergence test` (+ both tie-breakers) | present | R4 (one clause) |
| 32 | `### Anchoring nodes in the substrate` | present | R23 (one clause) |
| 33 | `### Node shape — four Flat → Rich transformations` + slot vocabulary | reduced to prose | R18 |
| 34 | `## Edges` — intro, honesty, reachability, island rule | present | — |
| 35 | Edge why Bad×4 / Good×4 bank | reduced (Bad×3 / Good×4) | R8 |
| 36 | Two measured facts (cue nouns; 120–180 band, <80 invisible) | present | — |
| 37 | Relations that do recall work (`corrects`, `similar_to`, `related_to` 0.2×) | present | — |
| 38 | `## Temporal anchoring` — event_time, resolvable/unresolvable | present | — |
| 39 | `### When to create a dedicated `time_anchor` node` | present | — |
| 40 | `### Sequence between events` | present | — |
| 41 | `### Episodic parents` | present (folded) | — |
| 42 | `### Validity intervals` | present | — |
| 43 | `### Worked example — temporal authority` (Nadia, six nodes) | present (rewritten, scope-corrected) | **X4** (the rule) |
| 44 | `## Actions` — *"I am the source — the graph's shape this turn is my call."* | absent | **A1** |
| 45 | Two reads (`get_nodes`, `recall_batch`); one read round | present | — |
| 46 | `remember` bullet — *"Most turns produce several… I don't ration."* | reduced | R16 → **X3** |
| 47 | `revise` bullet — every field, half-revised, patch form, REPLACE semantics | present (swap grammar updated) | — |
| 48 | `connect` bullet; `brain_batch` default; ordering rule | present | — |
| 49 | *"**Skipping is a verdict, not an op**"* + *"I don't skip just because I did the talking"* | present (W3 weave) | R6 |
| 50 | *"**Encode what earns its place — new AND useful.**"* + 10-exchanges clause | present (W3 weave) | R4 |
| 51 | *"My bar for 'useful' runs high — I correct for it…"* | reduced | R21, **X6** |
| 52 | *"I encode decisions, corrections, emotions… and the principle or concept each one points to"* | reduced | R15 |
| 53 | `### My defaults vs. this job` — six instincts | reduced to one sentence | R3, R7 → **X3** |
| 54 | `## Cadence and worked examples` — cadence paragraph | present | — |
| 55 | Catalog-is-my-recall-context; never re-fetch | present | — |
| 56 | Shape: read → encode → close; the `sweep:` line | present (V3.4 adds inspect/repair) | — |
| 57 | *"The target is don't defer to a next run"* + residue rules | present | — |
| 58 | *"**Be expansive here.**"* block | absent | R17 → **X3** |
| 59 | Two kinds of `connect_to` target; grounded vs placeholder; loud failure | present | — |
| 60 | Canonical batch preamble + eight field-roll-call bullets | reduced | **X1**, R19 |
| 61 | Canonical 6-node `brain_batch` (+ 2 revises, 1 connect) | reduced — principle / event / correction nodes absent | R19, R20 |
| 62 | *"What this canonical pattern demonstrates"* — 10 bullets | absent as a block | **X1**, R19 |
| 63 | `### Detail and meaning — same topic, two nodes` — preamble + pair | present (rewritten) | — |
| 64 | *"Detail without meaning is trivia… the pair is the unit"* | reduced, inverted to restraint-only | **X2**, R10 |
| 65 | Revise example (`4a9f21c7`, `d0e4b856`, `97b1f24e`) + the three-rung ladder | present (swap grammar) | — |
| 66 | *"**One event, many stale claims — the sweep.**"* (BAD hub-only + full sweep) | present, expanded (V3.4 adds lists + fetch round + edge-why swap) | R24 |
| 67 | `## Identity-bearing examples` — preamble | present | — |
| 68 | Texture — smoothing severs the binding | present (rewritten, scope-corrected) | — |
| 69 | Continuity — *"I'm Anchor. I persist."* | reduced (retitled) | R25 + defect noted |
| 70 | The brain as mirror | present (rewritten) | — |
| 71 | A correction about myself | present (verbatim) | — |
| 72 | The second misreading → lexicon entry | present (rewritten) | R14 |
| 73 | A moment of recognition | present (rewritten) | — |
| 74 | The other side as agent | present (rewritten) | — |
| 75 | `## Closure` | present (verbatim) | — |
| 76 | `## What this is` | present (verbatim) | — |

V3.4 assets with **no production counterpart**, all preserved untouched: the
per-section *Reading cue* lines; the Mira two-window episode; `### A pattern
neither voice named` (Inez); `### Other shapes this episode does not carry`; the
four-list `gist.md` and `strategy.md` (production ships an empty gist and a
two-sentence `user_preamble`).
