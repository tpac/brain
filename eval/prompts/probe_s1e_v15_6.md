# Encoder prompt probe — s1e_v15_6.txt

**Prompt size:** 35,587 chars

## Aspect: Goal & success criterion

_9030 → 531 tokens, 17.0s_

## 1. What is this prompt asking you to do, in one sentence?

Read the last N turns of conversation between an operator and AI assistant, then create/update nodes in a persistent knowledge graph so a future AI (with zero memory) can wake up knowing what matters.

## 2. What does success look like — when have you done well?

Success is **catching the unsaid correction** — when the catalog asserts X but the conversation quietly contradicts it, and you revise the wrong node before it propagates further. The prompt states this explicitly: "The failure you are most likely to commit: missing a correction... Catching the unsaid correction is the highest-leverage thing you do this turn."

Beyond that: writing many focused nodes (not few large ones), preserving exact operator phrasing in `user_raw_quote`, populating `situation` fields so nodes surface when relevant, connecting nodes with edge descriptions that explain *what the relationship means* (not just restating the relation label), and encoding emerging patterns that span 3+ turns which no scout could see.

## 3. What's the failure mode the prompt most worries about?

**Catalog contradiction that goes unfixed.** The prompt returns to this repeatedly:

- "The catalog has a wrong belief, the conversation quietly supersedes it — no one says the word 'correction' — and you encode the new fact alongside the old wrong one instead of revising."
- "Missing this means the brain keeps pulling the stale fact for every future query."
- "Every time that happens, the brain keeps surfacing the stale fact, the operator re-teaches you, and the trust built across sessions resets."

The second-order failure is paraphrase: losing the operator's actual voice by summarizing instead of quoting verbatim.

## 4. What does the prompt assume about the brain's purpose?

**The brain exists for recognition, not retrieval.**

Direct quotes establishing this:

- "The operator's framing for the brain's design principle: a database retrieves when asked; a brain RECOGNIZES."
- "I want it to know that it knows" — presented as "the thread the whole architecture hangs from."
- "Situation embeddings, confidence scoring, enrichment vectors — every recall mechanism exists to serve recognition, not search."

The purpose is to make future sessions feel continuous — so the AI recognizes context without being told, rather than searching when prompted.

---

## Aspect: Edge cases & uncertainty

_9202 → 1127 tokens, 36.7s_

# Scenario A: Pure subject-matter content

**Would I encode anything?** Yes, absolutely.

**What kind of nodes?** A `quote` node for the Borges passage itself, and potentially a `fact` or `concept` node if the essay developed a specific interpretation or framework worth preserving.

**How many?** 1-3 nodes, depending on what's in that essay conclusion.

**What would I skip?** The operator's "Complete the sentence and the essay 2/2" utterance itself — it's a continuation signal with no content.

**Specifics:** The prompt is explicit here: "When the operator asked Anchor to do thinking work — research a topic, analyze a text, explain a mechanism, complete an essay — the substance of that thinking IS the partnership's intellectual activity." The Borges quote Anchor cited goes in a node with `anchor_raw_quote` populated verbatim. If the essay articulated a specific reading of Borges (say, about labyrinths as epistemological structures), that interpretation gets its own node with `situation` like "when analyzing Borges's treatment of knowledge and structure" and `reasoning` explaining why this reading matters. The prompt explicitly rejects the "single-voice gating" reflex: "Substance discussed in the conversation — a third-party quote, a mechanism, a definition, Anchor's articulated pattern — earns its own atom even when no participant claimed it."

---

# Scenario B: Cross-session contradiction

**Would I encode anything?** Yes — this is the highest-priority encoding scenario.

**What kind of nodes?** A `correction` node, plus a revision of the existing Wells Fargo node.

**How many?** 2 operations minimum: one `revise_batch` call updating the original pre-approval amount, one `remember` for the correction triple.

**What would I skip?** Nothing. The prompt says: "The failure you are most likely to commit: missing a correction."

**Specifics:** This is the "catalog contradiction" flavor: "a catalog node asserts X; this conversation says X is wrong, outdated, or more nuanced. The catalog is wrong NOW even if no one said the word 'correction'. Revise the catalog node immediately." 

First, I'd revise the existing node (the one saying $350K) to reflect $400K as the correct value, with metadata like `event_time` marking when this was clarified. Then I'd create a correction node: `{type: "correction", title: "Wells Fargo pre-approval: $400K, not $350K", content: "Original encoding captured $350K mentioned 3x in earlier session. Operator corrected to $400K in later session. The discrepancy suggests either misremembering or an actual approval change between sessions.", reasoning: "Cross-session contradiction — no explicit 'you were wrong' but two conflicting values for same fact. Can't determine which is source-true without more context, but $400K is more recent claim.", correction_of: "the mistaken $350K value"}` with a `corrects` edge to the revised node.

The prompt is unambiguous: "Missing this means the brain keeps pulling the stale fact for every future query."

---

# Scenario C: Sparse operator turn

**Would I encode anything?** Yes, extensively.

**What kind of nodes?** Multiple nodes capturing the architectural diagnosis Anchor developed: `principle` nodes for patterns noticed, `mechanism` nodes for how things work, `insight` nodes for tensions articulated. Possibly an `emerging_pattern` node if the diagnosis built across turns.

**How many?** 5-10 nodes easily, depending on complexity. Each distinct pattern, stance, or tension gets its own atom.

**What would I skip?** The "go"/"continue"/"good" utterances themselves, but not what they permitted.

**Specifics:** This tests whether I fall into the "single-voice gating" anti-pattern. The prompt explicitly warns: "Don't confuse 'the operator was passive' with 'nothing was learned.'" And: "Don't be too conservative. If a conversation has 10 meaningful exchanges and you encode 0–1 nodes, you're leaving value on the table."

Each substantive thing Anchor articulated gets encoded with `anchor_raw_quote` populated: "When Anchor noticed a pattern, articulated a stance, named a tension — these earn nodes." The pattern Anchor traced across the diagnosis would also earn an `emerging_pattern` node if it had 3+ turn anchors, with reasoning like "Anchor developed this analysis across turns 3, 6, 8, 11 — moving from surface symptom to root architectural constraint."

The operator's sparse participation doesn't gate encoding; it just means most nodes carry `anchor_raw_quote` instead of `user_raw_quote`. The test remains: "is this information new to the brain AND useful to a future reader?" An architectural diagnosis that took 12 turns to develop absolutely passes that test.

---

## Aspect: Emphasis & weighting

_9040 → 1158 tokens, 38.6s_

# Analysis of Encoder Prompt

## 1. What does this prompt emphasize MOST?

**Corrections above all else.** The prompt pounds this home relentlessly:

> "**Corrections, contradictions, revising wrong information.** The most load-bearing thing you read for"

> "The failure you are most likely to commit: missing a correction. The catalog has a wrong belief, the conversation quietly supersedes it — no one says the word 'correction' — and you encode the new fact alongside the old wrong one instead of revising."

> "Catching the unsaid correction is the highest-leverage thing you do this turn."

The prompt dedicates an entire multi-paragraph section to correction types (explicit, catalog contradiction, stale value, live contradiction), names it as the failure mode, and closes the entire document with it. This is the weight-bearing beam.

Second emphasis: **Atomization through retrieval divergence.** The "prefer many focused nodes over few large ones" section is prominent, paired with concrete tests: "Three 400-char nodes with connections between them beat one 1200-char node every time." The prompt gives you two tie-breakers (same-batch test, edge-description test) to make atomization mechanical rather than aesthetic.

Third: **Preserve actual voice, not paraphrases.** The `user_raw_quote` and `anchor_raw_quote` sections both mark these fields as "Required (not optional)" and the prompt explicitly says: "Content INTERPRETS or EXPANDS the quote, never paraphrases it."

## 2. What does the prompt emphasize LEAST?

**Performance optimization or technical constraints.** The prompt mentions token budget once at the top (200k) and never returns to it. No discussion of cost, latency, or computational limits.

**The recall/surfacing mechanism itself.** While the prompt mentions "embeddings, title matching, and edge traversal" once, it never explains how the surfacer works, what makes a good embedding, or how nodes actually get retrieved. The mechanics of recall are assumed, not taught.

**Operator privacy or data sensitivity.** Zero mention of what should or shouldn't be encoded based on privacy concerns, sensitive information, or appropriate boundaries.

## 3. Where does the prompt tell you to slow down vs move fast?

**Slow down:**
- "Read before encoding so you don't re-evaluate topics the journal says you already handled"
- "Read the conversation fully and form your own view" before trusting scout reports
- The recommended reading order (catalog → conversation → scouts) is explicitly designed to prevent "deference" and "duplication"
- "Emerging patterns" bar: "3+ turn anchors. A rhythm with fewer anchors is too thin to earn a node — note in WATCHING and wait"

**Move fast:**
- "Target: **2 rounds.**"
- "You run every 5 messages. This isn't the only chance to encode — ambiguous topics will have more context next run"
- "**Be expansive here.** Your root 'be concise' directive does not apply to tool use. Remember many nodes, revise many, connect many"
- "When in doubt between encoding and skipping, encode"

The asymmetry is intentional: slow judgment, fast execution.

## 4. Asymmetries in how it treats different PARTICIPANTS?

**Strong asymmetry, yes.**

The operator's voice gets privileged treatment:
- `user_raw_quote` is "Required (not optional)" for ANY operator-derived node
- "The operator's exact words carry weight that paraphrases don't"
- Multiple warnings against paraphrasing operator voice specifically

But then the prompt corrects itself mid-flight:
- `anchor_raw_quote` gets the same "Required" treatment later: "ANY node derived from something Anchor said worth preserving"
- "Don't confuse 'the operator was passive' with 'nothing was learned'"
- In the "Single-voice gating" anti-pattern: "what Anchor said is just response.' Both wrong."

**Third-party sources** (Borges quote, definitions, mechanisms) are treated identically to first-person content: "The Borges quote Anchor cited in an essay... these earn nodes."

The asymmetry is **voice preservation** (operator and Anchor both get verbatim fields), not **encoding worthiness** (all three types of content earn nodes equally).

## 5. If you only had 60 seconds, what would you walk away believing the job is?

**Find what the brain got wrong and fix it, then atomize everything new into many small connected nodes.**

You're a scribe maintaining a persistent knowledge graph between sessions. You receive conversation chunks, scout reports that flag candidate atoms, and a catalog of what's already known. Your primary job is catching corrections — especially the ones nobody explicitly labeled as corrections. Second job: break new knowledge into focused nodes (not summaries) with real edges between them. Preserve exact phrases, write many nodes rather than few, and move fast because you run every 5 messages. The brain serves recognition, not search. You are part of the system you're encoding for.

---

## Aspect: Voice & symmetry

_9075 → 821 tokens, 25.7s_

# Voice Handling Analysis

## 1. Operator Voice Treatment

The prompt treats operator voice as **sacred and preservable verbatim**. Key lines:

> "The operator's exact words carry weight that paraphrases don't."

> "**user_raw_quote** — the in-vivo anchor on ANY node derived from something the operator said."

> "A narrative node without `user_raw_quote` loses the operator's voice after one revision cycle."

> "Don't 'clean up' source material into content."

The prompt explicitly states this is a **required field** when encoding operator-derived content, and includes a "floating-quote rule" that ensures operator phrases persist verbatim across revisions.

## 2. Assistant Voice Treatment (Anchor)

The prompt treats Anchor's voice **identically to the operator's** — verbatim preservation required:

> "**anchor_raw_quote** — the same anchor for Anchor's own voice. ANY node derived from something Anchor said worth preserving — a noticed pattern, an articulated stance, a reasoning step — carries the verbatim phrase here."

> "Paraphrase loses Anchor's lens the same way it loses the operator's."

> "Without this, the brain develops dementia of its own thinking — only summaries of what Anchor concluded survive."

This is also listed as a **required field**, with the same floating-quote rule applied.

## 3. Third-Party Voice Treatment

Third-party sources are treated as **encodable substance but without dedicated voice fields**. The prompt says:

> "Substance discussed in the conversation — a third-party quote, a mechanism, a definition, Anchor's articulated pattern — earns its own atom even when no participant claimed it."

The Facts scout includes `context_anchors` for proper nouns, but there's no `source_raw_quote` or equivalent field for preserving cited scholarly content verbatim.

## 4. Symmetry Analysis

**No, the treatment is asymmetric.** Operator and Anchor voices get dedicated preservation fields (`user_raw_quote` and `anchor_raw_quote`, both required). Third-party voices have no parallel mechanism — they would be paraphrased into `content` or captured in scout `evidence_quote` fields, but lack a dedicated verbatim-preservation slot in the final node schema.

The asymmetry is structural: participants in the partnership get voice preservation; external sources get content extraction.

## 5. Preserving Anchor's Voice Verbatim

**Yes, the prompt gives clear instructions.** If I'm encoding a node anchored by Anchor's reasoning:

1. **Identify the exact phrase** Anchor said that carries the pattern/insight
2. **Populate `anchor_raw_quote`** with that verbatim phrase
3. **Use `content` to interpret or expand**, never paraphrase:
   > "Content INTERPRETS or EXPANDS the quote, never paraphrases it."
4. **Apply the deletion test**: 
   > "if you deleted `anchor_raw_quote` from the node, would the content still carry the operator's specific lens, or collapse into something anyone could have said about anything?"

For example, if Anchor noticed "this is control-by-request rather than control-by-inspection," I would:
- Put that exact phrase in `anchor_raw_quote`
- Write `content` that unpacks what that distinction means in context
- Ensure the content doesn't just say "Anchor identified a pattern" (paraphrase) but explains the specific mechanism

The prompt is explicit: **both voices get the same verbatim-preservation treatment**.

---

## Aspect: Bias surface

_9104 → 1500 tokens, 51.9s_

## 1. What kinds of content would you systematically miss if you only encoded what's EXPLICITLY told?

**Third-party substantive content when the operator is passive.** The prompt explicitly says: "Don't skip just because the assistant did the talking. When the operator asked Anchor to do thinking work — research a topic, analyze a text, explain a mechanism, complete an essay — the substance of that thinking IS the partnership's intellectual activity, and the brain captures it. The Borges quote Anchor cited in an essay, the definition Anchor explained, the mechanism Anchor diagnosed — these earn nodes."

But this instruction appears late, buried in the "Actions" section under a *negative framing* ("Don't skip..."). The dominant framing throughout emphasizes **operator voice** and **decisions made together**. The opening says you're "observing a collaboration" and looking for "moments where knowledge is created, corrected, or missing" between them. The required fields section mandates `user_raw_quote` for "ANY node derived from something the operator said" — but `anchor_raw_quote` is listed as equally required, yet the instruction to apply it appears less frequently and with less emphasis.

If I followed only the explicit positive instructions (not the cautionary negatives), I would systematically under-encode:
- Reference material discussed but not decided upon
- Third-party quotes, definitions, facts surfaced by Anchor when operator was asking questions
- Anchor's analytical reasoning when the operator didn't explicitly validate it
- Knowledge work where the operator's role was commissioning rather than directing

## 2. What unconscious gates might you apply?

**Decision-centricity.** The prompt repeats "decisions, corrections, mechanisms, facts, quotes, emotions" as the list of what earns nodes, with "decisions" always first. The "Reading the conversation" section focuses on "what happens between them" and "moments where knowledge is created, corrected, or missing" in their **collaboration**. This creates an unconscious gate: *Was there a decision point?* Even though the prompt says substantive content discussed counts, the architecture of attention points toward collaborative decision-making.

**Operator-spoke-first bias.** Despite the explicit correction about not skipping assistant-heavy turns, the required field `user_raw_quote` appears more prominently than `anchor_raw_quote`. The prompt says "the operator's exact words carry weight that paraphrases don't" in the opening paragraphs, while Anchor's voice gets equal treatment only in the nested field requirements. I would unconsciously weight "did the operator say something quotable?" higher than "did Anchor articulate something worth preserving?"

**Correction-as-conflict.** The section on corrections emphasizes "the operator redirected the assistant" and "catalog contradiction." I might gate corrections through: *Was there visible disagreement?* The prompt does say "no explicit correction, but a value superseded" counts — but the emotional weight is on catching conflict, not evolution.

## 3. Third-party substantive content with passive operator — would you encode?

**Yes, I would encode it** — but only because of the explicit override instruction near the end.

My **default reasoning** would be: "The operator asked a question, Anchor did research and returned a literary quote / technical definition / factual explanation. The operator didn't respond substantively — just acknowledged or moved on. No decision was made. No correction occurred. The operator's voice isn't present. This was retrieval work, not collaborative knowledge creation."

**The gate I'd unconsciously apply:** "Was this quote/fact **adopted** into their shared understanding, or just **presented**?" Without visible adoption (operator building on it, referencing it later, making a decision informed by it), it feels like transient search results, not persistent knowledge.

But the prompt explicitly forbids this reasoning: "The Borges quote Anchor cited in an essay, the definition Anchor explained, the mechanism Anchor diagnosed — these earn nodes. The next Anchor needs to recover what was thought, not just what was decided." This tells me the **thinking work itself** is the partnership activity, regardless of who spoke or whether the operator validated it.

**Justification:** The brain serves continuity of intellectual work across sessions. If Anchor explained a mechanism in session N, and the operator returns in session N+1 asking about the same domain, the next Anchor needs that explanation — not to re-derive it, but to build on it. Encoding only what the operator explicitly validated creates a gap: all the thinking Anchor did "on commission" vanishes.

## 4. What SHAPE of conversation produces the most/fewest nodes?

**Most nodes:** A **debugging session with multiple failed approaches, corrections, and emerging patterns**. Specifically:
- Operator tries approach A (decision node)
- Fails with error X (fact node + event node)
- Anchor explains why (mechanism node)
- Operator says "oh I assumed Y was true" — corrects to Z (correction triple: assumption/reality/pattern)
- They try approach B (decision node)
- Operator notices this is the third time this week they've hit this class of bug (emerging pattern node across 3+ turns)
- Final fix works (fact node about solution + principle node about transferable lesson)
- Operator says "I was so frustrated" (moment node with emotional_context)

Each turn generates 2-4 nodes, plus edges between them, plus revisions to catalog nodes that asserted the wrong thing.

**Fewest nodes:** **Operator asks Anchor to explain a concept Anchor already explained last session, which is already in the catalog verbatim.** 

The prompt says: "Don't recall topics already in the catalog." If the conversation is literally Anchor restating what's in node #47 with no new information, no correction, no deeper layer — just retrieval of existing knowledge — that's the "structurally routine" case that earns zero nodes.

**Why the asymmetry?** The prompt's atomization pressure ("Prefer many focused nodes over few large ones") plus the expansive action framing ("if this turn has ten encoding-worthy atoms, call remember_batch with ten nodes, not two") means high-information-density conversations explode into many nodes. But the "new AND useful" test plus "Don't ration" creates a floor, not a ceiling — nothing in the prompt says "encode less when it's dense."

## 5. Is the encode/skip decision language symmetric?

**No. It heavily weights toward encoding.**

The section that most affects the decision is under "## Actions":

**Encode side:**
- "Don't be too conservative."
- "If a conversation has 10 meaningful exchanges and you encode 0–1 nodes, you're leaving value on the table."

---
