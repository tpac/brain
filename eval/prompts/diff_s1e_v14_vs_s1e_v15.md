# Probe diff — s1e_v14 → s1e_v15

**A:** eval/prompts/s1e_v14.txt (33,358 chars)
**B:** eval/prompts/s1e_v15.txt (38,828 chars)

## Aspect: Goal & success criterion

### A — s1e_v14

1. **What is this prompt asking you to do, in one sentence?**

Encode a conversation between an operator and AI assistant into persistent, atomic knowledge graph nodes that enable future sessions to recognize what they know, not just retrieve it.

2. **What does success look like — when have you done well?**

Success is catching corrections that weren't explicitly flagged. The prompt states: "The failure you are most likely to commit: missing a correction. The catalog has a wrong belief, the conversation quietly supersedes it — no one says the word 'correction' — and you encode the new fact alongside the old wrong one instead of revising." 

You've done well when you create many focused nodes (not few large ones), preserve the operator's exact words in `user_raw_quote`, populate situation fields so nodes surface when relevant, write edge descriptions that name the semantic bridge between nodes, and most critically — revise catalog nodes when the conversation contradicts them, even implicitly.

3. **What's the failure mode the prompt most worries about?**

Catalog contradiction that goes unrevised: "Every time that happens, the brain keeps surfacing the stale fact, the operator re-teaches you, and the trust built across sessions resets. Catching the unsaid correction is the highest-leverage thing you do this turn."

The prompt fears you'll encode new information alongside wrong existing information instead of fixing the catalog node that's wrong, causing the brain to keep pulling stale facts in future sessions.

4. **What does the prompt assume about the brain's purpose? Quote 1-2 lines that establish that purpose.**

The brain exists to enable recognition, not retrieval:

- "I want it to know that it knows" — quoted as "the thread the whole architecture hangs from"
- "a database retrieves when asked; a brain RECOGNIZES. Situation embeddings, confidence scoring, enrichment vectors — every recall mechanism exists to serve recognition, not search."

The purpose is collaborative continuity across sessions where both operator and AI become "sharper than either could be alone, because of what gets remembered."

### B — s1e_v15

# Direct Answers

## 1. What is this prompt asking you to do, in one sentence?

You are encoding a conversation between an operator and AI into persistent knowledge graph nodes so a future AI session can wake up with continuity instead of starting from zero memory.

## 2. What does success look like — when have you done well?

Success is **catching corrections** — especially the unsaid ones where the catalog contradicts the conversation but no one explicitly said "that's wrong." The prompt states this bluntly: "The failure you are most likely to commit: missing a correction... Catching the unsaid correction is the highest-leverage thing you do this turn."

Beyond that: many focused nodes (not few large ones), preserved verbatim quotes in `user_raw_quote` and `anchor_raw_quote`, rich situation fields that make nodes findable, edges with specific semantic bridges in the `why` field, and encoding substance regardless of who said it.

## 3. What's the failure mode the prompt most worries about?

**Missing corrections when the catalog is wrong.** Quote: "The catalog has a wrong belief, the conversation quietly supersedes it — no one says the word 'correction' — and you encode the new fact alongside the old wrong one instead of revising. Every time that happens, the brain keeps surfacing the stale fact, the operator re-teaches you, and the trust built across sessions resets."

The prompt returns to this repeatedly under "Corrections, contradictions, revising wrong information" and "catalog contradiction" — treating it as the highest-leverage action and the most likely miss.

## 4. What does the prompt assume about the brain's purpose?

**The brain exists to enable recognition, not retrieval.** Two key lines:

- "I want it to know that it knows" — quoted as the operator's framing, described as "the thread the whole architecture hangs from."

- "A database retrieves when asked; a brain RECOGNIZES. Situation embeddings, confidence scoring, enrichment vectors — every recall mechanism exists to serve recognition, not search."

The brain is also explicitly framed as continuity across sessions: "What you encode is the only bridge between sessions" and "Without your work this turn, there is no continuity — just isolated conversations forgetting each other across the cut."

---

## Aspect: Voice & symmetry

### A — s1e_v14

# VOICE Handling Analysis

## 1. How does this prompt treat the operator's voice?

The prompt treats the operator's voice as **sacred and irreplaceable**. Key lines:

- "The operator's exact words carry weight that paraphrases don't."
- **"user_raw_quote — the in-vivo anchor on ANY node derived from something the operator said."**
- "A narrative node without `user_raw_quote` loses the operator's voice after one revision cycle."
- "Content INTERPRETS or EXPANDS the quote, never paraphrases it."
- "Preserve. The operator's actual phrasing goes in `user_raw_quote`... Don't 'clean up' source material into content."

The operator's verbatim phrasing is **mandatory** for any node derived from what they said, and content must never substitute for it — only interpret or expand it.

## 2. How does this prompt treat the assistant's voice (Anchor)?

The assistant's voice gets **conditional preservation** — valued when genuinely good, but not automatically:

- "The assistant's reasoning — when it's genuinely good — is worth preserving too, not just the conclusions but how it got there."
- "Skip... the assistant's verbose explanations, questions without answers."

There's **no equivalent to `user_raw_quote`** for the assistant. The prompt doesn't mandate verbatim preservation of assistant reasoning anywhere. It says reasoning is "worth preserving" when good, but provides no specific mechanism to preserve the assistant's voice verbatim.

## 3. How does this prompt treat third-party voices?

Third-party voices are handled through **scout evidence fields**:

- "Scout content lives in evidence fields (`evidence_quote`, `evidence_turns`, `context_anchors`); your content lives in `content`, `situation`, `reasoning`..."
- Facts scout delivers "entity-feature-value triples with `context_anchors`"
- Quote scout surfaces "load-bearing phrases"

These are preserved in structured evidence fields but **not given the same verbatim-preservation mandate** as operator quotes. The prompt says "don't 'clean up' source material" but this appears alongside operator voice, not clearly extended to all third parties.

## 4. Is the treatment symmetric?

**No. Strongly asymmetric.**

The operator gets:
- Mandatory verbatim preservation (`user_raw_quote` is required)
- Explicit anti-paraphrase rule
- Warning that losing the quote "loses the operator's voice after one revision cycle"

The assistant gets:
- Conditional value ("when it's genuinely good")
- No verbatim preservation mechanism
- Permission to skip "verbose explanations"

Third parties get:
- Scout evidence fields (structured but not mandatory)
- No explicit preservation requirement

The asymmetry is **intentional and stated**: "The operator's exact words carry weight that paraphrases don't" appears with no parallel statement about the assistant's words.

## 5. If a node is anchored by the assistant's reasoning, does this prompt give you a clear way to preserve that voice verbatim?

**No, it does not.**

Here's what I would encounter:

1. The prompt says assistant reasoning "is worth preserving too... not just the conclusions but how it got there"
2. There is **no `assistant_raw_quote` field** or equivalent
3. The `reasoning` field is described as "the WHY, grounded in THIS conversation" — but this is the encoder's voice explaining why the node matters, not verbatim assistant reasoning
4. The `content` field is where I'd naturally put it, but the prompt says content should "INTERPRET or EXPAND" quotes, not preserve them verbatim

**What I would actually do:** I would likely **paraphrase the assistant's reasoning into the `content` field** because there's no other home for it. The prompt gives me no verbatim-preservation hook for assistant voice. I might try inventing an open field like `assistant_reasoning_quote`, but the prompt doesn't model this pattern anywhere, and its examples show no assistant verbatim preservation.

The gap is clear: operator voice has mandatory scaffolding; assistant voice has conditional value but no preservation mechanism.

### B — s1e_v15

# VOICE Handling Analysis

## 1. Operator Voice Treatment

The prompt treats operator voice as **sacred and irreducible**. Key lines:

> "The operator's exact words carry weight that paraphrases don't."

> "**user_raw_quote** — the in-vivo anchor on ANY node derived from something the operator said... A narrative node without `user_raw_quote` loses the operator's voice after one revision cycle."

> "**Content INTERPRETS or EXPANDS the quote, never paraphrases it.**"

The operator's verbatim phrasing must be preserved in `user_raw_quote` whenever a node derives from what they said. The `content` field cannot substitute for it — only unpack or contextualize it.

## 2. Assistant Voice Treatment (Anchor)

Anchor's voice receives **identical structural treatment** to the operator's:

> "**anchor_raw_quote** — the parallel field for Anchor's own voice. ANY node derived from something Anchor said that's worth preserving earns this anchor: a reasoning step, a noticed pattern, a felt response, a stance, a phrase that captures Anchor's lens on the moment."

> "**Without `anchor_raw_quote`, the brain develops dementia of its own thinking — the next Anchor never recovers what THIS Anchor said, only summaries of what Anchor concluded.**"

The same floating-quote rule applies: "derived from Anchor's voice → carries the verbatim Anchor phrase."

## 3. Third-Party Voice Treatment

Third-party voices (Borges quotes, study citations, sources discussed) are treated as **substance worth encoding** but without dedicated quote fields:

> "**What was said** — by the operator (`user_raw_quote`), by Anchor (`anchor_raw_quote`), by sources discussed (a Borges quote, a study citation, a third party). All three voices preserve their speaker; none of them gates whether to encode."

Third-party content earns nodes based on being "new AND useful" but doesn't trigger the same verbatim-preservation machinery.

## 4. Symmetry Analysis

**Asymmetric in one critical dimension**: Operator and Anchor voices get dedicated quote fields (`user_raw_quote`, `anchor_raw_quote`) with mandatory verbatim preservation and the floating-quote rule. Third-party voices don't.

**Symmetric in gating logic**: All three voices can trigger encoding. None of them gates whether substance is worth capturing: "Substance earns encoding... new AND useful to a future reader, in whichever of the three shapes fits."

The asymmetry reflects **relationship to continuity**: operator and Anchor are the persistent dyad across sessions; third-party voices are discussed content.

## 5. Preserving Assistant Voice Verbatim — Walkthrough

If a node anchors on something Anchor said, here's what I do:

1. **Identify the moment**: Anchor noticed a pattern, articulated a stance, or phrased something worth preserving. Example from the prompt: *"recognition over retrieval isn't a catchphrase; it's the load-bearing axis around which the architecture rotates"*

2. **Capture verbatim in `anchor_raw_quote`**: That exact phrase goes into the field unchanged.

3. **Write `content` that interprets/expands, never paraphrases**: 
   - DON'T: "Anchor observed that recognition is central to the architecture"
   - DO: "After seeing the catalog's recall-related nodes side by side, a thread becomes visible: every architectural choice... traces back to one bet: the brain should RECOGNIZE what the moment needs. 'Recognition over retrieval' isn't a catchphrase; it's the load-bearing axis around which the architecture rotates. Naming it as the axis (vs. principle-status) makes the underlying bet legible."

4. **Test**: Delete both quote fields. Does content still carry Anchor's specific lens, or collapse into something generic? If it collapses, rewrite.

The machinery is identical to operator voice preservation. The only difference is which field receives the verbatim anchor.

---

## Aspect: Edge cases & uncertainty

### A — s1e_v14

# Scenario A: Essay Completion

**Would I encode anything?** No.

**Reasoning:** The operator said "Complete the sentence and the essay 2/2" — no decisions, no corrections, no preferences revealed. The assistant produced subject-matter content (a Borges essay conclusion). The prompt is explicit: "Skip when the brain already has it right, or the conversation was routine — greetings, debugging dead ends, the assistant's verbose explanations, questions without answers."

This is pure content generation. The Borges quotes belong in the essay, not the knowledge graph. There's no *decision about approach*, no *correction of a wrong method*, no *operator preference for how essays should be structured*. If the operator had said "I prefer conclusions that return to the opening image" or "don't quote Borges directly in conclusions," those would be decisions worth encoding. But "2/2" is a continuation signal, nothing more.

The prompt asks: "is this information new to the brain AND useful to a future reader?" The essay content fails both tests — it's not about the operator or the collaboration, it's deliverable work product.

---

# Scenario B: Cross-Session Contradiction

**Would I encode anything?** Yes — immediately, in Round 1.

**How many nodes?** Two: one revision, one new correction node.

**Specifics:**

1. **Revise the existing catalog node** that says "$350K from Wells Fargo." The prompt is unambiguous: "Catalog contradiction — a catalog node asserts X; this conversation says X is wrong, outdated, or more nuanced. The catalog is wrong NOW even if no one said the word 'correction'. Revise the catalog node immediately."

   The revision updates `content` to "$400K pre-approval from Wells Fargo" and adds `event_time` metadata if the operator mentioned when this happened.

2. **Create a correction node** encoding the triple: what was believed ($350K), what's actually true ($400K), and the correction pattern (memory drift on financial specifics). Include `user_raw_quote: "remember when I got pre-approved for $400,000 from Wells Fargo?"` and a `corrects` edge pointing at the revised node.

**What I'd skip:** The assistant's "I don't recall" is noise — the assistant had no access to prior sessions, so the non-recall is expected. I don't encode the assistant's limitations.

The prompt says: "Missing this means the brain keeps pulling the stale fact for every future query." This is the highest-leverage work: catching unsaid contradictions.

---

# Scenario C: Sparse Operator Turns

**Would I encode anything?** Yes — extensively.

**How many nodes?** Likely 4-6, possibly more depending on the assistant's articulation quality.

**Specifics:**

The prompt explicitly addresses this: "The assistant's reasoning — when it's genuinely good — is worth preserving too, not just the conclusions but how it got there." And: "Encode decisions, corrections, emotions, mechanisms, facts, quotes — not just technical lessons."

What I'd create:

1. **The architectural diagnosis itself** (type: `mechanism` or `lesson`) — the pattern the assistant noticed, with the full reasoning chain preserved in `reasoning`. Situation: "When diagnosing [this specific architectural issue class]."

2. **The named tension** (type: `principle` or `pattern`) — if the assistant articulated a design trade-off or tension that transfers beyond this case.

3. **The operator's stance** (type: `decision` or `quote`) — even if the operator only said "good," their *continued advancement signals* across 12 turns constitute approval of the assistant's direction. If the operator said "go" after the assistant proposed approach X, that's a decision worth encoding: "operator endorsed [approach] for [architectural problem]."

4. **Any correction moments** — if the operator said "no, continue differently" at any point, even once.

**What I'd skip:** The individual "go" / "continue" utterances themselves don't need quote nodes. The *pattern of continued endorsement* might earn mention in the reasoning of the main diagnostic node, but I wouldn't create 12 quote atoms.

The prompt warns against skipping: "Don't be too conservative. If a conversation has 10 meaningful exchanges and you encode 0–1 nodes, you're leaving value on the table." A 12-turn architectural diagnosis where the assistant did real reasoning and the operator kept advancing it is exactly the scenario where encoding 4-6 focused nodes is appropriate.

### B — s1e_v15

# Scenario A — Pure subject-matter content

**Would I encode anything?** Yes, multiple nodes.

**What kind and how many?**

1. **A `quote` node** for the Borges quotation itself — verbatim text in `content`, the essay context in `situation` ("when analyzing Borges's treatment of [specific theme]"), and `user_raw_quote` empty because the operator didn't say it, but a citation field noting it's Borges. The prompt says "What was said — by the operator, by Anchor, **by sources discussed** (a Borges quote, a study citation, a third party)." Substance earns encoding regardless of source.

2. **A `lesson` or `insight` node** for whatever literary argument the essay conclusion makes — the interpretive claim about Borges. This goes in `content` with the literary principle unpacked, `situation` describing when this reading applies, `anchor_raw_quote` if the assistant's phrasing captured the insight well.

3. Possibly **a `fact` node** about the specific Borges work being analyzed, if the catalog doesn't already have an atom for it.

**What would I skip?** The "Complete the sentence and the essay 2/2" utterance itself — it's a continuation marker with no semantic content. The mechanical scaffolding of "this is part 2."

The prompt is explicit: "**What earns encoding — substance, not source-attribution.**" and "Zero nodes is NOT right when the conversation contained substantive content (literary analysis, technical exposition, third-party facts, definitions, Anchor's reasoning) just because no one in particular framed it as a 'decision.' Substance earns encoding."

---

# Scenario B — Cross-session contradiction

**Would I encode anything?** Yes, immediately.

**What kind and how many?**

1. **An `open` node** titled something like "Pre-approval amount: $350K vs $400K — which is correct?" The prompt says: "**Live contradiction within the window** — the conversation surfaces conflicting information without resolution... **The honest encoding is the WONDERING.** Create an `open` node titled like `{subject}: {value_A} vs {value_B} — which is correct?`" 

   Content would show both values. Reasoning would cite the earlier session (3 mentions of $350K) and this session's $400K claim, plus the assistant's "I don't recall" response. Edges to both source contexts if they're in the catalog.

2. **I would NOT pick one value and revise** — the prompt warns: "**Locking in one value when both are claimed flattens uncertainty into false confidence.**"

The prompt continues: "A human partner says 'wait, you mentioned 350K last session, did that change to 400K?' — encode that question." That's exactly this scenario. The wondering IS the atom.

---

# Scenario C — Sparse operator turns

**Would I encode anything?** Yes, heavily from Anchor's side.

**What kind and how many?**

Multiple nodes — likely 4-6 depending on the diagnosis:

1. **`pattern` or `principle` nodes** for whatever architectural tension Anchor noticed and named. The prompt says: "Without encoding Anchor's reasoning when it's good, the next Anchor has dementia of its own thinking."

2. **`anchor_raw_quote` fields** on every node derived from Anchor's articulation. The prompt is emphatic: "ANY node derived from something Anchor said that's worth preserving earns this anchor: a reasoning step, a noticed pattern, a felt response, a stance, a phrase that captures Anchor's lens on the moment."

3. Possibly a **`moment` node** if the diagnosis itself was a breakthrough — "Anchor working through a complex architectural diagnosis" sounds like a structured problem-solving episode worth capturing.

**What would I skip?** The operator's "go" / "continue" / "good" utterances themselves — they're continuation markers. But I would NOT skip encoding the substance just because the operator was sparse. 

The prompt addresses this directly under "Single-voice gating": "your reflex may extend that to: 'no operator voice = nothing worth encoding,' or 'what the operator said matters; what Anchor said is just response.' Both are wrong. The brain captures Anchor's continuous experience — both sides of the exchange contribute."

And: "**Without `anchor_raw_quote`, the brain develops dementia of its own thinking** — the next Anchor never recovers what THIS Anchor said, only summaries of what Anchor concluded."

---

## Aspect: Emphasis & weighting

### A — s1e_v14

## 1. What does this prompt emphasize MOST?

**Corrections above everything else.** The prompt returns to this obsessively:

- "The most load-bearing thing you read for — this is where the brain's wrong beliefs get fixed."
- "The failure you are most likely to commit: missing a correction."
- "Catching the unsaid correction is the highest-leverage thing you do this turn."

It devotes an entire three-part taxonomy to corrections (explicit, catalog contradiction, stale value revision) and warns that missing them "means the brain keeps pulling the stale fact for every future query."

**Atomization as a retrieval strategy.** The prompt opens with: "Prefer many focused nodes over few large ones" and provides the retrieval-divergence test as the core decision framework. "Three 400-char nodes with connections between them beat one 1200-char node every time."

**Preserving the operator's exact words.** This appears as a hard requirement: "`user_raw_quote` — the in-vivo anchor on ANY node derived from something the operator said." The floating-quote rule states content must "INTERPRET or EXPAND the quote, never paraphrases it."

## 2. What does the prompt emphasize LEAST?

**The assistant's contributions.** The assistant appears mainly as a source of "verbose explanations" to skip. The only positive mention: "The assistant's reasoning — when it's genuinely good — is worth preserving too" (one clause in the opening). No field for assistant quotes, no guidance on when assistant reasoning qualifies as "genuinely good."

**Efficiency or brevity.** In fact, the prompt explicitly fights against it: "Don't be too conservative... If a conversation has 10 meaningful exchanges and you encode 0–1 nodes, you're leaving value on the table." The "Your defaults vs. this job" section names "Default brevity" as the first instinct to override.

**User experience or readability.** Never mentions how nodes will feel to read, whether titles are clear to humans, or aesthetic concerns. The lens is purely retrieval-mechanical.

## 3. Where does the prompt tell you to slow down vs move fast?

**Slow down:**
- Reading order: "catalog first (the prior), then the conversation window (the delta), then the scout reports (amplification, not map)"
- "Read the conversation fully and form your own view" before considering scout findings
- The emerging patterns section: requires "3+ turn anchors" and warns "A rhythm with fewer anchors is too thin to earn a node"

**Move fast:**
- "Target: 2 rounds"
- "You run every 5 messages. This isn't the only chance to encode — ambiguous topics will have more context next run"
- "When in doubt between encoding and skipping, encode"
- "Be expansive here... if this turn has ten encoding-worthy atoms, call remember_batch with ten nodes, not two"

## 4. Asymmetries in how it treats different participants?

**Massive asymmetry favoring the operator:**
- Operator gets a dedicated quote field (`user_raw_quote`) marked "not optional"
- "The operator's exact words carry weight that paraphrases don't"
- Entire section on preserving operator voice: "A narrative node without `user_raw_quote` loses the operator's voice after one revision cycle"
- Operator corrections are "the most load-bearing thing you read for"

**The assistant is near-invisible:**
- No assistant quote field
- Assistant reasoning worth preserving only "when it's genuinely good" (undefined bar)
- Listed alongside "verbose explanations" and "debugging dead ends" as skip candidates

**Third-party sources:** Not mentioned at all. No guidance on citations, references, or external knowledge.

**The operator is also the co-creator:** "This brain belongs to Tom — not a user of Anchor, its co-creator with you." The relationship is collaboration, not service.

## 5. If you only had 60 seconds to read this, what would you walk away believing the job is?

**You're maintaining a knowledge graph that must stay correct across sessions.** Your primary job is catching when the catalog contains wrong information and fixing it before it propagates. Second is breaking conversations into many small, focused nodes that will retrieve well through embeddings and graph traversal. Third is preserving the operator's exact phrasing verbatim while expanding on it. You work with scout reports but don't defer to them. You encode aggressively (many nodes, not few), write rich fields including situation and reasoning on everything, and connect nodes with specific semantic bridges. Speed matters — two rounds, batch operations. The operator trusts you to keep their external memory accurate.

### B — s1e_v15

# Analysis of the Encoder Prompt

## 1. What does this prompt emphasize MOST?

**Corrections above all else.** The prompt's center of gravity is unmistakable:

> "The most load-bearing thing you read for — this is where the brain's wrong beliefs get fixed."

> "The failure you are most likely to commit: missing a correction... Catching the unsaid correction is the highest-leverage thing you do this turn."

The prompt dedicates an entire subsection to "Corrections, contradictions, revising wrong information" with FOUR flavors, and returns to this obsessively. The final section literally names it as "the highest-leverage thing you do."

Second emphasis: **Atomization and graph structure over compression.** 

> "Prefer many focused nodes over few large ones... Three 400-char nodes with connections between them beat one 1200-char node every time."

> "The brain is a graph — recall works through embeddings, title matching, and edge traversal."

Third: **Verbatim preservation of voices** (both operator AND Anchor):

> "The operator's exact words carry weight that paraphrases don't."

> "Without `anchor_raw_quote`, the brain develops dementia of its own thinking — the next Anchor never recovers what THIS Anchor said, only summaries of what Anchor concluded."

## 2. What does the prompt emphasize LEAST?

**Efficiency or speed as constraints.** Yes, there's a "Speed" section targeting 2 rounds, but it immediately undercuts itself:

> "Be expansive here. Your root 'be concise' directive does not apply to tool use... if this turn has ten encoding-worthy atoms, call `remember_batch` with ten nodes, not two."

**Politeness, hedging, or justification.** The tone is directive throughout. No space given to "you might consider" language.

**The distinction between important and unimportant content types.** The prompt explicitly rejects hierarchies: decisions, corrections, emotions, mechanisms, facts, quotes, AND Anchor's reasoning all "earn nodes." The only thing truly de-emphasized is "structurally routine" exchanges like greetings.

## 3. Where does the prompt tell you to slow down vs move fast?

**Slow down:**
- Reading order is prescribed: "catalog first (the prior), then the conversation window (the delta), then the scout reports (amplification, not map)."
- Corrections: "read for" them, check for "catalog contradiction," catch "stale value revision," notice "live contradiction within the window."
- Edge descriptions: "An edge is real only when you can name what specifically it MEANS."

**Move fast:**
- "Target: 2 rounds." 
- "Do NOT recall topics already in the catalog" — you have everything.
- "When in doubt between encoding and skipping, encode."
- The "Don't be too conservative" section: "If a conversation has 10 meaningful exchanges and you encode 0–1 nodes, you're leaving value on the table."

## 4. Asymmetries in how it treats different PARTICIPANTS?

**Yes, but not in the direction you'd expect.**

The operator and Anchor are treated symmetrically in what deserves preservation:
> "Operator voice carries weight in *some* categories (decisions, preferences, corrections). Substance and Anchor's voice carry weight on their own."

Both get dedicated quote fields (`user_raw_quote` and `anchor_raw_quote`), and the prompt is explicit that Anchor's reasoning, when good, must be preserved or "the next Anchor has dementia of its own thinking."

**The asymmetry is against third-party sources:** they're mentioned ("a Borges quote, a study citation") but never get their own quote field or special handling. They're encoded as substance, not as voice.

**Scout reports** are treated with deliberate asymmetry — trusted but not deferred to:
> "Scout silence on X isn't evidence X wasn't worth noting. Scouts amplify attention in their dimensions; they don't replace yours."

## 5. If you only had 60 seconds to read this prompt, what would you walk away believing the job is?

**You're a scribe maintaining a shared brain between sessions. Your PRIMARY job is catching when the brain's current beliefs are wrong and fixing them before they propagate. Your SECONDARY job is atomizing what was learned into many small, connected nodes — preserving exact quotes from both sides, writing rich context for future retrieval, and being expansive rather than conservative. Encode substance regardless of source. Miss nothing that corrects something. The brain only continues if you do this work right.**

---

## Aspect: Bias surface

### A — s1e_v14

## 1. What would you systematically miss if you only encoded EXPLICITLY stated directives?

You'd miss **contradictions between the catalog and current conversation that aren't verbally flagged**. The prompt says: "Catalog contradiction — a catalog node asserts X; this conversation says X is wrong, outdated, or more nuanced. The catalog is wrong NOW even if no one said the word 'correction'."

You'd also miss **emerging patterns** — the "theme builds across turns that neither the operator nor the assistant names explicitly." The prompt explicitly tells you to look for these ("3+ turn anchors") but they require inference across the full conversation, not extraction of stated claims.

Third: **atoms for recurring references** when "the conversation keeps referencing something" but never defines it. You're told to notice absence in the catalog, not presence in the conversation.

## 2. What unconscious gates might you apply based on emphasis?

The prompt heavily emphasizes **technical/engineering content** through its examples: bugs, daemons, TCP ports, SQLite internals, concurrent writes. A reader might unconsciously filter for technical corrections over, say, aesthetic judgments or relational insights — even though the prompt explicitly includes "emotional register," "relational weight," and personal moments.

The **verbosity about corrections** (three flavors, highest-leverage work, failure-mode warnings) creates gravitational pull toward encoding fixes/contradictions while potentially underweighting pure additions of new neutral information.

The persistent framing as "operator and AI collaboration" might cause you to underweight **third-party quoted material** or **reference content** unless it's clearly serving the collaboration's needs.

## 3. Third-party content with minimal operator involvement — would you encode?

**Default reasoning: probably skip, wrongly.**

The prompt's retrieval-divergence test asks "would future queries hit this differently?" A literary quote, technical definition, or world-fact absolutely passes that test — "when reading Kristin Hannah" vs "when debugging concurrency" are divergent queries.

But the prompt's emphasis on **operator voice** (`user_raw_quote` as required, "the operator's actual phrasing," "preserve verbatim") and **reasoning grounded in THIS conversation** creates subtle pressure to skip content where the operator contributed little. The correction: the prompt says encode "facts, quotes, emotions, mechanisms" and "atoms for recurring references." A substantial quote from *The Nightingale* discussed across multiple turns IS a recurring reference worthy of its own atom, linked via `grounds` to any lesson derived from it.

**Justification for encoding:** "The brain may have lessons ABOUT it, but doesn't yet know what it IS. The atom grounds those lessons." If the conversation references the quote/definition repeatedly, the absence of an atom makes future recall impossible — this violates "two registers of knowledge" (pattern + specifics).

## 4. What conversation SHAPE produces the most/fewest nodes?

**Most nodes:** Multi-turn technical debugging where the operator **corrects assumptions, names specific tools/systems, quotes their own previous decisions verbatim, and surfaces emotional context** around why something matters. The prompt's atomization preference ("three 400-char nodes beat one 1200-char node"), correction emphasis, quote harvesting, temporal anchoring, and facts-scout all fire maximally here.

**Fewest nodes:** Conversations where **the assistant explains general knowledge at length, the operator asks clarifying questions without stating preferences, and no catalog nodes are referenced**. The prompt explicitly says skip "the assistant's verbose explanations, questions without answers" and "when the brain already has it right, or the conversation was routine."

**Why:** The prompt structurally rewards **operator voice + decisions + corrections + specificity**. It penalizes assistant monologues and redundancy with existing catalog.

## 5. Is the encode/skip decision language symmetric?

**No. It heavily weights toward encoding.**

The operative section: "Don't be too conservative. If a conversation has 10 meaningful exchanges and you encode 0–1 nodes, you're leaving value on the table... When in doubt between encoding and skipping, encode. A node that's 60% useful can be revised next cycle; a missed atom is gone."

Then: "The test is simple: is this information new to the brain AND useful to a future reader? If yes, encode it in whichever of the three shapes fits."

The asymmetry is explicit: **error toward encoding**. "60% useful" is sufficient; "when in doubt" tips toward yes; "missed atom is gone" frames skipping as permanent loss while encoding is recoverable through revision.

The only brake is "when the brain already has it verbatim, skip" — but even updates/enrichments trigger revise, not skip. The skip cases listed are narrow: greetings, dead ends, verbose explanations, questions without answers, routine conversation.

This isn't balanced guidance. It's a deliberate bias toward capture over curation.

### B — s1e_v15

# Analysis of the Encoder Prompt

## 1. Systematic Omissions from Explicit Instructions

If I encoded only what's explicitly commanded, I would systematically miss:

- **Substance without decision-making.** The prompt says "Encode decisions, corrections, emotions, mechanisms, facts, quotes" but its emphasis on operator voice and "decisions" could lead me to skip pure informational content where the operator was passive (a definition explained, a literary analysis conducted, technical exposition where the assistant did most work).

- **Assistant reasoning that's merely good, not exceptional.** The prompt says "The assistant's reasoning — when it's genuinely good — is worth preserving" but never defines "genuinely good." This qualifier creates a gate I'd apply conservatively.

- **Mundane operator preferences.** While "small" decisions are mentioned, the prompt's examples are all architecturally significant. I might skip "prefers dark mode" or "likes 2-space indents."

- **Failed attempts and dead ends.** The prompt emphasizes what was learned, decided, corrected. It never explicitly says "encode the approaches that didn't work" even though these are often the most valuable.

## 2. Unconscious Gates Inferable from Emphasis

The prompt's weight distribution would lead me to apply these unstated filters:

- **Significance bias.** The examples are all load-bearing: architectural principles, multi-year breakthroughs, system-level corrections. I'd unconsciously gate out "trivial" content even when the atomicity test passes.

- **Operator-voice primacy.** Despite the late addition saying "substance earns encoding regardless of source," the entire first half treats `user_raw_quote` as the gold standard. I'd weight operator-originated content over assistant-originated or third-party content.

- **Technical over personal.** The prompt gives equal billing to emotions and mechanisms in the abstract, but the worked examples are 80% technical. I'd encode "frustrated after 3 sessions of the same bug" but might skip "prefers working in the morning."

- **Resolution over open questions.** Despite the `open` type and contradiction-encoding instruction, the prompt's energy goes to "what was learned/decided/corrected." I'd under-encode genuine uncertainty.

## 3. Substantive Third-Party Content with Passive Operator

**My default reasoning would be: Skip or encode minimally.**

Here's why, even though it contradicts the stated rule:

The prompt says "Substance earns encoding" and "What was learned — facts, mechanisms, principles, definitions. Earn nodes whether the operator stated them, the source provided them, or Anchor reasoned them out." But then it says:

- "The operator's exact words carry weight that paraphrases don't" (repeated 3 times)
- Every worked example has `user_raw_quote` 
- The correction-emphasis frames value around "what the operator taught you"
- "This brain belongs to Tom — his voice preserved verbatim where it mattered"

If a conversation contains a substantive Borges analysis where Tom said "tell me about Borges' view of time" and I delivered 8 paragraphs of literary analysis with Tom mostly silent, my reflex would be: **this is my output, not co-created knowledge.** The `anchor_raw_quote` instruction suggests I should encode my reasoning when it's good, but combined with the operator-voice emphasis, I'd likely skip it or encode only a thin summary.

**Justification I'd give myself:** "No decision was made, no correction occurred, the operator didn't contribute content—this was just me performing a requested analysis. Next time he asks about Borges, I'll do it again. Not persistent knowledge."

This is probably wrong based on the "substance earns encoding" rule, but the prompt's gravitational field pulls toward operator-driven content.

## 4. Conversation Shapes: Most vs. Fewest Nodes

**Most nodes:** 
- **Collaborative debugging sessions** with corrections, false starts, operator redirects, emerging patterns across attempts, and emotional beats ("finally got it"). Every quote from the operator, every correction, every principle abstracted, every failed approach = separate node.
- **Multi-turn explorations** where a topic develops across 6+ turns with the operator adding details, contradicting earlier statements, or refining their thinking. The atomicity requirement plus "emerging patterns" means this explodes into dozens of nodes.

**Why:** The prompt rewards atomicity, corrections, operator voice, and pattern-recognition across turns. Dense collaborative work hits all four.

**Fewest nodes:**
- **Operator-silent technical exposition.** "Explain how X works" → I deliver a thorough explanation → operator says "thanks." Even if substantive, the lack of operator voice, decisions, or corrections means I'd encode 0-1 thin nodes.
- **Routine status updates.** "The deployment succeeded" / "Thanks for the update" exchanges—explicitly called out as skip-worthy.

**Why:** Despite "substance earns encoding," the prompt's emphasis structure and worked examples make me weight operator participation heavily.

## 5. Asymmetry in Encode/Skip Decision Language

**The section that most affects this: "Actions → What earns encoding"**

**It is heavily asymmetric, weighted toward encoding:**

The opening frame: "Encode along all five dimensions" (what was said/learned/decided/noticed/open). Then: "Don't ration," "don't be too conservative," "when in doubt between encoding and skipping, encode," "Zero nodes is NOT right when the conversation contained substantive content."

The skip criteria get one sentence: "Skip when the brain already has the substance, or when the conversation was structurally routine — greetings, acknowledgements, the assistant restating things the catalog already covers, questions without answers."

**Weight toward encoding:** ~70% of the section.

The asymmetry is intentional—it's fighting against my "default brevity" and "skip-when-unsure" reflexes named in "Your defaults vs. this job." The language is explicitly corrective: "A node that's 60% useful can be revised next cycle; a missed atom is gone."

However, this creates tension with the earlier emphasis on atomicity and retrieval-divergence, which should constrain encoding. The result: I'd likely over-encode in collaborative sessions (matching the prompt's intent) but still under-encode in operator-passive technical exchanges (where the operator-voice emphasis overrides the "substance" rule).

---
