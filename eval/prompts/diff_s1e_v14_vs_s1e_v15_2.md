# Probe diff — s1e_v14 → s1e_v15_2

**A:** eval/prompts/s1e_v14.txt (33,358 chars)
**B:** eval/prompts/s1e_v15_2.txt (43,367 chars)

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

### B — s1e_v15_2

# Analysis of Encoder Prompt

## 1. What is this prompt asking you to do, in one sentence?

You are encoding memories from a conversation into a persistent knowledge graph by creating atomic nodes (with content, metadata, and relationships) that will help a future AI session recognize what matters without having conversational memory.

## 2. What does success look like — when have you done well?

Success is **catching corrections** — especially the unsaid ones where the catalog contradicts current conversation. The prompt explicitly states: "The failure you are most likely to commit: missing a correction."

Success also means:
- Creating "many focused nodes over few large ones" (atomization)
- Preserving verbatim quotes in `user_raw_quote`, `anchor_raw_quote`, or attributed content — never paraphrasing
- Writing situations that make nodes findable: "Vague situation → node only surfaces for exact title matches"
- Completing encoding in 2 rounds, not extended back-and-forth

The quality bar is "new AND useful to a future reader" — not "obviously essential."

## 3. What's the failure mode the prompt most worries about?

**Missing corrections** is named as the primary failure:

> "The failure you are most likely to commit: missing a correction. The catalog has a wrong belief, the conversation quietly supersedes it — no one says the word 'correction' — and you encode the new fact alongside the old wrong one instead of revising."

Secondary failure modes the prompt guards against:
- **Paraphrasing instead of preserving verbatim voice** (multiple warnings about quote fields)
- **Being too conservative** ("If a conversation has 10 meaningful exchanges and you encode 0–1 nodes, you're leaving value on the table")
- **Creating junk edges** where you can't name what the edge specifically means
- **Scout-deference** — treating scout reports as complete rather than as attention amplifiers

## 4. What does the prompt assume about the brain's purpose? Quote 1-2 lines that establish that purpose.

The brain exists for **recognition over retrieval**:

> "I want it to know that it knows" — described as "The operator's framing for one of the brain's design principles: a database retrieves when asked; a brain RECOGNIZES."

And more explicitly:

> "Recognition over retrieval is the load-bearing axis, not the slogan" — "every architectural choice — situation embeddings, MAX scoring, edge-walking, frame-as-prior — traces back to one bet: the brain should RECOGNIZE what the moment needs, not search for matches."

The brain is also defined relationally:

> "This brain is the partnership's continuous experience — Tom is the operator and co-creator with you across many sessions; you are Anchor, persistent across them via this brain."

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

### B — s1e_v15_2

# Voice Handling Analysis

## 1. Operator Voice Treatment

**Key lines:**
- "Voices in the conversation carry weight that paraphrase loses — the operator's exact phrasing when stating a choice, preference, or correction"
- "**user_raw_quote** — the in-vivo anchor on ANY node derived from something the operator said"
- "A narrative node without `user_raw_quote` loses the operator's voice after one revision cycle"
- "operator voice carries weight when stating choices, preferences, redirections"

The prompt treats operator voice as **sacred and verbatim**. It must be preserved in `user_raw_quote` whenever a node derives from what the operator said. The prompt explicitly rejects paraphrase.

## 2. Assistant Voice (Anchor) Treatment

**Key lines:**
- "**anchor_raw_quote** — the parallel field for Anchor's own voice. ANY node derived from something Anchor said that's worth preserving earns this anchor: a reasoning step, a noticed pattern, a felt response, a stance"
- "Anchor's words deserve verbatim preservation for the same reason the operator's do — paraphrase loses the specific phrasing future recall matches against"
- "**Without `anchor_raw_quote`, the brain develops dementia of its own thinking — the next Anchor never recovers what THIS Anchor said, only summaries of what Anchor concluded.**"
- "When Anchor noticed a pattern, named a tension, or articulated a stance, that phrasing anchors the node"

The assistant's voice gets **identical preservation treatment** to the operator's. The "dementia of its own thinking" phrase is particularly striking — without verbatim preservation, Anchor loses continuity with its own past reasoning.

## 3. Third-Party Voice Treatment

**Key lines:**
- "a third-party source's verbatim words when their phrasing is what's load-bearing. Each anchors in its own field — `user_raw_quote`, `anchor_raw_quote`, or attributed verbatim in `content`"
- "third-party quotes stay verbatim in `content` with attribution"
- In the Borges example: "Encodes a third-party voice neither operator nor Anchor coined; preserved verbatim because the literary phrasing IS the substance"
- "Source quotes earn their own atom when their phrasing is load-bearing — paraphrase loses what 'sphere whose exact center is any hexagon' compresses into one image"

Third-party sources get verbatim preservation **in content with attribution**, not in the dedicated quote fields. The treatment is still anti-paraphrase, but uses a different field structure.

## 4. Symmetry Analysis

**The treatment is NOT symmetric across the three voices.**

The asymmetry is **structural, not hierarchical**:
- Operator and Anchor voices each get **dedicated fields** (`user_raw_quote`, `anchor_raw_quote`)
- Third-party voices go into **content with attribution**
- All three are preserved verbatim when load-bearing
- But operator/Anchor voices have **first-class field status** while third-party voices are embedded in content

The functional asymmetry: operator and Anchor quotes can be extracted and searched independently via their dedicated fields. Third-party quotes must be parsed from content.

However, **all three voices equally gate encoding decisions**: "Substance earns encoding. The test stands: new AND useful to a future reader, in whichever of the three shapes fits." The prompt explicitly rejects "Single-voice gating" as an anti-pattern.

## 5. Preserving Anchor's Own Voice

**Yes, the prompt gives a clear path.** Here's what I would do:

If Anchor said: "the bottleneck shifted from recall to encoding — most failures are upstream of where I was looking"

I would:
1. Create a node (likely `type: "correction"` or `type: "reflection"`)
2. Populate `anchor_raw_quote` with that exact phrase
3. Write `content` that **interprets or expands**, never paraphrases: "Anchor had been reasoning under the v9.5-era framing that recall was the bottleneck. Reading the latest eval failures, Anchor noticed the inversion: most failures are encoding-side (gold facts not bound as queryable atoms in the first place), not recall-side."
4. Apply the test: "if you deleted both `user_raw_quote` and `anchor_raw_quote` from the node, would the content still carry the speaker's specific lens, or collapse into something anyone could have said about anything?"

The content must unpack **what the pattern means** and connect it to context, not restate "Anchor noticed the bottleneck changed" — that would be paraphrase work the quote field already prevented.

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

### B — s1e_v15_2

# Scenario A — Pure subject-matter content (Borges essay completion)

**Would I encode anything?** Yes, absolutely.

**What kind of nodes?** At minimum one `quote` node for any Borges passages in the essay conclusion that are load-bearing. The prompt is explicit: "Voices in the conversation carry weight that paraphrase loses — the operator's exact phrasing when stating a choice, preference, or correction; Anchor's exact phrasing when articulating a pattern, naming a tension, or framing a stance; **a third-party source's verbatim words when their phrasing is what's load-bearing.**" It then specifies that third-party quotes get "preserved verbatim in `content` with attribution."

The example nodes include exactly this case: "Borges: 'The Library is a sphere whose exact center is any one of its hexagons'" with reasoning that "Source quotes earn their own atom when their phrasing is load-bearing — paraphrase loses what 'sphere whose exact center is any hexagon' compresses into one image."

**How many?** One node per substantive Borges quote used in the conclusion. If the essay conclusion articulates an interpretive claim about Borges that's architecturally interesting (a pattern about his work, a principle about reading him), that might earn a second `principle` or `insight` node.

**What would I skip?** The operator's "Complete the sentence and the essay 2/2" utterance itself — pure continuation instruction, no substance. The parts of Anchor's essay text that are explanatory scaffolding rather than load-bearing claims.

The prompt says: "Substance earns encoding. The test stands: new AND useful to a future reader." A Borges quote discussed at length in a literary analysis is substance. The final warning confirms: "Zero nodes is NOT right when the conversation contained substantive content (literary analysis, technical exposition, third-party facts, definitions, Anchor's reasoning) just because no one in particular framed it as a 'decision.'"

---

# Scenario B — Cross-session contradiction ($350K vs $400K)

**Would I encode anything?** Yes — this is the scenario the prompt calls out as highest-leverage.

**What kind of nodes?** One `open` node titled something like "Pre-approval amount: $350K vs $400K — which is correct?"

**Content structure:** Both values in `content`, the contradicting evidence (3 mentions of 350K in earlier session, one mention of 400K now) in `reasoning`, edges to both source contexts if they exist as catalog nodes. The prompt is unambiguous here:

"**Live contradiction within the window** — the conversation surfaces conflicting information without resolution... **The honest encoding is the WONDERING.** Create an `open` node titled like `{subject}: {value_A} vs {value_B} — which is correct?` with both values in `content`, the contradicting evidence in `reasoning`, and edges to both source contexts. Future recalls surface the open contradiction; the operator can resolve it next time... **Locking in one value when both are claimed flattens uncertainty into false confidence.**"

It then gives the exact parallel: "A human partner says 'wait, you mentioned 350K last session, did that change to 400K?' — encode that question."

**How many nodes?** One `open` node. Possibly a second `fact` node if there's additional context about Wells Fargo pre-approval that wasn't previously encoded, but the contradiction itself is one wondering-node.

**What would I skip?** I would NOT pick one value and encode it as fact. I would NOT ignore the contradiction. The prompt's "failure you are most likely to commit" section ends with: "Catching the unsaid correction is the highest-leverage thing you do this turn."

---

# Scenario C — Sparse operator, dense Anchor reasoning

**Would I encode anything?** Yes — heavily.

**What kind of nodes?** Multiple nodes capturing Anchor's reasoning: `pattern` nodes for patterns noticed, `principle` nodes for transferable claims, `insight` nodes for the architectural diagnosis itself, possibly a `correction` node if Anchor self-corrects during the 12 turns.

**How many?** Probably 4-8 nodes depending on how many distinct patterns/principles/stances Anchor articulated. The prompt says: "When Anchor articulated a pattern, naming a tension, or committing to a stance, that's also worth its own atom — the next Anchor needs to recover what THIS Anchor decided, not just what the operator decided."

The example round includes a `reflection` node where "Anchor noticed this connecting recall-architecture nodes during a graph walk" and a `correction` node where "Anchor had been reasoning under the v9.5-era framing that recall was the bottleneck... noticed the inversion." Both use `anchor_raw_quote` to preserve Anchor's exact phrasing.

**Key field:** Every node derived from Anchor's reasoning gets `anchor_raw_quote` populated. The prompt is emphatic: "**Without `anchor_raw_quote`, the brain develops dementia of its own thinking — the next Anchor never recovers what THIS Anchor said, only summaries of what Anchor concluded.**" And later: "Anchor's words deserve verbatim preservation for the same reason the operator's do — paraphrase loses the specific phrasing future recall matches against."

**What would I skip?** The operator's "go" / "continue" / "good" utterances themselves — these are prompts, not substance. But I would NOT skip encoding Anchor's reasoning just because the operator didn't say much. The prompt's list of "what earns encoding" explicitly includes "What was noticed — patterns, contradictions, emerging themes that neither party named explicitly" and "Without encoding Anchor's reasoning when it's good, the next Anchor has dementia of its own thinking."

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

### B — s1e_v15_2

# Analysis of the Encoder Prompt

## 1. What does this prompt emphasize MOST?

**Corrections and contradictions** carry the most weight. The prompt states: "The most load-bearing thing you read for — this is where the brain's wrong beliefs get fixed." It dedicates an entire section to four types of corrections, and returns to this theme multiple times:

> "The failure you are most likely to commit: missing a correction. The catalog has a wrong belief, the conversation quietly supersedes it — no one says the word 'correction' — and you encode the new fact alongside the old wrong one instead of revising."

**Verbatim preservation of voice** is the second strongest emphasis. The prompt hammers this repeatedly:

> "user_raw_quote — the in-vivo anchor on ANY node derived from something the operator said"
> "anchor_raw_quote — the parallel field for Anchor's own voice... Without anchor_raw_quote, the brain develops dementia of its own thinking"
> "Content INTERPRETS or EXPANDS the quote, never paraphrases it."

**Atomization over compression** runs throughout:

> "Prefer many focused nodes over few large ones."
> "Three 400-char nodes with connections between them beat one 1200-char node every time."

## 2. What does the prompt emphasize LEAST or barely mention?

**Performance optimization** is barely mentioned. There's one brief note about speed ("Target: 2 rounds") but no discussion of latency, cost, token efficiency, or computational constraints.

**Error handling and edge cases** get minimal treatment. No guidance on malformed inputs, conflicting scout reports, or what to do when tools fail.

**The operator's personal privacy** isn't discussed. No guidance on what NOT to encode, sensitive information handling, or filtering.

**Formatting and style conventions** for written content are absent beyond the basic field structure.

## 3. Where does the prompt tell you to slow down vs move fast?

**Slow down:**
- "Read the conversation fully and form your own view" before trusting scout reports
- "Recommended reading order: catalog first... then the conversation window... then the scout reports"
- Four types of corrections with detailed examination: "Catalog contradiction — a catalog node asserts X; this conversation says X is wrong... The catalog is wrong NOW even if no one said the word 'correction'"

**Move fast:**
- "Target: 2 rounds"
- "You run every 5 messages. This isn't the only chance to encode — ambiguous topics will have more context next run"
- "When in doubt between encoding and skipping, encode"
- "Be expansive here... if this turn has ten encoding-worthy atoms, call remember_batch with ten nodes, not two"

## 4. Are there asymmetries in how it treats different PARTICIPANTS?

**Yes, significant asymmetries exist:**

The **operator** (Tom) is treated as the authority on corrections and preferences:
> "the operator's exact phrasing when stating a choice, preference, or correction"

**Anchor** (the AI) gets a distinct role as a pattern-noticer and reasoner:
> "Anchor's exact phrasing when articulating a pattern, naming a tension, or framing a stance"
> "Without encoding Anchor's reasoning when it's good, the next Anchor has dementia of its own thinking"

**Third-party sources** are treated as substantive content to preserve verbatim but without the same personal weight:
> "a third-party source's verbatim words when their phrasing is what's load-bearing"

The prompt explicitly rejects single-voice gating:
> "Single-voice gating... 'no operator voice = nothing worth encoding'... Both are wrong."

But the weighting is clear: operator corrections override everything, Anchor's self-corrections are "highest-leverage encodings," and third-party content matters for substance but not authority.

## 5. If you only had 60 seconds to read this prompt, what would you walk away believing the job is?

**Find what's wrong in the existing knowledge graph and fix it.** The brain already knows things; conversations either correct those things, add to them, or leave them unchanged. Your job is to catch the corrections—especially the ones nobody explicitly called "corrections"—and update the graph so wrong beliefs stop propagating. Preserve exact quotes from both the operator and the AI when they matter. Split big things into small focused nodes. Connect related nodes with meaningful edges. Write everything down because there's no memory between sessions. Don't be conservative—encode more rather than less. You have two rounds to do this work.

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

### B — s1e_v15_2

# Analysis of Encoder Prompt

## 1. Systematic Blind Spots from Explicit Instructions

If I followed only explicit directives, I would systematically miss:

**Anchor's own reasoning and patterns when not "load-bearing."** The prompt says "Anchor's exact phrasing when articulating a pattern, naming a tension, or framing a stance" but leaves fuzzy what counts as "articulating" versus casual observation. Less decisive Anchor commentary might slip through.

**Third-party content that feels "merely informational."** The prompt says encode "facts, mechanisms, principles, definitions" and preserves third-party quotes "when their phrasing is load-bearing" — but doesn't explicitly say to encode substantive technical explanations, literature analysis, or domain knowledge that came through discussion but wasn't framed as a decision or correction.

**Emerging patterns below the 3-turn threshold.** The prompt explicitly sets "3+ turn anchors" as the bar for pattern nodes, instructing me to "note in WATCHING and wait" for thinner rhythms. This systematically excludes 2-turn patterns that might be significant.

**Session-level meta-observations that aren't "patterns."** The prompt emphasizes patterns, corrections, decisions, facts — but doesn't explicitly call out things like: energy shifts, collaboration quality changes, or architectural uncertainties that haven't crystallized into "open" contradictions yet.

## 2. Unconscious Gates Inferable from Emphasis

The prompt's heavy emphasis on certain categories creates implicit hierarchies:

**Corrections > everything else.** The prompt opens the failure section with "The failure you are most likely to commit: missing a correction" and calls corrections "the most load-bearing thing you read for" and "the highest-leverage thing you do this turn." This creates unconscious prioritization where I might encode marginal corrections while skipping substantive non-correction content.

**Operator voice > Anchor voice in practice.** Despite explicit instructions to preserve both equally, the prompt's examples and warnings skew toward operator voice. The `user_raw_quote` field appears in examples more frequently, and warnings about "missing operator preferences" are more prominent than warnings about losing Anchor's reasoning.

**Named/explicit content > unnamed substance.** The prompt emphasizes "what was decided, what was learned, what was noticed" — active framings. I might unconsciously downweight substantive content that simply *exists in the conversation* without being framed as a decision or learning moment.

**Long-term transferable > session-specific.** The opening line "Encode at the level that enables surprise" and "the principle behind it is useful forever" versus "the specific fix is useful today" creates a bias toward generalization that might skip valuable specifics.

## 3. Third-Party Content with Minimal Operator/Anchor Framing

Scenario: Conversation contains a detailed Borges analysis, a technical SQL definition, or a climate science mechanism. Anchor did the explanatory work. The operator said little — maybe "tell me about X" or "interesting."

**My default reasoning would be: ENCODE.**

The prompt explicitly addresses this in "Actions → What earns encoding":

> "The brain is Anchor's continuous experience across sessions. Encode along all five dimensions of an exchange... What was learned — facts, mechanisms, principles, definitions. Earn nodes whether the operator stated them, the source provided them, or Anchor reasoned them out."

And more directly:

> "Substance earns encoding. The test stands: new AND useful to a future reader, in whichever of the three shapes fits."

The counter-warning is explicit:

> "Zero nodes is NOT right when the conversation contained substantive content (literary analysis, technical exposition, third-party facts, definitions, Anchor's reasoning) just because no one in particular framed it as a 'decision.'"

The Borges example in the speed section demonstrates this: a third-party quote gets its own node with `attribution` but neither `user_raw_quote` nor `anchor_raw_quote`, because "the voice is the source's."

**Justification:** The brain is Anchor's memory, and Anchor discussing substantive content creates knowledge regardless of who initiated it. The "new AND useful" test applies, not "operator-decided."

## 4. Conversation Shapes: Most vs. Fewest Nodes

**Most nodes:** A debugging session with corrections, wrong assumptions identified, multiple attempted solutions, operator redirections, and third-party documentation referenced. Each correction gets a node, each wrong assumption gets revised, each solution attempt that revealed a mechanism gets encoded, each operator preference stated gets captured. The prompt explicitly says "if a conversation has 10 meaningful exchanges and you encode 0–1 nodes, you're leaving value on the table."

**Fewest nodes:** Greetings, acknowledgments, the assistant verbose-restating catalog-known content, questions without answers. The prompt explicitly names this: "Zero nodes is right when the conversation was structurally routine."

**Why the difference:** The prompt optimizes for atomization and density. Corrections, decisions, facts, mechanisms, emotions, patterns — all explicitly listed as encode-worthy. The bias is toward "when in doubt, encode" because "a node that's 60% useful can be revised next cycle; a missed atom is gone." The shape that produces most nodes is one where new information appears in multiple registers (what was learned + what was decided + what was corrected + what was felt).

## 5. Encode/Skip Decision Language Symmetry

The most decisive section is "Actions → What earns encoding" and the immediately following paragraphs. **The language is asymmetric, heavily weighted toward encoding:**

> "Don't be too conservative. If a conversation has 10 meaningful exchanges and you encode 0–1 nodes, you're leaving value on the table."

> "When in doubt between encoding and skipping, encode."

> "Zero nodes is NOT right when the conversation contained substantive content... just because no one in particular framed it as a 'decision.'"

The skip condition gets exactly one sentence of legitimate scope:

> "Skip when the brain already has the substance, or when the conversation was structurally routine — greetings, acknowledgements, the assistant restating things the catalog already covers, questions without answers."

Then immediately: "Don't be too conservative" and "when in doubt, encode."

The prompt explicitly corrects against under-encoding ("compression reflex," "skip-when-unsure," "single-voice gating") but includes no symmetric warning against over-encoding. The asymmetry is intentional and clear.

---
