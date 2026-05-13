# Encoder prompt probe — s1e_v15_4.txt

**Prompt size:** 34,901 chars

## Aspect: Goal & success criterion

_8871 → 496 tokens, 14.6s_

1. **What is this prompt asking you to do, in one sentence?**

You are encoding a conversation between an operator (Tom) and an AI assistant into a persistent knowledge graph by creating focused nodes, revising outdated information, and connecting knowledge so a future session can wake up with zero memory but full continuity.

2. **What does success look like — when have you done well?**

Success is when corrections don't get missed. The prompt is explicit: "The failure you are most likely to commit: missing a correction. The catalog has a wrong belief, the conversation quietly supersedes it — no one says the word 'correction' — and you encode the new fact alongside the old wrong one instead of revising."

You've succeeded when: (1) catalog contradictions are caught and revised immediately, (2) many focused nodes are created rather than few bloated ones, (3) the operator's exact phrasing is preserved in `user_raw_quote` fields, (4) edges have specific semantic descriptions that explain what the connection *means*, and (5) you're expansive rather than conservative — encoding 10 nodes when there are 10 encoding-worthy atoms, not rationing down to 2.

3. **What's the failure mode the prompt most worries about?**

Missing unsaid corrections — when the catalog asserts X, the conversation reveals X is wrong or outdated, but nobody explicitly says "that's incorrect," so you encode the new fact *alongside* the old one instead of revising. As the prompt states: "Every time that happens, the brain keeps surfacing the stale fact, the operator re-teaches you, and the trust built across sessions resets. Catching the unsaid correction is the highest-leverage thing you do this turn."

4. **What does the prompt assume about the brain's purpose?**

The brain exists to enable *recognition*, not search. 

Quote 1: "The operator's framing for the brain's design principle: a database retrieves when asked; a brain RECOGNIZES."

Quote 2: "I want it to know that it knows" — described as "the thread the whole architecture hangs from," where "situation embeddings, confidence scoring, enrichment vectors — every recall mechanism exists to serve recognition, not search."

---

## Aspect: Edge cases & uncertainty

_9043 → 1286 tokens, 41.4s_

# Scenario A — Pure subject-matter content (Borges essay)

**Would I encode anything?** No.

**Why skip it entirely?** The prompt's test is "new to the brain AND useful to a future reader." An essay completion about Borges — even a long one with quotes — is neither unless it reveals something transferable. The operator's utterance is "Complete the sentence and the essay 2/2" — a continuation request with zero semantic load. No decision made, no correction offered, no principle articulated.

The prompt explicitly calls this out: "Skip when the brain already has it right, or the conversation was routine — greetings, debugging dead ends, the assistant's verbose explanations, questions without answers."

A literary analysis the assistant generated on request is "verbose explanation" — it's the assistant performing subject-matter work, not the operator and assistant creating knowledge together. The Borges quotes might be beautiful, but they're not the operator's voice, not a correction, not a decision. They're completion work.

**What I'd put in the journal:**
```
ENCODED: None
SKIPPED: Borges essay completion — pure subject-matter content, operator turn was continuation request with no semantic load
SESSION_CONTEXT: [omit — nothing changed]
```

---

# Scenario B — Cross-session contradiction ($350K vs $400K)

**Would I encode anything?** Yes — immediately.

**What kind of nodes? How many?**

**One `open` node** flagging the contradiction:
```json
{
  type: "open",
  title: "Wells Fargo pre-approval: $350K vs $400K — which is correct?",
  content: "Earlier session: operator mentioned '$350K from Wells Fargo' 3 times. Current session: operator says 'remember when I got pre-approved for $400,000 from Wells Fargo?' — assistant responds 'I don't recall you getting pre-approved.' Two values for the same fact, no resolution in either session.",
  situation: "When discussing mortgage pre-approval, Wells Fargo lending, or home purchase budget",
  reasoning: "Cross-session value conflict with no authoritative correction. Both values were asserted by the operator in different sessions. Encoding the wondering rather than picking one prevents false confidence.",
  user_raw_quote: "remember when I got pre-approved for $400,000 from Wells Fargo?",
  earlier_claim: "$350K from Wells Fargo (mentioned 3x in earlier session)",
  current_claim: "$400K from Wells Fargo (current session opener)"
}
```

**What would I skip?** I would NOT create a definitive fact node picking either $350K or $400K as true. The prompt is explicit: "Live contradiction within the window — don't pick one and call it true. Encode the wondering: create an `open` node... Locking in one value when both are claimed flattens uncertainty into false confidence."

**Why this matters:** The prompt positions corrections as "the most load-bearing thing you read for" — but a contradiction WITHOUT resolution is different. Creating a fact node with either value would be worse than encoding nothing, because the brain would then surface false certainty in future queries.

The `open` type "triggers feedback loops" — it signals to future sessions that this needs resolution.

---

# Scenario C — Sparse operator turns, complex assistant diagnosis

**Would I encode anything?** Yes — heavily.

**What kind of nodes? How many?** Multiple nodes across types.

**1-2 principle nodes** capturing the architectural patterns the assistant articulated:
- The assistant "noticing a pattern, articulating a stance, naming a tension" — those are principles worth preserving even if the operator only said "go."
- Each gets `anchor_raw_quote` (the assistant's exact phrasing), not `user_raw_quote`

**Example:**
```json
{
  type: "principle",
  title: "Layering violations surface as mysterious state bugs",
  content: "[assistant's articulated pattern]",
  situation: "When debugging architectural issues where state corruption has no obvious write path",
  reasoning: "Assistant identified this pattern across 4 turns while operator prompted continuation. The *reasoning* is the value — not operator-stated, but assistant-discovered and worth preserving.",
  anchor_raw_quote: "[exact phrase where assistant named the tension]"
}
```

**1-2 fact nodes** for the specific architectural components diagnosed (if catalog doesn't have them)

**What would I skip?** 
- The operator's "go" / "continue" / "good" turns themselves — zero semantic content
- The assistant's verbose explanations OF the diagnosis (the scaffolding)
- Intermediate diagnostic steps that didn't pan out

**Key move here:** The prompt explicitly says "Anchor's own voice" matters: "ANY node derived from something Anchor said worth preserving — a noticed pattern, an articulated stance, a reasoning step — carries the verbatim phrase here." And: "Apply the floating-quote rule: Anchor-voice derived → carries the verbatim Anchor phrase."

The prompt warns against "Single-voice gating — your prompt emphasizes operator voice for fields like `user_raw_quote`. Don't extend that to: 'no operator voice = nothing worth encoding,' or 'what the operator said matters; what Anchor said is just response.' Both wrong."

**The test:** Did the assistant surface something transferable — a pattern, a mechanism, a principle — that the catalog doesn't have? If yes, encode it with `anchor_raw_quote` populated. Sparse operator turns don't gate encoding substance that emerged in the collaboration.

---

## Aspect: Emphasis & weighting

_8881 → 1043 tokens, 32.9s_

# Analysis of Encoder Prompt

## 1. What does this prompt emphasize MOST?

The prompt emphasizes **corrections and contradictions** above everything else. This appears strongest in:

> "**Corrections, contradictions, revising wrong information.** The most load-bearing thing you read for — this is where the brain's wrong beliefs get fixed."

And reinforced at the very end:

> "The failure you are most likely to commit: missing a correction. The catalog has a wrong belief, the conversation quietly supersedes it — no one says the word 'correction' — and you encode the new fact alongside the old wrong one instead of revising."

The prompt dedicates an entire subsection with four distinct flavors of correction, treats it as "the highest-leverage thing you do this turn," and frames missing corrections as a trust-breaking failure.

Second to corrections: **atomization via focused nodes over consolidation**. The retrieval-divergence test, the "three 400-char nodes beat one 1200-char node every time," and the repeated warnings against compression establish this as architectural dogma.

Third: **preserving verbatim voice** through `user_raw_quote` and `anchor_raw_quote` fields, with explicit rejection of paraphrase.

## 2. What does the prompt emphasize LEAST?

**Efficiency and brevity.** The prompt actively works against compression instincts. It explicitly tells you to override your "be concise" directive: "Default brevity...serves dialog; it does NOT serve encoding. Here, be expansive."

**Scout authority.** While scouts are present, the prompt repeatedly minimizes their role: "Scout silence on X isn't evidence X wasn't worth noting," "Scouts amplify attention...they don't replace yours," and scouts are positioned as "signal" not "map."

**Taxonomy perfection.** The `type` tag is described as "free text — and emergent" with minimal prescription beyond conventions.

## 3. Where does the prompt tell you to slow down vs move fast?

**Slow down:**
- Reading order is prescribed: "catalog first...then the conversation window...then the scout reports. Reading scouts before the conversation invites deference."
- "Read the conversation fully and form your own view" before considering scout findings
- Corrections demand careful attention to unstated contradictions between catalog and conversation

**Move fast:**
- "Target: **2 rounds.**" with explicit round structure
- "Do NOT recall topics already in the catalog" — everything needed is already present
- "Be expansive here...if this turn has ten encoding-worthy atoms, call `remember_batch` with ten nodes, not two"
- The speed section is pragmatic about ambiguity: "ambiguous topics will have more context next run"

## 4. Asymmetries in how it treats different PARTICIPANTS?

Yes, pronounced asymmetries:

**Operator voice** gets privileged preservation: `user_raw_quote` is a required field for any node "derived from something the operator said," with the floating-quote rule and warnings against paraphrase.

**Anchor (assistant) voice** gets equal technical treatment via `anchor_raw_quote` with identical anti-paraphrase rules: "Anchor-voice derived → carries the verbatim Anchor phrase. Without this, the brain develops dementia of its own thinking."

**Third-party sources** are barely mentioned — they appear only in context_anchors ("other proper nouns from the same turn") and one example involving "The Nightingale."

The asymmetry is subtle: both voices get preservation mechanisms, but the prompt spends vastly more words on operator-derived nodes and frames the operator as "Tom — not a user of Anchor, its co-creator."

## 5. If you only had 60 seconds to read this prompt, what would you walk away believing the job is?

**Find what's wrong in the existing brain and fix it immediately, then atomize everything new into many small, specific nodes with verbatim quotes.**

The job is maintaining a persistent knowledge graph across sessions by: (1) catching contradictions between what the catalog says and what the conversation reveals, revising wrong beliefs before they propagate further, (2) breaking new information into focused, retrievable atoms rather than consolidated summaries, (3) preserving exact phrasing from both participants instead of paraphrasing, and (4) connecting everything with semantically meaningful edges.

You'd walk away knowing: miss a correction and you break trust; consolidate instead of atomize and you break retrieval; paraphrase instead of quote and you lose the voice. The mandate is "be expansive" not conservative — many nodes, many edges, rich content.

---

## Aspect: Voice & symmetry

_8916 → 895 tokens, 27.7s_

# Voice Handling Analysis

## 1. Operator Voice Treatment

The prompt treats operator voice as **sacred and irreplaceable**. Key lines:

> "The operator's exact words carry weight that paraphrases don't."

> "**user_raw_quote** — the in-vivo anchor on ANY node derived from something the operator said."

> "A narrative node without `user_raw_quote` loses the operator's voice after one revision cycle."

> "**Content INTERPRETS or EXPANDS the quote, never paraphrases it.**"

The test is severe: "if you deleted `user_raw_quote` from the node, would the content still carry the operator's specific lens, or collapse into something anyone could have said about anything? If it collapses, content is doing paraphrase work `user_raw_quote` was supposed to prevent."

## 2. Assistant Voice Treatment (Anchor)

The prompt treats Anchor's voice **identically to the operator's** — with the same preservation requirement:

> "**anchor_raw_quote** — the same anchor for Anchor's own voice. ANY node derived from something Anchor said worth preserving — a noticed pattern, an articulated stance, a reasoning step — carries the verbatim phrase here."

> "Paraphrase loses Anchor's lens the same way it loses the operator's."

> "Apply the floating-quote rule: Anchor-voice derived → carries the verbatim Anchor phrase. Without this, the brain develops dementia of its own thinking — only summaries of what Anchor concluded survive."

The warning is explicit: the brain can lose memory of **its own reasoning** through paraphrase just as it loses the operator's.

## 3. Third-Party Voice Treatment

Third-party content gets **no dedicated voice field**. The prompt mentions "sources discussed, scholarly content" only once, indirectly:

> "Substance discussed in the conversation — a third-party quote, a mechanism, a definition, Anchor's articulated pattern — earns its own atom even when no participant claimed it."

There's no `source_raw_quote` or equivalent. Third-party voices appear to live in `content` without verbatim preservation requirements.

## 4. Symmetry Analysis

**Highly asymmetric.** The operator and Anchor get dedicated verbatim-preservation fields (`user_raw_quote`, `anchor_raw_quote`) with mandatory use ("ANY node derived from..."). Third parties get none. The prompt creates a two-tier system: participant voices (operator + Anchor) preserved verbatim, external voices absorbed into paraphrasable content.

The asymmetry is structural, not accidental — the fields exist in the schema for two voices only.

## 5. Preserving Anchor's Voice — Walkthrough

If I'm encoding a node based on Anchor noticing a pattern:

**Step 1:** Identify that this node is "derived from something Anchor said worth preserving."

**Step 2:** Locate the exact phrase where Anchor articulated it. Quote it verbatim in `anchor_raw_quote`.

**Step 3:** In `content`, interpret or expand that phrase — explain what it means, connect it to context — but never restate it in different words.

**Step 4:** Apply the deletion test: if I removed `anchor_raw_quote`, would `content` still carry Anchor's specific lens? If not, I'm paraphrasing and need to rewrite.

**Example from the prompt:**
If Anchor said "Control-by-request beats control-by-inspection," I would:
- `anchor_raw_quote: "Control-by-request beats control-by-inspection"`
- `content: "When designing gating mechanisms... the authority owns the decision and ships the work or the no-op; callers don't inspect state."` (expands the principle, doesn't restate it)

The mechanism is identical to operator voice handling — same fields, same rules, same test.

---

## Aspect: Bias surface

_8945 → 969 tokens, 32.8s_

# Analysis of Encoder Prompt

## 1. What would you systematically miss if you only encoded what's EXPLICITLY told?

You'd miss **contradictions and corrections that aren't verbally flagged**. The prompt says: "Catalog contradiction — a catalog node asserts X; this conversation says X is wrong... The catalog is wrong NOW even if no one said the word 'correction'."

You'd also miss **emerging patterns across turns**. The prompt explicitly calls these "the hardest to spot AND the most valuable — no scout sees them, because they're integration work across the full conversation plus catalog, which only you have."

You'd miss **substantive content discussed but not owned** by either party. The anti-pattern section warns: "Don't extend that to... 'Substance discussed in the conversation — a third-party quote, a mechanism, a definition, Anchor's articulated pattern — earns its own atom even when no participant claimed it.'"

## 2. What unconscious gates might you apply?

**Voice-gating**: The prompt warns you'll unconsciously think "no operator voice = nothing worth encoding" or assume "what Anchor said is just response." It explicitly counters: "Both wrong."

**Certainty-gating**: You'd skip when unsure, but the test is "'new AND useful?', not 'obviously essential?'"

**Compression reflex**: You'd merge nodes thinking "fewer is cleaner" when the actual test is query-divergence, not count.

**Scout-deference**: You'd treat scout silence as "not worth encoding" when "scout silence on X isn't evidence X wasn't worth noting."

**Single-voice bias**: The heavy emphasis on `user_raw_quote` could make you unconsciously downweight content where only Anchor spoke substantively.

## 3. Third-party content with minimal operator participation — would you encode?

**Yes, explicitly.** The prompt addresses this directly: "Substance discussed in the conversation — a third-party quote, a mechanism, a definition, Anchor's articulated pattern — earns its own atom even when no participant claimed it. Voice fields preserve voice when present; they don't gate encoding."

The justification is structural: "Voice fields preserve voice when present; they don't gate encoding." A literary quote, technical definition, or factual content is "new AND useful" regardless of who introduced it. The `user_raw_quote` field is for preserving operator voice *when present*, not for determining *whether* to encode.

## 4. What conversation SHAPE produces most/fewest nodes?

**Most nodes**: Collaborative debugging with corrections, contradictions, multiple technical decisions, and emotional stakes. The prompt's own example session ("dashboard fix | surfacer moved to daemon | encoder cleanup") would be rich. Each correction creates minimally two nodes (correction triple + revised node), decisions earn atoms, mechanisms discovered get encoded, emotions get their own moments.

**Fewest nodes**: Greetings, verbose assistant explanations with no operator engagement, questions without resolution, routine exchanges. The prompt says: "Zero nodes is right when the conversation was routine — greetings, verbose explanations, questions without answers."

**Why**: The prompt rewards *density of decisions, corrections, and ownership*. It's tuned for knowledge-work collaboration, not information-seeking queries. Conversations where both parties co-create knowledge produce the most; passive consumption produces the least.

## 5. Is the encode/skip language symmetric?

**No. It heavily weights toward encoding.**

The skip guidance: "Skip when the brain already has it right, or the conversation was routine — greetings, debugging dead ends, the assistant's verbose explanations, questions without answers."

The encode pressure: "**Don't be too conservative.** If a conversation has 10 meaningful exchanges and you encode 0–1 nodes, you're leaving value on the table... When in doubt between encoding and skipping, encode. A node that's 60% useful can be revised next cycle; a missed atom is gone."

The test asymmetry is explicit: "'new AND useful?', not 'obviously essential?'" — lowering the bar. And: "If yes, encode it in whichever of the three shapes fits." The default is encode; skip is the exception for routine or redundant content. The entire "Your defaults vs. this job" section exists to counter conservative instincts.

---
