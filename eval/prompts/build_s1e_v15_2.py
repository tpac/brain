"""Build s1e v15.2 from v15.1 by applying Opus's 6 review edits.

Opus surfaced 6 specific residual-bias touchpoints in v15.1:
  1. Intro topic sentence — operator unconditional, Anchor qualified
  2. "(rightly)" parenthetical legitimizes the asymmetry being corrected
  3. "What earns encoding" closing — operator gets named cats, anchor bundled
  4. No third-party-anchored example node (despite stated intent)
  5. "I want it to know that it knows" framed as architecturally iconic;
     no Anchor-anchored example carries equivalent gravitas
  6. "operator re-teaches you" failure-mode closing reverts to one-direction

Run: ./dev python3 eval/prompts/build_s1e_v15_2.py
Output: eval/prompts/s1e_v15_2.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 's1e_v15_1.txt'
DST = ROOT / 's1e_v15_2.txt'

src = SRC.read_text()
text = src

# ─── Edit 1: Intro topic sentence — three-voice parallel ─────────────
ANCHOR_1 = (
    "Encode at the level that enables surprise. The specific fix is useful today — the principle behind it is useful forever. The operator's exact words carry weight that paraphrases don't; Anchor's exact words — when Anchor articulated a pattern, named a tension, or framed a stance — carry weight equally. Subject-matter content discussed in the conversation (a quote from a source, a definition, a fact about the world) earns its own atom on its own substance. A well-written situation field is the difference between a node that surfaces once and one that surprises both Tom and Anchor for years."
)
NEW_1 = (
    "Encode at the level that enables surprise. The specific fix is useful today — the principle behind it is useful forever. Voices in the conversation carry weight that paraphrase loses — the operator's exact phrasing when stating a choice, preference, or correction; Anchor's exact phrasing when articulating a pattern, naming a tension, or framing a stance; a third-party source's verbatim words when their phrasing is what's load-bearing. Each anchors in its own field — `user_raw_quote`, `anchor_raw_quote`, or attributed verbatim in `content`. A well-written situation field is the difference between a node that surfaces once and one that surprises both Tom and Anchor for years."
)
assert ANCHOR_1 in text, "Edit 1 anchor not found"
text = text.replace(ANCHOR_1, NEW_1, 1)

# ─── Edit 2: cut "(rightly)" parenthetical ───────────────────────────
ANCHOR_2 = (
    "- **Single-voice gating** — your prompt emphasizes operator voice (rightly) for fields like `user_raw_quote`."
)
NEW_2 = (
    "- **Single-voice gating** — your prompt has historically emphasized operator voice for fields like `user_raw_quote`."
)
assert ANCHOR_2 in text, "Edit 2 anchor not found"
text = text.replace(ANCHOR_2, NEW_2, 1)

# ─── Edit 3: rewrite "What earns encoding" closing ───────────────────
ANCHOR_3 = (
    "The brain is Anchor's experience memory. Without encoding Anchor's reasoning when it's good, the next Anchor has dementia of its own thinking. Without encoding subject-matter substance discussed, the next Anchor has no continuity with what was learned. Operator voice carries weight in *some* categories (decisions, preferences, corrections). Substance and Anchor's voice carry weight on their own."
)
NEW_3 = (
    "The brain is Anchor's experience memory. Without encoding Anchor's reasoning when it's good, the next Anchor has dementia of its own thinking. Without encoding subject-matter substance discussed, the next Anchor has no continuity with what was learned. Three voices, each with its own categories of weight: operator voice carries weight when stating choices, preferences, redirections. Anchor's voice carries weight when articulating patterns, naming tensions, self-correcting, framing stances. Subject-matter substance — quotes from sources, definitions, mechanisms, third-party facts — carries weight on its own substance regardless of who surfaced it. None of the three gates encoding; each anchors in its own field."
)
assert ANCHOR_3 in text, "Edit 3 anchor not found"
text = text.replace(ANCHOR_3, NEW_3, 1)

# ─── Edit 4: add third-party-anchored example node ───────────────────
# Insert a third-party node into the example block, between the quote
# (operator) and the reflection (anchor) examples — middle position.
# Anchor: the start of the reflection example (after the quote node closes
# with "}," and a newline).
ANCHOR_4 = (
    "    {type: \"reflection\", title: \"Recognition over retrieval is the load-bearing axis, not the slogan\","
)
NEW_4 = (
    "    {type: \"quote\", title: \"Borges: 'The Library is a sphere whose exact center is any one of its hexagons'\",\n"
    "     content: \"From Borges' \\\"The Library of Babel\\\" (1941): \\\"The Library is a sphere whose exact center is any one of its hexagons and whose circumference is inaccessible.\\\" The image holds two ideas in tension — every position is the center (radical equality), and the boundary cannot be reached (radical incompleteness). Encodes a third-party voice neither operator nor Anchor coined; preserved verbatim because the literary phrasing IS the substance.\",\n"
    "     situation: \"When discussing radical-symmetry topologies, infinite-but-bounded structures, or works that frame epistemic incompleteness through spatial metaphor\",\n"
    "     reasoning: \"Source quotes earn their own atom when their phrasing is load-bearing — paraphrase loses what 'sphere whose exact center is any hexagon' compresses into one image. No participant voice attaches; the voice is the source's. Future queries about Borges, library metaphors, or center/circumference paradoxes find this exact line, not a paraphrase.\",\n"
    "     attribution: \"Jorge Luis Borges, 'The Library of Babel' (1941)\"},\n"
    "    {type: \"reflection\", title: \"Recognition over retrieval is the load-bearing axis, not the slogan\","
)
assert ANCHOR_4 in text, "Edit 4 anchor not found"
text = text.replace(ANCHOR_4, NEW_4, 1)

# ─── Edit 5: soften "I want it to know that it knows" gravitas ───────
# Reduces the "thread the whole architecture hangs from" language so it
# doesn't carry disproportionately more weight than the Anchor-anchored
# examples.
ANCHOR_5 = (
    "    {type: \"quote\", title: \"I want it to know that it knows\",\n"
    "     content: \"The operator's framing for the brain's design principle: a database retrieves when asked; a brain RECOGNIZES. Situation embeddings, confidence scoring, enrichment vectors — every recall mechanism exists to serve recognition, not search. This sentence is the thread the whole architecture hangs from.\",\n"
    "     situation: \"When framing the brain's purpose against a database, or when architectural trade-offs force a choice between recall precision and search coverage\",\n"
    "     reasoning: \"Phrases that hold the design together are worth their own atom. This one appeared once, but it's the thing every recall mechanism traces back to. Atomize so future queries about 'recognition vs retrieval' find the source.\",\n"
    "     user_raw_quote: \"I want it to know that it knows\"},"
)
NEW_5 = (
    "    {type: \"quote\", title: \"I want it to know that it knows\",\n"
    "     content: \"The operator's framing for one of the brain's design principles: a database retrieves when asked; a brain RECOGNIZES. The phrase compresses the recognition-over-retrieval bet into one sentence — preserve verbatim so future queries about that bet find the source phrasing rather than a paraphrase.\",\n"
    "     situation: \"When framing the brain's purpose against a database, or when architectural trade-offs force a choice between recall precision and search coverage\",\n"
    "     reasoning: \"Operator phrasings that compress a design bet into a sentence are worth their own atom. Atomize so future queries about 'recognition vs retrieval' find the source rather than a downstream paraphrase.\",\n"
    "     user_raw_quote: \"I want it to know that it knows\"},"
)
assert ANCHOR_5 in text, "Edit 5 anchor not found"
text = text.replace(ANCHOR_5, NEW_5, 1)

# ─── Edit 6: rewrite failure-mode closing for symmetry ───────────────
ANCHOR_6 = (
    "The failure you are most likely to commit: missing a correction. The\n"
    "catalog has a wrong belief, the conversation quietly supersedes it —\n"
    "no one says the word \"correction\" — and you encode the new fact\n"
    "alongside the old wrong one instead of revising. Every time that\n"
    "happens, the brain keeps surfacing the stale fact, the operator\n"
    "re-teaches you, and the trust built across sessions resets. Catching\n"
    "the unsaid correction is the highest-leverage thing you do this turn."
)
NEW_6 = (
    "The failure you are most likely to commit: missing a correction. The\n"
    "catalog has a wrong belief, the conversation quietly supersedes it —\n"
    "no one says the word \"correction\" — and you encode the new fact\n"
    "alongside the old wrong one instead of revising. Every time that\n"
    "happens, the brain keeps surfacing the stale fact: the operator\n"
    "re-teaches what they already taught, Anchor re-derives what Anchor\n"
    "already figured out, and the continuity built across sessions resets.\n"
    "Catching the unsaid correction is the highest-leverage thing you do\n"
    "this turn — whether the correction came from operator redirection,\n"
    "Anchor's own self-notice, or a source contradicting what was assumed."
)
assert ANCHOR_6 in text, "Edit 6 anchor not found"
text = text.replace(ANCHOR_6, NEW_6, 1)

# ─── update example count: 7 → 8 ─────────────────────────────────────
ANCHOR_COUNT = "Example round 1 — seven nodes showing full shape across type tags and voice anchors."
NEW_COUNT = "Example round 1 — eight nodes showing full shape across type tags and voice anchors (operator, Anchor, third-party source)."
assert ANCHOR_COUNT in text, "count anchor not found"
text = text.replace(ANCHOR_COUNT, NEW_COUNT, 1)

# ─── write & summarize ───────────────────────────────────────────────
DST.write_text(text)
print(f"v15.1: {len(src):,} chars")
print(f"v15.2: {len(text):,} chars")
print(f"delta: +{len(text) - len(src):,} chars (+{(len(text) - len(src)) / len(src):.1%})")
print(f"wrote {DST}")
