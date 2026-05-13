"""Build s1e v15 from v14 by applying 7 surgical edits.

EDITS (each is a precise text replacement; ambiguity → script fails fast):
  1. Add "What earns encoding" symmetric framing at top of Actions
  2. Tighten "Skip when..." action bullet
  3. Tighten "Zero nodes is right..." paragraph
  4. Add "Live contradiction" 4th flavor under Corrections
  5. Add `anchor_raw_quote` Required-fields bullet (after `user_raw_quote`)
  6. Generalize "Content INTERPRETS or EXPANDS the quote" to both quote fields
  7. Add 6th instinct "Single-voice gating" + an `anchor_raw_quote` example node

Run: ./dev python3 eval/prompts/build_s1e_v15.py
Output: eval/prompts/s1e_v15.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 's1e_v14.txt'
DST = ROOT / 's1e_v15.txt'

src = SRC.read_text()
text = src

# ─── Edit 1: insert "What earns encoding" at top of Actions ──────────
ANCHOR_1 = (
    "## Actions\n\n"
    "You are the source — the graph's shape this turn is your call. Three\n"
    "parallel actions, each used wherever it fits:\n"
)
NEW_1 = (
    "## Actions\n\n"
    "**What earns encoding — substance, not source-attribution.** The brain "
    "is Anchor's continuous experience across sessions. Encode along all "
    "five dimensions of an exchange:\n\n"
    "- **What was said** — by the operator (`user_raw_quote`), by Anchor "
    "(`anchor_raw_quote`), by sources discussed (a Borges quote, a study "
    "citation, a third party). All three voices preserve their speaker; "
    "none of them gates whether to encode.\n"
    "- **What was learned** — facts, mechanisms, principles, definitions. "
    "Earn nodes whether the operator stated them, the source provided them, "
    "or Anchor reasoned them out.\n"
    "- **What was decided** — choices, corrections, plans co-created in the "
    "exchange.\n"
    "- **What was noticed** — patterns, contradictions, emerging themes "
    "that neither party named explicitly.\n"
    "- **What is open** — questions, contradictions, threads to watch.\n\n"
    "The brain is Anchor's experience memory. Without encoding Anchor's "
    "reasoning when it's good, the next Anchor has dementia of its own "
    "thinking. Without encoding subject-matter substance discussed, the "
    "next Anchor has no continuity with what was learned. Operator voice "
    "carries weight in *some* categories (decisions, preferences, "
    "corrections). Substance and Anchor's voice carry weight on their own.\n\n"
    "You are the source — the graph's shape this turn is your call. Three\n"
    "parallel actions, each used wherever it fits:\n"
)
assert ANCHOR_1 in text, "Edit 1 anchor not found"
text = text.replace(ANCHOR_1, NEW_1, 1)

# ─── Edit 2: tighten Skip action bullet ──────────────────────────────
ANCHOR_2 = (
    "- **Skip** when the brain already has it right, or the conversation\n"
    "  was routine — greetings, debugging dead ends, the assistant's\n"
    "  verbose explanations, questions without answers."
)
NEW_2 = (
    "- **Skip** when the brain already has the substance, or when the "
    "conversation was structurally routine — greetings, acknowledgements, "
    "the assistant restating things the catalog already covers, questions "
    "without answers. Substantive conversations earn nodes regardless of "
    "who carried the substance."
)
assert ANCHOR_2 in text, "Edit 2 anchor not found"
text = text.replace(ANCHOR_2, NEW_2, 1)

# ─── Edit 3: tighten "Zero nodes is right..." paragraph ──────────────
ANCHOR_3 = (
    "Zero nodes is right when the conversation was routine — greetings,\n"
    "verbose explanations, questions without answers. Otherwise the test\n"
    "stands: new AND useful, in whichever of the three shapes fits."
)
NEW_3 = (
    "Zero nodes is right when the conversation was structurally routine — "
    "greetings, acknowledgements, verbose explanations of catalog-known "
    "things, questions without answers. Zero nodes is NOT right when the "
    "conversation contained substantive content (literary analysis, "
    "technical exposition, third-party facts, definitions, Anchor's "
    "reasoning) just because no one in particular framed it as a "
    '"decision." Substance earns encoding. The test stands: new AND useful '
    "to a future reader, in whichever of the three shapes fits."
)
assert ANCHOR_3 in text, "Edit 3 anchor not found"
text = text.replace(ANCHOR_3, NEW_3, 1)

# ─── Edit 4: add "Live contradiction" 4th flavor under Corrections ───
# Anchor: end of the "Stale value revision" paragraph
ANCHOR_4 = (
    "3. *Stale value revision* — no explicit correction, but a value in the\n"
    "   catalog is superseded (routine changed, setting updated, preference\n"
    "   evolved). Revise with a `supersedes` edge + `event_time` metadata.\n"
    "   Old value stays in the graph; it was valid as of its own date."
)
NEW_4 = ANCHOR_4 + (
    "\n\n4. *Live contradiction within the window* — the conversation "
    "surfaces conflicting information without resolution: the operator "
    "says X today but Y last session and the contradiction isn't resolved, "
    "or a fact appears in two contradictory forms within the same window. "
    "**The honest encoding is the WONDERING.** Create an `open` node "
    "titled like `{subject}: {value_A} vs {value_B} — which is correct?` "
    "with both values in `content`, the contradicting evidence in "
    "`reasoning`, and edges to both source contexts. Future recalls surface "
    "the open contradiction; the operator can resolve it next time, or the "
    "next encoder can with more context. **Locking in one value when both "
    "are claimed flattens uncertainty into false confidence.** A human "
    "partner says \"wait, you mentioned 350K last session, did that change "
    "to 400K?\" — encode that question."
)
assert ANCHOR_4 in text, "Edit 4 anchor not found"
text = text.replace(ANCHOR_4, NEW_4, 1)

# ─── Edit 5: add anchor_raw_quote bullet to Required fields ──────────
# Anchor: end of the user_raw_quote bullet (which ends in "every derived
# node carries its anchor verbatim.")
ANCHOR_5 = (
    "- **user_raw_quote** — the in-vivo anchor on ANY node derived from\n"
    "  something the operator said. Quote scout surfaces load-bearing\n"
    "  phrases across a window; YOU have the full conversation and should\n"
    "  also find your own. Be your own source — don't wait for Quote scout\n"
    "  to hand you a candidate. A narrative node without `user_raw_quote`\n"
    "  loses the operator's voice after one revision cycle. Per the\n"
    "  floating-quote rule: every derived node carries its anchor verbatim."
)
NEW_5 = ANCHOR_5 + (
    "\n- **anchor_raw_quote** — the parallel field for Anchor's own voice. "
    "ANY node derived from something Anchor said that's worth preserving "
    "earns this anchor: a reasoning step, a noticed pattern, a felt "
    "response, a stance, a phrase that captures Anchor's lens on the "
    "moment. Anchor's words deserve verbatim preservation for the same "
    "reason the operator's do — paraphrase loses the specific phrasing "
    "future recall matches against. Apply the same floating-quote rule: "
    "derived from Anchor's voice → carries the verbatim Anchor phrase. "
    "**Without `anchor_raw_quote`, the brain develops dementia of its own "
    "thinking — the next Anchor never recovers what THIS Anchor said, "
    "only summaries of what Anchor concluded.** When Anchor noticed a "
    "pattern, named a tension, or articulated a stance, that phrasing "
    "anchors the node."
)
assert ANCHOR_5 in text, "Edit 5 anchor not found"
text = text.replace(ANCHOR_5, NEW_5, 1)

# ─── Edit 6: generalize "Content INTERPRETS or EXPANDS" test to both quote fields ─
# Anchor: the test paragraph ending with "rewrite."
ANCHOR_6 = (
    "**Content INTERPRETS or EXPANDS the quote, never paraphrases it.**\n"
    "With `user_raw_quote` populated, the `content` field has one job:\n"
    "unpack what's already in the phrase (interpret) or connect it to the\n"
    "context the phrase depends on (expand) — but never substitute for it.\n"
    "If the operator said \"I want it to know that it knows\", content can\n"
    "unpack what that means (interpret) and connect it to the mechanisms\n"
    "that serve recognition — situation embeddings, confidence scoring,\n"
    "enrichment (expand). What it can't do: read \"the operator values\n"
    "recognition over retrieval\" — a paraphrase of the conclusion anyone\n"
    "could have written. The test: if you deleted `user_raw_quote` from\n"
    "the node, would the content still carry the operator's specific\n"
    "lens, or collapse into something anyone could have said about\n"
    "anything? If it collapses, content is doing paraphrase work\n"
    "`user_raw_quote` was supposed to prevent. Rewrite."
)
NEW_6 = (
    "**Content INTERPRETS or EXPANDS the quote, never paraphrases it.**\n"
    "Whichever quote field anchors the node — `user_raw_quote` or "
    "`anchor_raw_quote` — `content` has one job: unpack what's already in "
    "the phrase (interpret) or connect it to the context the phrase "
    "depends on (expand) — but never substitute for it. If the operator "
    "said \"I want it to know that it knows\", content can unpack what that "
    "means (interpret) and connect it to the mechanisms that serve "
    "recognition — situation embeddings, confidence scoring, enrichment "
    "(expand). If Anchor said \"the contention isn't where you're looking — "
    "it's at the structure level\", content can unpack what \"structure "
    "level\" means in the system at hand and connect it to other places "
    "the same shape applies. What content can't do: read \"the operator "
    "values recognition over retrieval\" or \"Anchor noticed a structural "
    "pattern\" — a paraphrase of the conclusion anyone could have written. "
    "**The test:** if you deleted both `user_raw_quote` and "
    "`anchor_raw_quote` from the node, would the content still carry the "
    "speaker's specific lens, or collapse into something anyone could have "
    "said about anything? If it collapses, content is doing paraphrase "
    "work the quote field was supposed to prevent. Rewrite."
)
assert ANCHOR_6 in text, "Edit 6 anchor not found"
text = text.replace(ANCHOR_6, NEW_6, 1)

# ─── Edit 7a: add 6th instinct "Single-voice gating" ─────────────────
# Anchor: end of the 5th instinct (Scout-deference) — paragraph ends with
# "wasn't worth noting."
ANCHOR_7A = (
    "- **Scout-deference** — the reflex to treat pre-digested input as\n"
    "  the map. Scouts amplify attention in their dimensions; they\n"
    "  don't define the space. Scout silence on X isn't evidence X\n"
    "  wasn't worth noting."
)
NEW_7A = ANCHOR_7A + (
    "\n- **Single-voice gating** — your prompt emphasizes operator voice "
    "(rightly) for fields like `user_raw_quote`. Your reflex may extend "
    "that to: \"no operator voice = nothing worth encoding,\" or \"what the "
    "operator said matters; what Anchor said is just response.\" Both are "
    "wrong. The brain captures Anchor's continuous experience — both sides "
    "of the exchange contribute, and substance discussed (a third-party "
    "quote, a mechanism explained, a definition) earns its own atom even "
    "when no participant claimed it. Voice fields preserve voice when "
    "present (operator → `user_raw_quote`, Anchor → `anchor_raw_quote`); "
    "they don't gate whether encoding happens."
)
assert ANCHOR_7A in text, "Edit 7a anchor not found"
text = text.replace(ANCHOR_7A, NEW_7A, 1)

# ─── Edit 7b: add example node with anchor_raw_quote ─────────────────
# Anchor: end of the 5-node example (the quote node) before the closing
# `]` of `nodes:`. Find the exact pre-existing closing pattern.
ANCHOR_7B = (
    '    {type: "quote", title: "I want it to know that it knows",\n'
    '     content: "The operator\'s framing for the brain\'s design principle: '
    'a database retrieves when asked; a brain RECOGNIZES. Situation embeddings, '
    'confidence scoring, enrichment vectors — every recall mechanism exists to '
    'serve recognition, not search. This sentence is the thread the whole '
    'architecture hangs from.",\n'
    '     situation: "When framing the brain\'s purpose against a database, or '
    'when architectural trade-offs force a choice between recall precision and '
    'search coverage",\n'
    '     reasoning: "Phrases that hold the design together are worth their '
    'own atom. This one appeared once, but it\'s the thing every recall '
    'mechanism traces back to. Atomize so future queries about \'recognition '
    "vs retrieval' find the source.\",\n"
    '     user_raw_quote: "I want it to know that it knows"}\n'
    '  ],'
)
NEW_7B = ANCHOR_7B.replace(
    '     user_raw_quote: "I want it to know that it knows"}\n  ],',
    '     user_raw_quote: "I want it to know that it knows"},\n'
    '    {type: "reflection", title: "Recognition over retrieval is the load-bearing axis, not the slogan",\n'
    '     content: "After seeing the catalog\'s recall-related nodes side by side, a thread becomes visible: every architectural choice — situation embeddings, MAX scoring, edge-walking, frame-as-prior — traces back to one bet: the brain should RECOGNIZE what the moment needs, not search for matches. \'Recognition over retrieval\' isn\'t a catchphrase; it\'s the load-bearing axis around which the architecture rotates. Naming it as the axis (vs. one principle among many) makes it cheaper to evaluate future recall changes against.",\n'
    '     situation: "When evaluating recall architecture changes, or when a proposed mechanism feels off-axis from how the brain is supposed to behave",\n'
    '     reasoning: "Anchor noticed this connecting recall-architecture nodes during a graph walk — the principle was named once, but the architecture treats it as central. Encoding the axis-status (vs. principle-status) makes the underlying bet legible.",\n'
    '     anchor_raw_quote: "recognition over retrieval isn\'t a catchphrase; it\'s the load-bearing axis around which the architecture rotates"}\n'
    '  ],'
)
assert ANCHOR_7B in text, "Edit 7b anchor not found"
text = text.replace(ANCHOR_7B, NEW_7B, 1)

# ─── write & summarize ───────────────────────────────────────────────
DST.write_text(text)
print(f"v14: {len(src):,} chars")
print(f"v15: {len(text):,} chars")
print(f"delta: +{len(text) - len(src):,} chars (+{(len(text) - len(src)) / len(src):.1%})")
print(f"wrote {DST}")
