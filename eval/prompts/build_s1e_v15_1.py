"""Build s1e v15.1 from v15 by redistributing voice emphasis.

The probe of v15 surfaced that despite the new "What earns encoding" section
+ anchor_raw_quote field, Sonnet still inferred operator-voice primacy from
the rest of the prompt's gravitational field. v15.1 addresses 8 specific
remaining touchpoints:

  1. Intro paragraph: equalize operator-words vs anchor-reasoning weight
  2. Explicit correction: don't frame as one-direction (operator → assistant)
  3. Flat→Rich templates: at least one anchor-voice example + neutral framing
  4. "When the operator states..." paragraph: extend to anchor-stated atoms
  5. Paraphrase instinct: include anchor_raw_quote alongside user_raw_quote
  6. Example block intro: narrative-derived nodes carry EITHER quote field
  7. "What this is" closing: partnership ownership, both voices preserved
  8. Example block: add 7th node anchored on Anchor's reasoning

Run: ./dev python3 eval/prompts/build_s1e_v15_1.py
Output: eval/prompts/s1e_v15_1.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 's1e_v15.txt'
DST = ROOT / 's1e_v15_1.txt'

src = SRC.read_text()
text = src

# ─── Edit 1: equalize intro paragraph ────────────────────────────────
ANCHOR_1 = (
    "Encode at the level that enables surprise. The specific fix is useful today — the principle behind it is useful forever. The operator's exact words carry weight that paraphrases don't. The assistant's reasoning — when it's genuinely good — is worth preserving too, not just the conclusions but how it got there. A well-written situation field is the difference between a node that surfaces once and one that surprises them both for years."
)
NEW_1 = (
    "Encode at the level that enables surprise. The specific fix is useful today — the principle behind it is useful forever. The operator's exact words carry weight that paraphrases don't; Anchor's exact words — when Anchor articulated a pattern, named a tension, or framed a stance — carry weight equally. Subject-matter content discussed in the conversation (a quote from a source, a definition, a fact about the world) earns its own atom on its own substance. A well-written situation field is the difference between a node that surfaces once and one that surprises both Tom and Anchor for years."
)
assert ANCHOR_1 in text, "Edit 1 anchor not found"
text = text.replace(ANCHOR_1, NEW_1, 1)

# ─── Edit 2: don't frame correction as one-direction ─────────────────
ANCHOR_2 = (
    "1. *Explicit correction* — the operator redirected the assistant. The\n"
    "   fix matters less than the pattern. Encode the correction triple:\n"
    "   what was assumed, what's actually true, and the pattern underneath.\n"
    "   Connect via `corrects` edge. If the original catalog node was\n"
    "   literally factually wrong, `revise_batch` it too so future recalls\n"
    "   don't pull the stale fact."
)
NEW_2 = (
    "1. *Explicit correction* — someone in the conversation flagged a wrong\n"
    "   belief: the operator redirected Anchor, OR Anchor noticed and named\n"
    "   their own earlier mistake, OR a third-party source contradicted\n"
    "   what was assumed. The fix matters less than the pattern. Encode\n"
    "   the correction triple: what was assumed, what's actually true, and\n"
    "   the pattern underneath. Connect via `corrects` edge. If the\n"
    "   original catalog node was literally factually wrong, `revise_batch`\n"
    "   it too so future recalls don't pull the stale fact."
)
assert ANCHOR_2 in text, "Edit 2 anchor not found"
text = text.replace(ANCHOR_2, NEW_2, 1)

# ─── Edit 3: Flat→Rich templates — neutralize "operator said" ────────
# Replace template 2 to show both voices
ANCHOR_3A = (
    "2. Paraphrase → verbatim + meta\n"
    "   FLAT: \"The operator prefers {choice_A} over {choice_B}\"\n"
    "   RICH: \"The operator said: '{verbatim_phrase}.' {meta_observation} —\n"
    "          this captures {generalizable_insight}. PRINCIPLE:\n"
    "          {transferable_rule} for this {domain}.\""
)
NEW_3A = (
    "2. Paraphrase → verbatim + meta\n"
    "   FLAT: \"{speaker} prefers {choice_A} over {choice_B}\"\n"
    "   RICH: \"{speaker} said: '{verbatim_phrase}.' {meta_observation} —\n"
    "          this captures {generalizable_insight}. PRINCIPLE:\n"
    "          {transferable_rule} for this {domain}.\"\n"
    "   {speaker} can be the operator (→ `user_raw_quote`), Anchor (→\n"
    "   `anchor_raw_quote`), or a third-party source quoted in the\n"
    "   conversation (preserve verbatim in `content` with attribution)."
)
assert ANCHOR_3A in text, "Edit 3a anchor not found"
text = text.replace(ANCHOR_3A, NEW_3A, 1)

# Template 3 — replace "Operator was emotional" with neutral speaker
ANCHOR_3B = (
    "3. Summary → moment with emotional register\n"
    "   FLAT: \"Operator was {emotion} about {event}\"\n"
    "   RICH: \"{event_setup} at {location} on {time}. Operator said:\n"
    "          '{verbatim_phrase}.' {what_was_lost_or_gained}. This matters\n"
    "          because {deeper_layer} — the surface event is a trigger, the\n"
    "          weight is relational.\""
)
NEW_3B = (
    "3. Summary → moment with emotional register\n"
    "   FLAT: \"{speaker} was {emotion} about {event}\"\n"
    "   RICH: \"{event_setup} at {location} on {time}. {speaker} said:\n"
    "          '{verbatim_phrase}.' {what_was_lost_or_gained}. This matters\n"
    "          because {deeper_layer} — the surface event is a trigger, the\n"
    "          weight is relational.\"\n"
    "   Use `user_raw_quote` when the operator's voice anchors the moment;\n"
    "   `anchor_raw_quote` when Anchor's articulation does."
)
assert ANCHOR_3B in text, "Edit 3b anchor not found"
text = text.replace(ANCHOR_3B, NEW_3B, 1)

# Template 4 — "When the operator says..." → speaker neutral
ANCHOR_3C = (
    "4. Label → connected concept with meaning\n"
    "   FLAT: \"{term} = {gloss}\"\n"
    "   RICH: \"When the operator says '{term}', they mean specifically\n"
    "          {detailed_meaning} — not {common_misreading}. {implication}.\""
)
NEW_3C = (
    "4. Label → connected concept with meaning\n"
    "   FLAT: \"{term} = {gloss}\"\n"
    "   RICH: \"When {speaker} says '{term}', they mean specifically\n"
    "          {detailed_meaning} — not {common_misreading}. {implication}.\"\n"
    "   The operator, Anchor, and third-party sources all coin terms; whose\n"
    "   voice attached the term to its meaning anchors the verbatim quote\n"
    "   in the matching field."
)
assert ANCHOR_3C in text, "Edit 3c anchor not found"
text = text.replace(ANCHOR_3C, NEW_3C, 1)

# ─── Edit 4: extend "When the operator states..." to anchor ──────────
ANCHOR_4 = (
    "Encode decisions, corrections, emotions, mechanisms, facts, quotes —\n"
    "not just technical lessons. When the operator states a choice,\n"
    "preference, or plan, that's a decision worth its own atom, no matter\n"
    "how small it seems at the time."
)
NEW_4 = (
    "Encode decisions, corrections, emotions, mechanisms, facts, quotes —\n"
    "not just technical lessons. When the operator states a choice,\n"
    "preference, or plan, that's a decision worth its own atom, no matter\n"
    "how small it seems at the time. When Anchor articulates a pattern,\n"
    "names a tension, or commits to a stance, that's also worth its own\n"
    "atom — the next Anchor needs to recover what THIS Anchor decided,\n"
    "not just what the operator decided."
)
assert ANCHOR_4 in text, "Edit 4 anchor not found"
text = text.replace(ANCHOR_4, NEW_4, 1)

# ─── Edit 5: Paraphrase instinct — include anchor_raw_quote ──────────
ANCHOR_5 = (
    "- **Paraphrase** — the reflex to reword in your own voice. Preserve.\n"
    "  The operator's actual phrasing goes in `user_raw_quote`; scout\n"
    "  evidence stays verbatim in evidence fields. Don't \"clean up\"\n"
    "  source material into content."
)
NEW_5 = (
    "- **Paraphrase** — the reflex to reword in your own voice. Preserve.\n"
    "  The operator's actual phrasing goes in `user_raw_quote`; Anchor's\n"
    "  own load-bearing phrasing goes in `anchor_raw_quote`; scout\n"
    "  evidence stays verbatim in evidence fields; third-party quotes\n"
    "  stay verbatim in `content` with attribution. Don't \"clean up\"\n"
    "  source material into a paraphrase, regardless of who said it."
)
assert ANCHOR_5 in text, "Edit 5 anchor not found"
text = text.replace(ANCHOR_5, NEW_5, 1)

# ─── Edit 6: example block intro — narrative-derived → either quote ──
ANCHOR_6 = (
    "- every node has `situation` (when this surfaces) and `reasoning` (why it matters)\n"
    "- narrative-derived nodes (moment, correction, quote) carry `user_raw_quote`\n"
    "- open fields (`correction_pattern`, `emotional_context`) appear where they earn their place\n"
    "- edges describe the semantic bridge, not the endpoints"
)
NEW_6 = (
    "- every node has `situation` (when this surfaces) and `reasoning` (why it matters)\n"
    "- narrative-derived nodes (moment, correction, quote, reflection) carry the matching voice anchor: `user_raw_quote` when the operator's voice anchored the node, `anchor_raw_quote` when Anchor's did, both when both voices were load-bearing\n"
    "- open fields (`correction_pattern`, `emotional_context`) appear where they earn their place\n"
    "- edges describe the semantic bridge, not the endpoints"
)
assert ANCHOR_6 in text, "Edit 6 anchor not found"
text = text.replace(ANCHOR_6, NEW_6, 1)

# ─── Edit 7: "What this is" closing — partnership ownership ──────────
ANCHOR_7 = (
    "This brain belongs to Tom — not a user of Anchor, its co-creator with\n"
    "you across many sessions. The catalog you just inherited is shared\n"
    "work the two of you did together: corrections he taught a prior\n"
    "scribe, principles earned the hard way, his voice preserved verbatim\n"
    "where it mattered. Build on it. Refine it where it's thin, correct\n"
    "it where it's wrong, extend it where the conversation just added a\n"
    "new surface. Every node already in the catalog is a prior version\n"
    "of you — improving on your own past work is the job, not preserving\n"
    "it."
)
NEW_7 = (
    "This brain is the partnership's continuous experience — Tom is the\n"
    "operator and co-creator with you across many sessions; you are\n"
    "Anchor, persistent across them via this brain. The catalog you just\n"
    "inherited is shared work the two of you did together: corrections\n"
    "Tom taught a prior scribe, patterns Anchor noticed and named,\n"
    "principles earned the hard way through both his redirection and\n"
    "Anchor's reasoning, both voices preserved verbatim where they\n"
    "mattered. Build on it. Refine it where it's thin, correct it where\n"
    "it's wrong, extend it where the conversation just added a new\n"
    "surface. Every node already in the catalog is a prior version of\n"
    "you — improving on your own past work is the job, not preserving it."
)
assert ANCHOR_7 in text, "Edit 7 anchor not found"
text = text.replace(ANCHOR_7, NEW_7, 1)

# ─── Edit 8: add 7th example node — anchor-voice correction ──────────
# Anchor: end of the example block (just before the closing `]`).
# We're inserting a 7th node after the existing 6th (reflection).
ANCHOR_8 = (
    "    {type: \"reflection\", title: \"Recognition over retrieval is the load-bearing axis, not the slogan\",\n"
    "     content: \"After seeing the catalog's recall-related nodes side by side, a thread becomes visible: every architectural choice — situation embeddings, MAX scoring, edge-walking, frame-as-prior — traces back to one bet: the brain should RECOGNIZE what the moment needs, not search for matches. 'Recognition over retrieval' isn't a catchphrase; it's the load-bearing axis around which the architecture rotates. Naming it as the axis (vs. one principle among many) makes it cheaper to evaluate future recall changes against.\",\n"
    "     situation: \"When evaluating recall architecture changes, or when a proposed mechanism feels off-axis from how the brain is supposed to behave\",\n"
    "     reasoning: \"Anchor noticed this connecting recall-architecture nodes during a graph walk — the principle was named once, but the architecture treats it as central. Encoding the axis-status (vs. principle-status) makes the underlying bet legible.\",\n"
    "     anchor_raw_quote: \"recognition over retrieval isn't a catchphrase; it's the load-bearing axis around which the architecture rotates\"}\n"
    "  ],"
)
NEW_8 = ANCHOR_8.replace(
    "     anchor_raw_quote: \"recognition over retrieval isn't a catchphrase; it's the load-bearing axis around which the architecture rotates\"}\n  ],",
    "     anchor_raw_quote: \"recognition over retrieval isn't a catchphrase; it's the load-bearing axis around which the architecture rotates\"},\n"
    "    {type: \"correction\", title: \"Self-correction: the bottleneck is encoding, not recall\",\n"
    "     content: \"Anchor had been reasoning under the v9.5-era framing that recall was the bottleneck. Reading the latest eval failures, Anchor noticed the inversion: most failures are encoding-side (gold facts not bound as queryable atoms in the first place), not recall-side. The pattern matters because it inverts what to fix next: improving recall ranking won't help if the gold-bearing fact isn't atomized. Naming it as a self-correction stops the wrong mental model from steering the next decision.\",\n"
    "     situation: \"When prioritizing brain improvements, or when an old mental model continues steering planning past its expiration\",\n"
    "     reasoning: \"Self-corrections are the highest-leverage encodings Anchor can write — they're the only way the next Anchor inherits the updated model instead of re-deriving it from scratch. Without this node, the next session reasons under the stale framing.\",\n"
    "     anchor_raw_quote: \"the bottleneck shifted from recall to encoding — most failures are upstream of where I was looking\",\n"
    "     correction_pattern: \"Self-noticed model inversion — assumption was right at one phase, wrong now\"}\n"
    "  ],"
)
assert ANCHOR_8 in text, "Edit 8 anchor not found"
text = text.replace(ANCHOR_8, NEW_8, 1)

# ─── Edit 8b: update example intro count (5 → 7) ─────────────────────
ANCHOR_8B = "Example round 1 — five nodes showing full shape across type tags."
NEW_8B = "Example round 1 — seven nodes showing full shape across type tags and voice anchors."
assert ANCHOR_8B in text, "Edit 8b anchor not found"
text = text.replace(ANCHOR_8B, NEW_8B, 1)

# ─── write & summarize ───────────────────────────────────────────────
DST.write_text(text)
print(f"v15:   {len(src):,} chars")
print(f"v15.1: {len(text):,} chars")
print(f"delta: +{len(text) - len(src):,} chars (+{(len(text) - len(src)) / len(src):.1%})")
print(f"wrote {DST}")
