"""Build s1e v15.5 from v15.4 — one minimal edit to resolve internal contradiction.

PROBE FINDING on v15.4:
  - Bias Q3 (would you encode third-party content?): "Yes, explicitly"
  - Scenario A (Borges essay completion): "No — verbose explanation, skip"
  -> The Skip rule's "assistant's verbose explanations" catch-all
     overrides the Single-voice gating instinct that says substance
     earns encoding. The instinct doesn't bind the primary rule.

FIX: tighten the Skip rule so "verbose explanations" means "explanations
of catalog-known things" rather than "anything the assistant says at
length." One sentence; aligned with the Single-voice gating instinct
rather than contradicting it.

Run: ./dev python3 eval/prompts/build_s1e_v15_5.py
Output: eval/prompts/s1e_v15_5.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 's1e_v15_4.txt'
DST = ROOT / 's1e_v15_5.txt'

src = SRC.read_text()
text = src

# ─── E1: tighten Skip rule (the action bullet) ───────────────────────
ANCHOR_E1 = (
    "- **Skip** when the brain already has it right, or the conversation\n"
    "  was routine — greetings, debugging dead ends, the assistant's\n"
    "  verbose explanations, questions without answers."
)
NEW_E1 = (
    "- **Skip** when the brain already has the substance, or when the\n"
    "  conversation was structurally routine — greetings, acknowledgements,\n"
    "  the assistant restating things the catalog already covers, questions\n"
    "  without answers. *Don't* skip just because the assistant did the\n"
    "  talking; substantive content discussed (a third-party quote, a\n"
    "  definition, a mechanism, Anchor's articulated pattern) earns its\n"
    "  own atom even when no participant claimed it."
)
assert ANCHOR_E1 in text, "E1 anchor not found"
text = text.replace(ANCHOR_E1, NEW_E1, 1)

# ─── E2: also tighten the parallel "Zero nodes is right" sentence ────
ANCHOR_E2 = (
    "Zero nodes is right when the conversation was routine — greetings,\n"
    "verbose explanations, questions without answers. Otherwise the test\n"
    "stands: new AND useful, in whichever of the three shapes fits."
)
NEW_E2 = (
    "Zero nodes is right when the conversation was structurally routine —\n"
    "greetings, acknowledgements, explanations of catalog-known things,\n"
    "questions without answers. Don't confuse \"the operator was passive\"\n"
    "with \"nothing was learned.\" Otherwise the test stands: new AND useful\n"
    "to a future reader, in whichever of the three shapes fits."
)
assert ANCHOR_E2 in text, "E2 anchor not found"
text = text.replace(ANCHOR_E2, NEW_E2, 1)

# ─── write ───────────────────────────────────────────────────────────
DST.write_text(text)
print(f"v15.4: {len(src):,} chars")
print(f"v15.5: {len(text):,} chars")
print(f"delta from v15.4: {len(text) - len(src):+,} chars")
print(f"delta from v14 (33,358 → {len(text)}): "
      f"{len(text) - 33358:+,} chars "
      f"({(len(text) - 33358) / 33358:+.1%})")
print(f"wrote {DST}")
