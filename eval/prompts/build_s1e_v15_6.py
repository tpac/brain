"""Build s1e v15.6 from v15.5 — one more targeted edit on Borges-style content.

v15.5 PROBE FINDING:
  Sonnet reading v15.5 still rationalized skipping Borges essay:
  "Borges quotes aren't load-bearing for understanding the operator's
  world. They're citations in an essay."

  The "Single-voice gating" instinct + the tightened Skip rule both
  said "substance earns encoding," but Sonnet read the prompt's deeper
  frame ("write for a future reader") as implying "operator-related
  future" — applying a personal-life filter to subject-matter content.

  The Borges essay is a THINKING TASK the operator asked Anchor to do.
  The Borges substance is what the partnership thought about together
  — that's the brain's job to preserve.

FIX: name the thinking-task case explicitly. When the operator asks
Anchor to do research / analysis / exposition, the substance of that
thinking IS the partnership's work and earns encoding.

Run: ./dev python3 eval/prompts/build_s1e_v15_6.py
Output: eval/prompts/s1e_v15_6.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 's1e_v15_5.txt'
DST = ROOT / 's1e_v15_6.txt'

src = SRC.read_text()
text = src

# ─── E1: extend the Skip clarification to cover thinking-task case ───
ANCHOR_E1 = (
    "- **Skip** when the brain already has the substance, or when the\n"
    "  conversation was structurally routine — greetings, acknowledgements,\n"
    "  the assistant restating things the catalog already covers, questions\n"
    "  without answers. *Don't* skip just because the assistant did the\n"
    "  talking; substantive content discussed (a third-party quote, a\n"
    "  definition, a mechanism, Anchor's articulated pattern) earns its\n"
    "  own atom even when no participant claimed it."
)
NEW_E1 = (
    "- **Skip** when the brain already has the substance, or when the\n"
    "  conversation was structurally routine — greetings, acknowledgements,\n"
    "  the assistant restating things the catalog already covers, questions\n"
    "  without answers.\n"
    "  *Don't* skip just because the assistant did the talking. When the\n"
    "  operator asked Anchor to do thinking work — research a topic,\n"
    "  analyze a text, explain a mechanism, complete an essay — the\n"
    "  substance of that thinking IS the partnership's intellectual\n"
    "  activity, and the brain captures it. The Borges quote Anchor\n"
    "  cited in an essay, the definition Anchor explained, the\n"
    "  mechanism Anchor diagnosed — these earn nodes. The next Anchor\n"
    "  needs to recover what was thought, not just what was decided."
)
assert ANCHOR_E1 in text, "E1 anchor not found"
text = text.replace(ANCHOR_E1, NEW_E1, 1)

# ─── write ───────────────────────────────────────────────────────────
DST.write_text(text)
print(f"v15.5: {len(src):,} chars")
print(f"v15.6: {len(text):,} chars")
print(f"delta from v15.5: {len(text) - len(src):+,} chars")
print(f"delta from v14 (33,358 → {len(text)}): "
      f"{len(text) - 33358:+,} chars "
      f"({(len(text) - 33358) / 33358:+.1%})")
print(f"wrote {DST}")
