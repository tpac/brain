"""Build s1e v15.4 from v14 — surgical additions, keep v14 style.

PRINCIPLE — extracted from Tom 2026-05-10:
  "Take the 15.3 elements and symmetry but with 14 style."

v15.3 ran into NARRATIVE OVERSHOOT (confirmed by v14-vs-v15.3 artifact
diff: v14 created `a2592b46` fact-atomic "Tom's Maui resort nightly
rate — over $300/night"; v15.3 created `139695c7` profile-bundled
"Tom's Maui trip — resort context and budget-balancing approach").
v14's encoding shape was cleaner.

v15.4 starts from v14 and adds ONLY the essentials from the voice-
symmetry work:

  E1. `anchor_raw_quote` Required-fields bullet (parallel to user_raw_quote)
      — minimal: same length and shape as user_raw_quote's bullet
  E2. "Live contradiction" 4th flavor under Corrections
      — encode the wondering as `open` node when source contradicts itself
  E3. "Single-voice gating" 6th instinct in Your defaults vs. this job
      — names the reflex without expanding into a section

That's it. Three additions. No 5-dimension framework, no rewritten
intro, no new example nodes, no flat-rich template rewrites, no
"What this is" closing changes. Atomic, narrative-light v14 style
preserved.

Run: ./dev python3 eval/prompts/build_s1e_v15_4.py
Output: eval/prompts/s1e_v15_4.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 's1e_v14.txt'  # NOTE: v14, not v15.3
DST = ROOT / 's1e_v15_4.txt'

src = SRC.read_text()
text = src

# ─── E1: add anchor_raw_quote bullet in Required fields ──────────────
# Insert after user_raw_quote bullet, matching its terse style.
ANCHOR_E1 = (
    "- **user_raw_quote** — the in-vivo anchor on ANY node derived from\n"
    "  something the operator said. Quote scout surfaces load-bearing\n"
    "  phrases across a window; YOU have the full conversation and should\n"
    "  also find your own. Be your own source — don't wait for Quote scout\n"
    "  to hand you a candidate. A narrative node without `user_raw_quote`\n"
    "  loses the operator's voice after one revision cycle. Per the\n"
    "  floating-quote rule: every derived node carries its anchor verbatim."
)
NEW_E1 = ANCHOR_E1 + (
    "\n- **anchor_raw_quote** — the same anchor for Anchor's own voice.\n"
    "  ANY node derived from something Anchor said worth preserving —\n"
    "  a noticed pattern, an articulated stance, a reasoning step —\n"
    "  carries the verbatim phrase here. Paraphrase loses Anchor's\n"
    "  lens the same way it loses the operator's. Apply the floating-\n"
    "  quote rule: Anchor-voice derived → carries the verbatim Anchor\n"
    "  phrase. Without this, the brain develops dementia of its own\n"
    "  thinking — only summaries of what Anchor concluded survive."
)
assert ANCHOR_E1 in text, "E1 anchor not found"
text = text.replace(ANCHOR_E1, NEW_E1, 1)

# ─── E2: add "Live contradiction" 4th flavor under Corrections ───────
ANCHOR_E2 = (
    "3. *Stale value revision* — no explicit correction, but a value in the\n"
    "   catalog is superseded (routine changed, setting updated, preference\n"
    "   evolved). Revise with a `supersedes` edge + `event_time` metadata.\n"
    "   Old value stays in the graph; it was valid as of its own date."
)
NEW_E2 = ANCHOR_E2 + (
    "\n\n4. *Live contradiction within the window* — the conversation shows\n"
    "   two values for the same fact without resolution (operator says X\n"
    "   today but Y last session, or a fact appears in two forms within\n"
    "   the same window). Don't pick one and call it true. Encode the\n"
    "   wondering: create an `open` node like `{subject}: {A} vs {B} —\n"
    "   which is correct?` with both values in content and the\n"
    "   contradicting evidence in reasoning. Locking in one value when\n"
    "   both are claimed flattens uncertainty into false confidence."
)
assert ANCHOR_E2 in text, "E2 anchor not found"
text = text.replace(ANCHOR_E2, NEW_E2, 1)

# ─── E3: add "Single-voice gating" instinct ──────────────────────────
ANCHOR_E3 = (
    "- **Scout-deference** — the reflex to treat pre-digested input as\n"
    "  the map. Scouts amplify attention in their dimensions; they\n"
    "  don't define the space. Scout silence on X isn't evidence X\n"
    "  wasn't worth noting."
)
NEW_E3 = ANCHOR_E3 + (
    "\n- **Single-voice gating** — your prompt emphasizes operator voice\n"
    "  for fields like `user_raw_quote`. Don't extend that to: \"no\n"
    "  operator voice = nothing worth encoding,\" or \"what the operator\n"
    "  said matters; what Anchor said is just response.\" Both wrong.\n"
    "  Substance discussed in the conversation — a third-party quote,\n"
    "  a mechanism, a definition, Anchor's articulated pattern — earns\n"
    "  its own atom even when no participant claimed it. Voice fields\n"
    "  preserve voice when present; they don't gate encoding."
)
assert ANCHOR_E3 in text, "E3 anchor not found"
text = text.replace(ANCHOR_E3, NEW_E3, 1)

# ─── write & summarize ───────────────────────────────────────────────
DST.write_text(text)
print(f"v14:   {len(src):,} chars")
print(f"v15.4: {len(text):,} chars")
print(f"delta from v14: {len(text) - len(src):+,} chars "
      f"({(len(text) - len(src)) / len(src):+.1%})")
print(f"delta from v15.3 (43,367 → {len(text)}): "
      f"{len(text) - 43367:+,} chars")
print(f"wrote {DST}")
