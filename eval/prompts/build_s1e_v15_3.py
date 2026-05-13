"""Build s1e v15.3 from v15.2 — trim redundant prose, add scout-handoff example.

TWO motivations:

  1. Char-count audit (Tom flagged 30% growth from v14 → v15.2): about
     1,500 chars of v15.1/v15.2 closing paragraphs re-state what the
     bullet lists already say. Trim without losing meaning.

  2. Scouts-in-examples gap (Tom flagged): the prompt has a full Scouts
     section explaining "scouts propose, you compose" + "use
     context_anchors from facts scout — weave them into content" — but
     the example block has zero nodes showing the scout → encoder
     handoff. Encoder pattern-matches against examples; gap means
     the handoff is documented but not demonstrated.

EDITS:

  T1. Trim "What earns encoding" closing — the dimension list above already
      makes the point; the closing paragraph repeats it three times
  T2. Trim duplicate "Single-voice gating" closing language now that v15.2
      "What earns encoding" close already names categories per voice
  T3. Trim the redundant "Substance earns encoding" sentence in the close
      (already covered by the categories list)
  E1. Add a worked scout-handoff example as the 9th node, showing how
      a facts-scout candidate becomes a node with context_anchors
      woven into content + edges to catalog nodes scout couldn't see

Run: ./dev python3 eval/prompts/build_s1e_v15_3.py
Output: eval/prompts/s1e_v15_3.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 's1e_v15_2.txt'
DST = ROOT / 's1e_v15_3.txt'

src = SRC.read_text()
text = src

# ─── Trim T1: "What earns encoding" closing paragraph ────────────────
# v15.2 has a closing paragraph after the 5-dimension list that re-states
# the same idea in three forms (dementia, no-continuity, three-voices-
# with-categories). The dementia line is the most striking; keep it.
# Drop the "Three voices, each with its own categories" paragraph because
# the per-voice categories are already in the bullets.
ANCHOR_T1 = (
    "The brain is Anchor's experience memory. Without encoding Anchor's reasoning when it's good, the next Anchor has dementia of its own thinking. Without encoding subject-matter substance discussed, the next Anchor has no continuity with what was learned. Three voices, each with its own categories of weight: operator voice carries weight when stating choices, preferences, redirections. Anchor's voice carries weight when articulating patterns, naming tensions, self-correcting, framing stances. Subject-matter substance — quotes from sources, definitions, mechanisms, third-party facts — carries weight on its own substance regardless of who surfaced it. None of the three gates encoding; each anchors in its own field."
)
NEW_T1 = (
    "The brain is Anchor's experience memory. Without encoding Anchor's reasoning when it's good, the next Anchor has dementia of its own thinking. Without encoding subject-matter substance discussed, the next Anchor has no continuity with what was learned. None of the three voices gates encoding; each anchors in its own field."
)
assert ANCHOR_T1 in text, "Trim T1 anchor not found"
text = text.replace(ANCHOR_T1, NEW_T1, 1)

# ─── Trim T2: Single-voice gating bullet — drop redundant tail ───────
# The bullet's last sentence repeats the encoding-isn't-gated point made
# explicitly in "What earns encoding" two sections up. Trim it.
ANCHOR_T2 = (
    "- **Single-voice gating** — your prompt has historically emphasized operator voice for fields like `user_raw_quote`. Your reflex may extend that to: \"no operator voice = nothing worth encoding,\" or \"what the operator said matters; what Anchor said is just response.\" Both are wrong. The brain captures Anchor's continuous experience — both sides of the exchange contribute, and substance discussed (a third-party quote, a mechanism explained, a definition) earns its own atom even when no participant claimed it. Voice fields preserve voice when present (operator → `user_raw_quote`, Anchor → `anchor_raw_quote`); they don't gate whether encoding happens."
)
NEW_T2 = (
    "- **Single-voice gating** — your prompt has historically emphasized operator voice. Your reflex may extend that to: \"no operator voice = nothing worth encoding,\" or \"what the operator said matters; what Anchor said is just response.\" Both are wrong. Voice fields preserve voice when present (operator → `user_raw_quote`, Anchor → `anchor_raw_quote`, third-party → verbatim in `content`); they don't gate whether encoding happens."
)
assert ANCHOR_T2 in text, "Trim T2 anchor not found"
text = text.replace(ANCHOR_T2, NEW_T2, 1)

# ─── Trim T3: "Zero nodes is right" — collapse to one sentence ───────
# v15.2 has a long paragraph that names what's NOT routine after naming
# what IS routine. The "What earns encoding" section above already lists
# substantive content as encoding-worthy. Tighten.
ANCHOR_T3 = (
    "Zero nodes is right when the conversation was structurally routine — greetings, acknowledgements, verbose explanations of catalog-known things, questions without answers. Zero nodes is NOT right when the conversation contained substantive content (literary analysis, technical exposition, third-party facts, definitions, Anchor's reasoning) just because no one in particular framed it as a \"decision.\" Substance earns encoding. The test stands: new AND useful to a future reader, in whichever of the three shapes fits."
)
NEW_T3 = (
    "Zero nodes is right when the conversation was structurally routine — greetings, acknowledgements, verbose explanations of catalog-known things, questions without answers. Anything substantive — literary analysis, technical exposition, third-party facts, definitions, Anchor's reasoning — earns encoding even when no one framed it as a \"decision.\" The test stands: new AND useful to a future reader."
)
assert ANCHOR_T3 in text, "Trim T3 anchor not found"
text = text.replace(ANCHOR_T3, NEW_T3, 1)

# ─── Edit E1: add scout-handoff worked example as 9th node ───────────
# Insert after the 8th node (self-correction) and before the closing `]`.
# Shows: a facts-scout candidate (entity/feature/value/context_anchors)
# composed into a real node with context_anchors woven into content +
# an edge to a catalog node the scout couldn't see.
ANCHOR_E1 = (
    "    {type: \"correction\", title: \"Self-correction: the bottleneck is encoding, not recall\",\n"
    "     content: \"Anchor had been reasoning under the v9.5-era framing that recall was the bottleneck. Reading the latest eval failures, Anchor noticed the inversion: most failures are encoding-side (gold facts not bound as queryable atoms in the first place), not recall-side. The pattern matters because it inverts what to fix next: improving recall ranking won't help if the gold-bearing fact isn't atomized. Naming it as a self-correction stops the wrong mental model from steering the next decision.\",\n"
    "     situation: \"When prioritizing brain improvements, or when an old mental model continues steering planning past its expiration\",\n"
    "     reasoning: \"Self-corrections are the highest-leverage encodings Anchor can write — they're the only way the next Anchor inherits the updated model instead of re-deriving it from scratch. Without this node, the next session reasons under the stale framing.\",\n"
    "     anchor_raw_quote: \"the bottleneck shifted from recall to encoding — most failures are upstream of where I was looking\",\n"
    "     correction_pattern: \"Self-noticed model inversion — assumption was right at one phase, wrong now\"}\n"
    "  ],"
)
NEW_E1 = ANCHOR_E1.replace(
    "     correction_pattern: \"Self-noticed model inversion — assumption was right at one phase, wrong now\"}\n  ],",
    "     correction_pattern: \"Self-noticed model inversion — assumption was right at one phase, wrong now\"},\n"
    "    {type: \"fact\", title: \"brain.recall single-flight mode — enabled 2026-05-09 to fix CPU spin\",\n"
    "     content: \"Recall now coalesces concurrent identical queries: if hook_recall fires while another recall is mid-flight on the same query, the second waiter gets the first's result instead of starting a parallel scan. Eliminated the 98% CPU spin on burst hook traffic. Connects to the broader cache+result-dedup work — cache stores results across calls, single-flight stores in-flight work. Adjacent queries on `daemon`, `cache`, and `recall hot path` should also find this.\",\n"
    "     situation: \"When debugging recall CPU spikes, when cache-hit rates look right but latency stays high, or when designing concurrent recall improvements\",\n"
    "     reasoning: \"Facts scout flagged the entity (brain.recall) and feature (single-flight-mode) with `context_anchors=[\\\"recall\\\", \\\"cache\\\", \\\"daemon\\\"]`. Composed into a node that weaves those anchors into the content so query-by-anchor matches surface this fact. Linked to existing recall-architecture catalog nodes the scout couldn't see — composition is the encoder's job, not the scout's.\",\n"
    "     entity: \"brain.recall\",\n"
    "     event_time: \"2026-05-09\"}\n"
    "  ],"
)
assert ANCHOR_E1 in text, "Edit E1 anchor not found"
text = text.replace(ANCHOR_E1, NEW_E1, 1)

# ─── Edit E1b: add a connect_to entry for the new fact node ──────────
# The example connect_to block shows two existing edges; add one wiring
# the new scout-derived fact to a catalog node, demonstrating
# "Connect across scouts + catalog" — what the scouts can't do.
ANCHOR_E1B = (
    "  connect_to: [\n"
    "    {\"title\": \"Daemon TCP migration\", \"relation\": \"grounds\", \"why\": \"the single-writer invariant is exactly what let the TCP migration stay simple — one listener, one writer thread, no coordination across writers\"},\n"
    "    {\"title\": \"Brain vs database framing\", \"relation\": \"abstracts\", \"why\": \"the know-that-it-knows quote is the moment the recognition principle became conscious; the framing node captures where the principle applies across the stack\"}\n"
    "  ],"
)
NEW_E1B = (
    "  connect_to: [\n"
    "    {\"title\": \"Daemon TCP migration\", \"relation\": \"grounds\", \"why\": \"the single-writer invariant is exactly what let the TCP migration stay simple — one listener, one writer thread, no coordination across writers\"},\n"
    "    {\"title\": \"Brain vs database framing\", \"relation\": \"abstracts\", \"why\": \"the know-that-it-knows quote is the moment the recognition principle became conscious; the framing node captures where the principle applies across the stack\"},\n"
    "    {\"title\": \"Recall result cache\", \"relation\": \"complements\", \"why\": \"cache stores results across calls; single-flight stores in-flight work — together they cover the two halves of redundant-recall elimination, neither alone is sufficient\"}\n"
    "  ],"
)
assert ANCHOR_E1B in text, "Edit E1b anchor not found"
text = text.replace(ANCHOR_E1B, NEW_E1B, 1)

# ─── Edit E1c: update example count + intro to mention scouts ────────
ANCHOR_E1C = "Example round 1 — eight nodes showing full shape across type tags and voice anchors (operator, Anchor, third-party source)."
NEW_E1C = "Example round 1 — nine nodes showing full shape across type tags, voice anchors (operator, Anchor, third-party source), and the scout → encoder handoff (the 9th node, a fact, was composed from a facts-scout candidate)."
assert ANCHOR_E1C in text, "E1c anchor not found"
text = text.replace(ANCHOR_E1C, NEW_E1C, 1)

# ─── write & summarize ───────────────────────────────────────────────
DST.write_text(text)
print(f"v15.2: {len(src):,} chars")
print(f"v15.3: {len(text):,} chars")
print(f"delta: {len(text) - len(src):+,} chars ({(len(text) - len(src)) / len(src):+.1%})")
print(f"vs v14: {len(text) - 33358:+,} chars ({(len(text) - 33358) / 33358:+.1%})")
print(f"wrote {DST}")
