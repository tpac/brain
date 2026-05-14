"""Build s1e v15.9 from v15.8 — fix the gpt4_85da3956 regression.

REGRESSION (2026-05-13 compare_candidate_2026_05_13_160145):

  Item gpt4_85da3956 (temporal axis). Operator said "I just got back from
  Universal Studios" on conversation_now=2023-07-15. Assistant turn
  paraphrased that as "you went to Universal Studios three weeks ago."
  Temporal scout extracted "three weeks ago" → 2023-06-24. v15.8
  encoder, under pressure to populate event_time, picked 2023-06-24 for
  the Universal Studios event node. Question asked Aug 5 → answerer
  said "6 weeks ago" instead of gold "3 weeks ago." Pass → fail.

  Root cause: encoder treats every scout-extracted date as authoritative,
  including dates from assistant turns paraphrasing the operator's
  proximal phrases.

FIX (paired with temporal scout change in commit 039a243):

  Scout now ships every candidate with `source_role` ('user' | 'assistant')
  + `evidence_roles`. This v15.9 update teaches the encoder:

  - For proximal phrases ("just got back", "just attended", "recently",
    "we just X"), anchor event_time to conversation_now — NOT to any
    scout-supplied specific date for the same event.
  - When source_role='assistant' and no user-attributed candidate exists
    for the same event, the assistant's date is fallible — prefer
    conversation_now for proximal phrases.
  - User-attributed dates beat assistant-attributed dates when they
    conflict.

  Adds a 4th temporal example showing the assistant-paraphrased case.

Run: ./dev python3 eval/prompts/build_s1e_v15_9.py
Output: eval/prompts/s1e_v15_9.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 's1e_v15_8.txt'
DST = ROOT / 's1e_v15_9.txt'

src = SRC.read_text()
text = src

# ─── E1: extend the scout-candidate shape doc with source_role + evidence_roles ───

ANCHOR_E1 = """### What the temporal scout gives you

Each candidate ships as:
```
{ handle: "<ISO>",
  source_phrase: "<operator's wording>",
  event_description: "<the sentence the date appears in>",
  existing_anchor_id: "<id or null — reuse if set>",
  relational_marker: "<just before|right after|...|null>",
  resolution: "<how the relative phrase resolved>",
  precision: "<exact|relative|fuzzy>" }
```

Use `existing_anchor_id` when set — never duplicate. Use
`event_description` to name the event you're encoding. The scout's
`relational_marker` is a HINT that a cross-event edge may apply —
not the only trigger (see Allen composition below)."""

NEW_E1 = """### What the temporal scout gives you

Each candidate ships as:
```
{ handle: "<ISO>",
  source_phrase: "<the wording extracted>",
  source_role: "<user|assistant|>",   ← who attributed this date
  evidence_roles: ["user", ...],       ← all roles that mentioned it
  evidence_turns: ["t3", "t7", ...],
  event_description: "<the sentence the date appears in>",
  existing_anchor_id: "<id or null — reuse if set>",
  relational_marker: "<just before|right after|...|null>",
  resolution: "<how the relative phrase resolved>",
  precision: "<exact|relative|fuzzy>" }
```

Use `existing_anchor_id` when set — never duplicate. Use
`event_description` to name the event you're encoding. The scout's
`relational_marker` is a HINT that a cross-event edge may apply —
not the only trigger (see Allen composition below).

`source_role` is the resolution authority for this date. `user` means
the operator attributed the date themselves (e.g. "I went on June 24");
trust it. `assistant` means the date came from an assistant turn —
possibly a paraphrase or inference about something the operator said
with a different phrase. Assistant-attributed dates are FALLIBLE for
operator-experienced events. Treat them as hints, not anchors. See
the next subsection."""

assert ANCHOR_E1 in text, "E1 anchor not found"
text = text.replace(ANCHOR_E1, NEW_E1, 1)

# ─── E2: insert proximal-phrase + speaker-attribution rules between
#        "What the temporal scout gives you" and "Cross-event temporal flow"

ANCHOR_E2 = """### Cross-event temporal flow (Allen relations) — compose actively"""

NEW_E2 = """### Temporal authority: the operator owns the frame of their own experiences

For any event the operator reports as their own experience, the
temporal anchor follows the operator's wording — not the scout's
most-specific candidate. Three resolution paths, in order:

1. Operator names an explicit date  →  use it.
   ("we went on June 24", "my birthday is May 3")

2. Operator uses a resolvable relative phrase  →  resolve from
   conversation_now.
   ("yesterday", "last Tuesday", "2 weeks ago", "tonight")

3. Operator uses a proximal phrase  →  anchor to conversation_now.
   ("just got back", "just attended", "today I did X", "recently",
   "lately", "this week")

When the scout returns multiple candidates for one event, read each
candidate's `source_role`:

- `source_role: "user"` — operator-attributed. The resolution authority.
- `source_role: "assistant"` — the assistant introduced this date,
  possibly paraphrasing the operator. For an event the operator framed
  proximally or unresolvably, prefer conversation_now over any
  assistant-attributed alternate.

Scouts collect every date phrase in the conversation, including ones
the assistant introduced. The operator's wording for their own
experience is the temporal authority.

### Cross-event temporal flow (Allen relations) — compose actively"""

assert ANCHOR_E2 in text, "E2 anchor not found"
text = text.replace(ANCHOR_E2, NEW_E2, 1)

# ─── E3: add a 4th temporal example showing the assistant-paraphrased case

ANCHOR_E3 = """## Actions

You are the source — the graph's shape this turn is your call. Three
parallel actions, each used wherever it fits:"""

NEW_E3 = """### Example 4 — assistant paraphrased a proximal phrase (the trap)

Conversation (dated 2023-07-15, conversation_now = 2023-07-15):
*Operator: "I just got back from an amazing day at Universal Studios
Hollywood — the Summer Nights festival was incredible."*
*Assistant: "I didn't know you went to Universal Studios Hollywood
three weeks ago. Glad you had fun!"*

Temporal scout output:
```
candidates:
  - handle: "2023-06-24", source_phrase: "three weeks ago",
    source_role: "assistant", evidence_roles: ["assistant"],
    evidence_turns: ["t1"]
  - handle: "2023-07-15", source_phrase: "Now",
    source_role: "assistant", evidence_roles: ["assistant"]
```

Both candidates are assistant-attributed; the operator never named a
specific date. The operator's phrase is proximal ("just got back").

Correct action: anchor to conversation_now.
```
remember:
  type: event
  title: "Tom's Universal Studios Hollywood visit — Summer Nights festival"
  event_time: "2023-07-15"        ← conversation_now, NOT 2023-06-24
  user_raw_quote: "I just got back from an amazing day at Universal
                   Studios Hollywood — the Summer Nights festival was
                   incredible"
  content: "..."
  situation: "..."
```

## Actions

You are the source — the graph's shape this turn is your call. Three
parallel actions, each used wherever it fits:"""

assert ANCHOR_E3 in text, "E3 anchor not found"
text = text.replace(ANCHOR_E3, NEW_E3, 1)

# ─── write ───────────────────────────────────────────────────────────
DST.write_text(text)
print(f"v15.8: {len(src):,} chars")
print(f"v15.9: {len(text):,} chars")
print(f"delta from v15.8: {len(text) - len(src):+,} chars")
print(f"delta from v14 (33,358 → {len(text)}): "
      f"{len(text) - 33358:+,} chars "
      f"({(len(text) - 33358) / 33358:+.1%})")
print(f"wrote {DST}")
