"""Build s1e v15.10 from v15.8 — recover from v15.9's under-emission.

CONTEXT (2026-05-13 temporal-only 12-item eval):

  v15.9 demonstrated the principle correctly on gpt4_85da3956 (anchored
  Universal Studios to conversation_now, even wrote a `correction`
  node about the assistant's hallucinated "three weeks ago"). But the
  cohort-wide cost was steep: event_time emission dropped 30% (52/366
  in v15.8 → 33/387 in v15.9). Three items v15.8 had passed flipped to
  fail (982b5123, 71017276, 0bb5a684) because the encoder under-
  anchored events the answerer needed to compute temporal arithmetic.

ROOT CAUSE:

  v15.9's framing was restraint-heavy ("be careful about assistant-
  attributed dates", "operator owns the frame"). Sonnet learned the
  restraint and generalized it — "when in doubt, don't anchor."
  Compounded by my thin Example 4 (single node, placeholder content
  and situation, no companion events, no Allen edges). Sonnet's
  implicit lesson from the example: "for temporal cases, encode
  minimally."

  Compliance probe pattern: restraint rules hit ~100% compliance,
  generative rules hit ~20%. v15.9 added restraint without an
  equivalently-strong generative push.

FIX in v15.10:

  E1: keep v15.9's scout candidate spec update (source_role +
      evidence_roles + evidence_turns).

  E2: rewrite the temporal authority rule with DECISIVE default
      ("setting event_time is the default for events the operator
      experienced") and NARROW exceptions. Half the prose weight of
      v15.9's two-section version.

  E3: rebuild Example 4 around an ACL-recovery scene — five anchored
      events from one conversation, demonstrating all three resolution
      paths plus the source_role contrast. One node detailed in full
      (the surgery — the recovery anchor); the rest structurally
      complete with real (non-placeholder) content/situation. Allen
      edges link them into a recovery arc.

Run: ./dev python3 eval/prompts/build_s1e_v15_10.py
Output: eval/prompts/s1e_v15_10.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 's1e_v15_8.txt'
DST = ROOT / 's1e_v15_10.txt'

src = SRC.read_text()
text = src

# ─── E1: extend the scout-candidate shape doc with source_role + evidence_roles
# (same as v15.9 — this part wasn't the problem)

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

`source_role` shows who attributed each date — "user" means the
operator stated it in their own wording; "assistant" means the date
came from an assistant turn (possibly paraphrasing the operator).
Use it to break ties between contradictory candidates and to
discount assistant-only dates when the operator's wording supports
a different anchor. See the next subsection."""

assert ANCHOR_E1 in text, "E1 anchor not found"
text = text.replace(ANCHOR_E1, NEW_E1, 1)

# ─── E2: tightened temporal-authority rule — DECISIVE default, narrow exceptions

ANCHOR_E2 = """### Cross-event temporal flow (Allen relations) — compose actively"""

NEW_E2 = """### Temporal authority: the operator owns the frame

For any event the operator experienced, **set event_time**. The
operator's wording is the resolution authority — explicit dates,
resolvable relative phrases ("yesterday", "last Tuesday", "2 weeks
ago"), proximal phrases ("just got back", "today", "recently"), and
event-relative phrases ("a month after my surgery") all resolve
cleanly when the encoder follows the wording.

When scout candidates conflict with the operator's own wording, the
operator wins. Read `source_role` on each candidate: "user" means
the operator attributed the date — trust it. "assistant" means the
date came from a paraphrase; if it contradicts a user-attributed
date OR the operator's wording resolves against conversation_now,
prefer the operator. The encoder is not bound to use every scout
candidate — pick the ones the operator's wording supports.

**Setting event_time is the default for events the operator
experienced.** Narrow exceptions only:

- the phrase is genuinely unresolvable AND no event chain in the
  conversation pins it ("a while back", "at some point")
- the event is third-party — operator describes someone else's
  experience without dating it ("Sarah said she'd been to Lisbon
  but didn't say when")
- the framing is hypothetical or future-conditional ("if I move
  next year", "we might launch in Q3")

For the operator's own past or present experiences: anchor.
Example 4 shows the breadth.

### Cross-event temporal flow (Allen relations) — compose actively"""

assert ANCHOR_E2 in text, "E2 anchor not found"
text = text.replace(ANCHOR_E2, NEW_E2, 1)

# ─── E3: replace placeholder-thin Example 4 with the ACL recovery scene
#        Multiple events, multiple paths, full + structural shapes, edges.

ANCHOR_E3 = """## Actions

You are the source — the graph's shape this turn is your call. Three
parallel actions, each used wherever it fits:"""

NEW_E3 = """### Example 4 — temporal authority across the breadth (the wholistic case)

Conversation (dated 2025-05-13, conversation_now = 2025-05-13):

*Operator: "Just got back from physical therapy. Started this rehab
in March after I tore my ACL skiing last winter. PT thinks I can
start running again in about a month, which is wild because I've
been off my feet since the surgery on January 22nd."*

*Assistant: "Sounds like you've been recovering since November —
that's a long road."*

Temporal scout candidates (abridged shape):
```
- handle: "2025-05-13", source_phrase: "just got back",
  source_role: "user"     ← Path 3 (proximal)
- handle: "2025-03-15", source_phrase: "in March",
  source_role: "user"     ← Path 1 (explicit month, year inferable)
- handle: "2025-01-22", source_phrase: "January 22nd",
  source_role: "user"     ← Path 1 (explicit)
- handle: "2024-12-15", source_phrase: "last winter",
  source_role: "user", precision: "fuzzy"   ← Path 2 (resolvable relative)
- handle: "2025-06-13", source_phrase: "in about a month",
  source_role: "user"     ← Path 2 (future, resolvable)
- handle: "2024-11-15", source_phrase: "since November",
  source_role: "assistant"  ← CONTRADICTS user's "January 22nd" surgery
```

Five user-attributed anchors + one assistant-attributed contradiction.
The operator's "surgery on January 22nd" is explicit; the assistant's
"since November" is a paraphrase that conflicts. Operator wins —
discard the November candidate. Every user-attributed anchor gets
encoded.

Actions:

remember (the recovery anchor — Path 1, the surgery is the spine):
  type: event
  title: "Tom's ACL reconstruction surgery"
  event_time: "2025-01-22"
  user_raw_quote: "the surgery on January 22nd"
  content: "ACL reconstruction performed on 2025-01-22, following a
            ski injury from winter 2024-25. The surgery is the
            recovery's spine — every subsequent rehab milestone
            anchors against it. Operator has been off his feet
            since this date."
  situation: "When recalling ACL injury, surgery date, the recovery
              timeline, or any post-surgical milestone Tom references."
  keywords: "ACL surgery, reconstruction, January 2025, recovery start"
  reasoning: "Explicit date from operator (Path 1). Year inferable as
              2025 because conversation_now is 2025-05-13 and the
              operator describes ongoing recovery. The assistant's
              later 'since November' paraphrase is source_role:
              assistant and contradicts this user-attributed date —
              discard it; the operator wins."
  connect_to:
    - title: "Tom's PT session — post-surgery recovery progress"
      relation: "before"
      why: "surgery preceded today's PT visit by ~16 weeks"

remember (today's PT visit — Path 3, proximal):
  type: event
  title: "Tom's PT session — post-surgery recovery progress"
  event_time: "2025-05-13"
  user_raw_quote: "Just got back from physical therapy"
  content: "Routine PT visit ~16 weeks post-surgery. PT signaled
            green light on returning to running in ~1 month."
  situation: "When recalling PT visits, recovery progress checkpoints,
              or temporal references to mid-May 2025."
  keywords: "PT, physical therapy, ACL recovery, post-op week 16"

remember (rehab start — Path 1, fuzzy):
  type: event
  title: "Tom started formal ACL rehab program"
  event_time: "2025-03-15"
  user_raw_quote: "Started this rehab in March"
  content: "Formal rehab program began mid-March 2025, ~6 weeks
            post-surgery. Set the rhythm for the PT visits that
            followed. Specific day not given — encoded mid-month
            as the best resolution from 'in March'."
  situation: "When recalling rehab program structure, the recovery
              timeline, or events in March 2025."
  keywords: "rehab, recovery program, March 2025"
  connect_to:
    - title: "Tom's ACL reconstruction surgery"
      relation: "after"
      why: "rehab began ~6 weeks post-surgery"

remember (ski injury — Path 2, fuzzy resolvable):
  type: event
  title: "Tom's ACL tear — skiing accident, winter 2024-25"
  event_time: "2024-12-15"
  user_raw_quote: "I tore my ACL skiing last winter"
  content: "ACL tear occurred during skiing in winter 2024-25.
            Precise date not given; mid-December encoded as the
            standard ski-season midpoint. Operator can refine later
            if the exact date becomes relevant."
  situation: "When recalling the original injury, ski season
              experiences, or the cause of the surgery."
  keywords: "ACL tear, skiing injury, ski accident, winter 2024-25"
  connect_to:
    - title: "Tom's ACL reconstruction surgery"
      relation: "before"
      why: "ski injury preceded surgery by ~5 weeks"

remember (running goal — Path 2, future, open):
  type: open
  title: "Tom's running return target — ~mid-June 2025"
  event_time: "2025-06-13"
  user_raw_quote: "PT thinks I can start running again in about a month"
  content: "PT-prognosticated return-to-running window: approximately
            one month from 2025-05-13 → ~2025-06-13. Operator's
            framing ('which is wild') signals emotional weight on the
            milestone. Open until confirmed."
  situation: "When tracking recovery milestones, planning forward
              from PT prognosis, or revisiting running ambitions."
  keywords: "running, return-to-running, recovery milestone, future"

The breadth in one example: Path 3 (proximal "just got back" → today),
Path 1 (explicit "January 22nd"; explicit-month "March"), Path 2
(resolvable relative "last winter"; future "in about a month").
Five events, five event_time anchors, three temporal-flow edges.
The assistant's "since November" is the source_role contrast — a
paraphrase that contradicted the user's own wording and was rejected.

## Actions

You are the source — the graph's shape this turn is your call. Three
parallel actions, each used wherever it fits:"""

assert ANCHOR_E3 in text, "E3 anchor not found"
text = text.replace(ANCHOR_E3, NEW_E3, 1)

# ─── write ───────────────────────────────────────────────────────────
DST.write_text(text)
print(f"v15.8: {len(src):,} chars")
print(f"v15.10: {len(text):,} chars")
print(f"delta from v15.8: {len(text) - len(src):+,} chars")
print(f"delta from v14 (33,358 → {len(text)}): "
      f"{len(text) - 33358:+,} chars "
      f"({(len(text) - 33358) / 33358:+.1%})")
print(f"wrote {DST}")
