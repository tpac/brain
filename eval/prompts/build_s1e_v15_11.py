"""Build s1e v15.11 from v15.10 — restore correction node + Allen edges + tighten example.

CONTEXT (2026-05-14 structural analysis of v15.8 / v15.9 / v15.10):

  Per-item gold-anchor accuracy: v15.10 BEST of the three (anchors
  Universal which v15.8 missed; restores Book Lovers which v15.9 lost;
  hits BBQ-on-June-3rd exactly which neither v15.8 nor v15.9 got).
  Eval pass rate is downstream-limited (surface drops well-anchored
  nodes on temporal queries — the next-bottleneck the prior handoff
  named).

  BUT v15.10 lost two behaviors that v15.9 had:
    1. correction-type nodes (v15.9 wrote 2 explicit "assistant
       hallucinated X" callouts; v15.10 wrote 0). I dropped the
       "Wrong action (the trap)" paragraph from Example 4 during
       tightening, and Sonnet lost the signal.
    2. Allen-edge composition density (v15.8=29 temporal edges,
       v15.9=24, v15.10=12). v15.10 anchors event_time directly and
       relies on the answerer to sequence — the graph is "anchored
       but not sequenced."

FIX in v15.11 — targeted Example 4 edits ONLY:

  - E1 (scout candidate spec): UNCHANGED from v15.10
  - E2 (temporal authority principle): UNCHANGED from v15.10
  - E3 (Example 4): revised to:
      a. Trim each abbreviated node's content/situation to one line
         + "..." where extra detail is not teaching. The surgery
         node (the canonical anchor) keeps full detail.
      b. ADD a fact node about the surgeon — atomic personal-network
         fact, demonstrates good fact-type usage Tom flagged.
      c. ADD a correction node calling out the assistant's "since
         November" mistake. Restores the v15.9 behavior.
      d. Multiply Allen edges between event nodes — surgery BEFORE
         rehab, rehab BEFORE PT visit, ski injury MEETS surgery
         (adjacent), running goal AFTER PT visit. Re-emphasizes
         the "compose temporal flow actively" principle.

  Don't touch the temporal-authority section — that's working. Don't
  regress the decisive default — keep it.

Run: ./dev python3 eval/prompts/build_s1e_v15_11.py
Output: eval/prompts/s1e_v15_11.txt
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 's1e_v15_10.txt'
DST = ROOT / 's1e_v15_11.txt'

src = SRC.read_text()
text = src

# ─── E3: revise Example 4 — trim verbosity + add correction/fact nodes + Allen edges

ANCHOR_E3 = """### Example 4 — temporal authority across the breadth (the wholistic case)

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

## Actions"""

NEW_E3 = """### Example 4 — temporal authority across the breadth (the wholistic case)

Conversation (dated 2025-05-13, conversation_now = 2025-05-13):

*Operator: "Just got back from PT with Sarah at Riverside Rehab.
Started this program in March after I tore my ACL skiing last
winter. PT thinks I can start running again in about a month —
which is wild because I've been off my feet since the surgery
Dr. Chen did on January 22nd."*

*Assistant: "Sounds like you've been recovering since November —
that's a long road."*

Temporal scout candidates (abridged):
```
- "2025-05-13" / "just got back" / source_role: user   ← Path 3
- "2025-03-15" / "in March"      / source_role: user   ← Path 1
- "2025-01-22" / "January 22nd"  / source_role: user   ← Path 1
- "2024-12-15" / "last winter"   / source_role: user / precision: fuzzy   ← Path 2
- "2025-06-13" / "in about a month" / source_role: user   ← Path 2 (future)
- "2024-11-15" / "since November"   / source_role: assistant   ← CONTRADICTS Jan 22
```

Five user-attributed anchors + one assistant-attributed contradiction.
Operator wins — discard the November candidate.

Actions:

remember (the recovery anchor — Path 1, the spine of the arc):
  type: event
  title: "Tom's ACL reconstruction surgery by Dr. Chen"
  event_time: "2025-01-22"
  user_raw_quote: "the surgery Dr. Chen did on January 22nd"
  content: "ACL reconstruction performed on 2025-01-22 by Dr. Chen.
            Anchors the recovery — every subsequent rehab milestone
            sequences against this date. Operator off his feet since."
  situation: "When recalling ACL injury, surgery date, surgeon, or
              any post-surgical milestone Tom references."
  keywords: "ACL surgery, reconstruction, Dr. Chen, January 2025"
  reasoning: "Explicit date from operator (Path 1). Year 2025
              inferable from conversation_now and ongoing-recovery
              framing. Assistant's later 'since November' is
              source_role: assistant + contradicts this user-
              attributed date — discarded."
  connect_to:
    - title: "Tom's ACL tear — skiing, winter 2024-25"
      relation: "met_by"
      why: "surgery is adjacent to (~5 weeks after) the ski injury"
    - title: "Tom started formal ACL rehab program"
      relation: "before"
      why: "surgery preceded rehab start by ~6 weeks"

remember (PT visit today — Path 3, proximal):
  type: event
  title: "Tom's PT session at Riverside Rehab — week 16 post-op"
  event_time: "2025-05-13"
  user_raw_quote: "Just got back from PT with Sarah at Riverside Rehab"
  content: "Routine PT visit ~16 weeks post-surgery. PT cleared
            return-to-running window at ~1 month out..."
  situation: "When recalling PT visits, recovery progress checkpoints..."
  keywords: "PT, Sarah, Riverside Rehab, post-op week 16"
  connect_to:
    - title: "Tom started formal ACL rehab program"
      relation: "during"
      why: "this PT visit happens within the broader rehab program"

remember (rehab start — Path 1, month-only):
  type: event
  title: "Tom started formal ACL rehab program at Riverside"
  event_time: "2025-03-15"
  user_raw_quote: "Started this program in March"
  content: "Formal rehab began mid-March 2025, ~6 weeks post-surgery.
            Specific day not stated; mid-month encoded from 'in March'..."
  situation: "When recalling rehab program structure or events in March 2025..."
  keywords: "rehab, recovery program, Riverside Rehab, March 2025"

remember (ski injury — Path 2, fuzzy resolvable):
  type: event
  title: "Tom's ACL tear — skiing, winter 2024-25"
  event_time: "2024-12-15"
  user_raw_quote: "I tore my ACL skiing last winter"
  content: "ACL tear during skiing in winter 2024-25. Precise date
            not given; mid-December encoded as ski-season midpoint..."
  situation: "When recalling the original injury or ski season experiences..."
  keywords: "ACL tear, skiing injury, winter 2024-25"

remember (running goal — Path 2, future, open):
  type: open
  title: "Tom's running return target — ~mid-June 2025"
  event_time: "2025-06-13"
  user_raw_quote: "PT thinks I can start running again in about a month"
  content: "PT-prognosticated return-to-running window: ~1 month from
            2025-05-13 → ~2025-06-13. Open until confirmed..."
  situation: "When tracking recovery milestones or running ambitions..."
  keywords: "running, return-to-running, recovery milestone, future"
  connect_to:
    - title: "Tom's PT session at Riverside Rehab — week 16 post-op"
      relation: "after"
      why: "running window prognosis was given at today's PT visit"

remember (network atoms — the stable facts the operator named):
  type: fact
  title: "Tom's ACL surgeon: Dr. Chen"
  user_raw_quote: "the surgery Dr. Chen did on January 22nd"
  content: "Dr. Chen performed Tom's ACL reconstruction on 2025-01-22.
            Atomic fact for future recall of 'who was your surgeon'..."
  situation: "When Tom mentions surgeon, ACL surgery providers..."
  keywords: "Dr. Chen, ACL surgeon, orthopedic"

remember (network atoms — the stable facts the operator named):
  type: fact
  title: "Tom's PT: Sarah at Riverside Rehab"
  user_raw_quote: "PT with Sarah at Riverside Rehab"
  content: "Sarah at Riverside Rehab is Tom's physical therapist for
            ACL recovery. Atomic fact for future 'who's your PT'..."
  situation: "When Tom mentions PT, recovery practitioners..."
  keywords: "Sarah, Riverside Rehab, physical therapist"

remember (the trap — source_role discrimination as a graph fact):
  type: correction
  title: "Assistant's 'since November' is wrong — recovery started Jan 22"
  anchor_raw_quote: "the surgery Dr. Chen did on January 22nd"
  content: "Assistant glossed Tom's proximal phrasing as 'since
            November', which would put the recovery start ~6 months
            ago. Operator's own wording attributes the start to
            'January 22nd' (the surgery). Encoded the correction so
            future Anchor never propagates the November date..."
  situation: "When asked about when Tom's recovery started, when his
              surgery was, or whether November is involved in the
              ACL arc — recall this correction to override any
              assistant-paraphrased dates."
  keywords: "correction, assistant hallucination, recovery start date"
  reasoning: "Source_role: assistant on the November candidate +
              direct contradiction with user-attributed Jan 22.
              Created a correction node (not just discarded the
              candidate) so the rejection becomes a durable graph
              fact, not just an in-the-moment encoding choice."
  connect_to:
    - title: "Tom's ACL reconstruction surgery by Dr. Chen"
      relation: "anchored_to"
      why: "the correction defends this canonical surgery date"

The breadth in one example:

- Path 3 (proximal "just got back" → today, 2025-05-13)
- Path 1 (explicit "January 22nd"; explicit-month "March")
- Path 2 (resolvable relative "last winter"; future "in about a month")
- Five dated events, all with event_time
- Allen-edge composition: surgery `met_by` ski injury (adjacent),
  surgery `before` rehab, PT visit `during` rehab program, running
  goal `after` PT visit — the graph is sequenced, not just anchored
- Two `fact` nodes for the stable network atoms (Dr. Chen, Sarah at
  Riverside Rehab) — recall surface for "who's your surgeon"
- One `correction` node — the assistant's "since November" became a
  durable rejection, not just an in-the-moment discard, so future
  Anchor won't propagate it

## Actions"""

assert ANCHOR_E3 in text, "E3 anchor not found"
text = text.replace(ANCHOR_E3, NEW_E3, 1)

# ─── write ───────────────────────────────────────────────────────────
DST.write_text(text)
print(f"v15.10: {len(src):,} chars")
print(f"v15.11: {len(text):,} chars")
print(f"delta from v15.10: {len(text) - len(src):+,} chars")
print(f"delta from v14 (33,358 → {len(text)}): "
      f"{len(text) - 33358:+,} chars "
      f"({(len(text) - 33358) / 33358:+.1%})")
print(f"wrote {DST}")
