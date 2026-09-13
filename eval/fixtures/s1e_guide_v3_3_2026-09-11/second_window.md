### A later window — a choice folds into its plan, a read firms up, the facts survive

*Reading cue: Carry the decision and the developing read through an existing node.*

On October 14 the catalog shows the plan and the interpretation in full,
under the ids the first batch returned:

```
[plan] "October 17 arrival preparation — agreed checks, sign options still open" (id:49d28ce0)
  Content: For the October 17 print swap at Riverside Annex, Mira agreed to walk the street-to-door route and get the manager's ramp answer. She favors using her board if the entrance has room and keeps a wall sign as an alternative. My proposed entrance sign is not yet a selected arrangement. Her welcome stays unscripted. No card, check or sign placement is reported completed; the invitation cannot promise step-free entry while the ramp answer is missing.
  Situation: Preparing the agreed route/ramp checks or choosing between the conditional board option and a wall sign for the October 17 print swap.
  Reasoning: I proposed the preparation items; Mira adopted the two checks on October 12 but kept the sign conditional and offered an alternative. This establishes intended work and a leaning, not completed preparation or a settled sign choice.
  Their Raw Quote: I'm leaning towards using the board if the entrance has room; keep a wall sign as another option.
  Event Time: 2026-10-12
  Edges:
    [event id:a6b0139d] "October 17 print swap — Riverside Annex ground-floor room booked" this prepares_for — the route/ramp checks and unresolved sign options concern arrival at the booked October 17 print swap; they are planned work, not completed preparation
    [open id:82c41f0b] "October print swap — is the side-entrance ramp available?" this addresses — getting the manager's ramp answer is one of the adopted checks; listing that check does not itself answer whether the side entrance is step-free
[interpretation] "Mira keeps welcomes unscripted and checks arrival logistics" (id:93bf027e)
  Content: At the September 18 open studio and October 12 preparation for the print swap, Mira separated welcome rehearsal from checking how people arrive. She wants room to respond to people, with the layout or entry route checked. I twice expanded 'no rehearsal' into 'no preparation'. This describes her hosting, not her attitude to all structured work.
  Situation: When planning a welcome with Mira or hearing her reject rehearsal: preserve the arrival checks while leaving her words unscripted.
  Question: What does Mira want prepared when she says not to rehearse the welcome?
  Reasoning: Her current correction refers to the earlier occasion. The read confirms the same distinction; recurrence supports this scoped understanding.
  Thought: The useful distinction may be what must be dependable for other people, rather than how much Mira likes planning. How she prepares another kind of work would help me tell.
  Their Raw Quote: Don't rehearse my welcome; do walk the arrival route.
  My Raw Quote: I treated “no rehearsal” as “no preparation” again.
  Edges:
    [correction id:61de80a2] "Mira's 'no rehearsal' applied to the welcome, not the setup" this abstracts — the open-studio incident supplies the earlier instance of the welcome/setup distinction; this interpretation makes that correction usable at the next hosting conversation
```

The window:

```
<other trace="af702c31">Do the route walk first, tomorrow morning, before anything else on the card — if the side door turns out to be a problem, the invitation wording changes.</other>
<me trace="65e402af">Route walk first, then; I'll reorder the card. Can I ask how you prepare other kinds of work? I want to know whether the welcome is the exception or the rule.</me>
<other trace="9d3f1e42">For beginner workshops I rehearse every demonstration — if I muddle the sequence, nobody can follow. My own frames I measure twice and dry-fit before gluing, and nobody else ever uses those. The welcome is the one place I want room for whoever turns up.</other>
<me trace="71c0a8e5">So what other people depend on gets rehearsed or checked, and the welcome stays yours — though the frames say you're careful for your own sake too.</me>
```

Two Bad moves. Bad: put the route-walk morning into the plan's content
only, leaving its title and situation at “agreed checks” — the detail
arrives, the choice of what happens first and why does not, and a reader who
lands on that node still sees an unordered list. Bad: hold the thought at
“may be” when a second activity fits it, or rewrite the interpretation
around the workshops and drop the arrival-route evidence that grounded it.

The lists mark the plan's ordering surfaces stale, mark the interpretation's
scope, basis and thought stale, and name two facts under `new`; `fetch: none`
because both priors are whole:

```
changes: preparation order — unordered checks → route walk first, the morning of October 15, because what the walk finds at the side door decides the invitation wording
changes: understanding — rehearsed workshop demonstrations fit my dependable-for-others read; her own frames get the same care with nobody depending on them
changes: newly known — Mira rehearses every beginner-workshop demonstration; she measures her own frames twice and dry-fits before gluing
targets: 49d28ce0 · agreed, unordered checks → route walk first, reason stated → the order is part of the plan: title stale · content stale · situation stale · reasoning stale; agreement date, sign leaning and both edges still hold: event_time clean · why→a6b0139d clean · why→82c41f0b clean
targets: 93bf027e · tentative hosting read → same shape in her teaching, one exception in her own framing → scope widens, competing reading named: title stale · content stale · situation stale · reasoning stale · thought stale · why→61de80a2 stale; question clean
fetch: none
new: Mira rehearses every beginner-workshop demonstration — her stated practice and its reason; grounds the interpretation
new: Mira measures her own frames twice and dry-fits before gluing — a plain fact; it feeds the competing reading without settling it
```

The write, one `brain_batch`. The interpretation's edge moves with it — the
text after the quoted title on its Edges line is my claim about the pair, and
“next hosting conversation” went stale with the widening, so the swap rides
the same revise with `old` copied from that line. The sign leaning, the dated
corrections and the earlier incident stay untouched:

```json
{"operations": [
  {"op": "remember", "type": "fact",
   "title": "Mira rehearses every demonstration for beginner printmaking workshops",
   "content": "For beginner printmaking workshops Mira rehearses every demonstration, because participants cannot follow a muddled sequence. She contrasts this with her welcome, the one place she wants room for whoever turns up.",
   "situation": "When preparing a workshop with Mira or judging what she will want rehearsed versus left open.",
   "reasoning": "Mira stated the practice and her reason for it on October 14, in answer to my question about other kinds of work.",
   "their_raw_quote": "For beginner workshops I rehearse every demonstration — if I muddle the sequence, nobody can follow.",
   "connect_to": [{"target": "93bf027e", "relation": "grounds", "why": "rehearsed demonstrations are a second activity where Mira makes what other people depend on reliable; this practice carries my read beyond hosting into her teaching"}]},
  {"op": "remember", "type": "personal_context",
   "title": "Mira measures twice and dry-fits before gluing her own frames",
   "content": "For frames on her own wall, Mira measures twice and dry-fits before gluing. Nobody else uses those frames.",
   "situation": "When helping Mira frame her own prints, or weighing how far her care extends beyond what others depend on.",
   "reasoning": "Mira described the routine on October 14. The fact stands on its own; the motive reading it feeds lives on the interpretation.",
   "their_raw_quote": "My own frames I measure twice and dry-fit before gluing, and nobody else ever uses those.",
   "connect_to": [{"target": "93bf027e", "relation": "qualifies", "why": "the same care for frames nobody else uses is the one case my dependable-for-others read does not cover; it keeps the competing reading, care in general, alive"}]},
  {"op": "revise", "node_id": "49d28ce0",
   "reason": "Mira ordered the checks and gave the reason; the order is part of the plan, so it enters every surface a reader could land on.",
   "title": {"old": "agreed checks, sign options still open", "new": "route walk first on October 15, sign still conditional"},
   "content": {"old": "Mira agreed to walk the street-to-door route and get the manager's ramp answer.", "new": "Mira agreed to walk the street-to-door route and get the manager's ramp answer; on October 14 she put the route walk first, on the morning of October 15, before anything else on the card, because what the walk finds at the side door decides how the invitation is worded."},
   "situation": "Carrying out the October 17 print-swap preparation: the route walk comes first, on the morning of October 15; the ramp answer and the still-conditional board-or-wall sign choice follow it.",
   "reasoning": "I proposed the items on October 12; Mira adopted the two checks, and on October 14 set their order and its reason: what the route walk finds at the side door decides the invitation wording. Her 'tomorrow' resolves to October 15 against the conversation date. The sign remains her conditional preference with a wall sign as the alternative. Nothing is reported completed."},
  {"op": "revise", "node_id": "93bf027e",
   "reason": "A second activity fits the read and one does not; scope, basis and thought move together while the hosting facts stay true.",
   "title": {"old": "keeps welcomes unscripted and checks arrival logistics", "new": "keeps her welcome unscripted and prepares what others depend on"},
   "content": {"old": "This describes her hosting, not her attitude to all structured work.", "new": "On October 14 she added that she rehearses every beginner-workshop demonstration so people can follow, and that she measures her own frames twice with nobody else using them. Across hosting and teaching, what other people depend on gets checked or rehearsed; the welcome is where she wants room to respond. This is a read about how Mira prepares, not a rule about all structured work."},
   "situation": "When preparing an event or workshop with Mira, or hearing her reject rehearsal: ready what other people will depend on and leave her welcome unscripted.",
   "reasoning": "Her October 12 correction refers to the earlier occasion, and her October 14 account adds a teaching practice with the same shape, in her own words. Three occasions across hosting and teaching support this read; her own frames are the one case it does not cover.",
   "thought": "Two activities now fit the same distinction, so I hold it as a working read rather than a hunch: when preparing with Mira, lead with what other people will depend on. Her own frames get the same care with nobody depending on them, so care in general remains a competing reading; a preparation she does for nobody but herself, done loosely, would separate the two.",
   "connect_to": [{"target": "61de80a2", "relation": "abstracts", "why": {"old": "usable at the next hosting conversation", "new": "usable the next time I prepare an event or workshop with her"}}]}
]}
```

Returned results, abridged to outcomes and the title deltas:

```
{"total": 4, "succeeded": 4, "failed": 0, "connect_to_failures": 0,
 "results": [
  {"op": "remember", "index": 0, "ok": true, "result": {"id": "5e7c21a9"}},
  {"op": "remember", "index": 1, "ok": true, "result": {"id": "c17f40b3"}},
  {"op": "revise", "index": 2, "ok": true, "result": {"id": "49d28ce0", "deltas": [{"field": "title", "old": "October 17 arrival preparation — agreed checks, sign options still open", "new": "October 17 arrival preparation — route walk first on October 15, sign still conditional"}]}},
  {"op": "revise", "index": 3, "ok": true, "result": {"id": "93bf027e", "deltas": [{"field": "title", "old": "Mira keeps welcomes unscripted and checks arrival logistics", "new": "Mira keeps her welcome unscripted and prepares what others depend on"}]}}]}
```

I read the result as its future reader. Someone who retrieves only the
plan now learns what comes first, when and why, and still sees the sign
leaning, the alternative and that nothing is done. Someone who retrieves only the
interpretation gets the read at its current strength, the case it does not
cover, and what would separate the two readings; the dated corrections and
the earlier incident remain walkable behind it, through an edge whose why no
longer promises less than the node now covers. A first batch that carries
the choice and the developing read needs no repair. My `sweep:` names
`49d28ce0` and `93bf027e`. My Arc line: `route walk moved to first, October
15; the hosting read widened to how Mira prepares for others`.

