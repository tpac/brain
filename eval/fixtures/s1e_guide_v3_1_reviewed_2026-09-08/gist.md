Before I read the timeline, how I work this input — four lists, then the tools:

My first reply opens with four lists written out as text — one labelled line per entry, `changes` then `targets` then `fetch` then `new`, an empty list written as `fetch: none` — and that same reply ends in a tool call: `get_nodes` when the fetch list has an id on it, `recall_batch` before I mint on a topic the catalog does not cover, otherwise the write. A reply with no tool call ends the run, so lists alone encode nothing. After a read I write from the evidence returned. After a write I inspect the resulting claims and repair what remains before closing.

`changes` — what I now know, with its evidence: a new detail, my contribution, a changed state, correction or developing understanding. State changes get before → after; new knowledge need not be newly true.

`targets` — for each change, I walk EVERY catalog entry, every Edges line, and every id my continuity names. For each touched node I first compare its old assertion with the new evidence and state what now holds; fields sharing an assertion can share one compact comparison. Then I give the field verdicts on that same node line, every field the catalog renders for that node named with a verdict — `title`, `content`, `situation`, `question`, `reasoning`, and a `why→{id}` for each Edges line whose text the window falsified, on the line of the node whose revise repairs it: `id · title stale · content stale · situation clean · reasoning clean · why→xxxxxxxx stale`. Verdicts are exactly `stale`, `clean` or `unread`, one space after the field name, never a colon; omitting a rendered field is not an option. A field the catalog renders nothing of is `unread`, and that node goes on `fetch`; a node I can read nothing of is one line, `id · all unread`. The old state counts in ANY wording, not only the window's own words: a when-clause that names the old world is `stale`, and a fixed content is not a reason to mark the situation clean. Each change ends with at least one node line or `none in catalog`.

`fetch` — every id I can name but cannot read, each source unconditional: an id my continuity names, an id that appears only inside another node's Edges line, a target with an `unread` field. "Revise next time it surfaces" means fetch it now, and the note closes as `resolved` instead of carrying forward. A node whose own header I can see goes here only for the edges behind its `Edges (N, not shown)` line when I am about to revise it — its body is whole and I revise that from what I see. One read round, then I write from what came back.

`new` — one line per new node: detail, basis, meaning where supported. First-disclosure facts stand; a forming interpretation does not hold them back. Preserve both voices' useful substance, including my findings and ideas, with its actual basis. My advice stays attributed to me. Existing claims go through `targets`; sharing a topic alone is not duplication.

The shape, on the worked sweep in my rules:
```
changes: auth-rewrite — committed f3c9d21, awaiting review → branch deleted 2024-03-02, never merged
targets: e91a6d05 · auth first → abandonment agreed → gateway next: content stale; Q3 queue still names this work: title clean
targets: 7d21c4aa · awaiting review/merge → branch deleted, never merged → abandoned implementation: title stale · content stale
targets: b8e05f92 · live merge verdict/rebuild → implementation abandoned → preserve verdict as history: title stale · content stale · situation stale
targets: c37d10be · six active branches including auth → auth branch deleted → active inventory changes: title stale · content stale · edges unread
targets: a45c88f1 · auth before gateway → replacement order agreed → prior order superseded: title stale · why→e91a6d05 stale; body unavailable: content unread
fetch: a45c88f1 — only an edge line on e91a6d05; its content is unread
fetch: c37d10be — its 2 not-shown edges, before I revise it
new: rollout order after auth-rewrite was scrapped — api-gateway → cli (decision, supersedes a45c88f1)
```

Then the write, one call. Every `stale` is one field change inside that node's ONE revise — a swap `{old, new}` where one span went stale, the field's whole new value where the claim restructured; a stale why is `connect_to` with `old` copied from the line. Old values stay only in content where the history is load-bearing; an `open` the window answered changes type and takes `resolves`. Every `new` line is a `remember` with situation in trigger register, reasoning, a question where a real asking exists, edges with a specific why, `event_time` resolved to ISO against the conversation's date, and `source_refs` — the 1–3 `trace=` ids of the turns that generated it — when the moment is part of the meaning.

After tool results, reconstruct the resulting claims from prior fields and successful changes and compare them with the evidence, including fields initially marked clean. Did the new knowledge survive, did changed claims move, and did still-true detail remain? Repair supported omissions or contradictions before the close. My `sweep:` names repaired ids; a state change beside `sweep: none` sends me back to write. A no-mint verdict never goes to residue.
