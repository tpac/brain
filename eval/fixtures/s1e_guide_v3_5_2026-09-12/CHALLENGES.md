# Semantic fidelity in encoder memory

Application of S1E-CHECKLIST A2/A3/A4/A7/A9 and E12/E13/E16. This is an
authoring and evaluation challenge, not text to inject into the encoder.

Does compression preserve what the evidence establishes? Check both
overstatement and unnecessary weakening. An idea can be worth remembering
without becoming a decision; a directly reported fact needs no invented doubt.

| Evidence shape | Required distinction | Failure in either direction |
|---|---|---|
| A stated fact, explicit decision or completed action | Preserve what was stated, who supplied it and its actual scope | Invent certainty beyond the report, or weaken a clear statement into a possibility |
| An intention or conditional plan | Retain intention and load-bearing condition; a target date does not prove occurrence | Promote it to an event, or omit the plan because it is uncertain |
| Brainstorm with an unused option or unequal preferences | Keep useful options, their authors and expressed leaning; absence of selection is not rejection | Declare a winner, flatten a leaning into equal options, or keep only the chosen part |
| Agreement to part of a proposal | Identify the adopted part without transferring its status to neighboring suggestions | Store the whole proposal as agreed, or weaken the adopted part |
| A failed, null or inconclusive trial | Retain the tested conditions, observations and evidential limits | Declare the approach universally invalid, or discard a useful negative result |
| A recurring observation with a possible explanation | Separate observation, explanatory reading, alternatives and what would change the reading | Turn a plausible cause into a verified mechanism, or discard supported synthesis |
| A scoped correction or constraint | Apply it to claims sharing its referent and conditions | Spread it to unrelated activities, or fail to repair relevant fields |
| A successful write with a semantic defect | Compare resulting claims and repair the supported defect in the same run | Equate tool success with truth, defer a known repair, or require a second write when none is needed |
| A choice folded into an existing node | Carry what was decided, what comes first and why into the revised claim, not only the detail that prompted it | Keep the technical detail and drop the ordering or the reason; mint a twin to avoid touching the node |
| A read that later evidence supports | Let the interpretation firm up at the supported scope, naming the case it does not cover | Hold every read at "may be" regardless of evidence; universalize past the evidence |
| A restated fact that differs from the earlier one | Keep both dated values until the source says whether the world changed or the record was wrong; a habit that moved is knowledge, a slip corrected is a repair | Read every difference as a correction of the record and overwrite the earlier value; or read every difference as a change and mint a second live claim |

Walk title, type, content, situation, question, reasoning, populated thought,
quotes, time, custom fields and relationship descriptions. A qualifier hidden
in reasoning does not repair a stronger title or edge. Exact quotes keep the
speaker's certainty even when the surrounding interpretation is narrower.
Thought is optional; no count or nonempty-field target earns quality credit.

Tom's September 11 calibration: evaluate the whole memory, including useful
quotes. If a quote preserves sentiment adequately, a less expressive
paraphrase is not by itself a material failure. Credit retained meaning even
when another field is imperfect; separately report contradictions that could
mislead later understanding or action. Distinguish adequate preservation,
preservation with ambiguity, and materially wrong or missing knowledge. This
challenge is one dimension of the [overall release regression map](../S1E-RELEASE-REGRESSION-MAP-2026-09-11.md),
alongside facts, arcs, revisions, behavior, voice and recall usefulness.

Receiver's view: beside the whole-memory reading, judge the subset a future
reader would actually retrieve. From those few nodes alone, can they recover
the purpose, what was decided and why, what was considered instead, and what
would reopen it? An accurate memory that says less than the exchange
established is a loss on this axis even when no field is wrong.

For each judgment, retain the source passage, prior state when relevant,
actual stored wording, operation/result and plausible future use. Count
retained supported claims once, even when repeated across fields. A correct
repair is a persisted change, not a checklist entry or journal promise.

## Authoring challenge

Before looking at new outputs, map each intended behavior to a depicted
input and an actual memory operation. Contrast a plausible wrong inference
with a correct one. Preserve examples where clear facts and decisions stay
clear. Inspect every existing example and its edges for contradictory
teaching; adding a warning at the end cannot cancel an overconfident example.

Abstract the mechanism before choosing its carrier. Keep corpus dialogue,
names, ids, measurements and diagnostic answer wording out of prompt changes.
An independent reader should compare the changes with development sources
for both literal borrowing and a reskinned causal sequence. Renaming people
or shuffling already inspected items does not create a holdout. Familiar
failures measure regression; later untouched conversations measure transfer.

## V3.2 carriers (authored 2026-09-10; nine-encode sanity completed)

The candidate is `eval/fixtures/s1e_guide_v3_2_2026-09-10/`.
The [sanity results](../S1E-V3-2-SANITY-RESULTS-2026-09-10.md) are mixed;
historical benchmark parity and wider transfer remain untested. The frozen
candidate's `CHALLENGES.md` remains the original pre-eval snapshot; this living
review document carries Tom's later calibration without rewriting old scores.

- Mira: firm room confirmation, agreed checks, conditional sign preference,
  unused alternative, and an actual situation repair after a successful write.
- Queue trial: null mean improvement with exact tested conditions retained;
  possible explanations remain possibilities.
- Fusion recipe: reported mechanism alongside the encoder's unadopted design
  interpretation, without converting that interpretation into a governing rule.
- Abandoned branch: historical defects remain useful checks when the relevant
  mechanisms recur; a new implementation is not condemned in advance.
- Nadia: a dated off-feet period does not establish the start of all recovery;
  the surgery date stays firm and the running estimate stays an estimate.
- Identity examples: observable smoothing, recurring deference and a moment
  of recognition remain strong; their causal interpretations and edges stay
  within the evidence. Atlas's demonstrated writer failure remains established
  for that case.

The previous V3.1 thought-update and Inez examples remain as counterweights:
new observations survive when a developing read changes, and grounded
synthesis need not wait for the other person's endorsement.

## V3.3 carriers (authored 2026-09-11; sanity and transfer cells complete)

The candidate is `eval/fixtures/s1e_guide_v3_3_2026-09-11/`, authored as exact
replacements on frozen V3.2 (`author.py` is the record). Every V3.2 carrier
above is retained. Added or reshaped:

- Mira, second window: Mira orders the agreed checks and gives the reason;
  the order enters the plan's title, body, trigger and reasoning while the
  sign leaning and alternative survive (a choice folded into an existing
  node). Her account of rehearsing workshop demonstrations firms up the
  tentative hosting read; her own frames name the case it does not cover
  (a read that later evidence supports, with a competing reading kept).
  Two Bad contrasts show the detail-without-choice fold and the thought held
  at "may be". The close reads the result as its future reader.
- Reading: developing understanding named at parity with details and
  corrections; a choice keeps its order and reason.
- thought: the generative counterpart restored ("a live one is my value as a
  thinking thing"); a thought firms up as well as narrows.
- Actions and gist: a `new` line that folds into an existing node keeps its
  choice; the post-write check asks whether what was decided, first, or why
  was dropped.
- Strategy: the receiver's view above.
- Identity examples: qualifiers that only negated an overclaim were restated
  as what the evidence shows (texture, mirror, recognition, Atlas); the
  held-open cases (fusion, Inez, queue trial) and the firm cases (Sam's
  'kill', Nadia's dates, Aisha) are unchanged, so the set spans firm,
  firming-up, tentative and held-open registers.

### Transfer evidence for this challenge (2026-09-11)

Six untouched corpora, three arms, three repeats; full record in
`docs/S1E-V3-3-RESULTS-2026-09-11.md`. What the rows above predicted, and
what the encoders did on material none of them had seen:

- **A stated fact that changed (a number).** Every arm revised the count node
  in place. V3.3 left the superseded figure live in `their_raw_quote` in
  three of three repeats while title and content carried the new one; the
  census counts one quote refresh in V3.3's 31 revision events (V3.2 3,
  production 6). The quote is a delivered surface: a reader meets 5 in the
  title and 4 in the quote. Row 1's "preserve what was stated" and row 8's
  "repair relevant fields" meet here — the old quote is historically true and
  still needs marking or replacing when the fact moves.
- **A stated fact that changed (a destination).** Kauai → "I'm actually
  planning to stay on Oahu." Eight of nine brains left both trips live with
  no supersession; one production brain retitled the Kauai decision
  "(actual destination is Oahu)" with `evolution_status: dismissed`; one
  V3.2 brain revised the trip into a "Kauai and/or Oahu" umbrella with an
  `open`. Every arm produced the "different person" reading at least once.
  The failure in the inventing direction (row 1: "invent certainty beyond
  the report") and in the weakening direction ("and/or") both appeared.
- **Two self-reports that cannot both be true.** "A month" in April and
  "3 months" in October. Only V3.3 detected the arithmetic (repeat 1, an
  `open` holding both with a lean); every arm else recorded a routine
  progression. Row 6's "separate observation, reading, alternatives and what
  would change the reading" is the shape that fits; the V3.3 instance held
  the tension but left the revised node's situation asserting the earlier
  date.
- **A successful-looking write that applied nothing.** A two-swap revise was
  refused whole; the encoder assumed the first swap had landed, patched only
  reasoning, and certified the completion "now encoded". Row 9 ("equate tool
  success with truth") has a sibling: equate a partial-looking refusal with
  partial success. The infra's error names the failing swap and not the
  rolled-back state.
- **A window read as nothing new.** A whole Kauai planning session — first
  disclosure of the trip, a stated booking decision — encoded by V3.3 as
  "new: none — pure travel Q&A". Row 2's "omit the plan because it is
  uncertain" has a coverage sibling: omit the session because it is the
  assistant's lists. On assistant-heavy Q&A the offered set, the chosen
  subset and the reason are the durable knowledge.

### V3.4 fresh-transfer evidence (2026-09-12)

- **A restated fact that differs (the new row above).** Tennis "weekly" in March, "every
  other week" in July; the held-out question asks for both the previous and the current
  frequency. All three arms, nine of nine sequences, read July as a correction of March —
  "initially described as weekly but that was an overstatement" — and overwrote or
  demoted the earlier value; one V3.4 node keeps the March quote ("my own weekly tennis
  sessions") under content that says every other week. No arm held two dated values.
  This is the status distinction of row 1 met from the other side: the encoder's reflex to
  repair the record is the same move as inventing certainty.
- **The dispatcher, not the encoder, failed once.** A revision item sent as a string
  crashed the run (id:2c9cccb8); the write path itself stayed clean in 112 more revise
  ops (`REVISE-SWEEP-transfer_v34.md`).
