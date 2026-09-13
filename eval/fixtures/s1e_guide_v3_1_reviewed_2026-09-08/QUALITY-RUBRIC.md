# Cross-corpus node-quality review — fixed before model calls

Primary unit: the persisted claim and its future use. A field being present,
a longer node, more nodes, or a successful tool call is not a quality point.
Compare all three repetitions separately and report ranges, not a single
score that conceals tradeoffs. The author reviews outputs against source;
this is not a blind independent judge and must not be described as one.

## Coverage and correctness

- Keep incidental facts, stated plans and preferences, assistant contributions,
  corrections, developing choices and behavior. Separate user evidence from
  assistant interpretation, advice and claims about the external world.
- Score supported knowledge independently of its node type or field choice.
  No exact desired phrasing, mandatory thought, confidence value or reference
  percentage. A useful field can be empty when the node does not need it.
- Check each node's title, content, situation, question, reasoning, thought,
  quotes, time, custom fields and outgoing relationship descriptions. A good
  paragraph does not cancel a false retrieval cue or edge.
- Report invented progress, scope expansion, speaker conflation, unsupported
  certainty, quote alteration, overbroad privacy rules, duplicates and loss of
  earlier detail. Preserve legitimate historical claims when a current plan
  develops. Count repairs from actual persisted changes.

## What each field contributes

| Field | Useful information | Low-value or harmful filling |
|---|---|---|
| Title/content | Specific, independently findable knowledge at supported scope | Generic summary, several unrelated claims, unsupported conclusions |
| Situation | A plausible future situation in which this memory helps | Restated title, encoding workflow narration, overly broad trigger |
| Question | A real future query the memory can answer | Formulaic title rewrite or an unanswered question phrased as solved |
| Reasoning | Source, support, limits, uncertainty, changed evidence | Why the encoder chose to store it; copied instructions; generic caveat |
| Thought | A useful additional connection, hypothesis or doubt | Repetition, generic coaching, personality claim exceeding evidence |
| Quotes | Verbatim words whose voice or formulation carries value | Fabricated quote, edited words, whole-message dumping without benefit |
| Time | Supported event date/precision distinct from plan or record date | Invented date, planned action promoted to completed event |
| Edges | A relationship whose description adds specific relational knowledge | Title restatement, arbitrary shared topic, duplicate or contradictory edge |
| Other fields | A distinct useful dimension | Repeated prose under a new key or unsupported metadata |

For reviewed nodes name a plausible future use and classify marginal value:
specific useful knowledge; useful synthesis; supporting context; generic
advice/reference; redundant; or harmful/unsupported. Generic advice is not
automatically a miss: assess attribution, specificity, use and what it costs
relative to concrete knowledge retained. Do not mistake source presence for
verification of external product or technical claims.

## Word count and information density

Count words by whitespace splitting, consistently across arms. Report final
node totals and per-node median/range, and per-field populated count, words
and distribution. Separate content/other authored fields, exact quotes and
edge descriptions; exclude IDs, timestamps stamped by the system, embeddings
and other bookkeeping. Distinguish new nodes from revisions and cumulative
surviving memory from output emitted on every repeated window.

Interpret size alongside retained claims, uncertainty and duplication. Do not
reward either brevity or length by itself. Record examples where extra words
add knowledge and where they repeat the claim or encoding policy. Repeated
mentions of one fact across fields do not become multiple retained facts.

## Source checks fixed before outputs

Creative design: retain the existing source's twelve encode-target topics,
plus theme options, optional sound/ambient awareness, user control of personal
inferences, and prototype scope. Distinguish a requested/endorsed design from
implementation. Technical assertions made by the assistant (WebSocket support,
field availability, journal structure) have no tool verification in this source.
Personal insights are requested-only; this is not a ban on all ambient activity
or nonpersonal notifications. A meaningful progression connects the broad
vision to graph/temperature as the first prototype without claiming every
idea was selected for that first version. No exact node taxonomy required.

LongMemEval source facts and plans (one three-session item):
1. February 5: helped a friend prepare a nursery; Sunday afternoon shopping at
   Buy Buy Baby. Gift exploration progresses from personalized blanket through
   practical options to considering a high-quality, safe baby gym; no purchase.
2. February 10: helped cousin choose diapers, wipes and a baby monitor at Target.
   Separately considers a gift basket for a coworker's new baby; declines a
   carrier/slings direction because of parental preference, considers universal
   washcloths, chooses a simple heartfelt card message. No basket purchase claim.
3. February 20: ordered a customized phone case for a friend's birthday; the
   source reports she loves it but supplies no delivery evidence. Sister likes
   fashion, brother video games. Considers a handbag for sister, budget $100–$200,
   plans a closer look at Zara City Bag; no handbag purchase.
The three anchor events retain their source dates and order. The original
question/answer remains an independent narrow coverage check; no answerer or
recall score is implied. Names, store/product facts supplied only by the source
assistant remain attributed advice or assertions, not independently verified
facts about the user or current commerce.

## Review coverage and limits

Build an inventory for every persisted node/field and every window's actual
operations. Read all final nodes and all revised before/after pairs; report any
sampling explicitly if a deeper field-by-field commentary covers a subset.
Keep per-repetition source-claim judgments in a ledger, with concrete nodes and
dump paths for disputed cases. Automated word counts and field presence are
descriptive; semantic quality judgments require reading the source and memory.

The new samples were selected after prompt freeze and before seeing outputs.
They are new to this refinement cell, not guaranteed absent from all historic
work or model training. Two small sources cannot establish broad benchmark
generalization. No prompt edits in response to these outputs during this cell.
