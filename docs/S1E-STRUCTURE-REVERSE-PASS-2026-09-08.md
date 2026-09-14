# S1E structural reverse pass — 2026-09-08

Review of the measured guide v1.2 at `9a1727f` on `claude/sweet-lichterman-ba9854`. The initial structural pass and storyboard below preceded prompt authoring. Tom subsequently approved a separate v2 template/gist candidate; its exact diff, checks, and size trade are in [the v2 review](S1E-GUIDE-V2-REVIEW-2026-09-08.md). No default prompt, renderer, contract, runner, or eval code was changed, and nothing was promoted.

## Finding and first recommendation

The request teaches the desired *finished memory* much more continuously than it teaches the *work of arriving at that memory*. The guide's revision procedure supplies an inspectable sequence: identify a change, account for fields, fetch, patch. Creation, interpretation, and learning about the person are spread across prose and mostly finished-node examples. The encoder must assemble those into its own procedure while reading the actual conversation.

The first structural recommendation is to make **one complete encoding episode the organizing example**: prior memory and a mixed conversation → observations and implications → affected claims and new memories → necessary reads → writes → inspection of results → continuity. Rework the existing canonical example toward that shape before adding a new paragraph to `new`. The measured revise roll-call remains part of the episode. This is a proposed organizing principle, not evidence that a replacement will outperform the current guide.

The point is to expose the transformations Sonnet needs to learn. A finished fact, a finished correction, and a finished interpretation do not demonstrate how to notice all three in the same ordinary exchange. Conversely, a prescribed list can execute perfectly while its entries reflect a mistaken decision to skip the important fact.

## Scope and evidence

- **T**: [guide template](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/eval/candidate_prompts/s1e_guide_v1_2026-09-08.md), read across this discussion; line references below refer to this file.
- **G**: [guide gist](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/eval/candidate_prompts/s1e_gist_guide_v1_2026-09-08.md).
- **Assembly**: [system assembly](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/servers/scales/s1/encode.py:401), [user assembly](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/servers/scales/s1/encode.py:743), [request and capture](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/servers/scales/runner.py:489), [result continuation](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/servers/scales/runner.py:725).
- **Literal request**: [gym first round](/Users/tpac/AgentsContext/eval-corpus/f4897d/59524333/payloads/2026-09-08/s1e-ingest-5-5/000-round_payload.json). Its system starts with T exactly; the appendix and user blocks were read separately. This sample has no catalog, so it is not a representative catalog-size estimate.
- **Literal continuation**: [travel second round](/Users/tpac/AgentsContext/eval-corpus/f4897d/edced276_abs/payloads/2026-09-08/s1e-ingest-e-9/001-round_payload.json), including the assistant's lists, its batch, and the returned per-operation results.
- **Existing findings**: [hand ledger](/Users/tpac/AgentsContext/s1e-field-coverage-gold/ab_2026-09-01_03/ops9/ADJUDICATION.md:261), [pre-guide example inventory](/Users/tpac/AgentsContext/s1e-field-coverage-gold/ab_2026-09-01_03/EXAMPLE-SHAPE-INVENTORY-2026-09-07.md). Historical counts in the inventory describe the pre-guide template; guide v1.2 adds the depicted read. No benchmark outcomes were recomputed.

Labels used here: **observed** means visible in code, text, captures, or the existing ledger; **inference** means a proposed explanation or design consequence; **proposal** means untested and awaiting Tom's discussion.

The opening has an existing design rationale (brain nodes `99810ad5`, `ac53cee6`): remember → revise → connect there expresses weight, not execution order; “meaning” and the detail/meaning distinction were explicitly chosen. The process diagram below must not silently reinterpret that stance as an ordering bug. This review does not reopen or edit the settled opening wording.

## What is delivered, and when

The API call has separate `tools`, `system`, and `messages` fields. Anthropic documents its cached prompt prefix as tools → system → messages; a Markdown export placing tools last does not establish that serialization order. [Official prompt-prefix documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

```text
Tool capabilities and schemas
System: template → field reference → Arc → Review → finishing
User: preamble → continuity → optional failed-encode context
      → catalog → guide → timeline
Assistant: prose and tool call(s)
User: tool result(s)
Assistant: next action or final reply
```

The finishing block is last in the **system**, not text appended after the user timeline. The guide is last before the timeline. There is currently no task block after that timeline. Later rounds append tool results and the encoder's preceding replies; this changes the immediate decision context even though the same system remains.

Character measurements are source lengths, not token counts or attention measurements:

| Component | Characters | Basis |
|---|---:|---|
| T | 112,026 | Candidate file |
| G | 4,352 | Candidate file |
| Captured system | 118,794 | Gym request; T plus 6,768 runtime characters |
| Captured user preamble | 160 | Gym request; ordinary preamble |
| Captured user body | 19,222 | Gym request; guide plus timeline |
| Six tool schemas | 30,684 | Compact JSON reconstructed from branch code |

Captures retain full system/messages but **tool names only**. The schema length is a branch reconstruction, not a historical schema capture. Schema capabilities include `remember_batch`, `connect_batch`, `brain_batch`, `revise_batch`, `get_nodes`, and `recall_batch`. The mixed batch also exposes disconnect/archive/absorb, beyond the three acts emphasized by the template. This is a scope observation, not a recommendation to remove capabilities without review.

## Reverse derivation by section

For each section: what decision does it exist to improve, what behavior does its current shape teach, and when can the encoder use that information?

| Section and size | Job and general principle | Current shape and consequence | Proposed disposition to discuss |
|---|---|---|---|
| Opening, T 1–8; 2,035 chars | Establish ownership, continuity, generosity, detail plus meaning. **What is preserved should let a future self know and act.** | Strong first-person stance; already contains node/edge strategy as well as purpose. Similar motivation recurs in defaults, cadence, and coda. | Keep a clear orienting stance. Separate indispensable purpose from repeated implementation teaching; do not erase identity to save characters. |
| What I Receive, T 9–60; 5,881 | Interpret evidence and prior memory correctly. **Input labels should tell me what an object is and what it proves.** | A long legend teaches render syntax, provenance, reading order, anti-twin behavior, and covered-turn restrictions together. `encoded=true` is described as covered substance and excludes fresh atoms from those turns. | Put local interpretation cues on their objects; keep one small map. Investigate whether “already processed” is being treated as “nothing was missed” before changing covered-turn policy. |
| Reading, T 61–162; 5,627 | Detect facts, contradictions, developments, and patterns. **Observation and interpretation have different evidence requirements.** | Corrections get four worked verbal categories; patterns get a hard turn threshold; entity atoms require recurring references. The reader must infer how ordinary first-mention facts relate to these gates. | Make the evidence distinctions explicit in the central episode. Fact capture, pattern formation, and duplicate avoidance must not share one worthiness threshold. |
| Nodes, T 163–398; 13,916 | Make a claim useful outside this window. **A memory preserves a specific claim, its basis, and the circumstances in which it matters.** | Field craft, storage details, type conventions, atomization, selective provenance, and Flat→Rich transformations are interleaved. Richness examples emphasize abstraction and interpretation. | Retain craft with examples that show its payoff; keep mechanical field semantics in a consistent reference. Show a plain fact that is already sufficient, alongside one that supports an interpretation. |
| Edges, T 399–479; 4,433 | Preserve relationships as knowledge and maintain them when false. **A relationship can assert a claim that also needs revision.** | The section mainly teaches authoring a good new `why`. Repair is concentrated elsewhere, while catalog edges visually form a separate block. | Treat relationship claims as part of integration and inspection in the episode. Rendering changes remain a separate Tom gate; prose alone has not moved the missing shape. |
| Temporal, T 480–714; 12,126 | Resolve time and temporal authority; distinguish current state from history. **Interpret time from the conversation's evidence before serializing it.** | A substantial source→date→pseudo-operation example is stronger process teaching than finished nodes alone. Date reasoning and time-anchor construction rules share one section. | Preserve the temporal-authority transformation. Place it with evidence interpretation; leave format mechanics with the field contract. Keep approximate dates explicitly approximate. |
| Actions/defaults, T 715–855; 8,908 | Route knowledge into reads, remembers, revises, and connections. **Choose the memory change, then the tool that expresses it.** | Read/write lifecycle, field patch semantics, worthiness, voice rules, and motivational corrections coexist. Several duplicate the tools and later cadence. “Same topic” is used as a revise cue even though one topic can contain distinct claims. | Give operation selection one coherent home in the process. Distinguish matching a claim from merely sharing a topic. Preserve the patch semantics and field coverage the measured arm uses. |
| Cadence/examples, T 856–1402; 42,865 | Demonstrate execution, including integration with existing memory. **A worked episode should expose the decisions that cause its operations.** | The canonical batch presents high-quality finished nodes without its originating mixed timeline. The ladder teaches patch shapes; the guide sweep depicts plans, a read, and writes. The family does not share one full lifecycle. | Use a complete encoding episode as the organizing example, with smaller contrasts only where they teach a distinct transformation. Keep the revise ladder and roll-call's substance; reconcile competing round narratives. |
| Identity examples, T 1403–1615; 15,290 | Learn from relational experience, self-correction, and the person's behavior. **Concrete experience can change how I understand myself and my collaborator.** | Most examples are completed memories. The second-misreading example uniquely supplies prior catalog plus new exchange and shows the upgrade. The late appendix can imply identity is a special mode; this is an inference to test. | Carry the second-misreading transformation into ordinary encoding alongside facts and actions. Preserve identity and emotional texture; replace repeated output-only teaching only when the capability survives. |
| Closure/coda, T 1616–1622; 945 | Define the finished artifact and restore purpose. **Closing means the work is accounted for, including what actually succeeded.** | Placeholder documentation points to runtime closure; the coda repeats the mission. Elsewhere the guide says the reply after a write is the close, while cadence permits another write and field guidance asks for new connections from returned nodes. | One lifecycle must describe follow-up after results consistently. Review results before choosing closure. Shared contract and runner changes require Tom's gate. |

### Input and runtime sections are part of the same review

| Surface | Job | Structural finding |
|---|---|---|
| Tool definitions | Specify available operations and their exact meanings | Field descriptions repeat across create/revise/mixed schemas. They also teach strategy. Audit agreement with the template before trimming; `get_nodes` changes view with batch size. |
| Runtime field summary | Reference for constructing valid writes | It arrives after the long examples and includes storage implementation detail. Its returned-node advice initiates follow-up work that the guide's “write then close” narrative does not model. |
| Preamble | Initiate the current run | It should point into one operating procedure. Gold and longmem did not receive the same preamble; see the captured-composition correction below. |
| Continuity | Carry provisional unresolved observations and session context | It is encountered before fresh evidence. The ledger demonstrates no-mint verdicts becoming inherited reasons to skip. Prior notes must remain revisable evidence, not policy. |
| Catalog | Supply addressable prior claims for comparison | Full entries, lean entries, edge-only references, and residue-only IDs do not have equal behavioral reach. The ledger's missing-ID result is a presentation boundary, not merely a missing fetch sentence. |
| Guide | Turn principles into the next actions | It exposes field accounting on revises. `new` begins after material has already been selected. The recent proposed replacement wrongly routed existing-node revisions through a list whose execution rule says remember. |
| Timeline | Supply the new evidence and the connections between exchanges | Chronology and speaker/action distinctions matter. A behavior or arc can require integrating old covered text with a new exchange; a first-mention fact can be hidden inside a much longer response. |
| Tool results | Supply the evidence for the next decision | A real saved continuation has 10,587 characters of batch results, including successes and field deltas. That is a new input requiring interpretation, not just permission to emit DONE. No new failure rate is inferred from this sample. |
| Arc / Review / finishing | Hand forward the remaining state and terminate | The shared instructions define output formats and the runner's stopping condition. Their placement at the system tail must not be confused with a post-timeline task query. |

## Conflicts and incomplete transformations to resolve

These are observed textual seams. Their causal effects are hypotheses unless the ledger separately demonstrates them.

1. **Selection is assumed in the creation plan.** G 11 asks for nodes already chosen for minting. The gym/chandelier ledger failures show that writing this list can preserve a bad selection verdict. The desired observable transformation is evidence → memory decision; outputting a list alone is not evidence that discovery was complete. Moves failure patterns 5 and 6.

2. **Conditional richness becomes a universal pairing.** T 5 says detail and meaning *can* stand separately. T 1127 says “The pair is the unit.” The atomization test elsewhere permits separation only when retrieval diverges. A complete example must show when the detail is sufficient, when meaning adds an independent claim, and when an existing node should absorb the new evidence. This could affect over-interpretation and omission; it has not been isolated in an eval.

3. **The scope of the pattern threshold is difficult to transfer.** T 136 requires 3+ distinct turns. T 1536 makes a second occurrence the upgrade signal, with explicit confirmation from the person and earlier catalog evidence. These can be reconciled by evidence scope, but the teaching does not make that distinction operational. Do not convert this into a proposed numeric threshold change; the root question is what establishes an observation, an explicit preference, or an inferred regularity. Moves patterns 6 and 7 if the gate is being generalized to facts.

4. **The exemplar's verdicts outrun its shown evidence.** The sweep lists `b8e05f92 situation stale`, but its source catalog excerpt does not render that situation. The subsequent read does not fetch that node. The replacement situation is then authored. The example teaches the right result but omits the comparison that would justify the field verdict. Supply the actual old field in a future example revision. This relates to patterns 1, 2, and 4.

5. **The main canonical example still narrates an unfinished read.** T 1063 explains that content swaps would follow a get_nodes read, while the demonstrated call only performs title/type updates. The guide sweep now shows the read, but the large canonical example still stops short of it. Preserve its rich node craft while making its input→inspection→write sequence consistent. Relates to patterns 1 and 5.

6. **Round limits and follow-up duties compete.** T 874 forbids a second read. T 886 permits another encoding round. G 3 places the close after the write. Runtime field guidance tells the encoder to connect returned related nodes immediately. The runner actually continues whenever there is another tool call. Define a coherent sequence before shortening any of these. A repair after failed writes and a legitimate no-work close need different demonstrations. Pattern 9; shared mechanism changes remain gated.

7. **Read scope can defeat read depth.** G 9 makes every named but unrendered source unconditional. The branch tool description says over ten requested IDs returns SCAN with shortened content, while at most ten returns DETAIL; `rich=true` is available. “One read, every missing ID” and “write from complete stored words” therefore need a compatible policy. This is a static interface seam, not a claim that it caused the recorded five non-catalog misses. Pattern 3.

8. **Prior processing can become epistemic authority.** The continuity failure is measured. T 59's “not for fresh atoms” on covered turns is a related possible blind spot: the fact that a turn was processed does not establish that every useful atom survived. Investigate with a sequential replay before weakening duplicate avoidance. Patterns 6 and 7.

9. **The tools teach a broader menu than the task narrative.** The mixed batch exposes six operation variants while the template emphasizes three. Useful capabilities should remain available, but unnecessary routing decisions and duplicated semantics are part of the instruction burden. This is a tooling design question, not authorization to remove operations.

10. **Selective provenance has contradictory teaching.** T 334 says most nodes do not need the source flag; T 1093 says most lived nodes carry it and treats session-wide absence as a miss. Tom has resolved the intended principle: selective episodic visibility, not universal coverage. Reconcile during the appropriate section pass; do not make the coverage rate a quality objective.

## Placement: where each kind of teaching earns its position

This is a functional map, not an approved physical reordering:

```text
Stable orientation: whose memory, what preservation is for
        ↓
Prior claims + conversation: objects whose labels explain their role
        ↓
Working transformation: evidence → implications → memory changes
        ↓
Reads where a decision lacks evidence; incorporate what comes back
        ↓
Writes using the field and tool contract
        ↓
Returned results → completion, repair, or a remaining open question
        ↓
Arc and residue that carry only what the actions do not
```

Principles belong where they establish the task's enduring meaning. Local distinctions belong beside the object being interpreted. Tool schemas own exact operation semantics. Examples should enact the transitions between evidence, judgment, and action. The guide should activate that learned sequence against the current input. Closing guidance should describe what remains after actual results.

This does not establish that every instruction should move after the timeline. The existing local experiment supports the pre-timeline slot. Anthropic's general long-context guidance favors placing the query after long documents; that is a hypothesis for a later controlled comparison, not a reason to erase the measured placement. [Official long-context guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#long-context-prompting)

## First discussion topic: a complete encoding episode

The existing canonical batch teaches a rich result; the guide sweep teaches a procedure; the second-misreading example teaches integration across time. The proposal is to make those strengths agree on one lifecycle, using an ordinary mixed exchange as the central episode. It must visibly preserve both a concrete detail and the contextual distinction behind a correction, while showing whether an interpretation is already supported or still developing. Identity remains lived in that work.

This is not a new six-stage list format, a requirement to store every observation, a demand for a fact-plus-principle pair every time, or a decision to split the job into extra model calls. The first artifact to discuss is a storyboard of inputs, transformations, and resulting actions. Only then should the selected examples and adjacent wording change. Replace teaching that the episode subsumes rather than append another independent tutorial.

After Tom settles that shape, validation should compare actual captures with their intended composition; probe mixed and sequential conversations with contrasting evidence states; inspect facts, corrections, arcs, behavior scope, preservation during revision, and results handling; then run the existing gold and longmem comparisons plus held-out cases. Treat read-before-write, source attribution, and no-work windows as contrasts within the same capability. The current guide and best prior candidate remain distinct baselines. No answer-specific entities or benchmark questions belong in the candidate text.

## Challenge coverage and scratchpad follow-up

Tom's follow-up asks whether the existing challenges cover doubt and thought,
and whether the earlier stream's scratchpad proposal was implemented. This is
a coverage crosswalk for the proposed episode, not new encoder instructions
or a claim that the historical checklist's candidate work has landed.

The [checklist method](S1E-CHECKLIST.md#how-we-use-it) already requires a
reverse pass, several compatible challenge boxes per example, and weaving
missing shapes into existing examples. Its A1/A3/A4/A10 laws distinguish
mentioning a behavior from demonstrating it across the example set. E12
requires the recognition trigger to appear in the input; E13 asks whether a
check catches partial completion; E20 distinguishes rule recall from actual
application. Those are the right existing homes for most of this review.

**Doubt and thought are known, incompletely taught shapes.** D9 preserves
contradictory evidence without inventing certainty. E2/E3 gap 6 explicitly
names contested, unresolved knowledge: the examples assert flatly, with one
hedged thought attached to a confident correction. The historical
`thought` 0/120 finding is evidence for A1, not a measurement of today's
guide. The current T has a thought-field explanation and a canonical thought;
its revise examples do not demonstrate maintaining that field. The
checklist's v-next.7 matrix describes candidate additions, including four
thought shapes: a question left open, an idea suggested, a warning for next
time, and a read or curiosity worth following. Its September 4 recount
explicitly leaves those additions to a later template pass. They must not be
credited to this guide. Numeric confidence remains Tom's open decision.

The proposed episode and its smaller contrasts should cover these decisions:

| Decision to demonstrate | Existing challenge home | Evidence the example or eval must expose |
|---|---|---|
| Preserve a stated fact while its broader meaning remains uncertain | D1, D7, D9; E2/E3 gaps 6 and 11 | The fact survives on first mention; uncertainty about a pattern does not become a reason to discard its evidence. Moves guide failure pattern 6. |
| Add a useful personal read without turning it into attributed fact | D3, A1, A4, A10 | Show the source observation, the added connection or idea, and its appropriate scope. Also show an ordinary fact needing no thought. Addresses pattern 8; quality requires semantic review, not counting filled fields. |
| Let a thought change when new evidence changes the read | A10, D11, E12 | Depict the old thought and new evidence, then an appropriate revision. Preserve the still-true observation while changing the interpretation. This is a coverage proposal, not an isolated measured cause of the guide regression. |
| Keep a temporary encoding verdict from becoming future evidence | D9, E12, E13; extension to continuity review | A sequential case must distinguish an unresolved question from a prior decision to skip. A no-mint verdict must not become the next run's authority. Moves pattern 7. |
| Carry a decision through to its actual result | E13, E20 | Compare the working lists with reads, writes, and tool results. A complete-looking plan with omitted operations fails. Moves patterns 5 and 9. |

This does not require every example to carry every row. Each needs a coherent
shape and a coverage row, including what it deliberately leaves out. The
example set should distribute the remaining thought shapes rather than put
all four in one node. Source-reference frequency is not a success criterion;
Tom's selective visibility purpose governs it.

**The scratchpad exists in the branch's guide arm.** G asks the first assistant
reply to write `changes`, `targets`, `fetch`, and `new`, then call the required
tool. The runner returns those replies in `round_texts`, and the eval dumps
retain them. The existing `tools/worklist_kpi.py` in the handoff workspace
compares target verdicts with gold fields and actual operations: absent,
misjudged, or seen but not done. It does not establish that fact discovery was
complete, or that a thought was useful and appropriately grounded.

For the proposed revision, use these short working records to expose evidence,
decisions, and remaining unknowns. Revise what the existing lists account for
before adding a fifth list or a separate scratchpad phase. The example should
show the transition rather than supply a long narrated reasoning transcript
(A8 cautions against CoT-format imitation). Working notes are temporary;
`reasoning` preserves the basis of a stored claim, `thought` preserves a useful
take, and a substantive unresolved question can have a durable `open` home.
The existing review contract separately carries doubts, friction, surprises,
or forming patterns that actions do not capture. These are different purposes,
not interchangeable places to dump the working lists.

The lists-first preamble also exists behind a flag, but the longmem capture
correction below limits what was actually tested together. Production trace
storage of `round_texts` remains a gate. No prompt, checklist law, journal
contract, or runner behavior was changed by this follow-up.

## Storyboard v0 — a print swap, a correction, and a developing read

**Status: historical storyboard, subsequently implemented with corrections in the separate v2 candidate; not behaviorally evaluated.** T and G below still name the original guide files.
This fictional scene was composed for the general distinctions below; its
people, objects, wording, and event are not drawn from the evaluation items.
It is a storyboard, so the catalog labels and operation summaries are author
notation, not a proposed runtime grammar. The eventual prompt example must
use the actual render and tool shapes, with grounded example IDs and the
existing placeholder disclaimer.

### What the episode is trying to teach

An ordinary exchange can change what happened, add something simply worth
knowing, and change how I understand a person. Those discoveries have
different evidence requirements and different destinations in memory. I
account for them before choosing operations, then check the operations'
results. This is the proposed lesson; the storyboard must earn it through
shown evidence rather than announce that the encoder has been thorough.

The scene is preparation for a community print swap. It includes one new
possession fact, a booking update, an adopted preparation plan, and a
correction about the host's wishes. Access remains genuinely unresolved.
The encoder records this conversation; it does not contact the venue or
perform the physical checks.

### 1. What is already in memory

The conversation is dated **2026-10-12**. The catalog contains:

| Storyboard label | State actually shown before the conversation |
|---|---|
| `{event_id}` — an event, fully rendered | Title: **October 17 print swap — Riverside Annex booking pending**. Content: a hold for **2026-10-17, 17:00–19:00**; the room request is awaiting confirmation; visitors may swap prints or simply browse. Situation: preparing an invitation while the venue is still provisional. Question: where and when is the October print swap? Reasoning: Mira reported a hold, not a confirmed booking. Event time: **2026-10-17**. One edge to `{access_id}` says the entry question determines whether the invitation can promise step-free access. |
| `{access_id}` — an `open`, fully rendered | Title: **October print swap — which room, and can visitors enter without steps?** Content: the ground-floor room was requested; booking and the side entrance's ramp availability are unconfirmed. Situation: deciding what the invitation may say about the room and access. Question: which room is available, and is there a step-free route? Reasoning: neither the booking nor ramp availability has been confirmed. No outgoing edges in this fixture. |
| `{incident_id}` — a correction, lean | Only the header and title are visible: **Mira's “no rehearsal” applied to the welcome, not the setup**, dated **2026-09-18**. Its stored words and other fields are not yet available. |

The event's time and browsing option are deliberately still true. Its edge
is also still true. A successful revision preserves them. This supplies
clean fields and a clean relationship alongside stale ones; the example
must not teach that touching a node makes all of it wrong.

### 2. The mixed conversation

These are the complete short source turns for the main episode:

> **Mira:** The Annex confirmed the ground-floor room for the print swap on
> October 17, still five to seven. The manager hasn't answered about the
> ramp at the side entrance, so don't call it step-free yet. I keep a folding
> A3 display board in the blue cupboard at the studio; we can use that for
> the entrance sign.

> **Me:** Great. I'll keep the whole thing spontaneous and drop the setup
> walkthrough along with the welcome rehearsal.

> **Mira:** No — same distinction as last month's open studio. Don't
> rehearse my welcome; do walk the arrival route. I want to talk to people
> as they arrive, without having them struggle to find the door or get in.

> **Me:** I treated “no rehearsal” as “no preparation” again. For the prep
> card: walk the route from the street, get the ramp answer, and position
> the sign at the entrance. Leave your welcome unscripted. The room is
> confirmed; access is still a question.

> **Mira:** Yes, use that for the prep card. Keep the step-free claim out
> of the invitation until the manager answers.

What has *not* happened matters: no one has tested the route, obtained a
ramp answer, printed a sign, or established a general personality trait.
The assistant's concrete plan has been adopted; its mistaken first proposal
has been corrected. The example has evidence for both conclusions.

### 3. The first reply: use the existing four lists

The first list must include newly learned information as well as changes
to the world. “Unknown to memory → now stated” is different from “false
before → true now.” Keeping the `changes` label initially avoids adding a
new output section, but its definition and example must teach this broader
scope together.

Below is the proposed working record, at storyboard fidelity. The complete
example would render real field verdict lines with IDs copied from its
catalog. It would not include these explanatory paragraphs in the reply.

```text
changes: booking — room requested → ground-floor room confirmed; 17:00–19:00 unchanged
changes: access — ramp availability remains unknown; booking does not settle entry
changes: newly known — Mira's folding A3 display board is in the studio's blue cupboard
changes: preparation — my “drop the walkthrough” proposal was rejected; the three-item prep card was adopted
changes: how I read Mira — she distinguishes an unscripted welcome from a checked arrival route; the earlier incident may ground the recurrence
targets: {event_id} · title stale · content stale · situation stale · question clean · reasoning stale · event_time clean · why→{access_id} clean
targets: {access_id} · title stale · content stale · situation stale · question stale · reasoning stale
targets: {incident_id} · title clean · content unread · situation unread · question unread · reasoning unread
fetch: {incident_id} — read the earlier correction before claiming what the two occasions have in common
new: folding A3 display board — blue cupboard at Mira's studio; a first-disclosure fact
new: Mira's welcome/setup distinction — her explicit preference; use the read to establish its history and scope
new: October 17 arrival prep card — the adopted route / ramp answer / sign plan, not completed checks
```

This reply ends with **`get_nodes` for `{incident_id}`**. The read is a
depicted catalog-adjacent read. It does not claim to solve the measured
residue-only fetch failure or authorize the gated catalog-rendering change.

The `new` entry about Mira has sufficient current evidence as a stated
preference. What is pending is the claim about its recurrence, not permission
to preserve her words at all. A thinner read would reduce the historical
claim; it would not erase the current disclosure.

### 4. The read supplies the missing evidence

The returned correction says:

> At the September 18 open studio, I took Mira's “no rehearsal” to mean
> skipping preparation. She corrected the scope: “Don't rehearse my
> welcome. Check the room layout before people arrive.” The setup check
> remained part of the plan.

Its situation is the next time Mira rejects a rehearsal while planning a
welcome. Its reasoning attributes the distinction to that explicit
correction. No `thought` or outgoing edges are present in this fixture.

The read supports a connection across two occasions. It does **not**
establish that Mira dislikes structure, resists being managed, or prepares
every activity this way. The existing incident remains accurate. After the
read, its fields are clean; it needs a relationship to the new understanding,
not another incident node with nearly the same title.

This is the point at which the working record changes: unknown prior evidence
becomes available support for a bounded interpretation. The example must show
that small update before the write, without repeating all four lists.

### 5. The write: three new memories and two revised ones

These are semantic operation specifications, not executable pseudo-API.
The final example will show exact field values and the actual `brain_batch`
shape. Three remembers is what this scene earns, not a taught minimum.

| Operation | What the resulting memory carries | Why this destination is useful |
|---|---|---|
| `remember` — fact | **Mira's folding A3 display board — blue cupboard at the studio.** Preserve object, size, and location in the content and exact source quote; situation concerns finding a display surface or preparing a studio sign. Reasoning says Mira stated that she keeps the board there. It connects to the adopted prep card because the board supplies its sign. No thought or invented emotional register. | A concrete first-disclosure fact is enough. Its retrieval question differs from both hosting preferences and the event's booking. |
| `remember` — interpretation | **Mira keeps welcomes unscripted and checks arrival logistics.** Content grounds the scope in her explicit distinction at the open studio and print swap, including my repeated misreading. Situation is preparing a welcome with her or hearing “no rehearsal” in that setting. Carry her actual correction and my actual acknowledgment as the two voice fields. Connect by `abstracts` to the earlier incident. | Integrates a repeated correction into how I work with this person. It does not mint another copy of the incident or infer a global trait from a single utterance. |
| `remember` — decision | **October 17 arrival prep card — street route, ramp answer, entrance sign.** Preserve the assistant's three concrete items and Mira's adoption, dated to October 12; they are planned work for the October 17 event. Situation concerns preparing arrival or checking invitation claims. Connect to the event it serves and the access question it includes. | My contribution is substantive adopted work. It is preserved with its actual provenance and status, without making a claim that the physical checks happened. |
| `revise` — `{event_id}` | Change the pending-booking title, the corresponding content claim, the provisional-venue situation, and the reasoning that says only a hold is known. Preserve the **17:00–19:00** slot, **October 17** event time, browsing option, lookup question, and still-true access edge. | A booking update repairs every stale rendered claim without rewriting the whole event as a new node or losing unrelated detail. |
| `revise` — `{access_id}` | Narrow the title, content, situation, question, and reasoning to the unresolved ramp / entry question. The room is now known; step-free access is not. Keep type **`open`**. The revised event can carry **`partially_resolves`** to this node because its booking update answers the room part. | New knowledge reduces uncertainty without pretending the entire question was answered. This supplies the existing prose-only partial-closure branch. |

The interpretation's proposed `thought`, deliberately distinct from its
content and evidence basis:

> The useful distinction may be what must be dependable for other people,
> rather than how much Mira likes planning. How she prepares a teaching
> session would help me tell.

This is my tentative reading and a specific curiosity. It is not attributed
to Mira, assigned a numeric confidence, or expanded into a separate
personality node. It earns its place by suggesting what future evidence
would sharpen the interpretation. A following exchange will exercise that
possibility rather than leave “living field” as a claim in the prose.

In the eventual rendered example, the interpretation is a plausible place
for selective `source_refs`: the corrective exchange is useful to revisit
when presenting how I came to understand Mira. The board fact needs no scene
flag merely to satisfy coverage. Any illustrated refs must be copied from
trace IDs actually shown in that example's source, under the existing
placeholder discipline. Automatic action provenance is a separate channel.

### 6. Results determine the close

The example's tool result must show which five operations succeeded and the
resulting field changes. At storyboard stage, this is a semantic result
summary; the final text must use verified current tool-result formatting.

The closing check accounts for the board fact, the scoped interpretation,
the adopted prep card, every stale event/access field, and the unchanged
time and browsing option. It reads those outcomes from successful operations
and deltas. It does not use the presence of five planned operations as proof
that they happened.

The remaining access question already has an `open` node. The tentative
reading already has its `thought` home. They need not be copied into journal
residue. The close does not turn either into a “no-mint” policy for later
windows. Arc/Review formatting follows the existing contract.

A short BAD contrast beside this result can show a list containing the
board fact while the returned batch has no remember for it: that is an
omission despite a complete-looking list. This contrast teaches recognition
of partial work. It does not add the gated runner-side continuation or
promise automatic recovery from a tool-less reply.

### 7. Small follow-up: a thought changes, and the new fact also survives

In a later window, the interpretation above is now fully visible in the
catalog, including its exact prior thought. Mira says:

> For beginner printmaking workshops I rehearse every demonstration. If I
> muddle the sequence, people can't follow. The welcome is where I want
> room to respond to whoever turns up.

Two resulting operations keep the distinction honest:

1. **Remember the new stated practice:** Mira rehearses demonstrations for
   beginner printmaking workshops so participants can follow the sequence.
   Keep the exact quote and that teaching context. This is a new fact about
   her practice, not merely fuel for my interpretation.
2. **Revise the existing interpretation's `thought`:** the curiosity now
   has evidence. A candidate replacement is: “Rehearsing a demonstration
   can serve the same purpose as checking an arrival route: make the parts
   other people depend on reliable. Her welcome leaves room for response.
   That is a more useful distinction than spontaneous versus structured.”
   The original title and content about welcomes and arrival logistics
   remain true and need no patch. The new teaching fact's remember carries
   a specific `grounds` connection to the existing interpretation, making
   the basis walkable. The thought revise targets only that existing node;
   it does not try to resolve a newly minted sibling through revise.

The thought remains my synthesis; the new teaching fact supplies the
speaker's own account. The operations demonstrate a thought change without
a title/content rewrite, and preserve a fresh detail rather than consume it
into an abstraction. This is a compact second window of the same example,
not another full tutorial or a required extra encoding phase.

### Reverse pass and challenge coverage

| Beat | What its actual shape teaches | Tempting wrong move | Existing boxes and remaining risk |
|---|---|---|---|
| Board disclosed once | A concrete fact can be useful before any pattern exists | “Single session; wait for repetition,” or hide the board inside the event summary | D1, D7, D13; E2/E3 gap 11. Moves pattern 6. Risk: a conspicuously useful object is easier than a detail buried in a long exchange; held-out testing must include the latter. |
| Booking and access separate | Evidence can settle one claim while leaving another open | “Ground floor confirmed” becomes “step-free confirmed” | D9, D11, E12; partial-closure gap. Moves patterns 1/2/4. This is incomplete evidence, not conflicting sources; a contradictory-sources example is still needed elsewhere. |
| Correction plus earlier incident | The catalog lets a current correction inform a scoped understanding of the person | Another incident twin, or a sweeping personality claim | D3, D12, D13; T6, E24. Supports person-understanding coverage. Risk: explicit user confirmation is easier than an unspoken arc; do not credit this example with teaching that harder case. |
| Read before historical synthesis | A missing body is missing evidence; read it before asserting its contents | Infer the old incident's full meaning from its title | T8, A1, E12, E20. Moves pattern 5; does not solve residue-only invisibility (pattern 3). |
| Adopted assistant plan | Both voices can contribute specific knowledge, with different roles | Drop my substantive plan, or store any generic recommendation as a fact about Mira | D3, A4. Moves pattern 8. Adoption is evidence for this *joint plan*, not a new universal permission rule for encoding my own findings. |
| Thought and its follow-up | My own interpretation can be useful, provisional, and revisable | Attribute the thought to Mira, add it everywhere, or revise the thought while losing the new teaching fact | A1, A10, D7, D9, E12. One thought shape exercised across time; the whole-set distribution audit remains open. |
| Results and continuity | Work is complete when the memory changes succeeded; remaining uncertainty has an appropriate home | Close after lists, or carry “not worth minting” as authoritative residue | E13, E20. Moves patterns 5/7/9. The primary scene does not itself replay poisoned residue; that remains a sequential evaluation contrast. |

Global checks for this asset: A2 true/false contrasts; A3 consistency with
adjacent prose; A4 unintended defaults; A5 safe identifiers; A6 domain spread;
A7 independence from eval items; A8 concise working records; A9 no invented
claims presented as measured; A10 whole-set field distribution; B placement;
T2/C tool and contract compatibility. These are review obligations, not boxes
claimed closed because the storyboard names them.

Two scope limits matter. This explicitly confirmed preference cannot stand
in for discovering a behavior neither voice has named. And its unresolved
access question cannot stand in for preserving conflicting testimony. The
existing challenge gaps for both remain visible. Smaller supporting examples
must carry them; adding those lessons to this conversation would overpack it.

### Replacement map and size discipline

| Current carrier | Proposed disposition when writing the candidate |
|---|---|
| Canonical introduction, catalog, six-node batch, and explanatory recap, **T 952–1094** | Replace their organizing shape with this complete episode. Preserve field craft, mixed operations, numbers and quotes, meaningful connections, and selective fields in the new payloads. The episode's source and read replace commentary that currently assumes the missing evidence. |
| Existing canonical's independent action-derived finding and standalone old-to-old `connect` | Retain as compact supporting contrasts: this scene does not teach knowledge derived solely from an action result, or a connection whose endpoints neither change. Do not silently claim voice equality is fully taught by an adopted plan. |
| Existing canonical's rich moment and quote/meaning carriers | Keep their distinct lessons in the surviving example set, using the identity and detail/meaning examples where they genuinely overlap. Before cutting text, the inventory must identify the surviving carrier for the emotional moment and the phrase whose weight is not repetition. Naming “covered elsewhere” without identifying the carrier is insufficient. |
| Second-misreading example, **T 1513–1564** | This episode reuses its prior incident → current correction → scoped interpretation transformation. Initially keep it as the lexicon-specific contrast; shorten it only after comparing what its language lesson adds. The new scene does not justify deleting that distinction automatically. |
| Guide sweep, **T 1207–1402** | Retain the measured field roll-call and read/write carrier. Its repair distribution and edge case are not replaced by a booking example. Rendering its omitted old situation is an existing review item; edge-repair teaching remains Tom's gate. |
| G `changes` | Revise the current definition and its example to include newly learned facts and newly understood meaning alongside state transitions. Do not add a fifth list or confuse “new to memory” with “newly true.” |
| G `new` | Revise selection after the evidence pass: known facts, adopted plans, and supported interpretations have different grounds. Remove the universal “my delivered recommendation is a node” implication. Preserve first-class treatment of my own actual findings and ideas. |
| G `targets`, `fetch`, and write mapping | Preserve the working grammar and field accounting. Show how fetched evidence updates the plan before the write. This does not decide any renderer or runner gate. |
| `thought` explanation and nearby retention language | Reuse the existing field meaning. Let the follow-up teach maintenance; align contradictory wording only where the completed example exposes a conflict. No new mandatory field or numeric confidence rule. |

**Budget for the eventual candidate, not measured lengths:** aim to fit the
complete episode, small follow-up, and surviving canonical contrasts within
the current canonical's **16,924 characters**. Keep G near its existing
**4,352 characters** through replacement. The storyboard and this review
apparatus do not enter the encoder prompt. If the authored replacement
cannot fit while retaining necessary teaching, show the actual overage and
the displaced carriers before calling it an improvement. Concision alone
does not justify deleting a working example.

Conceptually new: a depicted mixed source, transitions into the working
lists, a returned read, outcome inspection, and a second window that updates
a thought. Reused: the field craft, first-person stance, distinction between
detail and meaning, revision sweep, partial-open semantics, and the earlier
incident's role in understanding a person. This is substantial example
rewriting, with bounded guide revisions; it is not an appended instruction
layer.

### What this draft establishes, and what it does not

The storyboard supplies source evidence for each planned memory change and
names the surviving example obligations. It does not establish model
transfer or improved benchmark performance. The next authoring step is to
turn this shape into a separate candidate while keeping placement fixed,
then re-run the example inventory and reverse pass on the actual text.
Captured-composition checks precede any gold or longmem run. A behavioral
evaluation needs independent mixed and sequential cases, including unspoken
behavior, contradicted claims, poisoned residue, mundane facts buried under
load, legitimate no-work windows, and partial execution. Evaluating on this
teaching scene would test imitation, not generalization.

## Captured-composition correction and gates

Handoff-gap **2d250d4b** records an observed composition error. All 23 saved first-round f4897d S1E captures carry the ordinary 160-character preamble. The runner script exported `BRAIN_S1E_LISTS_PREAMBLE=1` before `./dev`, but `./dev` sources [brain-env.sh](/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242/hooks/scripts/brain-env.sh:184), which unconditionally resets it to `0`. Gold's six patched captures do carry lists-first. Historical outcomes stay as adjudicated; the composition label must be corrected. Future use needs the flag set after that environment initialization and a captured-request check. No runner or env file was edited here.

Handoff-gap **86e96608** records the other correction: rules ranked never first in 9/9 probe interviews and last in 6/9, not last in 9/9.

Still Tom's gates: promotion/merge; shared closure/header/nudge changes; rendering residue/edge IDs into the catalog; continuation after a nonempty plan with no tool call; production round-text trace storage; edge repair mechanism vs example; confidence-to-label contract changes. No IsolatedBrain or new eval was launched for this review.

## Initial structural-pass verification stamp

At the start of this pass the worktree was clean with `9a1727f`, `069f22c`, `bb14326` as its top three commits. The only intended repository change from this pass is this review document.

- T SHA-256: `0560ffe825b6bdf30634a627129a57e5bfa09871949d6f9281a34935886a1e64`
- G SHA-256: `cc57e3ea8a595db0c2aefabb6d7f75284656135b7dcd7fc8dfe7c7670f18dddd`

The section sizes sum to 112,026 characters. They identify where teaching is concentrated; they do not measure attention or prove that a shorter candidate is better.
