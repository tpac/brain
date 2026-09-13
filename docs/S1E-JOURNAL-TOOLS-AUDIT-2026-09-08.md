# Journal deferral and tool-description audit

Tom asked whether journals let the encoder defer maintenance indefinitely and
whether MCP descriptions compete with the prompt. This is a read-only audit
of existing captures, source and the earlier stream's hand ledger. No prompt,
tool description, journal input, shared runtime or frozen eval input changed;
no new Sonnet call was made.

## Journal evidence: several different behaviors

| Saved case | Observation | What it establishes |
|---|---|---|
| V3 + cues R3 W3 | `friction · 3f26cbd7 content · swap produced a sentence repetition … minor redundancy, not a false claim; worth a clean swap next touch` | The encoder notices a repair it could make and assigns it to a later encounter. This is explicit deferral, but the issue is minor wording, not lost facts. |
| V2 R1 W2–W3 | `do not mint pattern node until Tom confirms a unifying rule or a fourth confirming instance appears` | A prior storage verdict becomes a condition on later action. Useful integration already exists in edges; the failure is the invented continuing endorsement/count gate, not mandatory absence of a pattern node. |
| V3 + cues R3 W1–W2 | `natural first task next session; no open node created since Tom closed immediately` | The source session ending is used as a reason to defer an open-node decision while the encoder itself is still running. The commitment is in the existing decision's content, so this does not establish that the commitment was lost. |
| V2 R2 W3 | Journal says `poster printed Saturday; no further open item`, but the poster node retains future printing | The model believes the maintenance is complete. This is false closure, not an explicit decision to revise later. |
| Courier/PDF notes before new evidence | Collection unconfirmed; investigation deferred by the user | Legitimate unresolved work. Encoding should preserve that state; it should not invent collection or perform the user's actual project work. |

Direct evidence:

- [V3 R3 W3 journal](../eval/results/s1e_guide_sanity_2026-09-08/v3_titles/repeat3/window3/next_continuity.txt)
  and [actual final response and operations](../eval/results/s1e_guide_sanity_2026-09-08/v3_titles/repeat3/window3/result.json).
- [V2 R1 W2 journal](../eval/results/s1e_guide_sanity_2026-09-08/v2_frozen/repeat1/window2/next_continuity.txt)
  and [W3 continuation](../eval/results/s1e_guide_sanity_2026-09-08/v2_frozen/repeat1/window3/next_continuity.txt).
- [V3 R3 W1 response](../eval/results/s1e_guide_sanity_2026-09-08/v3_titles/repeat3/window1/result.json).
- [V2 R2 W3 response](../eval/results/s1e_guide_sanity_2026-09-08/v2_frozen/repeat2/window3/result.json)
  and [persisted nodes](../eval/results/s1e_guide_sanity_2026-09-08/v2_frozen/repeat2/window3/nodes_after.json).

Read newly emitted Review text alongside harvested continuity: the renderer
also carries older lines. Reappearance alone does not mean the model emitted
the same note again. Nor is resolving a journal subject automatically a claim
that the underlying real-world task happened.

## This mechanism was already investigated by the earlier stream

The earlier stream's authoritative ledger is
`/Users/tpac/AgentsContext/s1e-field-coverage-gold/ab_2026-09-01_03/ops9/ADJUDICATION.md`,
sections “Why we miss”, “Probe result — d034485c” and “Probes — overall”.
Its transcripts are `ops9/probes/d034485c_run{1,2,3}.md` beside that ledger.
These are that stream's experiments, not new experiments here.

That case repeatedly carried a named stale node with “not in catalog — revise
next time it surfaces”. Interviews described this as a wait condition decided
while reading continuity, before applying the later fetch rule. The ledger's
later guide section gives the final cross-arm count: one fetch in 24 runs.
Do not substitute an earlier interim denominator for that final ledger count.

The same ledger records an important interview failure: a cued question induced
an account of a miss even in a run that fetched the node. Operation traces take
precedence over retrospective explanations. Interviews can suggest hypotheses;
they cannot independently establish what controlled a generation.

The current sanity cell differs: the relevant nodes are already in the catalog.
It can expose deferral and false clean judgments, but cannot retest missing-ID
retrieval or establish that catalog rendering is unnecessary.

## Where the current instruction conflict lives

The saved V3 system's ending includes:

- Review: a note to the next run for what actions do not capture, including
  doubt, friction and a forming pattern; an open stays visible until resolved.
- Working strategy: repair remaining supported changes before closing.
- Finishing: a read's results are followed by the write; the write's results
  by the final reply.

The continuity header calls its contents “not a to-do list”. The long-lived
nudge in `servers/trace_contract.py` says resolve or hand up, without explicitly
naming tool action. It starts at open ×5 and therefore did not fire in this
three-window cell. Do not attribute these outcomes to a nudge they never saw.

These instructions permit an interpretation in which a recognized concern
becomes future residue once a write has happened. They do not prove that
interpretation caused every missed field. V2 R3's successful repair round also
shows that the closing language is not an absolute prohibition in practice.

The distinction to teach is between an unknown requiring future evidence and
memory maintenance already supported by present evidence. The former may wait;
the latter has a current tool action. A tentative interpretation can be stored
tentatively without waiting for the person to endorse it. This is a proposed
design principle, not new prompt wording or permission to act on project tasks.

## Tool placement: capture order is not model order

`servers/scales/runner.py` captures a dictionary with keys `model`, `effort`,
`system`, `messages`, `tools`. Its last key is only a list of tool names. The
actual API call supplies `system=`, `messages=` and `tools=` as separate fields.
The full schemas are saved separately in each repetition's `tools.json`.

Anthropic documents the cached prompt prefix in the order **tools → system →
messages**. That contradicts treating the capture's final JSON key as evidence
that definitions come after the timeline. We cannot inspect the provider's
private serialization directly. [Anthropic prompt caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

The documented prefix and our saved components give this useful layout:

```text
Tool definitions
System: authored guide → generated fields → Arc/Review → cue strategy → Finishing
User: preamble → continuity → catalogs → gist → current timeline
Later rounds: assistant tool calls → user tool results
```

Finishing is last in the **system**, not after every component of the request.
In V3 R3 W3's closing round, the last user block is a tool result. No strategy
message is newly injected after that result. The system strategy is still
available earlier in the context. Moving JSON keys would not change this layout.

## What the actual six tool definitions say

Source: [saved tools](../eval/results/s1e_guide_sanity_2026-09-08/v3_titles/repeat1/tools.json).
S1E loads the six selected definitions from `brain_mcp.TOOLS` through
`encode._get_tool_schemas`; the generic interactive memory/setup preamble seen
by this chat is absent from this capture.

The saved pretty-printed JSON is 48,365 characters. Re-serialization with Python
`json.dumps(..., ensure_ascii=False)` is 31,683 characters; these are artifact
size measures, not measured model tokens. Across the six tools there are 117
string-valued description entries totaling 20,781 characters, including repeats.

| Finding | Assessment |
|---|---|
| `brain_batch`: “packed into ONE LLM round”; `connect_to`: a separate call forces a “needless second LLM round” | Correct intent is efficient batching. Combined with finishing language it could be generalized into avoidance of necessary repair/read rounds. Plausible influence, not established causality. |
| `brain_batch` says catalog targets resolve by title; target schema instructs exact IDs for existing nodes | Concrete textual inconsistency. The captured runs mostly used IDs, and there were no edge failures, so it is not an explanation for the observed stale poster reasoning. |
| `thought` says most nodes carry none and empty is correct | A legitimate non-mandatory field contract, also present in the template. It could interact with “pattern forming belongs in Review”, but does not by itself demonstrate suppression. Do not replace it with a thought quota. |
| Rich field descriptions live on `remember_batch`/`revise_batch`; mixed `brain_batch` mostly declares core fields and accepts the rest as open fields | A presentation asymmetry worth inspecting. Fields remain writable; it is not an API restriction. All arms did write additional fields. |
| Repeated swap/edge semantics, immutable fields, storage details and precision instructions | Adds competing reading load. Preserve argument validity, data preservation and ID rules when shortening; deleting essential contract text is not a clean test of verbosity. |

No description explicitly says “next run” or “next session”. The direct
future-deferral wording is in journals and the closing instructions. Tools
may reinforce a broader interpretation of the run; calling them the proven
cause would outrun the evidence.

## Next investigation before revising the candidate

Keep V3 + cues frozen while testing explanations. Rewriting the guide, journal
contract and tool descriptions together would hide which change moved behavior.

First, if new interviews are useful, continue copies of saved conversations
with no-write questions that do not presuppose a miss. Ask for a short account
of what is stored, what remains uncertain, and what supported maintenance is
still outstanding, with citations to visible fields or tool outcomes. Only then
ask which specific instruction supports waiting. Compare the answers with the
saved operations; label them retrospective self-report. Do not present suggested
causes before the first answer.

For causal evidence, compare one factor at a time: journal ownership/deferral
framing with factual content held fixed; or tool-description framing with tool
names, structural schemas, capabilities and essential contracts held fixed.
Keep the guide fixed and score persisted claims and actual repair rounds, not
whether the model repeats the new wording. Any behavioral cell retains Tom's
three repetitions and sequential memory carry within an arm. Scope and cost
should be reviewed before expanding the matrix.

Shared journal header/nudge/closure remain Tom's gates. A tool-description
candidate should first be an isolated eval artifact with the existing schema
checks, not an edit to the shared runtime. The proposed investigation has not
been run in this audit.
