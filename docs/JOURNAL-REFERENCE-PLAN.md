# Journal state and native tools

Status: implemented in `codex/journal-continuity`, based on `80d6246`; not merged.
The operator approved native tools, no additional Healer/Aspect model round,
and a five-minute cache breakpoint covering their entire prompt. This replaces
the earlier fenced-JSON Review transport. Merge to production main can activate
the code automatically through the daemon fingerprint check.

## Approved state and operations

One current journal view delivers the latest version of each selected item:

```text
JOURNAL — existing state
```
```json
{
  "items": [
    {
      "id": "journal_a1b2c3d4",
      "persist": true,
      "runsPersisted": 5,
      "reviewDue": true,
      "subject": "deployment",
      "text": "Verify the loaded source fingerprint."
    }
  ],
  "omitted": 0,
  "outsideSelection": 0
}
```

Optional labels use the encoder's own vocabulary. Undelivered feedback and
coverage failures are explicit. Empty state without notices renders nothing.
Complete items fit within the unchanged 8,000-character cap; Community's
48,000-character soft batch cap is unchanged.

The native `journal` tool accepts an `operations` array:

| Operation | Fields | Meaning |
|---|---|---|
| `note` | `subject`, `text`, optional `persist`, `label` | Independent observation; persist defaults false. |
| `edit` | `id` plus supplied `subject`, `text`, `persist`, `label` | Partial edit of the shown item. Empty label clears it. |
| `tell` / `ask` | `subject`, `text` | Reach live work through Thalamus. |
| `withdraw` | `subject`, `reason` | Withdraw the producer's live message independently. |

Every operation includes `op`; edits require a mutable field. The tool schema
and validator derive from the same operation fields. Invalid operations are
reported independently. There is no private resolve command. `persist:false`
allows aging without claiming that the underlying concern was settled.

Identity, `runsPersisted` and `reviewDue` are runtime-owned. Subject edits do not
change identity. Persistence advances once per invocation, including omitted
items, never per batch or retry. Text edits do not reset it; repeated toggles
cannot count an invocation twice. At five, reviewDue invites reconsideration
without expiring anything or forcing a message.

S1 purpose:

> I can sense how the conversation is progressing, but it can end abruptly.
> I can’t assume there will be another turn, or another chance to encode.

S2 purpose:

> My journal carries observations across runs of this work. Each run can develop,
> change, or settle what earlier runs noticed.

The prompt keeps judgment and purpose; the tool description carries mechanics.
No change means no journal call. Observations reflect what is known when the
call is authored; a proposed action is not a confirmed outcome. No mandatory
loose-end node, extra reflection round or end-of-session routing was introduced.

## Ownership and execution

- `trace_contract.py`: state rendering, tool schema, operation validation,
  strategy text, limits and pure historical adoption.
- `JournalBinding`: invocation selection, shown-version receipt, scoped tool
  dispatch, routing through Thalamus, separate Arc harvesting.
- `brain_traces.py`: current state, semantic validation, partial edits,
  append-only snapshots and feedback. `journal_notes` reads raw history.
- `TraceDAL`: SQL, checked identity allocator, single-snapshot state read,
  idempotent invocation/checkpoint markers and guarded writes.
- Generic runner: provider requests, caching, tool-loop execution and optional
  auxiliary terminal tools. It knows no journal field semantics.

The journal tool is encoder-scoped, not an unscoped public MCP command. Callers
cannot author session, unit, producer or invocation attribution. The binding
adds its schema and dispatcher together at the runner boundary.

| Encoder | Native path | Preserved behavior |
|---|---|---|
| S1 Scribe | `encode.py` passes `journal.bind_tools(...)` to the shared loop. | Session-scoped, XML-escaped continuity; Arc harvested independently. Failure notes use the journal owner. Main/gist prompts distinguish journal and node IDs. |
| Community | Each batch binds the shared journal tool; continuity refreshes before packing. | Proposal packing, membership reconciliation and rejection fingerprints remain driven by actual task outcomes. |
| Consolidation | Each batch binds the shared tool alongside its guarded dispatcher. | Archive allowlists, suppression, completion and current-member checks stay in their existing owners. |
| Healer | One response contains `submit_healings` and optional `journal` calls. | Existing batch-target/needed-field validators apply task results. No result is fed back for another model call. |
| Aspect | One response contains `submit_classifications` and optional `journal` calls. | Existing taxonomy/category/candidate validation and registry writes remain in place. No follow-up model call. |

For loop encoders, a response containing only `journal` executes and ends the
run. Mixed task/journal responses still receive ordinary task feedback. Journal
calls are excluded from task actions, graph writes, reads and failed-run partial
action counts. Terminal calls on the final response at the round limit execute
without another model request. Actual request counts come from per-round usage.

Single-shot task and journal calls are independent. Missing or multiple valid
task submissions fail visibly; journal failure cannot substitute its arguments
for task results. There is no final-text JSON extraction or Review fence parser.
The runtime logs failures; it never adds a retry round to repair journal output.

## State continuity and migration

A first real invocation folds all historical note pages before caps, then writes
one authoritative identity checkpoint. Legacy subject/inversion semantics exist
only in that cold adoption fold. Read-only inspection does not adopt or advance
counters. Modern notes, including failures and rejected live filings, always use
the current item snapshot shape.

Checkpoint, clock, latest snapshots, recent note-bearing chains and freshness
cursor come from one SQLite snapshot. Recent chains come from full trace history,
including superseded edits; deriving them from current items would incorrectly
keep old ordinary notes alive indefinitely under repeated edits.

Each invocation selects latest items from K note-bearing chains (S1=5, S2=3), plus
up to ten older persistent pins. Later batches refresh those selected IDs while
new private notes wait until the next invocation. Initial selection failure stays
unavailable for that invocation; refresh failure retains the last view.

Only versions actually shown authorize edits. Receipts are consumed by a journal
call; one batch of operations can edit several items. Omitted, stale, foreign or
reused references fail explicitly. No-op edits append nothing. A concurrent write
rejects stale edits while preserving independent additions. Private read/storage
failures do not prevent independent live operations, and failed feedback writes
cannot prevent later withdrawals. Explicit withdrawal alone affects live messages.

Removed runtime mechanisms: delimiter parsing, colon references, private closures,
headingless salvage, the old text renderer, alternate current views, fenced Review
parsing/writing/stripping and retired prompt simulation scripts. Arc keeps its own
fence scanner. Current eval callers use native bindings or raw run-chain history.
Frozen historical fixtures retain their original hashes and require pinned runtimes.

## Caching

`run_llm_once` now places five-minute breakpoints after tools/system and at the
end of the user message. The latter covers the entire request prefix. The stable
breakpoint permits reuse when batch content changes. Existing loops already cache
the full initial user prompt for five minutes and keep their one-hour stable
system/preamble caches.

Cache savings require matching prefixes within the TTL and the model's minimum
size. A five-minute cache write costs 1.25× ordinary input; a cache hit generally
costs 0.1×. No extra round is created to manufacture reuse. See the Anthropic
[prompt-caching contract](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

## Reviews and validation

Architecture review covered placement, unification, cohesion, coupling and altitude,
including independent review of the native migration. Existing owners are sufficient;
no new service/table or five copied journal implementations are warranted. Review
fixed a Healer edge-format import, final-response terminal-call loss, and journal
contamination of failed-run task accounting.

Shape requirement: one scoped journal tool across all five encoders, without extra
single-shot rounds. Operator-approved. Verdict: **KEEP** the resulting mechanism.
Probes: existing runner/binding reused; obsolete transport deleted; adoption stays
cold; one shared tool across siblings; validation stays at its owner; terminal
execution lives in the generic runner rather than copied encoder branches.
Coverage: native request/dispatch/results, all five callers, lifecycle tests,
retired callers and two targeted recalls. Recall reaffirmed verb-to-tool naming
and matching the prompt's finish condition to the runner. This does not establish
full encoding-quality improvement or production savings.

Final validation: **429 tests passed, 12 subtests passed in 12.05 seconds**.
`git diff --check` passed, and all four changed evaluator entrypoints parsed.

Deterministic coverage includes state identity/renaming, persistence clocks and
aging, cold adoption, stale/foreign receipts, read/write/routing failures, complete
JSON display, all five request paths, single-call tool execution, actual cache
markers, independent task results, terminal calls at the round limit and graph
action accounting. Contract/guardrail, S1, Community/Consolidation suppression and
reconciliation, Aspect, runner telemetry, client lifetime and dashboard tests run
alongside them. No guardrail allowlist was widened.

The current evaluator is `eval/journal_probe.py`. Default mode renders an isolated
production copy locally. `--llm` creates a fresh synthetic brain and sends only two
fictional same-subject notes, shared journal instructions and schemas to Anthropic.
That export was explicitly approved after automatic review requested confirmation.
`--s2` remains an optional isolated coordinator run through `brain.run_s2()`; it has
not been run for this change.

Ten native-tool model calls are recorded in `/private/tmp/journal-native-tools-probe/`:
all journal operations were valid, all targeted the intended item, no duplicate
notes were created, and all five unchanged second requests made no journal call.
Healer did rewrite an unchanged concern and retained persistence on the verified
entry. One Consolidation response omitted the synthetic task-completion tool;
Consolidation's production path is a loop with journal-only completion, so this
is a probe limitation, not evidence of a Healer/Aspect task loss.

Usage: 3,534 fresh input, 9,057 cache-creation, 4,285 cache-read and 731 output tokens.
These minimal protocol inputs differ from full encoder tasks. Cache hits occurred;
net savings and encoding quality are not established. The earlier fenced transport
probe had four missing headings and two duplicate notes; it is historical comparison
evidence, not a controlled quality or cost benchmark.

Before merge: review the final diff and decide whether these behavioral observations
need a prompt adjustment or production observation. After merge, verify loaded code,
real tool/result outcomes, journal continuity and actual cache/token costs. No current
redesign is deployed.
