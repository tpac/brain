# Action capture and encoder limits

The shared action policy applies to every host. The defaults remove Git calls
at capture and keep the encoder's action timeline small. User and assistant
messages, node content and provenance are separate from this action budget.

Add any overrides to `~/.config/brain/env`, then restart the daemon through the
normal launcher so it reloads that file:

```sh
BRAIN_ACTIONS_PROFILE=thin
BRAIN_ACTIONS_EXCLUDE_GIT=1
BRAIN_ACTIONS_MAX_LINES=40
BRAIN_ACTIONS_MAX_BYTES=6000
```

| Setting | Choices / bounds | Meaning |
|---|---|---|
| `PROFILE` | `thin`, `balanced`, `full` | Amount of per-turn detail before the total limit |
| `EXCLUDE_GIT` | `1` / `true`, `0` / `false` | Exclude a whole shell call containing a recognized Git invocation |
| `MAX_LINES` | 4–200 | Total action content lines, including the omission notice |
| `MAX_BYTES` | 512–24000 | Total serialized UTF-8 action bytes, including XML escaping, wrappers, indentation and the notice |

Names in the table have the `BRAIN_ACTIONS_` prefix. Invalid values warn and use
the defaults above; zero never means unlimited. XML wrapper lines count toward
bytes but not content lines. Every profile and the view-policy-off control obey
both total limits, including already-encoded turn stubs.

## What thin cuts and keeps

- **Git calls are excluded completely**, including a mixed call such as
  `git status && rg something .`. Future matching calls create no `tool_result`
  trace. Existing stored history stays intact; the same exclusion is applied
  when preparing its encoder view. Setting exclusion off resumes capture; it
  cannot restore calls excluded earlier.
- **Routine inspections and brain inbox/presence polls become counted groups**
  with tool and target cues. Import-first Python/shell snippets become opaque
  script counts; an import does not establish the script's effect or make it
  read-only. Other non-Git traces remain available for inspection.
- **Actions group within each conversation turn**, without preserving their
  internal order. Edit calls sharing a full target cue and raw tool become one
  counted row, even across intervening reads or tests. `(N edit calls)` counts
  calls sharing a caption, not identical patches or complete per-file touches;
  a multi-file patch's caption can name only its first file. `×N` on other cues
  means identical recorded summaries and classification identity. Omitted
  routine activity shares one rollup per turn.
- **The closing actions are reserved before deduplication**. Meaningful cues
  render separately as `Closing: …`, even if they repeat an earlier action.
  Closing boilerplate joins the routine rollup. Conversation turns stay ordered;
  action list order inside a turn cannot establish which test followed which edit.
- **Edits, diagnostic warnings and meaningful closing cues get priority** at
  allocation. Useful tests, deploy commands and stated script intent remain
  eligible as individual cues. A priority flood still stops at the total limit.
- **Overflow gets one counted `<action_limit>` notice** across the window,
  including the number of priority actions omitted. Selected records render
  in conversation-turn order; within a priority class, later turns and later
  prepared rows get space first.
  Missing detail does not mean no activity occurred.

`balanced` keeps the established 15/30 per-turn head budgets and exact-summary
deduplication. `thin` uses 5/10 head budgets plus grouping across each turn. These are
presentation targets, not total ceilings. `full` skips grouping/condensation;
existing provenance-based tool filtering still applies. The final whole-window
limit is always enforced. This bounds action input, not total encoder tokens,
model rounds, catalog size or output cost.

## Ownership and call paths

```text
Capture: hook sends raw facts + wire-only command
           → daemon normalizes tool kind
           → shared Git rule → trace append (or excluded result)

Encode:  traces → existing turn assembly
           → prepare all action blocks (filter + group/condense)
           → allocate once across the window → serialize → Sonnet prompt
```

`servers/action_policy.py` owns the immutable settings snapshot and literal Git
recognizer. `dispatch_observability.py` applies capture exclusion after kind
normalization; the hook transports the command without classifying it or
storing its full body. `encoder_actions.py` owns preparation, priority, counts
and serialization. `encoder_view.py` owns internal S1 presentation tuning and
provenance filtering. `encode.py` only coordinates the whole-window call.

Recognition reads shell syntax without executing it: command lists, pipelines,
literal executables, common wrappers, shell `-c`, substitutions and heredocs.
It does not resolve aliases, shell functions, computed executable names or
commands inside Python/other scripts. Encountering a function declaration
retains the call rather than treating its deferred body as execution.
Historical summaries may already be cropped and cannot
prove a later Git invocation. Unknown/malformed kinds are retained with a
diagnostic. If the full command exceeds the hook transport allowance, capture
also retains the row with a visible diagnostic and an error log rather than
silently treating its short summary as the whole command.
