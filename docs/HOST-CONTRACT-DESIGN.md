# Host Contract — Design

**Status — 2026-09-07:** design, shape approved by Tom, **0 of 5 steps built**. Merged to main
(`b56e63e`); docs-only, so no `servers/*.py` fingerprint change and no daemon restart. Nothing
under `servers/` or `hooks/` was touched. **Next move:** a from-above review before execution
— `/architecture-review` for code-claim verification and placement, noting it cannot simulate
post-step state (`96400de4`), so the step ordering in §10 is reasoned, not traced. Handoff and
verify-before-use list: brain node `a2594ee0`. One gate is still Tom's — see the end of §9.

**What this owns:** the boundary between a harness (Claude Code, Codex, later Grok or a local
model) and the host-neutral brain — how a harness is recognised, what vocabulary it speaks,
what we observe of it, and what we knowingly do not.

**What this does not own:** per-host research and gap analysis
([CODEX-ADAPTER-RESEARCH.md](CODEX-ADAPTER-RESEARCH.md)); the prompt-envelope row shape and
its two-phase plan (brain `cae5e153`, referenced from
[THALAMUS-ARCH-PLAN.md](THALAMUS-ARCH-PLAN.md)). **Explicitly deferred:** canonical argument
normalisation across tool shapes — Tom ruled 2026-09-07, "canonical arguments can be done
later." This doc must therefore promise classification and coverage, never argument rewriting.

Related: D-11 (`3c9c9012`) separated host-neutral service naming from adapter naming;
`137dc65c` extended it to adapter code. This applies the same line to **observation shape**.

> Reviewed by a parallel Codex session (01a07c8c): 15 findings across three parts, each marked
> MEASURED or INFERRED with its own side effects, all verified here against the code before
> acceptance. Six corrected errors in the first draft. Runtime samples marked
> `[codex-stream]` are its measurements, not mine.

---

## 1. The problem, traced

**The prompt path unites at `daemon_hooks.hook_recall:198`** (`ctx.set_env(model=…, host=…)`).
Both hosts run the same `pre_response_recall.py`; only the manifests differ.

**The tool path never unites.** `post_tool_trace._build_summary` branches per host tool name
at capture, writes the **raw** name into `metadata.tool`, and
`encoder_actions.parse_action:197-216` re-interprets that string at encode time — another
process, hours later.

**But "past the unite point everything is host-neutral" is false.** The first draft claimed
this from the absence of `'codex'`/`'claude-code'` literals under `servers/`. That grep looked
for host *names*, not host *shape*. Claude Code's `<task-notification>` envelope is recognised
and filtered at five sites in the host-neutral layer:

```
trace_contract.py:308   '<task-notification>' in (op_text or '')
trace_contract.py:315   WAKE_ENVELOPE_MARKER = "<task-notification>"
recall_laf.py:907       if is_machine_turn(r.get('content'))
daemon_hooks.py:324     drop wake envelopes
dal_logs.py:1354, 1455  filter on the marker
```

So host-envelope recognition is already downstream, and moving it to the boundary is part of
the work — not a bonus.

### The tool-name leak is a class, not a bug

| site | hardcoded | breaks when |
|---|---|---|
| `encoder_view.py:220` | `WRITE_ACTION_TOOLS = {'Edit','Write','NotebookEdit'}` | a host names its editor otherwise — **observed** (`apply_patch`, `63fde9b2`) |
| `encoder_actions.py:181,210,217,247,257` | `tool == 'Bash'` ×5 | a host names its shell `shell`/`run`/`terminal` |
| `daemon_hooks.py:746`, `dispatch_ops.py:75`, `brain_assembly.py:552,560` | `'Edit'` defaults | same |

The five `'Bash'` sites work on both current hosts **by coincidence** — they happen to agree
on that one name.

---

## 2. The contract

One declaration per harness, in the shape `PROMOTED_FIELDS` (`servers/contract.py`) and
`INTERACTION_DEFAULTS` + `INTERACTION_VALIDATORS` + `interaction_fingerprint()`
(`servers/interaction_defaults.py`) already establish: entries carry behaviour, consumers
derive, a validator holds the entry honest.

```
HOST_CONTRACT['codex'] = {
  'identity':   {'tells': [...], 'confidence': <rule>},        # §3
  'inputs':     {...},                                        # §4 — input classes
  'tools':      {'apply_patch': 'edit', 'Bash': 'shell', ...},
  'envelopes':  {'# Selected text:': 'keep',
                 '# Files mentioned by the user:': 'drop',
                 '## My request:': 'marker',
                 '<send_user_message_question_reply>': 'extract:question_reply'},
  'transcript': {'grammar': 'codex_rollout'},                  # see below
  'events':     {'declared': [...]},                           # D4
  'blind':      ['hosted_web'],                                # §5
  'record':     {'kind': 'rollout'},
  'verified':   {'host_version': '0.153.4', 'impl_identity': <hash>, 'vocab_version': 1},
}
```

**Kinds:** `edit` · `shell` · `read` · `search` · `agent` · `mcp`.
**Envelope policies:** `drop` · `keep` · `marker` · `extract:<name>` (a named pure extractor).

**No per-host `accessors` table.** The first draft had one. `turn_model()` is already one
host-agnostic function (payload-first, else transcript), so a per-host location map would
duplicate knowledge the function holds. What survives is `transcript.grammar`: a host-neutral
*interface* is not a host-neutral *implementation* — `turn_model`'s fallback reads only
Claude-style top-level `assistant`/`message.model`, and `post_response_track.py:17-47` scans
for top-level `type: human/user`, a predicate that matched **0 rows** against 7 nested
`item_completed UserMessage` rows on a real Codex rollout `[codex-stream]`. So "a new harness
is one entry, no code" is honestly scoped to **already-supported transcript grammars**.

**No `nesting` field.** Codex code mode wraps calls in `exec`, but our hooks already receive
leaf names — `post_tool_trace.py:72` consumes `data['tool_name']` directly, and traces show
`Bash` / `apply_patch` / `mcp__*` leaves. Code-mode hierarchy belongs to the **rollout
reader/reconciler** (§5), which must dedupe the `exec` wrapper against its leaf items.

### Consumers derive

| consumer | reads |
|---|---|
| normalizer (hook boundary) | `identity`, `inputs`, `tools`, `envelopes`, `transcript` |
| `encoder_view` / `encoder_actions` | `kind` — **plus retained raw name and MCP namespace+operation** |
| reconciliation | `record`, `blind`, `events` |
| a new harness | one entry + the conformance test |

`kind` alone is insufficient and the draft overpromised "kinds only": `kind == 'mcp'` cannot
distinguish `brain.recall` from another server's `recall`, so `is_brain_tool` and
`action_mode` still need namespace+operation or the retained raw name.

---

## 3. Identity is composite, recorded, and honest about doubt

Harness identity **cannot be made certain** at the hook boundary (`2f1ee97e`).
`CODEX_THREAD_ID` / `CODEX_SESSION_ID` exist in the session shell but are **absent from hook
processes** `[codex-stream]`.

| candidate | host-specific | guaranteed | defect |
|---|---|---|---|
| `CLAUDE_CODE_SESSION_ID` | yes | yes | Claude Code only — the one strong tell |
| `PLUGIN_DATA` name | **no — family** | yes | Codex sets CC's aliases deliberately, so any CC-compatible harness sets it (`483ba9d0`) |
| plugin-path root **value** (`~/.codex/`) | mostly | yes | a fork keeps `.codex`; a heuristic |
| `CODEX_INTERNAL_ORIGINATOR_OVERRIDE` | yes | **no** | internal, "override" semantics, Desktop only |
| payload `model` **key presence** | weak | yes | works because CC omits it; the **value** is never a host signal (custom providers) |

**Composite evidence; the row records which tells fired plus a status.** Low confidence is a
loud state that still functions — record, warn, decline to parse.

**Hazard the draft missed:** `SessionContext.set_env:230-259` ignores empty values, so a later
`''` retains the prior host — an unknown event would silently **inherit stale certainty**,
inverting this section's promise `[codex-stream]`. Fix with an explicit **per-event** status
and precedence, keeping last-known session state separate for display. Do not fix an
event-quality problem by erasing useful presence context.

**Four axes stay separate:** harness · app surface · model · vocabulary version. `session_meta`
reads `source='vscode'` on a Codex *Desktop* session `[codex-stream]`, so surface ≠ harness.
Desktop envelope rules therefore sit under an explicit surface/profile selection, not under
the harness name.

---

## 4. Input classes at the boundary

The envelope work assumed every input is operator text, possibly wrapped. It isn't:

1. **operator text** — plain.
2. **operator text inside a host envelope** — CC `<system-reminder>`, Codex `# …:` headers.
3. **structured envelope where one field is the operator's** —
   `<send_user_message_question_reply>` JSON, where only `answer` is theirs and `question` is
   **assistant** text (`44ac17e0`). Needs `extract:`, not a block verdict.
4. **host-injected function-call output** — `TurnStartParams.toolOutput {name, namespace?,
   output}`, accepted with `input: []`, arriving as `functionCallOutput` rather than
   `userMessage` `[codex-stream]`. Not operator speech at all and must never be attributed to
   the operator.

A message identity must **not** be keyed on `turn_id` alone: UserPromptSubmit can fire several
times within one turn/Stop chain (chain `s1r-01a07465-19` carries two `user_message` rows).

---

## 5. Coverage, and the two absences

- **Unknown input encountered** — arrived, unclassifiable. Detectable *at* the normalizer →
  record, warn, do not guess.
- **Event never observed** — never arrived. **Invisible at the normalizer by definition.** Only
  findable by reconciling against the host's own record.

Measured instance of the second: 4 hosted-web Extension operations in a 25s rollout window,
**zero** in S0 `[codex-stream]`.

**Invariant:** all **captured** entries pass the one normalizer, with explicitly measured
coverage gaps.

**Reconciliation needs a join key that does not exist yet.** `post_tool_trace.py:84-95` writes
session_id, stop chain, capped summary and `metadata.tool` — it **drops `tool_use_id`,
`turn_id`, `tool_response`**; the prompt hook forwards no host turn or message id. So source
IDs, event type, event-time/sequence and completion status must be added in **step 1**, or
step 4 is unbuildable. Old rows can then only reconcile heuristically, and a missing or
rotated host log must never be reported as full coverage.

`blind` is therefore not "assert these stay blind forever" but **detect changed blind status**,
including a family that becomes newly *captured*. Detection reports; it does not backfill.

**Coverage baseline.** Captured: `Bash`, `apply_patch`, brain MCP, app MCP
(`mcp__codex_app__read_thread` → `fe66a91d`). Missing: hosted web Extension calls.
**Untested, not failed:** ordinary free-text mid-tool-call, interrupts, queues — two probes
both routed through the async-question card. Claude Code separately drops mid-tool-call
messages entirely (`4b8ed058`).

---

## 6. Legacy rows — three states, no backfill

`parse_action` reads `metadata.tool` only for drop/stub; behaviour comes from the **summary
head**, and `_Action` carries no kind. So stamping `kind` changes nothing until `parse_action`
reads it — and after a cutover, every historical row and every row from an un-redeployed
client has no kind. A day of shadow traffic does not normalise old rows.

Read with **three** states, not two:

| state | row | behaviour |
|---|---|---|
| i | no normalisation stamp | legacy: today's summary-head path |
| ii | stamp + valid kind | normalised |
| iii | stamp present, kind unknown / failed / malformed / unsupported schema | **visible error policy — never a silent fall back to the summary head** |

Two states would disguise a new mapping failure as an ordinary legacy row. Keep `status`
distinct from the literal word "unknown".

**Accepted limitation:** historical `apply_patch` rows keep their lack of protection. No
backfill, and nothing fabricates an original vocabulary version a row never had. Mixed
old/new/unknown fixtures pin this boundary.

---

## 7. What moves

| from | today | to |
|---|---|---|
| `hook_common.host_name():269` | two env tells, `''` on miss | `identity.tells` + per-event status |
| `hook_common.turn_model():289` | payload else CC-only transcript scan | stays; `transcript.grammar` declares what it supports |
| `post_response_track.py:17-47` | CC-only top-level `type: human/user` scan | same treatment; a second transcript accessor |
| `post_tool_trace._build_summary():18-42` | nine per-tool-name branches | `tools` classification **beside** the existing summary — summary itself unchanged (deferred) |
| `post_tool_trace.py:84-95` | drops `tool_use_id`, `turn_id`, `tool_response` | add source IDs (step 1, additive) |
| `hook_common.tool_target_file():349` | regexes ONE filename out of `tool_input['command']`, discards the patch | preserve the patch text at capture |
| `encoder_view.py:220` | `WRITE_ACTION_TOOLS` name set | `kind == 'edit'` |
| `encoder_actions.py:181,210,217,247,257` | `tool == 'Bash'` ×5 | `kind == 'shell'` — **with `kind` as a new `_Action` field**, see §8 D7 |
| `trace_contract.py:308,315`; `recall_laf.py:907`; `daemon_hooks.py:324`; `dal_logs.py:1354,1455` | `<task-notification>` recognised downstream | boundary classification; consumers read stamped origin/role |

The patch body needs no rollout read: `tool_input['command']` already carries the patch text
at capture — `apply_patch` has no `file_path`, so the paths in existing `apply_patch: /…`
summaries can only have come from that regex. `FileChange.changes` is a second source, not the
only repair.

## 8. What stays (do not "fix" these)

- **`dispatch_common.is_brain_tool` / `_BRAIN_TOOL_RE`** — host-agnostic by pattern. The model
  the rest of this copies.
- **`DROPPED_ACTION_TOOLS` / `STUBBED_ACTION_TOOLS`** — brain tool names, reached only after
  `is_brain_tool()`. Host-neutral already.
- **`brain_mcp._stamp_caller_session:380-420`** — a **trust boundary**, not a host accessor: it
  HMAC-verifies every untrusted MCP call, strips `_caller_sig` in all branches, prefers the
  trusted CC env. Receiver-side verification stays at the receiver; a hook's normalisation
  result cannot replace it. Consequently "preserve raw" means **redacted** raw — never the
  signed caller pair (`hook_common:440-450` already strips `_caller_*` before recording).
- **`_bash_verb` / `GIT_WRITE_VERBS` ownership** — shell parsing belongs with `kind == 'shell'`.
  **But not a clean bill of health:** `_bash_verb` returns the *program* while
  `GIT_WRITE_VERBS` holds git *subcommands*, so `git push origin main` → `sub='git'` →
  **unprotected**, while bare `rm`/`mv`/`tag` are protected by name collision. Both directions
  wrong; `test_git_commit_harvests_subject` passes via the harvested `' · '`, masking it.
  **Pre-existing defect, its own fix, not resolved by a kind cutover.**
- **`hooks/*.json`** — manifests stay the events source of truth; the contract entry is
  declared and held to them by test (D4).

---

## 9. Decisions

| # | question | ruling / recommendation |
|---|---|---|
| D1 | where does it live? | **split:** `trace_contract.py` keeps the normalised **output** schema, validation and stamp subsets; new `servers/host_contract.py` holds **input** dialects, resolver references, evidence and coverage — and *references* the output schema rather than redefining it |
| D2 | how do hooks read it? | lazy import via `hook_common`, as `encoder_view.action_mode` and `is_brain_tool` already do — **but `HOST_CONTRACT` must be a leaf module with no path to `daemon_config`.** `post_tool_trace.py:97` is deliberately hand-rolled because importing `servers.daemon_client` measures ~44ms, ~22ms of it `daemon_config` md5-walking every `servers/*.py` for a code fingerprint at import time, on a path that fires ~2500×/day. A *lazy* import still pays that on first call, so the constraint was never "hooks can't import `servers/`" — it is "this hook can't afford to." Either the contract imports nothing heavy, or the hottest hook gets a cheaper read |
| D3 | where do `extract:<name>` extractors live? | beside the contract, one pure function per name in a registry dict. I/O-bearing transcript resolution stays **out** of that registry |
| D4 | `events` derived or declared? | **declared, test-verified.** Nothing parses the manifests at runtime. Assertions: `registered_contract(host) == registered_manifest(host)` **and** `registered_manifest(host) ⊆ supported_engine_events(host)`. Equality to all supported events is wrong — Codex registers 6 events / 10 handlers of 12 supported; CC registers 10 / 13. Keep the existing handler/matcher/timeout checks with explicit Codex-only exceptions |
| D5 | is `blind` enforced? | detect **changed** blind status in either direction; report only, no backfill |
| D6 | version policy | stamp vocabulary version **and** an implementation content-identity — hashing registry names would not notice a changed resolver or extractor body. Keep runtime-observed host version separate from "tested against 0.153.4" |
| D7 | the five `'Bash'` sites | fix in step 2, but **`kind` must be a new field on `_Action`, separate from raw/display name.** `_rollup_line` keeps ONE global `subs` counter rendered per display tool, so a bare kind swap would let two shell tools show each other's subcounts |
| D8 | canonical argument normalisation | **deferred** (Tom, 2026-09-07). Not in this work |

**Open — Tom's:** env_message phase 2 on the flip-day checklist, or its own
`s1_encode_eval`-gated step? Recommendation: its own step, since phase 2 reclassifies rows the
encoder currently reads (`ca3446a3`) and one eval failure would otherwise stall the checklist.

## 10. Step order

| # | step | risk |
|---|---|---|
| 0 | `HOST_CONTRACT` + CC and Codex entries + conformance test | no behaviour change |
| 1 | normalizer consumes it; add **shadow fields only** — kind, tell, status, versions, source IDs; preserve patch text at capture | additive, see below |
| 2 | flip consumers: `WRITE_ACTION_TOOLS`, the five `'Bash'` sites, the `'Edit'` defaults; three-state legacy read | first behaviour change |
| 3 | envelope handling: phase 1 + Codex entries + `extract:`; retire the downstream `<task-notification>` filters **after** §6's legacy path exists | removing them early re-admits machine chatter to recall/presence |
| 4 | reconciliation against the host's own record | needs step 1's source IDs |

**"Additive" has a hard boundary.** Consumers parse rendered summary text; edit summaries
truncate to a path and Bash commands to 200 chars, so a caption change is already an encoder
input change. Step 1 must leave existing summary, content, `metadata.tool` **and**
classification untouched. Envelope extraction, changed host decisions and any fail-closed
behaviour wait for an explicit cutover. Cost: two representations coexist briefly — which is
what makes a real before/after comparison possible instead of a silent change wearing an
additive label.

---

## Evidence

- `apply_patch` unprotected: `condense_actions` on both shapes, 39-action turn, 3 distinct
  paths → Codex 0/3 edit lines survive, CC 3/3. Node `63fde9b2`.
- `git push` unprotected / bare `rm` protected: probed via `_bash_verb` + `GIT_WRITE_VERBS`.
- Manifests parsed here: Codex 6 events / 10 handlers; CC 10 / 13.
- Hook-process env inventory (`pre-bash-safety.sh`, PID 59112) `[codex-stream]`. Node `2f1ee97e`.
- Codex envelope literals from `ChatGPT.app/Contents/Resources/app.asar` — nine headers plus
  the `## My request(?: for Codex)?:` marker; absent from the `codex` Rust binary, so this is
  desktop-app behaviour. Node `4c42b9da`.
- Question-reply envelope `[codex-stream]`. Node `44ac17e0`.
- `functionCallOutput` via `TurnStartParams.toolOutput`, 0.153.4 schema `[codex-stream]`.
- Hosted-web blind spot: 4 Extension ops in 25s, zero in S0 `[codex-stream]`.
- Approved architecture `1f2b3f89`; contract framing `ccc17d81`.

**Caveat carried from the review:** protocol presence is not end-to-end availability. The
0.153.4 schema documents capabilities (an app-server-control socket, an event-to-task bridge)
that are **absent or unverified** on the installed desktop build `[codex-stream]`. A contract
entry may only claim what has been observed on the build in use.
