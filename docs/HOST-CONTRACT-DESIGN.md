# Host Contract — Design

## Status — step 0 shipped and reviewed, step 1 next (2026-09-08) ◀ ACTIVE ARC

**Read first:** handoff node `66ba8b6c` (build step 1), decision `e8183445` (D9 + the review's
findings). Main `25e8165` = step 0 (`253375a`) + its lean-review fixes (`9ac4f58`); docs-only
head commit after.

**Central finding.** Reviewed from above (`/architecture-review` in design-doc mode, two
adversarial passes, every code claim re-verified); **placement changed to D9 by Tom** ("b. make
sure things can't drift"): classify at the first consumer's boundary — tool kinds daemon-side at
the S0 write door, envelopes hook-side. Step 0 built: `servers/host_contract.py`, the output
vocabularies and `tool_result` shape in `trace_contract`, boot validation, and the drift fences
(36 quoted host-shape sites in 12 files ratcheted both ways; manifests ↔ contract; hook mirrors;
leaf pin). One lean post-merge review pass: no blockers, five should-fix, all applied. No hook
changed; no behaviour changed.

**Locked:** D1–D11 (§9). **Open — Tom's:** env_message phase 2 placement (gates step 3 only).
**Do not reopen:** hook-side tool classification (D9); read-time backfill of legacy rows (D10);
canonical argument normalisation (D8, Tom); the `_bash_verb`/`GIT_WRITE_VERBS` defect
(`b09efd2a`, its own fix).

**Next builds:** step 1 (hooks send raw facts — tells, ids, `payload_keys`, capped patch;
write door builds via the builder THEN stamps; unknown kind → errors table; keys join the
required shape), then step 2 (D7 split, consumers flip to kind, before/after `s1_encode_eval`).

This doc cites **symbols, not line numbers** — the first draft's line refs drifted within a
day. The tests are the living truth; the doc says why.

**What this owns:** the boundary between a harness (Claude Code, Codex, later Grok or a local
model) and the host-neutral brain — how a harness is recognised, what vocabulary it speaks,
what we observe of it, what we knowingly do not, and **which test holds each of those facts
in place**.

**What this does not own:** per-host research and gap analysis
([CODEX-ADAPTER-RESEARCH.md](CODEX-ADAPTER-RESEARCH.md)); the prompt-envelope row shape and
its two-phase plan (brain `cae5e153`, referenced from
[THALAMUS-ARCH-PLAN.md](THALAMUS-ARCH-PLAN.md)). **Explicitly deferred:** canonical argument
normalisation across tool shapes — Tom ruled 2026-09-07, "canonical arguments can be done
later." This doc promises classification and coverage, never argument rewriting.

Related: D-11 (`3c9c9012`) separated host-neutral service naming from adapter naming;
`137dc65c` extended it to adapter code. This applies the same line to **observation shape**.

> Reviewed by a parallel Codex session (01a07c8c, 15 findings, six corrections accepted) and
> by this session's two adversarial passes. Runtime samples marked `[codex-stream]` are the
> Codex session's measurements; `[review]` marks this session's agents' measurements against
> live traces. Code-side claims were verified here by reading the code.

---

## 1. The problem, traced

**The prompt path unites in the daemon** at `daemon_hooks.hook_recall` → `ctx.set_env(model=…,
host=…)`. Both hosts run the same `pre_response_recall.py`; only the manifests differ. The hook
observes the host's tells; the daemon stamps.

**The tool path never unites.** `post_tool_trace._build_summary` branches per host tool name
at capture, writes the **raw** name into `metadata.tool`, and
`encoder_actions.parse_action` re-interprets that string at encode time — another process,
hours later. That asymmetry is the whole `apply_patch` defect (`63fde9b2`).

**But "past the unite point everything is host-neutral" is false.** Claude Code's
`<task-notification>` envelope is recognised downstream. Counted by the guardrail scan
(quoted literals only, `tests/test_host_shape_guardrail.py`): **three separate hardcodings**
of the literal — `trace_contract.WAKE_ENVELOPE_MARKER` (the owner), `pre_response_recall`
(the hook's wake→register_only routing), and `dashboard/queries/stats.py` (the dashboard may
not import `servers/`; the ratchet also counts its quoted mention in a comment — retirement
bookkeeping, not a decision) — plus readers of the constant in `daemon_hooks` (the surface
window), `recall_laf` (via `is_machine_turn`) and **three** `dal_logs` methods
(`active_sessions_by_turn`, the rich-presence read, `conversational_turns_since`). Moving
recognition to the boundary is part of the work, not a bonus. (`is_machine_turn` hardcoded the
literal seven lines above the constant that defines it; step 0 made it read the constant.)

### The tool-name leak is a class, not a bug

| site | hardcoded | breaks when |
|---|---|---|
| `encoder_view.WRITE_ACTION_TOOLS` | `{'Edit','Write','NotebookEdit'}` | a host names its editor otherwise — **observed** (`apply_patch`, `63fde9b2`) |
| `encoder_actions` — `_label`, `parse_action` ×2, `_rollup_line` ×2 | `tool == 'Bash'` ×5 | a host names its shell `shell`/`run`/`terminal` |
| `daemon_hooks.hook_pre_edit`, `dispatch_ops._handle_pre_edit`, `brain_assembly.pre_edit`, `hooks/scripts/pre_edit_suggest.py` | `'Edit'` defaults ×4 | same |

The five `'Bash'` sites work on both current hosts **by coincidence** — they agree on that one
name. So do the hook event names (`'Stop'` in `daemon_hooks.post_response_common`): Codex
copied Claude Code's vocabulary. Coincidence is not a contract; D4 makes the events one.

---

## 2. The contract

One declaration per harness, in the shape `PROMOTED_FIELDS` (`servers/contract.py`) and
`INTERACTION_DEFAULTS` + `INTERACTION_VALIDATORS` + `interaction_fingerprint()`
(`servers/interaction_defaults.py`) already establish: entries carry behaviour, consumers
derive, a validator holds the entry honest, a fingerprint names the code that ran. Built in
step 0 as `servers/host_contract.py`:

```
HOST_CONTRACT['codex'] = {
  'identity':        {'tells': ({'env': 'PLUGIN_DATA', 'strength': 'family'},
                                {'env': 'CODEX_INTERNAL_ORIGINATOR_OVERRIDE', 'strength': 'strong'})},
  'tools':           {'Bash': 'shell', 'apply_patch': 'edit', 'spawn_agent': 'agent'},
  'tool_patterns':   (('^mcp__', 'mcp'),),
  'matcher_aliases': {'Edit': 'apply_patch', 'Write': 'apply_patch', 'Agent': 'spawn_agent'},
  'envelopes':       {},                                    # step 3 — see §4
  'transcript':      {'grammar': 'codex_rollout', 'readable': False},
  'events':          (...6 registered...),                  # D4
  'engine_events':   frozenset(...12 documented...),
  'blind':           ('hosted_web',),                       # §5
  'record':          {'kind': 'rollout'},
  'manifest':        'hooks/hooks.codex.json',
  'verified':        {'host_version': '0.153.4'},
}
```

**The vocabularies live on the OUTPUT side**, in `servers/trace_contract.py`, and the contract
imports them to validate against: `ACTION_KINDS = ('edit','shell','read','search','agent','mcp')`,
`KIND_STATUS = ('ok','unknown')`, `HOST_STATUS = ('strong','family','ambiguous','unknown','legacy')`,
`ENVELOPE_POLICIES = ('drop','keep','marker')` + `ENVELOPE_EXTRACT_PREFIX`. The first draft
wrote them as prose inside the input file; a consumer importing the input dialects to read a
kind would re-open exactly the coupling §1 diagnoses. The encoder imports `trace_contract`,
never `host_contract`.

**What the two precedents have that the sketch lacked, now present:** `validate_host_contract()`
(every violation as `host: what`; the sync test asserts empty, `Brain.__init__` logs each to the
errors table — loud, non-fatal, no auto-heal, the `aspects` posture), `contract_fingerprint()`
(12-hex over the module's bytes, so a changed resolver body is a changed identity — D6),
`VOCAB_VERSION`, a numbered add-a-host recipe in the module docstring, and the read doors
`resolve_host` / `classify_tool` / `envelope_policy` that return an explicit status and never a
guess or a `KeyError`.

**No per-host `accessors` table** (`turn_model()` is already one host-agnostic function;
`transcript.grammar` + `readable` declares what our readers support — `post_response_track`'s
top-level `type: human/user` scan matched **0 rows** on a real Codex rollout `[codex-stream]`).
**No `nesting` field** (hooks already receive leaf names — `post_tool_trace` consumes
`data['tool_name']`; code-mode `exec` hierarchy belongs to the rollout reconciler, §5).

### Consumers derive

| consumer | reads |
|---|---|
| S0 write door (daemon) | `identity` (resolve), `tools` + `tool_patterns` (classify), `VOCAB_VERSION`, `contract_fingerprint()` |
| prompt hook | `envelopes` (step 3), the tell probe list (`all_tell_env_vars`) |
| `encoder_view` / `encoder_actions` | the stamped `kind` — **plus retained raw name and MCP namespace+operation** (`kind == 'mcp'` cannot separate `brain.recall` from another server's `recall`) |
| reconciliation (step 4) | `record`, `blind`, `events`, the row's source IDs |
| a new harness | one entry + the recipe; the tests fail loudly on a partial one |

---

## 3. Identity is composite, recorded, and honest about doubt

Harness identity **cannot be made certain** at the hook boundary (`2f1ee97e`).
`CODEX_THREAD_ID` / `CODEX_SESSION_ID` exist in the session shell but are **absent from hook
processes** `[codex-stream]`.

| candidate | host-specific | guaranteed | defect |
|---|---|---|---|
| `CLAUDE_CODE_SESSION_ID` | yes | yes | Claude Code only — the one strong tell |
| `PLUGIN_DATA` | **no — family** | yes | Codex sets CC's aliases deliberately, so any CC-compatible harness sets it (`483ba9d0`) |
| plugin-path root **value** (`~/.codex/`) | mostly | yes | a fork keeps `.codex`; a heuristic, not declared |
| `CODEX_INTERNAL_ORIGINATOR_OVERRIDE` | yes | **no** | internal, "override" semantics, Desktop only |
| payload `model` **key presence** | weak | yes | works because CC omits it; the **value** is never a host signal |

**The hook OBSERVES, the daemon RESOLVES.** The env is visible only in the hook process, so the
hook reports which declared tells are present (names only — `host_contract.all_tell_env_vars`
is the probe list; `hook_common.host_name` is held to it by `TestHookMirrors`). The rule lives
in `resolve_host`: one host's tells → `strong` or `family`; tells of two hosts →
**`ambiguous`, host `''`, never a pick** (5c1a0846's both-present case, now a state, not a
silent first-branch win); none → `unknown`. A client that sends a pre-resolved `host` string
and no tells stamps `legacy`.

**Hazard the first draft missed:** `SessionContext.set_env` ignores empty values, so a later
`''` retains the prior host — an unknown event would silently **inherit stale certainty**
`[codex-stream]`. `[review]` measured the gap: 2.8% of recent `tool_result` rows carry no host,
all in one Codex session that ran `apply_patch` before any prompt row — exactly the population
this contract classifies. Fix: per-event tells on every tool event (two env reads, zero
imports), `stamp_s0_session`'s `setdefault` already lets the per-event value win; and the
write door **never** calls `set_env(host=…)` from a tool event — the session's displayed host
stays the prompt path's.

**Four axes stay separate:** harness · app surface · model · vocabulary version. `session_meta`
reads `source='vscode'` on a Codex *Desktop* session `[codex-stream]`, so surface ≠ harness.

---

## 4. Input classes at the boundary

The envelope work assumed every input is operator text, possibly wrapped. It isn't:

1. **operator text** — plain.
2. **operator text inside a host envelope** — CC `<system-reminder>`, Codex `# …:` headers.
3. **structured envelope where one field is the operator's** —
   `<send_user_message_question_reply>` JSON, where only `answer` is theirs and `question` is
   **assistant** text (`44ac17e0`). Needs `extract:`, not a block verdict — the extractor is a
   registered pure function in `host_contract.EXTRACTORS`; the registry is **closed**, an
   unregistered name is a validator violation at contract time, not a first-prompt `KeyError`.

A message identity must **not** be keyed on `turn_id` alone: UserPromptSubmit can fire several
times within one turn/Stop chain (chain `s1r-01a07465-19` carries two recall runs — verified
here against the trace store).

`TurnStartParams.toolOutput` (`functionCallOutput`, not `userMessage` `[codex-stream]`) is
**not** an input class at the boundary: nothing shows a hook fires for it. It is host-injected
tool output that must never be attributed to the operator, and until a hook is observed
carrying it, it belongs to §5's reconciliation, not to the normalizer.

---

## 5. Coverage, and the two absences

- **Unknown input encountered** — arrived, unclassifiable. Detectable *at* the normalizer →
  stamp `unknown`, log to the errors table, do not guess.
- **Event never observed** — never arrived. **Invisible at the normalizer by definition.** Only
  findable by reconciling against the host's own record.

Measured instance of the second: 4 hosted-web Extension operations in a 25s rollout window,
**zero** in S0 `[codex-stream]`.

**Invariant:** all **captured** entries pass the one normalizer for their path, with
explicitly measured coverage gaps.

**Reconciliation needs a join key that does not exist yet.** `post_tool_trace` writes
session_id, stop chain, capped summary and `metadata.tool` — it **drops the IDs the payload
carries**: Codex documents `tool_use_id` + `turn_id` on Pre/PostToolUse
([CODEX-ADAPTER-RESEARCH.md](CODEX-ADAPTER-RESEARCH.md) §stdin); Claude Code's hooks reference
documents `tool_use_id` plus a `prompt_id` on every event. Step 1 adds them, **and adds
`payload_keys`** — the top-level stdin key names the hook saw, no values — so the contract's
claims about a host's payload are checked against every row instead of assumed (this retires
the "temporary diagnostic" ruling `1f2b3f89` carried). Old rows reconcile heuristically; a
missing or rotated host log must never be reported as full coverage.

`blind` is not "assert these stay blind forever" but **detect changed blind status**, including
a family that becomes newly *captured*. Detection reports; it does not backfill.

**Coverage baseline.** Captured: `Bash`, `apply_patch`, brain MCP, app MCP
(`mcp__codex_app__read_thread` → `fe66a91d`). Missing: hosted web Extension calls.
**Untested, not failed:** ordinary free-text mid-tool-call, interrupts, queues — two probes
both routed through the async-question card. Claude Code separately drops mid-tool-call
messages entirely (`4b8ed058`) — declared as its `blind` family.

---

## 6. Legacy rows — three states, no backfill

`parse_action` reads `metadata.tool` only for drop/stub; behaviour comes from the **summary
head**, and `_Action` carries no kind. So stamping `kind` changes nothing until `parse_action`
reads it — and every row written before the write door stamps has no kind.

Read with **three** states, not two:

| state | row | behaviour |
|---|---|---|
| i | no normalisation stamp | legacy: today's summary-head path |
| ii | stamp + `kind_status == 'ok'` | normalised |
| iii | stamp present, `kind_status == 'unknown'` / malformed | **visible error policy — never a silent fall back to the summary head** |

Two states would disguise a new mapping failure as an ordinary legacy row.

**No read-time backfill (D10).** Under daemon-side classification the raw name is preserved
and the row's host stamp exists on rows since 2026-09-06, so legacy rows *could* be classified
on read with today's vocabulary. Declined: it would retroactively protect historical
`apply_patch` rows under a vocabulary they never carried and make the encode eval a moving
target. Historical rows keep their gaps. Mixed old/new/unknown fixtures pin this boundary.

**Under D9 the un-redeployed-client state disappears** for the tool path: the daemon stamps
every row from every client from its first restart. State (i) is time-bounded to rows written
before that restart.

---

## 7. What moves

| from | today | to |
|---|---|---|
| `hook_common.host_name()` | two env tells, `''` on miss | reports which of `all_tell_env_vars()` are present; `resolve_host` in the daemon decides |
| `hook_common.turn_model()` | payload else CC-only transcript scan | stays; `transcript.grammar`/`readable` declares what it supports |
| `post_response_track` transcript scan | CC-only top-level `type: human/user` | same treatment; a second transcript accessor |
| `post_tool_trace` | 9 per-tool-name summary branches; drops IDs | **sends raw facts only**: + tells present, `tool_use_id`, `turn_id`/`prompt_id`, `payload_keys`, capped `tool_input['command']` for the patch body. Summary unchanged (deferred); no imports added |
| `dispatch_observability._handle_trace_append` | JSON-decode + `stamp_s0_session` | + `brain_traces` stamps `kind`/`kind_status`/`host_status`/`tells`/`vocab_version`/`impl_identity` via `build_tool_result_metadata` — the write door already stamps model/host here "because the hook itself stays a bare socket send" |
| `hook_common.tool_target_file()` | regexes ONE filename out of the patch, discards it | patch text preserved at capture (hook sends it) |
| `encoder_view.WRITE_ACTION_TOOLS` | name set | `kind == 'edit'` |
| `encoder_actions` `tool == 'Bash'` ×5 | raw display name | `kind == 'shell'` — with `kind` a new `_Action` field (D7) |
| `'Edit'` defaults ×4 | literal | kind from the contract, or no default |
| `WAKE_ENVELOPE_MARKER` readers + 3 literal copies | `<task-notification>` recognised downstream | boundary classification (step 3); consumers read stamped origin/role |

The patch body needs no rollout read: `tool_input['command']` carries it at capture —
`apply_patch` has no `file_path`, so the paths in existing `apply_patch: /…` summaries can only
have come from `tool_target_file`'s regex.

## 8. What stays (do not "fix" these)

- **`dispatch_common.is_brain_tool` / `_BRAIN_TOOL_RE`** — host-agnostic by pattern. The model
  the rest of this copies (`tool_patterns` is the same move).
- **`DROPPED_ACTION_TOOLS` / `STUBBED_ACTION_TOOLS`** — brain tool names, reached only after
  `is_brain_tool()`. Host-neutral already.
- **`brain_mcp._stamp_caller_session`** — a **trust boundary**, not a host accessor: it
  HMAC-verifies every untrusted MCP call, strips `_caller_sig` in all branches, prefers the
  trusted CC env. Receiver-side verification stays at the receiver. "Preserve raw" means
  **redacted** raw — never the signed caller pair (`hook_common.strip_caller_stamp`).
- **`post_tool_trace._build_summary`** — stays in the hook until canonical arguments: it needs
  `tool_input`'s per-tool fields. Its nine branch names are held to the contract by
  `TestHookMirrors.test_build_summary_branches_are_declared_tools`, so the two tables cannot
  disagree in silence. Its inline `"ref_type": "tool_result"` literal also stays — replacing
  it with a constant makes the file extractor-blind and fails
  `test_trace_contract_sync.test_extractor_sees_every_writer_file`.
- **`_bash_verb` / `GIT_WRITE_VERBS`** — shell parsing belongs with `kind == 'shell'`. **Not a
  clean bill of health:** `_bash_verb` returns the *program*, `GIT_WRITE_VERBS` holds git
  *subcommands*, so `git push origin main` → **unprotected** while bare `rm`/`mv`/`tag` are
  protected by collision. Pre-existing defect, its own fix (`b09efd2a`), not resolved here.
- **`hooks/*.json`** — manifests stay the events source of truth; the contract entry is
  declared and held to them by test (D4).

---

## 9. Decisions

| # | question | ruling |
|---|---|---|
| D1 | where does it live? | **split, vocabulary on the output side:** `trace_contract.py` owns `ACTION_KINDS`, `KIND_STATUS`, `HOST_STATUS`, `ENVELOPE_POLICIES`, `TOOL_RESULT_METADATA_SHAPE` + `build_tool_result_metadata`; `host_contract.py` owns per-host dialects and imports the vocabularies to validate against. The encoder imports `trace_contract`, never `host_contract` |
| D2 | how do hooks read it? | `host_contract` is a **leaf beside `brain_constants`** (imports only `trace_contract`; cold import ~13 ms including `re`, `trace_contract` alone 1–7 ms; no path to `daemon_config`, ~28 ms with its import-time fingerprint), pinned by `TestLeaf`. The prompt hook may import it (that path already pays `daemon_client`). **The PostToolUse hook never does** — under D9 it needs nothing from the contract |
| D3 | where do `extract:<name>` extractors live? | `host_contract.EXTRACTORS`, one pure function per name, **closed**: the validator refuses an envelope naming an unregistered extractor. Code, not JSON — no runtime process proposes a host entry (the `aspects_v1.json` case), and entries bind behaviour |
| D4 | `events` derived or declared? | **declared, test-verified both ways** (`TestContractManifestParity`): `contract.events == manifest events`, `manifest events ⊆ engine_events`, every matcher tool name declared (as tool or alias), every declared tool captured by a PostToolUse matcher. The engine lists live in the contract, dated by `verified.host_version` — Claude Code's reference lists 33 events at 2.1.263, not the 13 the first draft assumed |
| D5 | is `blind` enforced? | detect **changed** blind status in either direction; report only, no backfill |
| D6 | version policy | `VOCAB_VERSION` + `contract_fingerprint()` (module content identity) stamped per row; runtime-observed host version separate from `verified.host_version` |
| D7 | the five `'Bash'` sites | **two changes, in order:** (a) `_rollup_line` keeps ONE global `subs` Counter rendered per display tool — make it per tool first; this removes the cross-contamination on today's code and needs no kind at all; (b) then `kind` on `_Action` for the three capture-side predicates |
| D8 | canonical argument normalisation | **deferred** (Tom, 2026-09-07). Not in this work |
| **D9** | **where does classification run?** | **Classify at the first consumer's boundary** (Tom, 2026-09-08, option B). Tool kinds: the DAEMON, at the S0 write door — the first consumer is the encoder hours later; restart-deployable; every client's rows get the same vocabulary the same day; unknown-tool warnings land in the errors table, not a hook's swallowed stderr. Envelopes: the PROMPT HOOK, which consumes the class itself (wake → register_only, 4s timeout, decided before the daemon is called) — Tom's hook-declares ruling (`27945678`) unchanged. Identity: hook observes, daemon resolves. The honest cost: `_build_summary`'s branch table stays hook-side until canonical arguments, mirrored by test |
| **D10** | read-time classification of legacy rows? | **No.** See §6 |
| **D11** | how do we know it can't drift? | **Every surface pair that must agree has a named test; every KNOWN host literal (tool names, host keys, envelope tags) outside the contract is ratcheted.** A genuinely NEW host name is invisible to the ratchet by construction; its detector is the write door stamping `kind_status 'unknown'` into the errors table (step 1). The ledger below is part of the definition of done for each step |

**Open — Tom's:** env_message phase 2 on the flip-day checklist, or its own
`s1_encode_eval`-gated step? Recommendation: its own step, since phase 2 reclassifies rows the
encoder currently reads (`ca3446a3`) and one eval failure would otherwise stall the checklist.
Steps 0–2 do not depend on it; step 3 does.

## Drift ledger

Two kinds of agreement: **single-sourced** (one definition, importers derive — cannot drift)
and **mirrored** (a process boundary forces a copy — a test holds the copy to the source, both
ways). Anything not in this table is a gap; add the row before adding the code.

| surface A | surface B | why a copy exists | held by |
|---|---|---|---|
| `host_contract.tools` values | `trace_contract.ACTION_KINDS` | none — imported | `validate_host_contract` + `TestContractIsClean` |
| `envelopes` values | `ENVELOPE_POLICIES` / `EXTRACTORS` | none — imported; registry closed | validator + `TestValidatorHasTeeth.test_unregistered_extractor` |
| contract `events` | `hooks/hooks*.json` events | manifests are JSON the host reads | `TestContractManifestParity.test_declared_events_equal_registered_events` |
| contract `events` | `engine_events` (host docs, dated) | host's release cadence | `test_registered_events_within_engine_events` |
| manifest PostToolUse/PreToolUse matcher names | contract `tools` ∪ `matcher_aliases` | manifests are JSON | `test_matcher_tool_names_are_declared`, `test_every_declared_tool_is_captured_by_a_post_tool_matcher` |
| `post_tool_trace._build_summary` branch names **and rendered heads** (`'Bash: …'`) | contract `tools` ∪ aliases | the summary needs `tool_input` fields (hook-side until canonical args); the encoder reads the HEAD as the tool name, so the heads are the coupling that carries behaviour | `TestHookMirrors.test_build_summary_branches_are_declared_tools`, `test_build_summary_rendered_heads_are_declared_tools` |
| `hook_common.host_name` env reads and its returned host keys | `all_tell_env_vars()`; contract keys | env is visible only in the hook process | `TestHookMirrors.test_host_name_probes_only_declared_tells`; the ratchet fences the quoted keys until step 1 retires them (hook reports tells, daemon resolves) |
| `tool_result` stamped keys | `TOOL_RESULT_METADATA_SHAPE` + `TOOL_RESULT_NORMALIZATION_KEYS` | today the hook hand-builds `{'tool'}` and `stamp_s0_session` merges model/host after it; from step 1 the write door builds via the builder **then** stamps the session fields — the builder refuses non-contract keys, so stamp-then-build raises. The chokepoint checks required keys and their types only; extra keys pass | `TestToolResultShape`; `validate_trace_metadata` at `TraceDAL.append` |
| a NEW host tool name anywhere | nothing — undeclared names are invisible to a contract-derived scan | by construction | not the ratchet: the write door's `kind_status == 'unknown'` errors-table warning (step 1) |
| `contract_fingerprint()` | the code that classified a row | stamped per row | D6; `TestFingerprint` |
| `host_contract` import graph | `daemon_config` | hot-path cost | `TestLeaf` (subprocess pin, the `test_caller_stamp` pattern) |
| **every other file** | host tool names / host keys / envelope tags | must not exist | `test_host_shape_guardrail` — per-file ratchet, both ways, tool and key sets **derived** from the contract; baseline = today's 36 quoted sites in 12 files (quoted mentions in comments count, as retirement bookkeeping), each row naming the step that retires it. `hooks/adapters/` is not scanned: a host's own setup code is host-specific by design (`368b15af`) |
| `WAKE_ENVELOPE_MARKER` | `pre_response_recall`, `dashboard/queries/stats.py` ×2 | hook routing; dashboard may not import `servers/` | ratchet baseline until step 3; step 3 adds a dashboard mirror test (the `S0_SESSION_STAMP_FIELDS` pattern in `dashboard/queries/_meta.py`) |
| this doc | the code | prose | symbols only, no line numbers; the tests are the truth |

## 10. Step order

| # | step | risk | deploy |
|---|---|---|---|
| 0 | **BUILT.** `HOST_CONTRACT` + CC and Codex entries + validator + fingerprint; output vocabularies + `tool_result` shape/builder in `trace_contract` (required key: `tool` only — what every writer already sends); boot validation; the drift fences; `is_machine_turn` reads the constant | no behaviour change | merge; restart optional |
| 1 | hook sends raw facts (tells present, `tool_use_id`, `turn_id`/`prompt_id`, `payload_keys`, capped patch body) — **the only hook change in the plan**, additive; write door builds via `build_tool_result_metadata` **then** `stamp_s0_session` (build-then-stamp — the builder refuses `model`/`host`); `kind_status 'unknown'` → one errors-table row (the detector for a new host name); normalization keys join the shape's required set; `HOST_STATUS 'legacy'` for old clients; per-event `host` never mutates session env; `hook_common.host_name` retires its host-key returns (ratchet baseline lowered) | additive, see below | redeploy (hook) + restart |
| 2 | D7(a) per-tool `subs`; D7(b) `kind` on `_Action`; flip `WRITE_ACTION_TOOLS`, the five `'Bash'` sites, the four `'Edit'` defaults; three-state legacy read; ratchet baseline lowered for each retired site | first behaviour change; `s1_encode_eval` before/after on the shadow-stamped window | restart |
| 3 | envelopes: populate `envelopes` for both hosts + `extract:question_reply`; prompt hook classifies from the contract (retires its literal); retire the downstream `<task-notification>` readers **after** §6's legacy path exists; dashboard mirror test | removing readers early re-admits machine chatter to recall/presence | redeploy + restart; Tom's gate for phase 2 |
| 4 | reconciliation against the host's own record | needs step 1's source IDs | restart |

**"Additive" has a hard boundary.** Consumers parse rendered summary text; edit summaries
truncate to a path and Bash commands to 200 chars, so a caption change is already an encoder
input change. Step 1 leaves existing summary, content, `metadata.tool` **and** classification
behaviour untouched. **One knowing exception:** `recall_episodes`' `contains` filter greps the
whole metadata blob (`dal_logs`), so a capped patch body in metadata widens lexical matching on
every tool row — accepted, not accidental. Envelope extraction, changed host decisions and any
fail-closed behaviour wait for an explicit cutover. Cost: two representations coexist briefly —
which is what makes a real before/after comparison possible.

---

## Evidence

- `apply_patch` unprotected: `condense_actions` on both shapes, 39-action turn, 3 distinct
  paths → Codex 0/3 edit lines survive, CC 3/3. Node `63fde9b2`.
- `git push` unprotected / bare `rm` protected: probed via `_bash_verb` + `GIT_WRITE_VERBS`.
  Node `b09efd2a`.
- Manifests (jq): CC 10 events / 13 matcher groups / 14 command handlers; Codex 6 / 10 / 11.
  The first draft's "13 / 10" counted matcher groups; D4's test counts events.
- Import cost (`./dev python3`, this worktree): `servers.trace_contract` 1–7 ms and reaches no
  `daemon_config`; `servers.daemon_config` ~28–30 ms.
- Host-stamp gap: 28 of the 1000 most recent `tool_result` rows (2.8%) carry no host, all one
  Codex session on chain `s0-01a07ce3-0`, `apply_patch` ×3, no prompt rows `[review]`.
- Quoted host literals outside the contract: 36 sites in 12 files (guardrail baseline — tool
  names, host keys and envelope tags, after `is_machine_turn` stopped duplicating the marker;
  8 of the 36 are the two host keys, 4 of those in `hook_common.host_name`).
- Lean post-merge review of step 0 (one Opus pass, 2026-09-08): no blockers; five should-fix
  items, all applied — the validator now reports malformed shapes instead of raising (a raise
  would be swallowed by the boot guard and log nothing), the summary heads the hook renders
  are held to the contract, an intra-host duplicate tell is refused, the ledger's `tool_result`
  row now states build-then-stamp, and the ratchet's blind spot for undeclared names is
  written down with its real detector.
- Hook-process env inventory (`pre-bash-safety.sh`, PID 59112) `[codex-stream]`. Node `2f1ee97e`.
- Codex envelope literals from `ChatGPT.app/Contents/Resources/app.asar` — nine headers plus
  the `## My request(?: for Codex)?:` marker; absent from the `codex` Rust binary. Node `4c42b9da`.
- Question-reply envelope `[codex-stream]`. Node `44ac17e0`.
- Hosted-web blind spot: 4 Extension ops in 25s, zero in S0 `[codex-stream]`.
- Payload keys: Codex — [CODEX-ADAPTER-RESEARCH.md](CODEX-ADAPTER-RESEARCH.md) stdin section;
  Claude Code — hooks reference at 2.1.263 (`tool_use_id`, `prompt_id`). Not yet observed on a
  captured row: step 1's `payload_keys` closes that.
- Approved architecture `1f2b3f89`; contract framing `ccc17d81`; placement rule (D9) ruled by
  Tom 2026-09-08.

**Caveat carried from the review:** protocol presence is not end-to-end availability. The
0.153.4 schema documents capabilities that are **absent or unverified** on the installed
desktop build `[codex-stream]`. A contract entry may only claim what has been observed on the
build in use.
