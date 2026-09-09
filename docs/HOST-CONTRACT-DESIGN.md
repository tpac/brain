# Host Contract — Design

## Status — step 2 deployed; both hosts pass live capture/rendering (2026-09-08) ◀ ACTIVE ARC

**Step 2 checkpoint:** `codex/host-contract-step2`, based on main git:`450cff9`.
Implementation git:`8ade88b` is on main and deployed from `/Users/tpac/brain`.
D7(a) per-display-tool counters passed before D7(b) kind consumers were built.
Three read states, raw MCP identity, and all four absent pre-edit defaults are
covered; the legacy behavior is frozen, with four explicit ratcheted names.
Current ratchet: **19 sites in 5 files**. Independent functional review cleared
the consumer changes, including 1,044 legacy episodes across 15 busy windows.

The fixed snapshot/time comparison reaches 131 production action episodes and
five stamped edits. Patch retention rises from 2/5 to 5/5; two display lines and
95 characters are added. System/tools and all prompt text outside `<actions>`
are identical. Verification covers 656 distinct passing tests plus one existing
xfail (640 broad-tier passes; four permission-limited tests passed on rerun;
twelve eval-gate tests). The public-tree export/collection check passes.
**Deployment authorization:** after reviewing the before/after and clarifying
that step 2 changes encoder input preparation, Tom asked to deploy now and offered
to exercise Claude Code and Codex for production validation. Proceed with the
reviewed change and verify live capture plus rendered action retention. The paid
`s1_encode_eval` comparison remains unrun: automatic approval review rejected
sending its private frozen prompt to Anthropic without explicit payload/destination
authorization. Live capture/rendering evidence does not claim a model-output A/B.
Both live host checks now pass: Codex's distinct middle edit is omitted by the
old condenser and retained by the deployed one in a 76-action turn; Claude Code's
write and edit remain visible in a 41-action turn. The actual timeline renderer
consumes every tested edit. Deployment evidence is recorded below.

**Read first:** handoff id:`39ffdd98` (step 2), milestone id:`7e80cb51`
(review/deployment), decision id:`e8183445` (D9). Prior step 1 handoff id:`66ba8b6c`
is superseded; corrected gaps id:`f48d0402`, id:`661abc55`, id:`05fd53c3`.

**Central finding.** Every new dispatched S0 tool row, including sessionless rows,
gets its kind at the daemon write door; hooks send raw facts. Both hosts have live
stamped rows: Codex trace:`caad7588`, Claude Code trace:`6bc6775e`, contract fingerprint
`32da0b4f35b1`. Final follow-up tier: 315 passed, one existing xfail; independent
review cleared the attribution fix with 130 tests. Step 1 ratchet: 29 sites in 10 files.

**Locked:** D1–D11 (§9); raw tool names, summaries and encoder behavior preserved
through step 1. Tom approved retiring the new, unreleased prompt/Stop `host` wire
field and `hook_common.host_name` (id:`b3675380`).

**Open — Tom's:** env_message phase 2 placement, gating step 3 only. Steps 3–4
remain unbuilt. Step 2's code A/B uses a frozen database AND rendering clock;
`s1_encode_eval --compare` alone is insufficient (id:`05fd53c3`, id:`0c958683`).

**Do not reopen:** hook-side tool classification (D9); read-time backfill (D10);
canonical arguments (D8); `_bash_verb`/`GIT_WRITE_VERBS` (id:`b09efd2a`, separate fix);
retiring task-notification readers before step 3; doc field candidates as universal schemas.

**State checked 2026-09-08 16:20Z:** step 1 and its review fix reached main at
git:`d9fa401`; earlier docs git:`8a1d19c`/git:`f1d3bd4` are preserved ancestors.
Claude Code redeploy and Codex reinstall completed from the durable pinned checkout
`/Users/tpac/brain`, then on `codex/contract-host` at git:`d9fa401`; all 11 runtime
files matched both installed copies. Daemon code fingerprint `b2e83b6fa34ad0ba`
matched that revision; source ownership unchanged. A sibling subsequently merged
presence/MCP changes at git:`46f773d`. Those are outside this step's deployed revision.
Recheck main, the actual checkout branch and daemon `source_dir`/fingerprint before
any next merge or deploy; `/Users/tpac/brain` is not necessarily the main checkout.

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

**Before step 1, the tool path never united.** `post_tool_trace._build_summary` branches per host tool name
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
is the probe list; `hook_common.host_tells` is held to it by `TestHookMirrors`). The rule lives
in `resolve_host`: one host's tells → `strong` or `family`; tells of two hosts →
**`ambiguous`, host `''`, never a pick** (5c1a0846's both-present case, now a state, not a
silent first-branch win); none → `unknown`. A tool client with no `tells` key uses the session host and stamps `legacy`; an
explicitly empty tell list stamps `unknown`. Prompt/Stop hooks no longer send a resolved
`host`, and the daemon updates session host only from strong/family tells.

**Hazard the first draft missed:** `SessionContext.set_env` ignores empty values, so a later
`''` retains the prior host — an unknown event would silently **inherit stale certainty**
`[codex-stream]`. `[review]` measured the gap: 2.8% of recent `tool_result` rows carry no host,
all in one Codex session that ran `apply_patch` before any prompt row — exactly the population
this contract classifies. Fix: per-event tells on every tool event (three declared env probes, zero
new imports), the stamper explicitly writes the resolved event host, including empty on uncertainty,
before `stamp_s0_session`'s `setdefault` merge; and the
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

**Before step 1, reconciliation lacked source IDs.** `post_tool_trace` wrote
session_id, stop chain, capped summary and `metadata.tool`, dropping payload IDs: Codex documents `tool_use_id` + `turn_id` on Pre/PostToolUse
([CODEX-ADAPTER-RESEARCH.md](CODEX-ADAPTER-RESEARCH.md) §stdin); the Claude Code doc lookup
names `tool_use_id` and `prompt_id`, but those names remain candidates until observed.
Step 1 copies whichever IDs are present (both `turn_id` and `prompt_id` if supplied), **and adds
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

Before step 2, `parse_action` read `metadata.tool` only for drop/stub; behavior
came from the **summary head**. Step 2 carries `kind`, its internal read status
and raw tool identity on `_Action`; every row before the write-door cutover
still has no kind stamp.

Read with **three** states, not two:

| state | row | behaviour |
|---|---|---|
| i | no normalisation stamp | legacy: today's summary-head path |
| ii | stamp + `kind_status == 'ok'` | normalised |
| iii | stamp present, `kind_status == 'unknown'` / malformed | **visible error policy — never a silent fall back to the summary head** |

The reader detects a stamp by the presence of any of `kind`, `kind_status`,
`vocab_version`, `impl_identity`. Historical join IDs/host fields alone do not
constitute classification. Valid means a recognized kind with `ok`, a supported
integer vocabulary version (currently 1), a nonempty implementation identity and
raw tool string. Incomplete/invalid stamps and unsupported versions render a
plain-language `tool kind …; action unclassified: …` diagnostic that survives
drop/stub and rollup. It adds no new markup notation or prompt glossary syntax.
Dedup requires matching kind/read state and, for stamped rows, raw tool identity;
an unstamped row cannot absorb a protected edit or a diagnostic. MCP drop/stub
still checks the raw server/operation identity, and only for normalized MCP kinds
or the unchanged legacy path.

`LEGACY_SUMMARY_KINDS` holds the four pre-cutover shell/edit names, deliberately
not derived from the input contract (handoff gap id:`a9386c49`). The five normalized
predicates and write-tool set are retired, while these four legacy exceptions
remain visible in the ratchet. No historical patch gains protection.

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
| `hook_common.host_name()` | retired in step 1 | `host_tells()` reports which of `all_tell_env_vars()` are present; `resolve_host` in the daemon decides |
| `hook_common.turn_model()` | payload else CC-only transcript scan | stays; `transcript.grammar`/`readable` declares what it supports |
| `post_response_track` transcript scan | CC-only top-level `type: human/user` | same treatment; a second transcript accessor |
| `post_tool_trace` | 9 per-tool-name summary branches; drops IDs | **sends raw facts only**: + tells present, `tool_use_id`, `turn_id`/`prompt_id`, `payload_keys`, capped `tool_input['command']` for the patch body. Summary unchanged (deferred); no server imports added |
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
| `post_tool_trace._build_summary` / `_raw_metadata` branch names **and rendered summary heads** (`'Bash: …'`) | contract `tools` ∪ aliases | the summary needs `tool_input` fields (hook-side until canonical args); the encoder reads the HEAD as the tool name, so the heads are the coupling that carries behaviour | `TestHookMirrors.test_build_summary_branches_are_declared_tools`, `test_build_summary_rendered_heads_are_declared_tools` |
| `hook_common.HOST_TELL_ENV_VARS` + `host_tells()` | `all_tell_env_vars()` | env is visible only in the hook process; zero-import mirror | `TestHookMirrors.test_tell_probe_mirror_is_exact_and_observes_names_only`; step 1 retired all four host-key sites |
| `tool_result` stamped keys | `TOOL_RESULT_METADATA_SHAPE` + `TOOL_RESULT_NORMALIZATION_KEYS` | the hook sends raw metadata; the write door builds via the builder, writes resolved event host, **then** stamps the session fields — the builder refuses non-contract keys, so stamp-then-build raises. The chokepoint checks required keys and their types only; extra keys pass | `TestToolResultShape`; `validate_trace_metadata` at `TraceDAL.append` |
| a NEW host tool name anywhere | nothing — undeclared names are invisible to a contract-derived scan | by construction | not the ratchet: the write door's `kind_status == 'unknown'` errors-table warning (step 1) |
| `contract_fingerprint()` | the code that classified a row | stamped per row | D6; `TestFingerprint` |
| `host_contract` import graph | `daemon_config` | hot-path cost | `TestLeaf` (subprocess pin, the `test_caller_stamp` pattern) |
| **every other file** | host tool names / host keys / envelope tags | must not exist except the frozen D10 reader | `test_host_shape_guardrail` — per-file ratchet, both ways, tool and key sets **derived** from the contract; baseline = 19 quoted sites in 5 files after step 2 (14 old sites retired, 4 frozen legacy names added). Quoted comments count; `hooks/adapters/` remains host-specific by design (`368b15af`) |
| `WAKE_ENVELOPE_MARKER` | `pre_response_recall`, `dashboard/queries/stats.py` ×2 | hook routing; dashboard may not import `servers/` | ratchet baseline until step 3; step 3 adds a dashboard mirror test (the `S0_SESSION_STAMP_FIELDS` pattern in `dashboard/queries/_meta.py`) |
| this doc | the code | prose | symbols only, no line numbers; the tests are the truth |
| rollup display tool | its own subcommand counts | counts derive from the action records, never a global pool | `test_rollup_subcounts_belong_to_each_display_tool` |
| historical summary behavior | frozen `LEGACY_SUMMARY_KINDS` in the reader | D10 forbids deriving history from today's input contract; four intentional ratcheted names remain | `test_legacy_summary_behavior_is_frozen`, existing legacy action tests |
| stamped action vocabulary | reader's `SUPPORTED_ACTION_VOCAB_VERSIONS` | a changed meaning requires explicit reader support; unknown versions are diagnostic | `test_reader_supports_current_write_vocabulary`, `test_invalid_stamps_never_use_legacy_behavior` |
| kind/read state and raw MCP identity | action protection, drop/stub, dedup and diagnostics | derive from parsed records; same summary across states or namespaces is not the same action | `test_mixed_states_do_not_dedup_or_lose_diagnostics`, `test_normalized_mcp_namespace_policy` |

## 10. Step order

| # | step | risk | deploy |
|---|---|---|---|
| 0 | **BUILT.** `HOST_CONTRACT` + CC and Codex entries + validator + fingerprint; output vocabularies + `tool_result` shape/builder in `trace_contract` (required key: `tool` only — what every writer already sends); boot validation; the drift fences; `is_machine_turn` reads the constant | no behaviour change | merge; restart optional |
| 1 | **BUILT.** hook sends raw facts (tells present, `tool_use_id`, `turn_id`/`prompt_id`, `payload_keys`, capped patch body) — **the tool-capture hook change**, additive; write door builds via `build_tool_result_metadata` **then** `stamp_s0_session` (build-then-stamp — the builder refuses `model`/`host`); `kind_status 'unknown'` → one errors-table row (the detector for a new host name); normalization keys join the shape's required set; `HOST_STATUS 'legacy'` for old clients; per-event `host` never mutates session env; `hook_common.host_name` and the prompt/Stop host wire field retired with Tom’s approval (ratchet baseline lowered) | additive, see below | redeploy (hook) + restart |
| 2 | **DEPLOYED; both hosts verified live.** D7(a) per-tool `subs`; D7(b) kind consumers; four pre-edit defaults now empty; three-state read; ratchet lowered, frozen legacy exception explicit | first behavior change; fixed-input and live capture/rendering checks passed; paid model-output comparison unrun | git:`8ade88b`; daemon fingerprint `a5eb8f3f2b48ecee`; both installed copies synchronized |
| 3 | envelopes: populate `envelopes` for both hosts + `extract:question_reply`; prompt hook classifies from the contract (retires its literal); envelope rows land as their own correspondent (`env_message`, dial OFF in `S0_CONVERSATIONAL_INCOMING`) — the harness joins operator/stream/brain as a typed speaker (`3570a1bd`), so consumers select it out by ref_type instead of matching the marker text; retire the downstream `<task-notification>` readers **after** §6's legacy path exists; **verify the no-filter consumers** — `self_peek`'s turn_count (`dal_logs`) and the encoder lived timeline (`encode.py`, via `SAID_AND_DID_REF_TYPES`) never filtered the marker at all, so neither surfaces in the reader sweep or the ratchet; both should read correctly once the ref_type moves, so assert it rather than assume it; dashboard mirror test | removing readers early re-admits machine chatter to recall/presence. The mirror risk: a consumer that never filtered has no literal to retire and no ratchet entry, so it fails silently in the other direction — today the encoder renders a wake envelope as a `USER` turn (it opens a turn and is attributed to the operator), and `self_peek` counts it toward turn_count while the Scribe cadence does not | redeploy + restart; Tom's gate for phase 2 |
| 4 | reconciliation against the host's own record | needs step 1's source IDs | restart |

**"Additive" has a hard boundary.** Consumers parse rendered summary text; edit summaries
truncate to a path and Bash commands to 200 chars, so a caption change is already an encoder
input change. Step 1 leaves existing summary, content, `metadata.tool` **and** encoder classification
behaviour untouched. **One knowing tool-capture exception:** `recall_episodes`' `contains` filter greps the
whole metadata blob (`dal_logs`), so a capped patch body in metadata widens lexical matching on
every tool row — accepted, not accidental. Envelope extraction, encoder classification changes and fail-closed behaviour wait
for an explicit cutover. Tom explicitly approved the prompt/Stop identity cutover in
step 1: only strong/family tells update session host; unknown/ambiguous tells preserve
its last known value. Cost: two representations coexist briefly —
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
- Step 0 measured baseline of quoted host literals outside the contract: 36 sites in 12 files (tool
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
- Payload keys now observed on captured rows (id:`34e7be01`): Codex `turn_id`, Claude
  Code `prompt_id`, both `tool_use_id`; Codex subagents also send `agent_id`/`agent_type`.
  `payload_keys` records the actual top-level names. These observations do not make
  either host's current field set a universal schema.
- Approved architecture `1f2b3f89`; contract framing `ccc17d81`; placement rule (D9) ruled by
  Tom 2026-09-08.

**Caveat carried from the review:** protocol presence is not end-to-end availability. The
0.153.4 schema documents capabilities that are **absent or unverified** on the installed
desktop build `[codex-stream]`. A contract entry may only claim what has been observed on the
build in use.

### Step 1 implementation record

| choice | implementation / evidence |
|---|---|
| sessionless rows | Stamped at `_handle_trace_append`, including malformed/missing metadata retained under `raw`; no read-time backfill. |
| legacy vs uncertain | Only an absent `tells` key selects session host with `legacy`; present empty/ambiguous tells stamp empty event host. |
| source IDs | Preserve `tool_use_id`, `turn_id`, `prompt_id` independently when present. Required normalization join fields default to empty string/list for old clients; no IDs are derived. |
| patch cap | 16,384 characters; `patch_truncated_chars` records omitted characters. Patch capture adds one mirrored tool-name literal until canonical arguments. |
| unknown-kind logging | `_log_error('tool_kind_unknown', …)`; host/tool hash before message text prevents prefix-based dedup collisions in the logger's 100-character fingerprint. Uses existing 60-second dedup window and source/global rate limits. The event's raw session ID is passed explicitly, including empty for sessionless rows, and included in diagnostic context. |
| production behavior | Tool summaries and encoder behavior preserved; prompt/Stop host resolution cutover approved by Tom (`b3675380`). Old tool clients use the legacy path until cache reinstall. |
| review pass 1 — structure | Same-agent pre-commit call-boundary review: hook → dispatch → stamper/builder → session merge → DAL; prompt/Stop share one resolver/update helper. No read-time backfill, no encoder dependency on host_contract, no server import in the live tool-hook path. |
| review pass 2 — function | Same-agent pre-commit boundary review: old clients, pre-prompt/sessionless rows, ambiguous/empty/unknown tells, spoofed kind fields, source IDs, patch cap and caller-stamp redaction exercised. Summary/stop functions match main by AST; host contract matches main by AST (one retired-helper comment updated); encoder files match byte-for-byte. All seven action fields match across six tool families; a 48-action condensation matches before/after stamping. |
| review fixes | Explicit event host before session merge; preserve unknown tell names; remove retired host diagnostic; mock the prompt hook’s fast process exit in the harness. |
| tests | Expanded tier: 465 passed, one pre-existing xfail (576.96s). Final boundary suite after review fixes: 85 passed. After fast-forwarding the dashboard-only main head, the overlap/contract/deploy tier passed 185 tests with one pre-existing xfail (22.58s). The sandbox-only process-name failure passed with process inspection allowed. Installed the missing pinned pytest-timeout dependency; the later boundary run has no timeout warning. |
| independent review, requested after initial merge | A separate reviewer checked commit `8612ff7` against `bc29b60`, including real hook execution, writers/readers and a targeted simplify pass: 219 tests passed, no blocking finding. One diagnostic defect: unknown-kind errors inherited another session's global ID. No further simplification warranted. |
| review follow-up before fix commit | Added an explicit optional session ID to the existing logger door, avoiding a fabricated SessionContext. The reviewer cleared the fix with 130 passing tests, 16 subtests and six attribution-precedence probes. The new regression checks named and sessionless event attribution while another global session is active. Its initial read-only-property fixture mistake was corrected before this passing run. |
| final fix gate | Contract/guardrail/deploy/dispatch and trace tier: 315 passed, one existing xfail, 16 subtests (256.92s); 41 existing upstream embedder warnings. Core/session/LLM-latch run separately covered 136 other tests; the corrected attribution regression is included in the passing final tier. |


Reproduce the expanded step 1 tier (stage new files before the public-tree export gate):

```bash
./dev python3 -m pytest \
  tests/test_host_contract.py tests/test_host_shape_guardrail.py \
  tests/test_hooks_manifest_sync.py tests/test_trace_contract_sync.py \
  tests/test_caller_stamp.py tests/test_s0_session_stamp.py \
  tests/test_encoder_actions.py tests/test_raw_sql_guardrail.py \
  tests/test_traces_layer_guardrail.py tests/test_deploy_contract.py \
  tests/test_query_traces_truncation.py tests/test_trace_system.py \
  tests/test_tool_result_stamp.py tests/test_daemon_hooks.py \
  tests/test_hook_output_contract.py tests/test_run_hook_contract.py \
  tests/test_hook_daemon_call_logging.py tests/test_trace_integration.py \
  tests/test_self_delivery.py tests/test_contract_sync.py tests/test_session_context.py \
  -q --no-header -p no:cacheprovider
```

The logger follow-up also covers `tests/test_core.py`,
`tests/test_llm_rejection_latch.py` and `tests/test_dispatch_contract_sync.py`;
compose these with the tier above when changing the logging boundary.

**Step 2 eval preparation:** `eval/s1_encode_eval.py --compare` compares a prompt
file and tool set, not two code revisions. Freeze an isolated database/window and
select one session before the consumer change, then run both code versions on that
same input. Validate that the harness reaches the changed production action path;
two fresh snapshots of moving live data are not a paired baseline (brain id:`05fd53c3`).

### Step 2 implementation and evaluation record

The recording call stack stays `post_tool_trace` → `_handle_trace_append` →
`stamp_tool_result` (host resolution/classification, metadata builder, session
stamp) → `TraceDAL.append`. The consumption stack is `encode.run_encoding` →
`_build_user_content` → `_render_lived_sequence_timeline`; the renderer calls
`_lived_turns` to read episodes, then `condense_actions` → `parse_action`, `_dedup`
and `_rollup_line`. Step 2 replaces the parser's name-based behavior with a
three-state stamp read. The hook retains argument-shaped caption extraction;
the daemon owns input dialects; the encoder consumes output kinds and the
frozen legacy table. No new service or classification layer was introduced.

The pre-edit path stays `pre_edit_suggest` → `hook_pre_edit` → `brain.pre_edit`
→ `suggest` / `procedure_trigger`, with `_handle_pre_edit` as the direct API
ingress. Absent tool names now remain empty at all four doors.

The per-tool counter change was built and checked first (30 focused tests).
Kind consumers then passed 82 action/view tests. The author traced hook capture,
daemon stamping, episode reads, production prompt assembly, condensation and
pre-edit callers; no read-time classification or capture-caption changes were
introduced. The independent functional pass cleared the implementation with
1,044 legacy episodes across 15 windows, 73 adversarial parser/writer cases,
unknown floods, per-tool caps, and 13 pre-edit probes. Its harness follow-up
cleared the frozen-clock, offline-source and comparison-gate fixes.

The 27-file broad tier finished with 640 passes, one existing xfail and four
sandbox-only failures (process inspection and synthetic loopback binds). With
those permissions, the four passed. Staging the new eval test exposed D-8's
public-export requirement: tests reaching the omitted `eval/` tree must call
`tests.eval_optional.require_eval()` before importing it. Fixed; final export
and nine eval tests passed together (10 passes). Public export/collection is
verified; this does not claim a deployment.

`eval/host_contract_eval.py` prepares both code arms using production message,
catalog, system prompt, tools and condensation paths against one frozen
`IsolatedBrain` snapshot. A timezone-aware `--now` pins both catalog and timeline
rendering; `--check-pair BEFORE AFTER` rejects input drift outside actions,
including different execution settings, episode windows or tail budgets.
The optional `--encode` uses the existing dry-write `s1_encode_eval` helper:
Sonnet 4.6, 4,096 output tokens per call, default API effort, at most nine calls
per arm. Its applied settings are recorded separately from the frozen production
config (medium effort). Both arms use the same helper; this is a controlled
input comparison, not a claim of production-loop equivalence.

Tom requested fresh architecture, functional and simplification reviewers after
the initial pass. Architecture cleared placement before simplification began.
The fresh functional pass found two defects, both fixed and independently
rechecked: unknown/malformed multiline diagnostics now mark their omission;
the eval gate checks actual rendered edit counts and verifies measured lines
reach prompt action blocks, rather than trusting independently parsed protection
flags. It also preserves the unmeasured encoded-turn stubs between arms. The
reviewer's `ALL ACTIONS LOST` artifact now fails the gate. Simplification moved
model/token/tool-round settings into the dry-write runner and made its calls
and reporting share those constants; a fake-client test verifies both initial
and follow-up calls. One redundant type check was removed. All three reviewers
cleared the follow-up changes; three new tests bring the eval suite to twelve.
The final affected tier passed all 152 tests, and a fresh public-tree
export/collection check passed after those fixes.

Local artifacts for this run live under `/private/tmp/host-contract-step2/`:
`before-reviewed/`, `after-reviewed/`, `comparison-reviewed.json`, `prompt-final.diff`, test logs,
and `EVALUATION.md`. The baseline production code is an archive of git:`450cff9`
under `baseline-code/`; both arms use the same reviewed eval scripts and carry
source hashes. Regenerated payloads match the earlier prepared inputs byte for
byte; helper hashes now reflect the shared-constant simplification. Snapshot location is recorded
in `frozen-path.txt`. All payloads remain local; the external API call was
rejected before execution. Tom subsequently requested deployment and offered
hands-on Claude Code/Codex production testing. That authorizes proceeding with
live capture/rendering acceptance; the paid model-output comparison remains unrun.

To prepare each arm (from its own source checkout, using the same frozen path,
session and instant), then compare:

```bash
./dev python3 eval/host_contract_eval.py \
  --source-dir "$STEP2_FROZEN_DIR" --session "$STEP2_SESSION_ID" \
  --now 2026-09-08T17:25:48+00:00 --out "$STEP2_ARM_OUTPUT"
./dev python3 eval/host_contract_eval.py \
  --check-pair "$STEP2_BEFORE_OUTPUT" "$STEP2_AFTER_OUTPUT"
```

Any future `--encode` run still needs explicit external-payload approval and
fresh output directories; recheck its inputs against the prepared arms. For the
now-authorized deployment, merge verified main, recheck the actual pinned daemon
owner and fingerprint, update
the changed pre-edit hook in both installed copies, restart, and verify fresh
stamped rows. The env-message phase-2 decision still gates step 3 only.

### Step 2 deployment and live verification

Tom requested deployment and offered hands-on production tests after the local
reviews. Implementation git:`8ade88b` fast-forwarded the clean main checkout at
`/Users/tpac/.codex/worktrees/1f82/brain`, then the pinned service source at
`/Users/tpac/brain` on `codex/contract-host`. `redeploy.sh` built 235 files,
passed both packaged entrypoint imports, refreshed Claude Code and re-exec'd
the daemon. Ping verified the same pinned owner/database and code fingerprint
`a5eb8f3f2b48ecee`; contract fingerprint remains `32da0b4f35b1`.

Codex was reinstalled from its existing local `anchor-dev` marketplace with
version `0.9.0+codex.20260908194926`. The CLI removed the former `0.9.0` cache
path still referenced by this active task's hooks. Restored the same deployed
package at that path for session continuity and warmed both runtimes. The
compatibility copy encountered abandoned bootstrap locks from short-lived hook
startup attempts; process inspection and `kill(pid, 0)` proved each blocking
owner dead before an atomic rename. An uninterrupted installer then completed.
All 235 packaged files match Claude Code, the new Codex installation and the
active-task compatibility copy; only the new installation's intended manifest
version suffix differs. Fresh Codex tasks should use the new installation;
an old resident proxy's setup status is not evidence about the new installation.

Live test evidence (real host tool calls, no synthetic trace insertion):

| host | captured turn | result |
|---|---|---|
| Codex | session `01a0820b-ecac-72b2-aec9-1afc244d48b8`, user trace `2eecbf6b`, 76 actions at verification | three test edits have `kind=edit`, `kind_status=ok`, strong Codex identity; distinct middle edit `d5e35310` at position 71 is omitted by the baseline condenser and retained by deployed code; all three reach the actual timeline renderer |
| Claude Code | session `eec08916-3794-4f01-a097-f0375ae6e819`, user trace `74d67d3a`, 41 actions | 38 shell, one read and two edits; native Write `2a2a35c4` and Edit `c31fdabf` both have valid edit stamps and strong Claude Code identity; the edit at position 35 survives condensation, and both reach the actual timeline renderer |

The first Codex probe reused the creation's caption, so baseline dedup also
retained it. The distinct-file middle probe above removes that ambiguity.
Read-only verification used the daemon's existing `recall_episodes` and
`query_traces` doors, then the deployed condenser and timeline renderer with
real provenance. No model request or production memory write was made by the
verification script. No completed S1 encoding run existed yet in the fresh
Claude test session; this proves capture through input preparation, not model
output quality. Configuration validation reported only database-size maintenance
warnings. Artifacts: `/private/tmp/host-contract-step2/live-codex/`, `live-claude/`,
`deployed-daemon.json`, `installed-verification.json`, and deployment/runtime logs.
