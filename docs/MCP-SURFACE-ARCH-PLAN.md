# MCP Surface — Architecture Plan

## Scope

`servers/brain_mcp.py`'s tool surface: the 39 tool definitions, their names, descriptions and
schemas, the annotations they don't carry, the `initialize` capability block, and the four
consumer slices that read `TOOLS`. Everything below is a change to what an AGENT SEES, not to
what the daemon does — no dispatch handler, DAL, or scale logic is in scope.

**Baseline measured 2026-09-08** (`messages.count_tokens`, sonnet-4-6, minus a 530-token
empty-tools-block baseline; re-derive before acting, the recipe is in §Measurement):

| slice | source | tools | net tokens |
|---|---|---|---|
| every caller | `brain_mcp.TOOLS` | 39 | 19,314 |
| Anchor · eager | `_meta["anthropic/alwaysLoad"]` (`CRITICAL_TOOLS`) | 13 | 9,212 |
| Anchor · deferred | behind the host's ToolSearch | 26 | 10,069 |
| S1 Scribe | `ENCODING_TOOLS` (`scales/s1/encode.py:1404`) | 6 | 6,860 |
| S2 consolidation + community | `{brain_batch, get_nodes}` | 2 | 2,425 |

Four tools are half the catalog: `remember_batch` 2,299 · `brain_batch` 2,719 (gross) ·
`revise` 1,591 · `revise_batch` 1,528. Reads are cheap (`recall` 333, `get_nodes` 203).

**Prior art recalled and respected:**
- id:4ccb43eb — "memory operations only, no operational tools" (ping/health_check/save/config
  removed). `restart`, `eval`, `clear_errors`, `query_logs` are drift against this. Step 4.
- id:04ff3d58 / id:79b25bac — mechanics in MCP, strategy in the prompt. Step 5 is this rule
  applied to text that drifted the other way.
- id:807394de — an MCP description primes EVERY caller, not the one you were tuning.
- id:1b7984f8 — **hard fence.** The S1E prompt has three lines that delegate mechanics TO the
  MCP descriptions ("the parameter shapes live in the connect_to tool description", and two
  more). Strip MCP text before inlining those six items into the prompt and the encoder
  silently loses sibling resolution, ordering-agnosticism and NEW-wins. Step 5 owns this
  ordering; nothing else may touch encoder-facing description text first.
- id:55f960e5 — one source, three surfaces: a `TOOLS` edit reaches Anchor, S1 and S2 atomically
  after restart. Every step here has all-caller blast radius by construction.
- id:5a71e621 — `CRITICAL_TOOLS` was already curated to 13 (get_node and find_node_by_title
  dropped, get_nodes added). Step 2 finishes what that decision started.
- id:f358bba7 — tool names in prompt prose are load-bearing; an English synonym costs recall of
  the op. Any rename (Step 6) is a prose migration, not a one-line change.
- id:2caf3389 / id:3bab3268 — `SERVER_INSTRUCTIONS` is a LIVE lever on Claude Code (verified
  2026-09-08: it reaches the session system prompt verbatim). The old "dead channel" finding was
  Claude Desktop chat. Step 5 may move cross-tool guidance there instead of paying it per tool.

**Deploy contract for every step in this plan:** `servers/brain_mcp.py` is the one `servers/*`
file a daemon restart does NOT deploy — it needs `./redeploy.sh` (commit first) **and a new
session**. Schema or description changes must re-run `eval/mcp_batch_probe.py` and
`eval/mcp_schema_gate.py` before restart (CLAUDE.md). Diagnostic lens for description edits:
`eval/mcp_tool_interview.py --tool <name>`.

**Non-goals:** `revise_edge`'s batch-op placement (owned by the live worktree implementing
id:73d30b14 — edges as a field of the node's own `revise`). Prompt content itself, except where
Step 5 must inline what it removes.

---

## Step 1 — Tool annotations + the capability block

**Why first:** mechanical, no behaviour change on Claude Code, and it is the whole fix for a
known Codex symptom. Independent of every other step.

Nothing in the catalog carries `annotations` — 0 of 39. Under the spec the defaults are
deliberately pessimistic (`destructiveHint` defaults **true**, `openWorldHint` **true**), and
Codex CLI acts on them: unannotated tools are treated as maximum risk and prompt for approval on
every call, `readOnlyHint: true` tools are eligible for auto-approval AND are executed
concurrently (`agents.max_threads`, default 6), and a user with `destructive_enabled = false`
for our server is hard-blocked from calling anything at all — `recall` included. Claude Code
does not depend on the hints (it runs its own classifier in auto mode), so this step is
Codex-facing upside with no Claude-side risk.

**Do:**
1. Add an annotation map to `servers/contract.py` (contract-first; hooks and dispatch never
   hardcode). Fields per tool: `readOnlyHint`, `destructiveHint`, `idempotentHint`,
   `openWorldHint` (false for all 39 — the brain is a closed domain), and `title` (human display
   name, e.g. `recall` → "Recall memories").
2. Stamp them onto `TOOLS` at build time next to `_stamp_always_load`, with the same
   fail-loud-on-unknown-name check.
3. Fix the capability block: `handle_initialize` returns `capabilities: {"tools": {}}` while
   [brain_mcp.py:1340](../servers/brain_mcp.py) sends `notifications/tools/list_changed` on a
   daemon-fingerprint change. Declare `{"tools": {"listChanged": true}}` or the client is
   entitled to drop the notification.

**Do NOT derive `readOnlyHint` from `COMMAND_TABLE[...].is_write`.** Verified 2026-09-08: 38 of
39 tool names are keys in `daemon_dispatch.COMMAND_TABLE` (only `restart` is absent), but
`is_write` means "dirties brain.db", not "modifies its environment". Four tools are
`is_write=False` and still change state — `remind` (files a Thalamus item), `self_send` (writes
another stream's inbox), `self_inbox` (**drains** consume-once, `dispatch_self.py:89`), and
`thalamus_resolve` (resolves an item). A blind map mislabels all four as read-only, which on
Codex means auto-approved-and-parallelised writes. Classify by hand,
assert the count in a test. Also distinct from `brain_traces.stamp_tool_result`'s tool *kinds* —
that classifies the HOST's tools for trace analytics, different concern, different vocabulary.

The per-tool classification is Appendix A.

**Verify:** a contract-sync test asserting every tool in `TOOLS` has annotations and every
annotated name exists (the `_stamp_always_load` pattern); `tests/test_deploy_contract.py`;
manual check that a Codex session no longer prompts per call on a read tool.

---

## Step 2 — Delete three orphaned tools

`get_node` and `get_trace` are strict subsets of `get_nodes` / `get_traces`, and both are the
more expensive half of their pair (`get_trace` 430 net tokens vs `get_traces` 229 — 1,135 chars
of description). `CRITICAL_TOOLS` already dropped `get_node`; what remains is a deferred decoy
that costs a ToolSearch round-trip when the model reaches for the wrong one.

**`enrich` — the vestigial half of a superseded design (traced 2026-09-08).** It is the second
step of the V5 manual loop: `remember()` returns an `enrichment_prompt`, the agent fills in
question/anchor/bridge/keywords, then calls `enrich()`. Nothing automated does this. Its only
door is the MCP tool (`dispatch_write.py:1370` → `brain.store_enrichments`, whose only caller in
`servers/` is that handler); `ENCODING_TOOLS` and both S2 sets exclude it, so no encoder can emit
it, and `encoder_view.DROPPED_ACTION_TOOLS` drops it from the timeline the Scribe reads. The job
it did is now done by fields: the `question` field on `remember`/`revise` gets its own recall
embedding, and `_situation` enrichment rows are derived at write time (CLAUDE.md). Delete the
tool. `store_enrichments` / `_build_enrichment_prompt` / `ENRICHMENT_PROMPT_TEMPLATE` and
`tests/test_retired_fields.py`'s guard are a deeper retirement — name it, don't widen into it.

**Keep** `recall`/`recall_batch` and the three `_batch` pairs (`remember`, `revise`, `connect`).
Those are not singular/plural — they are different shapes for different consumers, and the field
prose is not actually paid twice: Anchor's eager set has `remember`, the Scribe's set has
`remember_batch`, neither sees the other. A `string | string[]` union would generate worse than
two tools.

**Callers to update before removal** (re-derive, perishable): `servers/scales/s1/encoder_view.py:183`
lists `get_node` in an allowlist; `COMMAND_TABLE` keeps the daemon command (leave it — the
daemon door is not the agent surface); grep both `get_node` and `"get_node"` dict-key form
(id:feedback param-removal rule) plus `get_trace` similarly, and check `dashboard/` and
`eval/` for tool-name references.

**Verify:** guardrail + contract tier, `eval/mcp_batch_probe.py`, and one live `get_nodes` call
with a single id after redeploy.

---

## Step 3 — Cut the interactions block out of the MCP

Six tools (`list_interactions`, `get_interaction`, `get_interaction_effective`,
`register_interaction`, `set_interaction_active`, `clear_interaction_override`), 907 net tokens,
all deferred. **The descriptions are accurate** — read them, they correctly state the
code-default-plus-override model, `register_interaction` even says "NEVER activates". The problem
is not stale text: six tools named `register_/set_/clear_interaction*` in a memory server imply
prompts are DB-managed, when the documented way to change a production default is *edit the .py
and merge*. They advertise the exception as the interface.

**Do:** remove all six from `TOOLS`. Keep every daemon command and `tests/interaction_override.py`
untouched — the maintainer path is `./dev check-overrides` plus the three-call recipe in
CLAUDE.md, which needs no MCP tool.

**Judgement call for the executing session:** keeping `get_interaction_effective` alone is
defensible — "what is `<name>` actually running?" is a real mid-session debugging question and
it is the only one of the six that can see both halves (default + override). Decide once, don't
keep two.

**Verify:** CLAUDE.md's "Deploy an override on THIS install" recipe still executes end-to-end
through the daemon after removal; contract tier.

---

## Step 4 — Operational-tool drift, and `eval`

`restart`, `eval`, `clear_errors`, `query_logs` contradict id:4ccb43eb. Ranked by actual risk:

- **`eval`** — **ships everywhere; annotate it honestly and make it ask to be replaced.**
  Tom's ruling 2026-09-08, reversing the env-flag gate this plan carried: a non-maintainer
  install is exactly where `eval` is needed, because it is the only way to diagnose a brain we
  cannot open ourselves. Hiding it would remove the remote-debug door and buy nothing — a shipped
  install is loopback-bound and single-user, the same risk profile that makes it acceptable here
  (id:d1dd9f2c). The protection is the annotation plus the host's approval, not absence. Keep
  the `COMMAND_TABLE` entry (`eval/oracle_audit/backfill_absorbed_into.py` uses the daemon
  command). See §"`eval` as a self-retiring escape hatch" for the description rewrite.
- **`restart`** — keep. It is how a dev session deploys, and it is the one tool name absent from
  `COMMAND_TABLE`. Annotate `destructiveHint: true`.
- **`query_logs`** — keep, read-only, genuinely used for diagnosis.
- **`clear_errors`** — weakest case. Decide: keep annotated destructive, or drop and leave it a
  daemon-only command.

**Verify:** `eval`'s annotations are the max-risk set (`destructiveHint` + `openWorldHint`
true); with `BRAIN_MCP_MAINTAINER` unset, `tools/list` carries the universal description and NOT
the improvement-loop paragraph; with it set, both.

### `eval` as a self-retiring escape hatch

`eval` exists because a capability has no door. Today that finding dies with the session: the
question gets answered and nobody learns which door was missing. The fix is a short paragraph in
the tool's own description — read at the moment the call is formed, which is the only moment the
question is live (the generation-time-enforcement property, id:55f960e5).

**Two audiences, so two descriptions.** Tom's ruling 2026-09-08: the improvement loop helps the
maintainer and is noise for everyone else — a normal user should never be asked to design an MCP
door. So the danger warning and the mechanics ship to everyone, and the loop paragraph is
appended only when `BRAIN_MCP_MAINTAINER=1`.

**Universal — always shipped:**

```
DANGEROUS — full capability. Evaluates an arbitrary Python expression inside the
daemon process with the live `brain` object in scope: it can read anything in
either database, and can write, archive or delete through any method it reaches.
The `safe_builtins` sandbox is weak and bypassable. Safe only because the daemon
is loopback-bound and single-user — treat every call as an operator-level action.

Use it to reach brain state that has no tool yet, including on an install nobody
can open directly. `brain` and `json` are in the expression's LOCALS, so a lambda
inside it NameErrors on `brain` AFTER its arguments have already run — side
effects included. Use flat expressions.
```

**Appended only under `BRAIN_MCP_MAINTAINER=1`:**

```
After using it, consider whether this should have been a door — an existing tool
with one more parameter or filter field, a new `brain_batch` op, or a tool of its
own. When it should, say so in a line and `remember` it as a `gap` node so the
suggestion accumulates instead of being rediscovered.
```

Deliberately a suggestion with two escape valves ("consider whether", "when it should") rather
than a per-call obligation — Tom's constraint: it must not become an annoyance. The `gap` node is
what makes it compound: a later session runs `filter_nodes(type='gap')` and reads the ranked list
of doors `eval` has been standing in for, instead of the same gap being rediscovered.

**The flag.** `~/.config/brain/env` is sourced unconditionally by `brain-env.sh:47` (it already
carries `BRAIN_OPERATOR_NAME`, `BRAIN_AGENT_NAME`), and `mcp-launch.sh` sources `brain-env.sh`
immediately before exec'ing `brain_mcp.py` — so a line in that file reaches `_build_tools()` with
no new plumbing. Default off; the shipped catalog never carries the paragraph. Do NOT key it off
`BRAIN_OPERATOR_NAME` — a name is not an opt-in.

Cost: `eval` goes from 62 net tokens to roughly 180 universal, ~240 with the maintainer
paragraph — paid only when a caller fetches the tool.

**Three notes for whoever writes this.** First, it is deliberately the one place this plan ADDS
prose to a description while Step 5 removes it. The rule being applied is "mechanics in MCP,
audience-specific craft in the audience's prompt" (id:04ff3d58), and the universal half is pure
mechanics; the maintainer half is audience-specific and therefore *conditioned*, not mixed into
one string — which is the shape Step 5 should aim for wherever an audience split is real. No
encoder can reach `eval`, so there is no cross-caller priming risk (id:807394de). Second, this
introduces the first install-conditional tool text in the catalog: `test_contract_sync` and the
annotation-coverage test must accept both shapes, and one of them should pin the maintainer
sentence under the flag so a later description trim cannot silently drop it. Third, consider
echoing one line of the loop on the RESULT as well, still flag-gated (`_format_result` already
special-cases per tool) — before the call the agent only knows it wants data; *after* the call it
knows which method it reached for, which is when "what door was missing?" is actually answerable.


---

## Step 5 — De-mix consumer-specific text (the heavy one)

**Blocked on the id:1b7984f8 fence — read it before starting.** Order is: audit every prompt
line that delegates mechanics to a tool description → inline those six items into the owning
prompt → only then remove text from MCP. Reversing this silently degrades the encoder.

Confirmed leaks (grep-verified 2026-09-08; several apparent hits were false positives —
"au**tom**atic", "**anchor**ing", "**Operator**s:"):

| where | text | whose |
|---|---|---|
| `remember` description | the whole `ENCODING CRAFT` / `LESSONS — climb the abstraction ladder` / `RICHNESS: Training rewards brevity` block | S1 Scribe coaching, read by Anchor too |
| `recall` description | "when the auto-surfaced context (~25 candidates per turn) didn't catch what you need" | Anchor-only — S1/S2 have no auto-surfaced context |
| `source_refs` (4 tools) | "the trace markers in your input", `[trace:<hex>]` | Scribe-only — only its input carries markers |
| `their_raw_quote` | "Their exact words — **my** counterpart's" | first person, Anchor's frame |
| `enrich` description | "after filling in the enrichment_prompt from remember()" | a workflow, not a contract |

Target shape: mechanics + shapes + failure modes stay; craft, stance and audience-specific
workflow move to the owning prompt. Front-load each description to a 1–2 sentence contract
(current outliers: `brain_batch` 2,399 chars, `remember` 1,413) and let the schema field
descriptions carry the detail. Cross-tool "when to reach for me" discipline that genuinely
applies to every caller can go in `SERVER_INSTRUCTIONS` — one place, no per-tool tax, delivered
on BOTH hosts (id:2caf3389; Codex honours the field too).

⚠ **But that valve is nearly full.** `SERVER_INSTRUCTIONS` is **501 characters** (measured
2026-09-08) against the ~512 our own Codex research says must stay self-contained — 11 characters
of headroom. Anything moved there displaces something already in it, so treat it as a fixed
budget to be re-drafted whole, not an overflow bin. If Step 5 wants real room there, raising the
budget needs a measurement of what Codex actually does past 512, which nobody has taken.

**Verify:** benchmark-first — `eval/s1_encode_eval.py` before and after, plus
`eval/mcp_tool_interview.py` on each edited tool to see how the consumer model reads the new
text. This step changes generation behaviour; do not ship it on inspection alone.

---

## Step 6 — Naming

We are compliant with [SEP-986](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/986)
(accepted, 1–64 chars, `[A-Za-z0-9_.\-/]`): longest mangled name is
`mcp__plugin_entity_brain__clear_interaction_override` = **52 of 64**. The prefix is the host's,
not ours — `mcp__plugin_entity_brain__x` as a plugin, `mcp__brain__x` via `claude mcp add brain`
and under Codex (our own `hooks/hooks.codex.json:54` matches `mcp__brain__.*`). Never hardcode a
full path anywhere.

One real problem: **`remember` and `remind` are one letter apart with unrelated semantics**
(writes a node / files a Thalamus item) — a textbook misroute pair. Bare verbs otherwise read
well here because the server name supplies the noun, so `recall`/`remember`/`enrich` are fine as
they are despite violating `verb_noun`.

**Cost, stated up front:** `remind` is in `CRITICAL_TOOLS`, the boot stance, `SKILL.md`, the
Thalamus docs and the delivery footer prose. Per id:f358bba7 the prose must move with the name in
the same commit. Do this AFTER Step 5 so prompt prose is edited once, not twice.

---

## Step 7 — `outputSchema` / `structuredContent` (own project, not this plan)

Every tool returns a text blob the caller re-parses. The 2025-06-18 revision added
`outputSchema` + `structuredContent`; for a server whose whole job is returning nodes and edges
this is the largest remaining modernization. It touches `_format_result`, `scales/runner.py:612`
(which reuses `_format_result` for encoder tool results) and every consumer's parsing. Scope it
separately; note that `PROTOCOL_VERSION` still defaults to `2024-11-05` while
`SUPPORTED_PROTOCOL_VERSIONS` already accepts `2025-06-18` and `2025-11-25`.

---

## Step 8 — Ordering experiment (independent, eval-gated)

Tool-selection position bias is measured, not folklore (ICLR 2026; shuffling a candidate toolset
dropped one open model 41% → 27%; "lost in the middle" shows mid-list tools at 22–52% vs 31–32%
at the ends — though that was at 741 tools). For Anchor at 39 tools the eager/deferred split
dominates and array order is second-order. Where it plausibly bites is the **6-tool Scribe
view**, which inherits `TOOLS` order and lands as `remember_batch, connect_batch, brain_batch,
revise_batch, get_nodes, recall_batch` — `brain_batch`, the op it should reach for most, sits
third, mid-list.

**Do:** move `brain_batch` first in `_build_tools()`'s literal (or sort the slice in
`_get_tool_schemas`), then A/B with `eval/s1_encode_eval.py`. Treat it as a hypothesis to
measure, not a fix to assert.

---

## Measurement

Re-derive the baseline before and after any step. Exact per-slice token cost:

```python
# ./dev python3 -
import anthropic, sys; sys.path.insert(0, '.')
from servers import brain_mcp
c = anthropic.Anthropic()
def cost(names):
    tools = [{'name': t['name'], 'description': t['description'],
              'input_schema': t['inputSchema']}
             for t in brain_mcp.TOOLS if t['name'] in names]
    n = c.messages.count_tokens(model='claude-sonnet-4-6',
                                messages=[{'role': 'user', 'content': 'x'}],
                                tools=tools).input_tokens
    base = c.messages.count_tokens(model='claude-sonnet-4-6',
                                   messages=[{'role': 'user', 'content': 'x'}]).input_tokens
    return n - base            # subtract another 530 for the fixed tools-block scaffolding
print(cost({t['name'] for t in brain_mcp.TOOLS}), cost(brain_mcp.CRITICAL_TOOLS))
```

---

## Host differences that constrain the design

| | Claude Code | Codex |
|---|---|---|
| tool name seen by the model | `mcp__plugin_<plugin>_<server>__<tool>` as a plugin, `mcp__<server>__<tool>` when added directly | `mcp__<server>__<tool>` (our `hooks.codex.json` matcher) |
| catalog pruning | automatic ToolSearch deferral above ~10% of context; `_meta["anthropic/alwaysLoad"]` forces eager (vendor extension, ignored elsewhere) | user-side allowlist: `enabled_tools` / `disabled_tools` in `config.toml`, edited by hand |
| annotations | not required — classifier-based risk assessment in auto mode | load-bearing: absent annotations = max risk = approval prompt every call; `readOnlyHint` enables auto-approval and concurrent execution; `destructive_enabled = false` hard-blocks any tool declaring `destructiveHint` |
| server `instructions` | injected into the system prompt (verified 2026-09-08) | honoured — keep the first 512 chars self-contained (`CODEX-ADAPTER-RESEARCH.md` §2.3) |
| per-tool timeouts | none exposed | `startup_timeout_sec`, `tool_timeout_sec` per server |
| identity | `CLAUDE_CODE_SESSION_ID` reaches the stdio server | no thread id to stdio servers (openai/codex#19937, closed not-planned) — bridged by the HMAC PreToolUse stamp (id:b71a1254) |

**Consequences for this plan:** Step 1 is worth more on Codex than on Claude Code and is what
makes our tools usable there without per-call approval. `alwaysLoad` buys nothing on Codex, so
`docs/CODEX-SETUP.md` should recommend an `enabled_tools` allowlist mirroring `CRITICAL_TOOLS`.
`SERVER_INSTRUCTIONS` (Step 5's escape valve) is read by BOTH hosts — Codex honours it, Claude
Code injects it — so it is the one description-side lever that reaches every caller for free.

---

## Appendix A — per-tool annotations (Step 1's payload)

**30 tools, not 39** — Step 2 deletes `get_node`, `get_trace`, `enrich`; Step 3 deletes the six
interactions tools. Neither set gets annotated. If Step 1 ships first, annotate the survivors and
let the deletions land unannotated.

`openWorldHint: false` for all 30 except `eval`. Per spec `destructiveHint` and `idempotentHint`
are only meaningful when `readOnlyHint` is false — **omit them on read tools** rather than
writing defaults nobody reads. `title` is the host's display label; without it a client shows the
mangled `mcp__plugin_entity_brain__…` name in its approval dialog.

### Reads — `readOnlyHint: true` (14)

| tool | title | note |
|---|---|---|
| `recall` | Recall memories | touches session-scoped access bookkeeping, not node data — still a read |
| `recall_batch` | Recall (multi-query) | |
| `recall_episodes` | Recall episodes | |
| `get_nodes` | Get memories by id | |
| `get_traces` | Get traces by id | |
| `find_node_by_title` | Find memory by title | |
| `filter_nodes` | Filter memories | |
| `query_traces` | Query traces | |
| `count_traces` | Count traces | |
| `query_logs` | Query brain logs | |
| `self_presence` | Live streams | |
| `self_peek` | Peek at a stream | |
| `self_outbox` | Sent-message receipts | |
| `thalamus_list` | Queued brain items | |

### Additive writes — RO false, **D false** (9)

| tool | title | I | why |
|---|---|---|---|
| `remember` | Save a memory | false | two calls = two nodes; a host retry on timeout must not duplicate |
| `remember_batch` | Save memories | false | same |
| `connect` | Link two memories | **true** | documented field-preserving upsert, no auto-strengthen on repeat |
| `connect_batch` | Link memories | **true** | same |
| `set_node_lock` | Lock a memory | **true** | flag set to a value |
| `self_send` | Message a stream | false | repeat = a second message in the inbox |
| `self_inbox` | Drain inbox | false | **consume-once** (`dispatch_self.py:89`) — the second call returns nothing. The tool `is_write` gets most wrong |
| `remind` | File a reminder | false | idempotent only when `dedup_key` is passed; hints are static, so take the conservative value |
| `thalamus_resolve` | Answer a queued item | **true** | resolving an already-resolved item adds nothing |

### Overwriting writes — RO false, **D true** (4)

| tool | title | I | why |
|---|---|---|---|
| `revise` | Revise a memory | false | specified fields are REPLACED; a `content` rewrite loses the old body, and a repeated `content_edits` fails its exact-match anchor |
| `revise_batch` | Revise memories | false | same |
| `revise_edge` | Revise an edge | false | overwrites relation / description / weight in place |
| `brain_batch` | Mixed memory ops | false | the union contains `archive`, `absorb` and `disconnect` — a union takes the max risk of its members |

### Operational — RO false, **D true** (3)

| tool | title | I | openWorld | why |
|---|---|---|---|---|
| `clear_errors` | Clear error log | true | false | deletes rows; clearing twice adds nothing |
| `restart` | Restart the daemon | false | false | tears down a live process — a host should always confirm |
| `eval` | Run arbitrary Python (full access) | false | **true** | `dispatch_ops.py:249` runs Python `eval()` with `brain` in locals and an explicitly weak `safe_builtins` sandbox; the handler's own docstring calls it "effectively arbitrary code execution", safe only because the daemon is loopback + single-user. Expression-only, but `brain.archive_node(...)` is an expression. Genuinely open-world — and the only tool whose `title` should read as a warning in an approval dialog |

### What each field actually buys us

- **`readOnlyHint`** — the whole Codex approval story, and it also makes those calls run
  concurrently (`agents.max_threads`, default 6). 14 tools become promptless there.
- **`destructiveHint`** — currently defaults **true** for all 39 because we declare nothing, so a
  Codex user with `destructive_enabled = false` cannot call `recall`. Declaring it correctly
  turns a blanket block into a 7-tool block.
- **`idempotentHint`** — retry safety. A host that auto-retries a timed-out call will duplicate a
  `remember` and silently no-op a repeated `connect`; the hint is how it knows which.
- **`openWorldHint`** — one honest `true` (`eval`) is worth more than 39 defaults.
- **`title`** — what a human reads in the approval dialog instead of
  `mcp__plugin_entity_brain__thalamus_resolve`.

---

## Appendix B — who actually hits a permission prompt

**The encoders never do.** S1 Scribe and both S2 encoders run inside the daemon and dispatch
through `scales/dispatch.py`, not through a host — no approval layer exists on that path.
Annotations therefore only change the experience of an INTERACTIVE session (Anchor in Claude
Code, or the operator in Codex). This is the reason Step 1 is cheap: it cannot regress encoding.

**Claude Code, this repo:** nothing prompts. `.claude/settings.json` carries a server-wide
prefix allow (`"mcp__plugin_entity_brain"`), which covers all 39. Claude Code's permission rules
match on tool NAME patterns and ignore annotations entirely, so Step 1 changes nothing here — a
fresh install without that rule prompts per tool on first use and can remember the answer.

**Codex: the prompt is once per tool, not once per call.** Measured in the rollout
(`docs/CODEX-ADAPTER-RESEARCH.md` §9): Codex asks on a tool's FIRST use — reads included — and
persists the answer as `[plugins."<id>".mcp_servers.brain.tools.<tool>] approval_mode = "approve"`.
A hook returning `permissionDecision: allow` does not suppress it, and the model never sees the
prompt (it reports "no approval"). So no annotation makes a tool prompt every time.

What the modes do, and where annotations land in them
(`default_tools_approval_mode` per server, `tools.<tool>.approval_mode` per tool):

| mode | behaviour |
|---|---|
| `auto` | no ask |
| `writes` | asks for **every tool not annotated `readOnlyHint: true`** — the mode where Step 1 pays off directly: 14 tools drop out of the ask |
| `prompt` | asks per tool on first use, remembers |
| `approve` | pre-approved — what our own onboarding writes server-wide (405666a), so the user answers ONE form instead of 39 prompts |

So Step 1's Codex value is narrower than "fewer prompts" and still real: `readOnlyHint` removes
14 tools from the `writes`-mode ask AND lets them execute concurrently, and a correct
`destructiveHint` means a user running `destructive_enabled = false` can still call 23 of the 30
tools instead of none. `brain_batch` being destructive costs one first-use confirmation, not a
confirmation per encode — there is no reason to soften the hint.

**`eval`'s real consumer, so Step 4 doesn't break it:** `eval/oracle_audit/backfill_absorbed_into.py`
uses the daemon `eval` COMMAND over TCP. Step 4 gates the MCP TOOL only — leave the
`COMMAND_TABLE` entry alone. (The `eval/` directory and the `eval` tool share a name and nothing
else; the eval platform does not call the tool.)
