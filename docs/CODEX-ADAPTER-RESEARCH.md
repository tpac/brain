# Codex Adapter — Research

## § State — 2026-09-05

S1 (host-portable hook stdout: `hook_common.emit_hook_output`, silence = no
opinion, block on Stop only) and S2 (`.codex-plugin/plugin.json`,
`hooks/hooks.codex.json`, newest-install MCP launcher, gates) are on main at
d5c1793 after two code reviews. Ruling that shaped them: the brain informs, it
never gates (the safety hook's block became a warning). Not yet: S4 signed
identity stamp (§5.1), the live empirical pass (§6) on Tom's ChatGPT desktop
app, and the daemon restart + `./redeploy.sh` that make the merged code live in
Claude Code. Handoff node in the brain: search "HANDOFF — Codex adapter".

Can Anchor run inside ChatGPT's Codex mode as a second host, with hooks? What is
there, what is missing, and how each gap closes. Researched 2026-09-05 against
the official docs (learn.chatgpt.com, developers.openai.com/plugins) and the
open-source Codex core (openai/codex @ 459a79e, 2026-09-05). No Codex install
was available on this machine, so every "untested" item below still needs the
empirical pass in §6.

Provenance markers: **[doc]** official documentation · **[src]** read in the
Codex source · **[ours]** read in this repo · **[measured]** run here ·
**[inferred]** follows from the above but not observed · **[untested]** needs a
live Codex.

---

## 1. Summary

**Codex is a hook host.** Since 2026-07-09 the Codex desktop app is the *Codex
mode* of the unified ChatGPT desktop app (Chat / Work / Codex). Codex mode, the
Codex CLI and the IDE extension share one config (`~/.codex/config.toml`) and
one hook engine whose events, stdin JSON, output JSON and even environment
variable names were built for Claude Code compatibility. **[doc][src]**

**Our plugin loads in Codex as-is — in format.** Codex discovers
`.claude-plugin/plugin.json` as a "Legacy" manifest, `.claude-plugin/
marketplace.json` as a "legacy-compatible" marketplace, `hooks/hooks.json`,
`.mcp.json` and `skills/*/SKILL.md` at their default paths, and exports
`CLAUDE_PLUGIN_ROOT` / `CLAUDE_PLUGIN_DATA` to plugin hook commands. **[src]**

**Three things break, one of them hard:**

| Gap | Severity | Bridge |
|---|---|---|
| **G1** MCP proxy identity — Codex passes no thread id to stdio MCP servers; every brain tool call arrives anonymous | Hard, but solvable | PreToolUse hook on `mcp__brain__.*` rewrites the tool arguments to carry a hook-signed `_caller_session` (§5.1) |
| **G3** `.mcp.json` command path — Legacy plugins get no `${CLAUDE_PLUGIN_ROOT}` expansion; the command is spawned literally | Blocking until fixed, small | Launcher resolution via the deterministic plugin-cache path or the hook-persisted `resolved.env` (§5.3) |
| **G2** Hook output shapes — our `{"decision":"approve"}` outputs fail Codex's strict schemas and register as hook *failures* on every turn | Noisy, small | Silent approve, suggestions via `additionalContext`, Stop emits JSON only (§5.2); CC-compatible |

Everything else is adaptation: a Codex hooks.json variant, a
`.codex-plugin/plugin.json` sibling manifest, `additionalContextLimit` on the
two injecting hooks, SessionEnd under 3 s, model-neutral identity wording,
a Codex install/redeploy path, and gate updates.

**Recommendation.** Build the Codex adapter before the ChatGPT Chat-mode tunnel
path (`f5b3fe55`): it carries the whole S0 layer, it exercises the second-adapter
architecture D-11 was designed for, and the identity mechanism it needs (§5.1)
is the same one the Chat-mode adapter would reuse. Estimated 2–3 days of build
plus a half-day empirical pass once a Codex is installed.

---

## 2. What Codex provides

### 2.1 Surfaces **[doc]**

| Surface | Hooks | Local stdio MCP | Plugins | Notes |
|---|---|---|---|---|
| ChatGPT desktop app, **Codex mode** | yes | yes | yes (Plugins tab, Personal marketplace) | every plan incl. Free; worktree "local environments" |
| Codex CLI | yes | yes | yes (`/plugins`) | `codex plugin marketplace add`, `/hooks` trust UI |
| IDE extension | yes | yes | **no** | shares config; can't install plugins |
| Codex cloud | — | — | — | isolated containers; no local daemon → out of scope |
| ChatGPT **Chat / Work** (web, desktop, mobile) | **no** | **no** | hosted plugins only | remote MCP via Secure MCP Tunnel — see `f5b3fe55` |

### 2.2 Plugin formats and discovery **[src]**

Manifest lookup (`utils/plugins/src/plugin_namespace.rs`,
`exec-server-protocol/src/protocol.rs`):

1. `plugin.json` at the plugin root **with** `"$schema":
   "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"` → format
   **AgentPlugin** (the cross-vendor Agent Plugins spec).
2. Otherwise, in order: `.codex-plugin/plugin.json`, `.claude-plugin/plugin.json`,
   `.cursor-plugin/plugin.json` → format **Legacy**. Ours is Legacy.

| Concern | Legacy (ours) | AgentPlugin |
|---|---|---|
| Hooks | loaded from `hooks/hooks.json` or manifest `hooks` (path, paths, inline) | **not loaded** (only allow-listed bundled cleanup hooks) |
| Skills | `skills/` recursive discovery | inventory-based |
| MCP config | `.mcp.json` — direct map or `mcp_servers`/`mcpServers` wrapper; server objects deserialize straight into `McpServerConfig` (so `startup_timeout_sec`, `tool_timeout_sec`, `env`, `env_vars`, `cwd` are all valid keys) | `./` commands resolved to plugin root; `${PLUGIN_ROOT}`/`${PLUGIN_DATA}` expanded in args/env/cwd; `PLUGIN_ROOT`/`PLUGIN_DATA` injected into the server env |
| **Variable expansion in `command`** | **none** — the string is spawned as-is (bare name → `PATH`; otherwise a path). No `${VAR}`, no tilde | see above |
| Tool budget | none | 8,000 bytes per tool spec, 64,000 bytes total; tools beyond are `Hidden` |
| Data dir (`PLUGIN_DATA`) | `~/.codex/plugins/data/<plugin>-<marketplace>/` | hashed dir |

The hooks-vs-expansion split is the fork that matters: **the format that gives
us hooks is the one without command expansion.** §5.3 bridges it.

Marketplaces **[doc][src]**: `$REPO_ROOT/.agents/plugins/marketplace.json`,
`$REPO_ROOT/.claude-plugin/marketplace.json` ("legacy-compatible"),
`~/.agents/plugins/marketplace.json` (personal). Add with
`codex plugin marketplace add <path | owner/repo | git-url>`. Install copies
the plugin to `~/.codex/plugins/cache/$MARKETPLACE/$PLUGIN/$VERSION/`
(`$VERSION` = `local` for local sources) and **loads from the cache copy**.
After changing a local plugin: update the directory the marketplace entry
points to, restart the app (or start a new CLI session). Enable state and
per-plugin MCP policy live in `~/.codex/config.toml` under
`plugins."<name>@<marketplace>".mcp_servers.<server>`.

Plugin name rule: lowercase kebab-case — `entity` qualifies. `userConfig` and
other unknown manifest keys are not part of the Legacy schema **[src]**; whether
they are ignored or rejected is **[untested]**.

### 2.3 Hooks **[doc][src]**

**Events.** SessionStart (source: startup | resume | clear | compact),
SessionEnd, UserPromptSubmit, PreToolUse, PostToolUse, PermissionRequest,
PreCompact, PostCompact, SubagentStart, SubagentStop, Stop, Interrupt.
Not present: WorktreeCreate, WorktreeRemove, ConfigChange, StopFailure.

**stdin (all events).** `session_id` (thread id, UUIDv7 string; subagent hooks
carry the parent's), `cwd`, `hook_event_name`, `transcript_path` (nullable;
format explicitly unstable), `model`, `permission_mode`. Turn-scoped events add
`turn_id`. UserPromptSubmit adds `prompt`; Stop adds `last_assistant_message`,
`stop_hook_active`; PreToolUse/PostToolUse add `tool_name`, `tool_input`,
`tool_use_id` (+ `tool_response`).

**Execution** (`hooks/src/engine/command_runner.rs`): the command string runs
through `$SHELL -lc "<command>"` (login shell from the session snapshot,
`/bin/sh` fallback), cwd = session cwd, **environment = the session's env
snapshot replayed** (not the live process env) plus per-handler env. Plugin
hooks additionally get `PLUGIN_ROOT`, `PLUGIN_DATA`, `CLAUDE_PLUGIN_ROOT`,
`CLAUDE_PLUGIN_DATA` (`hooks/src/engine/discovery.rs`). Own process group,
killed as a group on timeout. **Spawned directly by the Codex process, not
inside the sandbox** — the daemon on `127.0.0.1:47203` is reachable.

**Timeouts.** Default 600 s; per-handler `timeout` (seconds). SessionEnd and
Interrupt default to 1 s, **maximum 3 s**. Matching handlers run concurrently;
`async: true` runs a handler in the background (8 concurrent per session,
cannot block/rewrite).

**Output contract** (`hooks/src/engine/output_parser.rs`, generated schemas,
`additionalProperties: false`):

| Event | Accepted | Effect | Rejected / invalid |
|---|---|---|---|
| SessionStart, UserPromptSubmit | plain text; `{"hookSpecificOutput":{"hookEventName":…,"additionalContext":…}}`; `{"decision":"block","reason":…}` | text → **developer** message; block → prompt rejected | `decision:"approve"` — not in the enum (`block` only); stdout that *looks like* JSON but fails schema → hook **Failed** |
| PreToolUse | `hookSpecificOutput.permissionDecision` `deny` (+reason) / `allow` (+`updatedInput`); `additionalContext`; legacy `{"decision":"block","reason"}`; exit 2 + stderr | deny/block stops the call; `updatedInput` **replaces the MCP arguments object**; context → developer message | `decision:"approve"`, `ask`, `continue:false` → hook Failed (call continues); plain text ignored |
| PostToolUse | `decision:"block"` + reason; `continue:false`; `additionalContext` | replaces the tool result with the reason | plain text ignored |
| Stop | `{"decision":"block","reason":…}`; `continue:false`; exit 2 + stderr | block → **auto-continue with `reason` as the next user prompt** | **plain text is invalid** for Stop |
| SessionEnd | advisory only | — | — |

Exit 0 with empty stdout is success everywhere.

**Injection semantics.** `additionalContext` (and SessionStart/UserPromptSubmit
plain text) is recorded as a `developer`-role conversation item
(`core/src/context/hook_additional_context.rs`, `record_additional_contexts`).
A SessionStart hook matching `compact` delivers into the immediate continuation.

**Spill limit.** Model-visible hook output above ~2,500 tokens is written to
`<tmp>/hook_outputs/<session_id>/<uuid>.txt` and the model receives a head/tail
preview. Per-handler `additionalContextLimit` (0 = pass everything through).

**Matchers.** SessionStart → `source`; PreToolUse/PostToolUse/PermissionRequest
→ tool name: `Bash` (shell and unified exec), `apply_patch` (aliases `Edit`,
`Write`; stdin still says `apply_patch`), `mcp__<server>__<tool>`, `spawn_agent`
(alias `Agent`), other local function tools (`update_plan`, …). Hosted tools
(`WebSearch`) never fire. UserPromptSubmit, Stop, Interrupt ignore `matcher`.
`hooks/src/…/hook_names.rs` documents the aliases as deliberate Claude Code
compatibility.

**Trust.** Non-managed hooks (user, project, plugin) are skipped until the user
reviews and trusts the exact definition (hash-pinned; a changed hook needs
re-trust). CLI: `/hooks`. Desktop-app trust UI: app-server exposes `hooks/list`
and a trust status; the UI itself is **[untested]**. Feature flag
`features.hooks`: Stable, enabled by default (`features/src/lib.rs`).

### 2.4 MCP **[doc][src]**

**Spawn environment** (`rmcp-client/src/utils.rs`): the child env is cleared,
then rebuilt from an allowlist — `HOME LOGNAME PATH SHELL USER
__CF_USER_TEXT_ENCODING LANG LC_ALL TERM TMPDIR TZ` — plus the server's `env`
map and `env_vars` names, plus CA-bundle variables. **No session or thread
identity.** `CODEX_THREAD_ID` exists only for the model's shell tool executions
(`core/src/exec_env.rs`, `unified_exec`) — not hooks, not MCP servers. The
request to expose it to stdio servers (openai/codex#19937, 2026-04-28) is
closed as not planned.

**Lifecycle.** An `McpRuntime` is created per `Session` (= per thread) and its
connection set is built per runtime publish → **one stdio process per thread**,
started eagerly at session start (issue #21984 confirms). In the desktop app all
threads live in one app-server process, so every proxy shares the same parent
**[inferred]**. Prewarm worker per session. Defaults: `startup_timeout_sec` 10,
`tool_timeout_sec` 60, `mcp_optional_startup_grace_ms` 1000 (the initial tool
catalog waits only 1 s for optional servers; `required = true` waits the full
startup timeout).

**Tools.** Named `mcp__<server>__<tool>` (server = the `.mcp.json` key, so ours
become `mcp__brain__recall` — not CC's `mcp__plugin_entity_brain__recall`).
`enabled_tools`/`disabled_tools`, `tools.<tool>.output_token_limit`,
`default_tools_approval_mode` = `auto | prompt | writes | approve` — `writes`
prompts for every tool not annotated read-only. Codex **honors the MCP
`instructions` field** (keep the first 512 chars self-contained) — the field
Claude Code ignores (`3bab3268`). No tool-count budget for Legacy plugins; the
context cost of our 39-tool schema (75 k chars ≈ 19 k tokens **[measured]**) is
paid on every turn unless trimmed.

**cwd** for a Legacy stdio server = the thread's cwd unless `cwd` is set
(taken literally). Own process group.

### 2.5 Sandbox **[doc][src]**

Seatbelt (macOS) applies to commands the *model* spawns; default
`workspace-write` has **network off**. Hooks and MCP servers are spawned by
Codex itself outside that boundary (§2.3, §2.4). Consequence: the daemon is
reachable from hooks and from the proxy, but a brain **skill** that shells out
via the model's Bash tool to reach `localhost:47203` or the dashboard will hit
the network policy and need approval or `sandbox_workspace_write.network_access
= true` **[inferred][untested]**.

---

## 3. What we ship today, per component **[ours]**

| Component | Host dependency | Under Codex |
|---|---|---|
| `hooks/hooks.json` — 11 events | CC event names; matchers `Edit\|Write`, `Bash`, `mcp__.*`, and CC tool names `Read\|Glob\|Grep\|Agent\|WebSearch\|WebFetch\|NotebookEdit` | 7 events port 1:1; WorktreeCreate/Remove, ConfigChange, StopFailure don't exist; `Edit\|Write` and `Agent` work via aliases; `Read/Glob/Grep/WebSearch/WebFetch/NotebookEdit` never match (Codex has `apply_patch`, `Bash`, hosted search) |
| `hooks/scripts/*.sh` shims | `${CLAUDE_PLUGIN_ROOT}` in command strings; `resolve-brain-db.sh` resolves the plugin root from its own path | commands expand (login shell, alias env set); path-based resolution works from the plugin cache |
| `hook_common.get_hook_input` | reads `HOOK_INPUT`; backfills `session_id` from `CLAUDE_CODE_SESSION_ID` | every Codex event carries `session_id` on stdin — backfill never needed |
| `resolve-brain-db.sh`, `api-key-env.sh` | `CLAUDE_PLUGIN_DATA` (adoption scan, legacy default), `CLAUDE_PLUGIN_OPTION_*` (userConfig key / brain path) | `CLAUDE_PLUGIN_DATA` alias is set; `OPTION_*` never set → key and path come from `~/.config/brain/env` (already the primary path) |
| `pre_response_recall.py` | prints `{"decision":"approve"}` on slash/bang/short/register-only; else `hookSpecificOutput.additionalContext` | approve JSON **fails the Codex schema → hook Failed** on those turns (G2); context path is correct |
| `pre_bash_safety.py`, `pre_edit_suggest.py` (+ daemon `hook_pre_bash_safety`, `hook_pre_edit`) | `{"decision":"approve"\|"block","reason"}` | `block` accepted; `approve` (with or without reason) → **hook Failed** (G2); the suggestion `reason` is lost |
| `post_response_track.py` (Stop) | `{"decision":"block","reason"}` for self-message delivery; otherwise prints daemon `output` text | block works (auto-continue); any plain-text `output` is **invalid on Stop** (G2); transcript-JSONL fallback is CC-specific but unnecessary — the daemon writes `user_message` at prompt time and only needs `last_assistant_message` |
| `post_tool_trace.py` | `tool_name`/`tool_input`; summary builder knows CC tool names | `apply_patch` input is `{command: <patch>}` → falls to the generic branch; MCP names become `mcp__brain__*` |
| `session_end.py` | CC timeout 10 s; daemon call timeout 30 s | Codex caps SessionEnd at **3 s** (G10) |
| `config_change_host.py`, worktree hooks, `stop_failure_log.py` | CC-only events | never fire; worktree identity is derived from cwd at boot anyway |
| `servers/brain_mcp.py` `_stamp_caller_session` | `CLAUDE_CODE_SESSION_ID` env; **scrubs** any inbound `_caller_session` when the env var is absent | env var absent → every call anonymous (G1) |
| `hooks/scripts/mcp-launch.sh`, `.mcp.json` | command `${CLAUDE_PLUGIN_ROOT}/hooks/scripts/mcp-launch.sh`; cold-install wait up to 25 s | **no expansion → spawn fails** (G3); 25 s > Codex's 10 s default startup timeout |
| `.claude-plugin/plugin.json`, `marketplace.json` | CC manifest incl. `userConfig` | loads as Legacy; marketplace is legacy-compatible; `userConfig` ignored/unknown (G4) |
| `skills/*/SKILL.md` | frontmatter `name`+`description`; `/watch` depends on CC's Monitor tool; `/self-salvage` on `spawn_task` | skills load; `/watch` live-listener and `spawn_task` don't exist in Codex (G9) |
| Identity text (`skills/brain/SKILL.md`, `brain_voice.py`, `brain_assembly.py`) | "This isn't Claude with memory bolted on"; docstrings say Claude | model-specific wording under a GPT host (G6) |
| Daemon, dashboard, DBs, launchd | host-neutral (D-11, D-13) | unchanged |
| `build-plugin.sh`, `redeploy.sh` | CC plugin zip; `~/.claude/plugins/marketplaces/…` target | Codex needs a marketplace entry + cache refresh (G7) |
| `tests/test_deploy_contract.py` | version lockstep, adapter-name containment, host-neutrality, `.claude-plugin` allowlists | a second manifest must join the lockstep and allowlists (G8) |
| `runtime-state.sh` / `ensure-runtime.sh` | venv bootstrapped into `$PLUGIN_ROOT/venv` | the plugin root is a **cache copy** that a re-install may replace (G7) |

---

## 4. Identity model under Codex

| Where | Identity available | Source |
|---|---|---|
| Hook stdin | `session_id` = thread UUIDv7 (root thread for forks/subagents) | **[doc][src]** |
| Hook env | `PLUGIN_ROOT`, `PLUGIN_DATA`, CC aliases; **no** session/thread var | **[src]** `discovery.rs` |
| Model's shell tool | `CODEX_THREAD_ID` | **[src]** `exec_env.rs` |
| stdio MCP server env | nothing beyond the allowlist | **[src]** `utils.rs`; #19937 closed |
| MCP tool call arguments | whatever the model put there — plus whatever a PreToolUse hook rewrites in | **[doc][src]** `updatedInput` replaces the arguments object |

The daemon side already fits: `caller_session(args)` reads `session_id` or
`_caller_session`; `hook_recall` calls `get_or_create_session`; a UUID string
works everywhere `session_id[:8]` is used. Only the **stamping** is CC-bound.

---

## 5. Gaps and bridges

### 5.1 G1 — MCP proxy identity (the hard one)

**Mechanism.** A PreToolUse handler matching `mcp__brain__.*` reads
`session_id` and `tool_input` from stdin and returns

```json
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow",
  "updatedInput":{ …tool_input…, "_caller_session":"<session_id>",
                   "_caller_sig":"<hmac>" }}}
```

Codex replaces the MCP arguments with `updatedInput` **[doc][src]**, so the
proxy receives the identity on every call without the model's involvement.

**Trust.** Today the proxy scrubs inbound `_caller_session` because the model
can write arbitrary keys. Under Codex the hook — not the model — is the trusted
stamper, and the two must be distinguishable. Sign the stamp: HMAC-SHA256 of
`session_id` with a per-install secret (`~/.config/brain/hook-secret`, mode
600, created by the boot hook). The model sees tool arguments but never the
secret, so it cannot forge a valid pair. Proxy rule becomes: env var present →
env wins (CC); else a **valid signed** `_caller_session` is accepted; else
scrub. Add-only, CC unchanged.

**Cost.** One short Python process per brain tool call (no daemon round-trip:
read stdin, sign, print) — the same order as the PostToolUse trace hook that
already runs per tool call. **[untested]** whether `updatedInput` fires for MCP
tools before approval prompts, and end-to-end latency.

**Rejected alternatives.** Parent-PID handshake between boot hook and proxy —
collides in the desktop app (one app-server parent for all threads)
**[inferred]**. Model-supplied `session_id` — #19937 itself calls it
unreliable. Last-booted-session fallback — the last-writer-wins bug removed
2026-05-17.

### 5.2 G2 — Hook output hygiene (host-neutral fix)

Change the shared scripts and daemon handlers so that: approve = **no
stdout**; a suggestion or warning that should reach the model =
`hookSpecificOutput.additionalContext` (PreToolUse supports it); block stays
`{"decision":"block","reason":…}`; Stop prints **JSON or nothing**. Claude Code
treats empty stdout as approve and accepts `additionalContext` on PreToolUse,
so this is a single code path for both hosts. Touches: `pre_response_recall.py`
(`APPROVE`), `pre_bash_safety.py`, `pre_edit_suggest.py`, `post_response_track.py`,
`daemon_hooks.hook_pre_edit` / `hook_pre_bash_safety` return shapes, and
`hooks/HOOKS.md`.

### 5.3 G3 — `.mcp.json` command resolution

Legacy plugins get no variable expansion and `PLUGIN_ROOT` is not in the MCP
env. Options:

| Option | How | Trade-off |
|---|---|---|
| **a. Cache-path launcher (shipped)** | `bash -c` (no login shell — Codex's MCP env already carries PATH and HOME, and a profile that prints would corrupt the stdio stream) walks `${CODEX_HOME:-$HOME/.codex}/plugins/cache/*/entity/*/` and execs the **most recently modified** install's `mcp-launch.sh` | Codex keeps several version dirs side by side and activates the newest; a glob's first match is lexicographic (`0.10.0` before `0.9.0`, any number before `local`), so pick by mtime — the last install or re-copy wins on every marketplace |
| b. Hook-persisted pointer | boot hook writes `BRAIN_PLUGIN_ROOT` into `~/.config/brain/resolved.env` (it already persists resolution state); the launcher reads it | Races the eager MCP start on the very first session; correct from the second |
| c. AgentPlugin format | root `plugin.json` + `./hooks/scripts/mcp-launch.sh` gets proper expansion | **No hooks** in that format today — non-starter |
| d. Wait for Codex to expand `${PLUGIN_ROOT}` for Legacy `.mcp.json` | — | Not on their roadmap as far as the source shows |

Also in the Codex server object: `"startup_timeout_sec": 40` (our cold-install
wait is 25 s against a 10 s default), `"env_vars": ["XDG_DATA_HOME",
"XDG_CONFIG_HOME", "BRAIN_DB_DIR"]` so a customised brain location survives the
env allowlist, and the `instructions` text (§2.4) — a lever Claude Code lacks.
Keep CC's `.mcp.json` untouched; put the Codex server object inline under
`mcpServers` in `.codex-plugin/plugin.json` (§5.4).

### 5.4 G4 — Manifest and marketplace

Discovery reads `.codex-plugin/plugin.json` **before** `.claude-plugin/plugin.json`
(same Legacy format), so a Codex sibling manifest is not an overlay — it is
*the* manifest when present and must be complete: `name`, `version`,
`description`, `skills`, `mcpServers` (inline object, §5.3), `hooks`
(`./hooks/hooks.codex.json`, §5.5), `interface` if we want the Plugins-tab
card. Keep `version` in lockstep with `.claude-plugin/plugin.json` (extend the
existing deploy-contract test). The existing `.claude-plugin/marketplace.json`
is already a valid Codex marketplace; a `~/.agents/plugins/marketplace.json`
entry pointing at the repo (or the built package dir) is the dev install.

### 5.5 G5 — Codex hooks.json variant

`hooks/hooks.codex.json`: the seven shared events; drop the four CC-only ones;
PreToolUse matchers `Bash`, `Edit|Write`, plus the new `mcp__brain__.*`
identity handler; PostToolUse matchers `Bash`, `Edit|Write`, `mcp__.*`;
`additionalContextLimit: 0` (or a deliberate cap) on SessionStart and
UserPromptSubmit — a boot block plus standing items is ~1.5–2 k tokens and a
5-memory recall ~2.3 k tokens **[measured this session]**, right at the 2,500
spill threshold; `timeout` 3 on SessionEnd; `statusMessage` strings. Whether
Codex rejects a hooks file containing unknown event names is **[untested]** —
the variant avoids the question.

### 5.6 G6 — Identity text

`skills/brain/SKILL.md` ("This isn't Claude with memory bolted on") and the
`brain_voice.py` / `brain_assembly.py` docstrings name Claude. Under a GPT host
the stance text should be model-neutral, or the daemon takes a `host` on
`reset_session` and words it per host. Content decision — Tom's.

### 5.7 G7 — Install, redeploy, runtime

Dev loop: marketplace entry → `codex plugin marketplace add` (or the desktop
app's Personal tab) → install → **trust hooks** (`/hooks`) → new session.
Redeploy: refresh the directory the marketplace points at, restart. The venv
bootstrap writes into `$PLUGIN_ROOT/venv`, i.e. into the cache copy; whether a
refresh preserves it is **[untested]** — if not, every update pays the 60–90 s
cold bootstrap. Relocating the runtime to `PLUGIN_DATA` (`CLAUDE_PLUGIN_DATA`
under CC) fixes both hosts and is already noted as deferred in
`runtime-state.sh`.

### 5.8 G8 — Gates

`test_deploy_contract`: add `.codex-plugin/plugin.json` to the manifest
allowlists and the version-lockstep check; host-neutrality stays as is
(`servers/` and `dashboard/` must not learn `~/.codex/plugins` any more than
`~/.claude/plugins`). New: a Codex hooks.json validity test (event subset,
matchers, handler fields) and an output-shape test for the shared scripts
(no `decision:"approve"` anywhere).

### 5.9 G9 — Self-channel and skills

Stop `decision: block` → auto-continue is supported, so self-message delivery
at Stop works. `/watch`'s live listener needs CC's Monitor tool; `/self-salvage`
uses `spawn_task`. Both degrade to the Stop-hook drain under Codex. Model-run
shell reaching the daemon or dashboard may need a network approval (§2.5).

### 5.10 G10 — SessionEnd

Hook timeout must be ≤ 3; the daemon call inside should use ~2 s. The work
(`discard_session_context` + `save`) is fast; a kill mid-call is harmless.

### 5.11 G11 — PostToolUse trace

Add an `apply_patch` branch to `_build_summary` (input is `{command: <patch
text>}`); everything else falls through to the generic branch correctly.

---

## 6. Empirical pass (needs a Codex install)

Neither `codex` CLI nor a Codex-mode session exists on this machine yet
(`ChatGPT.app` is installed; `~/.codex` absent) **[measured]**.

1. Legacy plugin loads from a `.claude-plugin/marketplace.json` local path; hooks show in `/hooks` and trust sticks across sessions.
2. SessionStart: stdout vs `additionalContext` both land as developer context; spill behaviour with and without `additionalContextLimit: 0`.
3. UserPromptSubmit: per-turn context arrives before the model request; no hook-failure warnings after §5.2.
4. `.mcp.json`: which of §5.3 a/b spawns; startup within `startup_timeout_sec`; tools appear as `mcp__brain__*`; `instructions` visible in behaviour.
5. Identity hook: fires for `mcp__brain__*`; `updatedInput` reaches the proxy; signature verifies; per-call latency.
6. Stop block → continuation prompt carries a pending self-message.
7. Plugin refresh: does the cache copy keep `venv/`?
8. Desktop app: hook trust UI; threads created in worktrees resolve cwd/branch/project correctly at boot.
9. Approval prompts for brain writes under `default_tools_approval_mode = "writes"`; effect of `readOnlyHint` annotations.
10. Chat mode untouched by any of the above.

---

## 7. Build order

Each step runs cold in its own session.

| # | Step | Depends on | Size |
|---|---|---|---|
| S0 | Install Codex CLI (or use Codex mode); personal marketplace → repo | — | Tom |
| S1 | Output hygiene (§5.2) in shared scripts + daemon handlers + `HOOKS.md`; tests | — | ½ day |
| S2 | `hooks/hooks.codex.json` (§5.5) + `.codex-plugin/plugin.json` (§5.4) + deploy-contract lockstep/allowlists (§5.8) | S1 | ½ day |
| S3 | Codex `mcpServers` object with launcher resolution, `startup_timeout_sec`, `env_vars`, `instructions` (§5.3) | S2 | ½ day + E4 |
| S4 | Signed identity stamp: hook handler + proxy acceptance rule + secret provisioning (§5.1); tests | S2 | 1 day + E5 |
| S5 | Empirical pass E1–E10; encode findings; fix what breaks | S3, S4 | ½ day |
| S6 | Docs: README Codex install; `DISTRIBUTION-READINESS` decision for the second adapter; `HOOKS.md` | S5 | ¼ day |
| S7 | Runtime relocation to `PLUGIN_DATA` (§5.7) — both hosts | independent | 1 day, optional |

---

## 8. Decisions for Tom

1. **Ship as Legacy dual-manifest now** (hooks, no expansion, §5.3 workaround) vs wait for AgentPlugin-format hooks (expansion, tool budget, no date). Recommendation: Legacy now.
2. **Identity mechanism**: signed hook stamp (recommended) vs model-supplied `session_id`.
3. **Identity wording** under a GPT host (§5.6).
4. **Provenance and lock rights** for Codex-originated writes — `encoding_source` today assumes `anchor*` is Anchor; same question as for ChatGPT Chat mode (`f5b3fe55`).
5. **Runtime relocation to `PLUGIN_DATA`** now (S7) or after the empirical pass shows whether the cache copy keeps the venv.
6. **Codex install** on this Mac for the empirical pass.

---

## 9. Sources

Docs (fetched 2026-09-05; `.md` variants): learn.chatgpt.com/docs/hooks,
/docs/extend/mcp, /docs/plugins, /docs/build-plugins, /docs/sandboxing,
/docs/agent-approvals-security, /docs/config-file/environment-variables,
/docs/config-file/config-reference, /docs/app-server, /docs/build-skills;
developers.openai.com/plugins/build/plugins.

Source (openai/codex @ 459a79e): `codex-rs/hooks/src/engine/command_runner.rs`
(spawn, env replay, shell, process group), `…/engine/output_parser.rs` and
`hooks/schema/generated/*.json` (accepted shapes), `…/engine/discovery.rs`
(plugin hook env, trust), `…/events/user_prompt_submit.rs` (plain-text vs
invalid-JSON branches), `core/src/tools/hook_names.rs` (matcher aliases),
`core/src/context/hook_additional_context.rs` (developer role),
`core/src/exec_env.rs` + `protocol/src/shell_environment.rs` (CODEX_THREAD_ID),
`rmcp-client/src/utils.rs` (MCP env allowlist), `rmcp-client/src/
stdio_server_launcher.rs` (cwd default), `codex-mcp/src/plugin_config.rs` and
`agent_plugin_config.rs` (Legacy vs AgentPlugin MCP parsing and expansion),
`core-plugins/src/{manifest,loader,store,marketplace}.rs` (formats, defaults,
cache/data roots, hooks skipped for AgentPlugin), `utils/plugins/src/
plugin_namespace.rs` + `exec-server-protocol/src/protocol.rs` (manifest
discovery order), `core/src/mcp_tool_exposure.rs` (agent-plugin tool budget),
`core/src/session/{mcp,mcp_prewarm}.rs` + `codex-mcp/src/runtime.rs` (per-thread
runtime), `features/src/lib.rs` (hooks default), `protocol/src/thread_id.rs`
(UUIDv7), `external-agent-migration/src/hooks_cla.rs` (Codex's own Claude-hooks
importer: command hooks only, drops async/prompt/agent handlers).

Issues: openai/codex#19937 (thread id for stdio MCP — closed, not planned),
#21984 (MCP servers start per session), #8923 (session id exposure).

Brain: `7c786811` (Codex feasibility), `f5b3fe55` (ChatGPT Chat-mode path),
`3c9c9012` (D-11 two-identity model), `34278e20` (CC HTTP identity question),
`3bab3268` (MCP `instructions` dead in CC).
