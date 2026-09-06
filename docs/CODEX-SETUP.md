# Codex setup

After installation and runtime preparation, ask the assistant to finish Entity
setup. It calls `setup` with `action: status`. If enabled hook definitions need
approval, `action: review` asks for confirmation in the host. Accepting with
“Open Codex hook review” selected launches Codex's native review UI. The user
makes the trust decision in Codex and then asks the assistant to check again.

The native startup review is a terminal UI. No shell commands need to be typed
for new or modified enabled hooks. Codex may first ask whether to trust its
working directory. Other installed plugins may appear too; users should review
the definitions they intend to run. Entity never selects approval options or
writes hook trust state.

`definitions_trusted` means Codex reports this installation's definitions trusted
and enabled. It does not prove global feature flags, recording, recall, daemon
health, or all hooks work. `caller_identity_verified` describes the current
call's identity evidence; `runtime_verified` remains false. Confirm automatic
recall and capture in the host before declaring the installation fully ready.

## Fallbacks and scope

- Disabled hooks require manual `/hooks` settings review; the automatic startup
  review is for new or changed enabled definitions.
- Missing or disabled plugins are reported as not found. A successful MCP
  connection alone is never treated as hook approval.
- Missing form capability or an older MCP protocol produces instructions without
  launching anything. Unsupported Codex inspection produces an explicit error.
- macOS launches Terminal with a `.command` document. Other systems get
  a manual-review message. The target configuration directory is preserved.
- Direct marketplace runtime bootstrap can outlast the first MCP startup window.
  The checkout installer prebuilds the cache runtime; a direct marketplace cold
  install may still need a reconnect. This change is not a signed installer.
