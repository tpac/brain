# Codex setup

After installation and runtime preparation, ask **“Finish Entity setup.”**
The assistant checks two things: permission to use Entity tools and trust for
automatic-memory hooks. If needed, one in-app confirmation explains both.

Selecting **“Allow Entity tools and continue setup”** and accepting saves an
Entity-wide tool approval. This covers current and future tools from Entity's
brain server, including reading, changing and deleting memories and messaging
other Entity sessions. Existing restrictions on individual tools stay in place.
It applies until you change Entity's tool approval setting in Codex.

When hooks need trust, the same acceptance opens Codex's own hook review. You
make the hook-trust decision there. The popup does not grant hook trust. Return
to the chat and ask to check setup again. Saved tool permission may require the
existing Codex connection to reload; if prompts continue, quit and reopen the
app and check again.

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
- Missing or disabled plugins are reported as not found; a disabled MCP server
  must be enabled in Codex before setup can grant its tool permission. A successful MCP
  connection alone is never treated as hook approval.
- Missing form capability or an older MCP protocol produces instructions without
  launching anything. Unsupported Codex inspection produces an explicit error.
- macOS launches Terminal with a `.command` document. Other systems get
  a manual-review message. The target configuration directory is preserved.
- Direct marketplace runtime bootstrap can outlast the first MCP startup window.
  The checkout installer prebuilds the cache runtime; a direct marketplace cold
  install may still need a reconnect. This change is not a signed installer.
