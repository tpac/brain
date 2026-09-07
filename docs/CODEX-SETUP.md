# Codex setup

After installation, ask **“Finish Entity setup.”** On first use, Entity prepares
its private Python runtime and memory dependencies. Allow a few minutes for
this download; setup becomes available on the same connection when it finishes.
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

## If the confirmation closes or expires

The confirmation expires after two minutes. Nothing is approved by expiry;
the assistant explains this and offers another attempt. Say **“Finish Entity
setup”** to reopen it.

Skipping, closing, or continuing without selecting the checkbox leaves your
permissions unchanged. The assistant explains how to resume and does not
immediately reopen a declined confirmation. This does not disable previously
approved memory hooks: tools keep their existing permission settings, while
hooks that still need trust remain unavailable. The assistant reports these
two permissions separately.

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
- Codex allows up to five minutes for runtime preparation, with a six-minute
  host startup window that leaves time for the MCP server to initialize. Warm
  starts skip this wait. Claude keeps its existing 25-second launcher budget.
  If downloads fail or take longer, the launcher reports the bootstrap log path;
  check the connection and reconnect after preparation completes. The checkout
  installer still prepares the runtime before opening Codex.
