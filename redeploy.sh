#!/usr/bin/env bash
# redeploy.sh — push the brain repo into the installed plugin (dev → plugin).
#
# Code-only overlay: preserves venv/ py/ bin/ .runtime-ready, so there is NO
# ~15s cold runtime rebuild. Deps are refreshed only when requirements.txt
# changes. Restarts the daemon so daemon-side code goes live in the CURRENT
# session; a NEW session is only needed for MCP-surface / hook / manifest
# changes (see the note printed at the end).
#
# Dev/prod data split: this only swaps CODE. The brain db lives outside both
# trees (~/AgentsContext/brain) and is untouched. Run dev/eval against an
# isolated db via BRAIN_DB_DIR; the plugin resolves the real db on its own.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN="/Users/tpac/.claude/plugins/marketplaces/local-desktop-app-uploads/brain"

if [ -L "$PLUGIN" ]; then
  echo "ERROR: $PLUGIN is a symlink — refusing (that's the old bad-practice setup)." >&2
  exit 1
fi
[ -d "$PLUGIN" ] || { echo "ERROR: plugin dir not found: $PLUGIN" >&2; exit 1; }

# 1. Build a clean package. The manifest is git-derived (git ls-files) for the
#    runtime code dirs, so individual files can no longer silently rot out of it;
#    build-plugin.sh still fails loudly if a dir vanished or an explicit file is
#    missing. The smoke test in step 3.5 is the backstop for bad imports.
cd "$REPO"
./build-plugin.sh

# 2. Overlay code only. unzip -o overwrites packaged files; venv/ py/ bin/
#    .runtime-ready are not in the package, so they survive untouched.
unzip -o -q "$REPO/brain.plugin" -d "$PLUGIN"

# 3. Refresh deps only when requirements.txt actually changed.
NEWHASH="$(shasum "$REPO/requirements.txt" | awk '{print $1}')"
OLDHASH="$(cat "$PLUGIN/.deployed-reqs-hash" 2>/dev/null || true)"
if [ "$NEWHASH" != "$OLDHASH" ]; then
  echo "requirements.txt changed — refreshing deps..."
  "$PLUGIN/bin/uv" pip install --python "$PLUGIN/venv/bin/python" -r "$PLUGIN/requirements.txt"
  echo "$NEWHASH" > "$PLUGIN/.deployed-reqs-hash"
else
  echo "deps unchanged — skipping pip install."
fi

# 3.5. Smoke-test the DEPLOYED package: import the MCP entrypoint using the
#       plugin's own venv, from the plugin dir. Catches the manifest-omission /
#       broken-import class at DEPLOY time (e.g. a packaged brain_mcp.py that
#       imports a module which didn't ship) — loud here beats a silent "Failed
#       to connect" / zero MCP tools at the next session start. ~40ms: brain_mcp
#       is a thin TCP client to the daemon, so importing it loads no embedder.
echo "smoke-testing packaged imports..."
if ! ( cd "$PLUGIN" && "$PLUGIN/venv/bin/python" -c "import servers.brain_mcp" ); then
  echo "ERROR: packaged 'servers.brain_mcp' failed to import — aborting before daemon restart." >&2
  echo "       A required module is missing from the package or an import is broken." >&2
  exit 1
fi

# 4. Restart the daemon so daemon-side Python goes live now.
echo "restarting daemon..."
bash "$PLUGIN/hooks/scripts/restart-daemon.sh" || echo "  (daemon not running — it will boot fresh on next call)"

cat <<'EOF'

✓ Redeploy complete.

  Daemon-side changes (recall, encoding, scales, brain.py, scoring, embedder)
    → LIVE NOW (daemon re-exec'd). No new session needed.

  MCP-surface or wiring changes — START A NEW SESSION:
    • servers/brain_mcp.py             (resident MCP proxy, loaded once per session)
    • contract.py tool/field schemas   (tool list is fixed at the session handshake)
    • hooks/hooks.json, .mcp.json, .claude-plugin/plugin.json (read at session/plugin load)
EOF
