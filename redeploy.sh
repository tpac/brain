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

# 2. Replace package-sourced dirs wholesale, THEN unzip — don't just overlay.
#    `unzip -o` only OVERWRITES packaged files; it never deletes a file that was
#    removed from the manifest, so stale orphan modules drift in across deploys
#    (e.g. a since-deleted servers/*.py lingers and can still be imported).
#    Pruning the fully-package-sourced dirs first makes the deployed tree a
#    faithful mirror of the package. Runtime artifacts (venv/ py/ .runtime-ready,
#    bin/uv) are NOT in the package and survive — bin/ is mixed (ships launchers,
#    holds the runtime uv), so it is left to unzip -o overlay, not pruned.
#    ${PLUGIN:?} guards against an empty var turning this into `rm -rf /…`.
for _d in servers hooks skills dashboard data scripts .claude-plugin; do
  rm -rf "${PLUGIN:?}/$_d"
done
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

# 3.5. Smoke-test the DEPLOYED package: import BOTH entrypoints — the MCP server
#       (servers.brain_mcp, per-session, talks to the daemon over TCP) AND the
#       daemon itself (servers.daemon_server, what brain-daemon launches) —
#       using the plugin's own venv from the plugin dir. Catches a packaged
#       module that imports something which didn't ship, at DEPLOY time, before
#       the daemon restart — loud here beats a silent "Failed to connect" / dead
#       daemon at the next session start. ~250ms: neither import loads torch or
#       the embedder (lazy at runtime). NOTE: this exercises only MODULE-LEVEL
#       imports; lazy in-function imports aren't reached — git ls-files
#       completeness (build-plugin.sh) is what guarantees those ship.
echo "smoke-testing packaged imports..."
if ! ( cd "$PLUGIN" && "$PLUGIN/venv/bin/python" -c "import servers.brain_mcp, servers.daemon_server" ); then
  echo "ERROR: a packaged entrypoint (brain_mcp / daemon_server) failed to import —" >&2
  echo "       aborting before daemon restart. A required module is missing from the" >&2
  echo "       package or an import is broken." >&2
  exit 1
fi

# 4. Restart the daemon so daemon-side Python goes live now.
echo "restarting daemon..."
bash "$PLUGIN/hooks/scripts/rebrain-daemon" || echo "  (daemon not running — it will boot fresh on next call)"

cat <<'EOF'

✓ Redeploy complete.

  Daemon-side changes (recall, encoding, scales, brain.py, scoring, embedder)
    → LIVE NOW (daemon re-exec'd). No new session needed.

  MCP-surface or wiring changes — START A NEW SESSION:
    • servers/brain_mcp.py             (resident MCP proxy, loaded once per session)
    • contract.py tool/field schemas   (tool list is fixed at the session handshake)
    • hooks/hooks.json, .mcp.json, .claude-plugin/plugin.json (read at session/plugin load)
EOF
