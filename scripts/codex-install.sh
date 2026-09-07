#!/usr/bin/env bash
# codex-install.sh — install (or refresh) the Entity plugin into a local Codex
# (ChatGPT desktop's Codex mode) from this checkout, and verify the result.
#
# Codex installs a plugin by copying the directory its marketplace entry points
# at, WHOLESALE — no .gitignore, no manifest filter — and then loads from that
# cache copy: $CODEX_HOME/plugins/cache/<marketplace>/<plugin>/<version>/.
# Pointed at this checkout it copies ~10 GB / 120k files (.git, conversations,
# venv), and an interrupted copy leaves a half-tree with no manifest. So the
# marketplace source is the PACKAGED tree build-plugin.sh ships, materialized
# at dist/codex/<plugin> (redeploy.sh keeps it fresh too).
#
# Touches only: dist/codex/<plugin>, the personal marketplace file
# (~/.agents/plugins/marketplace.json — one entry, others preserved), and our
# marketplace's subtree of the Codex plugin cache. Never reads or writes brain
# data: no BRAIN_DB_DIR, no XDG data dir.
#
# Usage: scripts/codex-install.sh [--codex /path/to/codex]
#   CODEX_HOME        Codex home (default ~/.codex)
#   CODEX_MARKETPLACE personal marketplace name (default anchor-dev)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
MARKETPLACE="${CODEX_MARKETPLACE:-anchor-dev}"
MARKETPLACE_FILE="$HOME/.agents/plugins/marketplace.json"
PY="$REPO/venv/bin/python"; [ -x "$PY" ] || PY=python3
PLUGIN="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$REPO/.codex-plugin/plugin.json")"
SRC="$REPO/dist/codex/$PLUGIN"

# The codex CLI: --codex, then PATH, then the one bundled in the ChatGPT app
# (it is not on PATH by default).
CODEX=""
if [ "${1:-}" = "--codex" ]; then CODEX="${2:-}"; fi
[ -n "$CODEX" ] || CODEX="$(command -v codex || true)"
[ -n "$CODEX" ] || CODEX="/Applications/ChatGPT.app/Contents/Resources/codex"
[ -x "$CODEX" ] || { echo "ERROR: codex CLI not found on PATH or in the ChatGPT app — pass --codex <path>" >&2; exit 1; }

# 1. Package → dist/codex/<plugin>: exactly what ships, nothing else.
( cd "$REPO" && ./build-plugin.sh >/dev/null )
rm -rf "${SRC:?}"; mkdir -p "$SRC"; unzip -o -q "$REPO/brain.plugin" -d "$SRC"
[ -f "$SRC/.codex-plugin/plugin.json" ] || { echo "ERROR: packaged tree lacks .codex-plugin/plugin.json" >&2; exit 1; }

# 2. Personal marketplace → that tree. A source path resolves RELATIVE to the
#    marketplace root, which is $HOME for the personal file; an absolute path
#    silently yields a marketplace with no plugins.
case "$SRC" in "$HOME"/*) ;; *) echo "ERROR: $SRC is not under \$HOME — a personal marketplace path must be" >&2; exit 1;; esac
REL="./${SRC#"$HOME/"}"
mkdir -p "$(dirname "$MARKETPLACE_FILE")"
"$PY" - "$MARKETPLACE_FILE" "$MARKETPLACE" "$PLUGIN" "$REL" <<'EOF'
import json, os, sys
path, market, plugin, rel = sys.argv[1:]
doc = {"name": market, "plugins": []}
if os.path.exists(path):
    doc = json.load(open(path))
    if doc.get("name") != market:
        sys.exit("ERROR: %s defines marketplace %r, not %r — set CODEX_MARKETPLACE to match or edit the file"
                 % (path, doc.get("name"), market))
entry = {"name": plugin, "source": {"source": "local", "path": rel}}
others = [p for p in doc.get("plugins", []) if p.get("name") != plugin]
doc["plugins"] = others + [entry]
with open(path, "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
EOF

# 3. Staging dirs left by aborted installs (plugin-install-*) under OUR
#    marketplace: partial copies of whatever the marketplace pointed at.
for d in "$CODEX_HOME/plugins/cache/$MARKETPLACE"/plugin-install-*/; do
  [ -d "$d" ] || continue
  echo "removing stale install staging dir: $d"
  rm -rf "$d"
done

# 4. Install — remove first when present, so the cache copy IS the package
#    just built (a stale copy would keep running old hooks and proxy code).
if "$CODEX" plugin list 2>/dev/null | grep -q "^$PLUGIN@$MARKETPLACE  *installed"; then
  "$CODEX" plugin remove "$PLUGIN@$MARKETPLACE"
fi
"$CODEX" plugin add "$PLUGIN@$MARKETPLACE"

# 5. Verify: Codex's own view, then the cache copy's contents.
if ! "$CODEX" plugin list | grep "^$PLUGIN@$MARKETPLACE " | grep -q "installed, enabled"; then
  echo "ERROR: codex does not report $PLUGIN@$MARKETPLACE as installed, enabled:" >&2
  "$CODEX" plugin list | grep -A4 "Marketplace \`$MARKETPLACE\`" >&2 || true
  exit 1
fi
INSTALLED="$(ls -d "$CODEX_HOME/plugins/cache/$MARKETPLACE/$PLUGIN"/*/ | sort | tail -1)"
for f in .codex-plugin/plugin.json hooks/hooks.codex.json hooks/scripts/mcp-launch.sh hooks/scripts/stamp-caller-session.sh servers/brain_mcp.py; do
  [ -f "$INSTALLED$f" ] || { echo "ERROR: installed copy lacks $f ($INSTALLED)" >&2; exit 1; }
done
echo "✓ $PLUGIN@$MARKETPLACE installed at $INSTALLED ($(find "$INSTALLED" -type f | wc -l | tr -d ' ') files)"

# 6. Warm the runtime INSIDE the cache copy now. Codex runs hooks and the MCP
#    server from that copy; left cold, the first session's boot hook dies at
#    its 15 s timeout while uv downloads Python + deps. Preparing the runtime
#    here also avoids delaying MCP availability on the first connection.
echo "bootstrapping the runtime in the cache copy (first time: a few minutes)..."
bash "$INSTALLED/hooks/scripts/ensure-runtime.sh" >"$INSTALLED/.bootstrap.log" 2>&1 \
  || { echo "ERROR: runtime bootstrap failed — see $INSTALLED/.bootstrap.log" >&2; exit 1; }
echo "✓ runtime ready: $("$INSTALLED/venv/bin/python" -c 'import sys; print(sys.version.split()[0])')"
echo "  Next: open Codex and ask 'Finish Entity setup'. Entity will check tools and hooks"
echo "  and offer one confirmation for Entity tools and automatic-memory setup."
echo "  If hooks need trust, Codex opens its review; no commands need to be typed."
echo "  After approval, return to the chat and ask Entity to check setup again."
