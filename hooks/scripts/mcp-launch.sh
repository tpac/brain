#!/bin/bash
# brain — MCP server launcher
#
# Entry point from .mcp.json. Ensures the isolated Python runtime exists,
# then execs brain_mcp.py under the venv's Python. Claude Code speaks
# stdio MCP to this process, so stdout is reserved for MCP protocol —
# bootstrap output goes to stderr only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MCP_PLUGIN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/runtime-state.sh"

# ── Cold install: NEVER bootstrap inline — wait on the sentinel ────────────
# Claude Code gives this spawn 30s and never retries a failed connection. A
# fresh bootstrap can exceed that on slow networks, and racing it against the
# SessionStart hook's bootstrap SIGKILLed uv on the first laptop install
# (2026-07-17). So: kick the (mkdir-locked, so concurrency-safe) bootstrap
# detached in case nothing else has, then poll the sentinel with a deadline
# under CC's 30s. Ready in time → connect normally this session. Not ready →
# exit with a clear message; the NEXT session lands on the ~8ms fast path.
if ! brain_runtime_ready "$_MCP_PLUGIN_ROOT"; then
    # Subshell + nohup: detach from this launcher's process group — when the
    # 25s deadline below exits 1 and CC tears the MCP spawn down, a
    # group-directed signal must not kill the bootstrap mid-install.
    ( nohup "$SCRIPT_DIR/ensure-runtime.sh" \
        >> "$_MCP_PLUGIN_ROOT/.bootstrap.log" 2>&1 & )
    echo "[brain-mcp] cold install — waiting for runtime bootstrap (max 25s)..." >&2
    _mcp_deadline=$(( $(date +%s) + 25 ))
    while ! brain_runtime_ready "$_MCP_PLUGIN_ROOT"; do
        if [ "$(date +%s)" -ge "$_mcp_deadline" ]; then
            echo "[brain-mcp] runtime still bootstrapping — brain tools will be available next session (progress: $_MCP_PLUGIN_ROOT/.bootstrap.log)" >&2
            exit 1
        fi
        sleep 1
    done
    echo "[brain-mcp] runtime ready — connecting" >&2
fi

# Runtime present (fast path ~8ms) — wire env and exec the server.
source "$SCRIPT_DIR/brain-env.sh" 1>&2

# Exec MCP server under the venv Python. stdout stays clean for MCP protocol.
exec "$BRAIN_PYTHON" "$PLUGIN_DIR/servers/brain_mcp.py" "$@"
