#!/bin/bash
# brain — MCP server launcher
#
# Entry point from .mcp.json. Ensures the isolated Python runtime exists,
# then execs brain_mcp.py under the venv's Python. Claude Code speaks
# stdio MCP to this process, so stdout is reserved for MCP protocol —
# bootstrap output goes to stderr only.

set -euo pipefail

# Bootstrap runtime (idempotent — fast path ~8ms when sentinel is present)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/brain-env.sh" 1>&2

# Exec MCP server under the venv Python. stdout stays clean for MCP protocol.
exec "$BRAIN_PYTHON" "$PLUGIN_DIR/servers/brain_mcp.py" "$@"
