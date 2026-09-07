#!/bin/bash
# 5.7 — install smoke test: a stranger's first run, on a COPY of a tree.
# DEV TOOL: lives in scripts/, never ships.
#
#   Usage: install-smoke.sh <tree> [--keep]
#   Env:   see scripts/smoke-lib.sh (bootstrap timeout, uv cache, keep)
#
# The suite runs INSIDE the export tree, so it exercises only what a test
# imports. A file that ships, that no test touches, and that first boot needs
# on a machine without the author's setup is invisible to it. This is the gate
# for that gap: the cold SessionStart hook answers fast and bootstraps
# detached; the warm boot births the brain at the XDG service dir (D-13), the
# daemon comes up keyless, both entrypoints import, and the next non-hook
# shell resolves the brain through the ladder and persists resolved.env.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE_NAME=install-smoke
. "$REPO/scripts/smoke-lib.sh"

TREE="${1:?usage: install-smoke.sh <tree> [--keep]}"
[ "${2:-}" = "--keep" ] && SMOKE_KEEP=1

T0=$SECONDS
smoke_stage_new "$TREE"
smoke_install
say "PASS — a stranger's first run works ($((SECONDS - T0))s total)"
