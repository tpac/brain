#!/bin/bash
# brain — runtime bootstrap
#
# Installs an isolated Python 3.11+ runtime inside the plugin dir so the
# brain doesn't depend on the user's system Python version. Idempotent:
# writes $PLUGIN_DIR/.runtime-ready when complete, skips on subsequent runs.
#
# Layout after a successful run:
#   $PLUGIN_DIR/bin/uv                  # uv binary (platform-correct)
#   $PLUGIN_DIR/py/<version>/...        # standalone Python 3.11+ install
#   $PLUGIN_DIR/venv/bin/python         # venv using that Python
#   $PLUGIN_DIR/venv/bin/...            # fastembed, onnxruntime, anthropic, ...
#   $PLUGIN_DIR/.runtime-ready          # sentinel
#
# Hard-fail on any error. Silent degradation back to system Python is what
# this script exists to escape from.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SENTINEL="$PLUGIN_DIR/.runtime-ready"
UV_BIN="$PLUGIN_DIR/bin/uv"
PY_DIR="$PLUGIN_DIR/py"
VENV_DIR="$PLUGIN_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"
REQ_FILE="$PLUGIN_DIR/requirements.txt"

PY_VERSION="${BRAIN_PY_VERSION:-3.11}"
UV_VERSION="${BRAIN_UV_VERSION:-0.5.11}"  # pinned for reproducibility

# Fast path — already bootstrapped. Verify sentinel + venv python still exists.
if [ -f "$SENTINEL" ] && [ -x "$VENV_PY" ]; then
    exit 0
fi

# ── Concurrency guard ──────────────────────────────────────────
# brain-env.sh has ~13 sourcers (MCP spawn, every hook, launchers, plists);
# on a clean install they ALL hit the cold path simultaneously. Unserialized,
# two racers extract uv over each other: overwriting a RUNNING (ad-hoc
# signed) binary in place invalidates its code signature and macOS SIGKILLs
# it — observed as `uv python install ... Killed: 9` on the first laptop
# install (2026-07-17), which took the whole MCP connection down with it.
# mkdir is the atomic test-and-set (macOS ships no flock(1)): exactly one
# winner runs the bootstrap; losers wait for the sentinel, then fast-path.
LOCK_DIR="$PLUGIN_DIR/.runtime-bootstrap.lock"
LOCK_STALE_S=600     # a lock older than this is a dead winner — steal it
WAIT_MAX_S=600       # loser wait ceiling (callers' own timeouts kill sooner)

_now() { date +%s; }
_lock_age() {
    # seconds since the lock dir was created; 0 if unreadable (treat as fresh)
    local _m
    _m=$(stat -f %m "$LOCK_DIR" 2>/dev/null || stat -c %Y "$LOCK_DIR" 2>/dev/null) || { echo 0; return; }
    echo $(( $(_now) - _m ))
}

_deadline=$(( $(_now) + WAIT_MAX_S ))
while :; do
    # Re-check the fast path each round — the winner may have finished.
    if [ -f "$SENTINEL" ] && [ -x "$VENV_PY" ]; then
        exit 0
    fi
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        break  # we are the winner — run the bootstrap below
    fi
    if [ "$(_lock_age)" -gt "$LOCK_STALE_S" ]; then
        echo "[brain-boot] stale bootstrap lock (winner died?) — stealing" >&2
        rm -rf "$LOCK_DIR" 2>/dev/null || true
        continue
    fi
    if [ "$(_now)" -ge "$_deadline" ]; then
        echo "[brain-boot] FATAL: waited ${WAIT_MAX_S}s for a concurrent bootstrap that never finished" >&2
        exit 1
    fi
    sleep 1
done
# Winner: always release the lock — a FAILED bootstrap must not deadlock the
# other consumers (they retry-acquire and either succeed or fail loudly).
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

echo "[brain-boot] first-run setup — installing isolated Python $PY_VERSION + embedding runtime" >&2
echo "[brain-boot]   plugin dir: $PLUGIN_DIR" >&2

# ── 1. uv binary ───────────────────────────────────────────────
if [ ! -x "$UV_BIN" ]; then
    echo "[brain-boot]   [1/4] installing uv $UV_VERSION..." >&2
    mkdir -p "$PLUGIN_DIR/bin"

    # Detect platform triple for uv release artifact
    OS="$(uname -s)"
    ARCH="$(uname -m)"
    case "$OS-$ARCH" in
        Darwin-arm64)         UV_TRIPLE="aarch64-apple-darwin" ;;
        Darwin-x86_64)        UV_TRIPLE="x86_64-apple-darwin" ;;
        Linux-x86_64)         UV_TRIPLE="x86_64-unknown-linux-gnu" ;;
        Linux-aarch64|Linux-arm64) UV_TRIPLE="aarch64-unknown-linux-gnu" ;;
        *)
            echo "[brain-boot] FATAL: unsupported platform $OS-$ARCH" >&2
            exit 1
            ;;
    esac

    UV_URL="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$UV_TRIPLE.tar.gz"
    TMP_TAR="$PLUGIN_DIR/bin/uv.tar.gz"

    if ! curl -LsSf --fail --max-time 120 -o "$TMP_TAR" "$UV_URL"; then
        echo "[brain-boot] FATAL: uv download failed from $UV_URL" >&2
        exit 1
    fi

    # Archive contains uv-<triple>/uv and uv-<triple>/uvx. Extract to a temp
    # dir and rename into place: mv is atomic and never invalidates a running
    # binary's code signature, unlike extracting over bin/uv in place (macOS
    # SIGKILLs a running ad-hoc-signed executable whose file is overwritten).
    # Defense in depth on top of the bootstrap lock above.
    TMP_EXTRACT="$PLUGIN_DIR/bin/.uv-extract.$$"
    mkdir -p "$TMP_EXTRACT"
    if ! tar -xzf "$TMP_TAR" -C "$TMP_EXTRACT" --strip-components=1 "uv-$UV_TRIPLE/uv"; then
        echo "[brain-boot] FATAL: uv extraction failed" >&2
        rm -rf "$TMP_EXTRACT"
        exit 1
    fi
    chmod +x "$TMP_EXTRACT/uv"
    mv -f "$TMP_EXTRACT/uv" "$UV_BIN"
    rm -rf "$TMP_EXTRACT" "$TMP_TAR"

    if [ ! -x "$UV_BIN" ]; then
        echo "[brain-boot] FATAL: uv extracted but not executable at $UV_BIN" >&2
        exit 1
    fi
fi

# ── 2. Standalone Python ───────────────────────────────────────
if [ ! -d "$PY_DIR" ] || [ -z "$(ls -A "$PY_DIR" 2>/dev/null)" ]; then
    echo "[brain-boot]   [2/4] installing Python $PY_VERSION (isolated)..." >&2
    mkdir -p "$PY_DIR"
    if ! "$UV_BIN" python install "$PY_VERSION" --install-dir "$PY_DIR" >&2; then
        echo "[brain-boot] FATAL: Python install failed" >&2
        exit 1
    fi
fi

# ── 3. venv + deps ─────────────────────────────────────────────
if [ ! -x "$VENV_PY" ]; then
    echo "[brain-boot]   [3/4] creating venv + installing deps (~200MB)..." >&2
    if ! UV_PYTHON_INSTALL_DIR="$PY_DIR" \
         "$UV_BIN" venv "$VENV_DIR" --python "$PY_VERSION" >&2; then
        echo "[brain-boot] FATAL: venv creation failed" >&2
        exit 1
    fi
fi

if [ ! -f "$REQ_FILE" ]; then
    echo "[brain-boot] FATAL: $REQ_FILE missing from plugin" >&2
    exit 1
fi

# Install or refresh deps. uv's install is idempotent + incremental.
if ! VIRTUAL_ENV="$VENV_DIR" \
     "$UV_BIN" pip install --python "$VENV_PY" -r "$REQ_FILE" >&2; then
    echo "[brain-boot] FATAL: dep install failed" >&2
    exit 1
fi

# ── 4. Pre-fetch embedding model ───────────────────────────────
# Pull the embedding model into fastembed's cache now, during this (already
# blocking) first-run bootstrap, so the daemon's first load_model() is cache-fast
# instead of blocking the first SessionStart on a ~150MB download. Non-fatal: if
# this fails (offline, etc.), the daemon downloads it on first recall as before.
# Model name read from plugin.json so it never drifts from the daemon's default.
# Paths/names passed via env (not string-interpolated) so a quote/metachar in the
# install path or model name can't break the python invocation.
MODEL_NAME="$(BRAIN_PLUGIN_DIR="$PLUGIN_DIR" "$VENV_PY" -c "import json, os; print(json.load(open(os.path.join(os.environ['BRAIN_PLUGIN_DIR'], '.claude-plugin', 'plugin.json'))).get('config', {}).get('embedder', {}).get('model_name', 'nomic-ai/nomic-embed-text-v1.5-Q'))" 2>/dev/null || echo 'nomic-ai/nomic-embed-text-v1.5-Q')"
echo "[brain-boot]   [4/4] pre-fetching embedding model ($MODEL_NAME)..." >&2
if ! BRAIN_PREFETCH_MODEL="$MODEL_NAME" "$VENV_PY" -c "import os; from fastembed import TextEmbedding; TextEmbedding(model_name=os.environ['BRAIN_PREFETCH_MODEL'])" >&2; then
    echo "[brain-boot] WARN: model pre-fetch failed — daemon will download it on first recall (slower); continuing" >&2
fi

# ── Done ───────────────────────────────────────────────────────
touch "$SENTINEL"
echo "[brain-boot] runtime ready — $("$VENV_PY" -c 'import sys; print(sys.version.split()[0])')" >&2
