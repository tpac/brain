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

    # Archive contains uv-<triple>/uv and uv-<triple>/uvx — extract uv binary flat into bin/
    if ! tar -xzf "$TMP_TAR" -C "$PLUGIN_DIR/bin" --strip-components=1 "uv-$UV_TRIPLE/uv"; then
        echo "[brain-boot] FATAL: uv extraction failed" >&2
        exit 1
    fi
    rm -f "$TMP_TAR"
    chmod +x "$UV_BIN"

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
