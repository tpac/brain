#!/bin/bash
# brain — SessionStart hook: boots brain, prints context + consciousness signals.
# Output: full brain state for Claude's context (injected via SessionStart stdout)
#
# Brain DB resolution: see resolve-brain-db.sh — prefers $CLAUDE_PLUGIN_DATA/brain
# (Claude Code's standard plugin data location) and falls back to legacy paths.
# If no DB found AND no auto-create succeeds, boot fails cleanly (no /tmp fallback —
# silent data loss is worse).


# Save stdin early — inline python commands below would consume it.
# Extract session_id from hook input and pass as env var.
HOOK_STDIN=$(cat)
BRAIN_HOOK_SESSION_ID=$(echo "$HOOK_STDIN" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('session_id',''))" 2>/dev/null)
export BRAIN_HOOK_SESSION_ID
# cwd from hook input — fed to the daemon as session identity (the daemon never
# introspects the Claude env itself). See SessionContext.cwd / session_env_for.
BRAIN_HOOK_CWD=$(echo "$HOOK_STDIN" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('cwd',''))" 2>/dev/null)
export BRAIN_HOOK_CWD

# ── Resolve ANTHROPIC_API_KEY from canonical config location ──
# Single source: ${XDG_CONFIG_HOME:-$HOME/.config}/brain/env (mode 600,
# dotenv format). Matches the CLI-tool convention (gh, stripe, kubectl, ...).
# Env-var override: a key already in ANTHROPIC_API_KEY (shell export) wins.
# The daemon's dispatch.load_env mirrors the env-file/shell resolution but NOT
# the userConfig fallback below — it inherits ANTHROPIC_API_KEY from this hook's
# env on the direct-spawn path. A launchd-spawned daemon sees neither
# CLAUDE_PLUGIN_OPTION_* nor this export, so a userConfig-only key needs the env
# file there. See docs/DISTRIBUTION-READINESS.md (§2 onboarding).
BRAIN_ENV_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/brain/env"
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f "$BRAIN_ENV_FILE" ]; then
  set -a
  . "$BRAIN_ENV_FILE"
  set +a
fi

# Additive userConfig fallback (CLAUDE_PLUGIN_OPTION_<KEY>, plugins-reference):
# fill the key from the plugin-config value if the env file / shell didn't.
# Both casings checked (doc doesn't pin <KEY>'s case).
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  if [ -n "${CLAUDE_PLUGIN_OPTION_API_KEY:-}" ]; then
    export ANTHROPIC_API_KEY="$CLAUDE_PLUGIN_OPTION_API_KEY"
  elif [ -n "${CLAUDE_PLUGIN_OPTION_api_key:-}" ]; then
    export ANTHROPIC_API_KEY="$CLAUDE_PLUGIN_OPTION_api_key"
  fi
  # MIRROR a userConfig-resolved key to the env file (mode 600). Required, not
  # optional: CLAUDE_PLUGIN_OPTION_* exists only in hook executions, and this
  # export dies with the hook — the launchd-spawned daemon (a separate process
  # tree; installed by install-daemon-service.sh on every fresh macOS install)
  # resolves the key ONLY via dispatch.load_env = env file + shell. Without
  # the mirror, a user who fills the plugin's key field still runs a keyless
  # daemon: llm_unavailable in the dashboard while boot looks fine (first
  # laptop install, 2026-07-15). Never overwrites an existing key line.
  if [[ "${ANTHROPIC_API_KEY:-}" == sk-* ]] \
     && ! grep -q '^ANTHROPIC_API_KEY=' "$BRAIN_ENV_FILE" 2>/dev/null; then
    mkdir -p "$(dirname "$BRAIN_ENV_FILE")" 2>/dev/null
    printf 'ANTHROPIC_API_KEY=%s\n' "$ANTHROPIC_API_KEY" >> "$BRAIN_ENV_FILE" \
      && chmod 600 "$BRAIN_ENV_FILE" \
      && echo "[brain-boot] plugin-config key mirrored to $BRAIN_ENV_FILE (mode 600) for the background daemon" >&2
  fi
fi

# Missing key is NOT a boot failure — the daemon boots keyless by design
# (brain.llm_available gates surface/encode/S2/warms; everything local runs
# without it: runtime bootstrap, embedder, brain.db, traces, direct recall).
# Keep booting and present the key as the one remaining setup step — a
# designed onboarding stage, not an error.
BRAIN_KEYLESS_BOOT=0
if [ -z "${ANTHROPIC_API_KEY:-}" ] || [[ "${ANTHROPIC_API_KEY}" != sk-* ]]; then
  BRAIN_KEYLESS_BOOT=1   # read below: keyless warm boots bring the dashboard up
  cat <<EOF
🧠 Anchor — setup in progress

Your brain is initializing: local runtime, embedding model, memory database,
and the brain daemon are being set up automatically — no action needed for
those. Memory storage, history traces, and direct recall work from this
session on.

One step remains to complete setup — learning (writing new memories) and
automatic memory surfacing use the Anthropic API and need your key:

  → Open http://localhost:${DASHBOARD_PORT:-47303}/setup and paste it there.
    (That's your brain's local dashboard — it also shows what Anchor
    remembers, recalls and learns. Local-only, nothing leaves this machine.
    On a brand-new install the page comes alive when the first-run setup
    finishes, ~1-2 minutes.)

  Alternatives: the Anchor plugin's settings in Claude Code — fill the
  "Anthropic API key" field, then start a new session (it's applied at
  session start). Or the env file directly:
     mkdir -p "${XDG_CONFIG_HOME:-\$HOME/.config}/brain"
     printf 'ANTHROPIC_API_KEY=sk-ant-...\n' > "${XDG_CONFIG_HOME:-\$HOME/.config}/brain/env"
     chmod 600 "${XDG_CONFIG_HOME:-\$HOME/.config}/brain/env"

Get a key at https://console.anthropic.com/settings/keys — the dashboard
and env-file paths take effect on your next message, no restart needed.
EOF
  cat >&2 <<EOF
[brain-boot] ANTHROPIC_API_KEY not set — booting in local-only mode (no encode/surface).
[brain-boot] Key location: $BRAIN_ENV_FILE — or the /setup page on the dashboard.
EOF
fi

# ── Cold install: own the bootstrap DETACHED, answer instantly ─────────────
# resolve-brain-db.sh → brain-env.sh → ensure-runtime.sh blocks 60-90s on a
# truly fresh install — this hook has a 15s timeout, so inline bootstrap
# means CC kills us and the identity injection + key notice are DROPPED
# (first laptop install, 2026-07-17). Instead: launch ensure-runtime as a
# detached child (survives our death; its mkdir-lock makes concurrent
# launches from the MCP spawn safe), print a static no-Python notice, and
# exit. The key resolution + mirror above already ran (pure shell). The MCP
# launcher waits on the sentinel and connects in-session when the bootstrap
# is quick; otherwise the next session lands on the ~8ms fast path.
_BOOT_PLUGIN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$(dirname "$0")/runtime-state.sh"
if ! brain_runtime_ready "$_BOOT_PLUGIN_ROOT"; then
  # Chain the full first-run provisioning after the bootstrap, in order:
  # runtime → daemon launchd service (RunAtLoad starts the daemon — without
  # this the first session had no daemon until MCP's health monitor kicked
  # in) → dashboard (the setup URL printed above must come alive without
  # the user invoking /dashboard — it IS the presented key-entry path).
  # Positional-args bash -c: metachar-proof for any install path (quotes,
  # $, backticks — not just spaces). Subshell + nohup: detach from the
  # hook's job/process group so CC's teardown can't take the chain down.
  _BOOT_SD="$(dirname "$0")"
  ( nohup bash -c '"$1" && "$2" && "$3"' _ \
      "$_BOOT_SD/ensure-runtime.sh" \
      "$_BOOT_SD/install-daemon-service.sh" \
      "$_BOOT_SD/ensure-dashboard.sh" \
      >> "$_BOOT_PLUGIN_ROOT/.bootstrap.log" 2>&1 & )
  cat <<EOF

🧠 Anchor — first-run install in progress

Anchor is building its local runtime in the background (isolated Python +
embedding model — a couple of minutes on typical networks; progress:
$_BOOT_PLUGIN_ROOT/.bootstrap.log). Nothing to do. Memory tools usually
appear from the NEXT session on (this session only on a very fast
connection) — everything else about this session works normally.
EOF
  echo "[brain-boot] cold install — bootstrap detached, hook exiting fast" >&2
  exit 0
fi

# Keyless warm boot: make sure the dashboard (and with it the /setup page the
# notices point to) is actually up. The curl probe skips the fork + full env
# resolution when it already answers (the common case after the first boot);
# output goes to the bootstrap log — a dashboard that fails to come up must
# not die silently while the boot notice points users at its URL.
if [ "$BRAIN_KEYLESS_BOOT" = "1" ]; then
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 1 "http://127.0.0.1:${DASHBOARD_PORT:-47303}/" 2>/dev/null)" != "200" ]; then
    ( nohup "$(dirname "$0")/ensure-dashboard.sh" \
        >> "$(cd "$(dirname "$0")/../.." && pwd)/.bootstrap.log" 2>&1 & )
  fi
fi

source "$(dirname "$0")/resolve-brain-db.sh"

# ── Validate hooks.json schema ──
# Claude Code expects { "hooks": { ... } } wrapper but silently drops hooks
# if the schema is wrong (Zod parse failure caught and swallowed in pluginLoader).
# Catch this ourselves so a format mistake doesn't silently kill all hooks.
HOOKS_FILE="$PLUGIN_ROOT/hooks/hooks.json"
if [ -f "$HOOKS_FILE" ]; then
  if ! python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
if 'hooks' not in data or not isinstance(data['hooks'], dict):
    print('ERROR: hooks.json missing top-level \"hooks\" key.', file=sys.stderr)
    print('Expected: {\"hooks\": {\"SessionStart\": [...], ...}}', file=sys.stderr)
    print('Got top-level keys: ' + ', '.join(data.keys()), file=sys.stderr)
    sys.exit(1)
" "$HOOKS_FILE" 2>&1; then
    echo "⚠️  hooks.json schema validation FAILED — hooks will NOT fire"
    echo "   Claude Code silently drops hooks when the wrapper format is wrong."
    echo "   Fix: wrap all hook events inside a top-level \"hooks\" key."
  fi
fi

# No DB found AND auto-create failed — guide the user.
# (resolve-brain-db.sh tries to auto-create at $CLAUDE_PLUGIN_DATA/brain or in
# Cowork mounts; we only land here if those weren't writable or weren't set.)
if [ -z "$BRAIN_DB_DIR" ]; then
  echo ""
  echo "brain: No brain.db found and auto-create failed."
  echo ""
  echo "Two options:"
  echo ""
  echo "  1. CONNECT TO EXISTING BRAIN — Set the path to your brain folder:"
  echo "     In Claude Code settings or .claude/settings.json, add to env:"
  echo '       "BRAIN_DB_DIR": "/path/to/your/brain/folder"'
  echo "     The folder should contain (or will contain) brain.db."
  echo ""
  echo "  2. START FRESH — Create a new brain:"
  if [ -n "$CLAUDE_PLUGIN_DATA" ]; then
    echo "     mkdir -p \"\$CLAUDE_PLUGIN_DATA/brain\""
  else
    echo "     mkdir -p ~/AgentsContext/brain"
  fi
  echo "     Then restart this session. The brain will initialize automatically."
  echo ""
  echo "Searched locations:"
  echo "  - \$BRAIN_DB_DIR env var (not set)"
  echo "  - /sessions/*/mnt/AgentsContext/brain/ (Cowork — not found)"
  echo "  - \$CLAUDE_PLUGIN_DATA/brain/ ($([ -n "$CLAUDE_PLUGIN_DATA" ] && echo "not found at $CLAUDE_PLUGIN_DATA/brain" || echo "\$CLAUDE_PLUGIN_DATA not set"))"
  echo "  - \$HOME/AgentsContext/brain/ (not found, legacy)"
  echo ""
  exit 0
fi

# ── Provision the launchd service on fresh installs (macOS, idempotent) ──
# Install the LaunchAgent BEFORE ensure_daemon() so launchd owns the daemon from
# the first boot (KeepAlive + RunAtLoad). Without this, a fresh macOS install has
# no plist, so ensure_daemon's manages() is False and it direct-spawns a DETACHED
# daemon — no KeepAlive, no boot persistence. No-op on every non-fresh boot and
# off macOS. Non-fatal: on failure ensure_daemon's detached fallback still brings
# the daemon up (boot never has set -e).
bash "$(dirname "$0")/install-daemon-service.sh"

# ── Start daemon via ensure_daemon() — fcntl-locked singleton ──
# No inline spawning. ensure_daemon() handles: lock, ping, spawn, code-change restart.
PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd)" python3 -c "
from servers.daemon_client import ensure_daemon; import os, sys
db = os.path.join(os.environ.get('BRAIN_DB_DIR', ''), 'brain.db')
sys.stderr.write('[brain-boot] Daemon %s\n' % ('ready' if ensure_daemon(db) else 'FAILED'))
"

# ── Verify MCP server can start ──
# Claude Code starts the MCP server as a separate process. If it crashes (e.g. import error),
# Claude silently gets no brain tools. Catch that here and scream.
MCP_CRASH=""
PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd)" python3 -c "
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('$PLUGIN_ROOT')), ''))
try:
    from servers.brain_mcp import TOOLS
    sys.stderr.write('[brain-boot] MCP server OK — %d tools available\n' % len(TOOLS))
except Exception as e:
    # Write to stdout so it appears in session context
    print('⚠️  MCP SERVER BROKEN — Anchor will have NO direct brain tools')
    print('   Error: %s' % e)
    print('   The brain hooks still work, but recall/remember/connect MCP tools are dead.')
    print('   Fix the import error in servers/brain_mcp.py and restart the session.')
    sys.exit(1)
" 2>&1
MCP_EXIT=$?
if [ $MCP_EXIT -ne 0 ]; then
  # Also check for crash sentinel from a previous failed startup
  if [ -f /tmp/brain-mcp-crash.txt ]; then
    echo ""
    echo "Last MCP crash details:"
    head -5 /tmp/brain-mcp-crash.txt
  fi
fi

# Run boot
exec python3 "$(dirname "$0")/boot_brain.py"
