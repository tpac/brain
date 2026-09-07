#!/bin/bash
# 5.7 — upgrade smoke test: release N's user updates to N+1 and keeps their
# brain. DEV TOOL: lives in scripts/, never ships.
#
#   Usage: upgrade-smoke.sh <old-tree> <new-tree> [--keep]
#   Env:   see scripts/smoke-lib.sh
#
# The install smoke proves a stranger's first run. Only this proves the next
# release does not break the people who installed the last one — the
# population you have the day after publish. It installs OLD the way a
# stranger does (the whole install smoke), writes one lived memory through the
# daemon, installs NEW over it the way a marketplace update does (tree
# replaced, bootstrapped runtime kept — see smoke_overlay for the assumption),
# and runs the sessions after. Five layers get exercised at once: identity
# (same plugin, same brain path), the version signal, data living outside the
# plugin dir (D-13), schema migration at daemon boot, and daemon staleness
# convergence at session boot. What must hold: the daemon serves NEW's code,
# no node was lost and the lived one is still served, the entrypoints import,
# and a command still answers. The count may grow — a release that adds a
# seed node gap-fills it on first boot, and that is correct.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE_NAME=upgrade-smoke
. "$REPO/scripts/smoke-lib.sh"

OLD="${1:?usage: upgrade-smoke.sh <old-tree> <new-tree> [--keep]}"
NEW="${2:?usage: upgrade-smoke.sh <old-tree> <new-tree> [--keep]}"
[ "${3:-}" = "--keep" ] && SMOKE_KEEP=1
_version() { "$REPO/venv/bin/python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["version"])' "$1/.claude-plugin/plugin.json"; }
OLD_V="$(_version "$OLD")"; NEW_V="$(_version "$NEW")"
[ "$OLD_V" != "$NEW_V" ] || fail "old and new trees both read version $OLD_V — nothing to upgrade"

T0=$SECONDS
smoke_stage_new "$OLD"
say "installing $OLD_V as a stranger would"
smoke_install

# ── a life before the upgrade ───────────────────────────────────────────────
N0="$(smoke_node_count)" || _die "node count unavailable" count.err
LIVED="$(smoke_remember "upgrade smoke: lived on $OLD_V")" || _die "remember through the daemon failed" remember.err
N1="$(smoke_node_count)" || _die "node count unavailable" count.err
[ "$N1" -gt "$N0" ] || fail "remember did not land: $N0 -> $N1 nodes"
OLD_FP="$(smoke_fingerprint)" || _die "fingerprint unavailable" fingerprint.err
say "brain on $OLD_V: $N1 nodes (seed pack + 1 lived, id $LIVED), daemon fingerprint $OLD_FP"

# ── the update, and the sessions after it ──────────────────────────────────
smoke_overlay "$NEW"
say "installed $NEW_V over $OLD_V (runtime kept, brain untouched)"
smoke_converge
N2="$(smoke_node_count)" || _die "node count unavailable after the upgrade" count.err
[ "$N2" -ge "$N1" ] || fail "nodes lost across the upgrade: $N1 -> $N2"
smoke_node_present "$LIVED" || _die "the memory written on $OLD_V is gone after the upgrade ($LIVED)" present.err
[ -f "$BRAIN_HOME/brain.db" ] || fail "brain.db gone from $BRAIN_HOME"
say "converged: daemon on $NEW_V, $N2 nodes ($N1 before), lived memory served, MCP imports, commands answer"

say "PASS — $OLD_V → $NEW_V keeps the user's brain ($((SECONDS - T0))s total)"
