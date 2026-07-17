#!/bin/bash
# brain — the ONE definition of "is the isolated runtime ready?".
#
# Sourced by ensure-runtime.sh (fast path + loser-wait recheck),
# boot-brain.sh (cold-install detection) and mcp-launch.sh (cold wait).
# Five call sites used to hand-write this predicate; when the runtime
# layout changes (e.g. the deferred $CLAUDE_PLUGIN_DATA relocation, which
# also adds a requirements-hash check), THIS is the only place to touch —
# a split-brain cold-detection between the launchers re-creates the
# concurrent-bootstrap race.
#
# Usage: brain_runtime_ready "$PLUGIN_ROOT" && echo warm

brain_runtime_ready() {
    [ -f "$1/.runtime-ready" ] && [ -x "$1/venv/bin/python" ]
}
