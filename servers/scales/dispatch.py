"""Scale dispatch infrastructure — env loading + the write-command classification.

Shared by all scale agents (S1 Scribe, the S2 units, future scales). Both now
run IN-PROCESS on the daemon's brain and write through
`scales/s2/base.py::_make_encoder_dispatch`, which calls COMMAND_TABLE handlers
directly under `brain.write_lock` (the same lock daemon_server.py uses for
client requests, so cross-writer serialization is guaranteed).

This module no longer holds a dispatch factory. The legacy out-of-process path
— a background-thread `Brain(skip_embedder=True)` copy writing back over TCP via
`daemon_tcp_send` + `make_scale_dispatch` — was retired when S1 Scribe converged
onto the in-process pattern (S1 was its last user). What remains: `load_env` and
the canonical write-command CLASSIFICATION the attribution chokepoint reads.
"""

import os


def load_env():
    """Load ANTHROPIC_API_KEY from the canonical config location.

    Source: ${XDG_CONFIG_HOME:-~/.config}/brain/env (dotenv format, mode 600).
    Matches the CLI-tool convention (gh, stripe, kubectl, ...).

    A key already present in os.environ (real shell env, daemon plist) wins
    and the file is not read.
    """
    if os.environ.get('ANTHROPIC_API_KEY'):
        return
    xdg = os.environ.get('XDG_CONFIG_HOME') or os.path.join(
        os.path.expanduser('~'), '.config')
    env_path = os.path.join(xdg, 'brain', 'env')
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip()
                    if not os.environ.get(k):
                        os.environ[k] = v
    except Exception:
        # A profile-resolution failure must never crash an LLM call.
        return


# The write commands a scale agent can issue — the canonical classification read
# by the attribution chokepoint (s2/base.apply_encoder_attribution) and the
# contract test (test_contract_sync Layer 5: must be a subset of COMMAND_TABLE).
# (Formerly also the "route via TCP" set, back when out-of-process scale agents
# existed.)
WRITE_COMMANDS = {
    'remember', 'remember_batch', 'revise', 'revise_batch',
    'connect', 'connect_batch', 'brain_batch',
    'enrich',
    'trace_append', 'set_config',
}

# The subset of WRITE_COMMANDS that create or attribute nodes/edges and so must
# carry the scale agent's encoding_source. Every command here reads a top-level
# `encoding_source` and cascades it (the batch handlers fan it out per-op).
# DERIVED from WRITE_COMMANDS (single source of truth) minus the explicitly
# non-attributed writes — so a newly-added attributed write is tagged
# automatically and can't silently slip through as 'anchor' (the gap that once
# mislabelled brain_batch encodes and hid them from the dashboard's encoder view).
NON_ATTRIBUTED_WRITES = {'enrich', 'trace_append', 'set_config'}
ATTRIBUTED_WRITE_COMMANDS = WRITE_COMMANDS - NON_ATTRIBUTED_WRITES
