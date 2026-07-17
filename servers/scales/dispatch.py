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


def resolve_api_key() -> str:
    """The API key the daemon SHOULD be using RIGHT NOW.

    The env file is the canonical user-editable source — the dashboard's
    /setup form and boot-brain.sh's userConfig mirror both write it, and both
    promise "picked up automatically, no restart". load_env()'s no-override
    policy makes that promise false for key REPLACEMENT: once any key is in
    os.environ (first resolution, hook inheritance), a rewritten file is
    invisible (code review 2026-07-17, heal-truth finding). So: the file's
    current value wins when present; os.environ (real shell export, plist)
    is the fallback. Read every call — one stat + tiny read, only ever on
    LLM-availability checks, not the recall hot path.
    """
    xdg = os.environ.get('XDG_CONFIG_HOME') or os.path.join(
        os.path.expanduser('~'), '.config')
    env_path = os.path.join(xdg, 'brain', 'env')
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith('ANTHROPIC_API_KEY='):
                    v = line.split('=', 1)[1].strip()
                    if v:
                        return v
    except OSError:
        pass
    return os.environ.get('ANTHROPIC_API_KEY', '')


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


def stamp_project_provenance(cmd, cmd_args, project):
    """Enforce deterministic project provenance on an outgoing write's args.

    `project` on a node is PROVENANCE — the repo the session was working in
    when the node was learned (SessionContext.project, derived from cwd).
    It is never agent-authored: node-creating payloads get the session's
    value force-stamped; agent-supplied values on any other write are
    dropped. Mutates cmd_args in place; returns a list of warning strings
    for values that were overridden or dropped (callers surface them).

    project policy value:
      - None  → no-op. The caller has no session authority (bare handler
        call without an ambient session); an upstream chokepoint may already
        have stamped, so args pass through untouched.
      - ''    → authoritative "no project here": strip agent-supplied values
        (non-repo session, or an S2 unit — graph-scope work never invents
        provenance).
      - 'x'   → force-stamp onto node-creating payloads, strip elsewhere.
    """
    if project is None or not isinstance(cmd_args, dict):
        return []
    warnings = []

    def _force(node_dict, where):
        supplied = node_dict.get('project')
        if project:
            if supplied and supplied != project:
                warnings.append(
                    "%s: project is session-derived provenance — supplied "
                    "%r replaced with %r" % (where, supplied, project))
            node_dict['project'] = project
        elif 'project' in node_dict:
            # presence-based, not truthiness: an explicit '' is also
            # agent-authored and must not reach the node
            supplied = node_dict.pop('project')
            if supplied:
                warnings.append(
                    "%s: project is session-derived provenance and this "
                    "session has none — supplied %r dropped" % (where, supplied))

    def _strip(d, where):
        # presence-based: `project: ''` through a revise would WIPE birth
        # provenance (validate_field accepts '', revise writes the column) —
        # pop the key whenever it appears, warn when it carried a value
        if isinstance(d, dict) and 'project' in d:
            supplied = d.pop('project')
            if supplied:
                warnings.append(
                    "%s: project is session-derived provenance — "
                    "agent-supplied %r dropped (set at node creation, moved "
                    "only by migration)" % (where, supplied))

    if cmd == 'remember':
        _force(cmd_args, 'remember')
    elif cmd == 'remember_batch':
        for i, spec in enumerate(cmd_args.get('nodes') or []):
            if isinstance(spec, dict):
                _force(spec, 'remember_batch.nodes[%d]' % i)
    elif cmd in ('revise', 'revise_batch'):
        if cmd == 'revise':
            _strip(cmd_args, 'revise')
        else:
            for i, spec in enumerate(cmd_args.get('revisions') or []):
                _strip(spec, 'revise_batch.revisions[%d]' % i)
    elif cmd == 'brain_batch':
        # force-vs-strip derives from the op contract (creates_node flag on
        # BATCH_OP_SPECS), not a local op enumeration — a new node-creating
        # op added via the documented contract path inherits the stamp.
        from ..contract import BATCH_OP_SPECS
        for i, op in enumerate(cmd_args.get('operations') or []):
            if not isinstance(op, dict):
                continue
            where = 'brain_batch.operations[%d]' % i
            spec = BATCH_OP_SPECS.get(op.get('op'), {})
            if spec.get('creates_node'):
                _force(op, where)
            else:
                _strip(op, where)
    return warnings
