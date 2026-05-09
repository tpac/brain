"""Scale dispatch infrastructure — TCP communication and agent dispatch factory.

Shared by all scale agents (S1 encode, S2 session encode, future scales).

Two write paths exist:

1. **In-process (S2 encoder running in the daemon's pool)** — calls
   COMMAND_TABLE handlers directly under `brain.write_lock` (acquired in
   `scales/s2/base.py::_make_encoder_dispatch`). Same lock that
   daemon_server.py uses for client requests, so cross-writer
   serialization is guaranteed.
2. **Out-of-process (S1 encode subprocess, future scales)** — calls
   `daemon_tcp_send` here, which dispatches via the daemon and therefore
   goes through `_locked_exec` → `brain.write_lock`.

Either way, every write hits the same lock.
"""

import json
import os


def load_env():
    """Load .env file for API key. Shared by all scale agents.

    Search order (first existing file wins for any given key):
      1. ${BRAIN_DB_DIR}/.env   — runtime cache populated by boot-brain.sh
                                  from userConfig.anthropic_api_key (the
                                  Claude Code keychain). The standard path.
      2. ~/AgentsContext/brain/.env — same as (1) when BRAIN_DB_DIR is unset
                                      (e.g. ad-hoc scripts started outside
                                      the launchd plist).
      3. <plugin_root>/.env    — legacy in-repo .env, kept as fallback so
                                 pre-migration installs keep working until
                                 the user moves to userConfig via /plugin.

    A key already present in os.environ (real shell env, or a higher
    -priority earlier file) is never overwritten by a later source.
    """
    plugin_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    candidates = []
    db_dir = os.environ.get('BRAIN_DB_DIR')
    if db_dir:
        candidates.append(os.path.join(db_dir, '.env'))
    candidates.append(os.path.join(
        os.path.expanduser('~'), 'AgentsContext', 'brain', '.env'))
    candidates.append(os.path.join(plugin_root, '.env'))

    seen = set()
    for env_path in candidates:
        if env_path in seen or not os.path.exists(env_path):
            continue
        seen.add(env_path)
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        k, v = k.strip(), v.strip()
                        # Don't overwrite a key already set by an earlier
                        # (higher-priority) source.
                        if not os.environ.get(k):
                            os.environ[k] = v
        except Exception:
            # A profile-resolution failure must never crash an LLM call.
            # If a file is unreadable, fall through to the next candidate.
            continue


def daemon_tcp_send(cmd, args):
    """Send a command to the daemon via TCP.

    Used by background threads that must not write to DB directly
    (single-writer rule). Returns {"ok": bool, "result": ...}.
    """
    import socket
    port = 47200 + (os.getuid() % 100)
    msg = json.dumps({"cmd": cmd, "args": args}) + "\n"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(30)
    try:
        s.connect(("127.0.0.1", port))
        s.sendall(msg.encode())
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        return json.loads(data.decode().strip()) if data else {"ok": False, "error": "empty"}
    except Exception as e:
        return {"ok": False, "error": "daemon TCP: %s" % e}
    finally:
        s.close()


# Commands that must go through daemon TCP (all writes)
WRITE_COMMANDS = {
    'remember', 'remember_batch', 'revise', 'revise_batch',
    'connect', 'connect_batch', 'brain_batch',
    'enrich',
    'trace_append', 'set_config',
}


def make_scale_dispatch(read_brain, encoding_source='encoder:sonnet'):
    """Create a dispatch function for a scale agent.

    Reads use local read_brain (no lock contention).
    Writes go through daemon TCP (single-writer rule).
    encoding_source is set on all remember/revise calls.

    Args:
        read_brain: Brain instance for read operations (background thread's copy)
        encoding_source: encoding_source value for new/revised nodes

    Returns:
        dispatch(cmd, args) -> dict
    """
    from servers.daemon_dispatch import COMMAND_TABLE, check_unknown_keys

    def dispatch(cmd, cmd_args):
        if cmd in ('remember', 'remember_batch', 'revise'):
            cmd_args.setdefault('encoding_source', encoding_source)
        if cmd in WRITE_COMMANDS:
            return daemon_tcp_send(cmd, cmd_args)
        entry = COMMAND_TABLE.get(cmd)
        if entry:
            check_unknown_keys(cmd, entry, cmd_args, read_brain)
            return entry.handler(read_brain, cmd_args, [])
        return {"ok": False, "error": "Unknown command: %s" % cmd}

    return dispatch
