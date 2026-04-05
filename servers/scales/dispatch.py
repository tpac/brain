"""Scale dispatch infrastructure — TCP communication and agent dispatch factory.

Shared by all scale agents (S1 encode, S2 session encode, future scales).
Scale agents run in background threads with read-only Brain instances.
All writes go through daemon TCP (single-writer rule).
"""

import json
import os


def load_env():
    """Load .env file for API key. Shared by all scale agents."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())


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
    'connect', 'enrich', 'record_divergence', 'learn_vocabulary',
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
    from servers.daemon_dispatch import COMMAND_TABLE

    def dispatch(cmd, cmd_args):
        if cmd in ('remember', 'remember_batch', 'revise'):
            cmd_args.setdefault('encoding_source', encoding_source)
        if cmd in WRITE_COMMANDS:
            return daemon_tcp_send(cmd, cmd_args)
        entry = COMMAND_TABLE.get(cmd)
        if entry:
            return entry.handler(read_brain, cmd_args, [])
        return {"ok": False, "error": "Unknown command: %s" % cmd}

    return dispatch
