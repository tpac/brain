#!/usr/bin/env python3
"""
Brain MCP Server — Thin stdio proxy to brain daemon.

Zero-dependency MCP server (JSON-RPC 2.0 over stdio).
Forwards tool calls to the brain daemon via Unix socket.
Embedder loads once in the daemon; this process is just a relay.

Error policy: NEVER swallow errors silently. If something fails,
stderr gets a message and the caller gets a real error.
"""

import json
import os
import sys
import socket

# ── Daemon communication ──

SOCKET_PATH = "/tmp/brain-daemon-{}.sock".format(os.getuid())


def daemon_send(cmd, args=None, timeout=30.0):
    """Send command to brain daemon, return result dict."""
    if not os.path.exists(SOCKET_PATH):
        return {"ok": False, "error": "Daemon not running (no socket at {})".format(SOCKET_PATH)}

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(SOCKET_PATH)
        msg = json.dumps({"cmd": cmd, "args": args or {}}) + "\n"
        sock.sendall(msg.encode("utf-8"))
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        if data:
            return json.loads(data.decode("utf-8").strip())
        return {"ok": False, "error": "Empty response from daemon"}
    except socket.timeout:
        return {"ok": False, "error": "Daemon timeout ({}s)".format(timeout)}
    except Exception as e:
        return {"ok": False, "error": "Daemon connection error: {}".format(e)}
    finally:
        sock.close()


def ensure_daemon_running():
    """Start daemon if not running. Returns True if ready."""
    # Try ping first — fast path
    resp = daemon_send("ping", timeout=2.0)
    if resp.get("ok"):
        return True

    # Daemon not running — try to start it
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent not in sys.path:
        sys.path.insert(0, parent)

    try:
        from servers.daemon import ensure_daemon
        db_dir = os.environ.get("BRAIN_DB_DIR", "")
        if not db_dir:
            # Resolve DB path
            home = os.path.expanduser("~")
            candidate = os.path.join(home, "AgentsContext", "brain")
            if os.path.isfile(os.path.join(candidate, "brain.db")):
                db_dir = candidate

        if not db_dir:
            return False

        return ensure_daemon(os.path.join(db_dir, "brain.db"))
    except Exception as e:
        sys.stderr.write("[brain-mcp] Failed to start daemon: {}\n".format(e))
        return False


# ── MCP Protocol ──

SERVER_NAME = "brain"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2024-11-05"

# Tool definitions — what Claude sees as native tools
TOOLS = [
    {
        "name": "recall",
        "description": "Semantic recall from brain — searches nodes by meaning using embeddings. Returns ranked results with titles, content, types, confidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (semantic, not keyword)"},
                "limit": {"type": "integer", "description": "Max results (default 8)", "default": 8}
            },
            "required": ["query"]
        }
    },
    {
        "name": "remember",
        "description": "Store a new node in the brain. Types: decision, rule, lesson, concept, context, pattern, convention, mechanism, impact, constraint, purpose, mental_model, uncertainty, vocabulary, hypothesis, tension, aspiration, catalyst, interaction, meta_learning, failure_mode, performance, capability, arch_constraint, code_concept, fn_reasoning, param_influence, comment_anchor, bug_lesson.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "Node type"},
                "title": {"type": "string", "description": "Specific, scannable title"},
                "content": {"type": "string", "description": "Rich content with reasoning, tradeoffs, specifics"},
                "locked": {"type": "boolean", "description": "Lock node (for decisions, rules, lessons)", "default": False},
                "confidence": {"type": "number", "description": "Confidence 0.0-1.0", "default": 0.5},
                "keywords": {"type": "string", "description": "Space-separated keywords for search"},
                "project": {"type": "string", "description": "Project scope"},
                "emotion": {"type": "number", "description": "Emotional valence -1.0 to 1.0"}
            },
            "required": ["type", "title", "content"]
        }
    },
    {
        "name": "connect",
        "description": "Create a weighted edge between two brain nodes. Relations: related_to, caused_by, depends_on, contradicts, supports, produced, evolved_from, blocks, enables, example_of.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source node ID"},
                "target_id": {"type": "string", "description": "Target node ID"},
                "relation": {"type": "string", "description": "Edge relation type", "default": "related_to"},
                "weight": {"type": "number", "description": "Edge weight 0.0-1.0", "default": 0.5}
            },
            "required": ["source_id", "target_id"]
        }
    },
    {
        "name": "consciousness",
        "description": "Get brain consciousness signals — fading knowledge, tensions, vocabulary gaps, encoding health, errors, mental model drift, uncertainties, dream insights, reminders.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "context_boot",
        "description": "Full brain boot — returns stats, locked rules, consciousness signals, last session note. Use at session start.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user": {"type": "string", "default": "User"},
                "project": {"type": "string", "default": "default"},
                "task": {"type": "string"}
            },
        }
    },
    {
        "name": "set_config",
        "description": "Set a brain configuration value.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Config key"},
                "value": {"type": "string", "description": "Config value"}
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "get_config",
        "description": "Get a brain configuration value.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Config key"},
                "default": {"type": "string", "description": "Default if not set", "default": ""}
            },
            "required": ["key"]
        }
    },
    {
        "name": "health_check",
        "description": "Run brain health check — schema integrity, orphan nodes, edge consistency.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "auto_fix": {"type": "boolean", "description": "Auto-fix issues", "default": True}
            },
        }
    },
    {
        "name": "save",
        "description": "Force brain save to disk.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "ping",
        "description": "Check if brain daemon is alive.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "eval",
        "description": "Escape hatch — evaluate arbitrary Python expression on brain object. Variable 'brain' is the Brain instance. Use for methods not exposed as tools (e.g. remember_lesson, remember_impact, record_divergence, learn_vocabulary, etc).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python expression to eval (brain object available as 'brain')"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "engineering_context",
        "description": "Get engineering memory context — mechanisms, impacts, constraints, conventions for a project.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "default": "default"}
            },
        }
    },
]


def make_response(request_id, result):
    """Build a JSON-RPC 2.0 response."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id, code, message):
    """Build a JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_initialize(request_id):
    return make_response(request_id, {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
    })


def handle_tools_list(request_id):
    return make_response(request_id, {"tools": TOOLS})


def handle_tools_call(request_id, params):
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    # Map tool name to daemon command (1:1 for now)
    resp = daemon_send(tool_name, arguments)

    if resp.get("ok"):
        result_text = json.dumps(resp["result"], indent=2, default=str)
        return make_response(request_id, {
            "content": [{"type": "text", "text": result_text}]
        })
    else:
        error_msg = resp.get("error", "Unknown daemon error")
        # Try to restart daemon and retry once
        if "not running" in error_msg or "connection" in error_msg.lower():
            sys.stderr.write("[brain-mcp] Daemon down, attempting restart...\n")
            if ensure_daemon_running():
                resp = daemon_send(tool_name, arguments)
                if resp.get("ok"):
                    result_text = json.dumps(resp["result"], indent=2, default=str)
                    return make_response(request_id, {
                        "content": [{"type": "text", "text": result_text}]
                    })
                error_msg = resp.get("error", "Unknown daemon error after restart")

        return make_response(request_id, {
            "content": [{"type": "text", "text": "ERROR: {}".format(error_msg)}],
            "isError": True
        })


def handle_ping(request_id):
    return make_response(request_id, {})


def send(msg):
    """Write JSON-RPC message to stdout."""
    line = json.dumps(msg)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main():
    # Ensure daemon is running before accepting connections
    sys.stderr.write("[brain-mcp] Starting MCP server...\n")
    if not ensure_daemon_running():
        sys.stderr.write("[brain-mcp] FATAL: Cannot start brain daemon. Exiting.\n")
        sys.exit(1)
    sys.stderr.write("[brain-mcp] Daemon connected. Serving {} tools.\n".format(len(TOOLS)))

    # Main loop — read JSON-RPC from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stderr.write("[brain-mcp] Bad JSON: {}\n".format(e))
            continue

        method = msg.get("method", "")
        request_id = msg.get("id")
        params = msg.get("params", {})

        # Notifications (no id) — acknowledge silently
        if request_id is None:
            if method == "notifications/initialized":
                pass  # Client acknowledged init
            continue

        if method == "initialize":
            send(handle_initialize(request_id))
        elif method == "tools/list":
            send(handle_tools_list(request_id))
        elif method == "tools/call":
            send(handle_tools_call(request_id, params))
        elif method == "ping":
            send(handle_ping(request_id))
        else:
            send(make_error(request_id, -32601, "Method not found: {}".format(method)))


if __name__ == "__main__":
    main()
