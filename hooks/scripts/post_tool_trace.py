#!/usr/bin/env python3
"""PostToolUse trace capture — Scale 0 tool results.

Thin client: reads stdin JSON, sends trace event to daemon via TCP.
Attaches to the current stop's S0 chain (read from tmp file written by recall hook).
"""
import sys
import json
import socket
import os


def _build_summary(tool_name, tool_input):
    """Build a human-readable summary of the tool call."""
    if tool_name in ("Edit", "Write"):
        file_path = tool_input.get("file_path", "")
        old = (tool_input.get("old_string", "") or "")[:80]
        new = (tool_input.get("new_string", "") or tool_input.get("content", "") or "")[:80]
        s = "%s: %s" % (tool_name, file_path)
        if old:
            s += "\n  old: %s\n  new: %s" % (old, new)
        return s
    elif tool_name == "Bash":
        return "Bash: %s" % (tool_input.get("command", "") or "")[:200]
    elif tool_name == "Read":
        return "Read: %s" % tool_input.get("file_path", "")
    elif tool_name == "Glob":
        return "Glob: %s" % tool_input.get("pattern", "")
    elif tool_name == "Grep":
        return "Grep: %s in %s" % (tool_input.get("pattern", ""), tool_input.get("path", "."))
    elif tool_name == "Agent":
        return "Agent: %s" % (tool_input.get("description", "") or "")[:150]
    elif tool_name == "WebSearch":
        return "WebSearch: %s" % (tool_input.get("query", "") or "")[:150]
    elif tool_name == "WebFetch":
        return "WebFetch: %s" % (tool_input.get("url", "") or "")[:150]
    else:
        return "%s: %s" % (tool_name, json.dumps(tool_input)[:150])


def _read_current_stop(session_id):
    """Read current stop counter from tmp file written by recall hook."""
    try:
        path = "/tmp/brain-%s-current-stop.txt" % session_id
        if os.path.exists(path):
            with open(path) as f:
                return f.read().strip()
    except Exception:
        pass
    return "0"


def main():
    try:
        raw = sys.stdin.read()
        if not raw:
            return
        data = json.loads(raw)
    except Exception:
        return

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    session_id = data.get("session_id", "")

    summary = _build_summary(tool_name, tool_input)
    stop = _read_current_stop(session_id)

    # Attach to current stop's S0 chain
    chain_id = "s0-%s-%s" % (session_id[:8], stop)

    msg = json.dumps({
        "cmd": "trace_append",
        "args": {
            "chain_id": chain_id,
            "scale": "s0",
            "event_type": "delta",
            "ref_type": "tool_result",
            "summary": summary[:500],
            "metadata": json.dumps({"tool": tool_name}),
            "session_id": session_id,
        }
    })

    port = 47200 + (os.getuid() % 100)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(("127.0.0.1", port))
        sock.sendall((msg + "\n").encode())
        sock.recv(4096)
        sock.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
