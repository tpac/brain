#!/usr/bin/env python3
"""PostToolUse trace capture — Scale 0 raw tool results.

Thin client: reads stdin JSON, sends trace event to daemon via TCP.
Captures tool_name, tool_input summary, and tool output snippet.
"""
import sys
import json
import socket
import os

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

    # Build summary based on tool type
    if tool_name in ("Edit", "Write"):
        file_path = tool_input.get("file_path", "")
        old = (tool_input.get("old_string", "") or "")[:80]
        new = (tool_input.get("new_string", "") or tool_input.get("content", "") or "")[:80]
        summary = "%s: %s" % (tool_name, file_path)
        if old:
            summary += "\n  old: %s" % old
            summary += "\n  new: %s" % new
    elif tool_name == "Bash":
        cmd = (tool_input.get("command", "") or "")[:200]
        summary = "Bash: %s" % cmd
    elif tool_name == "Read":
        summary = "Read: %s" % tool_input.get("file_path", "")
    elif tool_name == "Glob":
        summary = "Glob: %s" % tool_input.get("pattern", "")
    elif tool_name == "Grep":
        summary = "Grep: %s in %s" % (tool_input.get("pattern", ""), tool_input.get("path", "."))
    elif tool_name == "Agent":
        summary = "Agent: %s" % (tool_input.get("description", "") or "")[:150]
    elif tool_name == "WebSearch":
        summary = "WebSearch: %s" % (tool_input.get("query", "") or "")[:150]
    elif tool_name == "WebFetch":
        summary = "WebFetch: %s" % (tool_input.get("url", "") or "")[:150]
    else:
        summary = "%s: %s" % (tool_name, json.dumps(tool_input)[:150])

    # Send to daemon
    msg = json.dumps({
        "cmd": "trace_append",
        "args": {
            "chain_id": "s0-%s-tool" % session_id[:8],
            "scale": "s0",
            "event_type": "delta",
            "ref_type": tool_name,
            "ref_id": tool_input.get("file_path", "") or tool_input.get("command", "")[:100] or "",
            "summary": summary[:500],
            "session_id": session_id,
        }
    })

    uid = os.getuid()
    port = 47200 + (uid % 100)
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
