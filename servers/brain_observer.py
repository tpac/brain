"""
brain — Observer Channel

A lightweight TCP server that streams brain events to any connected listener.
Completely separated from brain logic — pure observation, zero impact.

If nobody's listening, events are silently dropped.
If a listener connects, they get real-time brain activity:
  - Recall results (what was surfaced to Claude)
  - Encoding checkpoints (heartbeat, mirror, stats)
  - Pre-edit surfaces (rules before file changes)
  - Remember/connect calls (what Claude encoded)
  - Consciousness signals
  - Health check results

Protocol: newline-delimited JSON events on TCP.
Port: DAEMON_PORT + 100 (e.g., 47303)

Usage:
  # Watch brain activity in terminal:
  nc localhost 47303

  # Or from Claude Code preview pane via launch.json
"""

import json
import socket
import threading
import time
import sys

from servers.daemon_config import DAEMON_HOST, DAEMON_PORT

OBSERVER_PORT = DAEMON_PORT + 100
_lock = threading.Lock()
_clients = []  # list of connected sockets
_server_socket = None
_running = False


def _accept_loop(server_sock):
    """Accept new observer connections."""
    global _running
    while _running:
        try:
            server_sock.settimeout(1.0)
            client, addr = server_sock.accept()
            client.settimeout(5.0)
            # Send welcome message
            welcome = {
                "type": "connected",
                "message": "Brain observer connected",
                "port": OBSERVER_PORT,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            try:
                client.sendall(json.dumps(welcome).encode() + b"\n")
            except Exception:
                pass
            with _lock:
                _clients.append(client)
            _log("Observer client connected from %s" % str(addr))
        except socket.timeout:
            continue
        except Exception:
            if _running:
                continue
            break


def start():
    """Start the observer server. Called by daemon on startup.
    Non-blocking — runs accept loop in background thread."""
    global _server_socket, _running

    if _running:
        return

    try:
        _server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _server_socket.bind((DAEMON_HOST, OBSERVER_PORT))
        _server_socket.listen(3)
        _running = True

        t = threading.Thread(target=_accept_loop, args=(_server_socket,),
                             daemon=True, name="observer-accept")
        t.start()
        _log("Observer channel listening on %s:%d" % (DAEMON_HOST, OBSERVER_PORT))
    except Exception as e:
        _log("Observer failed to start: %s" % e)
        _running = False


def stop():
    """Stop the observer server."""
    global _running, _server_socket
    _running = False
    with _lock:
        for c in _clients:
            try:
                c.close()
            except Exception:
                pass
        _clients.clear()
    if _server_socket:
        try:
            _server_socket.close()
        except Exception:
            pass
        _server_socket = None


def emit(event_type, data=None, **kwargs):
    """Emit an event to all connected observers.

    If nobody's listening, this is a no-op (zero overhead).

    Args:
        event_type: e.g. "recall", "checkpoint", "remember", "pre_edit", "mirror"
        data: dict of event data
        **kwargs: additional fields merged into the event
    """
    with _lock:
        if not _clients:
            return  # Nobody listening — zero overhead

    event = {
        "type": event_type,
        "timestamp": time.strftime("%H:%M:%S"),
    }
    if data:
        event.update(data)
    if kwargs:
        event.update(kwargs)

    payload = json.dumps(event, default=str, ensure_ascii=False) + "\n"
    payload_bytes = payload.encode("utf-8")

    dead = []
    with _lock:
        for i, client in enumerate(_clients):
            try:
                client.sendall(payload_bytes)
            except Exception:
                dead.append(i)
        # Remove dead clients
        for i in reversed(dead):
            try:
                _clients[i].close()
            except Exception:
                pass
            _clients.pop(i)

    if dead:
        _log("Removed %d dead observer client(s)" % len(dead))


def has_listeners():
    """Check if anyone is listening. Use to skip expensive formatting."""
    with _lock:
        return len(_clients) > 0


def _log(msg):
    """Internal logging to stderr."""
    print("[observer] %s" % msg, file=sys.stderr, flush=True)
