#!/bin/bash
# brain-client.sh — Thin client for the brain daemon
#
# Usage: echo '{"cmd":"recall","args":{"query":"test"}}' | brain-client.sh
#   or:  brain-client.sh ping
#   or:  brain-client.sh recall '{"query":"test","limit":5}'
#
# Falls back to direct Python if daemon is not running.

SOCKET_PATH="/tmp/brain-daemon-$(id -u).sock"

# ── Send command to daemon via Python (most portable) ──
_send_to_daemon() {
  local cmd="$1"
  local args="${2:-{}}"

  python3 -c "
import socket, json, sys
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(10.0)
try:
    sock.connect('$SOCKET_PATH')
    msg = json.dumps({'cmd': '$cmd', 'args': $args}) + '\n'
    sock.sendall(msg.encode())
    data = b''
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        data += chunk
        if b'\n' in data:
            break
    print(data.decode().strip())
except Exception as e:
    print(json.dumps({'ok': False, 'error': str(e)}))
finally:
    sock.close()
" 2>/dev/null
}

# ── Check if daemon is running ──
is_daemon_running() {
  [ -S "$SOCKET_PATH" ]
}

# ── Main ──
if [ $# -ge 1 ]; then
  # Command-line mode: brain-client.sh <cmd> [<args-json>]
  CMD="$1"
  ARGS="${2:-\{\}}"
  _send_to_daemon "$CMD" "$ARGS"
elif [ ! -t 0 ]; then
  # Pipe mode: echo '{"cmd":"...","args":{}}' | brain-client.sh
  INPUT=$(cat)
  CMD=$(echo "$INPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('cmd',''))" 2>/dev/null)
  ARGS=$(echo "$INPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get('args',{})))" 2>/dev/null)
  _send_to_daemon "$CMD" "$ARGS"
else
  echo '{"ok": false, "error": "No command provided"}'
fi
