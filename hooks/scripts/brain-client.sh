#!/bin/bash
# brain-client.sh — Thin client for the brain daemon (TCP)
#
# Usage: echo '{"cmd":"recall","args":{"query":"test"}}' | brain-client.sh
#   or:  brain-client.sh ping
#   or:  brain-client.sh recall '{"query":"test","limit":5}'

DAEMON_HOST="127.0.0.1"
DAEMON_PORT=$((47200 + $(id -u) % 100))

# ── Send command to daemon via TCP ──
_send_to_daemon() {
  local cmd="$1"
  local args="${2:-{}}"

  python3 -c "
import socket, json, sys
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(10.0)
try:
    sock.connect(('$DAEMON_HOST', $DAEMON_PORT))
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
  python3 -c "
import socket, sys
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect(('$DAEMON_HOST', $DAEMON_PORT))
    sock.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

# ── Main ──
if [ $# -ge 1 ]; then
  CMD="$1"
  ARGS="${2:-\{\}}"
  _send_to_daemon "$CMD" "$ARGS"
elif [ ! -t 0 ]; then
  INPUT=$(cat)
  CMD=$(echo "$INPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('cmd',''))" 2>/dev/null)
  ARGS=$(echo "$INPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get('args',{})))" 2>/dev/null)
  _send_to_daemon "$CMD" "$ARGS"
else
  echo '{"ok": false, "error": "No command provided"}'
fi
