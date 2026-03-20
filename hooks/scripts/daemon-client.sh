#!/bin/bash
# Shared daemon client for brain hooks.
# Source this after resolve-brain-db.sh to get daemon_send() function.
#
# Usage:
#   source "$(dirname "$0")/resolve-brain-db.sh"
#   source "$(dirname "$0")/daemon-client.sh"
#   RESULT=$(daemon_send '{"cmd":"recall","args":{"query":"test","limit":5}}')
#   if [ $? -eq 0 ]; then
#     echo "Daemon handled it: $RESULT"
#   else
#     echo "Fallback to direct Python"
#   fi

BRAIN_SOCKET_PATH="/tmp/brain-daemon-$(id -u).sock"

# daemon_send CMD_JSON [TIMEOUT]
# Sends a JSON command to the daemon via Python socket client.
# Returns 0 + prints response JSON on success, returns 1 on failure.
daemon_send() {
  local cmd_json="$1"
  local timeout="${2:-10}"

  # Quick check: socket file must exist
  [ -S "$BRAIN_SOCKET_PATH" ] || return 1

  python3 -c '
import json, socket, sys

sock_path = "'"$BRAIN_SOCKET_PATH"'"
timeout = float("'"$timeout"'")
cmd_json = '"'"''"$cmd_json"''"'"'

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(timeout)
try:
    sock.connect(sock_path)
    sock.sendall((cmd_json + "\n").encode())
    data = b""
    while b"\n" not in data:
        chunk = sock.recv(65536)
        if not chunk:
            break
        data += chunk
    if data:
        resp = json.loads(data.decode().strip())
        if resp.get("ok"):
            print(json.dumps(resp.get("result", {})))
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        sys.exit(1)
except Exception:
    sys.exit(1)
finally:
    sock.close()
' 2>/dev/null
  return $?
}

# daemon_available — quick check if daemon socket exists
daemon_available() {
  [ -S "$BRAIN_SOCKET_PATH" ]
}
