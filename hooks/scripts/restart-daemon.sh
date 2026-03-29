#!/bin/bash
# Restart the brain daemon — clears cache, re-execs with fresh code.
# Usage: hooks/scripts/restart-daemon.sh
source "$(dirname "$0")/resolve-brain-db.sh"
PORT=$((47200 + $(id -u) % 100))

python3 -c "
import socket, json, sys
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(3)
try:
    sock.connect(('127.0.0.1', $PORT))
    sock.sendall(json.dumps({'cmd': 'restart', 'args': {}}).encode() + b'\n')
    data = sock.recv(4096)
    sock.close()
    resp = json.loads(data.decode().strip())
    if resp.get('ok'):
        print('Restart sent. Daemon will re-exec in ~4s.')
    else:
        print('Restart failed: %s' % resp.get('error', '?'))
        sys.exit(1)
except Exception as e:
    print('Cannot reach daemon on port $PORT: %s' % e)
    sys.exit(1)
"
