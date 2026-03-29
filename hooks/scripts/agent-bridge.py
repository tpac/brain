#!/usr/bin/env python3
"""Bridge script for Stop agent hook → brain daemon.

The agent hook can't use MCP tools directly. But it CAN run Bash.
This script takes a daemon command as arguments and returns the result.

Usage from agent hook (via Bash tool):
  python3 hooks/scripts/agent-bridge.py ping
  python3 hooks/scripts/agent-bridge.py recall "query text here"
  python3 hooks/scripts/agent-bridge.py remember '{"type":"lesson","title":"...","content":"..."}'
  python3 hooks/scripts/agent-bridge.py eval 'brain.get_config("stop_agent_prompt")'
"""
import sys
import os
import socket
import json

PORT = 47200 + os.getuid() % 100
HOST = '127.0.0.1'
TIMEOUT = 10


def daemon_call(cmd, args=None):
    """Send a command to the brain daemon via TCP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(TIMEOUT)
    try:
        s.connect((HOST, PORT))
        payload = {'cmd': cmd}
        if args:
            payload['args'] = args
        s.sendall((json.dumps(payload) + '\n').encode())
        data = b''
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
            if b'\n' in data:
                break
        return json.loads(data.decode().strip())
    except Exception as e:
        return {'ok': False, 'error': str(e)}
    finally:
        s.close()


def main():
    if len(sys.argv) < 2:
        print(json.dumps({'ok': False, 'error': 'Usage: agent-bridge.py <command> [args]'}))
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'ping':
        result = daemon_call('ping')

    elif cmd == 'recall':
        query = sys.argv[2] if len(sys.argv) > 2 else ''
        result = daemon_call('recall', {'query': query, 'limit': 5})

    elif cmd == 'remember':
        args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        result = daemon_call('remember', args)

    elif cmd == 'revise':
        args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        result = daemon_call('revise', args)

    elif cmd == 'connect':
        args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        result = daemon_call('connect', args)

    elif cmd == 'find_node_by_title':
        query = sys.argv[2] if len(sys.argv) > 2 else ''
        result = daemon_call('find_node_by_title', {'title_query': query})

    elif cmd == 'get_node':
        node_id = sys.argv[2] if len(sys.argv) > 2 else ''
        result = daemon_call('get_node', {'node_id': node_id})

    elif cmd == 'eval':
        code = sys.argv[2] if len(sys.argv) > 2 else ''
        result = daemon_call('eval', {'code': code})

    elif cmd == 'record_divergence':
        args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        result = daemon_call('record_divergence', args)

    elif cmd == 'learn_vocabulary':
        args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        result = daemon_call('learn_vocabulary', args)

    else:
        # Generic passthrough
        args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        result = daemon_call(cmd, args)

    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
