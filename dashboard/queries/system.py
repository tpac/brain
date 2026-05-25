"""System status — live/dead checks across every brain component.

This file used to open `sqlite3.connect(...)` directly three times despite
`db.ro_connect` existing for exactly this purpose — a pre-refactor leftover
the disconnection contract test couldn't catch (the connects had `mode=ro`
so they were technically allowed; they just bypassed the central helper).
Now all DB reads route through ro_connect.
"""

import json
import os
import socket

from ..daemon_client import DAEMON_HOST, DAEMON_PORT
from ..db import brain_db_path, logs_db_path, ro_connect
from ..log import warn


def _check_daemon() -> dict:
    """TCP ping the daemon. Doesn't go through `daemon_send` because we want
    the raw {pid, uptime_seconds, code_fingerprint} payload, and we want a
    shorter 2s timeout (this is on the health refresh path)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((DAEMON_HOST, DAEMON_PORT))
        sock.sendall(b'{"cmd":"ping","args":{}}\n')
        data = sock.recv(4096)
        sock.close()
        resp = json.loads(data.decode().strip()) if data else {}
        if resp.get("ok"):
            result = resp.get("result", {})
            return {
                'alive': True, 'pid': result.get('pid', '?'),
                'uptime': result.get('uptime_seconds', 0),
                'code_fingerprint': result.get('code_fingerprint', '')[:12],
            }
        return {'alive': False, 'error': resp.get('error', 'bad response')}
    except Exception as e:
        return {'alive': False, 'error': str(e)[:100]}


def _check_brain_db() -> dict:
    """Brain DB liveness: open + count nodes + report size."""
    path = brain_db_path()
    try:
        with ro_connect(path, timeout=2) as conn:
            if conn is None:
                return {'alive': False, 'error': 'File not found'}
            count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        size_mb = round(os.path.getsize(path) / 1048576, 1)
        return {'alive': True, 'nodes': count, 'path': path, 'size_mb': size_mb}
    except Exception as e:
        warn('queries.system', 'brain_db check failed', exc=e)
        return {'alive': False, 'error': str(e)[:100]}


def _check_logs_db() -> dict:
    """Logs DB liveness: open + trivial SELECT + report size."""
    path = logs_db_path()
    try:
        with ro_connect(path, timeout=2) as conn:
            if conn is None:
                return {'alive': False, 'error': 'File not found'}
            conn.execute("SELECT 1").fetchone()
        size_mb = round(os.path.getsize(path) / 1048576, 1)
        return {'alive': True, 'path': path, 'size_mb': size_mb}
    except Exception as e:
        warn('queries.system', 'logs_db check failed', exc=e)
        return {'alive': False, 'error': str(e)[:100]}


def _check_judge() -> dict:
    """Haiku judge health: success rate over last 20 S1 K judge_selected events."""
    try:
        with ro_connect(logs_db_path(), timeout=2) as conn:
            if conn is None:
                return {'alive': False, 'error': 'logs DB unavailable'}
            rows = conn.execute(
                "SELECT id, summary, created_at FROM trace_events "
                "WHERE scale = 's1' AND event_type = 'K' AND ref_type = 'judge_selected' "
                "ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
        total = len(rows)
        with_selection = sum(
            1 for r in rows if 'selected' in (r[1] or '') and not r[1].startswith('0')
        )
        last_time = rows[0][2] if rows else 'never'
        rate = round(with_selection * 100 / total) if total else 0
        return {
            'alive': total > 0,
            'success_rate': '%d%%' % rate,
            'last_success': last_time,
            'sample': '%d/%d with selections' % (with_selection, total),
        }
    except Exception as e:
        warn('queries.system', 'judge health check failed', exc=e)
        return {'alive': False, 'error': str(e)[:100]}


def _check_embedder() -> dict:
    """Embedder status — read from /tmp status file written by the daemon."""
    try:
        status_path = "/tmp/brain-status-%d.json" % os.getuid()
        if not os.path.exists(status_path):
            return {'alive': False, 'error': 'No status file'}
        with open(status_path) as f:
            ds = json.load(f)
        return {
            'alive': ds.get('embedder_ready', False),
            'model': ds.get('model_name', '?'),
        }
    except Exception as e:
        warn('queries.system', 'embedder status read failed', exc=e)
        return {'alive': False, 'error': str(e)[:100]}


def check_system_status() -> dict:
    """Probe each component and return a dict of {component: {alive, ...}}.

    Each checker is its own function — adding a new component is one new
    `_check_X()` + one dict entry below. Previously this was a 100-line
    sequential script.
    """
    return {
        'daemon':   _check_daemon(),
        'brain_db': _check_brain_db(),
        'logs_db':  _check_logs_db(),
        'judge':    _check_judge(),
        'embedder': _check_embedder(),
    }
