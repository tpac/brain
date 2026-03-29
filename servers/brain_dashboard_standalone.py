#!/usr/bin/env python3
"""
Standalone Brain Dashboard — completely independent from daemon.

Serves the dashboard HTML on port 47303. Queries the daemon for data via TCP.
If daemon is unavailable, shows a status message — doesn't crash.

Start: python3 servers/brain_dashboard_standalone.py
"""

import json
import os
import socket
import sqlite3
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

# ── Config ──
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", 47303))
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 47200 + (os.getuid() % 100)


def daemon_send(cmd, args=None, timeout=10):
    """Send a command to the daemon, return result or None."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((DAEMON_HOST, DAEMON_PORT))
        payload = json.dumps({"cmd": cmd, "args": args or {}}) + "\n"
        s.sendall(payload.encode("utf-8"))
        chunks = []
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            # Check if we have a complete JSON response
            try:
                json.loads(b"".join(chunks))
                break
            except json.JSONDecodeError:
                continue
        s.close()
        resp = json.loads(b"".join(chunks))
        if resp.get("ok"):
            return resp.get("result")
        return None
    except Exception:
        return None


def daemon_alive():
    """Quick check if daemon is responding."""
    result = daemon_send("ping", timeout=3)
    return result is not None


# ── SQLite direct read (fallback when daemon is down) ──
def _get_db_path():
    db_dir = os.environ.get("BRAIN_DB_DIR", os.path.join(os.path.expanduser("~"), "AgentsContext", "brain"))
    return os.path.join(db_dir, "brain.db")


def _direct_query(sql, args=(), db_path=None):
    """Direct read-only SQLite query — used when daemon is down."""
    import sqlite3
    path = db_path or _get_db_path()
    if not os.path.exists(path):
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)
        result = conn.execute(sql, args).fetchall()
        conn.close()
        return result
    except Exception:
        return []


# ── Hook Log — reads brain_dashboard.db for actual brain surface output ──

def _get_dashboard_db_path():
    db_dir = os.environ.get("BRAIN_DB_DIR", os.path.join(os.path.expanduser("~"), "AgentsContext", "brain"))
    return os.path.join(db_dir, "brain_dashboard.db")


def _query_hook_log(since_id=0, limit=50):
    """Read hook_log entries from brain_dashboard.db."""
    path = _get_dashboard_db_path()
    if not os.path.exists(path):
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)
        rows = conn.execute(
            "SELECT id, hook_name, timestamp, output_text, operator_text, metadata, session_id, user_prompt "
            "FROM hook_log WHERE id > ? ORDER BY id DESC LIMIT ?",
            (since_id, limit)
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "hook_name": r[1], "timestamp": r[2],
             "output_text": r[3] or "", "operator_text": r[4] or "",
             "metadata": r[5] or "", "session_id": r[6] or "",
             "user_prompt": r[7] if len(r) > 7 else ""}
            for r in rows
        ]
    except Exception:
        return []


def _query_encoding_activity(since_ts="", limit=30):
    """Read all encoding activity from brain.db — new nodes, revisions, connections, enrichments."""
    db = _get_db_path()
    if not os.path.exists(db):
        return []
    events = []
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
        where = "WHERE created_at > ?" if since_ts else "WHERE 1=1"
        args_base = (since_ts,) if since_ts else ()

        # New nodes
        rows = conn.execute(
            f"SELECT id, type, title, content, confidence, encoding_source, locked, created_at "
            f"FROM nodes {where} ORDER BY created_at DESC LIMIT ?",
            args_base + (limit,)).fetchall()
        for r in rows:
            events.append({
                "kind": "created", "id": r[0], "type": r[1], "title": r[2],
                "content": (r[3] or "")[:300], "confidence": r[4],
                "encoding_source": r[5], "locked": bool(r[6]), "timestamp": r[7]})

        # Revised nodes
        rows = conn.execute(
            "SELECT id, type, title, content, confidence, revised_at "
            "FROM nodes WHERE revised_at IS NOT NULL AND revised_at > ? "
            "ORDER BY revised_at DESC LIMIT ?",
            (since_ts or "1970-01-01", limit)).fetchall()
        for r in rows:
            events.append({
                "kind": "revised", "id": r[0], "type": r[1], "title": r[2],
                "content": (r[3] or "")[:300], "confidence": r[4], "timestamp": r[5]})

        # New connections (exclude co_accessed and emergent_bridge — organic noise)
        rows = conn.execute(
            f"SELECT e.source_id, e.target_id, e.relation, e.weight, e.created_at, "
            f"n1.title, n2.title "
            f"FROM edges e "
            f"LEFT JOIN nodes n1 ON n1.id = e.source_id "
            f"LEFT JOIN nodes n2 ON n2.id = e.target_id "
            f"{where.replace('created_at', 'e.created_at')} "
            f"AND e.relation NOT IN ('co_accessed', 'emergent_bridge') "
            f"ORDER BY e.created_at DESC LIMIT ?",
            args_base + (limit,)).fetchall()
        for r in rows:
            events.append({
                "kind": "connected", "source_title": r[5] or r[0][:12],
                "target_title": r[6] or r[1][:12], "relation": r[2],
                "weight": r[3], "timestamp": r[4]})

        # Enrichments
        rows = conn.execute(
            f"SELECT ne.node_id, ne.vector_type, ne.text, ne.created_at, n.title "
            f"FROM node_enrichments ne "
            f"LEFT JOIN nodes n ON n.id = ne.node_id "
            f"{where.replace('created_at', 'ne.created_at')} "
            f"ORDER BY ne.created_at DESC LIMIT ?",
            args_base + (limit,)).fetchall()
        for r in rows:
            events.append({
                "kind": "enriched", "node_title": r[4] or r[0][:12],
                "vector_type": r[1], "text": (r[2] or "")[:200], "timestamp": r[3]})

        conn.close()
        # Sort all by timestamp descending
        events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return events[:limit]
    except Exception:
        return []


def _get_logs_db_path():
    db_dir = os.environ.get("BRAIN_DB_DIR", os.path.join(os.path.expanduser("~"), "AgentsContext", "brain"))
    return os.path.join(db_dir, "brain_logs.db")


def _query_signal_queue():
    """Read signal_queue from brain_logs.db — all non-dismissed signals."""
    path = _get_logs_db_path()
    if not os.path.exists(path):
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)
        rows = conn.execute(
            "SELECT id, producer, signal_type, priority, content, content_chars, "
            "metadata, created_at, updated_at, ttl_seconds, times_surfaced, "
            "max_surfaces, last_surfaced_at, cooldown_seconds, preempt "
            "FROM signal_queue WHERE dismissed = 0 ORDER BY priority DESC"
        ).fetchall()
        conn.close()
        return [{
            'id': r[0], 'producer': r[1], 'signal_type': r[2],
            'priority': r[3], 'content': r[4], 'content_chars': r[5],
            'metadata': r[6], 'created_at': r[7], 'updated_at': r[8],
            'ttl_seconds': r[9], 'times_surfaced': r[10],
            'max_surfaces': r[11], 'last_surfaced_at': r[12],
            'cooldown_seconds': r[13], 'preempt': bool(r[14]),
        } for r in rows]
    except Exception:
        return []


def _query_assembler_comparison(limit=20):
    """Read assembler comparison log from brain_dashboard.db."""
    path = _get_dashboard_db_path()
    if not os.path.exists(path):
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)
        rows = conn.execute(
            "SELECT id, timestamp, user_prompt, old_chars, new_chars, new_output, stats "
            "FROM assembler_comparison ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [{
            'id': r[0], 'timestamp': r[1], 'user_prompt': r[2],
            'old_chars': r[3], 'new_chars': r[4],
            'new_output': r[5], 'stats': r[6],
        } for r in rows]
    except Exception:
        return []


# ── Unified Errors — aggregates errors from all system components ──

def _query_all_errors(limit=50, hours=24):
    """Read errors from all sources into a unified list."""
    errors = []
    logs_path = _get_logs_db_path()
    dash_path = _get_dashboard_db_path()
    since = "datetime('now', '-%d hours')" % hours

    # 1. Brain internal errors (debug_log where event_type='error')
    if os.path.exists(logs_path):
        try:
            conn = sqlite3.connect(f"file:{logs_path}?mode=ro", uri=True, timeout=3)
            rows = conn.execute(
                "SELECT id, created_at, source, metadata FROM debug_log "
                "WHERE event_type='error' AND created_at > %s "
                "ORDER BY created_at DESC LIMIT ?" % since, (limit,)).fetchall()
            for r in rows:
                meta = {}
                try:
                    meta = json.loads(r[3]) if r[3] else {}
                except Exception:
                    pass
                errors.append({
                    'source': 'brain', 'component': r[2], 'timestamp': r[1],
                    'error': meta.get('error', r[3] or '')[:200],
                    'context': meta.get('context', '')[:100],
                    'level': 'error'})
            conn.close()
        except Exception:
            pass

    # 2. Hook errors
    if os.path.exists(logs_path):
        try:
            conn = sqlite3.connect(f"file:{logs_path}?mode=ro", uri=True, timeout=3)
            rows = conn.execute(
                "SELECT id, created_at, hook_name, level, error, context FROM hook_errors "
                "WHERE created_at > %s ORDER BY created_at DESC LIMIT ?" % since,
                (limit,)).fetchall()
            for r in rows:
                errors.append({
                    'source': 'hook', 'component': r[2], 'timestamp': r[1],
                    'error': (r[4] or '')[:200], 'context': (r[5] or '')[:100],
                    'level': r[3] or 'error'})
            conn.close()
        except Exception:
            pass

    # 3. Conflicts
    if os.path.exists(logs_path):
        try:
            conn = sqlite3.connect(f"file:{logs_path}?mode=ro", uri=True, timeout=3)
            rows = conn.execute(
                "SELECT id, created_at, hook_name, rule_title, brain_decision, resolution "
                "FROM conflict_log WHERE created_at > %s "
                "ORDER BY created_at DESC LIMIT ?" % since, (limit,)).fetchall()
            for r in rows:
                errors.append({
                    'source': 'conflict', 'component': r[2], 'timestamp': r[1],
                    'error': 'Rule: %s — Decision: %s' % (r[3] or '?', r[4] or '?'),
                    'context': 'Resolution: %s' % (r[5] or 'pending'),
                    'level': 'warning'})
            conn.close()
        except Exception:
            pass

    # 4. Daemon down events (from dashboard)
    if os.path.exists(dash_path):
        try:
            conn = sqlite3.connect(f"file:{dash_path}?mode=ro", uri=True, timeout=3)
            rows = conn.execute(
                "SELECT id, timestamp, output_text FROM hook_log "
                "WHERE hook_name='DAEMON_DOWN' ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()
            for r in rows:
                errors.append({
                    'source': 'daemon', 'component': 'daemon_down', 'timestamp': r[1],
                    'error': (r[2] or '')[:200], 'context': '',
                    'level': 'critical'})
            conn.close()
        except Exception:
            pass

    # 5. Telemetry failures
    if os.path.exists(logs_path):
        try:
            conn = sqlite3.connect(f"file:{logs_path}?mode=ro", uri=True, timeout=3)
            rows = conn.execute(
                "SELECT id, timestamp, operation, error_message FROM brain_telemetry "
                "WHERE success=0 AND timestamp > %s "
                "ORDER BY timestamp DESC LIMIT ?" % since, (limit,)).fetchall()
            for r in rows:
                errors.append({
                    'source': 'telemetry', 'component': r[2], 'timestamp': r[1],
                    'error': (r[3] or '')[:200], 'context': '',
                    'level': 'warning'})
            conn.close()
        except Exception:
            pass

    # Sort by timestamp descending
    errors.sort(key=lambda e: e.get('timestamp', ''), reverse=True)
    return errors[:limit]


# ── System Status — live/dead check for all components ──

def _check_system_status():
    """Check health of all system components."""
    import socket as _socket
    status = {}

    # 1. Daemon — TCP ping
    try:
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.settimeout(2.0)
        port = 47200 + (os.getuid() % 100)
        sock.connect(("127.0.0.1", port))
        sock.sendall(b'{"cmd":"ping","args":{}}\n')
        data = sock.recv(4096)
        sock.close()
        resp = json.loads(data.decode().strip()) if data else {}
        if resp.get("ok"):
            result = resp.get("result", {})
            status['daemon'] = {
                'alive': True, 'pid': result.get('pid', '?'),
                'uptime': result.get('uptime_seconds', 0),
                'code_fingerprint': result.get('code_fingerprint', '')[:12]}
        else:
            status['daemon'] = {'alive': False, 'error': resp.get('error', 'bad response')}
    except Exception as e:
        status['daemon'] = {'alive': False, 'error': str(e)[:100]}

    # 2. Brain DB — file exists and readable
    brain_path = _get_db_path()
    try:
        if os.path.exists(brain_path):
            conn = sqlite3.connect(f"file:{brain_path}?mode=ro", uri=True, timeout=2)
            count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            conn.close()
            size_mb = round(os.path.getsize(brain_path) / 1048576, 1)
            status['brain_db'] = {'alive': True, 'nodes': count, 'path': brain_path, 'size_mb': size_mb}
        else:
            status['brain_db'] = {'alive': False, 'error': 'File not found'}
    except Exception as e:
        status['brain_db'] = {'alive': False, 'error': str(e)[:100]}

    # 3. Logs DB
    logs_path = _get_logs_db_path()
    try:
        if os.path.exists(logs_path):
            conn = sqlite3.connect(f"file:{logs_path}?mode=ro", uri=True, timeout=2)
            conn.execute("SELECT 1").fetchone()
            conn.close()
            size_mb = round(os.path.getsize(logs_path) / 1048576, 1)
            status['logs_db'] = {'alive': True, 'path': logs_path, 'size_mb': size_mb}
        else:
            status['logs_db'] = {'alive': False, 'error': 'File not found'}
    except Exception as e:
        status['logs_db'] = {'alive': False, 'error': str(e)[:100]}

    # 4. Dashboard DB
    dash_path = _get_dashboard_db_path()
    try:
        if os.path.exists(dash_path):
            conn = sqlite3.connect(f"file:{dash_path}?mode=ro", uri=True, timeout=2)
            last_entry = conn.execute("SELECT timestamp FROM hook_log ORDER BY id DESC LIMIT 1").fetchone()
            conn.close()
            size_mb = round(os.path.getsize(dash_path) / 1048576, 1)
            status['dashboard_db'] = {
                'alive': True, 'path': dash_path, 'size_mb': size_mb,
                'last_entry': last_entry[0] if last_entry else 'empty'}
        else:
            status['dashboard_db'] = {'alive': False, 'error': 'File not found'}
    except Exception as e:
        status['dashboard_db'] = {'alive': False, 'error': str(e)[:100]}

    # 5. Embedder — check via daemon status file
    try:
        status_path = "/tmp/brain-status-%d.json" % os.getuid()
        if os.path.exists(status_path):
            with open(status_path) as f:
                ds = json.load(f)
            status['embedder'] = {
                'alive': ds.get('embedder_ready', False),
                'model': ds.get('model_name', '?')}
        else:
            status['embedder'] = {'alive': False, 'error': 'No status file'}
    except Exception as e:
        status['embedder'] = {'alive': False, 'error': str(e)[:100]}

    # 6. Signal queue — count pending
    try:
        conn = sqlite3.connect(f"file:{logs_path}?mode=ro", uri=True, timeout=2)
        pending = conn.execute(
            "SELECT COUNT(*) FROM signal_queue WHERE dismissed=0").fetchone()[0]
        preempt = conn.execute(
            "SELECT COUNT(*) FROM signal_queue WHERE dismissed=0 AND preempt=1").fetchone()[0]
        conn.close()
        status['signal_queue'] = {'alive': True, 'pending': pending, 'preempt': preempt}
    except Exception as e:
        status['signal_queue'] = {'alive': False, 'error': str(e)[:100]}

    return status


# ── HTTP Server ──
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class DashboardHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._serve_html()
        elif path == "/api/stats":
            self._serve_stats()
        elif path == "/api/nodes":
            self._serve_nodes(params)
        elif path == "/api/graph":
            self._serve_graph(params)
        elif path == "/api/insights":
            self._serve_insights()
        elif path == "/api/status":
            self._serve_status()
        elif path == "/api/hook-log":
            self._serve_hook_log(params)
        elif path == "/api/encoding-activity":
            self._serve_encoding_activity(params)
        elif path == "/api/signal-queue":
            self._serve_signal_queue()
        elif path == "/api/assembler-comparison":
            self._serve_assembler_comparison(params)
        elif path == "/api/errors":
            self._serve_errors(params)
        elif path == "/api/system-status":
            self._serve_system_status()
        elif path.startswith("/api/node/"):
            node_id = path.split("/api/node/")[1]
            self._serve_node_detail(node_id)
        else:
            self._json_response(404, {"error": "Not found"})

    def _json_response(self, code, data):
        body = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_status(self):
        alive = daemon_alive()
        self._json_response(200, {
            "daemon": "alive" if alive else "unavailable",
            "dashboard": "running",
            "daemon_port": DAEMON_PORT,
        })

    def _serve_hook_log(self, params):
        """Return recent hook log entries — the actual brain surface output."""
        since_id = int(params.get("since_id", [0])[0])
        limit = int(params.get("limit", [50])[0])
        entries = _query_hook_log(since_id=since_id, limit=limit)
        latest_id = entries[0]["id"] if entries else since_id
        self._json_response(200, {"events": entries, "latest_id": latest_id})

    def _serve_encoding_activity(self, params):
        """Return recent encoding activity — nodes created, revised, connected, enriched."""
        since_ts = params.get("since", [""])[0]
        limit = int(params.get("limit", [30])[0])
        events = _query_encoding_activity(since_ts=since_ts, limit=limit)
        self._json_response(200, {"events": events})

    def _serve_signal_queue(self):
        """Return current signal queue state."""
        signals = _query_signal_queue()
        self._json_response(200, {"signals": signals})

    def _serve_assembler_comparison(self, params):
        """Return assembler vs old output comparison."""
        limit = int(params.get("limit", [20])[0])
        comparisons = _query_assembler_comparison(limit=limit)
        self._json_response(200, {"comparisons": comparisons})

    def _serve_errors(self, params):
        """Return unified errors from all system components."""
        hours = int(params.get("hours", [24])[0])
        limit = int(params.get("limit", [50])[0])
        errors = _query_all_errors(limit=limit, hours=hours)
        self._json_response(200, {"errors": errors, "count": len(errors)})

    def _serve_system_status(self):
        """Return live/dead status of all system components."""
        status = _check_system_status()
        self._json_response(200, {"status": status})

    def _serve_node_detail(self, node_id):
        """Lazy-loaded node detail: full content + connected nodes."""
        try:
            db = _get_db_path()
            # Node data
            row = _direct_query(
                "SELECT id, type, title, content, keywords, locked, emotion, "
                "access_count, confidence, encoding_source, created_at, last_accessed "
                "FROM nodes WHERE id = ?",
                args=(node_id,), db_path=db)
            if not row:
                return self._json_response(404, {"error": "Node not found"})
            r = row[0]
            node = {
                "id": r[0], "type": r[1], "title": r[2], "content": r[3],
                "keywords": r[4], "locked": bool(r[5]), "emotion": r[6],
                "access_count": r[7], "confidence": r[8], "encoding_source": r[9],
                "created_at": r[10], "last_accessed": r[11],
            }
            # Connected nodes
            edges = _direct_query(
                "SELECT e.target_id, e.relation, e.weight, n.type, n.title "
                "FROM edges e JOIN nodes n ON n.id = e.target_id "
                "WHERE e.source_id = ? ORDER BY e.weight DESC LIMIT 20",
                args=(node_id,), db_path=db)
            connections = [
                {"id": e[0], "relation": e[1], "weight": e[2],
                 "type": e[3], "title": e[4]}
                for e in edges
            ]
            self._json_response(200, {"node": node, "connections": connections})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _serve_stats(self):
        # Try direct SQLite read — works whether daemon is up or not
        try:
            db = _get_db_path()
            nodes = _direct_query("SELECT COUNT(*) FROM nodes WHERE archived = 0", db_path=db)
            edges = _direct_query("SELECT COUNT(*) FROM edges", db_path=db)
            locked = _direct_query("SELECT COUNT(*) FROM nodes WHERE locked = 1", db_path=db)
            types = _direct_query(
                "SELECT type, COUNT(*) FROM nodes WHERE archived = 0 GROUP BY type ORDER BY COUNT(*) DESC",
                db_path=db
            )
            recent = _direct_query(
                "SELECT COUNT(*) FROM nodes WHERE created_at > datetime('now', '-24 hours')",
                db_path=db
            )
            orphans = _direct_query("""
                SELECT COUNT(*) FROM nodes n WHERE archived = 0
                AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.source_id = n.id OR e.target_id = n.id)
            """, db_path=db)

            self._json_response(200, {
                "nodes": nodes[0][0] if nodes else 0,
                "edges": edges[0][0] if edges else 0,
                "locked": locked[0][0] if locked else 0,
                "recent_24h": recent[0][0] if recent else 0,
                "orphans": orphans[0][0] if orphans else 0,
                "types": {t: cnt for t, cnt in types},
                "daemon": "alive" if daemon_alive() else "unavailable",
            })
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _serve_nodes(self, params):
        try:
            db = _get_db_path()
            limit = int(params.get("limit", [50])[0])
            node_type = params.get("type", [None])[0]
            search = params.get("search", [None])[0]

            sql = "SELECT id, type, title, content, keywords, locked, emotion, access_count, created_at FROM nodes WHERE archived = 0"
            args = []
            if node_type:
                sql += " AND type = ?"
                args.append(node_type)
            if search:
                sql += " AND (title LIKE ? OR content LIKE ? OR keywords LIKE ?)"
                pat = "%%%s%%" % search
                args.extend([pat, pat, pat])
            sql += " ORDER BY created_at DESC LIMIT ?"
            args.append(limit)

            rows = _direct_query(sql, args, db_path=db)
            nodes = []
            for r in rows:
                nodes.append({
                    "id": r[0], "type": r[1], "title": r[2],
                    "content": r[3][:500] if r[3] else "",
                    "keywords": r[4], "locked": bool(r[5]),
                    "emotion": r[6], "access_count": r[7], "created_at": r[8],
                })
            self._json_response(200, {"nodes": nodes, "total": len(nodes)})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _serve_graph(self, params):
        try:
            db = _get_db_path()
            limit = int(params.get("limit", [80])[0])
            days = float(params.get("days", [30])[0])
            source = params.get("source", [None])[0]

            # Convert fractional days to minutes for SQLite
            minutes = int(days * 24 * 60)
            if minutes < 1:
                minutes = 5

            args = []
            nodes_sql = """
                SELECT id, type, title, locked, emotion, access_count, created_at
                FROM nodes WHERE archived = 0
                AND REPLACE(REPLACE(created_at, 'T', ' '), 'Z', '') > datetime('now', '-%d minutes')
            """ % minutes
            if source:
                nodes_sql += " AND encoding_source = ?"
                args.append(source)
            nodes_sql += " ORDER BY access_count DESC LIMIT ?"
            args.append(limit)
            rows = _direct_query(nodes_sql, tuple(args), db_path=db)
            node_ids = set()
            nodes = []
            for r in rows:
                node_ids.add(r[0])
                nodes.append({
                    "id": r[0], "type": r[1], "title": r[2][:60],
                    "locked": bool(r[3]), "emotion": r[4] or 0,
                    "access_count": r[5], "created_at": r[6],
                })

            edges = []
            if node_ids:
                placeholders = ",".join("?" * len(node_ids))
                edges_sql = """
                    SELECT source_id, target_id, relation, weight
                    FROM edges
                    WHERE source_id IN (%s) AND target_id IN (%s)
                """ % (placeholders, placeholders)
                id_list = list(node_ids)
                edge_rows = _direct_query(edges_sql, id_list + id_list, db_path=db)
                edges = [{"source": r[0], "target": r[1], "relation": r[2], "weight": r[3]}
                         for r in edge_rows]

            self._json_response(200, {"nodes": nodes, "edges": edges})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _serve_insights(self):
        try:
            db = _get_db_path()
            insights = []

            # Orphan locked nodes
            orphan_locked = _direct_query("""
                SELECT id, title, type, created_at FROM nodes
                WHERE locked = 1 AND archived = 0
                AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.source_id = nodes.id OR e.target_id = nodes.id)
            """, db_path=db)
            if orphan_locked:
                insights.append({
                    "severity": "high", "icon": "\U0001f512",
                    "title": "%d locked nodes are orphaned" % len(orphan_locked),
                    "detail": "Important memories disconnected from everything. Recall can't find them through graph traversal.",
                    "nodes": [{"id": r[0], "title": r[1], "type": r[2]} for r in orphan_locked],
                })

            # Thin nodes
            thin = _direct_query("""
                SELECT COUNT(*), AVG(LENGTH(content)) FROM nodes
                WHERE archived = 0 AND LENGTH(content) < 100
                AND created_at > datetime('now', '-7 days')
            """, db_path=db)
            if thin and thin[0][0] > 5:
                insights.append({
                    "severity": "medium", "icon": "\U0001f4cf",
                    "title": "%d thin nodes this week (avg %d chars)" % (thin[0][0], thin[0][1] or 0),
                    "detail": "Nodes under 100 chars lack context for future recall.",
                })

            # Precision loop health
            try:
                db_dir = os.path.dirname(db)
                logs_db = os.path.join(db_dir, "brain_logs.db")
                total = _direct_query("SELECT COUNT(*) FROM recall_log", db_path=logs_db)
                evaluated = _direct_query("SELECT COUNT(*) FROM recall_log WHERE precision_score IS NOT NULL", db_path=logs_db)
                total_n = total[0][0] if total else 0
                eval_n = evaluated[0][0] if evaluated else 0
                eval_pct = (eval_n / total_n * 100) if total_n > 0 else 0
                if eval_pct < 10 and total_n > 0:
                    insights.append({
                        "severity": "high", "icon": "\U0001f4ca",
                        "title": "Precision loop at %.1f%% (%d/%d evaluated)" % (eval_pct, eval_n, total_n),
                        "detail": "The brain can't learn which recalls help. The feedback loop is starving.",
                    })
            except Exception:
                pass

            # Zero quotes
            quotes = _direct_query("""
                SELECT COUNT(*) FROM nodes WHERE archived = 0
                AND created_at > datetime('now', '-7 days')
                AND (content LIKE '%Tom said%' OR content LIKE '%Tom:%'
                     OR content LIKE '%Claude:%' OR title LIKE '%quote%')
            """, db_path=db)
            types = dict(_direct_query(
                "SELECT type, COUNT(*) FROM nodes WHERE archived = 0 AND created_at > datetime('now', '-7 days') GROUP BY type",
                db_path=db
            ))
            total_recent = sum(types.values())
            if quotes and quotes[0][0] == 0 and total_recent > 5:
                insights.append({
                    "severity": "high", "icon": "\U0001f4ad",
                    "title": "Zero quotes preserved this week",
                    "detail": "Tom's exact words and Claude's own insights weren't captured.",
                })

            # Daemon status
            if not daemon_alive():
                insights.insert(0, {
                    "severity": "high", "icon": "\u26a0\ufe0f",
                    "title": "Daemon is not running",
                    "detail": "Brain daemon on port %d is not responding. Dashboard is showing read-only data from SQLite directly. Live features (SSE events, recall, encoding) are unavailable." % DAEMON_PORT,
                })

            self._json_response(200, {"insights": insights, "checked_at": time.strftime("%H:%M:%S")})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _serve_html(self):
        html = _build_dashboard_html()
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


def _build_dashboard_html():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brain Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a0f; color: #e0e0e0; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; overflow: hidden; height: 100vh; }
.tabs { display: flex; background: #111118; border-bottom: 1px solid #2a2a3a; }
.tab { padding: 10px 20px; cursor: pointer; color: #888; border-bottom: 2px solid transparent; transition: all 0.2s; }
.tab:hover { color: #ccc; }
.tab.active { color: #7eb8ff; border-bottom-color: #7eb8ff; }
.tab-content { display: none; height: calc(100vh - 42px); overflow: auto; }
.tab-content.active { display: block; }
.stats-bar { display: flex; gap: 16px; padding: 12px 16px; background: #111118; border-bottom: 1px solid #1a1a2a; flex-wrap: wrap; align-items: center; }
.stat { display: flex; flex-direction: column; align-items: center; min-width: 70px; }
.stat-value { font-size: 22px; font-weight: bold; color: #7eb8ff; }
.stat-label { font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 1px; }
.daemon-status { margin-left: auto; padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: bold; }
.daemon-status.alive { background: #1a3a1a; color: #33ff88; }
.daemon-status.unavailable { background: #3a1a1a; color: #ff6666; }
.feed { padding: 8px; }
.hook-entry { margin: 6px 0; border-radius: 6px; border-left: 3px solid #333; background: #111118; overflow: hidden; }
.hook-entry.boot { border-left-color: #ffaa33; }
.hook-entry.recall { border-left-color: #33ff88; }
.hook-entry.stop { border-left-color: #aa66ff; }
.hook-header { padding: 8px 12px; display: flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; }
.hook-header:hover { background: #1a1a2a; }
.hook-header .hook-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: bold; text-transform: uppercase; }
.hook-badge.boot { background: #3a2a1a; color: #ffaa33; }
.hook-badge.recall { background: #1a3a1a; color: #33ff88; }
.hook-badge.stop { background: #2a1a3a; color: #aa66ff; }
.hook-header .hook-time { color: #555; font-size: 11px; }
.hook-header .hook-session { color: #7eb8ff; font-size: 10px; font-family: monospace; background: #1a1a2a; padding: 1px 4px; border-radius: 3px; }
.hook-header .hook-id { color: #555; font-size: 10px; font-family: monospace; }
.hook-header .hook-size { color: #444; font-size: 10px; margin-left: auto; }
.hook-body { display: none; padding: 0 12px 10px; }
.hook-body.open { display: block; }
.hook-details-btn { background: #1a1a2a; border: 1px solid #2a2a3a; color: #7eb8ff; padding: 3px 10px; border-radius: 3px; font-size: 10px; cursor: pointer; margin-top: 6px; }
.hook-details-btn:hover { background: #2a2a4a; }
.hook-details { display: none; margin-top: 6px; }
.hook-details.open { display: block; }
.hook-details pre { background: #050510; border: 1px solid #1a1a3a; border-radius: 4px; padding: 10px; color: #998; font-size: 10px; line-height: 1.4; white-space: pre-wrap; word-break: break-word; max-height: 600px; overflow-y: auto; }
.hook-prompt { padding: 6px 12px; background: #0d1117; border-left: 3px solid #58a6ff; color: #c9d1d9; font-size: 12px; margin: 0 8px; font-style: italic; }
.hook-body pre { background: #0a0a12; border: 1px solid #1a1a2a; border-radius: 4px; padding: 10px; color: #bbb; font-size: 11px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; max-height: 500px; overflow-y: auto; }
.feed-toggle { display: flex; gap: 0; padding: 0 8px; margin-top: 4px; }
.feed-btn { background: #111118; border: 1px solid #2a2a3a; color: #666; padding: 6px 16px; cursor: pointer; font-family: inherit; font-size: 11px; transition: all 0.15s; }
.feed-btn:first-child { border-radius: 4px 0 0 4px; }
.feed-btn:last-child { border-radius: 0 4px 4px 0; border-left: none; }
.feed-btn.active { background: #1a1a2a; color: #7eb8ff; border-color: #3a3a5a; }
.enc-entry { padding: 8px 12px; margin: 4px 0; background: #111118; border-radius: 6px; border-left: 3px solid #333; font-size: 12px; }
.enc-entry.created { border-left-color: #33ff88; }
.enc-entry.revised { border-left-color: #ffaa33; }
.enc-entry.connected { border-left-color: #aa66ff; }
.enc-entry.enriched { border-left-color: #4a9eff; }
.enc-kind { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; text-transform: uppercase; margin-right: 6px; }
.enc-kind.created { background: #1a3a1a; color: #33ff88; }
.enc-kind.revised { background: #3a2a1a; color: #ffaa33; }
.enc-kind.connected { background: #2a1a3a; color: #aa66ff; }
.enc-kind.enriched { background: #1a2a4a; color: #4a9eff; }
.enc-title { color: #ccc; font-weight: bold; }
.enc-meta { color: #555; font-size: 10px; margin-top: 3px; }
.enc-content { color: #888; font-size: 11px; margin-top: 4px; max-height: 60px; overflow: hidden; white-space: pre-wrap; }
.explorer { padding: 12px; }
.search-bar { display: flex; gap: 8px; margin-bottom: 12px; }
.search-bar input { flex: 1; background: #1a1a2a; border: 1px solid #2a2a3a; color: #e0e0e0; padding: 8px 12px; border-radius: 4px; font-family: inherit; font-size: 13px; }
.search-bar select { background: #1a1a2a; border: 1px solid #2a2a3a; color: #e0e0e0; padding: 8px; border-radius: 4px; font-family: inherit; }
.node-card { padding: 10px 12px; margin: 4px 0; background: #111118; border-radius: 6px; border-left: 3px solid #333; cursor: pointer; transition: background 0.15s; }
.node-card:hover { background: #1a1a2a; }
.node-card .node-title { font-weight: bold; color: #ccc; margin-bottom: 4px; }
.node-card .node-meta { font-size: 11px; color: #666; display: flex; gap: 12px; }
.node-card .node-content { font-size: 11px; color: #888; margin-top: 6px; max-height: 60px; overflow: hidden; }
.node-card.expanded .node-content { max-height: none; }
.type-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }
.type-lesson { background: #1a2a4a; color: #4a9eff; }
.type-correction { background: #4a1a1a; color: #ff6666; }
.type-interaction { background: #1a4a2a; color: #33ff88; }
.type-rule { background: #4a3a1a; color: #ffaa33; }
.type-decision { background: #3a1a4a; color: #aa66ff; }
.type-mental_model { background: #1a3a3a; color: #33dddd; }
.type-mechanism { background: #3a3a1a; color: #dddd33; }
.type-vocabulary { background: #2a2a2a; color: #999; }
.type-context { background: #2a2a2a; color: #888; }
.type-bug_lesson { background: #4a1a1a; color: #ff8866; }
.locked-icon { color: #ffaa33; margin-left: 4px; }
.graph-container { position: relative; height: calc(100vh - 42px); }
.graph-controls { position: absolute; top: 10px; left: 10px; z-index: 10; display: flex; gap: 6px; flex-wrap: wrap; }
.graph-controls button, .graph-controls select { background: #1a1a2acc; border: 1px solid #2a2a3a; color: #ccc; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 11px; backdrop-filter: blur(4px); }
.graph-controls button:hover { background: #2a2a4a; }
canvas { width: 100%; height: 100%; }
.node-tooltip { position: absolute; background: #1a1a2aee; border: 1px solid #3a3a5a; padding: 10px; border-radius: 6px; max-width: 300px; font-size: 11px; pointer-events: none; display: none; z-index: 20; backdrop-filter: blur(8px); }
.node-detail { position: absolute; top: 0; right: 0; width: 380px; height: 100%; background: #0d0d15f0; border-left: 1px solid #2a2a3a; padding: 16px; overflow-y: auto; z-index: 15; backdrop-filter: blur(12px); font-size: 12px; }
.node-detail .nd-close { position: absolute; top: 8px; right: 12px; cursor: pointer; color: #666; font-size: 18px; }
.node-detail .nd-close:hover { color: #fff; }
.node-detail .nd-title { font-weight: bold; color: #fff; font-size: 14px; margin-bottom: 8px; padding-right: 24px; }
.node-detail .nd-meta { color: #666; font-size: 11px; margin-bottom: 12px; display: flex; flex-wrap: wrap; gap: 8px; }
.node-detail .nd-content { color: #bbb; white-space: pre-wrap; margin-bottom: 16px; line-height: 1.5; max-height: 300px; overflow-y: auto; border: 1px solid #1a1a2a; border-radius: 4px; padding: 10px; background: #0a0a12; }
.node-detail .nd-section { color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin: 12px 0 6px; }
.node-detail .nd-conn { padding: 6px 8px; margin: 3px 0; background: #111118; border-radius: 4px; border-left: 2px solid #333; cursor: pointer; }
.node-detail .nd-conn:hover { background: #1a1a2a; }
.node-detail .nd-conn-title { color: #ccc; font-size: 11px; }
.node-detail .nd-conn-meta { color: #555; font-size: 10px; }
.node-tooltip .tt-title { font-weight: bold; color: #fff; margin-bottom: 4px; }
.node-tooltip .tt-type { font-size: 10px; color: #888; }
.health { padding: 12px; }
.health-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; }
.health-card { background: #111118; border-radius: 8px; padding: 16px; border: 1px solid #1a1a2a; }
.health-card .hc-value { font-size: 28px; font-weight: bold; }
.health-card .hc-label { font-size: 11px; color: #666; margin-top: 4px; }
.health-card.ok .hc-value { color: #33ff88; }
.health-card.warn .hc-value { color: #ffaa33; }
.health-card.bad .hc-value { color: #ff6666; }
.no-daemon-banner { background: #3a1a1a; border: 1px solid #ff6666; color: #ff9999; padding: 10px 16px; font-size: 12px; text-align: center; }
</style>
</head>
<body>

<div class="tabs">
  <div class="tab active" onclick="switchTab('live')">Live</div>
  <div class="tab" onclick="switchTab('graph')">Graph</div>
  <div class="tab" onclick="switchTab('explorer')">Explorer</div>
  <div class="tab" onclick="switchTab('errors')">Errors</div>
  <div class="tab" onclick="switchTab('status')">Status</div>
  <div class="tab" onclick="switchTab('health')">Health</div>
</div>

<div id="tab-live" class="tab-content active">
  <div class="stats-bar" id="stats-bar"></div>
  <div id="daemon-banner"></div>
  <div class="feed-toggle">
    <button class="feed-btn active" onclick="switchFeed('surface')">Surface</button>
    <button class="feed-btn" onclick="switchFeed('encoding')">Encoding</button>
    <button class="feed-btn" onclick="switchFeed('queue')">Queue</button>
  </div>
  <div class="feed" id="feed"></div>
  <div class="feed" id="feed-encoding" style="display:none"></div>
  <div class="feed" id="feed-queue" style="display:none"></div>
</div>

<div id="tab-graph" class="tab-content">
  <div class="graph-container">
    <div class="graph-controls">
      <select id="graph-days" onchange="loadGraph()">
        <option value="0.003" selected>Last 5 min</option>
        <option value="0.02">Last 30 min</option>
        <option value="0.04">Last 1 hour</option>
        <option value="0.25">Last 6 hours</option>
        <option value="0.5">Last 12 hours</option>
        <option value="1">Last 1 day</option>
        <option value="7">Last 7 days</option>
        <option value="30">Last 30 days</option>
        <option value="365">All time</option>
      </select>
      <select id="graph-limit" onchange="loadGraph()">
        <option value="40">40 nodes</option>
        <option value="80" selected>80 nodes</option>
        <option value="150">150 nodes</option>
        <option value="300">300 nodes</option>
      </select>
      <select id="graph-source" onchange="loadGraph()">
        <option value="">All sources</option>
        <option value="manual">Manual (Claude)</option>
        <option value="auto">Auto-encode (Stop)</option>
        <option value="idle">Idle (dreams/consolidate)</option>
        <option value="hook">Hook (compaction/boot)</option>
      </select>
      <button onclick="loadGraph()">Refresh</button>
    </div>
    <canvas id="graph-canvas"></canvas>
    <div class="node-tooltip" id="tooltip"></div>
    <div class="node-detail" id="node-detail" style="display:none"></div>
  </div>
</div>

<div id="tab-explorer" class="tab-content">
  <div class="explorer">
    <div class="search-bar">
      <input type="text" id="search-input" placeholder="Search nodes..." onkeyup="searchNodes()">
      <select id="type-filter" onchange="searchNodes()">
        <option value="">All types</option>
      </select>
    </div>
    <div id="node-list"></div>
  </div>
</div>

<div id="tab-errors" class="tab-content">
  <div style="padding:8px;display:flex;gap:8px;align-items:center">
    <span style="color:#888;font-size:12px">Last</span>
    <select id="error-hours" onchange="loadErrors()" style="background:#111;color:#ccc;border:1px solid #333;padding:3px 8px;border-radius:4px">
      <option value="1">1h</option>
      <option value="6">6h</option>
      <option value="24" selected>24h</option>
      <option value="168">7d</option>
    </select>
    <button onclick="loadErrors()" style="background:#1a1a2a;color:#7eb8ff;border:1px solid #3a3a5a;padding:3px 12px;border-radius:4px;cursor:pointer">Refresh</button>
    <span id="error-count" style="color:#666;font-size:11px;margin-left:auto"></span>
  </div>
  <div class="feed" id="errors-feed"></div>
</div>

<div id="tab-status" class="tab-content">
  <div style="padding:8px">
    <button onclick="loadSystemStatus()" style="background:#1a1a2a;color:#7eb8ff;border:1px solid #3a3a5a;padding:4px 16px;border-radius:4px;cursor:pointer">Refresh</button>
  </div>
  <div id="status-grid" style="padding:0 8px;display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px"></div>
</div>

<div id="tab-health" class="tab-content">
  <div class="health" id="health-content"></div>
</div>

<script>
let daemonAlive = false;

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    const tabs = ['live','graph','explorer','errors','status','health'];
    t.classList.toggle('active', tabs[i] === name);
  });
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'graph') { setTimeout(() => { initGraph(); resizeCanvas(); if (graphNodes.length) render(); else loadGraph(); }, 50); }
  if (name === 'explorer') searchNodes();
  if (name === 'errors') loadErrors();
  if (name === 'status') loadSystemStatus();
  if (name === 'health') loadHealth();
}

async function loadStats() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    daemonAlive = d.daemon === 'alive';
    const statusClass = daemonAlive ? 'alive' : 'unavailable';
    const statusText = daemonAlive ? 'Daemon: alive' : 'Daemon: offline';
    document.getElementById('stats-bar').innerHTML =
      `<div class="stat"><span class="stat-value">${d.nodes}</span><span class="stat-label">Nodes</span></div>
       <div class="stat"><span class="stat-value">${d.edges}</span><span class="stat-label">Edges</span></div>
       <div class="stat"><span class="stat-value">${d.locked}</span><span class="stat-label">Locked</span></div>
       <div class="stat"><span class="stat-value">${d.recent_24h}</span><span class="stat-label">24h</span></div>
       <div class="stat"><span class="stat-value">${d.orphans}</span><span class="stat-label">Orphans</span></div>
       <div class="daemon-status ${statusClass}">${statusText}</div>`;

    const banner = document.getElementById('daemon-banner');
    if (!daemonAlive) {
      banner.innerHTML = '<div class="no-daemon-banner">Daemon is not running — showing read-only data from database. Live events unavailable.</div>';
    } else {
      banner.innerHTML = '';
    }

    const sel = document.getElementById('type-filter');
    const current = sel.value;
    sel.innerHTML = '<option value="">All types</option>';
    Object.entries(d.types).forEach(([t, c]) => {
      sel.innerHTML += `<option value="${t}" ${t===current?'selected':''}>${t} (${c})</option>`;
    });
  } catch(e) {}
}
loadStats();
setInterval(loadStats, 30000);

// Live feed — polls hook_log from brain_logs.db
let lastHookId = 0;
const MAX_ENTRIES = 100;

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function localTime(utcStr, mode) {
  if (!utcStr) return '';
  const d = new Date(utcStr.endsWith('Z') ? utcStr : utcStr + 'Z');
  if (isNaN(d)) return utcStr;
  if (mode === 'time') return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  return d.toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', second:'2-digit'});
}

function toggleDetails(btn) {
  const details = btn.nextElementSibling;
  details.classList.toggle('open');
  btn.textContent = details.classList.contains('open') ? 'Hide Details' : 'Full Details';
}

function formatMetadata(raw) {
  try {
    const obj = JSON.parse(raw);
    if (obj.distill_prompt) {
      return '=== FULL PROMPT SENT TO HAIKU ===\\n\\n' +
        obj.distill_prompt + '\\n\\n' +
        '=== STATS ===\\n' +
        'Model: ' + (obj.model || '?') + '\\n' +
        'Candidates: ' + (obj.candidates_count || '?') + '\\n' +
        'Fetch latency: ' + (obj.latency_fetch_ms || '?') + 'ms\\n' +
        'Distill latency: ' + (obj.latency_distill_ms || '?') + 'ms';
    }
    return JSON.stringify(obj, null, 2);
  } catch(e) {
    return raw;
  }
}

function toggleHookBody(el) {
  const body = el.parentElement.querySelector('.hook-body');
  body.classList.toggle('open');
}

async function pollHookLog() {
  try {
    const r = await fetch('/api/hook-log?since_id=' + lastHookId + '&limit=20');
    const d = await r.json();
    if (!d.events || !d.events.length) return;
    const feed = document.getElementById('feed');
    // Clear placeholder on first data
    if (feed.querySelector('.hook-placeholder')) feed.querySelector('.hook-placeholder').remove();
    // Events come newest-first from API; reverse to prepend in correct order
    const sorted = d.events.slice().reverse();
    for (const evt of sorted) {
      if (evt.id <= lastHookId) continue;
      const div = document.createElement('div');
      div.className = 'hook-entry ' + (evt.hook_name || '');
      const t = localTime(evt.timestamp, 'time');
      const chars = (evt.output_text || '').length;
      const sid = evt.session_id ? evt.session_id.substring(0, 8) : '';
      div.innerHTML =
        '<div class="hook-header" onclick="toggleHookBody(this)">' +
          '<span class="hook-badge ' + evt.hook_name + '">' + evt.hook_name + '</span>' +
          '<span class="hook-time">' + t + '</span>' +
          (sid ? '<span class="hook-session" title="' + escapeHtml(evt.session_id) + '">' + sid + '</span>' : '') +
          '<span class="hook-id">#' + evt.id + '</span>' +
          '<span class="hook-size">' + chars + ' chars</span>' +
        '</div>' +
        (evt.user_prompt ? '<div class="hook-prompt">' + escapeHtml(evt.user_prompt) + '</div>' : '') +
        '<div class="hook-body">' +
          '<pre>' + escapeHtml(evt.output_text || '(empty)') + '</pre>' +
          (evt.operator_text ? '<pre style="border-left:2px solid #ffaa33;margin-top:6px">' + escapeHtml(evt.operator_text) + '</pre>' : '') +
          (evt.metadata ? '<button class="hook-details-btn" onclick="toggleDetails(this)">Full Details</button><div class="hook-details"><pre>' + escapeHtml(formatMetadata(evt.metadata)) + '</pre></div>' : '') +
        '</div>';
      feed.prepend(div);
    }
    lastHookId = d.latest_id;
    // Cap entries
    while (feed.children.length > MAX_ENTRIES) feed.removeChild(feed.lastChild);
  } catch(e) {}
}

// Initial load: get recent history
(async function() {
  const feed = document.getElementById('feed');
  feed.innerHTML = '<div class="hook-placeholder" style="color:#666;padding:20px;text-align:center">Waiting for brain activity...</div>';
  await pollHookLog();
})();
setInterval(pollHookLog, 2000);

// Feed toggle
let activeFeed = 'surface';
function switchFeed(name) {
  activeFeed = name;
  document.querySelectorAll('.feed-btn').forEach(b => {
    const label = b.textContent.toLowerCase();
    b.classList.toggle('active', label.includes(name));
  });
  document.getElementById('feed').style.display = name === 'surface' ? 'block' : 'none';
  document.getElementById('feed-encoding').style.display = name === 'encoding' ? 'block' : 'none';
  document.getElementById('feed-queue').style.display = name === 'queue' ? 'block' : 'none';
  if (name === 'encoding' && !encodingLoaded) loadEncodingActivity();
  if (name === 'queue') loadSignalQueue();
}

// Encoding activity feed
let encodingLoaded = false;
let lastEncodingTs = '';

async function loadEncodingActivity() {
  try {
    const r = await fetch('/api/encoding-activity?limit=50' + (lastEncodingTs ? '&since=' + encodeURIComponent(lastEncodingTs) : ''));
    const d = await r.json();
    if (!d.events || !d.events.length) {
      if (!encodingLoaded) {
        document.getElementById('feed-encoding').innerHTML = '<div style="color:#666;padding:20px;text-align:center">No recent encoding activity</div>';
      }
      encodingLoaded = true;
      return;
    }
    const container = document.getElementById('feed-encoding');
    if (!encodingLoaded) container.innerHTML = '';
    encodingLoaded = true;

    for (const evt of d.events) {
      const div = document.createElement('div');
      div.className = 'enc-entry ' + evt.kind;
      const t = localTime(evt.timestamp);

      if (evt.kind === 'created') {
        div.innerHTML =
          '<span class="enc-kind created">created</span>' +
          '<span class="type-badge type-' + (evt.type||'') + '">' + (evt.type||'') + '</span> ' +
          (evt.locked ? '&#x1f512; ' : '') +
          '<span class="enc-title">' + escapeHtml(evt.title || '') + '</span>' +
          '<div class="enc-meta">' + t + ' · conf: ' + (evt.confidence||0).toFixed(2) + ' · source: ' + (evt.encoding_source||'?') + '</div>' +
          '<div class="enc-content">' + escapeHtml(evt.content || '') + '</div>';
      } else if (evt.kind === 'revised') {
        div.innerHTML =
          '<span class="enc-kind revised">revised</span>' +
          '<span class="type-badge type-' + (evt.type||'') + '">' + (evt.type||'') + '</span> ' +
          '<span class="enc-title">' + escapeHtml(evt.title || '') + '</span>' +
          '<div class="enc-meta">' + t + ' · conf: ' + (evt.confidence||0).toFixed(2) + '</div>' +
          '<div class="enc-content">' + escapeHtml(evt.content || '') + '</div>';
      } else if (evt.kind === 'connected') {
        div.innerHTML =
          '<span class="enc-kind connected">connected</span>' +
          '<span class="enc-title">' + escapeHtml(evt.source_title || '') + '</span>' +
          ' <span style="color:#aa66ff">—' + (evt.relation||'related_to') + '→</span> ' +
          '<span class="enc-title">' + escapeHtml(evt.target_title || '') + '</span>' +
          '<div class="enc-meta">' + t + ' · weight: ' + (evt.weight||0).toFixed(2) + '</div>';
      } else if (evt.kind === 'enriched') {
        div.innerHTML =
          '<span class="enc-kind enriched">enriched</span>' +
          '<span class="enc-title">' + escapeHtml(evt.node_title || '') + '</span>' +
          ' <span style="color:#4a9eff">(' + (evt.vector_type||'') + ')</span>' +
          '<div class="enc-meta">' + t + '</div>' +
          '<div class="enc-content">' + escapeHtml(evt.text || '') + '</div>';
      }
      container.prepend(div);
    }
    if (d.events.length) lastEncodingTs = d.events[0].timestamp;
  } catch(e) {}
}

setInterval(() => { if (activeFeed === 'encoding') loadEncodingActivity(); }, 3000);

// Signal Queue feed
async function loadSignalQueue() {
  try {
    const [queueR, compR] = await Promise.all([
      fetch('/api/signal-queue'),
      fetch('/api/assembler-comparison?limit=10')
    ]);
    const queueD = await queueR.json();
    const compD = await compR.json();
    const container = document.getElementById('feed-queue');

    let html = '';

    // Comparison banner
    if (compD.comparisons && compD.comparisons.length) {
      const latest = compD.comparisons[0];
      const pct = latest.old_chars ? Math.round((1 - latest.new_chars / latest.old_chars) * 100) : 0;
      html += '<div style="padding:10px 12px;background:#1a1a2a;border-radius:6px;margin:4px 0;font-size:12px">';
      html += '<span style="color:#888">Latest:</span> ';
      html += '<span style="color:#ff6666">' + latest.old_chars + ' chars (old)</span>';
      html += ' → <span style="color:#33ff88">' + latest.new_chars + ' chars (new)</span>';
      html += ' <span style="color:#7eb8ff">(' + pct + '% reduction)</span>';
      if (latest.user_prompt) html += '<div style="color:#58a6ff;font-style:italic;margin-top:4px">' + escapeHtml(latest.user_prompt) + '</div>';
      html += '</div>';
    }

    // Queue items
    if (!queueD.signals || !queueD.signals.length) {
      html += '<div style="color:#666;padding:20px;text-align:center">Queue empty — no pending signals</div>';
    } else {
      html += '<div style="color:#888;font-size:11px;padding:4px 8px">' + queueD.signals.length + ' signals in queue</div>';
      for (const sig of queueD.signals) {
        const priColor = sig.priority > 0.9 ? '#ff4444' : sig.priority > 0.7 ? '#ffaa33' : sig.priority > 0.5 ? '#ffff66' : '#666';
        const priBar = '<span style="display:inline-block;width:' + Math.round(sig.priority * 60) + 'px;height:4px;background:' + priColor + ';border-radius:2px;vertical-align:middle;margin-right:6px"></span>';
        const surfaced = sig.times_surfaced + (sig.max_surfaces ? '/' + sig.max_surfaces : '');
        const preemptBadge = sig.preempt ? ' <span style="color:#ff4444;font-size:9px;font-weight:bold">PREEMPT</span>' : '';

        html += '<div class="enc-entry" style="border-left-color:' + priColor + '">';
        html += priBar;
        html += '<span class="enc-kind" style="background:#1a1a2a;color:' + priColor + '">' + escapeHtml(sig.producer) + '</span> ';
        html += '<span class="enc-title">' + escapeHtml(sig.content).substring(0, 120) + '</span>' + preemptBadge;
        html += '<div class="enc-meta">';
        html += 'pri: ' + sig.priority.toFixed(2) + ' · surfaced: ' + surfaced + ' · type: ' + sig.signal_type;
        html += ' · ' + localTime(sig.created_at);
        if (sig.cooldown_seconds) html += ' · cooldown: ' + sig.cooldown_seconds + 's';
        html += '</div></div>';
      }
    }

    container.innerHTML = html;
  } catch(e) {
    document.getElementById('feed-queue').innerHTML = '<div style="color:#ff4444;padding:20px">Error loading queue: ' + e.message + '</div>';
  }
}

setInterval(() => { if (activeFeed === 'queue') loadSignalQueue(); }, 3000);

// Explorer
let expandedNode = null;
async function searchNodes() {
  const search = document.getElementById('search-input').value;
  const type = document.getElementById('type-filter').value;
  let url = '/api/nodes?limit=100';
  if (search) url += '&search=' + encodeURIComponent(search);
  if (type) url += '&type=' + encodeURIComponent(type);
  try {
    const r = await fetch(url);
    const d = await r.json();
    const list = document.getElementById('node-list');
    list.innerHTML = d.nodes.map(n => `
      <div class="node-card ${expandedNode===n.id?'expanded':''}" onclick="toggleNode('${n.id}', this)">
        <div class="node-title">
          <span class="type-badge type-${n.type}">${n.type}</span>
          ${n.locked ? '<span class="locked-icon">&#x1f512;</span>' : ''}
          ${n.title || '(untitled)'}
        </div>
        <div class="node-meta">
          <span>accessed: ${n.access_count}x</span>
          <span>emotion: ${(n.emotion||0).toFixed(1)}</span>
          <span>${localTime(n.created_at)}</span>
        </div>
        <div class="node-content">${(n.content||'').replace(/</g,'&lt;')}</div>
      </div>
    `).join('');
  } catch(e) {}
}
function toggleNode(id, el) {
  expandedNode = expandedNode === id ? null : id;
  el.classList.toggle('expanded');
}

// Errors
async function loadErrors() {
  const hours = document.getElementById('error-hours').value;
  try {
    const r = await fetch('/api/errors?hours=' + hours + '&limit=100');
    const d = await r.json();
    const feed = document.getElementById('errors-feed');
    document.getElementById('error-count').textContent = d.count + ' errors';

    if (!d.errors || !d.errors.length) {
      feed.innerHTML = '<div style="color:#4a4;padding:20px;text-align:center">✅ No errors in the last ' + hours + 'h</div>';
      return;
    }
    feed.innerHTML = '';
    for (const e of d.errors) {
      const div = document.createElement('div');
      const levelColor = {critical:'#ff4444',error:'#ff6644',warning:'#ffaa33',info:'#4a9eff'}[e.level] || '#888';
      div.style.cssText = 'padding:8px 12px;margin:4px 0;background:#111118;border-radius:6px;border-left:3px solid ' + levelColor + ';font-size:12px';
      const t = localTime(e.timestamp);
      div.innerHTML =
        '<span style="display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:bold;text-transform:uppercase;background:' + levelColor + '22;color:' + levelColor + '">' + (e.level || 'error') + '</span> ' +
        '<span style="color:#888;font-size:10px">' + (e.source || '') + '</span> ' +
        '<span style="color:#aaa;font-weight:bold">' + escapeHtml(e.component || '') + '</span>' +
        '<div style="color:#ccc;margin-top:3px">' + escapeHtml(e.error || '') + '</div>' +
        (e.context ? '<div style="color:#666;font-size:10px;margin-top:2px">' + escapeHtml(e.context) + '</div>' : '') +
        '<div style="color:#555;font-size:10px;margin-top:2px">' + t + '</div>';
      feed.appendChild(div);
    }
  } catch(e) {
    document.getElementById('errors-feed').innerHTML = '<div style="color:#f66;padding:20px">Failed to load errors: ' + e + '</div>';
  }
}

// System Status
async function loadSystemStatus() {
  try {
    const r = await fetch('/api/system-status');
    const d = await r.json();
    const grid = document.getElementById('status-grid');
    grid.innerHTML = '';

    const components = [
      {key: 'daemon', label: 'Brain Daemon', icon: '🧠'},
      {key: 'brain_db', label: 'Brain DB', icon: '💾'},
      {key: 'logs_db', label: 'Logs DB', icon: '📋'},
      {key: 'dashboard_db', label: 'Dashboard DB', icon: '📊'},
      {key: 'embedder', label: 'Embedder', icon: '🔮'},
      {key: 'signal_queue', label: 'Signal Queue', icon: '📡'},
    ];

    for (const comp of components) {
      const s = d.status[comp.key] || {alive: false, error: 'unknown'};
      const alive = s.alive;
      const card = document.createElement('div');
      card.style.cssText = 'background:#111118;border-radius:8px;padding:12px 16px;border:1px solid ' + (alive ? '#1a3a1a' : '#3a1a1a');

      let details = '';
      if (comp.key === 'daemon' && alive) {
        details = 'PID: ' + (s.pid || '?') + ' · Uptime: ' + Math.round((s.uptime || 0) / 60) + 'min';
      } else if (comp.key === 'brain_db' && alive) {
        details = s.nodes + ' nodes · ' + (s.size_mb || '?') + 'MB';
      } else if (comp.key === 'logs_db' && alive) {
        details = (s.size_mb || '?') + 'MB';
      } else if (comp.key === 'dashboard_db' && alive) {
        details = (s.size_mb || '?') + 'MB · Last: ' + localTime(s.last_entry);
      } else if (comp.key === 'embedder' && alive) {
        details = s.model || '?';
      } else if (comp.key === 'signal_queue' && alive) {
        details = s.pending + ' pending' + (s.preempt > 0 ? ' · ⚠️ ' + s.preempt + ' PREEMPT' : '');
      } else if (!alive) {
        details = s.error || 'unreachable';
      }

      const pathLine = s.path ? '<div style="font-size:9px;color:#444;margin-top:4px;word-break:break-all">' + escapeHtml(s.path) + '</div>' : '';
      card.innerHTML =
        '<div style="display:flex;align-items:center;gap:8px">' +
          '<span style="font-size:20px">' + comp.icon + '</span>' +
          '<div>' +
            '<div style="color:#ccc;font-weight:bold;font-size:13px">' + comp.label + '</div>' +
            '<div style="font-size:11px;margin-top:2px;color:' + (alive ? '#4a4' : '#f44') + '">' +
              (alive ? '● Live' : '● Down') +
            '</div>' +
          '</div>' +
          '<div style="margin-left:auto;font-size:10px;color:#666;text-align:right;max-width:200px;overflow:hidden;text-overflow:ellipsis">' + escapeHtml(details) + '</div>' +
        '</div>' + pathLine;
      grid.appendChild(card);
    }
  } catch(e) {
    document.getElementById('status-grid').innerHTML = '<div style="color:#f66;padding:20px">Failed to load status: ' + e + '</div>';
  }
}

// Auto-refresh status every 5s when tab is active
setInterval(() => {
  const statusTab = document.getElementById('tab-status');
  if (statusTab && statusTab.classList.contains('active')) loadSystemStatus();
}, 5000);

// Auto-refresh errors every 10s when tab is active
setInterval(() => {
  const errTab = document.getElementById('tab-errors');
  if (errTab && errTab.classList.contains('active')) loadErrors();
}, 10000);

// Health
async function loadHealth() {
  try {
    const statsR = await fetch('/api/stats');
    const insightsR = await fetch('/api/insights');
    const d = await statsR.json();
    const ins = await insightsR.json();
    const hc = document.getElementById('health-content');
    const orphanClass = d.orphans > 20 ? 'bad' : d.orphans > 5 ? 'warn' : 'ok';
    const sevColors = {high: '#ff6666', medium: '#ffaa33', low: '#7eb8ff'};
    const insightsHtml = (ins.insights || []).map(i => `
      <div style="background:#111118;border-radius:8px;padding:14px;margin:8px 0;border-left:4px solid ${sevColors[i.severity] || '#555'}">
        <div style="font-size:15px;font-weight:bold;color:${sevColors[i.severity]}">${i.icon} ${i.title}</div>
        <div style="color:#999;margin-top:6px;font-size:12px;line-height:1.5">${i.detail}</div>
        ${i.nodes ? '<div style="margin-top:8px;font-size:11px;color:#666">' + i.nodes.map(n =>
          '<div style="padding:2px 0">&#8226; ' + (n.title||'').substring(0,80) + ' <span style="color:#555">(' + (n.type||n.count||'') + ')</span></div>'
        ).join('') + '</div>' : ''}
      </div>
    `).join('');
    hc.innerHTML = `
      <div class="health-grid">
        <div class="health-card ok"><div class="hc-value">${d.nodes}</div><div class="hc-label">Total Nodes</div></div>
        <div class="health-card ok"><div class="hc-value">${d.edges}</div><div class="hc-label">Total Edges</div></div>
        <div class="health-card ok"><div class="hc-value">${d.locked}</div><div class="hc-label">Locked</div></div>
        <div class="health-card ${d.recent_24h > 0 ? 'ok' : 'warn'}"><div class="hc-value">${d.recent_24h}</div><div class="hc-label">Last 24h</div></div>
        <div class="health-card ${orphanClass}"><div class="hc-value">${d.orphans}</div><div class="hc-label">Orphans</div></div>
      </div>
      ${insightsHtml ? '<h3 style="color:#ccc;margin:20px 0 8px">Anchor Insights</h3>' + insightsHtml : '<div style="color:#33ff88;padding:20px;text-align:center;font-size:16px">No issues detected</div>'}
      <h3 style="color:#888;margin:20px 0 8px">Node Types</h3>
      <div class="health-grid">
        ${Object.entries(d.types).map(([t,c]) => `
          <div class="health-card ok" style="padding:10px">
            <span class="type-badge type-${t}">${t}</span>
            <span style="float:right;font-size:18px;font-weight:bold;color:#7eb8ff">${c}</span>
          </div>
        `).join('')}
      </div>
    `;
  } catch(e) { console.error(e); }
}

// Graph
let graphData = null, graphNodes = [], graphEdges = [];
let canvas, ctx;
let offsetX = 0, offsetY = 0, scale = 1;
let dragNode = null, dragStartX, dragStartY;
let hoveredNode = null, graphInited = false;

const TYPE_COLORS = {
  lesson: '#4a9eff', correction: '#ff6666', interaction: '#33ff88',
  rule: '#ffaa33', decision: '#aa66ff', mental_model: '#33dddd',
  mechanism: '#dddd33', vocabulary: '#666', context: '#555',
  bug_lesson: '#ff8866', pattern: '#ff66aa', boot: '#888',
  tension: '#ff4444', uncertainty: '#aaaaff', constraint: '#ff8833',
  impact: '#ff6644', convention: '#66aaff',
};

async function loadNodeDetail(nodeId) {
  const panel = document.getElementById('node-detail');
  panel.style.display = 'block';
  panel.innerHTML = '<div style="color:#666;padding:20px">Loading...</div>';
  try {
    const r = await fetch('/api/node/' + nodeId);
    const d = await r.json();
    const n = d.node;
    const conns = d.connections || [];
    panel.innerHTML = `
      <div class="nd-close" onclick="document.getElementById('node-detail').style.display='none'">&times;</div>
      <div class="nd-title"><span class="type-badge type-${n.type}">${n.type}</span> ${n.locked ? '&#x1f512;' : ''} ${(n.title||'').replace(/</g,'&lt;')}</div>
      <div class="nd-meta">
        <span>accessed: ${n.access_count}x</span>
        <span>confidence: ${(n.confidence||0).toFixed(2)}</span>
        <span>source: ${n.encoding_source||'?'}</span>
        <span>emotion: ${(n.emotion||0).toFixed(1)}</span>
        <span>${localTime(n.created_at)}</span>
      </div>
      ${n.keywords ? '<div style="color:#555;font-size:10px;margin-bottom:8px">'+n.keywords+'</div>' : ''}
      <div class="nd-section">Content</div>
      <div class="nd-content">${(n.content||'(empty)').replace(/</g,'&lt;')}</div>
      <div class="nd-section">Connections (${conns.length})</div>
      ${conns.map(c => `
        <div class="nd-conn" onclick="loadNodeDetail('${c.id}')">
          <div class="nd-conn-title"><span class="type-badge type-${c.type}">${c.type}</span> ${(c.title||'').replace(/</g,'&lt;').substring(0,60)}</div>
          <div class="nd-conn-meta">${c.relation} · weight ${(c.weight||0).toFixed(2)}</div>
        </div>
      `).join('')}
      ${conns.length === 0 ? '<div style="color:#555;padding:8px">No connections</div>' : ''}
    `;
  } catch(e) {
    panel.innerHTML = '<div style="color:#ff6666;padding:20px">Failed to load: ' + e.message + '</div>';
  }
}

async function loadGraph() {
  const days = document.getElementById('graph-days').value;
  const limit = document.getElementById('graph-limit').value;
  const source = document.getElementById('graph-source').value;
  let url = `/api/graph?days=${days}&limit=${limit}`;
  if (source) url += `&source=${source}`;
  try {
    const r = await fetch(url);
    graphData = await r.json();
    initForce();
  } catch(e) {}
}

function initGraph() {
  if (graphInited) return;
  graphInited = true;
  canvas = document.getElementById('graph-canvas');
  ctx = canvas.getContext('2d');
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  canvas.addEventListener('mousedown', onMouseDown);
  canvas.addEventListener('mousemove', onMouseMove);
  canvas.addEventListener('mouseup', onMouseUp);
  canvas.addEventListener('wheel', onWheel);
  loadGraph();
}

function resizeCanvas() {
  if (!canvas) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}

function initForce() {
  if (!graphData) return;
  const w = canvas.width / devicePixelRatio;
  const h = canvas.height / devicePixelRatio;
  graphNodes = graphData.nodes.map(n => ({
    ...n, x: w/2 + (Math.random()-0.5)*20,
    y: h/2 + (Math.random()-0.5)*20, vx: 0, vy: 0,
    radius: Math.max(4, Math.min(16, Math.sqrt(n.access_count || 1) * 2)),
  }));
  const idMap = {};
  graphNodes.forEach((n, i) => idMap[n.id] = i);
  graphEdges = graphData.edges.filter(e => idMap[e.source] !== undefined && idMap[e.target] !== undefined)
    .map(e => ({ source: idMap[e.source], target: idMap[e.target], relation: e.relation, weight: e.weight }));
  simulate();
}

function simulate() {
  let iterations = 0;
  function tick() {
    if (iterations > 300) { render(); return; }
    iterations++;
    const n = graphNodes.length;
    for (let i = 0; i < n; i++) {
      for (let j = i+1; j < n; j++) {
        let dx = graphNodes[j].x - graphNodes[i].x;
        let dy = graphNodes[j].y - graphNodes[i].y;
        let d = Math.sqrt(dx*dx + dy*dy) || 1;
        let force = 400 / (d * d);
        graphNodes[i].vx -= dx/d * force; graphNodes[i].vy -= dy/d * force;
        graphNodes[j].vx += dx/d * force; graphNodes[j].vy += dy/d * force;
      }
    }
    for (const e of graphEdges) {
      const a = graphNodes[e.source], b = graphNodes[e.target];
      let dx = b.x - a.x, dy = b.y - a.y;
      let d = Math.sqrt(dx*dx + dy*dy) || 1;
      let force = (d - 40) * 0.02;
      a.vx += dx/d * force; a.vy += dy/d * force;
      b.vx -= dx/d * force; b.vy -= dy/d * force;
    }
    const cx = (canvas.width/devicePixelRatio)/2, cy = (canvas.height/devicePixelRatio)/2;
    for (const node of graphNodes) {
      node.vx += (cx - node.x) * 0.005; node.vy += (cy - node.y) * 0.005;
      node.vx *= 0.9; node.vy *= 0.9;
      node.x += node.vx; node.y += node.vy;
    }
    render();
    requestAnimationFrame(tick);
  }
  tick();
}

function render() {
  const w = canvas.width / devicePixelRatio, h = canvas.height / devicePixelRatio;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);
  ctx.globalAlpha = 0.15;
  for (const e of graphEdges) {
    const a = graphNodes[e.source], b = graphNodes[e.target];
    ctx.strokeStyle = '#4a4a6a'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  for (const n of graphNodes) {
    ctx.fillStyle = TYPE_COLORS[n.type] || '#888';
    ctx.globalAlpha = n.locked ? 1 : 0.7;
    ctx.beginPath(); ctx.arc(n.x, n.y, n.radius, 0, Math.PI*2); ctx.fill();
    if (n.locked) { ctx.strokeStyle = '#ffaa33'; ctx.lineWidth = 1.5; ctx.stroke(); }
    if (n === hoveredNode) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); }
  }
  ctx.globalAlpha = 0.8; ctx.fillStyle = '#ccc';
  ctx.font = '9px SF Mono, monospace'; ctx.textAlign = 'center';
  for (const n of graphNodes) {
    if (n.radius > 6 || n === hoveredNode) ctx.fillText(n.title.substring(0, 25), n.x, n.y + n.radius + 12);
  }
  ctx.restore();
}

function getNodeAt(mx, my) {
  const x = (mx - offsetX) / scale, y = (my - offsetY) / scale;
  for (const n of graphNodes) {
    const dx = n.x - x, dy = n.y - y;
    if (dx*dx + dy*dy < (n.radius+4)*(n.radius+4)) return n;
  }
  return null;
}
function onMouseDown(e) {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  dragNode = getNodeAt(mx, my);
  if (dragNode) { dragStartX = mx; dragStartY = my; }
  else { dragStartX = mx - offsetX; dragStartY = my - offsetY; }
}
function onMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (dragNode) {
    dragNode.x += (mx - dragStartX) / scale; dragNode.y += (my - dragStartY) / scale;
    dragStartX = mx; dragStartY = my; render();
  } else if (e.buttons === 1) {
    offsetX = mx - dragStartX; offsetY = my - dragStartY; render();
  } else {
    const n = getNodeAt(mx, my);
    if (n !== hoveredNode) {
      hoveredNode = n; render();
      const tt = document.getElementById('tooltip');
      if (n) {
        tt.style.display = 'block'; tt.style.left = (mx+15)+'px'; tt.style.top = (my+15)+'px';
        tt.innerHTML = `<div class="tt-title">${n.title}</div><div class="tt-type"><span class="type-badge type-${n.type}">${n.type}</span> ${n.locked?'&#x1f512;':''} accessed ${n.access_count}x</div>`;
      } else { tt.style.display = 'none'; }
    }
  }
}
function onMouseUp(e) {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const moved = Math.abs(mx - dragStartX) + Math.abs(my - dragStartY);
  if (dragNode && moved < 5) {
    // Click, not drag — load node detail
    loadNodeDetail(dragNode.id);
  }
  dragNode = null;
}
function onWheel(e) {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  const newScale = scale * delta;
  if (newScale < 0.1 || newScale > 10) return;
  offsetX = mx - (mx - offsetX) * delta;
  offsetY = my - (my - offsetY) * delta;
  scale = newScale; render();
}
</script>
</body>
</html>'''


if __name__ == "__main__":
    server = ThreadedHTTPServer(("127.0.0.1", DASHBOARD_PORT), DashboardHandler)
    print("Brain Dashboard listening on http://127.0.0.1:%d" % DASHBOARD_PORT, flush=True)
    print("Daemon port: %d" % DAEMON_PORT, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.", flush=True)
        server.shutdown()
