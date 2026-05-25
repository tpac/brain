"""HTTP server + route dispatch for the Brain Dashboard.

This file contains no SQL and no presentation logic — routes either:
  * resolve a `/static/*` asset and serve it, or
  * call a `queries.*` function and JSON-encode the result.

Everything else lives in `dashboard.queries.*` (data access) and
`dashboard.static/*` (HTML/CSS/JS).
"""

import json
import os
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

from .daemon_client import DAEMON_PORT, daemon_alive
from .queries import (
    aspects,
    encoding,
    errors,
    explorer,
    graph,
    legacy,
    recalls,
    s2_runs,
    sessions,
    stats,
    system,
    traces,
)

# Dashboard listens here. Override with DASHBOARD_PORT env var (eval brains
# usually run a second dashboard on a different port).
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", 47303))

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Static MIME table — kept inline rather than mimetypes.guess_type, since the
# dashboard only ships three asset types and we want to enforce charset.
_STATIC_MIME = {
    '.html': 'text/html; charset=utf-8',
    '.css':  'text/css; charset=utf-8',
    '.js':   'application/javascript; charset=utf-8',
}


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class DashboardHandler(BaseHTTPRequestHandler):

    # Silence default access logging — there's already a steady stream from
    # the dashboard polling itself; logging every poll buries real signal.
    def log_message(self, format, *args):
        pass

    # ── Routing ──

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._serve_static_file("index.html")
        elif path.startswith("/static/"):
            self._serve_static_file(path[len("/static/"):])

        # Stats / health
        elif path == "/api/stats":
            self._json(200, stats.query_stats())
        elif path == "/api/status":
            self._json(200, {
                "daemon": "alive" if daemon_alive() else "unavailable",
                "dashboard": "running",
                "daemon_port": DAEMON_PORT,
            })
        elif path == "/api/system-status":
            self._json(200, {"status": system.check_system_status()})
        elif path == "/api/insights":
            self._json(200, {"insights": stats.query_insights()})
        elif path == "/api/aspects":
            self._json(200, {"aspects": aspects.query_aspects()})

        # Live decoding feed
        elif path == "/api/recalls":
            self._serve_recalls(params)
        elif path == "/api/sessions":
            self._json(200, sessions.query_recent_sessions())

        # Encoding
        elif path == "/api/encoding-activity":
            since_ts = params.get("since", [""])[0]
            limit = int(params.get("limit", [30])[0])
            self._json(200, {"events": encoding.query_encoding_activity(since_ts=since_ts, limit=limit)})
        elif path == "/api/encoding-runs":
            limit = int(params.get("limit", [10])[0])
            hours = int(params.get("hours", [24])[0])
            self._json(200, {"runs": encoding.query_encoding_runs(limit=limit, hours=hours)})

        # S2 unit runs
        elif path == "/api/consolidation-runs":
            hours = int(params.get("hours", ["24"])[0])
            self._json(200, {"runs": s2_runs.query_consolidation_runs(hours=hours)})
        elif path == "/api/community-runs":
            hours = int(params.get("hours", ["24"])[0])
            self._json(200, {"runs": s2_runs.query_community_runs(hours=hours)})
        elif path == "/api/healer-runs":
            hours = int(params.get("hours", ["24"])[0])
            self._json(200, {"runs": s2_runs.query_healer_runs(hours=hours)})
        elif path == "/api/consolidation-prompt":
            batch = int(params.get("batch", ["1"])[0])
            self._serve_consolidation_prompt(batch)

        # Traces
        elif path == "/api/traces":
            hours = int(params.get("hours", ["24"])[0])
            scale = params.get("scale", [""])[0]
            session_id = params.get("session", [""])[0]
            self._json(200, traces.query_traces(hours=hours, scale=scale, session_id=session_id))

        # Explorer + graph
        elif path == "/api/nodes":
            self._serve_nodes(params)
        elif path == "/api/graph3d":
            self._serve_graph3d()
        elif path == "/api/graph":
            # Legacy 2D graph endpoint — the 3D graph subsumed it. Kept so
            # external tooling that still hits this URL gets a sane response.
            self._serve_graph3d()
        elif path.startswith("/api/node/") and path.endswith("/corrections"):
            node_id = path[len("/api/node/"):-len("/corrections")]
            self._json(200, {"corrections": explorer.query_node_corrections(node_id)})
        elif path.startswith("/api/node/") and path.endswith("/source-refs"):
            node_id = path[len("/api/node/"):-len("/source-refs")]
            self._json(200, {"refs": explorer.query_node_source_refs(node_id)})
        elif path.startswith("/api/node/"):
            self._serve_node_detail(path.split("/api/node/")[1])

        # Errors / logs
        elif path == "/api/errors":
            self._serve_errors(params)

        # Legacy
        elif path == "/api/hook-log":
            since_id = int(params.get("since_id", [0])[0])
            limit = int(params.get("limit", [50])[0])
            entries = legacy.query_hook_log(since_id=since_id, limit=limit)
            latest_id = entries[0]["id"] if entries else since_id
            self._json(200, {"events": entries, "latest_id": latest_id})
        elif path == "/api/assembler-comparison":
            limit = int(params.get("limit", [20])[0])
            self._json(200, {"comparisons": legacy.query_assembler_comparison(limit=limit)})

        else:
            self._json(404, {"error": "Not found"})

    # ── Response helpers ──

    def _json(self, code: int, data):
        body = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static_file(self, relpath: str):
        # Defend against `/static/../something` — no traversal, no absolute
        # paths. Anything that escapes STATIC_DIR after resolution → 404.
        safe = os.path.normpath(os.path.join(STATIC_DIR, relpath))
        if not safe.startswith(STATIC_DIR + os.sep) and safe != STATIC_DIR:
            return self._json(404, {"error": "Not found"})
        if not os.path.isfile(safe):
            return self._json(404, {"error": "Not found"})
        ext = os.path.splitext(safe)[1].lower()
        mime = _STATIC_MIME.get(ext, 'application/octet-stream')
        with open(safe, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    # ── Routes that need a touch of glue ──

    def _serve_recalls(self, params):
        """Live recall feed. Cursor is timestamp-based (since_ts), with the
        legacy since_id key still accepted as no-op so old clients don't 500."""
        since_ts = params.get("since_ts", [""])[0]
        limit = int(params.get("limit", [50])[0])
        session_id = params.get("session_id", [""])[0]
        entries = recalls.query_recall_log(since_ts=since_ts, limit=limit, session_id=session_id)
        latest_ts = entries[0]["timestamp"] if entries else since_ts
        self._json(200, {"events": entries, "latest_ts": latest_ts})

    def _serve_errors(self, params):
        hours = int(params.get("hours", [24])[0])
        limit = int(params.get("limit", [50])[0])
        source = params.get("source", [""])[0]
        all_errors = errors.query_all_errors(limit=limit, hours=hours)
        if source:
            all_errors = [e for e in all_errors if (e.get('source') or '') == source]
        self._json(200, {"errors": all_errors, "count": len(all_errors)})

    def _serve_nodes(self, params):
        try:
            limit = int(params.get("limit", [50])[0])
            node_type = params.get("type", [None])[0]
            search = params.get("search", [None])[0]
            nodes = explorer.query_node_list(limit=limit, node_type=node_type, search=search)
            self._json(200, {"nodes": nodes, "total": len(nodes)})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _serve_node_detail(self, node_id: str):
        try:
            result = explorer.query_node_detail(node_id)
            if result is None:
                return self._json(404, {"error": "Node not found"})
            self._json(200, result)
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _serve_graph3d(self):
        try:
            self._json(200, graph.query_graph3d())
        except Exception as e:
            self._json(500, {"error": str(e), "trace": traceback.format_exc()})

    def _serve_consolidation_prompt(self, batch: int):
        # The S2 consolidation encoder writes its prompt to /tmp before
        # invoking Sonnet. The dashboard reads that file straight through.
        prompt_path = "/tmp/brain-consolidation-prompt-%d.json" % batch
        if not os.path.exists(prompt_path):
            return self._json(404, {"error": "No prompt file for batch %d" % batch})
        try:
            with open(prompt_path) as f:
                self._json(200, json.load(f))
        except Exception as e:
            self._json(500, {"error": str(e)})


def run(port: int = None) -> None:
    """Start the dashboard server."""
    port = port if port is not None else DASHBOARD_PORT
    server = ThreadedHTTPServer(("127.0.0.1", port), DashboardHandler)
    print("Brain Dashboard listening on http://127.0.0.1:%d" % port, flush=True)
    print("Daemon port: %d" % DAEMON_PORT, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.", flush=True)
        server.shutdown()
