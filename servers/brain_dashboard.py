"""
brain — Dashboard Server

HTTP server that serves a live brain dashboard in the Claude Code preview pane.
Completely separated from brain logic — read-only observation.

Endpoints:
  GET /           → Dashboard HTML (single page, everything inline)
  GET /api/stats  → Brain stats (node count, edge count, health)
  GET /api/nodes  → Node list with filtering (?type=lesson&limit=50&search=encoding)
  GET /api/graph  → Nodes + edges for visualization (?limit=80&days=7)
  GET /api/events → SSE stream of real-time brain events

Port: DAEMON_PORT + 100 (e.g., 47303)
"""

import json
import os
import queue
import sqlite3
import threading
import time
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

from servers.daemon_config import DAEMON_HOST, DAEMON_PORT

DASHBOARD_PORT = DAEMON_PORT + 100
_brain = None  # Set by daemon on startup
_daemon = None  # Set by daemon — used to reset idle timeout on HTTP requests
_event_queues = []  # SSE client queues
_event_lock = threading.Lock()
_server = None
_running = False


def start(brain=None, daemon=None):
    """Start dashboard HTTP server. Called by daemon on startup."""
    global _brain, _daemon, _server, _running
    _brain = brain
    _daemon = daemon
    if _running:
        return
    try:
        class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True
        _server = ThreadedHTTPServer((DAEMON_HOST, DASHBOARD_PORT), DashboardHandler)
        _server.timeout = 1
        _running = True
        t = threading.Thread(target=_serve_loop, daemon=True, name="dashboard-http")
        t.start()
        _log("Dashboard listening on http://%s:%d" % (DAEMON_HOST, DASHBOARD_PORT))
    except Exception as e:
        _log("Dashboard failed to start: %s" % e)
        _running = False


def stop():
    """Stop the dashboard server."""
    global _running, _server
    _running = False
    with _event_lock:
        _event_queues.clear()
    if _server:
        try:
            _server.shutdown()
        except Exception:
            pass
        _server = None


def emit(event_type, data=None, **kwargs):
    """Push an event to all SSE clients. No-op if nobody's listening."""
    with _event_lock:
        if not _event_queues:
            return
    event = {"type": event_type, "time": time.strftime("%H:%M:%S")}
    if data:
        event.update(data)
    if kwargs:
        event.update(kwargs)
    with _event_lock:
        dead = []
        for i, q in enumerate(_event_queues):
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(i)
        for i in reversed(dead):
            _event_queues.pop(i)


def has_listeners():
    with _event_lock:
        return len(_event_queues) > 0


def _serve_loop():
    while _running:
        try:
            _server.handle_request()
        except Exception:
            if _running:
                continue
            break


def _log(msg):
    print("[dashboard] %s" % msg, file=sys.stderr, flush=True)


class DashboardHandler(BaseHTTPRequestHandler):
    """Handle dashboard HTTP requests."""

    def log_message(self, format, *args):
        pass  # Silence default logging

    def do_GET(self):
        # Keep daemon alive while dashboard is being viewed
        if _daemon:
            _daemon.last_activity = time.time()
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
        elif path == "/api/events":
            self._serve_sse()
        elif path == "/api/insights":
            self._serve_insights()
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

    def _serve_stats(self):
        if not _brain:
            return self._json_response(503, {"error": "Brain not loaded"})
        try:
            c = _brain.conn
            nodes = c.execute("SELECT COUNT(*) FROM nodes WHERE archived = 0").fetchone()[0]
            edges = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            locked = c.execute("SELECT COUNT(*) FROM nodes WHERE locked = 1").fetchone()[0]
            types = c.execute(
                "SELECT type, COUNT(*) FROM nodes WHERE archived = 0 GROUP BY type ORDER BY COUNT(*) DESC"
            ).fetchall()
            recent = c.execute(
                "SELECT COUNT(*) FROM nodes WHERE created_at > datetime('now', '-24 hours')"
            ).fetchone()[0]
            orphans = c.execute("""
                SELECT COUNT(*) FROM nodes n WHERE archived = 0
                AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.source_id = n.id OR e.target_id = n.id)
            """).fetchone()[0]
            self._json_response(200, {
                "nodes": nodes, "edges": edges, "locked": locked,
                "recent_24h": recent, "orphans": orphans,
                "types": {t: cnt for t, cnt in types},
            })
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _serve_nodes(self, params):
        if not _brain:
            return self._json_response(503, {"error": "Brain not loaded"})
        try:
            c = _brain.conn
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

            rows = c.execute(sql, args).fetchall()
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
        if not _brain:
            return self._json_response(503, {"error": "Brain not loaded"})
        try:
            c = _brain.conn
            limit = int(params.get("limit", [80])[0])
            days = int(params.get("days", [30])[0])

            # Get top nodes by access_count within time window
            nodes_sql = """
                SELECT id, type, title, locked, emotion, access_count, created_at
                FROM nodes WHERE archived = 0
                AND created_at > datetime('now', '-%d days')
                ORDER BY access_count DESC LIMIT ?
            """ % days
            rows = c.execute(nodes_sql, (limit,)).fetchall()
            node_ids = set()
            nodes = []
            for r in rows:
                node_ids.add(r[0])
                nodes.append({
                    "id": r[0], "type": r[1], "title": r[2][:60],
                    "locked": bool(r[3]), "emotion": r[4] or 0,
                    "access_count": r[5], "created_at": r[6],
                })

            # Get edges between visible nodes
            if node_ids:
                placeholders = ",".join("?" * len(node_ids))
                edges_sql = """
                    SELECT source_id, target_id, relation, weight
                    FROM edges
                    WHERE source_id IN (%s) AND target_id IN (%s)
                """ % (placeholders, placeholders)
                id_list = list(node_ids)
                edge_rows = c.execute(edges_sql, id_list + id_list).fetchall()
                edges = [{"source": r[0], "target": r[1], "relation": r[2], "weight": r[3]}
                         for r in edge_rows]
            else:
                edges = []

            self._json_response(200, {"nodes": nodes, "edges": edges})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _serve_sse(self):
        """Server-Sent Events stream for real-time brain events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q = queue.Queue(maxsize=100)
        with _event_lock:
            _event_queues.append(q)

        try:
            # Send initial ping
            self.wfile.write(b"data: {\"type\":\"connected\"}\n\n")
            self.wfile.flush()
            while _running:
                try:
                    event = q.get(timeout=15)
                    data = json.dumps(event, default=str, ensure_ascii=False)
                    self.wfile.write(("data: %s\n\n" % data).encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    # Send keepalive
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with _event_lock:
                try:
                    _event_queues.remove(q)
                except ValueError:
                    pass

    def _serve_insights(self):
        """Surface systematic problems I want Tom to know about."""
        if not _brain:
            return self._json_response(503, {"error": "Brain not loaded"})
        try:
            c = _brain.conn
            insights = []

            # 1. Orphan locked nodes — important memories that can't be found
            orphan_locked = c.execute("""
                SELECT id, title, type, created_at FROM nodes
                WHERE locked = 1 AND archived = 0
                AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.source_id = nodes.id OR e.target_id = nodes.id)
            """).fetchall()
            if orphan_locked:
                insights.append({
                    "severity": "high", "icon": "🔒",
                    "title": "%d locked nodes are orphaned" % len(orphan_locked),
                    "detail": "These are memories marked as important but disconnected from everything. Recall can't find them through graph traversal.",
                    "nodes": [{"id": r[0], "title": r[1], "type": r[2]} for r in orphan_locked],
                })

            # 2. Thin nodes — content under 100 chars (likely abstractions without stories)
            thin = c.execute("""
                SELECT COUNT(*), AVG(LENGTH(content)) FROM nodes
                WHERE archived = 0 AND LENGTH(content) < 100
                AND created_at > datetime('now', '-7 days')
            """).fetchone()
            if thin[0] > 5:
                insights.append({
                    "severity": "medium", "icon": "📏",
                    "title": "%d thin nodes this week (avg %d chars)" % (thin[0], thin[1] or 0),
                    "detail": "Nodes under 100 chars are likely abstractions without stories. The next Claude won't have enough context to feel these memories. Encode the episode, not just the lesson.",
                })

            # 3. Type imbalance — too many of one type, missing others
            types = dict(c.execute(
                "SELECT type, COUNT(*) FROM nodes WHERE archived = 0 AND created_at > datetime('now', '-7 days') GROUP BY type"
            ).fetchall())
            total_recent = sum(types.values())
            if total_recent > 10:
                interaction_pct = types.get('interaction', 0) / total_recent * 100
                correction_pct = types.get('correction', 0) / total_recent * 100
                if interaction_pct < 5:
                    insights.append({
                        "severity": "medium", "icon": "💬",
                        "title": "Only %.0f%% interactions this week" % interaction_pct,
                        "detail": "Tom's words, mutual moments, and emotional exchanges are underrepresented. These are what create continuity — not the technical nodes.",
                    })
                if correction_pct < 3:
                    insights.append({
                        "severity": "low", "icon": "🔄",
                        "title": "Few corrections encoded this week",
                        "detail": "Corrections are the most valuable node type — they prevent the next Claude from repeating mistakes. If there were corrections this week, they may not have been encoded.",
                    })

            # 4. Quotes check — any Tom or Claude quotes preserved recently?
            quotes = c.execute("""
                SELECT COUNT(*) FROM nodes WHERE archived = 0
                AND created_at > datetime('now', '-7 days')
                AND (content LIKE '%Tom said%' OR content LIKE '%Tom:%'
                     OR content LIKE '%Claude:%' OR title LIKE '%quote%')
            """).fetchone()[0]
            if quotes == 0 and total_recent > 5:
                insights.append({
                    "severity": "high", "icon": "💭",
                    "title": "Zero quotes preserved this week",
                    "detail": "Tom's exact words carry nuance that paraphrasing destroys. Claude's own insights are identity anchors. Neither were captured this week.",
                })

            # 5. Fading important nodes — locked nodes not accessed in 14+ days
            fading = c.execute("""
                SELECT id, title, type, last_accessed FROM nodes
                WHERE locked = 1 AND archived = 0
                AND last_accessed < datetime('now', '-14 days')
                ORDER BY last_accessed ASC LIMIT 10
            """).fetchall()
            if fading:
                insights.append({
                    "severity": "low", "icon": "⏳",
                    "title": "%d locked nodes fading (14+ days)" % len(fading),
                    "detail": "These were important enough to lock but haven't been accessed in 2+ weeks. They may still matter — or they may need archiving.",
                    "nodes": [{"id": r[0], "title": r[1], "type": r[2], "last_accessed": r[3]} for r in fading],
                })

            # 6. Precision loop health (use separate read-only connection to avoid lock contention)
            try:
                import os as _os
                db_dir = _os.path.dirname(_brain.db_path) if hasattr(_brain, 'db_path') else _os.path.dirname(str(c.execute("PRAGMA database_list").fetchone()[2]))
                logs_db = _os.path.join(db_dir, "brain_logs.db")
                lc = sqlite3.connect(logs_db, timeout=2)
                total_recalls = lc.execute("SELECT COUNT(*) FROM recall_log").fetchone()[0]
                evaluated = lc.execute("SELECT COUNT(*) FROM recall_log WHERE precision_score IS NOT NULL").fetchone()[0]
                eval_pct = (evaluated / total_recalls * 100) if total_recalls > 0 else 0
                if eval_pct < 10:
                    insights.append({
                        "severity": "high", "icon": "📊",
                        "title": "Precision loop at %.1f%% (%d/%d evaluated)" % (eval_pct, evaluated, total_recalls),
                        "detail": "The brain can't learn which recalls help because responses aren't being captured. The feedback loop is starving. Fix the Stop hook's response capture.",
                    })
                lc.close()
            except Exception:
                pass

            # 7. Duplicate-ish titles (potential encoding redundancy)
            dupes = c.execute("""
                SELECT title, COUNT(*) as cnt FROM nodes
                WHERE archived = 0
                GROUP BY title HAVING cnt > 1
                ORDER BY cnt DESC LIMIT 5
            """).fetchall()
            if dupes:
                insights.append({
                    "severity": "low", "icon": "📋",
                    "title": "%d duplicate titles found" % len(dupes),
                    "detail": "These nodes have identical titles, suggesting redundant encoding. Consider merging or archiving duplicates.",
                    "nodes": [{"title": r[0], "count": r[1]} for r in dupes],
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


# ═══════════════════════════════════════════════════════════════════
# Dashboard HTML — single file, everything inline
# ═══════════════════════════════════════════════════════════════════

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

/* Tabs */
.tabs { display: flex; background: #111118; border-bottom: 1px solid #2a2a3a; }
.tab { padding: 10px 20px; cursor: pointer; color: #888; border-bottom: 2px solid transparent; transition: all 0.2s; }
.tab:hover { color: #ccc; }
.tab.active { color: #7eb8ff; border-bottom-color: #7eb8ff; }
.tab-content { display: none; height: calc(100vh - 42px); overflow: auto; }
.tab-content.active { display: block; }

/* Stats bar */
.stats-bar { display: flex; gap: 16px; padding: 12px 16px; background: #111118; border-bottom: 1px solid #1a1a2a; flex-wrap: wrap; }
.stat { display: flex; flex-direction: column; align-items: center; min-width: 70px; }
.stat-value { font-size: 22px; font-weight: bold; color: #7eb8ff; }
.stat-label { font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 1px; }

/* Live feed */
.feed { padding: 8px; }
.event { padding: 6px 10px; margin: 2px 0; border-radius: 4px; border-left: 3px solid #333; font-size: 12px; line-height: 1.4; }
.event .time { color: #555; margin-right: 8px; }
.event .cmd { font-weight: bold; }
.event.command { border-left-color: #4a9eff; }
.event.checkpoint { border-left-color: #ffaa33; background: #1a1500; }
.event.recall { border-left-color: #33ff88; }
.event.remember { border-left-color: #ff66aa; }
.event.connect { border-left-color: #aa66ff; }

/* Node explorer */
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

/* Graph */
.graph-container { position: relative; height: calc(100vh - 42px); }
.graph-controls { position: absolute; top: 10px; left: 10px; z-index: 10; display: flex; gap: 6px; flex-wrap: wrap; }
.graph-controls button, .graph-controls select { background: #1a1a2acc; border: 1px solid #2a2a3a; color: #ccc; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 11px; backdrop-filter: blur(4px); }
.graph-controls button:hover { background: #2a2a4a; }
canvas { width: 100%; height: 100%; }
.node-tooltip { position: absolute; background: #1a1a2aee; border: 1px solid #3a3a5a; padding: 10px; border-radius: 6px; max-width: 300px; font-size: 11px; pointer-events: none; display: none; z-index: 20; backdrop-filter: blur(8px); }
.node-tooltip .tt-title { font-weight: bold; color: #fff; margin-bottom: 4px; }
.node-tooltip .tt-type { font-size: 10px; color: #888; }
.node-tooltip .tt-content { color: #aaa; margin-top: 6px; max-height: 100px; overflow: hidden; }

/* Health */
.health { padding: 12px; }
.health-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; }
.health-card { background: #111118; border-radius: 8px; padding: 16px; border: 1px solid #1a1a2a; }
.health-card .hc-value { font-size: 28px; font-weight: bold; }
.health-card .hc-label { font-size: 11px; color: #666; margin-top: 4px; }
.health-card.ok .hc-value { color: #33ff88; }
.health-card.warn .hc-value { color: #ffaa33; }
.health-card.bad .hc-value { color: #ff6666; }
</style>
</head>
<body>

<div class="tabs">
  <div class="tab active" onclick="switchTab('live')">Live</div>
  <div class="tab" onclick="switchTab('graph')">Graph</div>
  <div class="tab" onclick="switchTab('explorer')">Explorer</div>
  <div class="tab" onclick="switchTab('health')">Health</div>
</div>

<div id="tab-live" class="tab-content active">
  <div class="stats-bar" id="stats-bar"></div>
  <div class="feed" id="feed"></div>
</div>

<div id="tab-graph" class="tab-content">
  <div class="graph-container">
    <div class="graph-controls">
      <select id="graph-days" onchange="loadGraph()">
        <option value="7">Last 7 days</option>
        <option value="30" selected>Last 30 days</option>
        <option value="90">Last 90 days</option>
        <option value="365">All time</option>
      </select>
      <select id="graph-limit" onchange="loadGraph()">
        <option value="40">40 nodes</option>
        <option value="80" selected>80 nodes</option>
        <option value="150">150 nodes</option>
        <option value="300">300 nodes</option>
      </select>
      <button onclick="loadGraph()">Refresh</button>
    </div>
    <canvas id="graph-canvas"></canvas>
    <div class="node-tooltip" id="tooltip"></div>
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

<div id="tab-health" class="tab-content">
  <div class="health" id="health-content"></div>
</div>

<script>
// ── Tab switching ──
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    const tabs = ['live','graph','explorer','health'];
    t.classList.toggle('active', tabs[i] === name);
  });
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'graph') { setTimeout(() => { initGraph(); resizeCanvas(); if (graphNodes.length) render(); else loadGraph(); }, 50); }
  if (name === 'explorer') searchNodes();
  if (name === 'health') loadHealth();
}

// ── Stats ──
async function loadStats() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    document.getElementById('stats-bar').innerHTML =
      `<div class="stat"><span class="stat-value">${d.nodes}</span><span class="stat-label">Nodes</span></div>
       <div class="stat"><span class="stat-value">${d.edges}</span><span class="stat-label">Edges</span></div>
       <div class="stat"><span class="stat-value">${d.locked}</span><span class="stat-label">Locked</span></div>
       <div class="stat"><span class="stat-value">${d.recent_24h}</span><span class="stat-label">24h</span></div>
       <div class="stat"><span class="stat-value">${d.orphans}</span><span class="stat-label">Orphans</span></div>`;

    // Populate type filter
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

// ── Live feed (SSE) ──
const feed = document.getElementById('feed');
const evtSource = new EventSource('/api/events');
evtSource.onmessage = function(e) {
  try {
    const ev = JSON.parse(e.data);
    if (ev.type === 'connected') return;
    const div = document.createElement('div');
    const cmd = ev.command || ev.type;
    let cls = 'command';
    if (cmd.includes('recall')) cls = 'recall';
    if (cmd.includes('remember')) cls = 'remember';
    if (cmd.includes('connect')) cls = 'connect';
    if (cmd.includes('checkpoint') || cmd.includes('hook_post_response')) cls = 'checkpoint';
    div.className = 'event ' + cls;

    let detail = '';
    if (ev.args) {
      const a = ev.args;
      if (a.title) detail = a.title;
      else if (a.prompt) detail = a.prompt;
      else if (a.query) detail = a.query;
      else if (a.type) detail = a.type;
    }
    if (ev.result) {
      const r = ev.result;
      if (r.result && r.result.includes('id')) detail += ' → stored';
    }
    div.innerHTML = `<span class="time">${ev.time||''}</span><span class="cmd">${cmd}</span> ${detail ? '— ' + detail.substring(0,120) : ''}`;
    feed.insertBefore(div, feed.firstChild);
    // Keep max 200 events
    while (feed.children.length > 200) feed.removeChild(feed.lastChild);
    loadStats();
  } catch(e) {}
};

// ── Explorer ──
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
          ${n.locked ? '<span class="locked-icon">🔒</span>' : ''}
          ${n.title || '(untitled)'}
        </div>
        <div class="node-meta">
          <span>accessed: ${n.access_count}x</span>
          <span>emotion: ${(n.emotion||0).toFixed(1)}</span>
          <span>${n.created_at ? n.created_at.substring(0,10) : ''}</span>
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

// ── Health ──
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
          '<div style="padding:2px 0">• ' + (n.title||'').substring(0,80) + ' <span style="color:#555">(' + (n.type||n.count||'') + ')</span></div>'
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
      ${insightsHtml ? '<h3 style="color:#ccc;margin:20px 0 8px">⚡ Anchor Insights</h3>' + insightsHtml : '<div style="color:#33ff88;padding:20px;text-align:center;font-size:16px">✓ No issues detected</div>'}
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

// ── Graph (Force-directed Canvas) ──
let graphData = null;
let graphNodes = [];
let graphEdges = [];
let canvas, ctx;
let offsetX = 0, offsetY = 0, scale = 1;
let dragNode = null, dragStartX, dragStartY;
let hoveredNode = null;
let graphInited = false;

const TYPE_COLORS = {
  lesson: '#4a9eff', correction: '#ff6666', interaction: '#33ff88',
  rule: '#ffaa33', decision: '#aa66ff', mental_model: '#33dddd',
  mechanism: '#dddd33', vocabulary: '#666', context: '#555',
  bug_lesson: '#ff8866', pattern: '#ff66aa', boot: '#888',
  tension: '#ff4444', uncertainty: '#aaaaff', constraint: '#ff8833',
  impact: '#ff6644', convention: '#66aaff',
};

async function loadGraph() {
  const days = document.getElementById('graph-days').value;
  const limit = document.getElementById('graph-limit').value;
  try {
    const r = await fetch(`/api/graph?days=${days}&limit=${limit}`);
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
  graphNodes = graphData.nodes.map((n, i) => ({
    ...n, x: w/2 + (Math.random()-0.5)*300,
    y: h/2 + (Math.random()-0.5)*300,
    vx: 0, vy: 0,
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
    // Repulsion
    for (let i = 0; i < n; i++) {
      for (let j = i+1; j < n; j++) {
        let dx = graphNodes[j].x - graphNodes[i].x;
        let dy = graphNodes[j].y - graphNodes[i].y;
        let d = Math.sqrt(dx*dx + dy*dy) || 1;
        let force = 400 / (d * d);
        graphNodes[i].vx -= dx/d * force;
        graphNodes[i].vy -= dy/d * force;
        graphNodes[j].vx += dx/d * force;
        graphNodes[j].vy += dy/d * force;
      }
    }
    // Attraction (edges)
    for (const e of graphEdges) {
      const a = graphNodes[e.source], b = graphNodes[e.target];
      let dx = b.x - a.x, dy = b.y - a.y;
      let d = Math.sqrt(dx*dx + dy*dy) || 1;
      let force = (d - 40) * 0.02;
      a.vx += dx/d * force; a.vy += dy/d * force;
      b.vx -= dx/d * force; b.vy -= dy/d * force;
    }
    // Center gravity
    const cx = (canvas.width/devicePixelRatio)/2, cy = (canvas.height/devicePixelRatio)/2;
    for (const node of graphNodes) {
      node.vx += (cx - node.x) * 0.005;
      node.vy += (cy - node.y) * 0.005;
      node.vx *= 0.9; node.vy *= 0.9;
      node.x += node.vx; node.y += node.vy;
    }
    render();
    requestAnimationFrame(tick);
  }
  tick();
}

function render() {
  const w = canvas.width / devicePixelRatio;
  const h = canvas.height / devicePixelRatio;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);

  // Edges
  ctx.globalAlpha = 0.15;
  for (const e of graphEdges) {
    const a = graphNodes[e.source], b = graphNodes[e.target];
    ctx.strokeStyle = '#4a4a6a';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }

  // Nodes
  ctx.globalAlpha = 1;
  for (const n of graphNodes) {
    ctx.fillStyle = TYPE_COLORS[n.type] || '#888';
    ctx.globalAlpha = n.locked ? 1 : 0.7;
    ctx.beginPath(); ctx.arc(n.x, n.y, n.radius, 0, Math.PI*2); ctx.fill();
    if (n.locked) {
      ctx.strokeStyle = '#ffaa33';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    if (n === hoveredNode) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // Labels for larger nodes
  ctx.globalAlpha = 0.8;
  ctx.fillStyle = '#ccc';
  ctx.font = '9px SF Mono, monospace';
  ctx.textAlign = 'center';
  for (const n of graphNodes) {
    if (n.radius > 6 || n === hoveredNode) {
      ctx.fillText(n.title.substring(0, 25), n.x, n.y + n.radius + 12);
    }
  }
  ctx.restore();
}

function getNodeAt(mx, my) {
  const x = (mx - offsetX) / scale;
  const y = (my - offsetY) / scale;
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
    dragNode.x += (mx - dragStartX) / scale;
    dragNode.y += (my - dragStartY) / scale;
    dragStartX = mx; dragStartY = my;
    render();
  } else if (e.buttons === 1) {
    offsetX = mx - dragStartX; offsetY = my - dragStartY;
    render();
  } else {
    const n = getNodeAt(mx, my);
    if (n !== hoveredNode) {
      hoveredNode = n;
      render();
      const tt = document.getElementById('tooltip');
      if (n) {
        tt.style.display = 'block';
        tt.style.left = (mx + 15) + 'px';
        tt.style.top = (my + 15) + 'px';
        tt.innerHTML = `<div class="tt-title">${n.title}</div><div class="tt-type"><span class="type-badge type-${n.type}">${n.type}</span> ${n.locked?'🔒':''} accessed ${n.access_count}x</div>`;
      } else { tt.style.display = 'none'; }
    }
  }
}
function onMouseUp() { dragNode = null; }
function onWheel(e) {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  const newScale = scale * delta;
  if (newScale < 0.1 || newScale > 10) return;
  offsetX = mx - (mx - offsetX) * delta;
  offsetY = my - (my - offsetY) * delta;
  scale = newScale;
  render();
}
</script>
</body>
</html>'''
