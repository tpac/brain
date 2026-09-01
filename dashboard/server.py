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

from . import db
from . import log as dashboard_log
from .daemon_client import DAEMON_PORT, daemon_alive, daemon_send
from . import contract
from .queries import (
    aspects,
    setup,
    encoding,
    errors,
    explorer,
    graph,
    insights_scanner,
    journals,
    recalls,
    s2_runs,
    self_channel,
    sessions,
    stats,
    system,
    traces,
)

# Dashboard listens here. Override with DASHBOARD_PORT env var (eval brains
# usually run a second dashboard on a different port). The launchd singleton
# (com.brain.dashboard) sets DASHBOARD_PORT=47303 explicitly. `or 47303` (not a
# dict default) so an empty-string DASHBOARD_PORT falls back instead of crashing.
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT") or 47303)

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
        if not self._loopback_guard():
            return self._json(403, {"error": "Forbidden"})
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._serve_static_file("index.html")
        elif path.startswith("/static/"):
            self._serve_static_file(path[len("/static/"):])

        # First-run setup (API key entry — the no-terminal onboarding path;
        # keyless boot notices point here)
        elif path == "/setup":
            self._serve_static_file("setup.html")
        elif path == "/api/setup-state":
            self._json(200, {"key_present": setup.api_key_present()})

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
        elif path == "/api/insights/live":
            # First adopter of the Prometheus envelope. Phase 2 migration
            # policy: new endpoints emit the envelope; existing routes
            # convert when their caller's render is being rewritten.
            self._json(200, contract.envelope_ok(insights_scanner.scan_all()))
        elif path == "/api/aspects":
            self._json(200, {"aspects": aspects.query_aspects()})
        elif path == "/api/dashboard-errors":
            # The dashboard's own error feed. Reads from log.py's ring buffer
            # — every warn() in queries/* / server.py / db.py lands here.
            # `?clear=1` resets the badge after the operator reads it.
            if params.get("clear", [""])[0] == "1":
                dashboard_log.clear()
                self._json(200, {"errors": [], "count": 0, "cleared": True})
            else:
                limit = int(params.get("limit", ["100"])[0])
                items = dashboard_log.recent(limit=limit)
                self._json(200, {"errors": items, "count": len(items)})

        # Live decoding feed
        elif path == "/api/recalls":
            self._serve_recalls(params)
        elif path == "/api/recall-prompt":
            recall_ref = params.get("recall_ref", [""])[0]
            chain_id = params.get("chain_id", [""])[0]
            pointer = params.get("pointer", [""])[0]
            self._json(200, recalls.query_recall_prompt(
                recall_ref=recall_ref, chain_id=chain_id, pointer=pointer))
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
            session_id = params.get("session_id", [""])[0]
            self._json(200, {"runs": encoding.query_encoding_runs(
                limit=limit, hours=hours, session_id=session_id)})
        elif path == "/api/encoding-prompt":
            chain_id = params.get("chain_id", [""])[0]
            self._json(200, encoding.query_encoding_prompt(chain_id=chain_id))

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
        elif path == "/api/aspect-runs":
            hours = int(params.get("hours", ["24"])[0])
            self._json(200, {"runs": s2_runs.query_aspect_runs(hours=hours)})
        elif path == "/api/consolidation-prompt":
            chain_id = params.get("chain_id", [""])[0]
            self._serve_consolidation_prompt(chain_id)

        # Encoder journals — every encoder's residue, one shape, one reader.
        elif path == "/api/journals":
            self._json(200, {"notes": journals.query_journal_notes(
                hours=int(params.get("hours", ["48"])[0]),
                scale=params.get("scale", [""])[0],
                unit=params.get("unit", [""])[0],
                session_id=params.get("session_id", [""])[0],
                tag=params.get("tag", [""])[0],
                subject=params.get("subject", [""])[0],
                limit=int(params.get("limit", ["300"])[0]))})
        elif path == "/api/journals/summary":
            self._json(200, journals.query_journal_summary(
                hours=int(params.get("hours", ["48"])[0])))

        # Traces
        elif path == "/api/traces":
            hours = int(params.get("hours", ["24"])[0])
            scale = params.get("scale", [""])[0]
            session_id = params.get("session", [""])[0]
            self._json(200, traces.query_traces(hours=hours, scale=scale, session_id=session_id))

        # Self-channel: stream↔stream messages + faithful boot captures
        elif path == "/api/self-messages":
            hours = int(params.get("hours", ["48"])[0])
            limit = int(params.get("limit", ["200"])[0])
            self._json(200, {"messages": self_channel.query_messages(hours=hours, limit=limit)})
        elif path == "/api/boot-renders":
            session_id = params.get("session", [""])[0]
            limit = int(params.get("limit", ["30"])[0])
            self._json(200, {"renders": self_channel.query_boot_renders(session_id=session_id, limit=limit)})
        elif path == "/api/self-presence":
            # Live roster — must go through the daemon (window + ranking logic).
            # Omit session_id so the operator sees every live stream. Returns
            # {streams:[], line:''} or an empty roster when the daemon is down.
            limit = int(params.get("limit", ["10"])[0])
            result = daemon_send("self_presence", {"limit": limit})
            self._json(200, result if isinstance(result, dict) else {"streams": [], "line": ""})

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

        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        """The dashboard's POST surfaces. The DB invariant holds for all three
        (CLAUDE.md: passive observer, never writes the DBs):

        * /api/self-send   → hands off to the daemon (the single writer) over
                             TCP, same as any MCP client. No write connection.
        * /api/setup-key   → writes ONE user-config file
                             (~/.config/brain/env, mode 600) — the no-terminal
                             onboarding path for the API key. Localhost-only
                             (server binds 127.0.0.1). Never touches a DB.
        * /api/recall      → runs the REAL recall pipeline through the daemon
                             with mark_accessed=False. POST because it carries
                             a query body, not because it writes: the flag makes
                             it a pure read, so the operator's searching never
                             enters access_count / fatigue and never shows up as
                             recall heat in the graph. Observing, not requesting.
        """
        if not self._loopback_guard():
            return self._json(403, {"error": "Forbidden"})
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/self-send":
            self._handle_self_send()
        elif path == "/api/setup-key":
            self._handle_setup_key()
        elif path == "/api/recall":
            self._handle_recall()
        else:
            self._json(404, {"error": "Not found"})

    def _handle_setup_key(self):
        """Accept {"api_key": "sk-..."} and persist it to the brain env file.

        The key value is treated as a secret end-to-end: validated, written
        0600, never logged, never echoed back in the response.
        """
        body, err = self._read_json_body()
        if err:
            return self._json(400, {"ok": False, "error": err})
        key = (body.get("api_key") or "").strip()
        ok, msg = setup.write_api_key(setup.brain_env_path(), key)
        self._json(200 if ok else 400, {"ok": ok, "message": msg})

    # Largest legitimate POST body (self-send message / setup key) is well
    # under this. Caps the read so a hostile Content-Length can't tie up a
    # handler thread allocating gigabytes.
    _MAX_BODY_BYTES = 65536

    def _loopback_guard(self) -> bool:
        """True when the request provably came from this machine's own pages.

        Two checks, closing two browser-as-confused-deputy attacks
        (security review 2026-07-17, findings 1+3):
        - Host must be loopback → kills DNS rebinding (attacker.com resolved
          to 127.0.0.1 arrives with Host: attacker.com).
        - Origin, when the browser sends one, must be a loopback origin →
          kills cross-site POSTs (CSRF); same-origin requests always pass.
        The server only ever binds 127.0.0.1, so legitimate traffic always
        satisfies both.
        """
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in ("127.0.0.1", "localhost", "::1"):
            return False
        origin = self.headers.get("Origin")
        if origin:
            try:
                ohost = (urlparse(origin).hostname or "").strip("[]")
            except ValueError:
                return False
            if ohost not in ("127.0.0.1", "localhost", "::1"):
                return False
        return True

    def _read_json_body(self):
        """Parse the request body as JSON; ({}, error_str) on any failure.

        Content-Type must be application/json: cross-origin fetch() with a
        JSON Content-Type is NOT a "simple request", so the browser
        preflights it — and this server answers no preflight (no
        do_OPTIONS, no CORS headers), so hostile pages are blocked. A
        text/plain body would sail through preflight-free, which is exactly
        the CSRF hole this check closes (security review 2026-07-17,
        finding 1).
        """
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        if ctype.lower() != "application/json":
            return {}, "Content-Type must be application/json"
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            length = 0
        if not length:
            return {}, "empty body"
        if length > self._MAX_BODY_BYTES:
            return {}, "body too large"
        try:
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8")), None
        except Exception as e:
            return {}, str(e)

    def _handle_recall(self):
        """Run the operator's query through the brain's own recall pipeline.

        Not a search box over the graph payload — the actual thing: LAF
        scoring, graph expansion, the same ranking a session gets. That's the
        point; a separate similarity search would answer a question the brain
        never asks. Requires the daemon (embeddings live in that process).

        mark_accessed=False and source='dashboard' are BOTH non-negotiable
        here: the first keeps the read out of the brain's heat/fatigue record,
        the second labels it honestly wherever it does appear.
        """
        body, err = self._read_json_body()
        if err:
            return self._json(400, {"ok": False, "error": "bad request body: %s" % err})
        query = (body.get("query") or "").strip()
        if not query:
            return self._json(400, {"ok": False, "error": "'query' is required"})
        try:
            limit = max(1, min(int(body.get("limit") or 12), 40))
        except (TypeError, ValueError):
            limit = 12
        if not daemon_alive():
            return self._json(503, {"ok": False, "error":
                "daemon is not running — recall needs the embedder, which "
                "lives in the daemon process"})
        # daemon_send ALREADY unwraps the daemon's {ok, result} envelope: it
        # returns the payload on success and None on any failure (see
        # daemon_client). Checking for `ok`/`result` here looked for keys that
        # were already stripped, so every successful recall was read as a
        # failure and reported as a flat "recall failed". The other caller in
        # this file (self_presence) had the convention right.
        payload = daemon_send("recall", {
            "query": query, "limit": limit,
            "source": "dashboard", "mark_accessed": False,
        })
        if not isinstance(payload, dict):
            return self._json(502, {"ok": False, "error":
                "the daemon did not answer the recall (down, timed out, or the "
                "command failed) — check the Logs tab"})
        # Trim to what the panel renders. The raw recall result carries full
        # node content plus several telemetry blocks (~100KB for 12 results);
        # the panel shows a ranked list, so shipping the rest is pure weight.
        nodes = []
        for n in (payload.get("results") or []):
            nodes.append({
                "id": n.get("id", ""),
                "title": n.get("title", ""),
                "type": n.get("type", ""),
                "content": (n.get("content") or "")[:400],
                "situation": (n.get("situation") or "")[:220],
                "confidence": n.get("confidence"),
                # `effective_activation` is where recall puts the blended
                # score on a result node (brain_recall STEP 7). It is NOT
                # exposed as `blended_score` — that name only exists on the
                # internal scored_results rows, so reading it here yielded
                # None for every result and the panel's score bars silently
                # rendered empty.
                "score": n.get("effective_activation"),
                "similarity": n.get("embedding_similarity"),
                "discovery": n.get("_discovery", ""),
                "created_at": n.get("created_at", ""),
                "last_accessed": n.get("last_accessed", ""),
                "access_count": n.get("access_count", 0),
            })
        stats = payload.get("_embedding_stats") or {}
        retr = payload.get("_retrieval_stats") or {}
        return self._json(200, {
            "ok": True,
            "query": query,
            "results": nodes,
            "mode": payload.get("_recall_mode", ""),
            "recall_ms": stats.get("recall_ms"),
            "brain_size": retr.get("brain_size"),
            "candidates": retr.get("candidates_after_floor"),
            "by_source": stats.get("results_by_source") or {},
        })

    def _handle_self_send(self):
        """Operator-authored send into the self-channel courier → daemon's
        self_send. The operator is not a stream of thought, so from_session
        carries a recognizable label ('operator-dashboard') rather than a
        session id — it renders attributed, never masquerading as another-me."""
        body, err = self._read_json_body()
        if err:
            return self._json(400, {"ok": False, "error": "bad request body: %s" % err})
        to = (body.get("to") or "").strip()
        message = (body.get("body") or "").strip()
        if not to or not message:
            return self._json(400, {"ok": False, "error": "both 'to' and 'body' are required"})
        if not daemon_alive():
            return self._json(503, {"ok": False, "error": "daemon unavailable — cannot send"})
        args = {
            "to": to,
            "body": message,
            "from_session": body.get("from_session") or "operator-dashboard",
        }
        result = daemon_send("self_send", args)
        if result is None:
            return self._json(502, {"ok": False, "error": "daemon returned no result"})
        self._json(200, {"ok": True, "result": result})

    # ── Response helpers ──

    def _json(self, code: int, data):
        # NO Access-Control-Allow-Origin header — the dashboard is strictly
        # same-origin (all JS uses relative fetches). The old wildcard `*`
        # let ANY website the user visited read the whole brain API
        # cross-origin (memory nodes, traces, recalls) despite the loopback
        # bind — the browser is the confused deputy. Security review
        # 2026-07-17, finding 2.
        body = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
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
        # no-cache: the browser must revalidate before reusing a cached asset,
        # so a dashboard deploy (new JS/CSS) takes effect on the next normal
        # refresh instead of silently running stale code until a hard-refresh.
        # Assets are tiny and served from localhost, so refetch cost is nil.
        self.send_header("Cache-Control", "no-cache")
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

    def _serve_consolidation_prompt(self, chain_id: str):
        # The S2 consolidation encoder records each batch's prompt via
        # brain.record_payload (full content); the dashboard reads it back
        # by chain layout from {BRAIN_DB_DIR}/payloads/ — direct file read,
        # daemon-down safe. Keyed by the run's chain_id (was: a batch number
        # against one hardcoded /tmp file, which only ever showed batch 1).
        if not chain_id:
            return self._json(200, {"error": "chain_id required"})
        files = db.chain_payload_files(chain_id, kind="prompt")
        if not files:
            # Legacy branch (time-bounded — delete once default views can't
            # reach pre-migration runs): chains from before the migration
            # recorded no payloads, but the retired writer's batch-1 tmp
            # file may still be on disk.
            legacy = os.path.join(os.environ.get('BRAIN_TMP_DIR', '/tmp'),
                                  'brain-consolidation-prompt-1.json')
            try:
                with open(legacy) as f:
                    return self._json(200, {
                        "user_content": json.load(f).get("user_content", ""),
                    })
            except OSError:
                pass
            except Exception as e:
                return self._json(200, {"error": str(e)})
            return self._json(200, {"error": "no prompt payload for %s "
                                             "(pruned, or recording gated "
                                             "off)" % chain_id})
        # Full prompts are 100KB+ each — bound the joined response so a
        # 10-batch run can't ship a multi-MB blob to one card expand.
        CAP = 500_000
        parts, used = [], 0
        for p in files:
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    body = f.read()
            except OSError as e:
                body = "(unreadable: %s)" % e
            if len(files) > 1:
                body = "── %s ──\n%s" % (os.path.basename(p), body)
            if used + len(body) > CAP:
                parts.append("… (%d more file(s) omitted — read them under "
                             "payloads/*/%s/)" % (
                                 len(files) - len(parts), chain_id))
                break
            parts.append(body)
            used += len(body)
        self._json(200, {"user_content": "\n\n".join(parts)})


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
