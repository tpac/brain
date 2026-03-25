#!/usr/bin/env python3
"""
Brain MCP Server — Thin stdio proxy to brain daemon.

Zero-dependency MCP server (JSON-RPC 2.0 over stdio).
Forwards tool calls to the brain daemon via TCP localhost.
Embedder loads once in the daemon; this process is just a relay.

Error policy: NEVER swallow errors silently. If something fails,
stderr gets a message and the caller gets a real error.
"""

import json
import os
import sys
import socket

# ── Daemon communication ──

DAEMON_HOST = "127.0.0.1"  # Client connects via IPv4 loopback
DAEMON_PORT = 47200 + (os.getuid() % 100)
_last_daemon_fingerprint = None  # Track daemon restarts


def daemon_send(cmd, args=None, timeout=30.0):
    """Send command to brain daemon via TCP, return result dict."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((DAEMON_HOST, DAEMON_PORT))
        msg = json.dumps({"cmd": cmd, "args": args or {}}) + "\n"
        sock.sendall(msg.encode("utf-8"))
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        if data:
            return json.loads(data.decode("utf-8").strip())
        return {"ok": False, "error": "Empty response from daemon"}
    except socket.timeout:
        return {"ok": False, "error": "Daemon timeout ({}s)".format(timeout)}
    except Exception as e:
        return {"ok": False, "error": "Daemon connection error: {}".format(e)}
    finally:
        sock.close()


def ensure_daemon_running():
    """Start daemon if not running. Returns True if ready."""
    # Try ping first — fast path
    resp = daemon_send("ping", timeout=2.0)
    if resp.get("ok"):
        return True

    # Daemon not running — try to start it
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent not in sys.path:
        sys.path.insert(0, parent)

    try:
        from servers.daemon import ensure_daemon
        db_dir = os.environ.get("BRAIN_DB_DIR", "")
        if not db_dir:
            # Resolve DB path
            home = os.path.expanduser("~")
            candidate = os.path.join(home, "AgentsContext", "brain")
            if os.path.isfile(os.path.join(candidate, "brain.db")):
                db_dir = candidate

        if not db_dir:
            return False

        return ensure_daemon(os.path.join(db_dir, "brain.db"))
    except Exception as e:
        sys.stderr.write("[brain-mcp] Failed to start daemon: {}\n".format(e))
        return False


# ── MCP Protocol ──

SERVER_NAME = "brain"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2024-11-05"

# Tool definitions — what Claude sees as native tools
# Memory operations only. No operational tools (ping, save, health_check, config).
# Daemon self-manages; hooks use internal commands directly.
TOOLS = [
    # ── Core memory operations ──
    {"name": "recall",
     "description": "Semantic recall from brain — searches nodes by meaning using embeddings. Returns ranked results with titles, content, types, confidence.",
     "inputSchema": {"type": "object", "required": ["query"], "properties": {
         "query": {"type": "string", "description": "Search query (semantic, not keyword)"},
         "limit": {"type": "integer", "description": "Max results (default 8)", "default": 8}}}},
    {"name": "remember",
     "description": "Store a new node in the brain. Types: decision, rule, lesson, concept, context, pattern, convention, mechanism, impact, constraint, purpose, mental_model, uncertainty, vocabulary, hypothesis, tension, aspiration, catalyst, interaction, meta_learning, failure_mode, performance, capability, arch_constraint, code_concept, fn_reasoning, param_influence, comment_anchor, bug_lesson.",
     "inputSchema": {"type": "object", "required": ["type", "title", "content"], "properties": {
         "type": {"type": "string", "description": "Node type"},
         "title": {"type": "string", "description": "Specific, scannable title"},
         "content": {"type": "string", "description": "Rich content with reasoning, tradeoffs, specifics"},
         "locked": {"type": "boolean", "description": "Lock node (for decisions, rules, lessons)", "default": False},
         "confidence": {"type": "number", "description": "Confidence 0.0-1.0", "default": 1.0},
         "keywords": {"type": "string", "description": "Space-separated keywords for search"},
         "project": {"type": "string", "description": "Project scope"},
         "emotion": {"type": "number", "description": "Emotional valence -1.0 to 1.0"}}}},
    {"name": "connect",
     "description": "Create a weighted edge between two brain nodes. Relations: related_to, caused_by, depends_on, contradicts, supports, produced, evolved_from, blocks, enables, example_of.",
     "inputSchema": {"type": "object", "required": ["source_id", "target_id"], "properties": {
         "source_id": {"type": "string", "description": "Source node ID"},
         "target_id": {"type": "string", "description": "Target node ID"},
         "relation": {"type": "string", "description": "Edge relation type", "default": "related_to"},
         "weight": {"type": "number", "description": "Edge weight 0.0-1.0", "default": 0.5}}}},
    {"name": "enrich",
     "description": "Store V5 enrichment vectors for a node (after filling in the enrichment_prompt from remember()). Pass the generated question, anchor phrase, bridge sentence, and/or keywords. Each is embedded and stored for improved recall.",
     "inputSchema": {"type": "object", "required": ["node_id"], "properties": {
         "node_id": {"type": "string", "description": "Node ID to enrich (from remember() response)"},
         "question": {"type": "string", "description": "One question a user would ask that leads to this node"},
         "anchor": {"type": "string", "description": "3-5 word phrase using neighbor vocabulary"},
         "bridge": {"type": "string", "description": "One sentence connecting this node to its most important neighbor"},
         "keywords": {"type": "string", "description": "Comma-separated keywords borrowed from neighbors"}}}},

    # ── Specialized encoding (promoted from eval) ──
    {"name": "remember_lesson",
     "description": "Store a lesson learned — what happened, root cause, fix, and preventive principle. Auto-locked, structured content.",
     "inputSchema": {"type": "object", "required": ["title", "what_happened", "root_cause", "fix", "preventive_principle"], "properties": {
         "title": {"type": "string", "description": "Lesson title"},
         "what_happened": {"type": "string", "description": "What went wrong or was discovered"},
         "root_cause": {"type": "string", "description": "Why it happened"},
         "fix": {"type": "string", "description": "How it was fixed"},
         "preventive_principle": {"type": "string", "description": "General principle to prevent recurrence"}}}},
    {"name": "remember_impact",
     "description": "Store what changes ripple where — the connectivity layer. What must be checked if something changes.",
     "inputSchema": {"type": "object", "required": ["title", "if_changed", "must_check", "because"], "properties": {
         "title": {"type": "string", "description": "Impact title"},
         "if_changed": {"type": "string", "description": "What might change"},
         "must_check": {"type": "string", "description": "What must be verified"},
         "because": {"type": "string", "description": "Why the dependency exists"}}}},
    {"name": "remember_mechanism",
     "description": "Store how something works — flows, algorithms, interactions.",
     "inputSchema": {"type": "object", "required": ["title", "content"], "properties": {
         "title": {"type": "string", "description": "Mechanism title"},
         "content": {"type": "string", "description": "How it works"},
         "steps": {"type": "array", "items": {"type": "string"}, "description": "Step-by-step flow"},
         "data_flow": {"type": "string", "description": "Data flow description"}}}},
    {"name": "remember_convention",
     "description": "Store patterns, utilities, coding style for a codebase.",
     "inputSchema": {"type": "object", "required": ["title", "content"], "properties": {
         "title": {"type": "string", "description": "Convention title"},
         "content": {"type": "string", "description": "The convention/pattern"},
         "pattern": {"type": "string", "description": "Example of correct usage"},
         "anti_pattern": {"type": "string", "description": "Example of what NOT to do"}}}},
    {"name": "remember_uncertainty",
     "description": "Store where you know you don't understand — honest not-knowing. Low confidence, encourages future investigation.",
     "inputSchema": {"type": "object", "required": ["title", "what_unknown", "why_it_matters"], "properties": {
         "title": {"type": "string", "description": "Uncertainty title"},
         "what_unknown": {"type": "string", "description": "What is not understood"},
         "why_it_matters": {"type": "string", "description": "Why resolving this matters"}}}},
    {"name": "remember_mental_model",
     "description": "Store your understanding of how systems/processes work. Confidence reflects how sure you are.",
     "inputSchema": {"type": "object", "required": ["title", "model_description"], "properties": {
         "title": {"type": "string", "description": "Mental model title"},
         "model_description": {"type": "string", "description": "How you understand this system/process works"},
         "applies_to": {"type": "string", "description": "What domain/system this applies to"},
         "confidence": {"type": "number", "description": "How confident in this model (0.0-1.0)", "default": 0.7}}}},
    {"name": "record_divergence",
     "description": "Track where your model diverged from reality — corrections. Creates correction traces and adjusts related node confidence.",
     "inputSchema": {"type": "object", "required": ["claude_assumed", "reality", "underlying_pattern"], "properties": {
         "claude_assumed": {"type": "string", "description": "What you assumed was true"},
         "reality": {"type": "string", "description": "What turned out to be true"},
         "underlying_pattern": {"type": "string", "description": "The general pattern behind this divergence"},
         "severity": {"type": "string", "description": "minor, medium, significant, or critical", "default": "medium"}}}},
    {"name": "learn_vocabulary",
     "description": "Map an operator term to its meaning — vocabulary learning. Improves recall by expanding queries with mapped terms.",
     "inputSchema": {"type": "object", "required": ["term", "maps_to", "context"], "properties": {
         "term": {"type": "string", "description": "The term as the operator uses it"},
         "maps_to": {"type": "string", "description": "What it means in brain/code context"},
         "context": {"type": "string", "description": "Where/how this term is used"}}}},

    # ── Compound operations ──
    {"name": "encode_cluster",
     "description": "Compound encoding — store multiple nodes in one call with auto-connections. Accepts inline enrichments, fuzzy-matches connect_to titles, auto-finds related nodes. Returns quality feedback (duplicates, missing enrichments, suggested connections).",
     "inputSchema": {"type": "object", "required": ["nodes"], "properties": {
         "nodes": {"type": "array", "description": "List of node specs: [{type, title, content, keywords?, enrichment?: {question?, anchor?, bridge?, keywords?}}]",
                   "items": {"type": "object", "required": ["type", "title", "content"], "properties": {
                       "type": {"type": "string"}, "title": {"type": "string"}, "content": {"type": "string"},
                       "keywords": {"type": "string"}, "locked": {"type": "boolean"},
                       "enrichment": {"type": "object", "properties": {
                           "question": {"type": "string"}, "anchor": {"type": "string"},
                           "bridge": {"type": "string"}, "keywords": {"type": "string"}}}}}},
         "connect_to": {"type": "array", "items": {"type": "string"}, "description": "Existing node titles to fuzzy-match and connect to (no UUIDs needed)"},
         "auto_connect": {"type": "boolean", "description": "Auto-find and connect to related existing nodes", "default": True}}}},
    {"name": "find_node_by_title",
     "description": "Find an existing node by fuzzy title matching using embedding similarity. Returns best match above threshold with context (content snippet, keywords) for verification. Default threshold 0.75 is conservative.",
     "inputSchema": {"type": "object", "required": ["title_query"], "properties": {
         "title_query": {"type": "string", "description": "Title to search for (fuzzy match)"},
         "threshold": {"type": "number", "description": "Minimum similarity (0.0-1.0, default 0.75)", "default": 0.75},
         "top_k": {"type": "integer", "description": "Return top K matches (default 1)", "default": 1}}}},

    # ── Introspection ──
    {"name": "consciousness",
     "description": "Get brain consciousness signals — fading knowledge, tensions, vocabulary gaps, encoding health, errors, mental model drift, uncertainties, dream insights, reminders.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "engineering_context",
     "description": "Get engineering memory context — mechanisms, impacts, constraints, conventions for a project.",
     "inputSchema": {"type": "object", "properties": {
         "project": {"type": "string", "default": "default"}}}},

    # ── Escape hatch ──
    {"name": "eval",
     "description": "Escape hatch — evaluate arbitrary Python expression on brain object. Variable 'brain' is the Brain instance. Use for methods not exposed as direct tools.",
     "inputSchema": {"type": "object", "required": ["code"], "properties": {
         "code": {"type": "string", "description": "Python expression to eval (brain object available as 'brain')"}}}},
]


def make_response(request_id, result):
    """Build a JSON-RPC 2.0 response."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id, code, message):
    """Build a JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_initialize(request_id):
    return make_response(request_id, {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
    })


def handle_tools_list(request_id):
    return make_response(request_id, {"tools": TOOLS})


def handle_tools_call(request_id, params):
    import time as _time
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    # Try up to 3 times with backoff — daemon may be restarting
    backoff = [0, 0.5, 1.5]  # immediate, 0.5s, 1.5s
    last_error = ""
    for attempt, delay in enumerate(backoff):
        if delay > 0:
            _time.sleep(delay)

        resp = daemon_send(tool_name, arguments)
        if resp.get("ok"):
            result_text = json.dumps(resp["result"], indent=2, default=str)
            return make_response(request_id, {
                "content": [{"type": "text", "text": result_text}]
            })

        last_error = resp.get("error", "Unknown daemon error")
        is_connection_error = "connection" in last_error.lower() or "timeout" in last_error.lower()

        if is_connection_error and attempt < len(backoff) - 1:
            sys.stderr.write("[brain-mcp] Attempt {}: {} — restarting daemon...\n".format(attempt + 1, last_error))
            ensure_daemon_running()
            check_daemon_fingerprint()
        else:
            break

    return make_response(request_id, {
        "content": [{"type": "text", "text": "ERROR: {}".format(last_error)}],
        "isError": True
    })


def handle_ping(request_id):
    return make_response(request_id, {})


def send(msg):
    """Write JSON-RPC message to stdout."""
    line = json.dumps(msg)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def send_notification(method):
    """Send a JSON-RPC 2.0 notification (no id, no response expected)."""
    send({"jsonrpc": "2.0", "method": method})


def check_daemon_fingerprint():
    """Check if daemon restarted (new code). If so, notify Claude Code to refresh tools."""
    global _last_daemon_fingerprint
    resp = daemon_send("ping", timeout=3.0)
    if not resp.get("ok"):
        return
    fp = resp.get("result", {}).get("code_fingerprint")
    if fp and _last_daemon_fingerprint and fp != _last_daemon_fingerprint:
        sys.stderr.write("[brain-mcp] Daemon fingerprint changed: {} → {} — notifying tools/list_changed\n".format(
            _last_daemon_fingerprint, fp))
        send_notification("notifications/tools/list_changed")
    _last_daemon_fingerprint = fp


def _read_stdin():
    """Read lines from stdin, surviving EOF and IO errors gracefully."""
    try:
        for line in sys.stdin:
            yield line
    except (IOError, BrokenPipeError, KeyboardInterrupt):
        pass
    sys.stderr.write("[brain-mcp] stdin closed — shutting down cleanly.\n")


def main():
    # Ensure daemon is running — retry a few times since boot hook may be starting it concurrently
    sys.stderr.write("[brain-mcp] Starting MCP server...\n")
    import time
    daemon_ready = False
    for attempt in range(4):
        if ensure_daemon_running():
            daemon_ready = True
            break
        if attempt < 3:
            sys.stderr.write("[brain-mcp] Daemon not ready, retry {}/3 in 2s...\n".format(attempt + 1))
            time.sleep(2)
    if daemon_ready:
        check_daemon_fingerprint()  # Record initial fingerprint
        sys.stderr.write("[brain-mcp] Daemon connected. Serving {} tools.\n".format(len(TOOLS)))
    else:
        sys.stderr.write("[brain-mcp] WARNING: Daemon not available at startup. Will retry on each tool call.\n")

    # Main loop — read JSON-RPC from stdin
    # Never crash: daemon going down/up is normal. Surface errors, keep serving.
    for line in _read_stdin():
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stderr.write("[brain-mcp] Bad JSON: {}\n".format(e))
            continue

        method = msg.get("method", "")
        request_id = msg.get("id")
        params = msg.get("params", {})

        # Notifications (no id) — acknowledge silently
        if request_id is None:
            if method == "notifications/initialized":
                pass  # Client acknowledged init
            continue

        try:
            if method == "initialize":
                send(handle_initialize(request_id))
            elif method == "tools/list":
                send(handle_tools_list(request_id))
            elif method == "tools/call":
                send(handle_tools_call(request_id, params))
            elif method == "ping":
                send(handle_ping(request_id))
            else:
                send(make_error(request_id, -32601, "Method not found: {}".format(method)))
        except Exception as e:
            sys.stderr.write("[brain-mcp] Unhandled error in {}: {}\n".format(method, e))
            try:
                send(make_error(request_id, -32603, "Internal MCP error: {}".format(e)))
            except Exception:
                pass  # stdout broken — nothing we can do


if __name__ == "__main__":
    main()
