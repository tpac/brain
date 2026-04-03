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

# Ensure parent dir is on sys.path so `from servers.X` works
# even when this file is run as a standalone script (not -m servers.brain_mcp)
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# ── Daemon communication ──

DAEMON_HOST = "127.0.0.1"  # Client connects via IPv4 loopback
DAEMON_PORT = 47200 + (os.getuid() % 100)
_last_daemon_fingerprint = None  # Track daemon restarts


# ── Contract-driven tool schema generation ──

def _generate_remember_schema():
    """Generate the 'remember' MCP tool schema from the contract."""
    from .contract import get_remember_fields as get_writable_fields

    TYPE_MAP = {"str": "string", "float": "number", "bool": "boolean", "int": "integer"}

    properties = {}
    for name, spec in get_writable_fields().items():
        prop = {"type": TYPE_MAP.get(spec.get("type", "str"), "string")}
        if spec.get("description"):
            prop["description"] = spec["description"]
        elif name == "type":
            prop["description"] = "Node type (decision, rule, lesson, mechanism, vocabulary, etc.)"
        elif name == "title":
            prop["description"] = "Specific, scannable title"
        elif name == "content":
            prop["description"] = "Rich content with reasoning, tradeoffs, specifics"
        elif name == "keywords":
            prop["description"] = "Space-separated keywords for search"
        if spec.get("default") is not None:
            prop["default"] = spec["default"]
        properties[name] = prop

    return {
        "name": "remember",
        "description": "Store a new node in the brain. Fields defined by contract — add new fields there, they appear here automatically.",
        "inputSchema": {
            "type": "object",
            "required": ["type", "title", "content"],
            "properties": properties,
        }
    }


def _generate_remember_batch_schema():
    """Generate the 'remember_batch' tool schema — array of remember() objects."""
    remember_schema = _generate_remember_schema()
    node_properties = remember_schema["inputSchema"]["properties"]
    return {
        "name": "remember_batch",
        "description": "Create multiple nodes in one call. Each node uses the same fields as remember(). Auto-connects new nodes to each other and to existing nodes matched by title.",
        "inputSchema": {
            "type": "object",
            "required": ["nodes"],
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "Array of node specs — same fields as remember()",
                    "items": {
                        "type": "object",
                        "required": ["type", "title", "content"],
                        "properties": node_properties,
                    },
                },
                "connect_to": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Existing node title to fuzzy-match"},
                            "why": {"type": "string", "description": "Why these are connected — what the relationship means"}
                        },
                        "required": ["title", "why"]
                    },
                    "description": "Existing node titles to connect to, with description of why they're connected.",
                },
                "auto_connect": {
                    "type": "boolean",
                    "description": "Auto-connect new nodes to each other",
                    "default": True,
                },
            },
        },
    }


def _generate_revise_schema():
    """Generate the 'revise' MCP tool schema from the contract."""
    from .contract import get_writable_fields

    TYPE_MAP = {"str": "string", "float": "number", "bool": "boolean", "int": "integer"}

    properties = {
        "node_id": {"type": "string", "description": "Full node ID to revise"},
        "reason": {"type": "string", "description": "Why this revision"},
    }
    for name, spec in get_writable_fields().items():
        prop = {"type": TYPE_MAP.get(spec.get("type", "str"), "string")}
        desc = spec.get("description", "")
        if spec.get("append_on_revise"):
            desc = (desc + " " if desc else "") + "(appended on revise, preserves history)"
        else:
            desc = (desc + " " if desc else "") + "(replaces existing value)"
        prop["description"] = desc.strip()
        properties[name] = prop

    return {
        "name": "revise",
        "description": "Update any field(s) on an existing brain node. Content is appended with revision history. All other fields are replaced.",
        "inputSchema": {
            "type": "object",
            "required": ["node_id", "reason"],
            "properties": properties,
        }
    }


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
    """Check if daemon is alive. Does NOT start it.

    Daemon lifecycle is managed by launchd (com.brain.daemon).
    The MCP plugin only connects — it never spawns the daemon.
    This prevents race conditions from multiple sessions/hooks competing.
    """
    resp = daemon_send("ping", timeout=3.0)
    if resp.get("ok"):
        return True

    sys.stderr.write("[brain-mcp] Daemon not responding. Managed by launchd — check: launchctl list | grep brain\n")
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
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Search query (semantic, not keyword)"},
         "node_id": {"type": "string", "description": "Look up a specific node by ID (skip search)"},
         "limit": {"type": "integer", "description": "Max results (default 8)", "default": 8},
         "neighbor_limit": {"type": "integer", "description": "Max neighbor nodes to include (default 3)", "default": 3}}}},
    _generate_remember_schema(),
    _generate_remember_batch_schema(),
    {"name": "connect",
     "description": "Create a weighted edge between two brain nodes. Relations: related_to, caused_by, depends_on, contradicts, supports, produced, evolved_from, blocks, enables, example_of.",
     "inputSchema": {"type": "object", "required": ["source_id", "target_id"], "properties": {
         "source_id": {"type": "string", "description": "Source node ID"},
         "target_id": {"type": "string", "description": "Target node ID"},
         "relation": {"type": "string", "description": "Edge relation type", "default": "related_to"},
         "weight": {"type": "number", "description": "Edge weight 0.0-1.0", "default": 0.5}}}},
    _generate_revise_schema(),
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

    # ── Lookup operations ──
    {"name": "find_node_by_title",
     "description": "Find an existing node by fuzzy title matching using embedding similarity. Returns best match above threshold with context (content snippet, keywords) for verification. Default threshold 0.75 is conservative.",
     "inputSchema": {"type": "object", "required": ["title_query"], "properties": {
         "title_query": {"type": "string", "description": "Title to search for (fuzzy match)"},
         "threshold": {"type": "number", "description": "Minimum similarity (0.0-1.0, default 0.75)", "default": 0.75},
         "top_k": {"type": "integer", "description": "Return top K matches (default 1)", "default": 1}}}},

    {"name": "get_node",
     "description": "Get a node by its exact ID. Returns full content, type, title, confidence, connections, metadata. Use when you already have a node ID from recall or find_node_by_title.",
     "inputSchema": {"type": "object", "required": ["node_id"], "properties": {
         "node_id": {"type": "string", "description": "Full node ID"}}}},

    # ── Introspection ──
    {"name": "consciousness",
     "description": "Get brain consciousness signals. Most signals migrated to signal queue — returns reminders only. Use queue_state for full signal view.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "dismiss_signal",
     "description": "Dismiss a signal from the brain's signal queue. Use when a signal has been acknowledged or is no longer relevant.",
     "inputSchema": {"type": "object", "properties": {
         "signal_id": {"type": "string", "description": "Signal ID to dismiss"},
         "producer": {"type": "string", "description": "Dismiss all signals from this producer"}}}},
    {"name": "queue_state",
     "description": "Get current signal queue state — all pending signals with priorities, surface counts, producers.",
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


def _format_result(tool_name, result):
    """Format tool result for MCP output.

    Recall gets structured text (same format as hooks) for readability.
    All other tools get JSON.
    """
    if tool_name == "recall" and isinstance(result, dict):
        from servers.brain_voice import BrainVoice
        results = result.get("results", [])
        # Strip _query_embedding — internal debug data, not for output
        result.pop("_query_embedding", None)

        lines = []
        if results:
            BrainVoice.format_recall_results(results, lines)
        else:
            lines.append("No results found.")

        # Show vocab context (connectors, not primary results)
        vocab = result.get("vocab_context", [])
        if vocab:
            lines.append("")
            lines.append("Related vocabulary:")
            for v in vocab[:5]:
                lines.append("  %s (id:%s)" % (v.get('title', ''), v.get('id', '')[:8]))

        # Show gap info if present
        gap = result.get("_gap")
        if gap:
            lines.append('No results above relevance threshold for: "%s"' % gap.get("query", ""))

        # Append recall stats
        stats = result.get("_embedding_stats", {})
        if stats:
            lines.append("---")
            lines.append("recall: %dms | mode: %s | sources: %s" % (
                stats.get("recall_ms", 0),
                result.get("_recall_mode", "?"),
                ", ".join("%s:%d" % (k, v) for k, v in
                          stats.get("results_by_source", {}).items() if v > 0)
            ))
        return "\n".join(lines)

    return json.dumps(result, indent=2, default=str)


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
            result_text = _format_result(tool_name, resp["result"])
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


def _health_monitor():
    """Background health monitor — pings daemon every 2s.

    If daemon dies:
    1. Attempts restart via ensure_daemon_running()
    2. Writes PREEMPT signal directly to signal queue (SQLite, no daemon)
    3. Logs to dashboard DB

    Runs as daemon thread — dies when MCP process exits.
    """
    import time
    import sqlite3

    consecutive_failures = 0
    PING_INTERVAL = 2.0
    FAILURE_THRESHOLD = 3  # Alert after 3 consecutive failures (6s)

    while True:
        time.sleep(PING_INTERVAL)
        try:
            resp = daemon_send("ping", timeout=2.0)
            if resp.get("ok"):
                if consecutive_failures > 0:
                    sys.stderr.write("[brain-mcp] Daemon recovered after %d failures\n" % consecutive_failures)
                consecutive_failures = 0
                continue
        except Exception:
            pass

        consecutive_failures += 1

        if consecutive_failures == FAILURE_THRESHOLD:
            sys.stderr.write("[brain-mcp] ALERT: Daemon unreachable for %ds — attempting restart\n" % (
                int(consecutive_failures * PING_INTERVAL)))

            # Write PREEMPT signal directly to signal queue (brain_logs.db)
            try:
                db_dir = os.environ.get("BRAIN_DB_DIR", "")
                if not db_dir:
                    home = os.path.expanduser("~")
                    candidate = os.path.join(home, "AgentsContext", "brain")
                    if os.path.isdir(candidate):
                        db_dir = candidate
                if db_dir:
                    logs_db = os.path.join(db_dir, "brain_logs.db")
                    conn = sqlite3.connect(logs_db, timeout=3)
                    # Only write daemon_down if not already dismissed
                    existing = conn.execute(
                        "SELECT dismissed FROM signal_queue WHERE id = 'health:daemon_down'"
                    ).fetchone()
                    if not existing or not existing[0]:
                        conn.execute(
                            """INSERT OR REPLACE INTO signal_queue
                               (id, producer, signal_type, priority, content, preempt, created_at,
                                times_surfaced, dismissed, cooldown_seconds)
                               VALUES (?, ?, ?, ?, ?, ?, datetime('now'), 0, 0, 300)""",
                            ("health:daemon_down", "system_health", "daemon_down", 0.85,
                             "⚠️ Brain daemon is DOWN. Recall and encoding disabled.",
                             0))  # NOT preempt — don't block recall for a restart blip
                    conn.commit()
                    conn.close()
            except Exception as e:
                sys.stderr.write("[brain-mcp] Failed to write PREEMPT signal: %s\n" % e)

            # Log to dashboard
            try:
                if db_dir:
                    dash_db = os.path.join(db_dir, "brain_dashboard.db")
                    conn = sqlite3.connect(dash_db, timeout=3)
                    conn.execute(
                        """INSERT INTO hook_log (hook_name, timestamp, output_text, operator_text, session_id)
                           VALUES (?, datetime('now'), ?, ?, ?)""",
                        ("DAEMON_DOWN",
                         "⚠️ Daemon unreachable — MCP health monitor detected failure",
                         "⚠️ DAEMON DOWN",
                         "mcp_health_monitor"))
                    conn.commit()
                    conn.close()
            except Exception:
                pass

            # Attempt restart
            try:
                ensure_daemon_running()
            except Exception as e:
                sys.stderr.write("[brain-mcp] Restart failed: %s\n" % e)

        elif consecutive_failures > FAILURE_THRESHOLD and consecutive_failures % 10 == 0:
            # Retry restart every 20 seconds
            sys.stderr.write("[brain-mcp] Still down after %ds — retrying restart\n" % (
                int(consecutive_failures * PING_INTERVAL)))
            try:
                ensure_daemon_running()
            except Exception:
                pass


def main():
    # Ensure daemon is running — retry a few times since boot hook may be starting it concurrently
    sys.stderr.write("[brain-mcp] Starting MCP server...\n")
    import time, threading
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

    # Start health monitor (daemon thread — dies with MCP process)
    health_thread = threading.Thread(target=_health_monitor, daemon=True)
    health_thread.start()
    sys.stderr.write("[brain-mcp] Health monitor started (2s interval).\n")

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
