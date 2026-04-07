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
    from servers.contract import get_remember_fields as get_writable_fields

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
    from servers.contract import get_writable_fields

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
        "description": "Update any field(s) on an existing brain node. Content is REPLACED (old saved to revision history). All other fields replaced.",
        "inputSchema": {
            "type": "object",
            "required": ["node_id", "reason"],
            "properties": properties,
        }
    }


def _build_revise_batch_schema():
    """Generate the 'revise_batch' MCP tool schema."""
    return {
        "name": "revise_batch",
        "description": "Revise multiple brain nodes in one call. Each revision can update content (replaced, history saved), metadata (reasoning, situation, etc.), or any revisable field. Use this instead of multiple revise() calls.",
        "inputSchema": {
            "type": "object",
            "required": ["revisions"],
            "properties": {
                "revisions": {
                    "type": "array",
                    "description": "List of revisions. Each must have node_id and reason, plus any fields to update.",
                    "items": {
                        "type": "object",
                        "required": ["node_id", "reason"],
                        "properties": {
                            "node_id": {"type": "string", "description": "Node ID to revise"},
                            "reason": {"type": "string", "description": "Why this revision"},
                            "content": {"type": "string", "description": "New content (replaces old, history saved)"},
                            "situation": {"type": "string", "description": "When is this relevant (gets own embedding)"},
                            "reasoning": {"type": "string", "description": "Why this was encoded"},
                            "user_raw_quote": {"type": "string", "description": "Operator's exact words"},
                            "anchor_raw_quote": {"type": "string", "description": "Anchor's exact words"},
                            "keywords": {"type": "string", "description": "Space-separated keywords"},
                            "confidence": {"type": "number", "description": "0-1 confidence score"},
                        },
                    },
                },
            },
        },
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
def _build_tools():
    """Build tool list at startup. If this fails, the MCP server is dead — scream about it."""
    try:
        return [
    # ── Core memory operations ──
    {"name": "recall",
     "description": "Semantic recall from brain — searches nodes by meaning using embeddings. Returns ranked results with titles, content, types, confidence. Supports dict filter for field-level filtering.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Search query (semantic, not keyword)"},
         "node_id": {"type": "string", "description": "Look up a specific node by ID (skip search)"},
         "filter": {"type": "object", "description": "Dict filter on node/metadata fields. Examples: {\"type\": {\"in\": [\"moment\"]}} or {\"anchor_raw_quote\": {\"exists\": true}} or {\"content\": {\"contains\": \"Anchor\"}}. Operators: exists, equals, in, contains, gte, lte. Node columns checked on result, other keys checked in metadata."},
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
    {"name": "connect_batch",
     "description": "Create multiple edges in one call.",
     "inputSchema": {"type": "object", "required": ["connections"], "properties": {
         "connections": {"type": "array", "description": "Array of connections to create", "items": {
             "type": "object", "required": ["source_id", "target_id"], "properties": {
                 "source_id": {"type": "string"}, "target_id": {"type": "string"},
                 "relation": {"type": "string", "default": "related_to"},
                 "weight": {"type": "number", "default": 0.5}}}}}}},
    {"name": "brain_batch",
     "description": "Execute multiple brain operations in one call — remember, revise, and connect mixed together. Each operation runs sequentially. Use for efficient multi-step encoding.",
     "inputSchema": {"type": "object", "required": ["operations"], "properties": {
         "operations": {"type": "array", "description": "Array of operations. Each has 'op' field: 'remember', 'revise', or 'connect', plus the fields for that operation.", "items": {
             "type": "object", "required": ["op"], "properties": {
                 "op": {"type": "string", "description": "Operation type: remember, revise, or connect"}}}}}}},
    _generate_revise_schema(),
    _build_revise_batch_schema(),
    {"name": "enrich",
     "description": "Store V5 enrichment vectors for a node (after filling in the enrichment_prompt from remember()). Pass the generated question, anchor phrase, bridge sentence, and/or keywords. Each is embedded and stored for improved recall.",
     "inputSchema": {"type": "object", "required": ["node_id"], "properties": {
         "node_id": {"type": "string", "description": "Node ID to enrich (from remember() response)"},
         "question": {"type": "string", "description": "One question a user would ask that leads to this node"},
         "anchor": {"type": "string", "description": "3-5 word phrase using neighbor vocabulary"},
         "bridge": {"type": "string", "description": "One sentence connecting this node to its most important neighbor"},
         "keywords": {"type": "string", "description": "Comma-separated keywords borrowed from neighbors"}}}},

    # Specialized tools REMOVED 2026-04-06:
    # record_divergence, learn_vocabulary — use remember(type='correction'/'vocabulary') instead.
    # remember_lesson, remember_impact, remember_mechanism, remember_convention,
    # remember_uncertainty, remember_mental_model — removed 2026-04-05.

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

    {"name": "get_nodes",
     "description": "Get multiple nodes by ID in one call. Returns full content, connections, metadata for each.",
     "inputSchema": {"type": "object", "required": ["node_ids"], "properties": {
         "node_ids": {"type": "array", "description": "Array of node IDs to fetch", "items": {"type": "string"}}}}},

    {"name": "recall_batch",
     "description": "Run multiple recall queries in one call. Returns results for each query.",
     "inputSchema": {"type": "object", "required": ["queries"], "properties": {
         "queries": {"type": "array", "description": "Array of search queries", "items": {"type": "string"}},
         "filter": {"type": "object", "description": "Dict filter applied to all queries. Same format as recall filter."},
         "limit": {"type": "integer", "description": "Max results per query (default 5)", "default": 5}}}},

    {"name": "filter_nodes",
     "description": "Structured query: filter nodes by any structural field (type, encoding_source, locked, confidence, etc.). Use for bulk lookups that semantic recall can't do — 'all corrections', 'nodes by encoder', 'low confidence nodes'. If no include/exclude/lt/gt given, lists all distinct values for discovery.",
     "inputSchema": {"type": "object", "required": ["field"], "properties": {
         "field": {"type": "string", "description": "Column to filter on (type, encoding_source, locked, confidence, project, etc.)"},
         "include": {"type": "array", "items": {"type": "string"}, "description": "Show only nodes where field matches one of these values"},
         "exclude": {"type": "array", "items": {"type": "string"}, "description": "Hide nodes where field matches one of these values"},
         "lt": {"type": "number", "description": "Less than (for numeric fields like confidence, emotion)"},
         "gt": {"type": "number", "description": "Greater than (for numeric fields)"},
         "limit": {"type": "integer", "description": "Max results (default 50, max 200)", "default": 50},
         "sort_by": {"type": "string", "description": "Sort column: created_at (default), confidence, access_count, title", "default": "created_at"},
         "sort_order": {"type": "string", "description": "asc or desc (default)", "default": "desc"}}}},

    {"name": "query_logs",
     "description": "Query brain operational logs — errors, debug events, and signals. Use this to diagnose brain health: hook timeouts, daemon errors, signal queue state, recall pipeline issues. Three sources available: 'errors' (hook failures like timeouts and crashes), 'debug' (daemon internal events), 'signals' (signal queue including daemon_down, brain_error). Use source='all' to get a merged timeline. Filter by level ('error', 'critical') or hook_name ('hook_recall', 'hook_post_response_track') to narrow results.",
     "inputSchema": {"type": "object", "properties": {
         "source": {"type": "string", "description": "Which log source: 'errors' (hook_errors table), 'debug' (debug_log table), 'signals' (signal_queue), or 'all' (merged timeline)", "default": "all", "enum": ["all", "errors", "debug", "signals"]},
         "hours": {"type": "integer", "description": "Look back window in hours (default 24)", "default": 24},
         "level": {"type": "string", "description": "Filter by severity: 'error', 'critical', or 'all'", "default": "all"},
         "hook_name": {"type": "string", "description": "Filter hook_errors by hook name (e.g. 'hook_recall', 'hook_pre_bash_safety')"},
         "limit": {"type": "integer", "description": "Max results per source (default 50, max 200)", "default": 50}}}},

    # ── Traces & Interactions ──
    {"name": "query_traces",
     "description": "Query the fractal trace system — O/K/Δ/outcome events at every scale (s0-s4). Use to inspect what happened: what was observed, what knowledge was selected, what changed, what the outcome was. Filter by scale, event_type, ref_type, session_id, or retrieve a full chain by chain_id. Use grouped=true with session_id to get chains with nested events. Traces are the learning loop — higher scales read lower scales' traces.",
     "inputSchema": {"type": "object", "properties": {
         "scale": {"type": "string", "description": "Filter by scale: 's0' (exchange), 's1' (turn), 's2' (session), 's3' (sleep), 's4' (growth). Empty = all."},
         "event_type": {"type": "string", "description": "Filter by type: 'O' (observation), 'K' (knowledge), 'delta' (changes), 'outcome'. Empty = all."},
         "chain_id": {"type": "string", "description": "Get all events in a specific chain. Overrides other filters."},
         "session_id": {"type": "string", "description": "Filter by session. Combine with grouped=true for chain-grouped results."},
         "ref_type": {"type": "string", "description": "Filter by ref_type: 'correction', 'recall_hit', 'encoding_run', 'tool_result', etc."},
         "grouped": {"type": "boolean", "description": "If true + session_id, return chains grouped with nested events instead of flat list.", "default": False},
         "hours": {"type": "integer", "description": "Look back window in hours (default 24)", "default": 24},
         "limit": {"type": "integer", "description": "Max results (default 100)", "default": 100}}}},

    {"name": "query_outcomes",
     "description": "Query outcome events — the learning signal. Outcomes are added retrospectively when we learn what happened next (corrections, future recalls). Use to find which chains got corrected vs validated.",
     "inputSchema": {"type": "object", "properties": {
         "chain_id": {"type": "string", "description": "Get outcomes for a specific chain."},
         "scale": {"type": "string", "description": "Filter by scale. Empty = all."},
         "hours": {"type": "integer", "description": "Look back window in hours (default 168 = 7 days)", "default": 168}}}},

    {"name": "count_traces",
     "description": "Count trace events grouped by a field. Use for quick overview: 'how many corrections?', 'events per type', 'chains per scale'.",
     "inputSchema": {"type": "object", "required": ["field"], "properties": {
         "field": {"type": "string", "description": "Group by: 'event_type', 'ref_type', 'chain_id', 'scale'"},
         "scale": {"type": "string", "description": "Filter by scale. Empty = all."},
         "hours": {"type": "integer", "description": "Look back window in hours (default 24)", "default": 24}}}},

    {"name": "list_interactions",
     "description": "List all registered interactions — versioned templates for every learnable boundary in the system (surfacer, encoder, voice, boot, etc.). Shows name, latest version, and total versions.",
     "inputSchema": {"type": "object", "properties": {}}},

    {"name": "get_interaction",
     "description": "Get a specific interaction template by name. Returns the template text, parameters, version, and who created it. Use to inspect or reference the current prompt/template for any system boundary.",
     "inputSchema": {"type": "object", "required": ["name"], "properties": {
         "name": {"type": "string", "description": "Interaction name (e.g. 'surface', 'encoding_agent', 'voice_surface', 'boot')"},
         "version": {"type": "integer", "description": "Specific version (default: latest)", "default": 0}}}},

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
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        crash_msg = "[brain-mcp] FATAL: Tool schema generation failed — MCP server cannot start.\n{}\n{}".format(e, tb)

        # Scream to stderr (Claude Code may log this)
        sys.stderr.write(crash_msg)
        sys.stderr.flush()

        # Write crash sentinel for boot hook to find
        crash_file = "/tmp/brain-mcp-crash.txt"
        try:
            with open(crash_file, "w") as f:
                f.write(crash_msg)
        except Exception:
            pass

        # Write signal to queue (direct SQLite — daemon may be fine, it's the MCP that's broken)
        try:
            import sqlite3
            db_dir = os.environ.get("BRAIN_DB_DIR", "")
            if not db_dir:
                candidate = os.path.join(os.path.expanduser("~"), "AgentsContext", "brain")
                if os.path.isdir(candidate):
                    db_dir = candidate
            if db_dir:
                logs_db = os.path.join(db_dir, "brain_logs.db")
                conn = sqlite3.connect(logs_db, timeout=3)
                conn.execute(
                    """INSERT OR REPLACE INTO signal_queue
                       (id, producer, signal_type, priority, content, preempt, created_at,
                        times_surfaced, dismissed, cooldown_seconds)
                       VALUES (?, ?, ?, ?, ?, ?, datetime('now'), 0, 0, 0)""",
                    ("mcp:startup_crash", "brain_mcp", "mcp_crash", 0.95,
                     "FATAL: Brain MCP server crashed on startup — Anchor has NO direct brain tools. Error: {}".format(e),
                     1))  # PREEMPT — this is critical
                conn.commit()
                conn.close()
        except Exception:
            pass

        raise  # Still crash — but now we've left evidence


TOOLS = _build_tools()


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

        # Graph neighbors from expansion (Layer 3 enrichment)
        graph_nbs = result.get("_graph_neighbors", [])
        if graph_nbs:
            lines.append("Related knowledge (via graph):")
            for nb in graph_nbs[:6]:
                edge_desc = " — %s" % nb["edge_description"] if nb.get("edge_description") else ""
                lines.append("  [%s] \"%s\" (%s%s)" % (
                    nb.get("type", "?"),
                    nb.get("title", "?")[:60],
                    nb.get("edge_type", "related"),
                    edge_desc))
                content = (nb.get("content") or "")[:150]
                if content:
                    lines.append("    %s" % content)
            lines.append("")

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
