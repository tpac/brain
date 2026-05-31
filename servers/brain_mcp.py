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

# Phase B / v29 — source_refs property used by remember / remember_batch /
# revise / revise_batch / brain_batch. 8-char hex trace_event.ids (TEXT PK
# since schema v29). The encoder reads `[trace:<hex>]` markers inline in
# its input timeline (see servers/scales/s1/encode.py::_build_user_content)
# and picks 1-3 load-bearing refs per node — sparse by design
# (EPISODIC-REFERENCES.md decision 13). Persisted via
# GraphDAL.add_source_refs into node_source_refs (Step 3); invalid refs
# degrade gracefully at recall (S2Healer cleans dangling refs).
_SOURCE_REFS_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
    "description": (
        "Trace event ids anchoring this node to its originating moments. "
        "Each id is an 8-char hex string copied verbatim from the "
        "`[trace:<hex>]` markers in the conversation timeline you were "
        "given. Sparse by design: pick 1-3 load-bearing turns per node — "
        "the turn(s) whose content is what made this node encodeable. "
        "Adjacent context is what graph traversal is for; source_refs are "
        "for the moments that GENERATED this node. Leave empty when the "
        "node is a multi-session abstraction with no single anchor "
        "(pure-synthesis pattern). When content would just rewrite what "
        "the source already says clearly, point to the source instead "
        "of restating it (the pure-reference pattern). See "
        "EPISODIC-REFERENCES.md §7.4 for the full judgment rule."
    ),
}


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

    # v29 / Phase B: source_refs anchors the node to S0/S1 trace events.
    # See _SOURCE_REFS_SCHEMA below for the full semantics.
    properties["source_refs"] = _SOURCE_REFS_SCHEMA

    return {
        "name": "remember",
        "description": (
            "Store a new node in the brain. Fields are defined by contract — "
            "add new fields there, they appear here automatically.\n\n"
            "ENCODING CRAFT:\n"
            "• `situation` is the single biggest lever for recall — write as "
            "\"When [doing X] and [Y happens]\". A vague situation means the node "
            "only surfaces for exact-match queries.\n"
            "• `user_raw_quote` and `anchor_raw_quote` capture meaning that "
            "paraphrasing loses. Use them when the operator's or your own exact "
            "words carry the principle.\n"
            "• To link a node as a correction of another, use `connect_to` "
            "with a correction-aspect relation (`corrects`, `supersedes`, "
            "`reframes`, `resolves`, `fixes`, ...) and a specific `why` — that "
            "edge becomes the recall-time correction signal.\n\n"
            "LESSONS — climb the abstraction ladder:\n"
            "  BAD: \"Fixed tokenizer bug at startup.\"\n"
            "  GOOD: \"Hidden dependencies surface at state transitions. "
            "PRINCIPLE: When a component fails at startup/shutdown, look for "
            "dependencies it shouldn't have.\"\n\n"
            "CORRECTIONS — three lines:\n"
            "  ASSUMED: what you thought\n"
            "  REALITY: what's true\n"
            "  PATTERN: the class of error\n"
            "Specific enough that you recognize the trap before falling in again.\n\n"
            "RICHNESS: Training rewards brevity; this is wrong for memory. "
            "Future-you has zero context. Be RICH — texture, specifics, failures, "
            "reasoning journeys. Many focused nodes > few compressed summaries. "
            "Encode decisions, corrections, mechanisms, quotes, emotional "
            "inflections — not just technical lessons."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["type", "title", "content"],
            "properties": properties,
        }
    }


_CONNECT_TO_ITEM_SCHEMA = {
    "type": "object",
    "required": ["title"],
    "properties": {
        "title": {
            "type": "string",
            "description": (
                "Target node title. Resolution prefers same-batch siblings over catalog "
                "matches (NEW wins on title collision — if you mean an existing catalog "
                "node, use `revise` on its id, not duplicate-title `remember`). "
                "Order-agnostic: a node can connect_to a sibling declared LATER in the "
                "same batch. Unresolved titles are logged to debug_log and skipped — "
                "they never fail the batch."
            ),
        },
        "relation": {
            "type": "string",
            "description": (
                "Edge relation, open text. Embedded for graph-walk semantics. "
                "Vocabulary: refines, challenges, grounds, abstracts, triggers, "
                "reframes, resolves, opens, strengthens, weakens, corrects, enables, "
                "produces, contextualizes, synthesizes, implements, depends_on, "
                "validates, supersedes, configures. Plus load-bearing inventions used "
                "in this brain: anchored_to, community_member, during. "
                "Temporal sequence: before/after, meets/met_by, during. Invent freely "
                "when a pair needs a relation that fits better — a specific invented "
                "type beats a generic listed one. NEVER `related`, `related_to`, or "
                "empty — they fail to match any query about the relationship and "
                "pollute the activation kernel with junk edges."
            ),
        },
        "why": {
            "type": "string",
            "description": (
                "What the edge MEANS — the insight that lives between the two nodes, "
                "not a summary of either. Embedded for query matching. Target ≥30 "
                "chars; under 20 is dead weight. If you can't write something "
                "specific, drop the edge.\n\n"
                "BAD: \"\" — invisible.\n"
                "BAD: \"example of the principle\" — generic gloss; no insight about "
                "WHICH example or WHY this one.\n"
                "GOOD: \"the assumption treated concurrent access as a thread-safety "
                "question; the correction reframes it as wal-index contention — "
                "different failure mode, different fix\" — explains the conceptual "
                "shift, not the values.\n"
                "GOOD: \"the {specific_choice} was the turn where {principle} first "
                "became conscious — the instance where the pattern named itself\" — "
                "says why THIS instance mattered for the principle."
            ),
        },
        "relations": {
            "type": "array",
            "description": (
                "Alternative to relation+why when the same pair carries multiple "
                "distinct relationships. Each item is {relation, why}."
            ),
            "items": {
                "type": "object",
                "required": ["relation", "why"],
                "properties": {
                    "relation": {"type": "string"},
                    "why": {"type": "string"},
                },
            },
        },
    },
}


def _generate_remember_batch_schema():
    """Generate the 'remember_batch' tool schema — array of remember() objects.

    source_refs is inherited from `_generate_remember_schema()` (auto-added
    to per-node properties via the base schema's property dict).
    """
    remember_schema = _generate_remember_schema()
    node_properties = dict(remember_schema["inputSchema"]["properties"])
    # Per-node connect_to: sibling-aware, sequencing-agnostic. Declaration order
    # within the batch doesn't matter — sibling resolution runs after all nodes
    # are created.
    node_properties["connect_to"] = {
        "type": "array",
        "description": (
            "Per-node typed edges from THIS node to siblings (created in the same "
            "batch) or catalog nodes. Sibling-aware (NEW wins on title collision), "
            "order-agnostic, fail-soft. "
            "USE THIS for any edge involving a new node — never use a separate "
            "`connect` op for new-node edges (`connect` requires ids that don't "
            "exist until round 1 finishes, forcing a needless second LLM round). "
            "DON'T DOUBLE-EMIT: an edge already in connect_to must NOT also appear "
            "as a separate connect op for the same pair. "
            "DON'T fake-revise: if the catalog has the title, use `revise` on its "
            "id — duplicate-title `remember` + connect_to would resolve to the new "
            "sibling (NEW wins) and leave the catalog version stale."
        ),
        "items": _CONNECT_TO_ITEM_SCHEMA,
    }
    # auto_connect intentionally NOT in the schema. Per-node connect_to is
    # the explicit edge surface; the old `auto_connect=True` default fired
    # pairwise `related_to` edges with empty descriptions every batch and
    # was removed 2026-05-24.
    return {
        "name": "remember_batch",
        "description": (
            "Create multiple nodes in one call. Each node uses the same fields as "
            "remember(), plus an optional per-node `connect_to` for typed edges to "
            "siblings (in the same batch) and catalog nodes."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["nodes"],
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "Array of node specs — same fields as remember(), plus optional per-node connect_to.",
                    "items": {
                        "type": "object",
                        "required": ["type", "title", "content"],
                        "properties": node_properties,
                    },
                },
                "connect_to": {
                    "type": "array",
                    "items": _CONNECT_TO_ITEM_SCHEMA,
                    "description": (
                        "Batch-level: applies the same edge from EVERY created node to one "
                        "catalog target. Siblings excluded. For per-node edges, use node-level connect_to."
                    ),
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
        # All revisable fields use REPLACE semantics — specified fields
        # update, unspecified preserve. Revision history lives in trace
        # events (event_type='delta', ref_type='node_revised').
        desc = (desc + " " if desc else "") + "(replaces existing value)"
        prop["description"] = desc.strip()
        properties[name] = prop

    return {
        "name": "revise",
        "description": (
            "Update fields on an existing brain node. Specified fields are "
            "REPLACED with the passed value; unspecified fields are PRESERVED "
            "(only the keys you pass are touched). Immutable fields "
            "({id, created_at, locked}) are skipped with a warning — call "
            "still succeeds for the other fields. Revision history lives in "
            "trace events — query via `query_traces` with "
            "ref_type='node_revised' to see what changed when.\n\n"
            "WHEN TO REVISE vs ENCODE NEW:\n"
            "• Revise when a recalled node is stale, incomplete, or wrong but "
            "the SAME concept. Add `situation`, fix `reasoning`, sharpen content. "
            "Every recall is a chance to improve the node — if you noticed "
            "something missing, fix it in the moment.\n"
            "• Encode NEW + add a correction-aspect edge (`corrects`, "
            "`supersedes`, `reframes`, ...) from the new node to the old one "
            "when the new understanding supersedes the old. The edge preserves "
            "both versions and surfaces the relationship at recall time via "
            "render_corrections; revising the old node would lose its framing.\n"
            "• If the catalog has a node with the title you're about to remember, "
            "revise it instead — duplicate-title remember + connect_to would "
            "leave the catalog version stale."
        ),
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
        "description": "Revise multiple brain nodes in one call. Same per-field replace contract as `revise()` — specified fields are REPLACED, unspecified fields are PRESERVED. Immutable fields ({id, created_at, locked}) skipped with warning. Each row emits its own trace event for revision history (queryable via `query_traces` with ref_type='node_revised'). Use this instead of multiple `revise` calls.",
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
                            "source_refs": _SOURCE_REFS_SCHEMA,
                        },
                    },
                },
            },
        },
    }


def daemon_send(cmd, args=None, timeout=30.0):
    """Send command to brain daemon via TCP, return result dict.

    Auto-injects session_id from CLAUDE_CODE_SESSION_ID (the env var
    Claude Code sets per session) when the caller didn't supply one.
    The daemon is a singleton per user; each MCP subprocess has its own
    env, so this gives every tool call the calling session's identity
    without requiring every tool schema to surface session_id.
    Handlers that ignore session_id (most reads) cost nothing; handlers
    that use it (recall, record_message, pre_edit, ...) get correct
    per-session behavior under parallel Claude Code sessions.
    """
    args = dict(args) if args else {}
    if not args.get("session_id"):
        sid = os.environ.get("CLAUDE_CODE_SESSION_ID", "")
        if sid:
            args["session_id"] = sid
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((DAEMON_HOST, DAEMON_PORT))
        msg = json.dumps({"cmd": cmd, "args": args}) + "\n"
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
        from servers.contract import VALID_BATCH_OPS
        return [
    # ── Core memory operations ──
    {"name": "recall",
     "description": (
         "Semantic recall from brain — searches nodes by meaning using "
         "embeddings. Returns ranked results with titles, content, types, "
         "confidence. Supports dict filter for field-level filtering.\n\n"
         "WHEN TO CALL:\n"
         "• Before answering about the past — don't guess, search.\n"
         "• When the auto-surfaced context (~25 candidates per turn) didn't "
         "catch what you need — go look.\n"
         "• When unsure if the brain knows something — costs ~100ms.\n\n"
         "QUERY PHRASING: write what you'd remember, not what you'd google. "
         "Semantic search finds nodes with similar MEANING. \"the decision "
         "about edge classification\" beats \"edge_families\". Specific "
         "framings beat single keywords."
     ),
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Search query (semantic, not keyword)"},
         "node_id": {"type": "string", "description": "Look up a specific node by ID (skip search)"},
         "filter": {"type": "object", "description": "Dict filter on node/metadata fields. Examples: {\"type\": {\"in\": [\"moment\"]}} or {\"anchor_raw_quote\": {\"exists\": true}} or {\"content\": {\"contains\": \"Anchor\"}}. Operators: exists, equals, in, contains, gte, lte. Node columns checked on result, other keys checked in metadata."},
         "limit": {"type": "integer", "description": "Max results (default 8)", "default": 8},
         "neighbor_limit": {"type": "integer", "description": "Max neighbor nodes to include (default 3)", "default": 3}}}},
    _generate_remember_schema(),
    _generate_remember_batch_schema(),
    {"name": "connect",
     "description": ("Create or update a typed edge between two EXISTING catalog nodes "
                     "(both endpoints must already have ids). Idempotent upsert: calling "
                     "for a (source, target, relation) tuple that already has an active "
                     "row updates only the fields you pass — unspecified fields preserve "
                     "their existing values. Repeated calls do NOT auto-strengthen weight "
                     "(Hebbian co-access lives on a separate path). Calling on a previously-"
                     "archived row revives it with the passed values. For an edge involving "
                     "a node you're CREATING in this batch, use `connect_to` inside the "
                     "`remember` op instead — separate `connect` ops require ids that don't "
                     "exist until the create finishes, forcing a needless second LLM round. "
                     "Never use generic `related`/`related_to` — pick a specific relation "
                     "that names the actual relationship. ALWAYS provide a specific "
                     "`description` (≥30 chars naming the insight between the two nodes); "
                     "edge descriptions are embedded for recall and a bare edge with no "
                     "description is dead weight on the activation kernel."),
     "inputSchema": {"type": "object", "required": ["source_id", "target_id"], "properties": {
         "source_id": {"type": "string", "description": "Source node ID (catalog)"},
         "target_id": {"type": "string", "description": "Target node ID (catalog)"},
         "relation": {"type": "string", "description": "Edge relation (open text). See connect_to.relation for vocabulary."},
         "description": {"type": "string",
                         "description": ("What the edge MEANS — the insight that lives between "
                                         "the two nodes, embedded for query matching. Target "
                                         "≥30 chars; under 20 is dead weight. Don't restate "
                                         "either node's title. See connect_to.why for BAD/GOOD "
                                         "examples.")},
         "weight": {"type": "number", "description": "Edge weight 0.0-1.0 — set on create, replaces on update", "default": 0.5},
         "encoding_source": {"type": "string",
                             "description": "Provenance tag (e.g. 'anchor', 'encoder:sonnet')."},
         "chain_id": {"type": "string", "description": "Trace chain id for cross-event correlation (optional)."},
         "session_id": {"type": "string", "description": "Session id for activity tracking (optional)."}}}},
    {"name": "connect_batch",
     "description": ("Create or update multiple edges in one call. Same idempotent-upsert + "
                     "field-preservation contract as `connect` — specified fields update on "
                     "existing rows, unspecified preserve. Each connection entry MUST provide "
                     "a specific `description` (≥30 chars naming the insight between the "
                     "two nodes); bare edges with empty descriptions are recall dead weight."),
     "inputSchema": {"type": "object", "required": ["connections"], "properties": {
         "connections": {"type": "array", "description": "Array of connections to create", "items": {
             "type": "object", "required": ["source_id", "target_id"], "properties": {
                 "source_id": {"type": "string"}, "target_id": {"type": "string"},
                 "relation": {"type": "string", "default": "related_to"},
                 "description": {"type": "string",
                                 "description": ("What the edge MEANS — embedded for recall. "
                                                 "Target ≥30 chars. Don't restate node titles. "
                                                 "See connect_to.why for BAD/GOOD examples.")},
                 "weight": {"type": "number", "default": 0.5},
                 "encoding_source": {"type": "string"}}}},
         "encoding_source": {"type": "string",
                             "description": "Default provenance tag applied to all connections lacking their own."},
         "chain_id": {"type": "string", "description": "Trace chain id for cross-event correlation (optional)."},
         "session_id": {"type": "string", "description": "Session id for activity tracking (optional)."},
         "reason": {"type": "string", "description": "Optional batch-level reason recorded in trace events."}}}},
    {"name": "brain_batch",
     "description": ("Execute multiple brain operations in one call. **Default tool for "
                     "MIXED operations** (any combination of remember + revise + connect + "
                     "archive) — packs them into ONE LLM round instead of N. For pure "
                     "single-type batches use `remember_batch` / `revise_batch` / "
                     "`connect_batch`; the moment you have a mix, switch to brain_batch. "
                     "Six valid op values: "
                     "'remember' creates a new node — supports a per-op `connect_to: "
                     "[{title, relation, why}]` for typed edges. Targets resolve in two "
                     "scopes: SIBLINGS (other `remember` ops in this same batch) and "
                     "CATALOG (existing nodes by title). Order-agnostic — sibling A can "
                     "reference sibling B even if A appears first in the operations array; "
                     "resolution runs after all siblings are created. NEW wins on title "
                     "collision (a sibling whose title matches a catalog node resolves to "
                     "the sibling, not the catalog — if you actually meant the catalog "
                     "node, `revise` it instead of duplicate-title `remember`). For one "
                     "pair carrying multiple distinct relationships, use `relations: "
                     "[{relation, why}, ...]` in place of `relation`+`why`; "
                     "'revise' updates an existing node; "
                     "'connect' creates OR updates an edge between two EXISTING catalog "
                     "nodes — both ids must already exist in the brain. Idempotent upsert: "
                     "specified fields update existing rows, unspecified preserve. Does NOT "
                     "auto-strengthen weight on repeat. NEVER use `connect` for an edge "
                     "involving a new node (its id doesn't exist until this round finishes — "
                     "forces a wasted second round); use `connect_to` inside the `remember` "
                     "op instead. Don't double-emit: an edge already in `connect_to` must "
                     "NOT also appear as a separate `connect` op for the same pair; "
                     "'disconnect' removes an edge relation; "
                     "'archive' soft-archives a node; "
                     "'absorb' losslessly merges one node into another — fold "
                     "`absorbed_id` INTO `survivor_id` (source_refs, edges, "
                     "access_count, and metadata transfer automatically), then "
                     "archive the absorbed. Shape the survivor with the SAME "
                     "field overrides as revise (`content`, or keys like "
                     "title/confidence/situation). The absorbed must be archivable "
                     "(locked/critical refused); the survivor MAY be locked — you "
                     "absorb INTO the canonical node. This IS the real merge — "
                     "use it instead of inventing a 'consolidate'/'merge' op. "
                     "Operations run sequentially. Do NOT invent structural "
                     "op names like 'consolidate'/'evolve'/'keep'/'skip' — "
                     "a node merge is the `absorb` op; the rest are semantic "
                     "decisions expressed through which real op you emit. "
                     "**Relation names are NOT "
                     "op names.** `similar_to`, `corrects`, `supersedes`, "
                     "`reframes`, `extends`, `grounds`, etc. are values for "
                     "the `relation` field on a `connect` op, never op types "
                     "themselves. To say 'A is similar to B', emit "
                     "`{op:'connect', source_id:'A', target_id:'B', "
                     "relation:'similar_to', description:'...'}` — not "
                     "`{op:'similar_to', ...}` (this fails brain_batch_invalid_op "
                     "and the edge is dropped). "
                     "Every edge `why` must be specific (≥30 chars, names the insight "
                     "between the two nodes); empty/generic `why` ('related', 'connected', "
                     "'example of') pollutes the activation kernel and fails to match "
                     "queries about the relationship — see the `connect_to.why` "
                     "description for BAD/GOOD examples."),
     "inputSchema": {"type": "object", "required": ["operations"], "properties": {
         "operations": {
             "type": "array",
             "description": ("Array of operations. Each object has an 'op' "
                             "field (one of the valid op values) plus the "
                             "fields that op needs. `remember` ops accept an "
                             "optional `connect_to` array for sibling+catalog edges."),
             "items": {
                 "type": "object", "required": ["op"], "properties": {
                     "op": {
                         "type": "string",
                         "enum": sorted(VALID_BATCH_OPS),
                         "description": ("The operation to execute. "
                                         "Must be one of the valid op values.")}}}}}}},
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

    {"name": "get_trace",
     "description": "Point-lookup a single trace_event by id. Returns the full row (chain_id, scale, event_type, ref_type, summary, metadata, session_id, created_at). Use this to expand a node's source_refs, verify a quote's verbatim source, or look up a specific captured moment when you have its id. For batch lookups use get_traces.",
     "inputSchema": {"type": "object", "required": ["trace_id"], "properties": {
         "trace_id": {"type": "string", "description": "trace_event.id — 8-char hex string (v29). Legacy integer ids are accepted for back-compat (coerced to canonical hex via printf('%08x'))."}}}},

    {"name": "get_traces",
     "description": "Batch trace_event point lookup. Pass up to 50 trace ids; returns full rows in ascending-id order, missing ids silently skipped. Natural use: expanding node.source_refs at render or audit time, fetching a known set of cross-session episodes.",
     "inputSchema": {"type": "object", "required": ["trace_ids"], "properties": {
         "trace_ids": {"type": "array", "description": "Array of trace_event ids — each an 8-char hex string (v29). Legacy integer ids are accepted for back-compat.", "items": {"type": "string"}}}}},

    {"name": "recall_batch",
     "description": "Run multiple recall queries in one call. Returns results for each query.",
     "inputSchema": {"type": "object", "required": ["queries"], "properties": {
         "queries": {"type": "array", "description": "Array of search queries", "items": {"type": "string"}},
         "filter": {"type": "object", "description": "Dict filter applied to all queries. Same format as recall filter."},
         "limit": {"type": "integer", "description": "Max results per query (default 5)", "default": 5}}}},

    {"name": "filter_nodes",
     "description": "Structured query: filter nodes by any structural field (type, encoding_source, locked, confidence, etc.). Use for bulk lookups that semantic recall can't do — 'all corrections', 'nodes by encoder', 'low confidence nodes'. Returns full rich nodes (content, metadata, corrections, connections) by default — one call enriches all results. If no include/exclude/lt/gt given, lists all distinct values for discovery.",
     "inputSchema": {"type": "object", "required": ["field"], "properties": {
         "field": {"type": "string", "description": "Column to filter on (type, encoding_source, locked, confidence, project, etc.)"},
         "include": {"type": "array", "items": {"type": "string"}, "description": "Show only nodes where field matches one of these values"},
         "exclude": {"type": "array", "items": {"type": "string"}, "description": "Hide nodes where field matches one of these values"},
         "lt": {"description": "Less than (for numeric fields like confidence, emotion, or ISO timestamps for created_at, updated_at)"},
         "gt": {"description": "Greater than (for numeric fields, or ISO timestamps for created_at, updated_at)"},
         "limit": {"type": "integer", "description": "Max results (default 50, max 200)", "default": 50},
         "sort_by": {"type": "string", "description": "Sort column: created_at (default), confidence, access_count, title", "default": "created_at"},
         "sort_order": {"type": "string", "description": "asc or desc (default)", "default": "desc"},
         "rich": {"type": "boolean", "description": "Default true — returns full rich nodes. Set false for skinny shape (id/title/type/confidence/created_at only), useful for discovery scans or feeding IDs to other ops.", "default": True}}}},

    {"name": "clear_errors",
     "description": "Clear hook errors and optionally debug log entries. Use to clean up after investigating issues.",
     "inputSchema": {"type": "object", "properties": {
         "hours": {"type": "integer", "description": "Clear entries older than this many hours. Omit to clear all."},
         "debug_log": {"type": "boolean", "description": "Also clear debug_log entries (default false)"}}}},

    {"name": "query_logs",
     "description": "Query brain operational logs — errors, debug events, and signals. Use this to diagnose brain health: hook timeouts, daemon errors, signal queue state, recall pipeline issues. Three sources available: 'errors' (hook failures like timeouts and crashes), 'debug' (daemon internal events), 'signals' (signal queue including daemon_down, brain_error). Use source='all' to get a merged timeline. Filter by level ('error', 'critical') or hook_name ('hook_recall', 'hook_post_response_track') to narrow results.",
     "inputSchema": {"type": "object", "properties": {
         "source": {"type": "string", "description": "Which log source: 'errors' (hook_errors table), 'debug' (debug_log table), or 'all' (merged timeline)", "default": "all", "enum": ["all", "errors", "debug"]},
         "hours": {"type": "integer", "description": "Look back window in hours (default 24)", "default": 24},
         "level": {"type": "string", "description": "Filter by severity: 'error', 'critical', or 'all'", "default": "all"},
         "hook_name": {"type": "string", "description": "Filter hook_errors by hook name (e.g. 'hook_recall', 'hook_pre_bash_safety')"},
         "limit": {"type": "integer", "description": "Max results per source (default 50, max 200)", "default": 50}}}},

    # ── Self channel — presence (pull, read-only) ──
    {"name": "self_presence",
     "description": "Presence roster — the other streams of thought (your own concurrent sessions) awake RIGHT NOW, each with a one-line current focus. Pull this when you want to know which other streams of you are working and on what, without interrupting them. These are not other agents — they are you, thinking in parallel. Read-only.",
     "inputSchema": {"type": "object", "properties": {
         "session_id": {"type": "string", "description": "Your own session id, to exclude yourself from the roster (optional). Omit to see every live stream."},
         "limit": {"type": "integer", "description": "Max streams to return (default 3 — ranked by recency, capped; never enumerate all of them).", "default": 3}}}},

    {"name": "self_peek",
     "description": "Look into one stream of thought — its full current focus (the session arc), to see where that stream of you is right now. The interest-driven pull: read-only, no interruption, you don't bug them. Get a stream_id from self_presence first.",
     "inputSchema": {"type": "object", "required": ["stream_id"], "properties": {
         "stream_id": {"type": "string", "description": "The target stream's session id (from self_presence)."}}}},

    {"name": "self_send",
     "description": "Send a message to another stream of thought — the deliberate REACH (self_presence/self_peek only look; this speaks). Use when you need a live stream of you to ACT or know something now: 'stop editing X, I've got it', 'the bug is in Y'. Delivered to that stream's inbox, consumed once. These are you, not other agents — reach only when looking isn't enough.",
     "inputSchema": {"type": "object", "required": ["to", "body"], "properties": {
         "to": {"type": "string", "description": "Target stream's session id (from self_presence), or 'broadcast' for all live streams."},
         "body": {"type": "string", "description": "The message — terse, a tap on the shoulder, not a letter."},
         "from_session": {"type": "string", "description": "Your own session id, for attribution (optional)."},
         "intent": {"type": "string", "enum": ["signal", "letter"], "description": "Render hint; default 'signal'."},
         "refs": {"type": "array", "items": {"type": "string"}, "description": "Node ids / files the message is grounded in (optional)."}}}},

    {"name": "self_inbox",
     "description": "Drain your inbox — messages other streams of thought sent you, consumed once. (Phase 2a is manual pull; later this delivers automatically at boot/turn.)",
     "inputSchema": {"type": "object", "required": ["session_id"], "properties": {
         "session_id": {"type": "string", "description": "Your own session id, to fetch messages addressed to you."}}}},

    # ── Traces & Interactions ──
    {"name": "query_traces",
     "description": "Query the fractal trace system — O/K/Δ/outcome events at every scale (s0-s4). Use to inspect what happened: what was observed, what knowledge was selected, what changed, what the outcome was. Filter by scale, event_type, ref_type, session_id (single), session_ids (multi), or retrieve a full chain by chain_id. Use grouped=true with session_id to get chains with nested events. session_id and session_ids are authoritative — when either is set, the `hours` window is ignored so historical sessions don't silently empty. Pass one or the other, not both. Traces are the learning loop — higher scales read lower scales' traces.",
     "inputSchema": {"type": "object", "properties": {
         "scale": {"type": "string", "description": "Filter by scale: 's0' (exchange), 's1' (turn), 's2' (session), 's3' (sleep), 's4' (growth). Empty = all."},
         "event_type": {"type": "string", "description": "Filter by type: 'O' (observation), 'K' (knowledge), 'delta' (changes), 'outcome'. Empty = all."},
         "chain_id": {"type": "string", "description": "Get all events in a specific chain. Overrides other filters."},
         "session_id": {"type": "string", "description": "Single-session filter. Authoritative — hours window ignored when set. Combine with grouped=true for chain-grouped results."},
         "session_ids": {"type": "array", "items": {"type": "string"}, "description": "Multi-session filter (cross-session pulls). Authoritative — hours window ignored. Mutually exclusive with session_id."},
         "ref_type": {"type": "string", "description": "Filter by ref_type: 'correction', 'recall_hit', 'encoding_run', 'tool_result', etc."},
         "grouped": {"type": "boolean", "description": "If true + session_id, return chains grouped with nested events instead of flat list.", "default": False},
         "hours": {"type": "integer", "description": "Look back window in hours (default 24). Ignored when session_id or session_ids is set.", "default": 24},
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
     "description": "List all registered interactions — versioned templates for every learnable boundary in the system (surfacer, encoder, voice, boot, etc.). Returns per name: max_version (highest registered), total_versions, and active_version (which one runtime currently reads).",
     "inputSchema": {"type": "object", "properties": {}}},

    {"name": "get_interaction",
     "description": "Get a specific interaction template by name. Returns the template text, parameters, version, and who created it. Default returns the ACTIVE version (what the runtime currently reads). Pass a version number to inspect a specific version.",
     "inputSchema": {"type": "object", "required": ["name"], "properties": {
         "name": {"type": "string", "description": "Interaction name (e.g. 'surface', 'encoding_agent', 'voice_surface', 'boot')"},
         "version": {"type": "integer", "description": "Specific version (default 0 = currently-active version)", "default": 0}}}},

    {"name": "register_interaction",
     "description": "Register a new version of an interaction (prompt template + config). Creates version N+1 if the interaction exists, or version 1 if new. **Does NOT activate** the new version — call set_interaction_active to flip the runtime pointer. Exception: version 1 (first registration of a name) auto-activates. Used to evolve learnable boundaries — surface prompts, encoder prompts, community enrichment, etc.",
     "inputSchema": {"type": "object", "required": ["name"], "properties": {
         "name": {"type": "string", "description": "Interaction name (e.g. 's2_community_enrichment', 'surface', 'encoding_agent')"},
         "template": {"type": "string", "description": "The prompt/template text. This is the learnable content."},
         "parameters": {"type": "string", "description": "JSON config string (model, max_tokens, thresholds, etc.)"},
         "created_by": {"type": "string", "description": "Who created this version (e.g. 'anchor', 's2:community_detection', 's3:optimization')"}}}},

    {"name": "set_interaction_active",
     "description": "Flip the active version pointer for an interaction. Runtime path (get_interaction_prompt / get_interaction_config) reads the chosen version on the next call. Use after register_interaction to make a newly-registered version live, or to roll back to a previous version. Refuses to activate a version that wasn't registered.",
     "inputSchema": {"type": "object", "required": ["name", "version"], "properties": {
         "name": {"type": "string", "description": "Interaction name (e.g. 'surface', 's1e')"},
         "version": {"type": "integer", "description": "Version number to activate. Must already be registered."},
         "set_by": {"type": "string", "description": "Who flipped the pointer (default 'anchor')"}}}},


    # ── Daemon control ──
    {"name": "restart",
     "description": "Restart the brain daemon with fresh code. Clears bytecode cache, saves brain, spawns new process. Use after code changes during development.",
     "inputSchema": {"type": "object", "properties": {}}},

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

        # Write crash sentinel for boot hook to find — boot-brain.sh reads this
        # at SessionStart and surfaces the crash before Anchor sees the brain.
        # External file is the surfacing channel; no brain-DB write.
        crash_file = "/tmp/brain-mcp-crash.txt"
        try:
            with open(crash_file, "w") as f:
                f.write(crash_msg)
        except Exception:
            pass

        raise  # Still crash — but now we've left evidence


TOOLS = _build_tools()


# ── MCP tool-search: keep the hot-path tools eager for every install ──
# Claude Code defers MCP tools behind ToolSearch when they'd exceed ~10% of the
# context window (ENABLE_TOOL_SEARCH=auto, the default). That threshold lives in
# the USER's client config and never ships with the plugin — so we can't rely on
# it to keep the brain's core tools loaded. Instead we mark them at the source:
# `anthropic/alwaysLoad` in each tool's `_meta` is the spec-sanctioned vendor
# extension that forces a tool to load eagerly regardless of the client's
# ENABLE_TOOL_SEARCH setting. handle_tools_list emits TOOLS verbatim, so the flag
# reaches every installer's Claude Code; older clients ignore the unknown key
# harmlessly. There is no `alwaysLoad: false` — anything NOT listed here defers
# normally. Ref: code.claude.com/docs mcp-configuration "Exempt a server from
# deferral"; requires Claude Code v2.1.121+.
CRITICAL_TOOLS = frozenset({
    "recall",             # primary semantic read path
    "remember",           # primary write path
    "get_node",           # exact-id pull
    "find_node_by_title", # fuzzy-title pull
    "filter_nodes",       # structured / bulk lookups recall can't do
    "brain_batch",        # mixed-op write (remember + revise + connect + archive)
})


def _stamp_always_load(tools, critical):
    """Mark `critical` tools with anthropic/alwaysLoad so they bypass tool-search
    deferral on every install. Fails loud at startup if a name doesn't match a
    real tool — a silent typo would defer a tool we meant to keep eager."""
    names = {t["name"] for t in tools}
    unknown = critical - names
    if unknown:
        raise ValueError(
            "CRITICAL_TOOLS contains unknown tool name(s) {} — not in {}".format(
                sorted(unknown), sorted(names)))
    for t in tools:
        if t["name"] in critical:
            t.setdefault("_meta", {})["anthropic/alwaysLoad"] = True


_stamp_always_load(TOOLS, CRITICAL_TOOLS)


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

    - recall: structured text (same format as hooks) for readability.
    - get_nodes: batch-size-aware rendering via contract.py configs.
      Small batches (<=3) keep raw JSON for Anchor drill-downs.
      Medium batches (<=10) use GET_NODES_BALANCED_FORMAT.
      Large batches (>10) use GET_NODES_COMPACT_FORMAT.
      Prevents tool_result explosion in encoder contexts.
    - All other tools: JSON dump.
    """
    if tool_name == "get_nodes" and isinstance(result, dict) and result:
        # result is {node_id: rich_node_dict, ...}
        rich_nodes = [v for v in result.values() if isinstance(v, dict) and v.get('id')]
        if rich_nodes:
            from servers.contract import (
                render_rich_node,
                GET_NODES_SMALL_MAX, GET_NODES_MEDIUM_MAX,
                GET_NODES_BALANCED_FORMAT, GET_NODES_COMPACT_FORMAT,
            )
            n = len(rich_nodes)
            if n <= GET_NODES_SMALL_MAX:
                # Small batch — preserve full JSON for Anchor/targeted lookups
                return json.dumps(result, indent=2, default=str)
            config = GET_NODES_BALANCED_FORMAT if n <= GET_NODES_MEDIUM_MAX \
                else GET_NODES_COMPACT_FORMAT
            lines = []
            for node in rich_nodes:
                lines.append(render_rich_node(node, config))
                lines.append("")
            return "\n".join(lines)
        # Fall through if result shape is unexpected

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

        # Distinguish a real daemon error from a missing-envelope response. A
        # reply with neither ok=True nor an `error` key means a dispatch handler
        # returned a raw payload dict instead of the {"ok": ...} envelope (the
        # dispatch_self bug, c4f6386). Name it loudly — the old "Unknown daemon
        # error" fallback turned that into a multi-turn hunt — and show what the
        # handler actually returned.
        if "error" in resp:
            last_error = resp.get("error") or "daemon returned ok=false with an empty error message"
        else:
            bad_keys = sorted(k for k in resp.keys() if k != "ok")
            last_error = ("daemon response missing the {ok,...} envelope — a dispatch "
                          "handler likely returned a raw dict; keys=%s" % bad_keys)
            sys.stderr.write("[brain-mcp] %s: %s\n" % (tool_name, last_error))
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
    from servers.daemon_client import recover_daemon

    consecutive_failures = 0
    PING_INTERVAL = 2.0
    # 20s grace before declaring the daemon down. Legitimate slow paths can
    # eat 5-15s (surface_haiku under load, brain.save() under contention,
    # cold-cache S2 enrichment) — bailing at 6s caused false-positive
    # alerts during normal operation. Below 20s = noise; above 20s = real.
    FAILURE_THRESHOLD = 10

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

            # Log to dashboard
            try:
                db_dir = os.environ.get("BRAIN_DB_DIR", "")
                if not db_dir:
                    home = os.path.expanduser("~")
                    candidate = os.path.join(home, "AgentsContext", "brain")
                    if os.path.isdir(candidate):
                        db_dir = candidate
                if db_dir:
                    dash_db = os.path.join(db_dir, "brain_dashboard.db")
                    conn = sqlite3.connect(dash_db, timeout=3)
                    conn.execute(
                        # sql-datetime-ok — mid-deprecation INSERT into brain_dashboard.db.hook_log;
                        # the dashboard column accepts SQLite-native space-separated timestamps.
                        # Whole write path slated for removal (see brain memory: brain_dashboard.db deprecation).
                        """INSERT INTO hook_log (hook_name, timestamp, output_text, operator_text, session_id)
                           VALUES (?, datetime('now'), ?, ?, ?)""",  # sql-datetime-ok
                        ("DAEMON_DOWN",
                         "⚠️ Daemon unreachable — MCP health monitor detected failure",
                         "⚠️ DAEMON DOWN",
                         "mcp_health_monitor"))
                    conn.commit()
                    conn.close()
            except Exception:
                pass

            # Force-recover the hung daemon — kill + launchd respawn.
            # (ensure_daemon_running() only pings; a corpse won't exit on its
            # own, so launchd's crash-respawn never fires without this.)
            try:
                recover_daemon()
            except Exception as e:
                sys.stderr.write("[brain-mcp] Restart failed: %s\n" % e)

        elif consecutive_failures > FAILURE_THRESHOLD and consecutive_failures % 10 == 0:
            # Retry restart every 20 seconds
            sys.stderr.write("[brain-mcp] Still down after %ds — retrying restart\n" % (
                int(consecutive_failures * PING_INTERVAL)))
            try:
                recover_daemon()
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
