#!/usr/bin/env python3
"""
brain — CLI client for bash and subagents.

Talks to the brain daemon via Unix socket. Outputs JSON.
Subagents that don't have MCP tools can use this from bash.

Usage:
    brain recall "how does the recall pipeline work" --limit 5
    brain remember --type lesson --title "..." --content "..."
    brain connect <source_id> <target_id> --relation validates
    brain ping
    brain status
    brain eval "brain.get_due_reminders()"

With agent DB isolation:
    brain recall "query" --db /tmp/brain-agent-123.db
"""

import argparse
import json
import os
import sys

# Resolve imports
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from servers.daemon_client import send_command, is_daemon_running, ensure_daemon


def _print_json(data):
    """Print JSON to stdout, compact for piping."""
    print(json.dumps(data, indent=2, default=str))


def _ensure_running():
    """Make sure daemon is up. Returns True or exits."""
    if is_daemon_running():
        return True
    # Try to start
    db_dir = os.environ.get("BRAIN_DB_DIR",
                            os.path.join(os.path.expanduser("~"), "AgentsContext", "brain"))
    db_path = os.path.join(db_dir, "brain.db")
    if not os.path.isfile(db_path):
        print(json.dumps({"ok": False, "error": f"brain.db not found at {db_path}"}))
        sys.exit(1)
    if not ensure_daemon(db_path):
        print(json.dumps({"ok": False, "error": "Failed to start daemon"}))
        sys.exit(1)
    return True


def cmd_recall(args):
    _ensure_running()
    resp = send_command("recall", {
        "query": args.query,
        "limit": args.limit,
    }, timeout=15.0)
    _print_json(resp)


def cmd_remember(args):
    _ensure_running()
    params = {
        "type": args.type,
        "title": args.title,
        "content": args.content,
    }
    if args.keywords:
        params["keywords"] = " ".join(args.keywords)
    if args.confidence is not None:
        params["confidence"] = args.confidence
    if args.locked:
        params["locked"] = True
    if args.project:
        params["project"] = args.project
    resp = send_command("remember", params, timeout=15.0)
    _print_json(resp)


def cmd_connect(args):
    _ensure_running()
    resp = send_command("connect", {
        "source_id": args.source_id,
        "target_id": args.target_id,
        "relation": args.relation,
        "weight": args.weight,
    })
    _print_json(resp)


def cmd_enrich(args):
    _ensure_running()
    params = {"node_id": args.node_id}
    if args.question:
        params["question"] = args.question
    if args.anchor:
        params["anchor"] = args.anchor
    if args.bridge:
        params["bridge"] = args.bridge
    if args.keywords:
        params["keywords"] = args.keywords
    resp = send_command("enrich", params, timeout=15.0)
    _print_json(resp)


def cmd_ping(args):
    resp = send_command("ping", timeout=2.0)
    _print_json(resp)


def cmd_status(args):
    if not is_daemon_running():
        _print_json({"ok": True, "result": {"status": "not_running"}})
        return
    resp = send_command("ping", timeout=2.0)
    if resp.get("ok"):
        result = resp["result"]
        # Enrich with status file data
        from servers.daemon_config import get_status_path
        status_path = get_status_path()
        if os.path.exists(status_path):
            try:
                with open(status_path) as f:
                    status = json.load(f)
                result.update(status)
            except Exception:
                pass
        _print_json({"ok": True, "result": result})
    else:
        _print_json(resp)


def cmd_eval(args):
    _ensure_running()
    resp = send_command("eval", {"code": args.code}, timeout=30.0)
    _print_json(resp)


def cmd_consciousness(args):
    _ensure_running()
    resp = send_command("consciousness", timeout=10.0)
    _print_json(resp)


def cmd_raw(args):
    """Send raw JSON command to daemon."""
    _ensure_running()
    try:
        msg = json.loads(args.json_str)
    except json.JSONDecodeError as e:
        _print_json({"ok": False, "error": f"Invalid JSON: {e}"})
        sys.exit(1)
    resp = send_command(msg.get("cmd", ""), msg.get("args", {}), timeout=30.0)
    _print_json(resp)


def main():
    parser = argparse.ArgumentParser(
        prog="brain",
        description="Brain CLI — talk to the brain daemon from bash")

    sub = parser.add_subparsers(dest="command")

    # recall
    p = sub.add_parser("recall", help="Semantic recall from brain")
    p.add_argument("query", help="Search query")
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=cmd_recall)

    # remember
    p = sub.add_parser("remember", help="Store a new node")
    p.add_argument("--type", default="context", help="Node type")
    p.add_argument("--title", required=True)
    p.add_argument("--content", required=True)
    p.add_argument("--keywords", nargs="*")
    p.add_argument("--confidence", type=float)
    p.add_argument("--locked", action="store_true")
    p.add_argument("--project")
    p.set_defaults(func=cmd_remember)

    # connect
    p = sub.add_parser("connect", help="Create edge between nodes")
    p.add_argument("source_id")
    p.add_argument("target_id")
    p.add_argument("--relation", default="related_to")
    p.add_argument("--weight", type=float, default=0.5)
    p.set_defaults(func=cmd_connect)

    # enrich
    p = sub.add_parser("enrich", help="Store enrichment vectors for a node")
    p.add_argument("node_id")
    p.add_argument("--question")
    p.add_argument("--anchor")
    p.add_argument("--bridge")
    p.add_argument("--keywords")
    p.set_defaults(func=cmd_enrich)

    # ping
    p = sub.add_parser("ping", help="Check if daemon is alive")
    p.set_defaults(func=cmd_ping)

    # status
    p = sub.add_parser("status", help="Daemon status with brain stats")
    p.set_defaults(func=cmd_status)

    # eval
    p = sub.add_parser("eval", help="Eval Python on brain object")
    p.add_argument("code", help="Python expression (brain object available)")
    p.set_defaults(func=cmd_eval)

    # consciousness
    p = sub.add_parser("consciousness", help="Get consciousness signals")
    p.set_defaults(func=cmd_consciousness)

    # raw
    p = sub.add_parser("raw", help="Send raw JSON command")
    p.add_argument("json_str", help='JSON: {"cmd":"...","args":{}}')
    p.set_defaults(func=cmd_raw)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
