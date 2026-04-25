"""Snapshot replay — replay a real conversation through v14+SPLIT encoder
against a snapshot of the live brain. Compare to what production (v12)
actually encoded for that same session.

Why this matters: the synthetic A/B harness (s1s_ab_wiring_check.py) runs
on a fresh seed brain (16 nodes), which under-tests revise paths and
catalog-aware behaviors. This harness uses the REAL prod catalog at a
point in time, so revise/connect-to-catalog/sibling-aware connect_to all
fire against meaningful state. The conversation actually ran on prod, so
production's outputs are the v12 baseline — for free, no rerun needed.

Usage:
    ./dev python3 eval/s1s_snapshot_replay.py \\
        --snapshot ~/AgentsContext/brain/brain.db.bak-pre-situation-migration \\
        --conversation ~/.claude/projects/-Users-tpac-brain/c9a23893-...jsonl \\
        --run-name first_replay
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "eval" / "reports" / "snapshot_replay"


def load_jsonl_messages(jsonl_path: str) -> List[Dict[str, str]]:
    """Extract user/assistant text pairs from a Claude session JSONL."""
    messages: List[Dict[str, str]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            etype = entry.get("type")
            if etype not in ("user", "assistant"):
                continue
            msg = entry.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or etype
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
                text = "\n".join(t for t in texts if t)
            elif isinstance(content, str):
                text = content
            else:
                text = ""
            text = (text or "").strip()
            if not text:
                continue
            if role == "user" and len(text) < 3:
                continue  # skip noise like "ok"
            messages.append({"role": role, "content": text[:8000]})
    return messages


def session_id_from_jsonl(jsonl_path: str) -> str:
    """Pull the original session_id so we can compare to production's
    encoder traces for that same session."""
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            sid = entry.get("sessionId")
            if sid:
                return sid
    raise ValueError(f"no sessionId found in {jsonl_path}")


def fetch_production_baseline(orig_session_id: str) -> Dict[str, Any]:
    """Pull what production (v12) actually encoded for this session.
    Reads the LIVE brain_logs.db read-only — never writes."""
    logs_path = os.path.expanduser("~/AgentsContext/brain/brain_logs.db")
    if not os.path.exists(logs_path):
        return {"error": f"no live logs at {logs_path}"}
    con = sqlite3.connect(f"file:{logs_path}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT chain_id, metadata FROM trace_events "
        "WHERE session_id = ? AND ref_type = 'encoding_run' "
        "AND event_type = 'delta' AND metadata IS NOT NULL "
        "ORDER BY id",
        (orig_session_id,)
    ).fetchall()

    cycles = []
    total_actions = 0
    total_writes = 0
    nested_op_counts: Dict[str, int] = {}
    for chain_id, metadata in rows:
        try:
            m = json.loads(metadata)
        except Exception:
            continue
        if "rounds" not in m:
            continue
        cycles.append({
            "chain": chain_id,
            "rounds": m.get("rounds"),
            "actions": m.get("actions"),
            "writes": m.get("write_actions"),
            "outcomes": m.get("outcomes"),
        })
        total_actions += int(m.get("actions") or 0)
        total_writes += int(m.get("write_actions") or 0)
        for a in m.get("action_details", []) or []:
            inp = a.get("input") or {}
            for op in inp.get("operations", []):
                nested_op_counts[op.get("op", "?")] = (
                    nested_op_counts.get(op.get("op", "?"), 0) + 1)
    return {
        "session_id": orig_session_id,
        "encoding_cycles": cycles,
        "total_cycles": len(cycles),
        "total_actions": total_actions,
        "total_writes": total_writes,
        "nested_op_counts": nested_op_counts,
    }


def replay(snapshot: str, conversation: str, run_name: str) -> Dict[str, Any]:
    if not os.path.exists(snapshot):
        raise FileNotFoundError(f"snapshot not found: {snapshot}")
    if not os.path.exists(conversation):
        raise FileNotFoundError(f"conversation not found: {conversation}")

    work = REPORTS_DIR / run_name
    work.mkdir(parents=True, exist_ok=True)
    db_dest = work / "brain.db"
    logs_dest = work / "brain_logs.db"

    print(f"[replay] copying snapshot ({os.path.getsize(snapshot)/1e6:.1f}MB) → {db_dest}",
          flush=True)
    shutil.copy2(snapshot, db_dest)
    # logs DB starts fresh — replay's traces shouldn't mix with production history
    if logs_dest.exists():
        logs_dest.unlink()

    # Set BRAIN_DB_DIR so any code that resolves it gets the copy, not live.
    os.environ["BRAIN_DB_DIR"] = str(work)

    print(f"[replay] parsing conversation: {Path(conversation).name}", flush=True)
    orig_session_id = session_id_from_jsonl(conversation)
    messages = load_jsonl_messages(conversation)
    user_turns = sum(1 for m in messages if m["role"] == "user")
    print(f"[replay] {len(messages)} messages, {user_turns} user turns "
          f"(orig session_id: {orig_session_id})",
          flush=True)

    # Pull production baseline BEFORE we open our own brain (read-only on live).
    print(f"[replay] fetching production baseline (v12 actually-encoded)…",
          flush=True)
    prod_baseline = fetch_production_baseline(orig_session_id)
    print(f"[replay] prod: {prod_baseline.get('total_cycles', 0)} cycles, "
          f"{prod_baseline.get('total_actions', 0)} actions, "
          f"ops: {prod_baseline.get('nested_op_counts', {})}",
          flush=True)

    # Open Brain on the snapshot copy. Schema migrations run on first open.
    from servers.brain import Brain
    brain = Brain(str(db_dest))

    # Register v14+SPLIT prompt as `s1e` in the snapshot copy (idempotent).
    from eval.s1s_v13_prompt import extract_v13_prompt
    from servers.daemon_dispatch import COMMAND_TABLE
    v14_prompt = extract_v13_prompt()
    print(f"[replay] v14+SPLIT prompt: {len(v14_prompt)} chars", flush=True)

    def dispatch(cmd, args=None):
        entry = COMMAND_TABLE.get(cmd)
        if not entry:
            return {"ok": False, "error": f"unknown: {cmd}"}
        return entry.handler(brain, args or {}, [])

    reg = dispatch("register_interaction", {
        "name": "s1e", "template": v14_prompt, "parameters": "",
        "created_by": "snapshot_replay",
    })
    if not reg.get("ok"):
        raise RuntimeError(f"register_interaction failed: {reg}")
    print(f"[replay] registered s1e v{reg.get('result', {}).get('version', '?')}",
          flush=True)

    # Run replay using the existing _drive_encoding helper (mirrors prod path).
    from eval.s1s_ab_wiring_check import _drive_encoding, _fetch_encoding_events, _fetch_new_nodes, _fetch_new_edges

    replay_session_id = f"replay-{run_name}-{os.getpid()}"
    t_mark = time.time()
    print(f"[replay] driving encoder over {user_turns} turns (muster=True)…",
          flush=True)

    stats = _drive_encoding(brain, messages,
                            dispatch=dispatch,
                            session_id=replay_session_id,
                            muster_enabled=True,
                            log_prefix=f"[{run_name}]")

    encoding_events = _fetch_encoding_events(brain, replay_session_id)
    new_nodes = _fetch_new_nodes(brain, t_mark)
    new_edges = _fetch_new_edges(brain, t_mark)

    elapsed_total = time.time() - t_mark

    # Comparison summary
    replay_cycles = []
    replay_op_counts: Dict[str, int] = {}
    for ev in encoding_events:
        m = ev.get("metadata") or {}
        if "rounds" not in m:
            continue
        replay_cycles.append({
            "rounds": m.get("rounds"),
            "actions": m.get("actions"),
            "writes": m.get("write_actions"),
            "outcomes": m.get("outcomes"),
        })
        for a in m.get("action_details", []) or []:
            inp = a.get("input") or {}
            for op in inp.get("operations", []):
                replay_op_counts[op.get("op", "?")] = (
                    replay_op_counts.get(op.get("op", "?"), 0) + 1)

    summary = {
        "run_name": run_name,
        "snapshot": snapshot,
        "conversation": conversation,
        "orig_session_id": orig_session_id,
        "replay_session_id": replay_session_id,
        "messages": len(messages),
        "user_turns": user_turns,
        "elapsed_s": round(elapsed_total, 1),
        "production_v12": prod_baseline,
        "replay_v14": {
            "encoding_cycles": replay_cycles,
            "total_cycles": len(replay_cycles),
            "nested_op_counts": replay_op_counts,
            "new_nodes": len(new_nodes),
            "new_edges": len(new_edges),
        },
        "stats": stats,
    }

    (work / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"[replay] done — wrote {work}/summary.json ({elapsed_total:.1f}s)",
          flush=True)

    try:
        brain.close()
    except Exception:
        pass

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True,
                        help="Path to brain.db.bak-* snapshot")
    parser.add_argument("--conversation", required=True,
                        help="Path to ~/.claude/projects/.../session.jsonl")
    parser.add_argument("--run-name", required=True,
                        help="Output dir name under eval/reports/snapshot_replay/")
    args = parser.parse_args()
    summary = replay(args.snapshot, args.conversation, args.run_name)

    # Print the comparison
    print("\n" + "=" * 70)
    print("PRODUCTION (v12, actually ran)  vs  REPLAY (v14+SPLIT, simulation)")
    print("=" * 70)
    p = summary["production_v12"]
    r = summary["replay_v14"]
    print(f"Cycles:          v12 {p.get('total_cycles', 0)}      v14 {r.get('total_cycles', 0)}")
    print(f"Total actions:   v12 {p.get('total_actions', 0)}      v14 {sum(c.get('actions') or 0 for c in r.get('encoding_cycles', []))}")
    print(f"Op counts        v12 {dict(sorted(p.get('nested_op_counts', {}).items()))}")
    print(f"                 v14 {dict(sorted(r.get('nested_op_counts', {}).items()))}")
    print(f"New nodes (v14): {r.get('new_nodes', 0)}")
    print(f"New edges (v14): {r.get('new_edges', 0)}")
    print(f"\nDetails: {REPORTS_DIR}/{args.run_name}/summary.json")


if __name__ == "__main__":
    main()
