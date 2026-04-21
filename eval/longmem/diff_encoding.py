"""Targeted script: replay specific items under two encoder prompt versions
and dump what each encoded. Lets us see exactly what the temporal addition
changed in the encoder's output for items that regressed.

Usage:
  ./dev python3 eval/longmem/diff_encoding.py --qids fca762bc 54026fce \\
      --versions 5 6

Keeps the brain DBs so we can re-inspect later.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def dump_nodes(brain) -> list:
    """All non-seed nodes with their key fields."""
    rows = brain.conn.execute("""
        SELECT n.id, n.type, n.title, n.content, n.encoding_source
        FROM nodes n
        WHERE n.encoding_source != 'anchor:seed' OR n.encoding_source IS NULL
        ORDER BY n.created_at
    """).fetchall()
    out = []
    for nid, ntype, title, content, src in rows:
        kv = dict(brain.conn.execute(
            "SELECT key, value FROM node_metadata_kv WHERE node_id=? "
            "AND key IN ('situation','question','event_time','reasoning')",
            (nid,)).fetchall())
        out.append({
            "id": nid[:8], "type": ntype, "title": title,
            "content": (content or "")[:300],
            "situation": kv.get("situation", "")[:150],
            "event_time": kv.get("event_time", ""),
            "src": src,
        })
    return out


def dump_edges(brain, node_ids: list) -> list:
    """All edges between the given nodes."""
    if not node_ids:
        return []
    phs = ",".join("?" * len(node_ids))
    rows = brain.conn.execute(f"""
        SELECT e.source_id, e.target_id, er.relation, er.description
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        WHERE e.source_id IN ({phs}) AND e.target_id IN ({phs})
        AND er.archived = 0
    """, node_ids * 2).fetchall()
    return [
        {"source": s[:8], "target": t[:8], "relation": r, "why": (d or "")[:80]}
        for s, t, r, d in rows
    ]


def register_prompt(version: int):
    """Point s1e at a previously-registered version."""
    from servers.daemon_client import send_command
    r = send_command('get_interaction', {'name': 's1e', 'version': version})
    if not r.get('ok'):
        raise RuntimeError(f"failed to fetch v{version}: {r}")
    t = r['result']['template']
    # Register as a new version (becomes latest — runtime reads latest)
    send_command('register_interaction', {
        'name': 's1e', 'template': t,
        'parameters': r['result']['parameters'],
        'created_by': f'diff_encoding:cloned_v{version}',
    })


def run_for_item(brain, item: dict, log_prefix: str):
    """Replay one item's haystack. Does NOT do the final query — just ingest."""
    from eval.longmem.replay import replay_item
    session_id = f"diff-{item['question_id']}"
    replay_item(brain, session_id, item["haystack_sessions"],
                haystack_dates=item.get("haystack_dates"),
                log_prefix=log_prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qids", nargs="+", required=True,
                        help="Question IDs to replay (from oracle)")
    parser.add_argument("--versions", type=int, nargs="+", default=[5, 6],
                        help="s1e prompt versions to compare")
    parser.add_argument("--oracle", default="eval/longmem/data/longmemeval_oracle.json")
    args = parser.parse_args()

    # Load env
    from pathlib import Path
    envf = Path(".env")
    if envf.exists():
        for line in envf.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                key, val = k.strip(), v.strip().strip('"').strip("'")
                if not os.environ.get(key):
                    os.environ[key] = val

    with open(args.oracle) as f:
        data = json.load(f)

    items = [it for it in data if it["question_id"] in args.qids]
    assert len(items) == len(args.qids), f"missing qids: {set(args.qids) - {i['question_id'] for i in items}}"

    results = {}  # {qid: {version: {nodes, edges}}}
    for version in args.versions:
        print(f"\n{'='*70}\n[diff] REGISTERING s1e v{version} as latest\n{'='*70}", flush=True)
        register_prompt(version)

        for item in items:
            qid = item["question_id"]
            print(f"\n--- item {qid} / version {version} ---", flush=True)

            from eval.longmem.fresh_brain import create_fresh_eval_brain
            path = os.path.expanduser(f"~/AgentsContext/brain-diff-v{version}-{qid}")
            brain = create_fresh_eval_brain(path=path, wipe=True)
            run_for_item(brain, item, log_prefix=f"[v{version}/{qid}]")

            nodes = dump_nodes(brain)
            node_ids_full = [r[0] for r in brain.conn.execute(
                "SELECT id FROM nodes WHERE encoding_source != 'anchor:seed' OR encoding_source IS NULL"
            ).fetchall()]
            edges = dump_edges(brain, node_ids_full)

            results.setdefault(qid, {})[version] = {"nodes": nodes, "edges": edges, "path": path}
            try:
                brain.close()
            except Exception:
                pass

    # Report
    print(f"\n\n{'#'*70}\n# ENCODING DIFF\n{'#'*70}")
    for qid, by_ver in results.items():
        print(f"\n\n=== {qid} — '{next(i for i in items if i['question_id']==qid)['question'][:100]}' ===")
        print(f"    gold: {str(next(i for i in items if i['question_id']==qid)['answer'])[:120]}")
        for version in args.versions:
            r = by_ver[version]
            print(f"\n--- v{version} ({len(r['nodes'])} nodes, {len(r['edges'])} edges) — {r['path']}")
            for n in r["nodes"]:
                tm = f" event_time={n['event_time']}" if n.get("event_time") else ""
                print(f"  [{n['type']}] {n['title']}{tm}")
                if n["situation"]:
                    print(f"      sit: {n['situation']}")
                if n["content"][:120].strip() and n["content"][:120] != n["title"][:120]:
                    print(f"      c:   {n['content'][:160]}")
            if r["edges"]:
                print(f"  edges:")
                for e in r["edges"]:
                    print(f"    {e['source']} --{e['relation']}--> {e['target']}    {e['why']}")

    # Re-register v6 as final state (so runtime stays on latest temporal)
    print(f"\n[diff] restoring s1e to v6 as latest")
    register_prompt(6)


if __name__ == "__main__":
    main()
