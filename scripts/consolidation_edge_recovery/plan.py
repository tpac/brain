"""
Consolidation edge recovery — PLAN stage.

Historic S2 consolidation runs archived original nodes but DELETED their edges
instead of rewriting them to point at the synthesized new node. This script
builds a recovery plan (JSON + human-readable report) — pure read, no writes.

Flow:
1. Load archived → new_node mapping from s2 consolidation trace deltas.
2. Chain-resolve new_node to current canonical form (follow `consolidated_into`).
3. Pick best backup per orphan (most recent with edges intact).
4. Extract edges from that backup (skip meta relations).
5. Chain-resolve other endpoint of each edge.
6. Dedup against existing live edges.
7. Write plan.json + REPORT.md.

Run: ./dev python3 scripts/consolidation_edge_recovery/plan.py
"""
import sqlite3
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from servers.daemon_config import resolve_db_dir  # noqa: E402

BRAIN_DIR = Path(resolve_db_dir())
LIVE_BRAIN = BRAIN_DIR / "brain.db"
LIVE_LOGS = BRAIN_DIR / "brain_logs.db"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

META_RELATIONS = {"similar_to", "consolidated_into"}
# Relations excluded from recovery — co_accessed is 79% of the graph and represents
# usage patterns, not semantic meaning. See decision d1d1a90c: no traversal value,
# pure noise at recovery time.
NOISE_RELATIONS = {"co_accessed"}
EXCLUDED_RELATIONS = META_RELATIONS | NOISE_RELATIONS
CHAIN_DEPTH_CAP = 5


def ro_conn(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_consolidation_links(logs_db, live_db):
    """Parse s2 consolidation trace deltas → [(archived_id, new_node_id, chain_id, ts, action_type)].

    Authoritative mapping: connect ops with relations `consolidated_into` or `evolves_from`.
    Direction is inconsistent across encoder runs — resolve by checking which endpoint
    is archived in the live DB.
    """
    rows = logs_db.execute("""
        SELECT chain_id, metadata, created_at
        FROM trace_events
        WHERE chain_id LIKE 's2-%consolidation%' AND event_type='delta'
        ORDER BY created_at
    """).fetchall()

    # Cache archived status
    archived_cache = {}
    def is_archived(nid):
        if nid not in archived_cache:
            r = live_db.execute("SELECT archived FROM nodes WHERE id=?", (nid,)).fetchone()
            archived_cache[nid] = r[0] if r else None
        return archived_cache[nid]

    links = []
    skipped_ambiguous = 0
    for chain, md_str, ts in rows:
        md = json.loads(md_str)
        for ad in md.get("action_details", []):
            ops = (ad.get("input") or {}).get("operations") or []
            for op in ops:
                if op.get("op") != "connect":
                    continue
                rel = op.get("relation")
                # Lineage relations — the encoder used several interchangeable
                # names across runs. All signal "X was the old form of Y."
                if rel not in ("consolidated_into", "evolves_from",
                               "supersedes", "corrects"):
                    continue
                a = op.get("source_id")
                b = op.get("target_id")
                if not a or not b:
                    continue
                a_arch = is_archived(a)
                b_arch = is_archived(b)
                # Pick archived side as original, live side as new
                if a_arch == 1 and b_arch == 0:
                    archived_id, new_id = a, b
                elif b_arch == 1 and a_arch == 0:
                    archived_id, new_id = b, a
                elif a_arch == 1 and b_arch == 1:
                    # Both archived — chain-resolve the new side later
                    # Pick the one that's NOT in archive ops of this batch as "new"
                    batch_archives = {op2.get("id") or op2.get("node_id")
                                      for op2 in ops if op2.get("op") == "archive"}
                    if a in batch_archives and b not in batch_archives:
                        archived_id, new_id = a, b
                    elif b in batch_archives and a not in batch_archives:
                        archived_id, new_id = b, a
                    else:
                        skipped_ambiguous += 1
                        continue
                else:
                    # Both live (unusual) or one missing — skip
                    skipped_ambiguous += 1
                    continue
                action_type = "CONSOLIDATE" if rel == "consolidated_into" else f"LINEAGE:{rel}"
                links.append((archived_id, new_id, chain, ts, action_type))

    if skipped_ambiguous:
        print(f"[plan] {skipped_ambiguous} connect ops skipped (ambiguous direction)")
    return links


def chain_resolve(live_db, node_id, depth=0, seen=None):
    """Follow consolidated_into forward to current canonical node. Returns (canonical_id, was_chained, archived)."""
    if seen is None:
        seen = set()
    if depth >= CHAIN_DEPTH_CAP or node_id in seen:
        return node_id, False, None
    seen.add(node_id)
    row = live_db.execute("SELECT archived FROM nodes WHERE id=?", (node_id,)).fetchone()
    if not row:
        return node_id, depth > 0, None  # missing
    archived = row[0]
    if not archived:
        return node_id, depth > 0, 0
    nxt = live_db.execute("""
        SELECT e.target_id
        FROM edges e
        JOIN edge_relations er ON e.edge_id = er.edge_id
        WHERE e.source_id=? AND er.relation='consolidated_into'
        LIMIT 1
    """, (node_id,)).fetchone()
    if not nxt:
        return node_id, depth > 0, 1  # archived dead end
    return chain_resolve(live_db, nxt[0], depth + 1, seen)


def extract_edges_for_node(backup_db, node_id):
    """Get all non-meta edges for a node from a backup. Returns list of dicts."""
    out = []
    # Outgoing
    rows = backup_db.execute("""
        SELECT e.edge_id, e.source_id, e.target_id, e.weight AS edge_weight,
               er.relation, er.description, er.weight AS rel_weight, er.encoding_source
        FROM edges e
        JOIN edge_relations er ON e.edge_id = er.edge_id
        WHERE e.source_id=?
    """, (node_id,)).fetchall()
    for r in rows:
        rel = r[4]
        if rel in EXCLUDED_RELATIONS:
            continue
        out.append({
            "direction": "out",
            "other_endpoint": r[2],
            "relation": rel,
            "description": r[5] or "",
            "weight": r[6],
            "edge_weight": r[3],
            "encoding_source": r[7] or "",
        })
    # Incoming
    rows = backup_db.execute("""
        SELECT e.edge_id, e.source_id, e.target_id, e.weight AS edge_weight,
               er.relation, er.description, er.weight AS rel_weight, er.encoding_source
        FROM edges e
        JOIN edge_relations er ON e.edge_id = er.edge_id
        WHERE e.target_id=?
    """, (node_id,)).fetchall()
    for r in rows:
        rel = r[4]
        if rel in EXCLUDED_RELATIONS:
            continue
        out.append({
            "direction": "in",
            "other_endpoint": r[1],
            "relation": rel,
            "description": r[5] or "",
            "weight": r[6],
            "edge_weight": r[3],
            "encoding_source": r[7] or "",
        })
    return out


def pick_backup_for(node_id, backups_ranked, backup_cache):
    """Return (backup_path, edges_count) for the first backup where node has edges."""
    for path, _mtime in backups_ranked:
        if path not in backup_cache:
            try:
                conn = ro_conn(path)
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                if "edge_relations" not in tables or "edges" not in tables:
                    conn.close()
                    backup_cache[path] = None
                    continue
                # Require v22+ edges schema (has edge_id column)
                cols = {r[1] for r in conn.execute("PRAGMA table_info(edges)").fetchall()}
                if "edge_id" not in cols:
                    conn.close()
                    backup_cache[path] = None
                    continue
                backup_cache[path] = conn
            except Exception:
                backup_cache[path] = None
                continue
        conn = backup_cache[path]
        if conn is None:
            continue
        cnt = conn.execute("""
            SELECT COUNT(*) FROM (
              SELECT edge_id FROM edges WHERE source_id=?
              UNION ALL
              SELECT edge_id FROM edges WHERE target_id=?
            )
        """, (node_id, node_id)).fetchone()[0]
        if cnt > 0:
            return path, cnt
    return None, 0


def live_edge_exists(live_db, source, target, relation):
    """Check if (source→target, relation) already exists as active edge."""
    row = live_db.execute("""
        SELECT 1
        FROM edges e
        JOIN edge_relations er ON e.edge_id = er.edge_id
        WHERE e.source_id=? AND e.target_id=? AND er.relation=?
          AND COALESCE(er.archived, 0) = 0
        LIMIT 1
    """, (source, target, relation)).fetchone()
    return row is not None


def node_meta(db, node_id):
    row = db.execute("SELECT title, type, archived FROM nodes WHERE id=?", (node_id,)).fetchone()
    if not row:
        return None
    return {"id": node_id, "title": row[0], "type": row[1], "archived": row[2]}


def main():
    print(f"[plan] reading {LIVE_BRAIN}")
    live = ro_conn(LIVE_BRAIN)
    logs = ro_conn(LIVE_LOGS)

    # Rank backups by mtime, newest first (prefer recent to match any node's latest pre-archival state)
    backups = []
    for f in os.listdir(BRAIN_DIR):
        p = BRAIN_DIR / f
        if (f.startswith("brain.db.bak-") or f.startswith("brain.db.v") or f == "brain.db.backup-pre-v14-20260317-144005") \
                and not f.endswith("-shm") and not f.endswith("-wal") \
                and "corrupted" not in f:
            backups.append((str(p), p.stat().st_mtime))
    backups.sort(key=lambda x: -x[1])  # newest first
    print(f"[plan] {len(backups)} candidate backups")

    links = load_consolidation_links(logs, live)
    print(f"[plan] {len(links)} (archived, new_node) links from traces")

    # Deduplicate by archived_id (keep first encounter = earliest consolidation)
    first_link = {}
    for aid, nid, chain, ts, action in links:
        if aid and aid not in first_link:
            first_link[aid] = (nid, chain, ts, action)

    # Filter to true orphans (zero edges in live db)
    orphans = []
    for aid in first_link:
        has_out = live.execute("SELECT 1 FROM edges WHERE source_id=? LIMIT 1", (aid,)).fetchone()
        has_in = live.execute("SELECT 1 FROM edges WHERE target_id=? LIMIT 1", (aid,)).fetchone()
        if not has_out and not has_in:
            orphans.append(aid)
    print(f"[plan] {len(orphans)} orphans (no edges in live db)")

    backup_cache = {}
    plan_entries = []
    skipped = []

    for aid in orphans:
        new_id, chain, ts, action = first_link[aid]
        orig_meta = node_meta(live, aid)

        if not new_id:
            skipped.append({"orphan": aid, "reason": "no new_node in trace", "meta": orig_meta})
            continue

        canonical_new, was_chained, new_archived = chain_resolve(live, new_id)
        canonical_meta = node_meta(live, canonical_new)

        if new_archived == 1 or canonical_meta is None or canonical_meta.get("archived") == 1:
            skipped.append({
                "orphan": aid,
                "reason": "synthesized chain terminates at archived/missing node",
                "original_new_id": new_id,
                "canonical_new_id": canonical_new,
                "meta": orig_meta,
            })
            continue

        bak_path, edge_count = pick_backup_for(aid, backups, backup_cache)
        if not bak_path:
            skipped.append({"orphan": aid, "reason": "no backup with edges found", "meta": orig_meta})
            continue

        bak = backup_cache[bak_path]
        raw_edges = extract_edges_for_node(bak, aid)

        resolved_edges = []
        for e in raw_edges:
            other = e["other_endpoint"]
            can_other, other_chained, other_arch = chain_resolve(live, other)
            other_meta = node_meta(live, can_other)
            if other_meta is None or other_meta.get("archived") == 1:
                e["status"] = "skip"
                e["skip_reason"] = "other endpoint archived/missing and no live canonical"
                e["canonical_other"] = can_other
                e["other_title"] = other_meta["title"] if other_meta else None
                resolved_edges.append(e)
                continue
            if can_other == canonical_new:
                e["status"] = "skip"
                e["skip_reason"] = "self-edge after canonicalization"
                e["canonical_other"] = can_other
                resolved_edges.append(e)
                continue
            # Dedup against live
            if e["direction"] == "out":
                src, tgt = canonical_new, can_other
            else:
                src, tgt = can_other, canonical_new
            if live_edge_exists(live, src, tgt, e["relation"]):
                e["status"] = "dup"
                e["skip_reason"] = "edge already exists in live"
                e["canonical_other"] = can_other
                e["other_title"] = other_meta["title"]
                e["resolved_source"] = src
                e["resolved_target"] = tgt
                resolved_edges.append(e)
                continue
            e["status"] = "restore"
            e["canonical_other"] = can_other
            e["other_title"] = other_meta["title"]
            e["other_chained"] = other_chained
            e["resolved_source"] = src
            e["resolved_target"] = tgt
            resolved_edges.append(e)

        plan_entries.append({
            "archived_original": orig_meta,
            "canonical_new": canonical_meta,
            "new_chained": was_chained,
            "action_type": action,
            "backup": os.path.basename(bak_path),
            "consolidation_chain": chain,
            "consolidation_ts": ts,
            "edges": resolved_edges,
        })

    # Stats
    n_restore = sum(1 for e in plan_entries for x in e["edges"] if x["status"] == "restore")
    n_dup = sum(1 for e in plan_entries for x in e["edges"] if x["status"] == "dup")
    n_skip_edge = sum(1 for e in plan_entries for x in e["edges"] if x["status"] == "skip")
    rel_counts = defaultdict(int)
    for e in plan_entries:
        for x in e["edges"]:
            if x["status"] == "restore":
                rel_counts[x["relation"]] += 1

    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_orphans": len(orphans),
        "planned_orphans": len(plan_entries),
        "skipped_orphans": len(skipped),
        "edge_stats": {
            "to_restore": n_restore,
            "duplicates": n_dup,
            "skipped_edges": n_skip_edge,
        },
        "relation_breakdown": dict(sorted(rel_counts.items(), key=lambda kv: -kv[1])),
        "entries": plan_entries,
        "skipped": skipped,
    }

    out_json = OUT_DIR / "plan.json"
    out_json.write_text(json.dumps(plan, indent=2))
    print(f"[plan] wrote {out_json}")
    print(f"[plan] orphans planned: {len(plan_entries)} / skipped: {len(skipped)}")
    print(f"[plan] edges: restore={n_restore} dup={n_dup} skip={n_skip_edge}")

    # Human-readable report
    lines = ["# Consolidation Edge Recovery — Plan\n"]
    lines.append(f"- Generated: {plan['generated_at']}")
    lines.append(f"- Total orphans (zero edges in live): **{plan['total_orphans']}**")
    lines.append(f"- Planned for recovery: **{plan['planned_orphans']}**")
    lines.append(f"- Skipped orphans: {plan['skipped_orphans']}")
    lines.append(f"- Edges to restore: **{n_restore}**")
    lines.append(f"- Duplicates (already live): {n_dup}")
    lines.append(f"- Skipped edges (dead endpoints/self): {n_skip_edge}")
    lines.append("\n## Relation breakdown (to restore)\n")
    for rel, cnt in sorted(rel_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{rel}`: {cnt}")

    lines.append("\n## Skip reasons (orphan-level)\n")
    skip_reasons = defaultdict(int)
    for s in skipped:
        skip_reasons[s["reason"]] += 1
    for reason, cnt in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {reason}: {cnt}")

    lines.append("\n## Backup usage\n")
    bak_counts = defaultdict(int)
    for e in plan_entries:
        bak_counts[e["backup"]] += 1
    for b, c in sorted(bak_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{b}`: {c} orphans")

    lines.append("\n## Sample entries (first 10)\n")
    for e in plan_entries[:10]:
        lines.append(f"\n### {e['archived_original']['id']} · {e['archived_original']['title']!r}")
        lines.append(f"- → canonical new: `{e['canonical_new']['id']}` · {e['canonical_new']['title']!r}"
                     + (" (chained)" if e["new_chained"] else ""))
        lines.append(f"- backup: `{e['backup']}`")
        lines.append(f"- consolidation: `{e['consolidation_chain']}` at {e['consolidation_ts']}")
        by_status = defaultdict(int)
        for x in e["edges"]:
            by_status[x["status"]] += 1
        lines.append(f"- edges: {dict(by_status)}")
        for x in e["edges"][:5]:
            tag = x["status"].upper()
            arrow = "→" if x["direction"] == "out" else "←"
            other_title = x.get("other_title") or "?"
            lines.append(f"  - [{tag}] {arrow} `{x.get('canonical_other', x['other_endpoint'])}` "
                         f"({other_title!r}) · `{x['relation']}` — {x['description'][:80]!r}")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"[plan] wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
