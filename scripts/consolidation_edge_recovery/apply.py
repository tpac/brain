"""
Consolidation edge recovery — APPLY stage.

Reads plan.json, dispatches `connect` ops through the daemon to restore the
edges that CONSOLIDATE/EVOLVE historically deleted. Takes a backup first.
Writes a run log alongside the plan.

Run:
    ./dev python3 scripts/consolidation_edge_recovery/apply.py               # dry run
    ./dev python3 scripts/consolidation_edge_recovery/apply.py --execute     # real run
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from servers.daemon_client import send_command  # noqa: E402

from servers.daemon_config import resolve_db_dir  # noqa: E402

BRAIN_DIR = Path(resolve_db_dir())
LIVE_BRAIN = BRAIN_DIR / "brain.db"
OUT_DIR = Path(__file__).parent / "output"
PLAN_PATH = OUT_DIR / "plan.json"
LOG_PATH = OUT_DIR / "apply.log.jsonl"

ENCODING_SOURCE = "migration:consolidation_edge_recovery-20260421"
BATCH_SIZE = 50


def backup_live_db():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    bak = BRAIN_DIR / f"brain.db.bak-pre-edge-recovery-{ts}"
    print(f"[apply] backing up live DB → {bak}")
    shutil.copy2(LIVE_BRAIN, bak)
    return bak


def load_restore_edges(plan):
    """Flatten plan into a list of edges to write."""
    edges = []
    for entry in plan["entries"]:
        for e in entry["edges"]:
            if e["status"] != "restore":
                continue
            edges.append({
                "source_id": e["resolved_source"],
                "target_id": e["resolved_target"],
                "relation": e["relation"],
                "description": e.get("description", ""),
                "weight": e.get("weight", 0.5),
                "archived_original_id": entry["archived_original"]["id"],
                "canonical_new_id": entry["canonical_new"]["id"],
            })
    return edges


def log_result(log_file, record):
    log_file.write(json.dumps(record) + "\n")
    log_file.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="Actually perform writes. Without this flag, dry-run only.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap total edges written (for staged verification).")
    args = ap.parse_args()

    if not PLAN_PATH.exists():
        print(f"[apply] plan not found at {PLAN_PATH} — run plan.py first")
        sys.exit(1)

    plan = json.loads(PLAN_PATH.read_text())
    edges = load_restore_edges(plan)
    print(f"[apply] plan has {len(edges)} edges to restore "
          f"across {plan['planned_orphans']} orphan nodes")
    if args.limit:
        edges = edges[:args.limit]
        print(f"[apply] limit active → executing first {len(edges)}")

    rel_summary = defaultdict(int)
    for e in edges:
        rel_summary[e["relation"]] += 1
    print(f"[apply] relation breakdown:")
    for rel, cnt in sorted(rel_summary.items(), key=lambda kv: -kv[1]):
        print(f"  {rel:30s}  {cnt}")

    if not args.execute:
        print("\n[apply] DRY RUN — pass --execute to actually write.")
        print(f"[apply] would write {len(edges)} edges via daemon brain_batch")
        print(f"[apply] encoding_source = {ENCODING_SOURCE!r}")
        return

    # Real run
    # Verify daemon is up
    try:
        ping = send_command("ping", timeout=5.0)
        if not ping.get("ok"):
            print(f"[apply] daemon ping failed: {ping}")
            sys.exit(2)
    except Exception as e:
        print(f"[apply] daemon unreachable: {e}")
        sys.exit(2)

    # Backup
    bak = backup_live_db()
    print(f"[apply] backup complete: {bak}")

    # Open log
    log_file = LOG_PATH.open("w")
    log_result(log_file, {
        "event": "apply_start",
        "ts": datetime.now(timezone.utc).isoformat(),
        "total_edges": len(edges),
        "encoding_source": ENCODING_SOURCE,
        "backup": str(bak),
    })

    total = len(edges)
    success = 0
    failed = 0
    t0 = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch = edges[batch_start:batch_start + BATCH_SIZE]
        operations = [{
            "op": "connect",
            "source_id": e["source_id"],
            "target_id": e["target_id"],
            "relation": e["relation"],
            "description": e["description"],
            "weight": e["weight"],
        } for e in batch]

        try:
            resp = send_command("brain_batch", {
                "operations": operations,
                "encoding_source": ENCODING_SOURCE,
            }, timeout=60.0)
        except Exception as e:
            print(f"[apply] batch {batch_start}..{batch_start+len(batch)} FAILED: {e}")
            for e_meta in batch:
                log_result(log_file, {"event": "edge", "ok": False, "error": str(e), **e_meta})
            failed += len(batch)
            continue

        if not resp.get("ok"):
            print(f"[apply] batch dispatcher rejected: {resp}")
            for e_meta in batch:
                log_result(log_file, {"event": "edge", "ok": False,
                                      "error": resp.get("error", "unknown"),
                                      **e_meta})
            failed += len(batch)
            continue

        results = resp.get("result", {}).get("results") or []
        # Each result has ok + op index; align by index into this batch
        by_index = {r.get("index"): r for r in results}
        for i, e_meta in enumerate(batch):
            r = by_index.get(i, {})
            ok = r.get("ok", False)
            log_result(log_file, {
                "event": "edge",
                "ok": ok,
                "error": r.get("error") if not ok else None,
                **e_meta,
            })
            if ok:
                success += 1
            else:
                failed += 1

        done = batch_start + len(batch)
        rate = done / max(time.time() - t0, 0.01)
        print(f"[apply] {done}/{total}  ok={success} fail={failed}  ({rate:.1f} edges/s)")

    elapsed = time.time() - t0
    log_result(log_file, {
        "event": "apply_end",
        "ts": datetime.now(timezone.utc).isoformat(),
        "success": success,
        "failed": failed,
        "elapsed_sec": elapsed,
    })
    log_file.close()

    print(f"\n[apply] done in {elapsed:.1f}s")
    print(f"[apply] success: {success}  failed: {failed}")
    print(f"[apply] log: {LOG_PATH}")
    if failed:
        print("[apply] failed rows are in the log — review before rerunning")


if __name__ == "__main__":
    main()
