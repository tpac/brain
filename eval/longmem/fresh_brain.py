"""Fresh eval brain at ~/AgentsContext/brain-eval/ — dashboard-watchable.

Per-item flow:
  1. wipe_eval_dir()            — nuke prior state (or keep for inspection)
  2. create_fresh_eval_brain()  — new brain, seed pack auto-loads via brain init
  3. replay item's haystack     — ingestion populates the brain
  4. query                      — answer from surfaced K
  5. reset_to_seeds(brain)      — between items: drop user nodes, keep seeds

The dashboard can be launched in a second window with:
    BRAIN_DB_DIR=~/AgentsContext/brain-eval/ python dashboard/brain_dashboard_standalone.py

It's a passive reader — no writes, no lock contention with the eval run.
"""
import os
import shutil
import sqlite3
from typing import Optional

EVAL_BRAIN_DIR = os.path.expanduser("~/AgentsContext/brain-eval")


def per_item_brain_dir(qid: str, run_name: str = None) -> str:
    """Return a per-item eval brain path, isolated from other items and runs.

    Path shape: ~/AgentsContext/brain-eval-{run_name}/{qid}/
    If run_name is None, falls back to a generic per-item dir under
    ~/AgentsContext/brain-eval-items/{qid}/ — handy for one-off scripts.
    """
    if run_name:
        base = os.path.expanduser(f"~/AgentsContext/brain-eval-{run_name}")
    else:
        base = os.path.expanduser("~/AgentsContext/brain-eval-items")
    return os.path.join(base, qid)


def wipe_eval_dir(path: str = EVAL_BRAIN_DIR) -> None:
    """Delete the eval brain directory if it exists. Destructive — confirm usage in caller."""
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[fresh-brain] wiped {path}", flush=True)
    os.makedirs(path, exist_ok=True)


def create_fresh_eval_brain(path: str = EVAL_BRAIN_DIR, wipe: bool = True):
    """Create a new Brain instance at the eval path with seed pack loaded.

    Args:
        path: brain directory (default: ~/AgentsContext/brain-eval)
        wipe: wipe the directory first (default True — fresh start)

    Returns:
        Brain instance
    """
    if wipe:
        wipe_eval_dir(path)
    else:
        os.makedirs(path, exist_ok=True)

    os.environ["BRAIN_DB_DIR"] = path
    db_path = os.path.join(path, "brain.db")

    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from servers.brain import Brain
    brain = Brain(db_path=db_path)
    print(f"[fresh-brain] brain ready at {path}", flush=True)
    return brain


def reset_to_seeds(brain) -> dict:
    """Drop all non-seed nodes and their edges. Keeps the seed pack intact.

    Use between LongMemEval items so each item starts from a clean-plus-seeds state
    without paying the seed-loading cost again.

    Returns:
        {"nodes_removed": N, "edges_removed": N, "seeds_kept": N}
    """
    conn: sqlite3.Connection = brain.conn

    # Which nodes are seeds
    seed_ids = {row[0] for row in conn.execute(
        "SELECT id FROM nodes WHERE encoding_source = 'anchor:seed'"
    ).fetchall()}
    seeds_kept = len(seed_ids)

    # Count + delete non-seed nodes (cascade to node_enrichments, metadata_kv via FK or explicit)
    non_seed_ids = [row[0] for row in conn.execute(
        "SELECT id FROM nodes WHERE encoding_source != 'anchor:seed' OR encoding_source IS NULL"
    ).fetchall()]
    nodes_removed = len(non_seed_ids)

    if non_seed_ids:
        placeholders = ",".join("?" * len(non_seed_ids))
        # Delete dependent rows first (schema may not have FK cascade everywhere)
        for table in ("node_enrichments", "node_metadata_kv", "node_embeddings", "node_vectors"):
            try:
                conn.execute(f"DELETE FROM {table} WHERE node_id IN ({placeholders})", non_seed_ids)
            except sqlite3.OperationalError:
                pass  # table may not exist in all schema versions
        conn.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", non_seed_ids)

    # Edges: keep only those where both endpoints are seeds
    edges_before = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    if seed_ids:
        seed_placeholders = ",".join("?" * len(seed_ids))
        seed_list = list(seed_ids)
        # Find edges to delete (at least one endpoint is non-seed)
        edge_ids_to_remove = [row[0] for row in conn.execute(
            f"SELECT edge_id FROM edges WHERE source_id NOT IN ({seed_placeholders}) "
            f"OR target_id NOT IN ({seed_placeholders})",
            seed_list + seed_list
        ).fetchall()]
        if edge_ids_to_remove:
            eph = ",".join("?" * len(edge_ids_to_remove))
            conn.execute(f"DELETE FROM edge_relations WHERE edge_id IN ({eph})", edge_ids_to_remove)
            conn.execute(f"DELETE FROM edges WHERE edge_id IN ({eph})", edge_ids_to_remove)
    edges_after = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    edges_removed = edges_before - edges_after

    conn.commit()

    print(f"[fresh-brain] reset: removed {nodes_removed} nodes, {edges_removed} edges, kept {seeds_kept} seeds",
          flush=True)
    return {"nodes_removed": nodes_removed, "edges_removed": edges_removed, "seeds_kept": seeds_kept}


if __name__ == "__main__":
    # Smoke test: create fresh, show counts, reset, show counts again
    brain = create_fresh_eval_brain()
    n_count = brain.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    e_count = brain.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    print(f"[smoke] after create: {n_count} nodes, {e_count} edges")

    # Add a fake non-seed node to test reset
    brain.remember(type="observation", title="fake eval artifact",
                   content="this should be wiped by reset_to_seeds",
                   encoding_source="encoder:test")
    n_count = brain.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    print(f"[smoke] after fake add: {n_count} nodes")

    reset_to_seeds(brain)
    n_count = brain.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    e_count = brain.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    print(f"[smoke] after reset: {n_count} nodes, {e_count} edges (expect 16 nodes, 16 edges)")
