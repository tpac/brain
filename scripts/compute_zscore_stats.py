#!/usr/bin/env python3
"""Compute per-node z-score statistics for contrastive recall scoring.

For each node, computes mean and std of cosine similarity across a diverse
set of queries. Stores in node_metadata_kv via MetadataDAL.

At recall time: z_score = (cosine - mean) / std
This measures SURPRISE (how unusually relevant this node is to THIS query)
rather than raw similarity.

Usage:
    python3 scripts/compute_zscore_stats.py              # Compute + store
    python3 scripts/compute_zscore_stats.py --dry-run    # Compute only, don't store
    python3 scripts/compute_zscore_stats.py --check      # Show current coverage
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# Diverse query set — covers different brain areas, query styles, and vocabulary.
# These calibrate what "normal" cosine looks like for each node.
CALIBRATION_QUERIES = [
    # Technical — brain internals
    "How does the recall pipeline work",
    "Fix the encoding agent bug",
    "Deploy the dashboard changes",
    "Create a new node type",
    "Remove dead code from the codebase",
    "Run the tests and check results",
    "Design the API for this feature",
    "Debug the crash in brain_recall.py",
    "What are the different edge types",
    "How does session management work",
    # Operational
    "Write documentation for this module",
    "Refactor the module into smaller files",
    "Optimize memory usage in the daemon",
    "Handle the error gracefully",
    "Update the database schema",
    "Check the performance metrics",
    "Connect to the database",
    "Install the dependencies",
    # Conversational / personal
    "Tell me about Tom",
    "What is a community in the brain",
    "How does Haiku select nodes",
    "What makes Anchor different from Claude",
    "What did we decide about encoding quality",
    "Why did we choose this architecture",
    "What are the open questions right now",
    # Abstract / inferential
    "Something about this feels wrong",
    "Are we making progress or going in circles",
    "What would happen if the brain had 10000 nodes",
    "What is the most important thing we learned",
    "How do we know if the brain is actually helping",
    # Short / vague
    "Lets start coding",
    "What about the decoding side",
    "Good morning",
    "Sounds good",
    "Can you check that",
    # Domain-specific actions
    "I want to delete all archived nodes",
    "I want to optimize the scoring formula",
    "The hook is taking 15 seconds",
    "The encoder creates too many nodes",
    "I keep seeing the same nodes every time",
]


def compute_stats(brain):
    """Compute mean and std cosine for each node across calibration queries.

    Returns: {node_id: (mean, std)}
    """
    from servers import embedder
    from servers.dal import EmbeddingDAL

    if not embedder.is_ready():
        print("ERROR: Embedder not ready")
        return {}

    # Embed all calibration queries
    print(f"Embedding {len(CALIBRATION_QUERIES)} calibration queries...")
    query_vecs = []
    for q in CALIBRATION_QUERIES:
        vec = embedder.embed(q)
        if vec:
            query_vecs.append(vec)
    print(f"  {len(query_vecs)} embedded successfully")

    # Also add real queries from S1R traces for better calibration
    try:
        import json
        trace_rows = brain._trace_dal.conn.execute(
            "SELECT metadata FROM trace_events "
            "WHERE scale = 's1' AND ref_type = 'recall' AND event_type = 'O' "
            "ORDER BY created_at DESC LIMIT 30"
        ).fetchall()
        for (meta,) in trace_rows:
            try:
                q = json.loads(meta).get('query', '')[:300]
                if q and len(q.strip()) > 10:
                    vec = embedder.embed(q)
                    if vec:
                        query_vecs.append(vec)
            except (json.JSONDecodeError, TypeError):
                pass
        print(f"  +{len(query_vecs) - len(CALIBRATION_QUERIES)} from real S1R traces")
    except Exception as e:
        print(f"  (skipped trace queries: {e})")

    print(f"  Total calibration vectors: {len(query_vecs)}")

    # Load all node embeddings
    emb_dal = EmbeddingDAL(brain.conn)
    emb_rows = emb_dal.get_all_with_context(exclude_archived=True)
    print(f"Scoring {len(emb_rows)} nodes against {len(query_vecs)} queries...")

    import numpy as np

    stats = {}
    t0 = time.time()
    for i, row in enumerate(emb_rows):
        nid = row['node_id']
        blob = row['embedding']
        if not blob:
            continue

        cosines = []
        for qvec in query_vecs:
            sim = embedder.cosine_similarity(qvec, blob)
            cosines.append(sim)

        mean = float(np.mean(cosines))
        std = float(max(np.std(cosines), 0.001))
        stats[nid] = (mean, std)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i + 1}/{len(emb_rows)} ({rate:.0f} nodes/s)")

    elapsed = time.time() - t0
    print(f"Computed stats for {len(stats)} nodes in {elapsed:.1f}s")
    return stats


def store_stats(brain, stats):
    """Store z-score stats in node_metadata_kv via MetadataDAL."""
    from servers.dal_metadata import MetadataDAL
    from servers.brain_constants import ZSCORE_STATS_KEY_MEAN, ZSCORE_STATS_KEY_STD

    mdal = MetadataDAL(brain.conn)

    means = {nid: str(round(mean, 6)) for nid, (mean, _) in stats.items()}
    stds = {nid: str(round(std, 6)) for nid, (_, std) in stats.items()}

    print(f"Storing {len(means)} mean values...")
    count_m = mdal.bulk_set_key(ZSCORE_STATS_KEY_MEAN, means)

    print(f"Storing {len(stds)} std values...")
    count_s = mdal.bulk_set_key(ZSCORE_STATS_KEY_STD, stds)

    brain.conn.commit()
    print(f"Stored: {count_m} means, {count_s} stds")
    return count_m


def check_coverage(brain):
    """Show current z-score stats coverage."""
    from servers.dal_metadata import MetadataDAL
    from servers.brain_constants import ZSCORE_STATS_KEY_MEAN, ZSCORE_STATS_KEY_STD

    mdal = MetadataDAL(brain.conn)
    n_mean = mdal.nodes_with_field(ZSCORE_STATS_KEY_MEAN)
    n_std = mdal.nodes_with_field(ZSCORE_STATS_KEY_STD)

    total = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE archived = 0"
    ).fetchone()[0]
    with_emb = brain.conn.execute(
        "SELECT COUNT(*) FROM node_embeddings ne "
        "JOIN nodes n ON n.id = ne.node_id WHERE n.archived = 0"
    ).fetchone()[0]

    print(f"Z-score stats coverage:")
    print(f"  Total active nodes:     {total}")
    print(f"  With embeddings:        {with_emb}")
    print(f"  With zscore_mean:       {n_mean} ({100 * n_mean / max(with_emb, 1):.0f}%)")
    print(f"  With zscore_std:        {n_std} ({100 * n_std / max(with_emb, 1):.0f}%)")


def main():
    parser = argparse.ArgumentParser(description='Compute z-score stats for contrastive recall')
    parser.add_argument('--dry-run', action='store_true', help='Compute only, do not store')
    parser.add_argument('--check', action='store_true', help='Show current coverage only')
    args = parser.parse_args()

    from tests.isolated_brain import IsolatedBrain

    if args.check:
        with IsolatedBrain(cleanup=False) as env:
            check_coverage(env.brain)
        return

    if args.dry_run:
        with IsolatedBrain() as env:
            stats = compute_stats(env.brain)
            if stats:
                import numpy as np
                means = [m for m, _ in stats.values()]
                stds = [s for _, s in stats.values()]
                print(f"\nStats summary (dry run):")
                print(f"  Mean of means: {np.mean(means):.4f}")
                print(f"  Mean of stds:  {np.mean(stds):.4f}")
                print(f"  Nodes with std < 0.02: {sum(1 for s in stds if s < 0.02)}")
        return

    # Production: compute and store on the REAL brain
    from servers.brain import Brain

    db_dir = os.environ.get('BRAIN_DB_DIR', os.path.join(os.path.expanduser('~'), 'AgentsContext', 'brain'))
    db_path = os.path.join(db_dir, 'brain.db')

    if not os.path.exists(db_path):
        print(f"ERROR: brain.db not found at {db_path}")
        sys.exit(1)

    # Load .env for API keys (embedder may need config)
    env_path = os.path.join(ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

    brain = Brain(db_path)
    try:
        stats = compute_stats(brain)
        if stats:
            store_stats(brain, stats)
            check_coverage(brain)
    finally:
        brain.save()
        brain.close()


if __name__ == '__main__':
    main()
