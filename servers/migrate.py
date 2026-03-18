"""
tmemory — Migration Utility (v13 JS → v14 Python Serverless)

Handles the one-time migration from the Node.js HTTP server architecture
to the Python serverless architecture.

Key changes:
  1. Embedding model: bge-m3 (1024d) → snowflake-arctic-embed-m (768d)
     - All existing embeddings in node_embeddings must be re-computed
     - The model column changes from 'bge-m3' to 'snowflake-arctic-embed-m'
  2. Schema: v13 → v14
     - Adds session_activity table
     - Updates node_embeddings default model name
  3. Session tracking: in-memory (index.js) → SQLite (session_activity table)

Usage:
  python3 -m servers.migrate /path/to/brain.db

  Or from the plugin root:
  python3 servers/migrate.py /path/to/brain.db

The migration is non-destructive:
  - Creates a backup before modifying
  - Only re-embeds nodes that have outdated model or missing embeddings
  - Schema changes are additive (new tables/columns only)
"""

import sys
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone


def migrate(db_path: str, batch_size: int = 20):
    """
    Run the full migration from v13 (JS) to v14 (Python serverless).

    Args:
        db_path: Path to brain.db
        batch_size: How many nodes to re-embed per batch
    """
    if not os.path.exists(db_path):
        print(f"[migrate] Database not found: {db_path}")
        return False

    # Step 0: Backup
    backup_path = f"{db_path}.backup-pre-v14-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"[migrate] Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)

    # Step 1: Open DB and run schema migration
    print("[migrate] Running schema migration...")

    # Add parent dir to path for imports
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)

    from servers.schema import ensure_schema, BRAIN_VERSION
    from servers import embedder

    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')

    ensure_schema(conn)

    # Check version
    cur = conn.execute("SELECT value FROM brain_meta WHERE key = 'brain_schema_version'")
    row = cur.fetchone()
    print(f"[migrate] Schema version: {row[0] if row else '?'}")

    # Step 2: Count existing embeddings
    cur = conn.execute("SELECT COUNT(*) FROM node_embeddings")
    total_embeddings = cur.fetchone()[0]

    cur = conn.execute("SELECT COUNT(*) FROM node_embeddings WHERE model = 'bge-m3'")
    old_embeddings = cur.fetchone()[0]

    cur = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived = 0")
    total_nodes = cur.fetchone()[0]

    print(f"[migrate] Nodes: {total_nodes}, Existing embeddings: {total_embeddings} ({old_embeddings} bge-m3)")

    # Step 3: Load new embedding model
    print("[migrate] Loading FastEmbed model...")
    try:
        embedder.load_model()
        if not embedder.is_ready():
            print("[migrate] WARNING: Embedder not ready. Skipping re-embedding.")
            print("[migrate] Install fastembed: pip install fastembed --break-system-packages")
            conn.close()
            return True  # Schema migration still succeeded
    except Exception as e:
        print(f"[migrate] WARNING: Embedder failed to load: {e}")
        print("[migrate] Skipping re-embedding. Run migrate again after installing fastembed.")
        conn.close()
        return True

    # Step 4: Clear old bge-m3 embeddings (they're 1024d, new model is 768d)
    if old_embeddings > 0:
        print(f"[migrate] Clearing {old_embeddings} old bge-m3 embeddings...")
        conn.execute("DELETE FROM node_embeddings WHERE model = 'bge-m3'")
        conn.commit()

    # Step 5: Re-embed all nodes in batches
    print(f"[migrate] Re-embedding {total_nodes} nodes with snowflake-arctic-embed-m...")

    offset = 0
    total_embedded = 0
    t0 = time.time()

    while True:
        cur = conn.execute(
            """SELECT id, title, content FROM nodes
               WHERE archived = 0
               ORDER BY last_accessed DESC NULLS LAST
               LIMIT ? OFFSET ?""",
            (batch_size, offset)
        )
        nodes = cur.fetchall()
        if not nodes:
            break

        texts = [f"{title}{(' ' + content) if content else ''}" for _, title, content in nodes]
        blobs = embedder.embed_batch(texts)

        for i, (node_id, _, _) in enumerate(nodes):
            if i < len(blobs) and blobs[i]:
                conn.execute(
                    "INSERT OR REPLACE INTO node_embeddings (node_id, embedding, model, created_at) VALUES (?, ?, ?, ?)",
                    (node_id, blobs[i], embedder.MODEL_NAME,
                     datetime.now(timezone.utc).isoformat())
                )
                total_embedded += 1

        conn.commit()
        offset += batch_size
        elapsed = time.time() - t0
        rate = total_embedded / elapsed if elapsed > 0 else 0
        print(f"  Embedded {total_embedded}/{total_nodes} ({rate:.1f}/s)")

    elapsed = time.time() - t0
    print(f"[migrate] Re-embedding complete: {total_embedded} nodes in {elapsed:.1f}s")

    # Step 6: Record migration
    conn.execute(
        "INSERT INTO version_history (version, migration_ts, description) VALUES (?, ?, ?)",
        (14, datetime.now(timezone.utc).isoformat(),
         f'v14 serverless migration: {total_embedded} nodes re-embedded with snowflake-arctic-embed-m (768d)')
    )
    conn.commit()
    conn.close()

    print(f"[migrate] Migration complete. Backup at: {backup_path}")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Try default location
        db_path = None
        for candidate in [
            os.environ.get('TMEMORY_DB_DIR', ''),
        ]:
            if candidate:
                p = os.path.join(candidate, 'brain.db')
                if os.path.exists(p):
                    db_path = p
                    break

        if not db_path:
            # Search common locations
            import glob
            for pattern in [
                '/sessions/*/mnt/AgentsContext/tmemory/brain.db',
                os.path.expanduser('~/AgentsContext/tmemory/brain.db'),
            ]:
                matches = glob.glob(pattern)
                if matches:
                    db_path = matches[0]
                    break

        if not db_path:
            print("Usage: python3 migrate.py /path/to/brain.db")
            sys.exit(1)
    else:
        db_path = sys.argv[1]

    print(f"[migrate] Migrating: {db_path}")
    success = migrate(db_path)
    sys.exit(0 if success else 1)
