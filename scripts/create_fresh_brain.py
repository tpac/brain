#!/usr/bin/env python3
"""Create a fresh brain database pair (brain.db + brain_logs.db).

Usage:
    python3 scripts/create_fresh_brain.py [output_dir]

If output_dir is not specified, creates in a temp directory and prints the path.
Creates both brain.db and brain_logs.db with the current schema (no data).

Use for: testing, development, fresh installs, isolated evals.
"""

import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_fresh_brain(output_dir: str = None, skip_embedder: bool = True) -> str:
    """Create a fresh brain.db + brain_logs.db in output_dir.

    Args:
        output_dir: Directory to create databases in. Created if missing.
                   If None, uses a temp directory.
        skip_embedder: If True (default), don't load the embedding model.

    Returns:
        Path to brain.db
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='brain_fresh_')

    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, 'brain.db')

    if os.path.exists(db_path):
        print("ERROR: %s already exists. Won't overwrite." % db_path)
        sys.exit(1)

    from servers.brain import Brain
    brain = Brain(db_path, skip_embedder=skip_embedder)

    # Seed interactions (judge, encoding_agent, etc.)
    try:
        from servers.interaction_seed import seed_interactions
        seed_interactions(brain)
    except Exception as e:
        print("WARNING: interaction seeding failed: %s" % e)

    brain.save()
    brain.close()

    return db_path


if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else None
    db_path = create_fresh_brain(output_dir)
    print("Created fresh brain at: %s" % db_path)
    print("  brain.db:      %s" % db_path)
    print("  brain_logs.db: %s" % os.path.join(os.path.dirname(db_path), 'brain_logs.db'))
