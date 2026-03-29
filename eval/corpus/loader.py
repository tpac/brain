"""Corpus loader — load and validate conversation files for eval suites.

Conversation format:
{
    "id": "conv_001",
    "source": "tom_session_2026-03-25",
    "category": "architecture_decisions",
    "exchanges": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "ground_truth": {
        "encode_targets": [
            {"type": "decision", "topic": "chose TCP over Unix sockets", "exchange_range": [3, 5]}
        ],
        "decode_queries": [
            {"query": "why TCP over Unix sockets?", "expected_topics": ["TCP over Unix sockets"]}
        ]
    }
}
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional


CORPUS_DIR = Path(__file__).parent


def load_conversation(path: str) -> Dict:
    """Load and validate a single conversation file."""
    with open(path) as f:
        conv = json.load(f)

    required = ['id', 'category', 'exchanges']
    for field in required:
        if field not in conv:
            raise ValueError("Conversation %s missing field: %s" % (path, field))

    if not conv['exchanges'] or len(conv['exchanges']) < 2:
        raise ValueError("Conversation %s needs at least 2 exchanges" % path)

    for i, ex in enumerate(conv['exchanges']):
        if 'role' not in ex or 'content' not in ex:
            raise ValueError("Exchange %d in %s missing role or content" % (i, path))
        if ex['role'] not in ('user', 'assistant'):
            raise ValueError("Exchange %d in %s has invalid role: %s" % (i, path, ex['role']))

    return conv


def load_corpus(category: Optional[str] = None, corpus_dir: Optional[str] = None) -> List[Dict]:
    """Load all conversations from the corpus directory.

    Args:
        category: Filter to specific category (e.g. 'architecture_decisions')
        corpus_dir: Override corpus directory path
    """
    cdir = Path(corpus_dir) if corpus_dir else CORPUS_DIR
    conversations = []

    for f in sorted(cdir.glob("*.json")):
        try:
            conv = load_conversation(str(f))
            if category and conv.get('category') != category:
                continue
            conversations.append(conv)
        except (json.JSONDecodeError, ValueError) as e:
            print("Warning: skipping %s: %s" % (f.name, e))

    return conversations


def get_categories(corpus_dir: Optional[str] = None) -> List[str]:
    """List all categories in the corpus."""
    convs = load_corpus(corpus_dir=corpus_dir)
    return sorted(set(c['category'] for c in convs))


def export_from_message_stream(db_path: str, session_id: str, output_path: str,
                               category: str = "real_session"):
    """Export a session from message_stream to corpus format.

    Args:
        db_path: Path to brain_logs.db
        session_id: Session ID to export
        output_path: Where to write the JSON file
        category: Category label for the conversation
    """
    import sqlite3

    conn = sqlite3.connect(db_path, timeout=3)
    rows = conn.execute(
        "SELECT role, content FROM message_stream "
        "WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    ).fetchall()
    conn.close()

    if not rows:
        raise ValueError("No messages found for session %s" % session_id)

    exchanges = [{"role": r[0], "content": r[1]} for r in rows if r[1]]

    conv = {
        "id": "session_%s" % session_id[:12],
        "source": "message_stream/%s" % session_id,
        "category": category,
        "exchanges": exchanges,
        "ground_truth": {
            "encode_targets": [],
            "decode_queries": []
        }
    }

    with open(output_path, 'w') as f:
        json.dump(conv, f, indent=2)

    return len(exchanges)
