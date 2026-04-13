"""DAL for node metadata key-value store.

All metadata reads/writes go through this class. No raw SQL for metadata
anywhere else in the codebase. Keys are validated against the contract.

Usage:
    dal = MetadataDAL(conn)
    dal.set(node_id, 'reasoning', 'Why this was encoded')
    dal.set_many(node_id, {'reasoning': '...', 'anchor_raw_quote': '...'})
    meta = dal.get(node_id)  # → {'reasoning': '...', 'anchor_raw_quote': '...'}
    dal.delete(node_id, 'reasoning')
"""

import sqlite3
from typing import Dict, Optional, List


class MetadataDAL:
    """Key-value metadata access for nodes. Single table: node_metadata_kv."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get(self, node_id: str) -> Dict[str, str]:
        """Get all metadata for a node as a dict. Returns {} if none."""
        rows = self.conn.execute(
            'SELECT key, value FROM node_metadata_kv WHERE node_id = ?',
            (node_id,)).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_field(self, node_id: str, key: str) -> Optional[str]:
        """Get a single metadata field. Returns None if not set."""
        row = self.conn.execute(
            'SELECT value FROM node_metadata_kv WHERE node_id = ? AND key = ?',
            (node_id, key)).fetchone()
        return row[0] if row else None

    def get_fields(self, node_id: str, keys: List[str]) -> Dict[str, str]:
        """Get specific metadata fields. Returns dict with only present keys."""
        if not keys:
            return {}
        placeholders = ','.join('?' * len(keys))
        rows = self.conn.execute(
            'SELECT key, value FROM node_metadata_kv WHERE node_id = ? AND key IN (%s)' % placeholders,
            [node_id] + keys).fetchall()
        return {r[0]: r[1] for r in rows}

    def set(self, node_id: str, key: str, value: str) -> None:
        """Set a single metadata field. Overwrites if exists."""
        if value is None or str(value).strip() == '':
            return  # Don't store empty values
        self.conn.execute(
            'INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) VALUES (?, ?, ?)',
            (node_id, key, str(value)))

    def set_many(self, node_id: str, metadata: Dict[str, str]) -> int:
        """Set multiple metadata fields at once. Returns count written."""
        count = 0
        for key, value in metadata.items():
            if value is not None and str(value).strip():
                self.conn.execute(
                    'INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) VALUES (?, ?, ?)',
                    (node_id, key, str(value)))
                count += 1
        return count

    def delete(self, node_id: str, key: str) -> bool:
        """Delete a single metadata field. Returns True if existed."""
        cursor = self.conn.execute(
            'DELETE FROM node_metadata_kv WHERE node_id = ? AND key = ?',
            (node_id, key))
        return cursor.rowcount > 0

    def delete_all(self, node_id: str) -> int:
        """Delete all metadata for a node. Returns count deleted."""
        cursor = self.conn.execute(
            'DELETE FROM node_metadata_kv WHERE node_id = ?',
            (node_id,))
        return cursor.rowcount

    def nodes_with_field(self, key: str) -> int:
        """Count how many nodes have a specific metadata field set."""
        row = self.conn.execute(
            'SELECT COUNT(DISTINCT node_id) FROM node_metadata_kv WHERE key = ?',
            (key,)).fetchone()
        return row[0] if row else 0

    def total_nodes(self) -> int:
        """Count how many nodes have any metadata."""
        row = self.conn.execute(
            'SELECT COUNT(DISTINCT node_id) FROM node_metadata_kv').fetchone()
        return row[0] if row else 0

    def get_all_by_key(self, key: str) -> Dict[str, str]:
        """Get all node_id→value pairs for a given key. For bulk loading."""
        rows = self.conn.execute(
            'SELECT node_id, value FROM node_metadata_kv WHERE key = ?',
            (key,)).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_paired_keys(self, key1: str, key2: str) -> Dict[str, tuple]:
        """Get paired values for two keys per node. Returns {node_id: (val1, val2)}.

        Used for loading z-score stats (mean + std as a pair).
        Nodes must have BOTH keys to be included.
        """
        rows = self.conn.execute(
            'SELECT a.node_id, a.value, b.value '
            'FROM node_metadata_kv a '
            'JOIN node_metadata_kv b ON a.node_id = b.node_id AND b.key = ? '
            'WHERE a.key = ?',
            (key2, key1)).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def get_nodes_with_flag(self, key: str, value: str = 'true') -> List[str]:
        """Get node IDs where a flag is set to a specific value.

        Used for needs_enrichment flags, etc.
        """
        rows = self.conn.execute(
            'SELECT node_id FROM node_metadata_kv WHERE key = ? AND value = ?',
            (key, value)).fetchall()
        return [r[0] for r in rows]

    def clear_flag(self, node_id: str, key: str) -> bool:
        """Delete a flag after it's been processed. Returns True if existed."""
        return self.delete(node_id, key)

    def bulk_set_key(self, key: str, node_values: Dict[str, str]) -> int:
        """Set the same key on many nodes at once. Returns count written."""
        count = 0
        for node_id, value in node_values.items():
            if value is not None and str(value).strip():
                self.conn.execute(
                    'INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) VALUES (?, ?, ?)',
                    (node_id, key, str(value)))
                count += 1
        return count

    def field_coverage(self) -> Dict[str, int]:
        """Get count of nodes per metadata key. For health monitoring."""
        rows = self.conn.execute(
            'SELECT key, COUNT(DISTINCT node_id) FROM node_metadata_kv GROUP BY key'
        ).fetchall()
        return {r[0]: r[1] for r in rows}
