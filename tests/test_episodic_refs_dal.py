"""DAL methods for episodic references (v27 substrate).

  - TraceDAL: store_embeddings / get_embeddings / find_unembedded
  - GraphDAL: add_source_refs / get_source_refs / get_nodes_referencing

Pure-DAL tests: in-memory SQLite, no live Brain, no embedder.
"""

import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.dal import TraceDAL, GraphDAL
from servers.schema import ensure_schema, ensure_logs_schema


def _open_in_memory():
    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def _seed_node(conn, node_id):
    conn.execute(
        "INSERT INTO nodes (id, type, title) VALUES (?, ?, ?)",
        (node_id, 'concept', node_id))


def _seed_trace(conn, scale='s0', ref_type='user_message',
                summary='hello', session_id='sess-1'):
    """Insert a trace_event row and return its id."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        'INSERT INTO trace_events '
        '(chain_id, scale, event_type, ref_type, summary, '
        ' metadata, session_id, created_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        ('chain-x', scale, 'K' if ref_type == 'user_message' else 'delta',
         ref_type, summary, None, session_id, now))
    conn.commit()
    return cur.lastrowid


class TraceEmbeddingsDALTest(unittest.TestCase):
    def setUp(self):
        self.conn = _open_in_memory()
        ensure_logs_schema(self.conn)
        self.dal = TraceDAL(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_store_and_get_single(self):
        n = self.dal.store_embeddings(
            [(1, b'\x01\x02', 'tom said hi')], model='nomic')
        self.assertEqual(n, 1)
        got = self.dal.get_embeddings([1])
        self.assertEqual(got, {1: b'\x01\x02'})

    def test_store_skips_null_vectors(self):
        n = self.dal.store_embeddings(
            [(1, b'\x00', 'ok'), (2, None, 'skip'), (3, b'\x02', 'ok2')],
            model='nomic')
        self.assertEqual(n, 2)
        self.assertEqual(set(self.dal.get_embeddings([1, 2, 3]).keys()),
                         {1, 3})

    def test_store_replaces_on_conflict(self):
        self.dal.store_embeddings([(1, b'\xaa', 'first')], model='nomic')
        self.dal.store_embeddings([(1, b'\xbb', 'second')], model='nomic')
        got = self.dal.get_embeddings([1])
        self.assertEqual(got[1], b'\xbb')

    def test_text_truncation_500_chars(self):
        long_text = 'x' * 1000
        self.dal.store_embeddings([(7, b'\x01', long_text)], model='nomic')
        row = self.conn.execute(
            'SELECT text FROM trace_embeddings WHERE trace_id = 7'
        ).fetchone()
        self.assertEqual(len(row[0]), 500)

    def test_get_embeddings_empty(self):
        self.assertEqual(self.dal.get_embeddings([]), {})
        self.assertEqual(self.dal.get_embeddings([999]), {})

    def test_find_unembedded_returns_newest_first(self):
        t1 = _seed_trace(self.conn, summary='first')
        t2 = _seed_trace(self.conn, summary='second')
        t3 = _seed_trace(self.conn, summary='third')
        rows = self.dal.find_unembedded(
            limit=5, scales=['s0'],
            ref_types=['user_message', 'assistant_message', 'tool_result'])
        self.assertEqual([r['id'] for r in rows], [t3, t2, t1])

    def test_find_unembedded_skips_already_embedded(self):
        t1 = _seed_trace(self.conn, summary='first')
        t2 = _seed_trace(self.conn, summary='second')
        self.dal.store_embeddings([(t1, b'\x01', 'first')], model='nomic')
        rows = self.dal.find_unembedded(
            limit=5, scales=['s0'], ref_types=['user_message'])
        self.assertEqual([r['id'] for r in rows], [t2])

    def test_find_unembedded_respects_limit(self):
        for _ in range(10):
            _seed_trace(self.conn)
        rows = self.dal.find_unembedded(
            limit=3, scales=['s0'], ref_types=['user_message'])
        self.assertEqual(len(rows), 3)

    def test_find_unembedded_filters_by_scale(self):
        _seed_trace(self.conn, scale='s0')
        _seed_trace(self.conn, scale='s1', ref_type='recall')
        rows = self.dal.find_unembedded(
            limit=5, scales=['s0'], ref_types=['user_message'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['scale'], 's0')

    def test_find_unembedded_filters_by_ref_type(self):
        _seed_trace(self.conn, ref_type='user_message')
        _seed_trace(self.conn, ref_type='assistant_message')
        rows = self.dal.find_unembedded(
            limit=5, scales=['s0'], ref_types=['user_message'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['ref_type'], 'user_message')

    def test_find_unembedded_requires_scales(self):
        with self.assertRaises(ValueError):
            self.dal.find_unembedded(
                limit=5, scales=[], ref_types=['user_message'])

    def test_find_unembedded_requires_ref_types(self):
        with self.assertRaises(ValueError):
            self.dal.find_unembedded(
                limit=5, scales=['s0'], ref_types=[])

    def test_find_unembedded_respects_since_cutoff(self):
        """The `since` window excludes traces older than the cutoff —
        bounds embed scope to traces created after identity stamping
        went live (avoids polluting the vector neighborhood with
        sentinel-tagged historical traces)."""
        import time as _time
        t1 = _seed_trace(self.conn, summary='old')
        # Force old created_at on t1
        self.conn.execute(
            "UPDATE trace_events SET created_at = '2025-01-01T00:00:00+00:00' "
            "WHERE id = ?", (t1,))
        self.conn.commit()
        t2 = _seed_trace(self.conn, summary='recent')
        rows = self.dal.find_unembedded(
            limit=5, scales=['s0'], ref_types=['user_message'],
            since='2025-06-01T00:00:00+00:00')
        ids = [r['id'] for r in rows]
        self.assertIn(t2, ids)
        self.assertNotIn(t1, ids)

    def test_find_unembedded_no_since_returns_all(self):
        """since=None preserves legacy behavior (no time filter)."""
        t1 = _seed_trace(self.conn, summary='one')
        rows = self.dal.find_unembedded(
            limit=5, scales=['s0'], ref_types=['user_message'])
        self.assertIn(t1, [r['id'] for r in rows])

    def test_decode_metadata_single_encoded(self):
        """Post-fix metadata is single-encoded JSON — one json.loads pass."""
        meta = self.dal._decode_metadata('{"tool": "Bash"}')
        self.assertEqual(meta, {'tool': 'Bash'})

    def test_decode_metadata_double_encoded_legacy(self):
        """Pre-fix tool_result traces stored '"{\\"tool\\": \\"Bash\\"}"'
        (the dispatch json.dumps'd a string client payload, then
        TraceDAL.append json.dumps'd that again). Defensive decode
        unwraps both layers."""
        import json as _json
        double = _json.dumps('{"tool": "Bash"}')  # produces '"{\"tool\": \"Bash\"}"'
        self.assertEqual(self.dal._decode_metadata(double),
                         {'tool': 'Bash'})

    def test_decode_metadata_handles_garbage(self):
        self.assertEqual(self.dal._decode_metadata(None), {})
        self.assertEqual(self.dal._decode_metadata(''), {})
        self.assertEqual(self.dal._decode_metadata('not-json'), {})
        # JSON that decodes to a non-dict primitive — return {} not crash
        self.assertEqual(self.dal._decode_metadata('42'), {})


class SourceRefsDALTest(unittest.TestCase):
    def setUp(self):
        self.conn = _open_in_memory()
        ensure_schema(self.conn)
        self.dal = GraphDAL(self.conn)
        _seed_node(self.conn, 'node_a')
        _seed_node(self.conn, 'node_b')

    def tearDown(self):
        self.conn.close()

    def test_add_and_get_preserves_order(self):
        n = self.dal.add_source_refs('node_a', [42, 17, 99])
        self.assertEqual(n, 3)
        self.assertEqual(self.dal.get_source_refs('node_a'), [42, 17, 99])

    def test_add_ignores_duplicates(self):
        first = self.dal.add_source_refs('node_a', [10, 20])
        second = self.dal.add_source_refs('node_a', [10, 30])
        self.assertEqual(first, 2)
        self.assertEqual(second, 1)  # only 30 new; 10 ignored
        self.assertEqual(self.dal.get_source_refs('node_a'), [10, 20, 30])

    def test_add_empty_inputs_returns_zero(self):
        self.assertEqual(self.dal.add_source_refs('', [1, 2]), 0)
        self.assertEqual(self.dal.add_source_refs('node_a', []), 0)

    def test_get_source_refs_empty_node(self):
        self.assertEqual(self.dal.get_source_refs('node_b'), [])
        self.assertEqual(self.dal.get_source_refs('nonexistent'), [])

    def test_get_nodes_referencing_cohort(self):
        self.dal.add_source_refs('node_a', [100, 200])
        self.dal.add_source_refs('node_b', [200, 300])
        cohort = self.dal.get_nodes_referencing(200)
        self.assertEqual(sorted(cohort), ['node_a', 'node_b'])
        self.assertEqual(self.dal.get_nodes_referencing(100), ['node_a'])
        self.assertEqual(self.dal.get_nodes_referencing(999), [])

    def test_fk_cascade_clears_refs(self):
        self.dal.add_source_refs('node_a', [5, 6, 7])
        self.conn.execute("DELETE FROM nodes WHERE id = 'node_a'")
        self.assertEqual(self.dal.get_source_refs('node_a'), [])


if __name__ == '__main__':
    unittest.main()
