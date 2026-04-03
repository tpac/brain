"""Tests for redistribution.py — vector blending, fidelity, bridge detection.

Tests the pure computation functions without needing a full brain DB.
Integration tests (test on real DB copy) are in tests/integration/.
"""

import unittest
import struct
import math
import sqlite3
import sys

sys.path.insert(0, '.')

from servers.redistribution import (
    blend_vectors,
    cosine_sim,
    get_blend_ratio,
    is_bridge_node,
    BLEND_RATIOS,
    FIDELITY_RESET_THRESHOLD,
)


def make_vec(*values):
    """Create a bytes vector from float values."""
    return struct.pack(f'{len(values)}f', *values)


def vec_to_list(blob):
    dim = len(blob) // 4
    return list(struct.unpack(f'{dim}f', blob))


class TestBlendVectors(unittest.TestCase):
    """Test vector blending formula: ratio * original + (1-ratio) * neighbor."""

    def test_full_original(self):
        """ratio=1.0 should return original (normalized)."""
        orig = make_vec(1.0, 0.0, 0.0)
        neigh = make_vec(0.0, 1.0, 0.0)
        result = blend_vectors(orig, neigh, 1.0)
        vals = vec_to_list(result)
        self.assertAlmostEqual(vals[0], 1.0, places=3)
        self.assertAlmostEqual(vals[1], 0.0, places=3)

    def test_full_neighbor(self):
        """ratio=0.0 should return neighbor (normalized)."""
        orig = make_vec(1.0, 0.0, 0.0)
        neigh = make_vec(0.0, 1.0, 0.0)
        result = blend_vectors(orig, neigh, 0.0)
        vals = vec_to_list(result)
        self.assertAlmostEqual(vals[0], 0.0, places=3)
        self.assertAlmostEqual(vals[1], 1.0, places=3)

    def test_half_blend(self):
        """ratio=0.5 should be between original and neighbor."""
        orig = make_vec(1.0, 0.0, 0.0)
        neigh = make_vec(0.0, 1.0, 0.0)
        result = blend_vectors(orig, neigh, 0.5)
        vals = vec_to_list(result)
        # Both components should be equal (normalized)
        self.assertAlmostEqual(vals[0], vals[1], places=3)

    def test_result_is_normalized(self):
        """Output should be L2 normalized."""
        orig = make_vec(3.0, 4.0, 0.0)
        neigh = make_vec(0.0, 3.0, 4.0)
        result = blend_vectors(orig, neigh, 0.7)
        vals = vec_to_list(result)
        norm = math.sqrt(sum(v*v for v in vals))
        self.assertAlmostEqual(norm, 1.0, places=3)


class TestCosineSim(unittest.TestCase):

    def test_identical(self):
        v = make_vec(1.0, 2.0, 3.0)
        self.assertAlmostEqual(cosine_sim(v, v), 1.0, places=3)

    def test_orthogonal(self):
        a = make_vec(1.0, 0.0, 0.0)
        b = make_vec(0.0, 1.0, 0.0)
        self.assertAlmostEqual(cosine_sim(a, b), 0.0, places=3)

    def test_opposite(self):
        a = make_vec(1.0, 0.0, 0.0)
        b = make_vec(-1.0, 0.0, 0.0)
        self.assertAlmostEqual(cosine_sim(a, b), -1.0, places=3)


class TestGetBlendRatio(unittest.TestCase):

    def test_locked(self):
        self.assertEqual(get_blend_ratio({'locked': True}), BLEND_RATIOS['locked'])

    def test_high_confidence(self):
        self.assertEqual(get_blend_ratio({'confidence': 0.95}), BLEND_RATIOS['high_confidence'])

    def test_low_confidence(self):
        self.assertEqual(get_blend_ratio({'confidence': 0.3}), BLEND_RATIOS['low_confidence'])

    def test_normal(self):
        self.assertEqual(get_blend_ratio({'confidence': 0.7}), BLEND_RATIOS['normal'])

    def test_no_confidence(self):
        self.assertEqual(get_blend_ratio({}), BLEND_RATIOS['normal'])


class TestBridgeNode(unittest.TestCase):

    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        self.conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY)")
        self.conn.execute("CREATE TABLE edges (source_id TEXT, target_id TEXT, weight REAL, edge_type TEXT)")
        # Create nodes in 2 communities
        for nid in ['a', 'b', 'c', 'd', 'e', 'bridge']:
            self.conn.execute("INSERT INTO nodes VALUES (?)", (nid,))

    def test_bridge_detected(self):
        """Node with equal-weight edges to 2 OTHER communities = bridge."""
        # bridge is in community 2, connects equally to community 0 and 1
        communities = {'a': 0, 'b': 0, 'c': 1, 'd': 1, 'bridge': 2}
        self.conn.execute("INSERT INTO edges VALUES ('bridge', 'a', 0.7, 'related')")
        self.conn.execute("INSERT INTO edges VALUES ('bridge', 'b', 0.7, 'related')")
        self.conn.execute("INSERT INTO edges VALUES ('bridge', 'c', 0.7, 'related')")
        self.conn.execute("INSERT INTO edges VALUES ('bridge', 'd', 0.7, 'related')")
        self.assertTrue(is_bridge_node(self.conn, 'bridge', communities))

    def test_non_bridge(self):
        """Node with edges to only 1 community = not bridge."""
        communities = {'a': 0, 'b': 0, 'c': 1, 'd': 1, 'e': 0}
        self.conn.execute("INSERT INTO edges VALUES ('e', 'a', 0.7, 'related')")
        self.conn.execute("INSERT INTO edges VALUES ('e', 'b', 0.7, 'related')")
        self.assertFalse(is_bridge_node(self.conn, 'e', communities))

    def test_dominant_community_not_bridge(self):
        """Node with one community dominating by 2x+ = not bridge."""
        communities = {'a': 0, 'b': 0, 'c': 1, 'bridge': 0}
        self.conn.execute("INSERT INTO edges VALUES ('bridge', 'a', 0.9, 'related')")
        self.conn.execute("INSERT INTO edges VALUES ('bridge', 'b', 0.9, 'related')")
        self.conn.execute("INSERT INTO edges VALUES ('bridge', 'c', 0.3, 'related')")
        self.assertFalse(is_bridge_node(self.conn, 'bridge', communities))

    def test_no_community_assignment(self):
        self.assertFalse(is_bridge_node(self.conn, 'unknown', {}))

    def tearDown(self):
        self.conn.close()


class TestFidelityThreshold(unittest.TestCase):

    def test_threshold_value(self):
        self.assertEqual(FIDELITY_RESET_THRESHOLD, 0.50)

    def test_blend_stays_above_threshold(self):
        """70/30 blend of orthogonal vectors should stay above 0.50 fidelity."""
        orig = make_vec(1.0, 0.0, 0.0)
        neigh = make_vec(0.0, 1.0, 0.0)
        result = blend_vectors(orig, neigh, 0.7)
        fidelity = cosine_sim(result, orig)
        self.assertGreater(fidelity, FIDELITY_RESET_THRESHOLD)


if __name__ == '__main__':
    unittest.main()
